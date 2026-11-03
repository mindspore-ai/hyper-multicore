# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Locked wheel preparation and project-owned, content-addressed framework environments."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
import textwrap
import sys
import time
import urllib.parse
import zipfile
from email.parser import BytesParser
from .command_runner import message
from .build_workspace import framework_directory
from .build_utils import atomic_json
from . import framework_store as STORE


def digest_json(value) -> str:
    """Return a deterministic input identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_targets(path: Path) -> list[dict]:
    """Read version pairs independently of the current interpreter's wheel tags."""
    from packaging.version import Version

    config = json.loads(path.read_text())
    if config.get("schema_version") != 2:
        raise ValueError("Unsupported framework matrix schema")
    targets = [
        {
            **target,
            "indexes": config["indexes"],
            "build_requirements": config["build_requirements"],
            "torch_release": Version(target["torch_version"]).base_version,
            "pta_release": Version(target["pta_build_version"]).base_version,
        }
        for target in config["targets"]
    ]
    if (
        not targets
        or len({t["id"] for t in targets}) != len(targets)
        or len({t["pta_release"] for t in targets}) != len(targets)
    ):
        raise ValueError("Empty or duplicate framework targets")
    for target in targets:
        validate_target(target)
    return targets


def validate_target(target: dict) -> None:
    """Require exact framework versions and named indexes, without per-platform artifacts."""
    from packaging.version import Version

    for field in ("id", "torch_release", "pta_release", "torch_version", "pta_build_version"):
        if not isinstance(target.get(field), str) or not re.fullmatch(r"[a-zA-Z0-9.+_-]+", target[field]):
            raise ValueError(f"Unsafe framework target field: {field}")
    for distribution, field in (("torch", "torch_version"), ("pta", "pta_build_version")):
        version = Version(target[field])
        if version.base_version != target[f"{distribution}_release"]:
            raise ValueError(f"Framework branch differs from build version: {field}")
    validate_sources_and_requirements(target)


def validate_sources_and_requirements(target: dict):
    """Validate common resolver settings without prescribing a complete package list."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    for name in ("torch", "pta", "dependencies"):
        url = urllib.parse.urlparse(target["indexes"][name])
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError(f"Framework indexes require credential-free HTTPS URLs: {name}")
    for text in target["build_requirements"]:
        requirement = Requirement(text)
        if (
            requirement.url
            or requirement.extras
            or canonicalize_name(requirement.name) in ("torch", "torch-npu", "pytest", "pybind11")
        ):
            raise ValueError(f"Invalid direct build requirement: {text}")


def isolated_environment(python: Path) -> dict:
    """Preserve CANN/toolchain settings but remove inherited framework configuration."""
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "TORCH_INSTALL_PREFIX", "Torch_DIR", "Caffe2_DIR"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    # Framework-owned paths are chosen from the target interpreter, never inherited from another venv.
    for key in ("LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "CMAKE_PREFIX_PATH"):
        if key in env:
            env[key] = os.pathsep.join(
                p
                for p in env[key].split(os.pathsep)
                if "site-packages/torch" not in p
                and "framework/envs/" not in p
                and "site-packages/hyper_multicore" not in p
                and "/build/native/payload/" not in p
                and "/build/native/components/" not in p
            )
    return env


def cann_tool_environment(python: Path) -> dict:
    """Expose only the selected CANN tool modules to native compiler workers, never to pip."""
    env = isolated_environment(python)
    cann = Path(env["ASCEND_HOME_PATH"]).resolve()
    paths = [cann / "python/site-packages", cann / "opp/built-in/op_impl/ai_core/tbe"]
    if not all(path.is_dir() for path in paths):
        raise ValueError(f"Missing installed CANN compiler Python paths: {paths}")
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    env["HI_PYTHON"] = str(python)
    return env


def wheel_record(path: Path, source: str) -> dict:
    """Inspect a selected wheel using current interpreter tags and primary metadata."""
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename, canonicalize_name

    filename_name, filename_version, _, tags = parse_wheel_filename(path.name)
    if not tags.intersection(sys_tags()):
        raise ValueError(f"Wheel tags do not match build platform: {path.name}")
    with zipfile.ZipFile(path) as archive:
        entries = [n for n in archive.namelist() if len(Path(n).parts) == 2 and n.endswith(".dist-info/METADATA")]
        if len(entries) != 1:
            raise ValueError(f"Ambiguous wheel metadata: {path}")
        meta = BytesParser().parsebytes(archive.read(entries[0]))
    if canonicalize_name(meta["Name"]) != filename_name or str(filename_version) != meta["Version"]:
        raise ValueError(f"Wheel distribution identity mismatch: {path}")
    checksum = file_digest(path)
    return {
        "name": canonicalize_name(meta["Name"]),
        "version": meta["Version"],
        "filename": path.name,
        "sha256": checksum,
        "source": source,
    }


def file_digest(path: Path) -> str:
    """Hash large framework wheels without retaining their bytes in memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify_wheel(path: Path, package: dict) -> None:
    """Validate downloaded bytes against an automatically generated local snapshot."""
    from packaging.utils import canonicalize_name

    if Path(package["filename"]).name != package["filename"]:
        raise ValueError("Unsafe cached wheel filename")
    actual = wheel_record(path, package.get("source", ""))
    if actual["sha256"] != package["sha256"]:
        raise ValueError(f"Wheel SHA256 mismatch: {path}")
    if actual["name"] != canonicalize_name(package["name"]) or actual["version"] != package["version"]:
        raise ValueError(f"Cached wheel distribution mismatch: {path}")


def environment_key(target: dict, base_python: Path) -> str:
    """Separate version requests, Python runtimes and architectures, not transitive versions."""
    return digest_json(
        {
            "schema": 2,
            "base_python": str(base_python.resolve()),
            "base_version": sys.version,
            "abi": sys.implementation.cache_tag,
            "arch": platform.machine(),
            "request": target,
        }
    )


def pip_environment(python: Path) -> dict:
    """Keep source selection explicit even when the caller has custom pip settings."""
    env = isolated_environment(python)
    for key in list(env):
        if key.startswith("PIP_") and key not in ("PIP_CACHE_DIR", "PIP_NO_CACHE_DIR"):
            env.pop(key)
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def download_roots(target: dict, directory: Path, native: Path, python: Path, run) -> list[Path]:
    """Let pip select each exact framework wheel from its own official source."""
    roots = directory / "roots"
    roots.mkdir()
    result = []
    for name, index, field in (("torch", "torch", "torch_version"), ("torch-npu", "pta", "pta_build_version")):
        cached = STORE.cached_root(native, target, name, target[field], verify_wheel)
        if cached is not None:
            message(f"PTA {target['pta_build_version']}: reusing verified local {name} wheel")
            result.append(cached)
            continue
        destination = roots / name
        destination.mkdir()
        command = [
            str(python),
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--progress-bar",
            "off",
            "--index-url",
            target["indexes"][index],
            "--dest",
            str(destination),
        ]
        run(
            [*command, f"{name}=={target[field]}"], label=f"download-{target['id']}-{name}", env=pip_environment(python)
        )
        wheels = list(destination.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(
                f"No unique {name} wheel for {target['id']}/{sys.implementation.cache_tag}/{platform.machine()}"
            )
        record = wheel_record(wheels[0], target["indexes"][index])
        if record["name"] != name or record["version"] != target[field]:
            raise ValueError(f"Unexpected framework wheel: {record}")
        result.append(wheels[0])
    return result


def report_digest(download_info: dict) -> str:
    """Accept pip 23's legacy hash field as well as newer installation reports."""
    archive = download_info["archive_info"]
    digest = archive.get("hashes", {}).get("sha256")
    if digest is None:
        algorithm, _, digest = archive.get("hash", "").partition("=")
        if algorithm != "sha256":
            raise ValueError("Resolver report requires a SHA256 wheel digest")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid resolver wheel SHA256")
    return digest


def download_resolution(target: dict, directory: Path, native: Path, python: Path, run) -> dict:
    """Resolve necessary dependencies once, then download only that reported closure."""
    from packaging.utils import canonicalize_name

    roots = download_roots(target, directory, native, python, run)
    report = directory / "pip-report.json"
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--only-binary=:all:",
            "--progress-bar",
            "off",
            "--index-url",
            target["indexes"]["dependencies"],
            "--report",
            str(report),
            *map(str, roots),
            *target["build_requirements"],
        ],
        label=f"resolve-{target['id']}",
        env=pip_environment(python),
    )
    entries = json.loads(report.read_text())["install"]
    requirements = directory / "download-requirements.txt"
    wheels = directory / "wheels"
    wheels.mkdir()
    missing = reported_downloads(entries, native, wheels)
    if missing:
        requirements.write_text(
            "".join(f"{e['download_info']['url']} --hash=sha256:{report_digest(e['download_info'])}\n" for e in missing)
        )
        run(
            [
                str(python),
                "-m",
                "pip",
                "download",
                "--no-deps",
                "--require-hashes",
                "--only-binary=:all:",
                "--progress-bar",
                "off",
                "--dest",
                str(wheels),
                "-r",
                str(requirements),
            ],
            label=f"download-closure-{target['id']}",
            env=pip_environment(python),
        )
    packages = resolution_packages(target, entries, wheels)
    # Root URLs in pip's report refer to a transient local directory. Record the configured source instead.
    for entry in entries:
        name = canonicalize_name(entry["metadata"]["name"])
        if name in ("torch", "torch-npu"):
            entry["download_info"]["url"] = next(
                p["source"] for p in packages if p["name"] == name
            )
    report.write_text(json.dumps({"install": entries}, indent=2) + "\n")
    requirements.unlink(missing_ok=True)
    shutil.rmtree(directory / "roots")
    return {"request": target, "packages": packages, "created_at": time.time()}


def reported_downloads(entries, native, wheels):
    """A fresh pip report fixes identity; matching bytes need not be downloaded again."""
    missing = []
    for entry in entries:
        url = entry["download_info"]["url"]
        filename = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
        package = {
            "filename": filename,
            "name": entry["metadata"]["name"],
            "version": entry["metadata"]["version"],
            "sha256": report_digest(entry["download_info"]),
            "source": url,
        }
        cached = STORE.wheel_path(native, package)
        if cached.exists():
            verify_wheel(cached, package)
            try:
                os.link(cached, wheels / filename)
            except OSError:
                shutil.copyfile(cached, wheels / filename)
        else:
            missing.append(entry)
    return missing


def resolution_packages(target: dict, entries: list, directory: Path) -> list[dict]:
    """Reject extra, duplicate or changed artifacts before publishing a resolved wheelhouse."""
    from packaging.utils import canonicalize_name

    expected = {canonicalize_name(e["metadata"]["name"]): e for e in entries}
    if len(expected) != len(entries):
        raise ValueError("Duplicate resolved distributions")
    packages = []
    for path in sorted(directory.iterdir()):
        record = wheel_record(path, "")
        if record["name"] not in expected:
            raise ValueError(f"Unexpected or duplicate resolved wheel: {path}")
        entry = expected.pop(record["name"])
        info = entry["download_info"]
        if record["version"] != entry["metadata"]["version"] or record["sha256"] != report_digest(info):
            raise ValueError(f"Resolved wheel identity/hash mismatch: {path}")
        record["source"] = target["indexes"].get(
            {"torch": "torch", "torch-npu": "pta"}.get(record["name"], ""), info["url"]
        )
        packages.append(record)
    if expected:
        raise ValueError(f"Missing resolved wheels: {sorted(expected)}")
    return packages


def resolved_environment(target: dict, native: Path, python: Path, run, key: str) -> tuple[Path, dict]:
    """Publish a complete resolution atomically; cached corruption never triggers network repair."""
    parent = framework_directory(native, "resolutions")
    parent.mkdir(parents=True, exist_ok=True)
    destination = STORE.snapshot_path(native, target, key)
    if parent.is_symlink():
        raise ValueError(f"Refusing symlinked framework resolution: {parent}")
    reused = destination.exists()
    if destination.exists():
        record = STORE.load_snapshot(destination, target, key)
    else:
        message(f"PTA {target['pta_build_version']}: resolving platform wheels and dependencies; no completed snapshot")
        temporary = Path(tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=parent))
        try:
            record = download_resolution(target, temporary, native, python, run)
            destination, record = STORE.publish_snapshot(temporary, native, record, target, key, verify_wheel)
        finally:
            shutil.rmtree(temporary)
    message(f"PTA {target['pta_build_version']}: verifying {len(record['packages'])} shared local wheels")
    for package in record["packages"]:
        verify_wheel(STORE.wheel_path(native, package), package)
    message(f"PTA {target['pta_build_version']}: wheel snapshot {'reused' if reused else 'ready'}")
    return destination, record


def prepare_environment(target: dict, native: Path, run, base_python: Path, *, identity_probe=None) -> Path:
    """Resolve once and install a local-only snapshot into a project-owned environment."""
    validate_target(target)
    key = environment_key(target, base_python)
    prefix = f"cp{sys.version_info.major}{sys.version_info.minor}-{platform.machine()}-torch{target['torch_release']}-pta{target['pta_build_version']}"
    directory = framework_directory(native, "envs") / f"{prefix}-{key[:20]}"
    python = directory / "bin/python"
    ready = directory / "ready.json"
    if ready.exists():
        message(f"PTA {target['pta_build_version']}: checking cached environment")
        record = json.loads(ready.read_text())
        if record.get("environment_key") != key:
            raise ValueError(f"Corrupt environment completion marker: {directory}")
        destination = STORE.snapshot_path(native, target, key)
        if not destination.is_file():
            raise ValueError(f"Missing successful environment snapshot: {destination}")
        _, snapshot = resolved_environment(target, native, python, run, key)
        STORE.update_reference(native, directory, destination, snapshot)
        verify_environment(python, record, run, identity_probe=identity_probe)
        environment_summary(target, python, record, "REUSE")
        return python
    if directory.exists():
        message(f"PTA {target['pta_build_version']}: removing incomplete environment", "WARN")
        shutil.rmtree(directory)
    message(f"PTA {target['pta_build_version']}: creating environment; no completed match for Python/platform/request")
    directory.mkdir(parents=True)
    pending = directory / "pending.json"
    pending.write_text(
        json.dumps(
            {
                "owner": "hyper-multicore-framework-env",
                "environment_key": key,
                "target": target["id"],
                "pta_release": target["pta_release"],
                "created_at": time.time(),
            }
        )
    )
    run([str(base_python), "-m", "venv", str(directory)], label=f"venv-{target['id']}")
    resolution, record = resolved_environment(target, native, python, run, key)
    message(f"PTA {target['pta_build_version']}: installing {len(record['packages'])} verified local wheels")
    install = directory / "install-requirements.txt"
    STORE.install_manifest(native, record, install)
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-cache-dir",
            "--no-deps",
            "--require-hashes",
            "--only-binary=:all:",
            "-r",
            str(install),
        ],
        label=f"install-{target['id']}",
        env=pip_environment(python),
    )
    verify_environment(python, record, run, identity_probe=identity_probe)
    record.update(
        environment_key=key, target=target["id"], pta_release=target["pta_release"], resolution=str(resolution)
    )
    temporary = directory / "ready.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, ready)
    pending.unlink(missing_ok=True)
    environment_summary(target, python, record, "READY")
    return python


def environment_summary(target: dict, python: Path, record: dict, action: str):
    """Print actual installed versions for both newly created and reused environments."""
    detail = "verified; no download or reinstallation" if action == "REUSE" else "installed and verified"
    message(
        f"PTA {target['pta_build_version']}: environment {'reused' if action == 'REUSE' else 'ready'}; {len(record['packages'])} packages {detail}"
    )
    packages = " ".join(f"{p['name']}=={p['version']}" for p in sorted(record["packages"], key=lambda p: p["name"]))
    print(
        "  packages:\n"
        + textwrap.fill(
            packages,
            width=100,
            initial_indent="    ",
            subsequent_indent="    ",
            break_on_hyphens=False,
            break_long_words=False,
        ),
        flush=True,
    )


def verify_environment(python: Path, record: dict, run, *, identity_probe=None) -> None:
    """Verify the actual successful snapshot without requiring its versions in source control."""
    expected = {p["name"]: p["version"] for p in record["packages"]}
    code = 'from importlib.metadata import version\nimport json,sys\nexpected=json.loads(sys.argv[1])\nactual={n:version(n) for n in expected}\nif actual != expected:\n    raise RuntimeError(f"Framework environment changed: {actual!r} != {expected!r}")'
    if identity_probe is None:
        run(
            [str(python), "-c", code, json.dumps(expected)],
            label=f"identity-{python.parent.parent.name}",
            env=isolated_environment(python),
        )
    else:
        identity_probe(python, expected)
    run(
        [str(python), "-m", "pip", "check"], label=f"pip-check-{python.parent.parent.name}", env=pip_environment(python)
    )


def environment_record(directory: Path):
    """Read successful environments or explicitly owned abandoned installs only."""
    marker = directory / "ready.json"
    if marker.is_file():
        return json.loads(marker.read_text())
    try:
        record = json.loads((directory / "pending.json").read_text())
    except (OSError, ValueError):
        return None
    key = record.get("environment_key", "")
    if (
        record.get("owner") == "hyper-multicore-framework-env"
        and re.fullmatch(r"[0-9a-f]{64}", key)
        and directory.name.endswith(key[:20])
    ):
        return record
    return None


def rollback_environments(candidates, branches):
    """Retain exactly the newest previously successful rollback for each active branch."""
    keep = set()
    for branch in branches:
        successful = [
            (d, m) for d, m in candidates if m.get("pta_release") == branch and m.get("last_successful_build")
        ]
        if successful:
            keep.add(max(successful, key=lambda item: item[1]["last_successful_build"])[0])
    return keep


def prune_environments(native: Path, selected: list[Path], branches: set[str]) -> list[str]:
    """After full success retain selected environments and one successful rollback per branch."""
    current = {p.parent.parent.resolve() for p in selected}
    candidates = []
    for directory in (framework_directory(native, "envs")).iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        record = environment_record(directory)
        if record is None:
            continue
        if directory.resolve() in current:
            record["last_successful_build"] = time.time()
            atomic_json(directory / "ready.json", record)
        else:
            candidates.append((directory, record))
    keep = rollback_environments(candidates, branches)
    removed = []
    for directory, _ in candidates:
        if directory not in keep:
            shutil.rmtree(directory)
            removed.append(str(directory))
    return removed
