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
"""Exact framework pairs in reusable isolated environments, with pip-managed dependencies."""

from __future__ import annotations
import hashlib
from importlib import import_module
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
from .build_workspace import environment_directory
from .build_utils import atomic_json


def digest_json(value) -> str:
    """Return a deterministic input identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_targets(path: Path) -> list[dict]:
    """Read version pairs independently of the current interpreter's wheel tags."""
    version_type = import_module("packaging.version").Version

    config = json.loads(path.read_text())
    if config.get("schema_version") != 2:
        raise ValueError("Unsupported framework matrix schema")
    reference = config.get("reference_cann")
    if reference is not None and not isinstance(reference, str):
        raise ValueError("reference_cann must be a string or null")
    targets = [
        {
            **target,
            "indexes": config["indexes"],
            "build_requirements": config["build_requirements"],
            "torch_release": version_type(target["torch_version"]).base_version,
            "pta_release": version_type(target["pta_build_version"]).base_version,
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
    version_type = import_module("packaging.version").Version

    for field in ("id", "torch_release", "pta_release", "torch_version", "pta_build_version"):
        if not isinstance(target.get(field), str) or not re.fullmatch(r"[a-zA-Z0-9.+_-]+", target[field]):
            raise ValueError(f"Unsafe framework target field: {field}")
    for distribution, field in (("torch", "torch_version"), ("pta", "pta_build_version")):
        version = version_type(target[field])
        if version.base_version != target[f"{distribution}_release"]:
            raise ValueError(f"Framework branch differs from build version: {field}")
    validate_sources_and_requirements(target)


def validate_index_url(address: str, name: str) -> None:
    """Apply the same HTTPS source rules to primary and additional indexes."""
    if not isinstance(address, str):
        raise ValueError(f"Framework index must be an HTTPS URL string: {name}")
    url = urllib.parse.urlparse(address)
    if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError(f"Framework indexes require credential-free HTTPS URLs: {name}")


def validate_sources_and_requirements(target: dict):
    """Validate common resolver settings without prescribing a complete package list."""
    requirement_type = import_module("packaging.requirements").Requirement
    canonicalize_name = import_module("packaging.utils").canonicalize_name

    for name in ("torch", "pta", "dependencies"):
        validate_index_url(target["indexes"][name], name)
    extra = target["indexes"].get("dependencies_extra", [])
    if not isinstance(extra, list):
        raise ValueError("indexes.dependencies_extra must be a list of HTTPS URLs")
    for address in extra:
        validate_index_url(address, "dependencies_extra")
    for text in target["build_requirements"]:
        requirement = requirement_type(text)
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
    storage = environment_directory(Path(__file__).resolve().parents[2] / "build/native")
    # Framework-owned paths are chosen from the target interpreter, never inherited from another venv.
    for key in ("LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "CMAKE_PREFIX_PATH"):
        if key in env:
            env[key] = os.pathsep.join(
                p
                for p in env[key].split(os.pathsep)
                if "site-packages/torch" not in p
                and "/build/native/envs/" not in p
                and not Path(p).resolve().is_relative_to(storage)
                and "site-packages/hyper_multicore" not in p
                and "/build/native/payload/" not in p
            )
    return env


def build_environment(python: Path, *, isolated=False) -> dict:
    """Use the caller's development environment unless a matrix venv is selected."""
    env = isolated_environment(python) if isolated else dict(os.environ)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    return env


def cann_tool_environment(python: Path, *, isolated=False) -> dict:
    """Add available toolkit modules without replacing developer search paths."""
    env = build_environment(python, isolated=isolated)
    cann = Path(env["ASCEND_HOME_PATH"]).resolve()
    paths = [cann / "python/site-packages", cann / "opp/built-in/op_impl/ai_core/tbe"]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH", ""),
                                                   *(str(p) for p in paths if p.is_dir())]))
    env["HI_PYTHON"] = str(python)
    return env


def wheel_record(path: Path, source: str) -> dict:
    """Inspect a selected wheel using current interpreter tags and primary metadata."""
    sys_tags = import_module("packaging.tags").sys_tags
    utilities = import_module("packaging.utils")

    filename_name, filename_version, _, tags = utilities.parse_wheel_filename(path.name)
    if not tags.intersection(sys_tags()):
        raise ValueError(f"Wheel tags do not match build platform: {path.name}")
    with zipfile.ZipFile(path) as archive:
        entries = [n for n in archive.namelist() if len(Path(n).parts) == 2 and n.endswith(".dist-info/METADATA")]
        if len(entries) != 1:
            raise ValueError(f"Ambiguous wheel metadata: {path}")
        meta = BytesParser().parsebytes(archive.read(entries[0]))
    if utilities.canonicalize_name(meta["Name"]) != filename_name or str(filename_version) != meta["Version"]:
        raise ValueError(f"Wheel distribution identity mismatch: {path}")
    checksum = file_digest(path)
    return {
        "name": utilities.canonicalize_name(meta["Name"]),
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


def environment_request(target: dict) -> dict:
    """Keep extra mirrors out of environment identity without modifying provenance."""
    request = dict(target)
    indexes = request.get("indexes")
    if isinstance(indexes, dict):
        request["indexes"] = {name: value for name, value in indexes.items() if name != "dependencies_extra"}
    return request


def environment_key(target: dict, base_python: Path) -> str:
    """Separate version requests, Python runtimes and architectures, not transitive versions."""
    return digest_json(
        {
            "schema": 2,
            "base_python": str(base_python.resolve()),
            "base_version": sys.version,
            "abi": sys.implementation.cache_tag,
            "arch": platform.machine(),
            "request": environment_request(target),
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


def extra_index_arguments(target: dict) -> list[str]:
    """Use the configured extra indexes for framework downloads and dependency installs."""
    return [argument for address in target["indexes"].get("dependencies_extra", [])
            for argument in ("--extra-index-url", address)]


def download_roots(target: dict, directory: Path, python: Path, run) -> list[Path]:
    """Let pip select exact framework wheels from configured primary and extra indexes."""
    roots = directory / "roots"
    roots.mkdir()
    result = []
    for name, index, field in (("torch", "torch", "torch_version"), ("torch-npu", "pta", "pta_build_version")):
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
            *extra_index_arguments(target),
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


def prepare_environment(target: dict, native: Path, run, base_python: Path, *, identity_probe=None) -> Path:
    """Reuse an exact framework pair without requiring its historical download snapshot."""
    validate_target(target)
    key = environment_key(target, base_python)
    prefix = (f"cp{sys.version_info.major}{sys.version_info.minor}-{platform.machine()}"
              f"-torch{target['torch_release']}-pta{target['pta_build_version']}")
    directory = environment_directory(native) / f"{prefix}-{key[:20]}"
    python = directory / "bin/python"
    ready = directory / "ready.json"
    if directory.is_symlink() or ready.is_symlink():
        raise ValueError(f"Refusing symlinked framework environment: {directory}")
    reused = ready.is_file()
    if reused:
        record = json.loads(ready.read_text())
        previous = record.get("request")
        if (record.get("environment_key") != key or not isinstance(previous, dict)
                or environment_request(previous) != environment_request(target)):
            raise ValueError(f"Corrupt environment completion marker: {directory}")
        message(f"PTA {target['pta_build_version']}: checking installed framework pair")
    else:
        if directory.exists():
            if environment_record(directory) is None:
                raise ValueError(f"Unattributed framework environment: {directory}")
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        record = {"owner": "hyper-multicore-framework-env", "environment_key": key,
                  "request": target, "target": target["id"], "pta_release": target["pta_release"],
                  "created_at": time.time()}
        atomic_json(directory / "pending.json", record)
        message(f"PTA {target['pta_build_version']}: creating isolated environment")
        run([str(base_python), "-m", "venv", str(directory)], label=f"venv-{target['id']}")
        record["installation"] = install_framework(target, directory, python, run)
    record["installed_packages"] = verify_environment(python, target, run, identity_probe=identity_probe)
    atomic_json(ready, record)
    (directory / "pending.json").unlink(missing_ok=True)
    environment_summary(target, record, "REUSE" if reused else "READY")
    return python


def install_framework(target, directory, python, run):
    """Use pip's download cache and resolver; retain provenance, not a private wheelhouse."""
    canonicalize_name = import_module("packaging.utils").canonicalize_name

    with tempfile.TemporaryDirectory(prefix="framework-install-", dir=directory) as temporary:
        temporary = Path(temporary)
        roots = download_roots(target, temporary, python, run)
        report = temporary / "pip-report.json"
        run([str(python), "-m", "pip", "install", "--only-binary=:all:", "--progress-bar", "off",
             "--index-url", target["indexes"]["dependencies"], *extra_index_arguments(target),
             "--report", str(report),
             *map(str, roots), *target["build_requirements"]],
            label=f"install-{target['id']}", env=pip_environment(python))
        installation = json.loads(report.read_text())
        # Local root URLs would point into a deleted temporary directory.
        for entry in installation["install"]:
            name = canonicalize_name(entry["metadata"]["name"])
            if name in ("torch", "torch-npu"):
                entry["download_info"]["url"] = target["indexes"]["torch" if name == "torch" else "pta"]
        return installation


def environment_summary(target: dict, record: dict, action: str):
    """Print actual installed versions for both newly created and reused environments."""
    detail = "Torch/PTA pair verified; no download or reinstallation" if action == "REUSE" else "installed and verified"
    message(
        f"PTA {target['pta_build_version']}: environment {'reused' if action == 'REUSE' else 'ready'}; "
        f"{len(record['installed_packages'])} installed packages; {detail}"
    )
    packages = " ".join(f"{p['name']}=={p['version']}" for p in sorted(record["installed_packages"], key=lambda p: p["name"]))
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


def verify_environment(python: Path, target: dict, run, *, identity_probe=None) -> list:
    """Only Torch/PTA versions constrain reuse; observe all other installed versions."""
    expected = {"torch": target["torch_version"], "torch-npu": target["pta_build_version"]}
    if identity_probe is not None:
        return identity_probe(python, expected)
    code = (
        "import json,sys; from importlib import metadata; "
        "expected=json.loads(sys.argv[1]); actual={n:metadata.version(n) for n in expected}\n"
        "if actual != expected: "
        "raise RuntimeError(f'Framework environment changed: {actual!r} != {expected!r}')\n"
        "print(json.dumps(sorted([{'name':d.metadata['Name'],'version':d.version} "
        "for d in metadata.distributions()],key=lambda p:p['name'].lower())))"
    )
    return json.loads(run([str(python), "-c", code, json.dumps(expected)],
                          label=f"identity-{python.parent.parent.name}",
                          env=isolated_environment(python), capture=True))


def environment_record(directory: Path):
    """Read successful environments or explicitly owned abandoned installs only."""
    marker = directory / "ready.json"
    if marker.is_symlink() or (directory / "pending.json").is_symlink():
        return None
    if marker.is_file():
        try:
            record = json.loads(marker.read_text())
        except (OSError, ValueError):
            return None
        key = record.get("environment_key", "")
        if (re.fullmatch(r"[0-9a-f]{64}", key) and directory.name.endswith(key[:20])
                and isinstance(record.get("request"), dict)):
            return record
        return None
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
    for directory in environment_directory(native).iterdir():
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
