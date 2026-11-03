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
"""Authenticate compile/link probes and retain referenced preflight workspaces."""

import json
import platform
import re
import shlex
import shutil
import sys
import time
from pathlib import Path

from . import build_workspace as WORKSPACE
from .build_utils import atomic_json
from .command_runner import message


SCHEMA_VERSION = 2


def development_inputs(paths, cann, digest):
    """Hash development trees actually searched by the probe, including CMake contracts."""
    files = {}
    for directory in (Path(paths["python_include"]), Path(paths["torch_cmake"]), cann / "include"):
        if not directory.is_dir():
            raise ValueError(f"Missing preflight development directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                files[str(path)] = digest(path)
    return files


def configuration_options(contract):
    if contract is None:
        return []
    options = ["-G", contract["generator"]["name"]]
    if "backend" in contract["programs"]:
        options.append(f"-DCMAKE_MAKE_PROGRAM={contract['programs']['backend']['path']}")
    return options


def verify_configured_tools(work, contract, digest):
    """Refuse to certify a configure step that selected a different effective toolchain."""
    if contract is None:
        return
    values = {}
    for line in (work / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and ":" in line.partition("=")[0]:
            key, _, value = line.partition("=")
            values[key.partition(":")[0]] = value
    if values.get("CMAKE_GENERATOR") != contract["generator"]["name"]:
        raise RuntimeError("Configured preflight generator differs from its fingerprint")
    configured = {"backend": "CMAKE_MAKE_PROGRAM", "compiler": "CMAKE_CXX_COMPILER"}
    for name, variable in configured.items():
        if name not in contract["programs"]:
            continue
        path = Path(values[variable]).resolve()
        expected = contract["programs"][name]
        if str(path) != expected["path"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"Configured preflight {name} differs from its fingerprint")


def dependency_files(work, run, env, version):
    """Read generated Make/Ninja compiler dependencies and included CMake modules."""
    files = set(ninja_dependencies(work, run, env, version))
    for depfile in work.rglob("*.o.d"):
        contents = depfile.read_text().replace("\\\n", " ")
        files.update(Path(p) for p in shlex.split(contents.partition(":")[2]) if Path(p).is_absolute())
    makefile = work / "CMakeFiles/Makefile.cmake"
    if makefile.exists():
        contents = makefile.read_text()
        block = re.search(r"set\(CMAKE_MAKEFILE_DEPENDS(.*?)\n\s*\)", contents, re.S)
        if block:
            files.update(Path(p) if Path(p).is_absolute() else work / p for p in re.findall(r'"([^"\n]+)"', block[1]))
    return sorted(p for p in files if p.is_file())


def ninja_dependencies(work, run, env, version):
    """Ninja consumes depfiles; query its retained dependency database instead."""
    build = work / "build.ninja"
    if not build.is_file():
        return []
    text = run(
        ["ninja", "-C", str(work), "-t", "deps"],
        label=f"preflight-compiler-dependencies-{version}",
        env=env,
        capture=True,
    )
    files = [Path(line.strip()) for line in text.splitlines() if line.startswith(" ") and line.strip().startswith("/")]
    for line in build.read_text().splitlines():
        if "RERUN_CMAKE" in line:
            paths = line.partition("|")[2].replace("$ ", "\\ ")
            files.extend(Path(p) for p in shlex.split(paths) if Path(p).is_absolute())
    return files


def linked_files(output, env, run, version):
    text = run(["ldd", str(output)], label=f"preflight-dependencies-{version}", env=env, capture=True)
    if "not found" in text:
        raise RuntimeError(f"Missing preflight link dependency: {text}")
    return sorted(set(re.findall(r"(?:=>\s+)?(/[^\s]+)\s+\(", text)))


def valid_record(work, key, env, run, version, digest):
    marker = work / "complete.json"
    if not marker.is_file():
        return False, "no verified preflight success record"
    try:
        record = json.loads(marker.read_text())
        if record.get("schema_version") != SCHEMA_VERSION or record.get("fingerprint") != key:
            return False, "preflight success record does not match inputs"
        output = work / "libhyper_multicore_framework_probe.so"
        if digest(output) != record["output_sha256"]:
            return False, "preflight output changed"
        if any(digest(Path(p)) != value for p, value in record["extra_inputs"].items()):
            return False, "consumed preflight dependency changed"
        if linked_files(output, env, run, version) != record["linked_files"]:
            return False, "resolved preflight libraries changed"
    except (OSError, ValueError, KeyError):
        return False, "preflight record or dependency missing/invalid"
    return True, "verified preflight inputs and outputs reused"


def complete_probe(work, key, digest, env, run, version, started, guard, write_json):
    output = work / "libhyper_multicore_framework_probe.so"
    linked = linked_files(output, env, run, version)
    files = sorted(set(dependency_files(work, run, env, version)) | {Path(p) for p in linked})
    for path in files:
        if not path.is_relative_to(work) and max(path.stat().st_mtime_ns, path.stat().st_ctime_ns) > started:
            raise RuntimeError(f"Preflight dependency changed during compilation: {path}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": key,
        "output_sha256": digest(output),
        "extra_inputs": {str(p): digest(p) for p in files},
        "linked_files": linked,
    }
    digest.validate()
    if guard:
        guard.check_source()
        guard.created.add(work)
    write_json(work / "complete.json", record)
    digest.validate()
    if guard:
        guard.check_source()


def ensure_probe(work, key, args, python, env, run, version, digest, guard, root, write_json, contract=None):
    """Only certify successful frozen inputs; failures and interruption revoke certification."""
    WORKSPACE.check_workspace(work, root)
    forced = args.clean or (contract is not None and not contract["reusable"])
    hit, reason = (
        (False, "forced preflight validation") if forced else valid_record(work, key, env, run, version, digest)
    )
    if hit:
        digest.validate()
        return True, reason
    if work.is_symlink():
        raise ValueError(f"Refusing symlinked preflight work directory: {work}")
    digest.discard_tree(work)
    if work.exists():
        shutil.rmtree(work)
    WORKSPACE.claim_workspace(work, root)
    started = time.time_ns()
    try:
        run(
            [
                "cmake",
                "-S",
                str(root / "hyper_multicore/cmake/framework_probe"),
                "-B",
                str(work),
                f"-DPython3_EXECUTABLE={python}",
                "-DCMAKE_BUILD_TYPE=Release",
                *configuration_options(contract),
            ],
            label=f"preflight-configure-{version}",
            env=env,
        )
        run(
            ["cmake", "--build", str(work), "--parallel", str(args.jobs)], label=f"preflight-compile-{version}", env=env
        )
        run([str(python), "-c", "import ctypes,os,sys,torch; ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW)",
             str(work / "libhyper_multicore_framework_probe.so")],
            label=f"preflight-load-{version}", env=env, capture=True)
        verify_configured_tools(work, contract, digest)
        complete_probe(work, key, digest, env, run, version, started, guard, write_json)
    except BaseException:
        (work / "complete.json").unlink(missing_ok=True)
        raise
    return False, reason


def read_index(path):
    """An invalid index is never evidence that a workspace is disposable."""
    if not path.exists():
        return {"schema": 2, "platforms": {}}, True
    try:
        value = json.loads(path.read_text())
        if path.is_symlink() or value.get("schema") != 2 or not isinstance(value["platforms"], dict):
            return None, False
        for modes in value["platforms"].values():
            if not isinstance(modes, dict) or set(modes) - {"all", "current"}:
                return None, False
            for refs in modes.values():
                if not isinstance(refs, list) or not refs or any(not valid_reference(ref) for ref in refs):
                    return None, False
        return value, False
    except (OSError, ValueError, KeyError, AttributeError):
        return None, False


def valid_reference(ref):
    return (
        isinstance(ref, dict)
        and isinstance(ref.get("fingerprint"), str)
        and re.fullmatch(r"[0-9a-f]{64}", ref["fingerprint"]) is not None
        and isinstance(ref.get("directory"), str)
        and WORKSPACE.FRAMEWORK_SLOT.fullmatch(ref["directory"]) is not None
    )


def owned_workspace(path, root):
    """Uncertified probes are owned only with an attributable CMake source."""
    if path.is_symlink() or not path.is_dir() or not WORKSPACE.FRAMEWORK_SLOT.fullmatch(path.name):
        return False
    cache = path / "CMakeCache.txt"
    if cache.is_symlink():
        return False
    try:
        lines = cache.read_text().splitlines()
    except (OSError, UnicodeError):
        return False
    expected = str((root / "hyper_multicore/cmake/framework_probe").resolve())
    return f"CMAKE_HOME_DIRECTORY:INTERNAL={expected}" in lines


def collect_unreferenced(directory, keep, root):
    removed, retained = [], []
    for path in directory.iterdir():
        if path.name == "index.json" or path.name in keep:
            continue
        if owned_workspace(path, root):
            shutil.rmtree(path)
            removed.append(path.name)
        else:
            retained.append(path.name)
    return removed, retained


def finish(native, root, mode, events):
    """Called after the build transaction commits, under the same exclusive build lock."""
    directory = native / "preflight"
    index, missing = read_index(directory / "index.json")
    if index is None:
        message("Preflight retention: invalid reference index; preserving all workspaces", "WARN")
        events.append({"stage": "preflight-retention", "status": "invalid-index", "removed": []})
        return
    scope = f"cp{sys.version_info.major}{sys.version_info.minor}-{platform.machine()}"
    refs = [
        {"fingerprint": e["fingerprint"], "directory": e["directory"]} for e in events if "preflight_cache_hit" in e
    ]
    if not refs or any(not valid_reference(ref) for ref in refs):
        raise ValueError("Cannot register incomplete preflight references")
    index["schema"] = 2
    modes = index["platforms"].setdefault(scope, {})
    modes[mode] = refs
    atomic_json(directory / "index.json", index)
    keep = {ref["directory"] for group in index["platforms"].values() for items in group.values() for ref in items}
    removed, retained = ([], [])
    if "all" in modes and not (missing and mode != "all"):
        removed, retained = collect_unreferenced(directory, keep, root)
    events.append(
        {
            "stage": "preflight-retention",
            "mode": mode,
            "protected": sorted(keep),
            "removed": removed,
            "unknown_retained": retained,
        }
    )
    message(f"Preflight retention: {len(keep)} referenced workspaces, {len(removed)} old workspaces removed")
    if retained:
        message(f"Preflight retention: preserving {len(retained)} unattributed entries", "WARN")
