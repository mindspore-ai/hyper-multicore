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
"""One local developer vendor snapshot per architecture/SoC; never cache framework adapters."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time

from .build_utils import atomic_json
from .build_workspace import check_workspace, claim_workspace, safe_component
from .build_publication import publish_payload
from .command_runner import message


SCRIPTS = (
    "build.sh", "_build/assemble_multicore_source.py", "_build/merge_multicore_vendors.py",
    "_build/vendor_cache.py", "_build/prepare_dependencies.py", "_build/dependencies.lock.json",
    "cmake/hardening.cmake",
    "shmem/_build/shmem_sdk.sh",
)
SOURCE_TREES = ("ops", "cmake/vendor", "shmem/cmake/shmem_wrapper", "shmem/_build/patches")
ENVIRONMENT = (
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "ASCEND_HOME_PATH", "CC", "CXX", "CFLAGS", "CXXFLAGS",
    "CPPFLAGS", "LDFLAGS", "CMAKE_BUILD_TYPE", "CMAKE_GENERATOR", "CMAKE_PREFIX_PATH",
    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
)


def sha256(path: Path) -> str:
    """Hash file contents without loading compiler binaries into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_files(root: Path) -> dict:
    """Check exact snapshot membership without following redirected cache contents."""
    if root.is_symlink():
        raise ValueError(f"Symlinked vendor snapshot root: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinked vendor cache content: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256(path)
    if not files:
        raise ValueError(f"Empty vendor inputs or snapshot: {root}")
    return files


def development_files(root: Path) -> dict:
    """Record SDK/source file contents and resolved symlink targets, including device headers."""
    return {str(p.relative_to(root)): {"path": str(p.resolve()), "sha256": sha256(p)}
            for p in sorted(root.rglob("*")) if p.is_file()}


def tool_inputs(environment: dict) -> dict:
    """Identify the selected tools and compiler wrappers by actual executable bytes."""
    names = ["bisheng", "asc_opc", "gcc", "g++", "ld", "as", "ar", "cmake", "make"]
    names.extend(shlex.split(environment[name])[0] for name in ("CC", "CXX") if environment[name])
    result = {}
    for name in names:
        found = shutil.which(name, path=environment["PATH"])
        if found is None:
            raise ValueError(f"Cannot identify vendor tool: {name}")
        path = Path(found).resolve()
        result[name] = {"path": str(path), "sha256": sha256(path)}
    return result


def vendor_inputs(root: Path, sdk: Path, soc: str, environment: dict) -> dict:
    """A bounded local fingerprint, not certification of an arbitrary CANN installation."""
    component = root / "hyper_multicore"
    cann = Path(environment["ASCEND_HOME_PATH"]).resolve()
    version = cann / "opp/version.info"
    if not version.is_file():
        raise ValueError("CANN version record unavailable")
    sources = {name: sha256(component / name) for name in SCRIPTS}
    for name in SOURCE_TREES:
        sources[name] = development_files(component / name)
    sdk_files = {name: development_files(sdk / name) for name in ("include", "src/device", "lib")}
    if any(not files for files in sdk_files.values()):
        raise ValueError("SDK vendor inputs unavailable")
    return {"schema": 1, "arch": platform.machine(), "soc": soc, "sources": sources,
            "sdk": sdk_files, "cann_root": str(cann), "cann_version_info": version.read_text(),
            "tools": tool_inputs(environment), "environment": environment,
            "python": {"path": str(Path(sys.executable).resolve()), "version": sys.version}}


def fingerprint(inputs: dict) -> str:
    """Use one stable digest for the explicitly bounded vendor inputs."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_snapshot(directory: Path, inputs: dict) -> dict | None:
    """Damaged or mismatched owned snapshots are misses; never relax path ownership."""
    try:
        record = json.loads((directory / "record.json").read_text())
        if (record["fingerprint"] == fingerprint(inputs) and isinstance(record.get("origin"), dict)
                and record["files"] == tree_files(directory / "vendor")):
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def prepare(root: Path, native: Path, soc: str, sdk: Path, request_path: Path, clean: bool = False) -> None:
    """Stage a verified hit or a request to publish a freshly built vendor after commit."""
    started = time.monotonic()
    directory = native / "vendor-cache" / safe_component(platform.machine() + "-" + soc)
    check_workspace(directory, root)
    if clean and directory.exists():
        shutil.rmtree(directory)
    environment = {name: os.environ.get(name, "") for name in ENVIRONMENT}
    request = {"soc": soc, "directory": str(directory), "sdk": str(sdk), "environment": environment,
               "reused": False, "origin": json.loads(os.environ.get("HP_VENDOR_ORIGIN", "{}"))}
    reason = "--clean requested" if clean else "no matching complete snapshot"
    try:
        inputs = vendor_inputs(root, sdk, soc, environment)
        request["fingerprint"] = fingerprint(inputs)
        request["origin"]["environment"] = {k: inputs[k] for k in
                                            ("cann_root", "cann_version_info", "tools", "environment", "python")}
        record = None if clean else read_snapshot(directory, inputs)
        if record is not None:
            reused = request_path.parent / ("vendor-reuse-" + soc) / "hyper_multicore_nn"
            shutil.copytree(directory / "vendor", reused)
            request.update(reused=True, candidate=str(reused), origin=record["origin"])
            reason = "cache hit"
    except (OSError, ValueError) as error:
        request.pop("fingerprint", None)
        reason = f"cache unavailable; rebuilding ({error})"
    request["lookup_seconds"] = time.monotonic() - started
    atomic_json(request_path, request)
    message(f"Vendor {soc}: {reason}; checked in {request['lookup_seconds']:.2f}s")


def built(request_path: Path, candidate: Path, started: float) -> None:
    """Remember the fresh candidate, without publishing a reusable cache entry."""
    request = json.loads(request_path.read_text())
    request.update(reused=False, candidate=str(candidate), compile_seconds=time.time() - started)
    atomic_json(request_path, request)
    message(f"Vendor {request['soc']}: compiled in {request['compile_seconds']:.1f}s")


def publish(root: Path, requests: list[dict]) -> None:
    """Only the committed build calls this; failed maintenance cannot undo its artifact."""
    for request in requests:
        if request["reused"] or "fingerprint" not in request:
            continue
        started = time.monotonic()
        inputs = vendor_inputs(root, Path(request["sdk"]), request["soc"], request["environment"])
        if fingerprint(inputs) != request["fingerprint"]:
            message(f"Vendor {request['soc']}: inputs changed; snapshot not saved", "WARN")
            continue
        candidate = Path(request["candidate"])
        staged = candidate.parent / "cache-publication"
        check_workspace(staged, root)
        claim_workspace(staged, root)
        shutil.copytree(candidate, staged / "vendor")
        atomic_json(staged / "record.json", {"fingerprint": fingerprint(inputs), "inputs": inputs,
                    "files": tree_files(staged / "vendor"), "origin": request["origin"]})
        directory = Path(request["directory"])
        check_workspace(directory, root)
        # The staging ownership describes its final location, without changing the cache protocol.
        marker = {"schema": 1, "root": str(root.resolve()), "kind": directory.parent.name, "slot": directory.name}
        atomic_json(staged / ".workspace.json", marker)
        directory.parent.mkdir(parents=True, exist_ok=True)
        publish_payload(staged, directory)
        request["publish_seconds"] = time.monotonic() - started
        message(f"Vendor {request['soc']}: snapshot saved in {request['publish_seconds']:.2f}s")


def main() -> None:
    """Serve the private common-build handoff; no public build options are added."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare_args = sub.add_parser("prepare")
    for name in ("root", "native", "sdk", "request"):
        prepare_args.add_argument("--" + name, type=Path, required=True)
    prepare_args.add_argument("--soc", required=True)
    prepare_args.add_argument("--clean", action="store_true")
    built_args = sub.add_parser("built")
    built_args.add_argument("--request", type=Path, required=True)
    built_args.add_argument("--candidate", type=Path, required=True)
    built_args.add_argument("--started", type=float, required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.root, args.native, args.soc, args.sdk, args.request, args.clean)
    else:
        built(args.request, args.candidate, args.started)


if __name__ == "__main__":
    main()
