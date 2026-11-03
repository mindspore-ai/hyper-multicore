# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Build-time artifact identity and payload validation; no runtime package import."""

from __future__ import annotations
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
import time
from contextlib import contextmanager
from packaging.version import Version
from ._bootstrap import load_module
from .build_utils import version_label

COMPONENT_VERSION = "0.1.0.dev0"
_MANIFEST = load_module("native_manifest")
REQUIRED_LIBRARIES = (
    "lib/vendors/hyper_multicore_nn/op_api/lib/libcust_opapi.so",
    "shmem/lib/libhyper_multicore_shmem_ops.so",
    "shmem/lib/shmem/libhyper_multicore_shmem.so",
    "shmem/lib/shmem/libhyper_multicore_shmem_utils.so",
    "shmem/lib/shmem/aclshmem_bootstrap_uid.so",
    "shmem/lib/shmem/aclshmem_bootstrap_config_store.so",
)
_VENDOR_OPERATORS = (
    "hyper_mega_moe",
    "hyper_mega_moe_grad",
    "hyper_mega_mhc",
    "hyper_mega_mhc_grad",
    "hyper_mega_gate_route",
    "hyper_mega_gate_route_grad",
)


def artifact_version(pta_version: str) -> str:
    """Return the component version tagged with the actual PTA distribution version.

    Args:
        pta_version: Full installed torch-npu version, preserving post releases.

    Returns:
        Component version with a safe local label; the original PTA version remains in metadata.
    """
    return str(Version(f"{COMPONENT_VERSION}+pt.{version_label(pta_version)}"))


def torch_op_plugin_version(torch_root: Path) -> str:
    """Read the compiled Torch interface version independently of its distribution label."""
    header = torch_root / "include/torch/csrc/api/include/torch/version.h"
    contents = header.read_text()
    if re.search(r"^#include <torch/headeronly/version.h>", contents, re.MULTILINE):
        header = torch_root / "include/torch/headeronly/version.h"
        contents = header.read_text()
    numbers = []
    for component in ("MAJOR", "MINOR"):
        values = re.findall(rf"^#define TORCH_VERSION_{component} ([0-9]+)", contents, re.MULTILINE)
        if len(values) != 1:
            raise RuntimeError(f"Missing or ambiguous Torch {component} definition in {header}")
        numbers.append(values[0])
    return f"V{numbers[0]}R{numbers[1]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(root: Path) -> dict:
    inputs = {}
    for directory in ("hyper_multicore", "scripts"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and (
                path.name == "hyper_multicore_build_info"
                or path.suffix
                in {".py", ".asc", ".cpp", ".cc", ".h", ".hpp", ".sh", ".bash", ".patch", ".json", ".ini", ".cmake", ".txt"}
            ):
                if any(part in {"lib", "__pycache__"} for part in path.relative_to(root).parts):
                    continue
                inputs[path.relative_to(root).as_posix()] = _sha256(path)
    for name in ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt"):
        inputs[name] = _sha256(root / name)
    native_test = root / "tests/ut/shmem/native_log_test.cpp"
    inputs["tests/ut/shmem/native_log_test.cpp"] = _sha256(native_test) if native_test.is_file() else None
    build_commit = None
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=False, capture_output=True)
        if result.returncode == 0:
            build_commit = result.stdout.decode().strip()
    return {
        "build_commit": build_commit,
        "source_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
    }


@contextmanager
def identity_phase(timings, name):
    """Diagnostic timings never enter artifact fingerprints."""
    started = time.monotonic()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.monotonic() - started


class ProbeHashes:
    """Read overlapping inputs once per probe and reject changing file generations."""

    def __init__(self):
        self.files = {}

    def __call__(self, path):
        path = Path(path)
        stat = path.stat()
        generation = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if path in self.files:
            old, value = self.files[path]
            if old != generation:
                raise RuntimeError(f"Framework input changed during identity collection: {path}")
            return value
        value = _sha256(path)
        after = path.stat()
        if generation != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError(f"Framework input changed during identity collection: {path}")
        self.files[path] = generation, value
        return value

    def validate(self):
        """Keep generation checks even for files read only once at probe start."""
        for path in self.files:
            self(path)

    def discard_tree(self, directory):
        """Discard generated probe files before intentionally recreating their work tree."""
        self.files = {p: value for p, value in self.files.items() if not p.is_relative_to(directory)}


def header_digest(directory, digest_file):
    """Preserve the original path/content digest algorithm."""
    digest = hashlib.sha256()
    for header in sorted(directory.rglob("*")):
        if header.is_file():
            digest.update(header.relative_to(directory).as_posix().encode())
            digest.update(bytes.fromhex(digest_file(header)))
    return digest.hexdigest()


def verify_package_versions(expected: dict, timings: dict) -> None:
    """Check recorded matrix packages without imposing local release requirements."""
    with identity_phase(timings, "package_versions"):
        actual = {name: metadata.version(name) for name in expected}
        if actual != expected:
            raise RuntimeError(f"Framework environment changed: {actual!r} != {expected!r}")


def build_identity(root: Path, soc_list: str, *, source=None, timings=None, expected_versions=None) -> dict:
    """Collect the original identity with per-call shared inputs and separate diagnostics."""
    timings = {} if timings is None else timings
    if sys.version_info[:2] < (3, 10):
        raise RuntimeError("Native builds require Python >=3.10")
    verify_package_versions(expected_versions or {}, timings)
    pta = _MANIFEST.framework_package("torch_npu")
    pta_version = pta["version"]
    with identity_phase(timings, "framework_import"):
        import torch
    pta_root = Path(pta["root"])
    torch_root = Path(torch.__file__).resolve().parent
    torch_version = str(getattr(torch, "__version__", "unknown"))
    headers_root = pta_root / "include" / "third_party" / "op-plugin"
    if not (headers_root / "op_plugin/include/npu_cpp_extension.h").is_file():
        raise RuntimeError(f"Missing installed op-plugin header: {headers_root}")
    digest_file = ProbeHashes()
    with identity_phase(timings, "headers_hash"):
        headers = {
            p.relative_to(headers_root).as_posix(): digest_file(p)
            for p in sorted(headers_root.rglob("*"))
            if p.is_file()
        }
        header_hashes = {
            name: header_digest(directory, digest_file)
            for name, directory in (
                ("torch_headers_sha256", torch_root / "include"),
                ("pta_headers_sha256", pta_root / "include"),
            )
        }
    with identity_phase(timings, "source_identity"):
        source = _source_identity(root) if source is None else source
    info = identity_metadata(root, soc_list, torch, torch_version, pta_version, headers, source)
    info.update(header_hashes)
    with identity_phase(timings, "libraries_hash"):
        info["libtorch_npu_sha256"] = digest_file(pta_root / "lib/libtorch_npu.so")
        info["torch_libraries_sha256"] = {
            name: digest_file(torch_root / "lib" / name) for name in ("libc10.so", "libtorch.so", "libtorch_cpu.so")
        }
    digest_file.validate()
    info["build_fingerprint"] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return info


def cann_identity(cann_root: Path) -> dict:
    """Record local toolkit facts without requiring a particular version or version file."""
    cann_version = cann_root / "opp/version.info"
    contents = cann_version.read_text() if cann_version.is_file() else ""
    versions = [line.partition("=")[2].strip() for line in contents.splitlines()
                if line.partition("=")[0].strip() == "Version"]
    return {
        "cann_root": str(cann_root.resolve()),
        "cann_version": ", ".join(versions) or "unknown",
        "cann_version_info": contents,
        "cann_version_sha256": _sha256(cann_version) if cann_version.is_file() else None,
    }


def identity_metadata(root, soc_list, torch, torch_version, pta_version, headers, source):
    return {
        "schema_version": 1,
        "component_version": COMPONENT_VERSION,
        "artifact_version": artifact_version(pta_version),
        "pta_version": pta_version,
        "torch_version": torch_version,
        "op_plugin_version": torch_op_plugin_version(Path(torch.__file__).resolve().parent),
        "torch_commit": getattr(torch.version, "git_version", None),
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "host_arch": platform.machine(),
        "soc_list": soc_list.split(","),
        **cann_identity(Path(os.environ["ASCEND_HOME_PATH"])),
        "op_plugin_headers_sha256": hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest(),
        "dependency_lock_sha256": _sha256(root / "hyper_multicore/_build/dependencies.lock.json"),
        **source,
    }


def probe_details():
    """Return build-only development paths without changing the framework identity schema."""
    import torch

    return {
        "python_include": sysconfig.get_paths()["include"],
        "torch_cmake": str(Path(torch.__file__).resolve().parent / "share/cmake"),
    }


def _payload_hashes(root: Path) -> dict[str, str]:
    """Hash every packaged resource except the identity manifest itself."""
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "build_info.json"
    }


def _kernel_references(value):
    """Read CANN binary and per-operator index references, including nested binInfo."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"binPath", "jsonPath", "jsonFilePath"}:
                if not isinstance(item, str) or not item:
                    raise RuntimeError(f"Invalid kernel index reference: {key}={item!r}")
                yield item
            else:
                yield from _kernel_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _kernel_references(item)


def _kernel_index_resources(kernel_root: Path, index: Path) -> list[Path]:
    """Require every indexed kernel variant and its metadata/object pair."""
    if not index.is_file():
        raise RuntimeError(f"Missing native payload resource: {index}")
    try:
        contents = json.loads(index.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise RuntimeError(f"Invalid kernel index: {index}") from error
    references = list(_kernel_references(contents))
    if not references:
        raise RuntimeError(f"Missing kernel index references: {index}")
    required = []
    for relative in references:
        path = kernel_root / relative
        if not path.resolve().is_relative_to(kernel_root.resolve()):
            raise RuntimeError(f"Kernel index reference outside payload: {relative}")
        required.append(path)
        if path.suffix == ".json":
            required.append(path.with_suffix(".o"))
    return required


def _validate_vendor_resources(root: Path, soc_list: list[str], host_arch: str) -> None:
    """Apply the vendor build's required runtime-resource checks at packaging time."""
    if (
        not isinstance(soc_list, list)
        or not soc_list
        or any(soc not in {"ascend910b", "ascend910_93"} for soc in soc_list)
    ):
        raise RuntimeError(f"Missing or unsupported payload SoC list: {soc_list}")
    if host_arch not in {"aarch64", "x86_64"}:
        raise RuntimeError(f"Missing or unsupported payload Host architecture: {host_arch}")
    vendor = root / "lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe"
    host_libraries = [
        vendor / "op_tiling/liboptiling.so",
        vendor / f"op_tiling/lib/linux/{host_arch}/libcust_opmaster_rt2.0.so",
        root / f"lib/vendors/hyper_multicore_nn/op_proto/lib/linux/{host_arch}/libcust_opsproto_rt2.0.so",
    ]
    for soc in soc_list:
        indexes = [vendor / f"kernel/config/{soc}/binary_info_config.json"]
        required = [*host_libraries, vendor / f"config/{soc}/aic-{soc}-ops-info.json", *indexes]
        for operator in _VENDOR_OPERATORS:
            index = vendor / f"kernel/config/{soc}/{operator}.json"
            required.append(index)
            indexes.append(index)
            for suffix in (".o", ".json"):
                matches = sorted((vendor / f"kernel/{soc}/{operator}").glob(f"*{suffix}"))
                if not matches:
                    raise RuntimeError(f"Missing native payload resource: {soc}/{operator}/*{suffix}")
                required.extend(matches)
        for index in indexes:
            required.extend(_kernel_index_resources(vendor / "kernel", index))
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty native payload resource: {path.relative_to(root)}")


def validate_payload(root: Path, info: dict) -> None:
    """Require the complete native payload and its recorded file hashes.

    Args:
        root: Prepared hyper_multicore payload directory.
        info: Its generated build_info object.

    Raises:
        RuntimeError: If identity, required files or native checksums are invalid.
    """
    for relative in (*REQUIRED_LIBRARIES, "lib/set_env.bash"):
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing native payload file: {relative}")
    if (root / "lib/framework").exists() or (root / "shmem/lib/framework").exists():
        raise RuntimeError("Expected only the shared Torch adapter in each variant; stale framework layout")
    _validate_vendor_resources(root, info.get("soc_list"), info.get("host_arch"))
    try:
        manifest = _MANIFEST.read_manifest(root)
        _MANIFEST.check_integrity(root, info, manifest)
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(str(error)) from error
    expected_version = (
        COMPONENT_VERSION if info.get("build_mode") == "all" else artifact_version(info["variants"][0]["pta_version"])
    )
    if info["artifact_version"] != expected_version:
        raise RuntimeError("Payload version does not match its framework matrix")
    if info.get("build_mode") not in ("current", "all"):
        raise RuntimeError("Unknown payload build mode")
    if info["build_mode"] == "current" and len(manifest["variants"]) != 1:
        raise RuntimeError("Current-environment payload must contain one adapter pair")
    _validate_payload_hashes(root, info)


def _validate_payload_hashes(root: Path, info: dict):
    """Verify the complete file set and contents independently of layout validation."""
    recorded = info.get("payload_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise RuntimeError("Missing complete payload checksum manifest: payload_sha256")
    actual = _payload_hashes(root)
    missing = sorted(recorded.keys() - actual.keys())
    unrecorded = sorted(actual.keys() - recorded.keys())
    if missing or unrecorded:
        raise RuntimeError(f"Payload file set mismatch: missing={missing}, unrecorded={unrecorded}")
    for relative, digest in actual.items():
        if recorded[relative] != digest:
            raise RuntimeError(f"Changed payload resource: {relative}")


def main() -> None:
    """Prepare identity before compilation, finalize its resources, or validate a payload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--soc-list", default="ascend910b,ascend910_93")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--finalize", type=Path, metavar="IDENTITY_JSON")
    action.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        info = build_identity(Path(__file__).resolve().parents[2], args.soc_list)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(info["build_fingerprint"])
    elif args.finalize:
        info = json.loads(args.finalize.read_text(encoding="utf-8"))
        info["native_sha256"] = {p.relative_to(args.output).as_posix(): _sha256(p) for p in args.output.rglob("*.so")}
        info["payload_sha256"] = _payload_hashes(args.output)
        validate_payload(args.output, info)
        (args.output / "build_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    else:
        validate_payload(args.output, json.loads((args.output / "build_info.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
