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
"""Standard-library contracts for native adapter selection and payload integrity."""

from __future__ import annotations

import ast
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import re

SHMEM_DEPENDENCIES = (
    "shmem/libhyper_multicore_shmem_utils.so",
    "shmem/libhyper_multicore_shmem.so",
    "libhyper_multicore_shmem_ops.so",
)
SHMEM_RESOURCES = (*SHMEM_DEPENDENCIES,
                   "shmem/aclshmem_bootstrap_uid.so", "shmem/aclshmem_bootstrap_config_store.so")


def framework_package(package: str) -> dict:
    """Describe the package selected by Python, including source trees without dist-info."""
    spec = util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"Missing {package} package")
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    version = source_package_version(root)
    if version == "unknown":
        try:
            distribution = metadata.distribution(package.replace("_", "-"))
            if Path(distribution.locate_file(package)).resolve() == root:
                version = distribution.version
        except metadata.PackageNotFoundError:
            pass
    return {"version": version, "root": str(root)}


def source_package_version(root: Path) -> str:
    """Read literal version declarations without importing a backend or executing source."""
    version = "unknown"
    for filename in ("version.py", "__init__.py"):
        path = root / filename
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeError):
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    version = node.value.value
    return version


def release(version: str) -> str:
    """Extract a numeric branch, or retain an opaque label without version admission rules."""
    match = re.match(r"[vV]?(\d+\.\d+\.\d+)", version)
    return match[1] if match else version


def file_sha256(path: Path) -> str:
    """Hash a file without reading its entire contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resource_path(root: Path, relative: str) -> Path:
    """Resolve an owned relative path, rejecting traversal and escaped symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Invalid payload path: {relative!r}")
    path = Path(relative)
    if path.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise ValueError(f"Unsafe payload path: {relative!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Escaped payload path: {relative!r}")
    return resolved


def variant_selection_key(variant: dict):
    """Validate the unique adapter ID and stable branch/ABI selector."""
    identifier = variant.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-zA-Z0-9._-]+", identifier):
        raise ValueError(f"Invalid adapter ID: {identifier!r}")
    if not isinstance(variant.get("torch_cxx11_abi"), bool):
        raise ValueError("Adapter C++ ABI must be boolean")
    key = (variant.get("torch_release"), variant.get("pta_release"), variant["torch_cxx11_abi"])
    if not all(isinstance(v, str) and release(v) == v for v in key[:2]):
        raise ValueError("Adapter selection requires branch or opaque framework labels")
    return identifier, key


def validate_adapter_paths(root: Path, variant: dict, backends: dict):
    """Require the shared Torch adapter and one binding for each packaged backend."""
    expected_torch = f"adapters/{variant['id']}/libhyper_multicore_torch.so"
    expected_shmem = {name: f"adapters/{variant['id']}/shmem/{name}/hyper_multicore_shmem_torch.so"
                      for name in backends}
    if variant.get("torch_so") != expected_torch or variant.get("shmem_so") != expected_shmem:
        raise ValueError(f"Invalid adapter paths for {variant['id']}")
    for expected in (expected_torch, *expected_shmem.values()):
        path = resource_path(root, expected)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing adapter resource: {expected}")


def validate_architecture(root: Path, name: str, backend: dict) -> None:
    """Validate one backend's canonical paths, architecture and SoC membership."""
    if not re.fullmatch(r"arch[0-9]+", name) or backend.get("shmem_root") != f"shmem/lib/{name}":
        raise ValueError(f"Invalid SHMEM backend root: {name}")
    kernel_root = f"lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe/kernel/{name}"
    if backend.get("kernel_root") != kernel_root or not isinstance(backend.get("sdk_soc_type"), str):
        raise ValueError(f"Invalid architecture resources: {name}")
    arch = backend.get("npu_arch")
    selected = backend.get("soc_list")
    if not isinstance(arch, int) or isinstance(arch, bool) or arch <= 0:
        raise ValueError(f"Invalid SHMEM architecture: {name}")
    if not isinstance(selected, list) or not selected or any(not isinstance(s, str) for s in selected):
        raise ValueError(f"Missing SHMEM backend SoCs: {name}")
    for relative in SHMEM_RESOURCES:
        path = resource_path(root, f"{backend['shmem_root']}/{relative}")
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing SHMEM resource: {path}")


def validate_architectures(root: Path, backends: dict) -> None:
    """Validate SDK resources and reject overlapping backend identities."""
    if not isinstance(backends, dict) or not backends:
        raise ValueError("Missing SHMEM backends")
    for name, backend in backends.items():
        validate_architecture(root, name, backend)
    arches = [backend["npu_arch"] for backend in backends.values()]
    socs = [soc for backend in backends.values() for soc in backend["soc_list"]]
    if len(set(arches)) != len(arches) or len(set(socs)) != len(socs):
        raise ValueError("Duplicate SHMEM backend architecture or SoC")


def read_manifest(root: Path) -> dict:
    """Read and validate framework and backend resources before loading any SO."""
    manifest = json.loads((root / "adapters/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported native adapter manifest schema")
    if not manifest.get("python_abi") or not manifest.get("host_arch"):
        raise ValueError("Missing manifest Python ABI or host architecture")
    backends = manifest.get("architectures")
    validate_architectures(root, backends)
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Adapter manifest must contain at least one variant")
    ids, keys = set(), set()
    for variant in variants:
        identifier, key = variant_selection_key(variant)
        if identifier in ids or key in keys:
            raise ValueError("Duplicate adapter ID or ambiguous selection key")
        ids.add(identifier)
        keys.add(key)
        validate_adapter_paths(root, variant, backends)
    return manifest


def select_variant(manifest: dict, actual: dict) -> dict:
    """Select by branch and ABI; external CANN and post versions are diagnostic only."""
    for key in ("python_abi", "host_arch"):
        if actual[key] != manifest[key]:
            raise ValueError(f"Native {key} mismatch: built={manifest[key]}, actual={actual[key]}")
    key = (release(actual["torch_version"]), release(actual["pta_version"]), actual["torch_cxx11_abi"])
    matches = [v for v in manifest["variants"]
               if (v["torch_release"], v["pta_release"], v["torch_cxx11_abi"]) == key]
    if len(matches) != 1:
        available = [(v["torch_release"], v["pta_release"], v["torch_cxx11_abi"]) for v in manifest["variants"]]
        raise ValueError(f"Expected one adapter for {key}; found {len(matches)}; available={available}")
    return matches[0]


def select_shmem_backend(manifest: dict, npu_arch: int) -> tuple[str, dict]:
    """Match a device architecture to exactly one packaged SHMEM backend."""
    matches = [(name, entry) for name, entry in manifest["architectures"].items()
               if entry["npu_arch"] == npu_arch]
    if len(matches) != 1:
        raise ValueError(f"No packaged SHMEM backend for npu_arch={npu_arch}; "
                         f"available={list(manifest['architectures'])}")
    return matches[0]


def validate_build_records(info: dict, manifest: dict):
    """Match platform and requested adapter IDs to actual build provenance."""
    if info.get("schema_version") != 3:
        raise ValueError("Native payload requires schema 3; rebuild with bash build.sh")
    if any(info.get(k) != manifest[k] for k in ("python_abi", "host_arch")):
        raise ValueError("Build identity and manifest platform disagree")
    validate_backend_provenance(info, manifest)
    records = info.get("variants")
    if not isinstance(records, list):
        raise ValueError("Missing adapter build identities")
    if sorted(v["id"] for v in records) != sorted(v["id"] for v in manifest["variants"]):
        raise ValueError("Build identity and adapter manifest disagree")
    if sorted(info.get("requested_target_ids", [])) != sorted(v["id"] for v in records):
        raise ValueError("Missing requested adapter targets")
    return records


def validate_backend_provenance(info: dict, manifest: dict) -> None:
    """Match the selected SoCs and SDK identities to the runtime index."""
    backends = manifest["architectures"]
    built = info.get("architectures", {})
    if set(built) != set(backends) or any(
            any(built[name].get(key) != value for key, value in entry.items()) for name, entry in backends.items()):
        raise ValueError("Build identity and SHMEM backends disagree")
    socs = [soc for entry in backends.values() for soc in entry["soc_list"]]
    if sorted(socs) != sorted(info.get("soc_list", [])):
        raise ValueError("Build identity and backend SoCs disagree")


def validate_variant_provenance(record: dict, variant: dict):
    """Match each adapter selector and library path to its actual framework build."""
    if (release(record["torch_version"]) != variant["torch_release"]
            or release(record["pta_version"]) != variant["pta_release"]):
        raise ValueError(f"Adapter branch disagrees with actual build: {variant['id']}")
    if any(record.get(k) != variant.get(k)
           for k in ("torch_release", "pta_release", "torch_cxx11_abi", "torch_so", "shmem_so")):
        raise ValueError(f"Adapter provenance mismatch: {variant['id']}")


def payload_members(root: Path, info: dict):
    """Require an exact native-resource set, including files not selected for loading."""
    recorded = info.get("payload_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Missing complete payload checksum manifest: payload_sha256")
    actual = {p.relative_to(root).as_posix() for prefix in ("lib", "shmem/lib", "adapters")
              for p in (root / prefix).rglob("*") if p.is_file()}
    if actual != set(recorded):
        raise ValueError(f"Payload file set mismatch: missing={sorted(set(recorded)-actual)}, "
                         f"unrecorded={sorted(actual-set(recorded))}")
    return recorded, actual


def validate_resource_hashes(root: Path, info: dict, recorded: dict, actual: set):
    """Check every recorded resource and require separate native-library digests."""
    for relative, digest in recorded.items():
        if file_sha256(resource_path(root, relative)) != digest:
            raise ValueError(f"Changed payload resource: {relative}")
    for relative in actual:
        if relative.endswith(".so") and info.get("native_sha256", {}).get(relative) != recorded[relative]:
            raise ValueError(f"Missing native checksum: {relative}")


def check_integrity(root: Path, info: dict, manifest: dict) -> None:
    """Validate native membership, framework provenance and all recorded resource hashes."""
    records = validate_build_records(info, manifest)
    for variant in manifest["variants"]:
        record = next(v for v in records if v["id"] == variant["id"])
        validate_variant_provenance(record, variant)
    recorded, actual = payload_members(root, info)
    validate_resource_hashes(root, info, recorded, actual)
