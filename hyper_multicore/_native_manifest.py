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

import hashlib
import json
from pathlib import Path
import re


def release(version: str) -> str:
    """Extract a stable framework branch without imposing a post-release floor."""
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:\.post\d+)?(?:\+[a-zA-Z0-9._-]+)?", version)
    if match is None:
        raise ValueError(f"Unrecognized framework release: {version!r}")
    return match[1]


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
    if type(variant.get("torch_cxx11_abi")) is not bool:
        raise ValueError("Adapter C++ ABI must be boolean")
    key = (variant.get("torch_release"), variant.get("pta_release"), variant["torch_cxx11_abi"])
    if not all(isinstance(v, str) and release(v) == v for v in key[:2]):
        raise ValueError("Adapter selection requires explicit stable branches")
    return identifier, key


def validate_adapter_paths(root: Path, variant: dict):
    """Require both nonempty adapter libraries at their owned canonical paths."""
    for field, filename in (("torch_so", "libhyper_multicore_torch.so"),
                            ("shmem_so", "hyper_multicore_shmem_torch.so")):
        expected = f"adapters/{variant['id']}/{filename}"
        if variant.get(field) != expected:
            raise ValueError(f"Invalid {field} for adapter {variant['id']}")
        path = resource_path(root, expected)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing adapter resource: {expected}")


def read_manifest(root: Path) -> dict:
    """Read and validate the unique adapter-pair index before loading any SO."""
    manifest = json.loads((root / "adapters/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported native adapter manifest schema")
    if not manifest.get("python_abi") or not manifest.get("host_arch"):
        raise ValueError("Missing manifest Python ABI or host architecture")
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
        validate_adapter_paths(root, variant)
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


def validate_build_records(info: dict, manifest: dict):
    """Match platform and requested adapter IDs to actual build provenance."""
    if info.get("schema_version") != 2:
        raise ValueError("Native payload requires schema 2; rebuild with bash build.sh")
    if any(info.get(k) != manifest[k] for k in ("python_abi", "host_arch")):
        raise ValueError("Build identity and manifest platform disagree")
    records = info.get("variants")
    if not isinstance(records, list):
        raise ValueError("Missing adapter build identities")
    if sorted(v["id"] for v in records) != sorted(v["id"] for v in manifest["variants"]):
        raise ValueError("Build identity and adapter manifest disagree")
    if sorted(info.get("requested_target_ids", [])) != sorted(v["id"] for v in records):
        raise ValueError("Missing requested adapter targets")
    return records


def validate_variant_provenance(record: dict, variant: dict):
    """Match each adapter selector and library path to its actual framework build."""
    if release(record["torch_version"]) != variant["torch_release"] or release(record["pta_version"]) != variant["pta_release"]:
        raise ValueError(f"Adapter branch disagrees with actual build: {variant['id']}")
    if any(record.get(k) != variant.get(k) for k in ("torch_release", "pta_release", "torch_cxx11_abi", "torch_so", "shmem_so")):
        raise ValueError(f"Adapter provenance mismatch: {variant['id']}")


def payload_members(root: Path, info: dict):
    """Require an exact native-resource set, including files not selected for loading."""
    recorded = info.get("payload_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Missing complete payload checksum manifest: payload_sha256")
    actual = {p.relative_to(root).as_posix() for prefix in ("lib", "shmem/lib", "adapters")
              for p in (root / prefix).rglob("*") if p.is_file()}
    if actual != set(recorded):
        raise ValueError(f"Payload file set mismatch: missing={sorted(set(recorded)-actual)}, unrecorded={sorted(actual-set(recorded))}")
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
    """Validate native membership, pair provenance and all recorded resource hashes."""
    records = validate_build_records(info, manifest)
    for variant in manifest["variants"]:
        record = next(v for v in records if v["id"] == variant["id"])
        validate_variant_provenance(record, variant)
    recorded, actual = payload_members(root, info)
    validate_resource_hashes(root, info, recorded, actual)
