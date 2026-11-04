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
"""Validate staged artifacts and atomically expose complete filesystem outputs."""

from __future__ import annotations
import hashlib
import re
import base64
import csv
import ctypes
from email.parser import BytesParser
import io
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from . import artifacts
from .command_runner import Commands
from ._bootstrap import load_module

MANIFEST = load_module("native_manifest")


def publish_wheel(source: Path, destination: Path) -> None:
    """Expose a complete wheel atomically, preserving any prior successful artifact."""
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def exchange_directories(first: Path, second: Path) -> None:
    """Exchange Linux directory entries without a missing-payload interval."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic native payload publication requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(first), -100, os.fsencode(second), 2) != 0:  # AT_FDCWD, RENAME_EXCHANGE
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(second))


def publish_payload(candidate: Path, selected: Path) -> None:
    """Keep the visible payload complete even if interrupted immediately after publication."""
    try:
        if selected.is_symlink():
            raise ValueError(f"Refusing to replace an external payload symlink: {selected}")
        if selected.exists() and not selected.is_dir():
            raise ValueError(f"Native payload destination is not a directory: {selected}")
        if selected.exists():
            exchange_directories(candidate, selected)
        else:
            os.replace(candidate, selected)
    finally:
        if candidate.exists():
            shutil.rmtree(candidate)


def validate_elf(path: Path, arch: str, run: Commands, env: dict) -> None:
    """Keep target architecture, RPATH, private dependencies and ELF hardening explicit."""
    data = run(["readelf", "-hW", "-lW", "-dW", str(path)], label=f"elf-{path.name}", env=env, capture=True)
    expected = "AArch64" if arch == "aarch64" else "Advanced Micro Devices X86-64"
    if expected not in data or "GNU_RELRO" not in data or "BIND_NOW" not in data:
        raise ValueError(f"Missing architecture or ELF hardening: {path}")
    if re.search(r"GNU_STACK[^\n]*RWE", data):
        raise ValueError(f"Executable stack is forbidden: {path}")
    if "(RPATH)" in data or "(RUNPATH)" in data or "Shared library: [libcust_opapi.so]" in data:
        raise ValueError(f"Forbidden adapter runtime dependency/path: {path}")
    if "Shared library: [libshmem" in data:
        raise ValueError(f"Generic SHMEM dependency: {path}")
    linked = run(["ldd", str(path)], label=f"dependencies-{path.name}", env=env, capture=True)
    missing = [line.strip().split()[0] for line in linked.splitlines() if "not found" in line]
    if any(not name.startswith(("libtorch", "libc10")) for name in missing):
        raise ValueError(f"Missing adapter dependencies: {missing}")


def validate_wheel(wheel: Path, payload: Path, info: dict) -> None:
    """Verify archive membership, RECORD, identities and native payload before publication."""
    with zipfile.ZipFile(wheel) as archive:
        names, record_path, metadata_path = wheel_identity_members(archive)
        validate_wheel_record(archive, record_path, names)
        validate_wheel_metadata(archive, metadata_path, info)
        validate_wheel_payload(archive, names, payload, info)


def wheel_identity_members(archive):
    """Require unique safe entries and exactly one distribution identity."""
    names = archive.namelist()
    if len(names) != len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
        raise ValueError("Wheel contains duplicate or unsafe entries")
    records = [n for n in names if len(Path(n).parts) == 2 and n.endswith(".dist-info/RECORD")]
    metas = [n for n in names if len(Path(n).parts) == 2 and n.endswith(".dist-info/METADATA")]
    if len(records) != 1 or len(metas) != 1:
        raise ValueError("Wheel requires unique RECORD and METADATA")
    return names, records[0], metas[0]


def validate_wheel_record(archive, record_path, names):
    """Require RECORD to cover every archive member with its correct size and digest."""
    entries = list(csv.reader(io.StringIO(archive.read(record_path).decode())))
    if len(entries) != len({row[0] for row in entries}) or {row[0] for row in entries} != set(names):
        raise ValueError("Wheel RECORD membership mismatch")
    for name, checksum, size in entries:
        if name == record_path:
            continue
        contents = archive.read(name)
        expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode()
        if checksum != expected or size != str(len(contents)):
            raise ValueError(f"Wheel RECORD mismatch: {name}")


def validate_wheel_metadata(archive, metadata_path, info):
    """Verify the component identity and unconstrained framework dependency."""
    metadata = BytesParser().parsebytes(archive.read(metadata_path))
    if metadata["Name"].replace("_", "-") != "hyper-multicore" or metadata["Version"] != info["artifact_version"]:
        raise ValueError("Wheel distribution identity mismatch")
    from packaging.requirements import Requirement

    torch = [Requirement(v) for v in metadata.get_all("Requires-Dist", []) if Requirement(v).name.lower() == "torch"]
    if len(torch) != 1 or str(torch[0].specifier) or torch[0].url:
        raise ValueError("Wheel must declare Torch without a version constraint or direct URL")


def validate_wheel_payload(archive, names, payload, info):
    """Check complete native membership, resource hashes and assembled identity bytes."""
    expected = {"hyper_multicore/" + n for n in info["payload_sha256"]}
    actual = {
        n
        for n in names
        if any(n.startswith("hyper_multicore/" + prefix + "/") for prefix in ("lib", "shmem/lib", "adapters"))
    }
    if actual != expected:
        raise ValueError("Wheel native payload membership mismatch")
    for name, checksum in info["payload_sha256"].items():
        if hashlib.sha256(archive.read("hyper_multicore/" + name)).hexdigest() != checksum:
            raise ValueError(f"Wheel payload mismatch: {name}")
    for name in ("build_info.json", "adapters/manifest.json"):
        if archive.read("hyper_multicore/" + name) != (payload / name).read_bytes():
            raise ValueError(f"Wheel identity mismatch: {name}")


class SourceGuard:
    """Reject mixed repository sources before and after formal publication."""

    def __init__(self, root: Path):
        self.root = root
        self.frozen = self.observe()

    def observe(self):
        """Record content and file generations, including edit-and-restore changes."""
        paths = []
        for name in ("hyper_multicore", "scripts"):
            paths.extend(p for p in (self.root / name).rglob("*")
                         if p.is_file() and not {"lib", "__pycache__"}.intersection(p.relative_to(self.root).parts))
        paths.extend(self.root / name for name in
                     ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt", "tests/ut/shmem/native_log_test.cpp"))
        generations = {}
        for path in paths:
            if path.is_file():
                stat = path.stat()
                generations[str(path)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return artifacts._source_identity(self.root), generations

    def check_source(self):
        if self.observe() != self.frozen:
            raise RuntimeError("Source inputs changed during build; refusing mixed-source payload")
