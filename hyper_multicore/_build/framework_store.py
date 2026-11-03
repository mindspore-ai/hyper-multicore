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
"""Store immutable framework wheels and publish verified dependency snapshots."""

import errno
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path

from .build_utils import atomic_json
from .build_workspace import framework_directory


STORAGE_VERSION = 2


def wheel_path(native, package):
    filename, digest = package["filename"], package["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or Path(filename).name != filename or not filename.endswith(".whl"):
        raise ValueError("Invalid shared wheel identity")
    root = framework_directory(native, "wheels")
    result = root / filename
    if result.is_symlink():
        raise ValueError(f"Refusing symlinked shared wheel storage: {result}")
    return result


def install_text(native, record):
    return "".join(f"{wheel_path(native, p)} --hash=sha256:{p['sha256']}\n" for p in record["packages"])


def install_manifest(native, record, destination):
    """Preserve operational local-only installation commands when a snapshot moves its bytes."""
    text = install_text(native, record)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(destination)


def clear_incoming(directory):
    """Only called under the project lock; interrupted private wheel imports are disposable."""
    for path in directory.glob(".incoming-*"):
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def import_wheel(source, native, package, verify):
    """Publish immutable verified bytes without replacing a conflicting filename."""
    if source.is_symlink():
        raise ValueError(f"Refusing symlinked wheel source: {source}")
    destination = wheel_path(native, package)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clear_incoming(destination.parent)
    verify(source, package)
    if destination.exists():
        try:
            verify(destination, package)
        except ValueError as error:
            raise ValueError(f"Conflicting shared wheel filename: {destination}") from error
        return destination, False
    temporary = Path(tempfile.mkdtemp(prefix=".incoming-", dir=destination.parent))
    candidate = temporary / package["filename"]
    try:
        try:
            os.link(source, candidate)
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
            shutil.copyfile(source, candidate)
        verify(candidate, package)
        candidate.chmod(0o444)
        os.replace(candidate, destination)
    finally:
        shutil.rmtree(temporary)
    return destination, True


def snapshot_path(native, target, key):
    """Short names are labels; the full key remains authoritative in the record."""
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Invalid framework environment key")
    label = f"cp{sys.version_info.major}{sys.version_info.minor}-{platform.machine()}-pt{target['pta_build_version']}-{key[:12]}"
    return framework_directory(native, "resolutions") / (label + ".json")


def load_snapshot(path, target, key):
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked resolution: {path}")
    record = json.loads(path.read_text())
    if (
        record.get("environment_key") != key
        or record.get("request") != target
        or record.get("wheel_storage") != STORAGE_VERSION
    ):
        raise ValueError(f"Framework snapshot key/request conflict: {path}")
    report = path.with_suffix(".pip-report.json")
    if report.is_symlink() or hashlib.sha256(report.read_bytes()).hexdigest() != record["report_sha256"]:
        raise ValueError(f"Framework pip report integrity mismatch: {report}")
    return record


def publish_snapshot(snapshot, native, record, target, key, verify):
    """Publish current flat wheel bytes and report before the authoritative snapshot."""
    destination = snapshot_path(native, target, key)
    if destination.exists():
        return destination, load_snapshot(destination, target, key)
    for package in record["packages"]:
        import_wheel(snapshot / "wheels" / package["filename"], native, package, verify)
    report = snapshot / "pip-report.json"
    if report.is_symlink():
        raise ValueError(f"Refusing symlinked pip report: {report}")
    report_value = json.loads(report.read_text())
    report_value["_hyper_multicore_environment_key"] = key
    contents = (json.dumps(report_value, indent=2) + "\n").encode()
    updated = {
        **record,
        "wheel_storage": STORAGE_VERSION,
        "environment_key": key,
        "report_sha256": hashlib.sha256(contents).hexdigest(),
    }
    report_path = destination.with_suffix(".pip-report.json")
    if report_path.exists():
        orphan = json.loads(report_path.read_text())
        if orphan.get("_hyper_multicore_environment_key") != key:
            raise ValueError(f"Framework snapshot short-ID conflict: {report_path}")
    atomic_json(report_path, report_value)
    atomic_json(destination, updated)
    return destination, updated


def update_reference(native, environment, destination, record):
    """The stable path checks without rewriting already correct operational metadata."""
    marker = environment / "ready.json"
    ready = json.loads(marker.read_text())
    if (
        ready.get("environment_key") != record["environment_key"]
        or ready.get("packages") != record["packages"]
        or ready.get("request") != record["request"]
    ):
        raise ValueError(f"Changed environment snapshot: {marker}")
    install = environment / "install-requirements.txt"
    expected = install_text(native, record)
    if not install.is_file() or install.read_text() != expected:
        install_manifest(native, record, install)
    if ready.get("resolution") != str(destination):
        atomic_json(marker, {**ready, "resolution": str(destination)})


def cached_root(native, target, name, version, verify):
    """Only a completed, source-matching snapshot can supply a framework root."""
    source = target["indexes"]["torch" if name == "torch" else "pta"]
    for path in sorted((framework_directory(native, "resolutions")).glob("*.json")):
        if path.name.endswith(".pip-report.json") or path.is_symlink():
            continue
        try:
            record = json.loads(path.read_text())
            key = record["environment_key"]
            if path != snapshot_path(native, record["request"], key):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        matches = [p for p in record["packages"] if (p["name"], p["version"], p["source"]) == (name, version, source)]
        if matches:
            load_snapshot(path, record["request"], key)
            result = wheel_path(native, matches[0])
            verify(result, matches[0])
            return result
    return None
