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
"""Authenticate native stages and explain input changes without weakening checks."""

import hashlib
import json
import shutil
from pathlib import Path

from . import artifacts as HELPERS, build_workspace as WORKSPACE, framework_environments as FRAMEWORK
from ._bootstrap import load_module
from .build_utils import atomic_json as write_json, timed
from .command_runner import message


ROOT = Path(__file__).resolve().parents[2]
CACHE_SCHEMA_VERSION = 3
MANIFEST = load_module("native_manifest")


def source_observation() -> dict:
    """Freeze source identity and file generations; restoring old bytes is still a change."""
    generations = {}
    for directory in ("hyper_multicore", "scripts"):
        for path in (ROOT / directory).rglob("*"):
            relative = path.relative_to(ROOT)
            if "lib" in relative.parts or "__pycache__" in relative.parts or not path.is_file():
                continue
            stat = path.stat()
            generations[relative.as_posix()] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    for name in ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt", "tests/ut/shmem/native_log_test.cpp"):
        path = ROOT / name
        if path.is_file():
            stat = path.stat()
            generations[name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return {"identity": HELPERS._source_identity(ROOT), "generations": generations}


class CacheTransaction:
    """Commit only unchanged stages and revoke this run's new markers on failure."""

    def __init__(self):
        self.frozen = source_observation()
        self.created = set()
        self.input_checks = []
        self.scopes = {}
        self.metrics = {}

    def __enter__(self):
        self.check_source()
        return self

    def check(self, directory=None):
        """Validate one stage at certification, or all registered inputs at final commit."""
        checks = [self.scopes[directory]] if directory in self.scopes else self.input_checks
        with timed(self.metrics, "stage-input-validation" if directory else "final-input-validation"):
            for check in checks:
                check()
        self.check_source()

    def check_source(self):
        """Observe repository changes without repeatedly traversing unrelated SDK inputs."""
        with timed(self.metrics, "source-validation"):
            if source_observation() != self.frozen:
                raise RuntimeError("Source inputs changed during build; refusing mixed-source cache or payload")

    def prepare(self, directory: Path):
        """Discard abandoned task outputs without touching any reusable stage."""
        self.check_source()
        if directory.is_symlink():
            raise ValueError(f"Refusing symlinked native work directory: {directory}")
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        WORKSPACE.claim_workspace(directory, ROOT)
        self.created.add(directory)

    def complete(self, directory: Path, record: dict):
        """Publish a stage marker only after output validation and a fresh input check."""
        self.check(directory)
        write_json(directory / "complete.json", {**record, "cache_schema": CACHE_SCHEMA_VERSION})
        self.check(directory)

    def __exit__(self, exception_type, _exception, _traceback):
        try:
            if exception_type is None:
                self.check()
        except BaseException:
            self.revoke()
            raise
        if exception_type is not None:
            self.revoke()

    def revoke(self):
        """Keep successful pre-existing caches; remove only this run's completion claims."""
        for directory in self.created:
            (directory / "complete.json").unlink(missing_ok=True)


def hashes(root: Path) -> dict:
    """Hash every resource in a completed stage, excluding its external completion record."""
    return {p.relative_to(root).as_posix(): MANIFEST.file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def cache_valid(directory: Path, key: str) -> bool:
    """Never reuse an incomplete or changed native stage."""
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record.get("cache_schema") != CACHE_SCHEMA_VERSION:
        return False
    if record.get("fingerprint") != key:
        return False
    if "inputs" in record and FRAMEWORK.digest_json(record["inputs"]) != key:
        raise ValueError(f"Native cache input record mismatch: {directory}")
    if record.get("fingerprint") != key or record.get("hashes") != hashes(directory / "payload"):
        raise ValueError(f"Native cache integrity mismatch: {directory}")
    if "sdk_root" in record:
        sdk = Path(record["sdk_root"])
        handoff = directory / "sdk.json"
        if not handoff.is_file() or MANIFEST.file_sha256(handoff) != record.get("sdk_manifest_sha256"):
            raise ValueError(f"Changed SDK handoff: {directory}")
        if not sdk.is_dir() or hashes(sdk) != record.get("sdk_hashes"):
            raise ValueError(f"Missing or changed SDK inputs: {sdk}")
    return True


LABELS = {
    "source": "source inputs changed",
    "compiler": "compiler/toolchain changed",
    "flags": "effective options/search paths changed",
    "framework": "framework headers/libraries changed",
    "common": "shared native inputs changed",
    "development": "development headers/libraries changed",
    "schema": "preflight cache format changed",
}


def compact_inputs(inputs):
    """Summarize duplicated development manifests, never the inputs used for certification."""
    result = dict(inputs)
    for name in ("development", "cann_development"):
        files = result.get(name)
        if isinstance(files, dict) and files.get("summary_schema") != 1:
            data = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            result[name] = {"summary_schema": 1, "sha256": hashlib.sha256(data).hexdigest(), "file_count": len(files)}
    return result


def normalize_records(records):
    """Diagnostics tolerate missing/corrupt entries and read the former expanded representation."""
    if not isinstance(records, dict):
        return {}
    return {
        name: {**record, "inputs": compact_inputs(record["inputs"])}
        for name, record in records.items()
        if isinstance(record, dict)
        and isinstance(record.get("inputs"), dict)
        and isinstance(record.get("fingerprint"), str)
    }


class CacheDiagnostics:
    """Compare against the most recent successful build; diagnostics are never a gate."""

    def __init__(self, native, events):
        self.path = native / "cache-inputs.json"
        self.events, self.pending = events, {}
        try:
            self.previous = normalize_records(json.loads(self.path.read_text()))
        except (OSError, ValueError):
            self.previous = {}

    def observe(self, scope, key, inputs, directory):
        inputs = compact_inputs(inputs)
        self.pending[scope] = {"fingerprint": key, "inputs": inputs}
        previous = self.previous.get(scope, {})
        changes = {
            name: {"previous": previous.get("inputs", {}).get(name), "current": value}
            for name, value in inputs.items()
            if previous.get("inputs", {}).get(name) != value
        }
        if previous and "schema" in changes:
            reason = "preflight cache format upgrade; revalidating once"
        elif previous.get("fingerprint") == key:
            reason = "inputs unchanged"
        elif previous:
            reason = "; ".join(LABELS.get(name, f"{name} changed") for name in changes) or "cache identity changed"
        else:
            reason = "no recorded successful inputs; historical differences unavailable"
        self.events.append(
            {
                "stage": "cache-inputs",
                "task": scope,
                "fingerprint": key,
                "reason": reason,
                "differences": changes if previous else {},
            }
        )
        if not (directory / "complete.json").exists():
            message(f"{scope}: {reason}")

    def publish(self):
        """Call only after complete publication and input validation succeed."""
        write_json(self.path, {**self.previous, **self.pending})
