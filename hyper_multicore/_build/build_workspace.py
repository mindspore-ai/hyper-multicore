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
"""Locate, protect and manage build workspaces, locks and formal outputs."""

import fcntl
import json
import os
import platform
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from .build_utils import atomic_json
from .command_runner import message


FRAMEWORK_SLOT = re.compile(r"cp[0-9]+-[A-Za-z0-9_]+-torch[A-Za-z0-9_.-]+-pt[A-Za-z0-9_.-]+-abi[01]\Z")


def safe_component(value):
    value = str(value).replace("+", ".")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Invalid build workspace component: {value}")
    return value


def framework_slot(identity):
    fields = [identity[name] for name in ("python_abi", "host_arch", "torch_version", "pta_version")]
    fields.append(int(identity["torch_cxx11_abi"]))
    python, arch, torch, pta, abi = map(safe_component, fields)
    # CPU is the configured Torch distribution; retain local version labels to avoid collisions.
    return f"{python}-{arch}-torch{torch.removesuffix('.cpu')}-pt{pta}-abi{abi}"


def common_slot(socs):
    return safe_component(platform.machine() + "-" + socs.replace(",", "-"))


def claim_workspace(directory, root):
    directory.mkdir(parents=True, exist_ok=True)
    marker = {"schema": 1, "root": str(root.resolve()), "kind": directory.parent.name, "slot": directory.name}
    try:
        atomic_json(directory / ".workspace.json", marker)
    except BaseException:
        if not any(directory.iterdir()):
            directory.rmdir()
        raise


def check_workspace(directory, root):
    """A human-created directory must never be erased just because its name matches a slot."""
    if directory.parent.is_symlink() or directory.is_symlink():
        raise ValueError(f"Refusing symlinked native workspace: {directory}")
    if not directory.exists() or not any(directory.iterdir()):
        return
    try:
        marker = json.loads((directory / ".workspace.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Unattributed native workspace; refusing replacement: {directory}") from error
    expected = {"schema": 1, "root": str(root.resolve()), "kind": directory.parent.name, "slot": directory.name}
    if marker != expected:
        raise ValueError(f"Foreign native workspace; refusing replacement: {directory}")


def framework_directory(native: Path, kind: str) -> Path:
    """Keep framework storage together and reject redirected managed roots."""
    if kind not in ("envs", "resolutions", "wheels"):
        raise ValueError(f"Unknown framework directory: {kind}")
    parent = native / "framework"
    directory = parent / kind
    if parent.is_symlink() or directory.is_symlink():
        raise ValueError(f"Refusing symlinked framework storage: {directory}")
    return directory


RUN_DIRECTORY = re.compile(r"\d{8}-\d{6}-\d+\Z")


def lock_owner(lock) -> str:
    """Metadata explains the wait; stale or incomplete metadata never decides ownership."""
    lock.seek(0)
    try:
        owner = json.loads(lock.read(4096))
        return f"owner PID {owner['pid']}, mode {owner['mode']}, started {owner['started_at']}"
    except (ValueError, KeyError, TypeError):
        return "owner metadata unavailable"


def wait_for_lock(lock, interval=10.0):
    """Cancellation while waiting never truncates owner metadata or touches its outputs."""
    started = time.monotonic()
    next_notice = started
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            now = time.monotonic()
            if now >= next_notice:
                message(f"Waiting for build lock ({now - started:.0f}s elapsed); {lock_owner(lock)}")
                next_notice = now + interval
            time.sleep(0.1)


@contextmanager
def build_lock(native: Path, mode: str):
    """Keep one persistent inode; flock, not metadata or PID existence, decides ownership."""
    with (native / ".build.lock").open("a+") as lock:
        wait_for_lock(lock)
        try:
            lock.seek(0)
            lock.truncate()
            json.dump({"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": mode}, lock)
            lock.flush()
            yield
        finally:
            lock.seek(0)
            lock.truncate()
            lock.flush()
            fcntl.flock(lock, fcntl.LOCK_UN)


def remove_generated(parent: Path, matches):
    """Remove only known generated children, without traversing redirected directories."""
    if parent.is_symlink():
        raise ValueError(f"Refusing to clean a symlinked build directory: {parent}")
    if not parent.exists():
        return
    for child in parent.iterdir():
        if not matches(child.name):
            continue
        if child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def clear_temporary_outputs(native: Path):
    """Preserve native caches, environments, source repositories and published payloads."""
    remove_generated(native / "staging", RUN_DIRECTORY.fullmatch)
    remove_generated(
        native / "payload", lambda name: name.startswith(".payload-") and RUN_DIRECTORY.fullmatch(name[9:])
    )


def prepare_logs(native: Path) -> Path:
    """Run only while holding the build lock, after command-line validation."""
    clear_temporary_outputs(native)
    logs = native / "logs"
    if logs.is_symlink():
        raise ValueError(f"Refusing symlinked build logs: {logs}")
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("build.log", "tools.log", "stages.json"):
        path = logs / name
        if path.is_symlink():
            raise ValueError(f"Refusing symlinked build log: {path}")
        path.unlink(missing_ok=True)
    (logs / "stages.json").write_text("[]\n")
    (logs / "tools.log").touch()
    return logs


def clear_distribution(directory: Path):
    """dist represents only this build, not a historical artifact archive."""
    if directory.is_symlink():
        raise ValueError(f"Refusing symlinked distribution directory: {directory}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


class BuildOutputs:
    """One commit flag governs both visible outputs; uncommitted builds leave neither."""

    def __init__(self, distribution: Path, payload: Path) -> None:
        """Name the only two managed outputs and start before commit."""
        self.distribution, self.payload = distribution, payload
        self.committed = False

    def __enter__(self) -> "BuildOutputs":
        """Begin a real attempt after the caller acquires the build lock."""
        self.committed = False
        try:
            clear_distribution(self.distribution)
        except BaseException:
            self.clear_payload()
            raise
        return self

    def clear_payload(self) -> None:
        """Remove only the formal managed output, without following redirected paths."""
        if self.payload.is_symlink() or self.payload.parent.is_symlink():
            raise ValueError(f"Refusing symlinked formal payload: {self.payload}")
        if self.payload.exists():
            shutil.rmtree(self.payload)

    def __exit__(self, *_: object) -> None:
        """Clear uncommitted output on failure, cancellation or an early return."""
        if not self.committed:
            try:
                clear_distribution(self.distribution)
            finally:
                self.clear_payload()
