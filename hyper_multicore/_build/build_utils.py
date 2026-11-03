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
"""Small standard-library filesystem and timing primitives shared by build modules."""

from contextlib import contextmanager
import json
from pathlib import Path
import time
from collections.abc import Iterator


def atomic_json(path: Path, value: object) -> None:
    """Replace complete JSON and remove private partial writes on failure or cancellation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def timed(metrics: dict, name: str) -> Iterator[None]:
    """Accumulate counters without adding diagnostic data to authentication inputs."""
    started = time.monotonic()
    try:
        yield
    finally:
        record = metrics.setdefault(name, {"stage": name, "calls": 0, "elapsed_seconds": 0.0})
        record["calls"] += 1
        record["elapsed_seconds"] += time.monotonic() - started


@contextmanager
def measured(events: list, stage: str) -> Iterator[None]:
    """Record one controller phase, including a failed or interrupted phase."""
    started = time.monotonic()
    try:
        yield
    finally:
        events.append({"stage": stage, "elapsed_seconds": time.monotonic() - started})
