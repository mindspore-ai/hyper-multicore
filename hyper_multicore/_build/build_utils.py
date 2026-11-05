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
import hashlib
import json
from pathlib import Path
import re
import time
from collections.abc import Iterator


def version_label(version: str) -> str:
    """Encode a framework label for filenames without imposing version syntax."""
    label = version.lower().replace("+", ".").replace("!", ".")
    if len(label) <= 80 and re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", label):
        return re.sub(r"[._-]+", ".", label)
    readable = re.sub(r"[^a-z0-9]+", ".", label).strip(".")[:64].rstrip(".") or "unknown"
    digest = hashlib.sha256(version.encode()).hexdigest()[:12]
    return f"{readable}.{digest}"


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
def measured(events: list, stage: str) -> Iterator[None]:
    """Record one controller phase, including a failed or interrupted phase."""
    started = time.monotonic()
    try:
        yield
    finally:
        events.append({"stage": stage, "elapsed_seconds": time.monotonic() - started})
