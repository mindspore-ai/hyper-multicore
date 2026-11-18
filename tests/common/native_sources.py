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
"""Locate production sources in a checkout or a Jenkins test bundle."""

import tarfile
from pathlib import Path


def native_source(relative: Path, output: Path) -> Path:
    """Return one source file, extracting it into a test-owned directory when needed."""
    source_root = Path(__file__).resolve().parents[2]
    source = source_root / relative
    if source.is_file():
        return source

    with tarfile.open(source_root / "hyper-multicore.tar") as archive:
        members = [member for member in archive.getmembers()
                   if member.isfile() and Path(member.name).parts[-len(relative.parts):] == relative.parts]
        if len(members) != 1:
            raise RuntimeError(f"The test bundle must contain exactly one copy of {relative}")
        with archive.extractfile(members[0]) as content:
            source = output / relative.name
            source.write_bytes(content.read())
    return source
