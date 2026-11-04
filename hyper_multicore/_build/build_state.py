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
"""Explicit per-build state and results, separate from the tool process runner."""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from .command_runner import Commands


@dataclass
class PreparedFramework:
    """One selected interpreter and its collected build-only interface evidence."""

    python: Path
    identity: dict
    packages: list = field(default_factory=list)


@dataclass
class BuildResult:
    """Values needed after publication; no dynamic attributes on runner or transaction."""

    payload: Path
    pythons: list[Path]
    branches: set[str]
    pairs: int
    started: float


class BuildContext:
    """Own task state explicitly while sharing the runner's ordered stage records."""

    def __init__(self, root: Path, native: Path) -> None:
        """Collect task state without adding it to the command executor."""
        self.root, self.native = root, native
        self.directory = native / "logs"
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        self.events = []
        self.commands = Commands(self.directory, root=root, events=self.events,
                                 scratch=native / "staging" / self.run_id / "tool-output")
        self.frameworks = {}
        self.vendor_requests = []
        self.isolated = False
