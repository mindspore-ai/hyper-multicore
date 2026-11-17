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
"""Dump-specific safe stage directories, independent of profiler names."""

from collections.abc import Sequence
import re

from hyper_multicore.diagnostics.common.graph import operator_stage_name
from hyper_multicore.scheduler.graph import OperatorNode


_SAFE_FILE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _stage_names(operators: Sequence[OperatorNode]) -> dict[str, str]:
    """Reject duplicate, colliding or reserved dump stage directories."""
    names = {}
    used_directories = set()
    for operator in operators:
        name = operator_stage_name(operator)
        directory = _SAFE_FILE_NAME.sub("_", name)
        if directory in ("", ".", "..", "__kernel_boundary__") or directory in used_directories:
            raise ValueError(f"duplicate, colliding or reserved MegaKernel dump stage directory: {name!r}")
        used_directories.add(directory)
        names[operator.name] = name
    return names
