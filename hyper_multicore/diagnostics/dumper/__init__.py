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
"""Public Tensor dump entry points; underscored modules are internal implementations."""

from hyper_multicore.diagnostics.dumper.api import (
    DumpConfig, DumpLevel, DumperSession, dump,
)
from hyper_multicore.diagnostics.dumper._inspect import compare_tensors, load_tensor
from hyper_multicore.diagnostics.dumper._task import (
    TaskDumpTensorView, TaskDumpTensorViewContext, UnsupportedTaskDumpPort,
)

__all__ = [
    "DumpConfig", "DumpLevel", "DumperSession", "dump",
    "compare_tensors", "load_tensor",
    "TaskDumpTensorView", "TaskDumpTensorViewContext", "UnsupportedTaskDumpPort",
]
