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
"""Abstract base class for all operator fill configs."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from hyper_multicore.scheduler.config import RuntimeConfigC, TaskDescC


class FillConfig(ABC):
    """
    Abstract base for all fill configs.

    Subclasses hold per-operator config data and implement fill() with the
    task-building logic.  OperatorNode.fill_config holds an instance;
    gen_runtime_data calls op.fill_config.fill(cfg, op, tsv).
    """

    @abstractmethod
    def fill(self, cfg: RuntimeConfigC, op: Any, tsv: Any) -> None:
        """Fill tasks into cfg for the given op using runtime state tsv."""


_TASK_DTYPE = np.dtype(TaskDescC)


def fill_task_block(cfg: RuntimeConfigC, start: int, count: int, template: TaskDescC) -> np.ndarray:
    """Copy common fields into this operator's contiguous slice of the host ABI."""
    storage = np.frombuffer(cfg.all_tasks, dtype=_TASK_DTYPE)
    if start < 0 or count < 0 or start + count > len(storage):
        raise IndexError("task block exceeds runtime capacity")
    block = storage[start:start + count]
    block[:] = np.frombuffer(template, dtype=storage.dtype, count=1)[0]
    return block
