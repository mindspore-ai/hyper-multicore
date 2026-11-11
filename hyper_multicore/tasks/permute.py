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
"""Fill configs for deterministic route reduction tasks."""
from dataclasses import dataclass
from typing import ClassVar

from hyper_multicore.scheduler.config import (
    RuntimeConfigC,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)
from hyper_multicore.scheduler.graph import OperatorNode
from hyper_multicore.tasks.task_base import FillConfig
from hyper_multicore.tasks.utils import (
    advance_tsv_vector,
    advance_tsv_vector_only,
    make_tensor_desc,
)


@dataclass
class _RouteFillConfig(FillConfig):
    """Build the common static descriptor shape for route vector tasks."""

    dependent_event: int = 0
    trigger_event: int = 0
    top_k: int = 1
    local_completion_event: int = 0
    shared_output_slot: int = -1
    add_shared_residual: bool = True
    _task_type: ClassVar[TaskType]
    _advance_event: ClassVar[bool] = True

    def _extra_value_1(self) -> int:
        """Return the optional task-specific second scalar."""
        return 0

    def fill(
        self,
        cfg: RuntimeConfigC,
        op: OperatorNode,
        tsv: TaskSplitValue,
    ) -> None:
        """Append route vector tasks with identical tensor descriptors.

        Args:
            cfg: Runtime receiving vector tasks and their dependencies.
            op: Token-reduction node with fixed native tensor ports.
            tsv: Expert topology and running vector queue offsets.
        """
        task_num = op.task_num
        param = op.param_positions
        for i in range(task_num):
            task = TaskDescC()
            task.task_type = self._task_type
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
            task.num_inputs = len(op.inputs)
            task.num_outputs = len(op.outputs)
            task.dependent_event = self.dependent_event
            task.trigger_event = self.trigger_event
            task.extra_value_0 = self.top_k
            task.extra_value_1 = self._extra_value_1()
            if self.shared_output_slot >= 0 and self.add_shared_residual:
                task.extra_value_1 |= 2

            for j, spec in enumerate(op.inputs):
                task.inputs[j] = make_tensor_desc(
                    tensor_type=spec.tensor_type,
                    data_type=spec.dtype_size,
                    input_position=param[j],
                    shape=spec.shape,
                )

            for j, spec in enumerate(op.outputs):
                task.outputs[j] = make_tensor_desc(
                    tensor_type=spec.tensor_type,
                    data_type=spec.dtype_size,
                    input_position=param[task.num_inputs + j],
                    shape=spec.shape,
                )
                if j == 0 and self.shared_output_slot >= 0:
                    task.outputs[j].tensor_type = 2
                    task.outputs[j].dynamic_dim = self.shared_output_slot

            if self.local_completion_event:
                # Weighted unpermute reuses its fourth input as an owned snapshot.
                # Its unused fourth dimension carries the local read-completion fence.
                task.inputs[3].dim[3] = self.local_completion_event
            task.task_index = i
            task.task_split_num = task_num
            task.task_split_value = op.split_value
            task.tiling_data_position = 0xFFFFFFFF
            cfg.all_tasks[tsv.pre_task_num + i] = task
            cfg.vector_task_indices[tsv.pre_vector_task_num + i] = (
                tsv.pre_task_num + i
            )

        cfg.task_index_num[1] += task_num
        cfg.all_event_num_triggers[self.trigger_event] = task_num
        if self._advance_event:
            advance_tsv_vector(tsv, task_num, event_group_size=1)
        else:
            advance_tsv_vector_only(tsv, task_num)


@dataclass
class UnpermuteReduceFillConfig(_RouteFillConfig):
    """Fill deterministic backward no-scale route reductions."""

    _task_type = TaskType.TASK_UNPERMUTE_REDUCE_GRAD


@dataclass
class WeightedUnpermuteFillConfig(_RouteFillConfig):
    """Fill forward weighted reductions or acquire-only terminal epilogues."""

    acquire_remote_writes: bool = False
    _task_type = TaskType.TASK_WEIGHTED_UNPERMUTE

    def _extra_value_1(self) -> int:
        """Encode whether this task is an acquire-only terminal epilogue."""
        return int(self.acquire_remote_writes)


@dataclass
class TopKWeightGradFillConfig(_RouteFillConfig):
    """Fill independent route-output/dy dot products for router gradients."""

    _task_type = TaskType.TASK_TOPK_WEIGHT_GRAD
    _advance_event = False
