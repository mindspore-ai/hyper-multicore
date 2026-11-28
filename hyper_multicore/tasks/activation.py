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
"""Lower BF16 GLU activations to vector tasks with shared row and event splitting."""

import math
import struct
from dataclasses import dataclass
from enum import IntEnum

from hyper_multicore.scheduler.config import (
    TaskDescC, TensorDescC, RuntimeConfigC,
    TaskAiCoreType, TaskType, TaskSplitValue,
    MAX_TENSOR_DIMS, SITU_TASKS, SITU_RUNTIME_PROTOCOL_VERSION,
)
from hyper_multicore.scheduler.graph import OperatorNode, OpType
from hyper_multicore.tasks.task_base import FillConfig
from hyper_multicore.tasks.utils import advance_tsv_vector


ACTIVATION_TASK_TYPES = {
    OpType.SWIGLU: TaskType.TASK_SWI_GLU,
    OpType.SWIGLU_GRAD: TaskType.TASK_SWI_GLU_GRAD,
    OpType.SITU_GLU: TaskType.TASK_SITU_GLU,
    OpType.SITU_GLU_GRAD: TaskType.TASK_SITU_GLU_GRAD,
}


class SwiGluClampPosition(IntEnum):
    """Gate-branch clamp position encoded by SwiGLU and SwiGLUGrad tasks."""

    BEFORE_SILU = 0
    AFTER_SILU = 1


class ActivationTaskIteration(IntEnum):
    """TILED covers one tile; PERSISTENT_LANES visits tile lane + n * lanes.

    Persistent descriptors encode group * lanes + lane in task_index and
    groups * lanes in task_split_num. task_split_value is the row tile size.
    Empty groups do no row work and retain their original completion credit.
    """

    TILED = 0
    PERSISTENT_LANES = 2


def encode_activation_bound(value: float | None, *, name: str) -> int:
    """Validate and encode an optional positive float32 activation bound.

    Args:
        value: Activation bound; None encodes the disabled value zero.
        name: Configuration field used in validation errors.

    Returns:
        Unsigned integer carrying the bound's float32 bit pattern.

    Raises:
        ValueError: If the bound is not a finite positive float32 value.
    """
    if value is None:
        return 0
    try:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("expected a real activation bound")
        encoded = struct.pack("<f", value)
        rounded = struct.unpack("<f", encoded)[0]
        if not math.isfinite(rounded) or rounded <= 0:
            raise ValueError("activation bound is not positive and finite")
    except (OverflowError, TypeError, ValueError, struct.error):
        raise ValueError(
            f"{name} must be None or a finite positive float32-representable number, "
            f"got {value!r}."
        ) from None
    return struct.unpack("<I", encoded)[0]


@dataclass
class ActivationFillConfig(FillConfig):
    """
    Fill an activation task with its tensor operands, row partition and completion events.

    ``clamp_limit`` is serialized in ``TaskDesc.extra_value_0``.  Zero keeps
    the unclamped path; a positive value enables clipped SwiGLU.
    SiTUGLU parameters are carried in the activation tiling input.
    """

    clamp_limit: float | None = None
    clamp_after_activation: bool = False

    def fill(
        self,
        cfg: RuntimeConfigC,
        op: OperatorNode,
        tsv: TaskSplitValue,
    ) -> None:
        """Serialize activation tasks into the runtime configuration.

        Args:
            cfg: Runtime configuration receiving task descriptors.
            op: Activation operator node to serialize.
            tsv: Topology and task split values for the current graph.
        """
        task_type = ACTIVATION_TASK_TYPES[op.op_type]
        clamp_bits = encode_activation_bound(self.clamp_limit, name="clamp_limit")
        if task_type in SITU_TASKS:
            if clamp_bits or self.clamp_after_activation:
                raise ValueError("SiTUGLU does not support SwiGLU clipping")
            cfg.protocol_version |= SITU_RUNTIME_PROTOCOL_VERSION

        num_triggers = getattr(op, "compact_lanes", tsv.per_expert_seq // op.split_value)
        task_num     = op.task_num
        param        = op.param_positions

        for i in range(task_num):
            task = TaskDescC()
            task.task_type        = task_type
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
            task.num_inputs       = len(op.inputs)
            task.num_outputs      = len(op.outputs)

            for j, spec in enumerate(op.inputs):
                td = TensorDescC()
                td.tensor_type     = 1   # all activation inputs are tensor lists
                td.data_type       = spec.dtype_size
                td.input_position  = param[j]
                td.base_ptr_offset = i * spec.shape[1] * op.split_value
                td.dynamic_shape   = int(spec.is_dynamic)
                for k in range(min(len(spec.shape), MAX_TENSOR_DIMS)):
                    td.dim[k] = spec.shape[k]
                task.inputs[j] = td

            for j, spec in enumerate(op.outputs):
                td = TensorDescC()
                td.tensor_type     = 1   # all activation outputs are tensor lists
                td.data_type       = spec.dtype_size
                td.input_position  = param[task.num_inputs + j]
                td.base_ptr_offset = i * spec.shape[1] * op.split_value
                td.dynamic_shape   = int(spec.is_dynamic)
                for k in range(min(len(spec.shape), MAX_TENSOR_DIMS)):
                    td.dim[k] = spec.shape[k]
                task.outputs[j] = td

            ev_idx = i // num_triggers
            task.dependent_event = tsv.pre_pre_event_num + ev_idx + 1
            task.trigger_event   = tsv.pre_event_num + ev_idx + 1
            cfg.all_event_num_triggers[task.trigger_event] = num_triggers

            task.task_index           = i
            task.task_split_num       = task_num
            task.task_split_value     = op.split_value
            task.extra_value_0        = clamp_bits
            task.extra_value_1        = (SwiGluClampPosition.AFTER_SILU
                                        if self.clamp_after_activation and clamp_bits else 0)
            if hasattr(op, "compact_lanes"):
                task.extra_value_1 |= ActivationTaskIteration.PERSISTENT_LANES
            task.tiling_data_position = op.tiling_position

            cfg.all_tasks[tsv.pre_task_num + i]               = task
            cfg.vector_task_indices[tsv.pre_vector_task_num + i] = tsv.pre_task_num + i

        cfg.task_index_num[1] += task_num
        advance_tsv_vector(tsv, task_num, event_group_size=tsv.single_rank_expert_num)
