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
"""Task-type decoding for MegaMoE's existing 32-bit task option ABI."""
from enum import IntEnum
from typing import Any

from hyper_multicore.tasks.gmm import GroupedMatmulTaskMode
from hyper_multicore.tasks.alltoall import ShmemTaskPartition
from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.tasks.swiglu import SwiGluClampPosition, SwiGluTaskIteration
from .probability_abi import ProbabilityLayout


class MegaMoeActivationWeighting(IntEnum):
    """Activation math, distinct from a transport payload with the same bit."""
    UNWEIGHTED = 0
    BEFORE_FC2 = 4


class MegaMoeTransportPayload(IntEnum):
    """Scalar traffic completing before the original token completion credit."""
    TOKENS = 0
    TOKENS_WITH_PROBABILITIES = 4
    TOKENS_WITH_PROBABILITY_GRADIENTS = 8


def activation_weighting(task: Any) -> MegaMoeActivationWeighting:
    """Decode activation options only for activation tasks."""
    if task.task_type not in (TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
        raise ValueError("activation weighting requires a SwiGLU task")
    return MegaMoeActivationWeighting(task.extra_value_1 & 4)


def transport_payload(task: Any) -> MegaMoeTransportPayload:
    """Decode mutually exclusive transport payloads only for transfer tasks."""
    if task.task_type not in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
        raise ValueError("transport payload requires a SHMEM task")
    return MegaMoeTransportPayload(task.extra_value_1 & 12)


def _activation_options(task: Any) -> bool:
    allowed = int(SwiGluTaskIteration.PERSISTENT_LANES | SwiGluClampPosition.AFTER_SILU
                  | MegaMoeActivationWeighting.BEFORE_FC2)
    if task.extra_value_1 & ~allowed:
        raise ValueError(f"invalid activation flags {task.extra_value_1:#x}")
    if task.extra_value_1 & SwiGluClampPosition.AFTER_SILU and task.extra_value_0 == 0:
        raise ValueError("clamp position requires a clamp limit")
    return activation_weighting(task) == MegaMoeActivationWeighting.BEFORE_FC2


def _transport_options(task: Any) -> bool:
    allowed = int(ShmemTaskPartition.RUNTIME_ROWS | MegaMoeTransportPayload.TOKENS_WITH_PROBABILITIES
                  | MegaMoeTransportPayload.TOKENS_WITH_PROBABILITY_GRADIENTS)
    if task.extra_value_1 & ~allowed:
        raise ValueError(f"invalid transport flags {task.extra_value_1:#x}")
    payload = transport_payload(task)
    if (task.task_type == TaskType.TASK_SHMEM_GET_MEM
            and payload == MegaMoeTransportPayload.TOKENS_WITH_PROBABILITY_GRADIENTS):
        raise ValueError("GET cannot return probability gradients")
    return payload != MegaMoeTransportPayload.TOKENS


def _validate_operands(task: Any, weighted: bool, inputs: int) -> None:
    if task.num_inputs != inputs or task.num_outputs != (2 if weighted else 1):
        raise ValueError("task operands do not match its token and probability options")
    if not weighted:
        return
    layout = ProbabilityLayout.from_descriptor(task.outputs[1])
    positions = {TaskType.TASK_SWI_GLU: 16, TaskType.TASK_SWI_GLU_GRAD: 25}
    if layout.parameter_position != positions.get(task.task_type, layout.parameter_position):
        raise ValueError("probability operand uses the wrong forward/backward port")


def validate_task_options(task: Any) -> None:
    """Reject unknown bits, conflicting payloads and missing auxiliary operands."""
    kind = task.task_type
    if kind == TaskType.TASK_GROUPED_MATMUL:
        if task.extra_value_1 not in tuple(GroupedMatmulTaskMode) or task.num_inputs != 3 or task.num_outputs != 1:
            raise ValueError("invalid grouped matmul task mode or operands")
    elif kind in (TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
        _validate_operands(task, _activation_options(task), 1 if kind == TaskType.TASK_SWI_GLU else 2)
    elif kind in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
        _validate_operands(task, _transport_options(task), 4)
