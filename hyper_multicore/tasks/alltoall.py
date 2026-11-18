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
"""AllToAll fill config: dispatch and combine (forward + backward)."""
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

from hyper_multicore.scheduler.config import (
    TaskDescC, RuntimeConfigC,
    TaskAiCoreType, TaskType, TaskSplitValue,
)
from hyper_multicore.scheduler.graph import OperatorNode
from hyper_multicore.tasks.task_base import FillConfig, fill_task_block
from hyper_multicore.tasks.utils import (
    advance_tsv_vector, advance_tsv_vector_only,
)


class ShmemTaskPartition(IntEnum):
    """Partition each peer/expert transfer into contiguous row ranges.

    FIXED_ROWS uses task_split_value rows per task. RUNTIME_ROWS divides the
    actual runtime row count evenly among that PUT's task descriptors.
    GET continues to use fixed row tiles.
    Both modes publish one completion credit per task, including empty ranges.
    """

    FIXED_ROWS = 0
    RUNTIME_ROWS = 2


class AllToAllType(Enum):
    """
    MoE AllToAll semantic type — determines event wiring.

    DISPATCH — transfer source tokens into expert storage via configured PUT or GET.
        dependent_event = pre_pre_event_num + 0
        trigger_event   = pre_event_num + global_expert + 1
        trigger_count   = task_num * ep // all_expert_num

    COMBINE — gather expert results back to the originating rank.
        dependent_event = pre_pre_event_num + (i // per_g_e_num) % sre + 1
        trigger_event   = all_event_num
        trigger_count   = task_num

    OTHER — reserved for AllToAll patterns outside the MoE dispatch/combine semantic.
    """
    DISPATCH = 1
    COMBINE  = 2
    OTHER    = 3


@dataclass
class AllToAllFillConfig(FillConfig):
    """
    AllToAll fill config covering dispatch and combine for fwd and bwd.

    moe_type : AllToAllType
        Determines event wiring; see AllToAllType for details.

    advance : "vector" | "vector_only"
        "vector"      — advance_tsv_vector(tsv, task_num, event_group_size=event_group)
                        Used by: dispatch (fwd/bwd), combine forward.
        "vector_only" — advance_tsv_vector_only(tsv, task_num)
                        Only advances pre_task_num/pre_vector_task_num.
                        Used by: combine backward.

    event_group : int
        Only effective when advance="vector".
        dispatch (fwd/bwd): all_expert_num
        combine forward:    1
    """
    moe_type:    AllToAllType = AllToAllType.DISPATCH
    advance:     str          = "vector"   # "vector" | "vector_only"
    event_group: int          = 1          # only used when advance="vector"

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Emit configured dispatch and PUT combine tasks with completion events.

        Args:
            cfg: Runtime receiving task descriptors and event thresholds.
            op: Communication node with fixed tensor argument positions.
            tsv: Topology and running graph offsets.
        """
        task_num    = op.task_num
        per_g_e_num = task_num // tsv.all_expert_num
        param       = op.param_positions

        if task_num:
            dispatch = self.moe_type == AllToAllType.DISPATCH
            pull = dispatch and tsv.dispatch_mode == "pull"
            template = TaskDescC()
            template.task_type = TaskType.TASK_SHMEM_GET_MEM if pull else TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            template.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
            template.num_inputs = len(op.inputs)
            template.num_outputs = len(op.outputs)
            template.task_split_num = task_num
            template.task_split_value = op.split_value
            template.extra_value_1 = ShmemTaskPartition.RUNTIME_ROWS if hasattr(op, "compact_lanes") else 0
            template.tiling_data_position = 0xFFFFFFFF

            for j, spec in enumerate(op.inputs):
                operand = template.inputs[j]
                operand.input_position = param[j]
                operand.data_type = spec.dtype_size
                if j == 1:
                    operand.tensor_type = spec.tensor_type
                    operand.dynamic_shape = int(spec.is_dynamic)
            out_spec = op.outputs[0]
            out = template.outputs[0]
            out.tensor_type = out_spec.tensor_type
            out.data_type = out_spec.dtype_size
            out.input_position = param[len(op.inputs)]
            out.dynamic_shape = int(out_spec.is_dynamic)

            tasks = fill_task_block(cfg, tsv.pre_task_num, task_num, template)
            index = np.arange(task_num, dtype=np.int64)
            if per_g_e_num == 0:
                raise ValueError("communication schedule needs at least one task per expert")
            expert = index // per_g_e_num
            tasks["task_index"] = index
            for j in range(len(op.inputs)):
                if j != 1:
                    tasks["inputs"][:, j]["base_ptr_offset"] = expert
            events = np.ctypeslib.as_array(cfg.all_event_num_triggers)
            if dispatch:
                owner = (tsv.rank_id * tsv.single_rank_expert_num + expert % tsv.single_rank_expert_num
                         if pull else expert)
                trigger = tsv.pre_event_num + owner + 1
                tasks["dependent_event"] = tsv.pre_pre_event_num
                tasks["trigger_event"] = trigger
                events[trigger] = task_num * tsv.ep // tsv.all_expert_num
            else:
                tasks["dependent_event"] = tsv.pre_pre_event_num + expert % tsv.single_rank_expert_num + 1
                tasks["trigger_event"] = tsv.all_event_num
                events[tsv.all_event_num] = task_num
            queue = np.ctypeslib.as_array(cfg.vector_task_indices)
            queue[tsv.pre_vector_task_num:tsv.pre_vector_task_num + task_num] = tsv.pre_task_num + index

        cfg.task_index_num[1] += task_num
        if self.advance == "vector":
            advance_tsv_vector(tsv, task_num, event_group_size=self.event_group)
        else:   # "vector_only"
            advance_tsv_vector_only(tsv, task_num)


class ShmemTransferEngine(IntEnum):
    """Transport-owned engine bit; independent of dependency polling and payload."""
    MTE = 0
    UDMA = 16


def set_transfer_engine(task: TaskDescC, engine: str) -> None:
    """Encode an engine without touching partition, scalar payload or runtime fields."""
    if engine not in ("mte", "udma"):
        raise ValueError("SHMEM transfer engine must be mte or udma")
    task.extra_value_1 = (task.extra_value_1 & ~16) | int(
        ShmemTransferEngine.UDMA if engine == "udma" else ShmemTransferEngine.MTE)
