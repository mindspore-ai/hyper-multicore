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
"""GMM fill config: all GroupedMatMul variants (fwd up_proj/down_proj, bwd act/gate/w1/w2 grad)."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Set

import numpy as np

from hyper_multicore._core_spec import DEFAULT_CORE_SPEC
from hyper_multicore.scheduler.config import (
    TaskDescC, RuntimeConfigC,
    TaskAiCoreType, TaskType, TaskSplitValue,
    MAX_TENSOR_DIMS,
)
from hyper_multicore.tasks.task_base import FillConfig, fill_task_block
from hyper_multicore.scheduler.graph import OperatorNode
from hyper_multicore.tasks.utils import (
    advance_tsv_cube, advance_tsv_cube_only,
)


class GroupedMatmulTaskMode(IntEnum):
    """Mutually exclusive grouped-matmul coverage modes in extra_value_1."""

    PER_EXPERT = 0
    SINGLE_GROUP = 1
    BATCHED = 2
    BATCHED_READY = 3


@dataclass
class GmmFillConfig(FillConfig):
    """
    GMM fill config covering all variants (fwd up_proj/down_proj, bwd act_grad/w2_grad/
    gate_grad/w1_grad).  Tensor attributes (tensor_type, dtype_size, is_dynamic,
    transpose, shape) are read from TensorSpec.

    offset_inputs : Set[int]
        Input indices that receive base_ptr_offset and dynamic_shape.
        activation GMMs (up_proj, down_proj, act_grad, gate_grad): {0}
        weight-grad GMMs (w2_grad, w1_grad): {0, 1}

    transpose_inputs : Set[int]
        Additional transposed operands, without changing shared TensorSpec objects.

    rank_in_event : bool
        True  → dependent_event adds single_rank_expert_num * rank_id.
                 GMM1 (fwd/bwd), GMM4/w2_grad (bwd).
        False → no rank offset.
                 GMM2 (fwd/bwd), GMM3/w1_grad (bwd).

    global_trigger : bool
        False → trigger_event = pre_event_num + data_index + 1 (per-expert).
                 GMM1/GMM2 (activation-gradient path).
        True  → trigger_event = all_event_num (global).
                 GMM4/GMM3 (weight-gradient path).

    out_offset : bool
        True  → out.base_ptr_offset = data_index * 4096 * shape[1].
        False → 0 (weight-grad output writes to a dedicated buffer).

    advance : "cube" | "cube_only" | "cube_custom"
        "cube"        — standard cube advance (also advances events).
        "cube_only"   — advances task/cube counters only (no event update).
                        Used by: GMM1 bwd.
        "cube_custom" — manual: pre_task_num/pre_cube_task_num += task_num,
                        pre_pre_event_num = pre_event_num,
                        pre_event_num += event_delta.
                        Used by: GMM4 bwd, GMM3 bwd.

    event_delta : int
        Only used when advance="cube_custom".
        GMM4 bwd: single_rank_expert_num.  GMM3 bwd: 1.

    num_cube_cores : int
        Number of Cube cores.
    """
    offset_inputs:  Set[int] = field(default_factory=lambda: {0})
    # Shared activation specs retain their producer's interpretation.
    transpose_inputs: Set[int] = field(default_factory=set)
    rank_in_event:  bool     = False
    global_trigger: bool     = False
    out_offset:     bool     = True
    advance:        str      = "cube"   # "cube" | "cube_only" | "cube_custom"
    event_delta:    int      = 0        # only used when advance="cube_custom"
    num_cube_cores: int      = DEFAULT_CORE_SPEC.num_cube_cores

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Serialize operand interpretation and event dependencies.

        Args:
            cfg: Runtime descriptor to populate.
            op: Operator supplying operands and split metadata.
            tsv: Current task indices and expert dimensions.
        """
        task_num = op.task_num
        param    = op.param_positions
        glist_j  = len(op.inputs) - 1   # last input is always group_list

        # Every Cube sees the same operand schema. Only expert offsets, event
        # numbers and queue indices vary across this operator's tasks.
        if task_num:
            template = TaskDescC()
            template.task_type = TaskType.TASK_GROUPED_MATMUL
            template.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
            template.num_inputs = len(op.inputs)
            template.num_outputs = len(op.outputs)
            template.task_split_num = task_num
            template.task_split_value = op.split_value
            template.tiling_data_position = op.tiling_position
            for j, spec in enumerate(op.inputs):
                operand = template.inputs[j]
                operand.input_position = param[j]
                operand.tensor_type = 0 if j == glist_j else spec.tensor_type
                operand.data_type = spec.dtype_size
                operand.dynamic_shape = int(j in self.offset_inputs and spec.is_dynamic)
                operand.transpose_flag = int(spec.transpose or j in self.transpose_inputs)
                for k, size in enumerate(spec.shape[:MAX_TENSOR_DIMS]):
                    operand.dim[k] = size

            out_spec = op.outputs[0]
            out = template.outputs[0]
            out.tensor_type = out_spec.tensor_type
            out.data_type = out_spec.dtype_size
            out.input_position = param[len(op.inputs)]
            out.dynamic_shape = int(out_spec.is_dynamic)
            if self.out_offset:
                for k, size in enumerate(out_spec.shape[:MAX_TENSOR_DIMS]):
                    out.dim[k] = size

            tasks = fill_task_block(cfg, tsv.pre_task_num, task_num, template)
            index = np.arange(task_num, dtype=np.int64)
            expert = index // self.num_cube_cores
            tasks["task_index"] = index
            for j, spec in enumerate(op.inputs):
                if j in self.offset_inputs:
                    tasks["inputs"][:, j]["base_ptr_offset"] = expert * (4096 * spec.shape[1])
            if self.out_offset:
                tasks["outputs"][:, 0]["base_ptr_offset"] = expert * (4096 * out_spec.shape[1])

            rank_offset = tsv.single_rank_expert_num * tsv.rank_id if self.rank_in_event else 0
            tasks["dependent_event"] = tsv.pre_pre_event_num + rank_offset + expert + 1
            trigger = tsv.all_event_num if self.global_trigger else tsv.pre_event_num + expert + 1
            tasks["trigger_event"] = trigger
            np.ctypeslib.as_array(cfg.all_event_num_triggers)[trigger] = self.num_cube_cores
            queue = np.ctypeslib.as_array(cfg.cube_task_indices)
            queue[tsv.pre_cube_task_num:tsv.pre_cube_task_num + task_num] = tsv.pre_task_num + index

        self._advance_descriptor_state(cfg, tsv, task_num)

    def _advance_descriptor_state(self, cfg, tsv, task_num):
        """Advance queue and event cursors after the serialized task block."""
        cfg.task_index_num[0] += task_num
        if self.advance == "cube":
            advance_tsv_cube(tsv, task_num, event_group_size=task_num // self.num_cube_cores)
        elif self.advance == "cube_only":
            advance_tsv_cube_only(tsv, task_num)
        else:   # "cube_custom"
            tsv.pre_task_num      += task_num
            tsv.pre_cube_task_num += task_num
            tsv.pre_pre_event_num  = tsv.pre_event_num
            tsv.pre_event_num     += self.event_delta
