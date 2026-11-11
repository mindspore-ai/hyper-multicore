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
"""MegaMoe-owned expert semantics shared by its diagnostic tools."""

__all__ = []

from dataclasses import dataclass
from typing import Any

from hyper_multicore.scheduler.config import (
    TaskDescC,
    TaskSplitValue,
)
from hyper_multicore.scheduler.graph import OperatorNode, OpType


MEGA_MOE_OWNER_LABEL = "Expert"


@dataclass(frozen=True)
class _MegaMoeDiagnosticContext:
    """Host values needed to resolve expert ownership after task construction."""

    topology: TaskSplitValue
    num_cube_cores: int


def _require_diagnostic_context(context: Any) -> _MegaMoeDiagnosticContext:
    if not isinstance(context, _MegaMoeDiagnosticContext):
        raise TypeError(
            f"MegaMoe diagnostic context must be _MegaMoeDiagnosticContext, got {type(context).__name__}"
        )
    return context


def _resolve_expert_owner(
    operator: OperatorNode,
    task_desc: TaskDescC,
    raw_context: Any,
) -> int:
    """Resolve the global expert ID from a graph operator and its concrete task."""
    context = _require_diagnostic_context(raw_context)
    topology = context.topology
    rank_owner_base = topology.rank_id * topology.single_rank_expert_num

    if operator.op_type == OpType.GMM:
        return rank_owner_base + task_desc.task_index // (task_desc.extra_value_0 or context.num_cube_cores)

    if operator.op_type in (OpType.SWIGLU, OpType.SWIGLU_GRAD):
        if task_desc.task_split_value == 0:
            raise ValueError("SwiGLU diagnostic owner requires a non-zero task_split_value")
        num_triggers = topology.per_expert_seq // task_desc.task_split_value
        if num_triggers == 0:
            raise ValueError("SwiGLU diagnostic owner resolved a zero task count per expert")
        return rank_owner_base + task_desc.task_index // num_triggers

    if operator.op_type == OpType.ALLTOALL:
        task_count_per_expert = task_desc.task_split_num // topology.all_expert_num
        if task_count_per_expert == 0:
            raise ValueError("AllToAll diagnostic owner resolved a zero task count per expert")
        expert_index = task_desc.task_index // task_count_per_expert
        if operator.name == "dispatch":
            if topology.dispatch_mode == "pull":
                return rank_owner_base + expert_index % topology.single_rank_expert_num
            return expert_index
        if operator.name == "combine":
            return rank_owner_base + expert_index % topology.single_rank_expert_num
        raise ValueError(f"MegaMoe AllToAll operator has no expert owner rule: {operator.name!r}")

    raise ValueError(f"MegaMoe operator type has no expert owner rule: {operator.op_type}")


def _mega_moe_owner_options(
    topology: TaskSplitValue,
    *,
    num_cube_cores: int,
) -> dict[str, Any]:
    """Supply only MegaMoe's owner semantics to the shared diagnostics adapter."""
    if not isinstance(topology, TaskSplitValue):
        raise TypeError(f"topology must be TaskSplitValue, got {type(topology).__name__}")
    if not isinstance(num_cube_cores, int) or isinstance(num_cube_cores, bool) or num_cube_cores <= 0:
        raise ValueError(f"num_cube_cores must be a positive integer, got {num_cube_cores!r}")
    context = _MegaMoeDiagnosticContext(topology=topology, num_cube_cores=num_cube_cores)
    return {
        "owner_label": MEGA_MOE_OWNER_LABEL,
        "owner_resolver": _resolve_expert_owner,
        "owner_context": context,
    }
