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
"""Graph-declared probability operands and scalar effects lowered with tokens."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from typing import Any

from hyper_multicore.scheduler.graph import TensorEffect
from hyper_multicore.scheduler.config import TaskType

from .task_options import (
    MegaMoeActivationWeighting, MegaMoeTransportPayload, activation_weighting, transport_payload, validate_task_options,
)
from .probability_abi import (  # re-export the existing internal import contract
    ProbabilityLayout, ProbabilityPlane, PROBABILITY_PLANES, PROBABILITY_DTYPE_BYTES,
    PROBABILITY_POSITION_FORWARD, PROBABILITY_POSITION_BACKWARD,
)


__all__ = [
    "ProbabilityLayout", "ProbabilityPlane", "PROBABILITY_PLANES", "PROBABILITY_DTYPE_BYTES",
    "PROBABILITY_POSITION_FORWARD", "PROBABILITY_POSITION_BACKWARD", "probability_stride",
    "probability_layout_for_task", "probability_capture_plane", "declare_probability_dataflow",
]

def probability_stride(spec: Any) -> int:
    """Return the conservative routed-slot bound of one probability plane."""
    return spec.ep_size * spec.plan_tokens * spec.top_k


def probability_layout_for_task(task: Any) -> ProbabilityLayout | None:
    """Read an optional MegaMoe probability operand from an already scheduled task."""
    if task.task_type in (TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
        present = activation_weighting(task) == MegaMoeActivationWeighting.BEFORE_FC2
    elif task.task_type in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
        present = transport_payload(task) != MegaMoeTransportPayload.TOKENS
    else:
        present = False
    if not present:
        return None
    return ProbabilityLayout.from_descriptor(task.outputs[task.num_outputs - 1])


def probability_capture_plane(operator: Any) -> ProbabilityPlane:
    """Derive the local diagnostic slice from the graph's declared scalar effects."""
    effect, = operator.tensor_effects
    local_regions = effect.writes or effect.reads
    planes = {"source_p": ProbabilityPlane.SOURCE, "received_p": ProbabilityPlane.RECEIVED,
              "received_dp": ProbabilityPlane.RECEIVED_GRADIENT, "returned_dp": ProbabilityPlane.RETURNED_GRADIENT}
    if len(local_regions) != 1:
        raise ValueError("probability capture requires one local scalar region")
    return planes[local_regions[0]]


@dataclass(frozen=True)
class ProbabilityTaskBinding:
    """Logical scalar effects lowered with their token task and original event."""

    layout: ProbabilityLayout
    option: MegaMoeActivationWeighting | MegaMoeTransportPayload
    reads: tuple[str, ...]
    writes: tuple[str, ...] = ()
    remote_writes: tuple[str, ...] = ()


@dataclass
class ProbabilityFillConfig:
    """Compose the existing math/transfer filler with an explicit graph operand."""

    token_config: Any
    binding: ProbabilityTaskBinding

    def fill(self, config: Any, operator: Any, topology: Any) -> None:
        """Serialize both token and scalar operands from the same logical node."""
        token_operator = copy(operator)
        token_operator.outputs = operator.outputs[:-1]
        token_operator.param_positions = operator.param_positions[:-1]
        token_operator.fill_config = self.token_config
        start = topology.pre_task_num
        self.token_config.fill(config, token_operator, topology)
        for index in range(start, start + operator.task_num):
            task = config.all_tasks[index]
            task.outputs[task.num_outputs] = self.binding.layout.descriptor()
            task.num_outputs += 1
            task.extra_value_1 |= int(self.binding.option)
            validate_task_options(task)


def declare_probability_dataflow(graph: Any, stride: int, *, backward: bool, pull: bool) -> None:
    """Declare scalar aliases/effects before splitting, lowering and scheduling.

    Dispatch carries source p with tokens. Activation consumes received p; its
    backward writes received dp. CombineGrad returns dp with dX. All transfers
    finish scalar and token payloads before publishing the original credit.
    """
    position = PROBABILITY_POSITION_BACKWARD if backward else PROBABILITY_POSITION_FORWARD
    layout = ProbabilityLayout(stride, position)
    bindings = {
        "dispatch": ProbabilityTaskBinding(layout, MegaMoeTransportPayload.TOKENS_WITH_PROBABILITIES,
                                            ("source_p",), remote_writes=("received_p",)),
        "swiglu_grad" if backward else "swiglu": ProbabilityTaskBinding(
            layout, MegaMoeActivationWeighting.BEFORE_FC2, ("received_p",),
            writes=("received_dp",) if backward else ()),
    }
    if backward:
        bindings["combine"] = ProbabilityTaskBinding(
            layout, MegaMoeTransportPayload.TOKENS_WITH_PROBABILITY_GRADIENTS,
            ("received_dp",), remote_writes=("returned_dp",))
    for name, binding in bindings.items():
        operator = graph.get_op(name)
        operand = layout.tensor_spec()
        reads, writes, remote, remote_reads = binding.reads, binding.writes, binding.remote_writes, ()
        if name == "dispatch" and pull:
            reads, writes, remote, remote_reads = (), ("received_p",), (), ("source_p",)
        operator.outputs.append(operand)
        operator.param_positions.append(position)
        operator.split_spec = replace(operator.split_spec,
                                      split_output_dims=[*operator.split_spec.split_output_dims, -1])
        operator.tensor_effects = (TensorEffect(operand, reads, writes, remote, remote_reads,
                                                completion="token_and_scalar"),)
        operator.fill_config = ProbabilityFillConfig(operator.fill_config, binding)
