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
"""Autograd handoff between current MXFP8 root Parameters and physical execution."""

from dataclasses import dataclass
from typing import Any

import torch

from .mxfp8_weights import MXFP8ExpertWeightBinding


@dataclass(frozen=True)
class MXFP8ForwardState:
    """Owned output, ordinary saved activations and execution metadata for backward."""

    output: torch.Tensor
    saved_tensors: tuple[torch.Tensor, ...]
    metadata: Any
    workspace: Any = None


class _MXFP8ExpertsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        root1: torch.Tensor,
        root2: torch.Tensor,
        ids: torch.Tensor,
        probs: torch.Tensor,
        shapes: tuple[tuple[int, int, int], tuple[int, int, int]],
        execution: Any,
        grad_enabled: bool,
    ) -> torch.Tensor:
        ctx.weight1 = MXFP8ExpertWeightBinding(root1, shapes[0])
        ctx.weight2 = MXFP8ExpertWeightBinding(root2, shapes[1])
        retain_backward = grad_enabled and any(ctx.needs_input_grad)
        state = execution.forward(
            hidden, ids, probs, ctx.weight1.resolve('forward'), ctx.weight2.resolve('forward'), retain_backward,
        )
        ctx.execution = execution
        ctx.execution_metadata = state.metadata
        ctx.save_for_backward(*state.saved_tensors)
        ctx.workspace = state.workspace if retain_backward else None
        if ctx.workspace is not None:
            ctx.workspace.track_graph(ctx)
        return state.output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[Any, ...]:
        # HP pre-backward all-gather replaces these fields on the retained roots.
        # Never reuse the forward-only physical views held by native launch code.
        weight1 = ctx.weight1.resolve('backward')
        weight2 = ctx.weight2.resolve('backward')
        dx, dp, dw1, dw2 = ctx.execution.backward(
            gradient, ctx.saved_tensors, ctx.execution_metadata, weight1, weight2,
        )
        dw1 = dw1.reshape(ctx.weight1.root.shape).to(dtype=ctx.weight1.root.dtype)
        dw2 = dw2.reshape(ctx.weight2.root.shape).to(dtype=ctx.weight2.root.dtype)
        if ctx.workspace is not None and not torch._C._autograd._get_current_graph_task_keep_graph():
            ctx.workspace.release_graph(ctx)
        return dx, dw1, dw2, None, dp, None, None, None


def execute_mxfp8_experts(
    hidden: torch.Tensor,
    root1: torch.Tensor,
    root2: torch.Tensor,
    ids: torch.Tensor,
    probs: torch.Tensor,
    shapes: tuple[tuple[int, int, int], tuple[int, int, int]],
    execution: Any,
) -> torch.Tensor:
    """Keep root identity across phases and expose physical execution to autograd.

    ``execution.forward`` consumes current column-only weight storage and returns
    MXFP8ForwardState. ``execution.backward`` receives freshly resolved dual-axis
    weights and returns dX, dProbs, dW1, dW2. Neither interface may retain a weight
    view as an assumption about the next FSDP phase.
    """
    return _MXFP8ExpertsFunction.apply(hidden, root1, root2, ids, probs, shapes, execution, torch.is_grad_enabled())
