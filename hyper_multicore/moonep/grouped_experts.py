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
"""Ordinary grouped-GEMM experts with official MoonEP parameter transfers."""
from __future__ import annotations

from typing import Any

import torch


def _gmm(x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    import torch_npu

    # Ordinary GMM reads SHMEM-backed weights slowly on A3. Stage only the
    # current projection in ordinary device memory; do not retain it in ctx or
    # change the shared bank's ownership, lifetime, or backward refresh rules.
    transposed = not weight.is_contiguous() and weight.transpose(1, 2).is_contiguous()
    owned = (weight.transpose(1, 2) if transposed else weight).clone()
    weight = owned.transpose(1, 2) if transposed else owned
    return torch_npu.npu_grouped_matmul(
        [x], [weight], group_list=counts, split_item=3, group_type=0, group_list_type=1)[0]


def _weight_gradient(x: torch.Tensor, dy: torch.Tensor, counts: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    import torch_npu

    # GMM's BF16 output_dtype=float32 mode is unsupported by CANN 9.0.
    # Its existing additive API computes straight into FP32, including cold zeros.
    gradient = torch.zeros(shape, dtype=torch.float32, device=x.device)
    if x.shape[0]:
        torch_npu.npu_grouped_matmul_add_(gradient, x, dy, counts.cumsum(0), group_list_type=0)
    return gradient


class TransferredExperts(torch.autograd.Function):
    """Save owned activations and home weights, never mutable staging views."""

    @staticmethod
    def forward(
        ctx: Any, hidden: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
        counts: torch.Tensor, copies: torch.Tensor, resources: Any, activation: Any,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        ctx.resources = resources
        ctx.activation = activation
        ctx.shapes = weight1.shape, weight2.shape
        resources.claim()
        try:
            bank1, bank2 = resources.prefetch((weight1, weight2), copies)
            if hidden.shape[0]:
                up = _gmm(hidden, bank1, counts)
                activated = (activation(up) * probs.unsqueeze(-1)).to(up.dtype)
                output = _gmm(activated, bank2, counts)
            else:
                up = hidden.new_empty((0, bank1.shape[-1]))
                activated = hidden.new_empty((0, bank2.shape[1]))
                output = hidden.new_empty(hidden.shape)
            generation = torch.tensor(resources.cache_generation, dtype=torch.int64, device="cpu")
            ctx.save_for_backward(hidden, up, activated, weight1, weight2, counts, copies, generation, probs)
            resources.graphs.add(ctx)
            return output
        finally:
            resources.release()

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple:
        # Unpacking can trigger checkpoint recomputation; acquire staging later.
        hidden, up, activated, weight1, weight2, counts, copies, generation, probs = ctx.saved_tensors
        ctx.maybe_clear_saved_tensors()
        resources = ctx.resources
        resources.claim()
        try:
            if generation.item() == resources.cache_generation:
                bank1, bank2 = (projection.bank[:resources.local_experts + copies.shape[1]]
                                for projection in resources.projections)
            else:
                bank1, bank2 = resources.prefetch((weight1, weight2), copies)
            dw2 = _weight_gradient(activated, dy, counts, bank2.shape)
            if hidden.shape[0]:
                dactivation = _gmm(dy, bank2.transpose(1, 2), counts)
                with torch.enable_grad():
                    projected = up.detach().requires_grad_(True)
                    probabilities = probs.detach().requires_grad_(True)
                    output = (ctx.activation(projected) * probabilities.unsqueeze(-1)).to(projected.dtype)
                    dprojected, dprobs = torch.autograd.grad(
                        output, (projected, probabilities), dactivation,
                    )
                dx = _gmm(dprojected, bank1.transpose(1, 2), counts) if ctx.needs_input_grad[0] else None
            else:
                dprojected = up.new_empty(up.shape)
                dprobs = torch.zeros_like(probs)
                dx = hidden.new_empty(hidden.shape) if ctx.needs_input_grad[0] else None
            dw1 = _weight_gradient(hidden, dprojected, counts, bank1.shape)
            owners = resources.return_generated_gradients(
                (dw1, dw2), copies, output_dtypes=(weight1.dtype, weight2.dtype))
            keep_graph = torch._C._autograd._get_current_graph_task_keep_graph()
            if not keep_graph:
                resources.graphs.discard(ctx)
            return dx, owners[0].reshape(ctx.shapes[0]), owners[1].reshape(ctx.shapes[1]), None, None, None, None, dprobs
        finally:
            resources.release()
