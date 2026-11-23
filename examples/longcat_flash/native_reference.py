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
"""Independent native NPU EP LongCat reference using fused one-shot permute/unpermute."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu


class _AllToAllSingle(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Differentiable variable-length all_to_all_single across an explicit EP group."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        send_sizes: list[int],
        receive_sizes: list[int],
        group: Any,
    ) -> torch.Tensor:
        """Dispatch rows along dimension 0 using HCCL all_to_all_single."""
        ctx.send_sizes = send_sizes
        ctx.receive_sizes = receive_sizes
        ctx.group = group
        outputs = inputs.new_empty((sum(receive_sizes), *inputs.shape[1:]))
        dist.all_to_all_single(
            outputs,
            inputs.contiguous(),
            output_split_sizes=receive_sizes,
            input_split_sizes=send_sizes,
            group=group,
        )
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Reverse split sizes to scatter gradients back to source ranks."""
        grad_inputs = grad_outputs.new_empty((sum(ctx.send_sizes), *grad_outputs.shape[1:]))
        dist.all_to_all_single(
            grad_inputs,
            grad_outputs.contiguous(),
            output_split_sizes=ctx.send_sizes,
            input_split_sizes=ctx.receive_sizes,
            group=ctx.group,
        )
        return grad_inputs, None, None, None


class _GroupedMatmul(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Compute native GMM forward and both gradients in expert-major order."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weights: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """Run the NPU grouped matrix multiplication."""
        ctx.save_for_backward(inputs, weights, groups)
        return torch_npu.npu_grouped_matmul(
            x=[inputs],
            weight=[weights],
            bias=[],
            group_list=groups,
            split_item=3,
            group_type=0,
            group_list_type=0,
        )[0]

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Use native grouped GEMMs for input and expert-weight gradients."""
        inputs, weights, groups = ctx.saved_tensors
        gradient = gradient.contiguous()
        dx = torch_npu.npu_grouped_matmul(
            x=[gradient],
            weight=[weights.transpose(-1, -2)],
            bias=[],
            group_list=groups,
            split_item=3,
            group_type=0,
            group_list_type=0,
        )[0]
        dw = (
            torch.zeros_like(weights)
            if inputs.numel() == 0
            else torch_npu.npu_grouped_matmul(
                x=[inputs.transpose(-1, -2)],
                weight=[gradient],
                bias=[],
                group_list=groups,
                split_item=2,
                group_type=2,
                group_list_type=0,
            )[0]
        )
        return dx, dw, None


def _permute(hidden: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep an autograd-connected empty input when a source has no rows."""
    if not hidden.shape[0]:
        return hidden, ids.new_empty((0,), dtype=torch.int32)
    permuted, mapping = torch_npu.npu_moe_token_permute(hidden.contiguous(), ids.to(torch.int32).contiguous())
    return permuted, mapping.reshape(-1).to(torch.int32).contiguous()


def _unpermute(
    hidden: torch.Tensor,
    mapping: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restore token order with optional fused FP32 probability weighting."""
    if not mapping.numel():
        return hidden[:0]
    if probs is None:
        return torch_npu.npu_moe_token_unpermute(hidden, mapping)
    return torch_npu.npu_moe_token_unpermute(hidden, mapping, probs=probs.float().contiguous())


def _exchange_counts(
    counts: torch.Tensor,
    group: Any,
    ep_size: int,
) -> tuple[torch.Tensor, list[int], list[int], int]:
    """Exchange per-expert real counts and compute all-to-all split sizes and global real total."""
    received = torch.empty_like(counts)
    if ep_size > 1:
        dist.all_to_all_single(received, counts.contiguous(), group=group)
    else:
        received.copy_(counts)
    local = counts.numel() // ep_size
    send_by_rank = counts.view(ep_size, local).sum(1)
    recv_by_rank = received.view(ep_size, local).sum(1)
    total_real_tensor = send_by_rank.sum().unsqueeze(0)
    if ep_size > 1:
        dist.all_reduce(total_real_tensor, op=dist.ReduceOp.SUM, group=group)
    segments = torch.stack((send_by_rank, recv_by_rank)).tolist()
    return received, segments[0], segments[1], int(total_real_tensor.item())


def _expert_gemms(
    hidden: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor],
    counts: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    """Apply GMM1, SwiGLU (fused or hf) and GMM2 in EOI layout."""
    gate, down = weights
    if not hidden.shape[0]:
        return hidden + (gate.reshape(-1)[:1].sum() + down.reshape(-1)[:1].sum()) * 0
    groups = counts.to(torch.int64).cumsum(0)
    projected = _GroupedMatmul.apply(hidden, gate.transpose(1, 2), groups)
    if precision == "fused":
        intermediate = torch_npu.npu_swiglu(projected)
    elif precision == "hf":
        first, second = projected.chunk(2, -1)
        intermediate = F.silu(first) * second
    else:
        raise ValueError(f"precision must be fused or hf, got {precision!r}")
    return _GroupedMatmul.apply(intermediate, down.transpose(1, 2), groups)


class NativeLongcatExperts:
    """Native NPU LongCat reference matching one-shot permute + fused unpermute semantics."""

    def __init__(
        self,
        num_experts: int,
        zero_experts: int,
        ep_size: int,
        group: Any,
        precision: str = "fused",
        *,
        gate_up_padded: bool = True,
    ) -> None:
        """Bind global real/zero expert counts, EP topology and SwiGLU rounding mode."""
        self.num_experts = num_experts
        self.zero_experts = zero_experts
        self.total_experts = num_experts + zero_experts
        self.ep_size = ep_size
        self.group = group
        self.precision = precision
        self.gate_up_padded = gate_up_padded

    def __call__(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        router_weights: torch.Tensor,
        *,
        expert_weights: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run one-shot permute, real-expert EP GMM/SwiGLU, stitch identity suffix, fused unpermute."""
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        ids_i32 = expert_ids.to(torch.int32).contiguous()
        all_permuted, mapping = _permute(hidden_flat, ids_i32)

        all_counts = torch.bincount(ids_i32.reshape(-1).long(), minlength=self.total_experts)
        real_counts = all_counts[: self.num_experts]
        received_counts, send_sizes, receive_sizes, total_real_across_ep = _exchange_counts(
            real_counts, self.group, self.ep_size
        )

        if total_real_across_ep == 0:
            return _unpermute(all_permuted, mapping, router_weights).reshape_as(hidden)

        local = self.num_experts // self.ep_size
        gate, down = expert_weights
        real_gate = gate[:local] if self.gate_up_padded and gate.shape[0] > local else gate

        local_real_rows = sum(send_sizes)
        total_slots = all_permuted.shape[0]
        if local_real_rows == total_slots:
            real_x = all_permuted
            zero_x = None
        else:
            real_x, zero_x = torch.split(all_permuted, [local_real_rows, total_slots - local_real_rows], dim=0)

        received = (
            _AllToAllSingle.apply(real_x, send_sizes, receive_sizes, self.group)
            if self.ep_size > 1
            else real_x
        )
        local_ids = torch.arange(local, device=hidden.device).repeat(self.ep_size)
        local_ids = local_ids.repeat_interleave(received_counts).reshape(-1, 1)
        expert_input, owner_mapping = _permute(received, local_ids)
        local_counts = received_counts.view(self.ep_size, local).sum(0)
        expert_output = _expert_gemms(expert_input, (real_gate, down), local_counts, self.precision)
        owner_output = _unpermute(expert_output, owner_mapping)
        real_y = (
            _AllToAllSingle.apply(owner_output, receive_sizes, send_sizes, self.group)
            if self.ep_size > 1
            else owner_output
        )

        all_y = real_y if zero_x is None else torch.cat((real_y, zero_x), dim=0)
        return _unpermute(all_y, mapping, router_weights).reshape_as(hidden)


NativeExperts = NativeLongcatExperts
