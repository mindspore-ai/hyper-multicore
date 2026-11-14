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
"""Local expert shards and ordinary all-to-all EP for standalone example comparisons."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, List, Sequence
import math
import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_func

_NPU_GROUPED_MATMUL = None
# The fused token permutation requires both index values and the number of indices below this limit.
_NPU_PERMUTE_INDEX_LIMIT = (1 << 24) - 1


def _get_npu_grouped_matmul() -> Any:
    """Build and cache the autograd wrapper after the Torch backend is selected."""
    global _NPU_GROUPED_MATMUL  # pylint: disable=global-statement
    if _NPU_GROUPED_MATMUL is not None:
        return _NPU_GROUPED_MATMUL

    import torch_npu  # pylint: disable=C0415

    class _NPUGroupedMatmul(torch.autograd.Function):  # pylint: disable=abstract-method
        """Autograd wrapper around ``torch_npu.npu_grouped_matmul``."""

        @staticmethod
        def forward(ctx: Any, inputs: Any, weight: Any, group_list: Any) -> Any:
            """Run one grouped matrix multiplication and retain backward inputs."""
            ctx.save_for_backward(inputs, weight, group_list)
            return torch_npu.npu_grouped_matmul(
                x=[inputs],
                weight=[weight],
                bias=[],
                group_list=group_list,
                split_item=3,
                group_type=0,
                group_list_type=0,
            )[0]

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, None]:
            """Compute grouped input and weight gradients."""
            inputs, weight, group_list = ctx.saved_tensors
            grad_output = grad_output.contiguous()
            grad_inputs = torch_npu.npu_grouped_matmul(
                x=[grad_output],
                weight=[weight.transpose(-1, -2)],
                bias=[],
                group_list=group_list,
                split_item=3,
                group_type=0,
                group_list_type=0,
            )[0]
            numel = inputs.numel()
            if (
                not torch.compiler.is_compiling()
                and isinstance(numel, int)
                and numel == 0
            ):
                grad_weight = torch.zeros_like(weight)
            else:
                grad_weight = torch_npu.npu_grouped_matmul(
                    x=[inputs.transpose(0, 1)],
                    weight=[grad_output],
                    bias=[],
                    group_list=group_list,
                    split_item=2,
                    group_type=2,
                    group_list_type=0,
                )[0]
            return grad_inputs, grad_weight, None

    _NPU_GROUPED_MATMUL = _NPUGroupedMatmul
    return _NPU_GROUPED_MATMUL


def npu_grouped_swiglu(
    inputs: Any,
    gate_up_weight: Any,
    down_weight: Any,
    tokens_per_expert: Any,
) -> Any:
    """Run packed local SwiGLU experts with two grouped GEMMs.

    Args:
        inputs: Expert-major routed tokens with shape ``[T, H]``.
        gate_up_weight: Packed projection weights with shape ``[E, 2I, H]``.
        down_weight: Down projection weights with shape ``[E, H, I]``.
        tokens_per_expert: Token count for each local expert, shape ``[E]``.

    Returns:
        Expert-major output with shape ``[T, H]``.

    Raises:
        ValueError: If called for a non-NPU tensor or incompatible shapes.
    """
    if inputs.device.type != "npu":
        raise ValueError("npu_grouped_swiglu requires inputs on an Ascend NPU")
    if gate_up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("grouped SwiGLU weights must be three-dimensional")
    if gate_up_weight.shape[0] != tokens_per_expert.numel():
        raise ValueError("tokens_per_expert must contain one count per local expert")

    import torch_npu  # pylint: disable=C0415

    grouped_matmul = _get_npu_grouped_matmul()
    group_list = torch.cumsum(tokens_per_expert.to(torch.int64), dim=0)
    gate_up = grouped_matmul.apply(
        inputs,
        gate_up_weight.transpose(1, 2),
        group_list,
    )
    intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
    return grouped_matmul.apply(
        intermediate,
        down_weight.transpose(1, 2),
        group_list,
    )


def differentiable_all_to_all_single(input_tensor: Tensor, input_splits: Sequence[int],
                                     output_splits: Sequence[int], group: Any) -> Tensor:
    """Variable-split all-to-all with autograd support for EP token dispatch/combine."""
    out_total = sum(output_splits)
    output = torch.empty(
        out_total, *input_tensor.shape[1:],
        dtype=input_tensor.dtype, device=input_tensor.device,
    )
    return dist_func.all_to_all_single(
        output, input_tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )


def exchange_splits_via_all_to_all(input_tensor: Tensor, group: Any) -> Tensor:
    """All-to-all a per-rank split vector, returning the received counts.

    Used by the token dispatchers to learn how many tokens every peer sends
    them; the exchange carries no gradient, so plain ``dist`` collectives are
    used.  Runs asynchronously with an explicit ``handle.wait()`` rather than
    ``async_op=False``: the implicit cross-stream sync is NCCL-only, and on
    HCCL the compute stream may read the output before the collective write is
    visible, producing garbage values that blow up downstream allocations.
    """
    output = torch.empty(
        [input_tensor.shape[0]],
        dtype=input_tensor.dtype, device=input_tensor.device,
    )
    handle = dist.all_to_all_single(
        output, input_tensor, group=group, async_op=True,
    )
    if handle is not None:
        handle.wait()
    return output


def _generate_permute_indices(tokens_per_expert_group, experts_per_rank: int, num_ranks: int):
    """Generate permutation indices for rank-major → expert-major reordering.

    After all-to-all, received tokens are laid out in rank-major order::

        [rank0·expert0 tokens | rank0·expert1 tokens | ... |
         rank1·expert0 tokens | rank1·expert1 tokens | ...]

    Expert computation requires expert-major order::

        [all tokens for local expert 0 | all tokens for local expert 1 | ...]

    Args:
        tokens_per_expert_group: 1-D integer tensor of shape
            ``[num_ranks * experts_per_rank]``.  Entry ``[r * E + e]`` is the
            number of tokens received from rank ``r`` for local expert ``e``.
        experts_per_rank: Number of experts owned by each rank.
        num_ranks: EP degree (total number of ranks in the EP group).

    Returns:
        Tuple of:

        - ``permuted_indices``: 1-D long tensor of length
          ``total_received_tokens``.  ``permuted_indices[i]`` is the source
          position in the rank-major buffer for destination position ``i`` in
          the expert-major buffer.
        - ``num_tokens_per_expert``: 1-D integer tensor of length
          ``experts_per_rank`` with the token count per local expert.
    """
    counts = tokens_per_expert_group
    counts_2d = counts.view(num_ranks, experts_per_rank)
    num_tokens_per_expert = counts_2d.sum(dim=0)
    total = int(num_tokens_per_expert.sum())
    if total == 0:
        return (counts.new_zeros(0, dtype=counts.dtype), num_tokens_per_expert)
    src_offsets_rm = counts.cumsum(0) - counts
    src_offsets_em = src_offsets_rm.view(num_ranks, experts_per_rank).T.contiguous().view(-1)
    counts_em = counts_2d.T.contiguous().view(-1)
    block_src_starts = src_offsets_em.repeat_interleave(counts_em)
    dst_block_starts = counts_em.cumsum(0) - counts_em
    dst_block_starts_per_token = dst_block_starts.repeat_interleave(counts_em)
    intra = torch.arange(0, total, device=counts.device) - dst_block_starts_per_token
    permuted_indices = (block_src_starts + intra).long()
    return (permuted_indices, num_tokens_per_expert)

def _permute(x, tokens_per_expert_group, ep_degree: int, num_local_experts: int):
    """Apply rank-major → expert-major permutation to routed tokens.

    Args:
        x: Received token tensor of shape
            ``[sum(tokens_per_expert_group), *feature_dims]``.
        tokens_per_expert_group: 1-D integer tensor of shape
            ``[ep_degree * num_local_experts]`` (output of the first
            all-to-all that exchanges token counts).
        ep_degree: EP group size (number of ranks).
        num_local_experts: Number of experts owned by this rank.

    Returns:
        Tuple of:

        - ``original_shape``: shape of *x* before permutation.
        - ``permuted_x``: tokens reordered to expert-major layout.
        - ``permuted_indices``: permutation indices (needed for
          :func:`_unpermute`).
        - ``num_tokens_per_expert``: token count per local expert.
        - Whether the indices came from the fused NPU permutation.
    """
    original_shape = x.shape
    if (x.device.type == "npu" and x.dtype == torch.bfloat16 and x.ndim == 2
            and 0 < x.shape[0] < _NPU_PERMUTE_INDEX_LIMIT and num_local_experts <= _NPU_PERMUTE_INDEX_LIMIT):
        import torch_npu  # pylint: disable=C0415

        counts = tokens_per_expert_group.reshape(ep_degree, num_local_experts)
        expert_ids = torch.arange(num_local_experts, dtype=torch.int32, device=x.device).repeat(ep_degree)
        row_experts = torch.repeat_interleave(
            expert_ids, tokens_per_expert_group, output_size=x.shape[0],
        ).reshape(-1, 1)
        permuted_x, permuted_indices = torch_npu.npu_moe_token_permute(x, row_experts)
        return original_shape, permuted_x, permuted_indices, counts.sum(dim=0), True
    permuted_indices, num_tokens_per_expert = _generate_permute_indices(
        tokens_per_expert_group, num_local_experts, ep_degree,
    )
    permuted_x = x[permuted_indices]
    return original_shape, permuted_x, permuted_indices, num_tokens_per_expert, False

def _unpermute(out, original_shape, permuted_indices, *, fused: bool):
    """Reverse the permutation applied by :func:`_permute`.

    Args:
        out: Expert-major output tensor of shape
            ``[sum(num_tokens_per_expert), *feature_dims]``.
        original_shape: Shape before permutation (from :func:`_permute`).
        permuted_indices: Permutation indices from :func:`_permute`.
        fused: Whether the indices came from ``npu_moe_token_permute``.

    Returns:
        Token tensor restored to the rank-major layout received after
        all-to-all, with shape ``original_shape``.
    """
    if fused:
        import torch_npu  # pylint: disable=C0415

        return torch_npu.npu_moe_token_unpermute(out, permuted_indices)
    result = out.new_zeros(*original_shape)
    result[permuted_indices] = out
    return result

@dataclass
class DispatchContext:
    """Per-forward split and permutation state retained through combine.

    The grouped-expert forward passes this context directly from dispatch to
    combine; autograd retains the tensors needed for backward independently.
    """
    input_splits: List[int]
    output_splits: List[int]
    input_shape: tuple[int, ...]
    permuted_indices: Any
    fused_permutation: bool
    probs_input_shape: Optional[tuple[int, ...]] = None

class AllToAllTokenDispatcher:
    """Token dispatch and combine via all-to-all for expert parallelism.

    Provides :meth:`dispatch` and :meth:`combine` as static methods that
    receive and return a :class:`DispatchContext` object.  This decouples
    the all-to-all token routing logic from the parallel style class so
    that it can be reused or tested independently.

    Callers (e.g. :class:`ExpertParallel`) are responsible for storing the
    context between the paired dispatch/combine calls.
    """

    @staticmethod
    def dispatch(module: Module, inputs: tuple, group: Any) -> tuple:
        """Dispatch tokens to their assigned ranks via all-to-all.

        Called explicitly by the local grouped-expert module.  Receives
        the module's forward inputs and returns transformed inputs.

        Args:
            module: The ``GroupedExperts`` module.
            inputs: Tuple ``(routed_input, num_tokens_per_expert)`` or
              ``(routed_input, num_tokens_per_expert, permuted_probs)`` where
                ``routed_input`` has shape ``[total_tokens, dim]``, and
                ``num_tokens_per_expert`` has shape ``[num_experts]``, and
                ``permuted_probs`` has shape ``[total_tokens]``.
            group: EP process group (1-D).

        Returns:
            Tuple ``(permuted_local_input, local_token_counts, ctx)`` or
            ``(permuted_local_input, local_token_counts, permuted_probs, ctx)``
            depending on whether *permuted_probs* was provided -
            the first two elements are the transformed inputs for local
            expert computation; *ctx* is a :class:`DispatchContext`
            carrying the updated state to be stored by the caller.
        """
        del module
        (routed_input, num_tokens_per_expert) = (inputs[0], inputs[1])
        permuted_probs = inputs[2] if len(inputs) > 2 else None
        ep_group = group
        ep_size = dist.get_world_size(group)
        num_local_experts = num_tokens_per_expert.shape[0] // ep_size
        counts_out = exchange_splits_via_all_to_all(num_tokens_per_expert, ep_group)
        input_splits = num_tokens_per_expert.view(ep_size, num_local_experts).sum(dim=1).tolist()
        output_splits = counts_out.view(ep_size, num_local_experts).sum(dim=1).tolist()
        dispatched = differentiable_all_to_all_single(routed_input, input_splits, output_splits, group=ep_group)
        input_shape, permuted, permuted_indices, local_counts, fused = _permute(
            dispatched, counts_out, ep_size, num_local_experts,
        )
        probs_input_shape, permuted_probs_reordered = AllToAllTokenDispatcher._exchange_and_reorder_probs(
            permuted_probs, input_splits, output_splits, permuted_indices, fused, ep_group,
        )
        ctx = DispatchContext(input_splits=input_splits, output_splits=output_splits, input_shape=input_shape,
                              permuted_indices=permuted_indices, fused_permutation=fused,
                              probs_input_shape=probs_input_shape)
        if permuted_probs is not None:
            return (permuted, local_counts, permuted_probs_reordered, ctx)
        return (permuted, local_counts, ctx)

    @staticmethod
    def _exchange_and_reorder_probs(permuted_probs, input_splits, output_splits, indices, fused, ep_group):
        """Exchange ``permuted_probs`` via a2a and reorder to expert-major.

        Returns ``(probs_input_shape, permuted_probs_reordered)``; both are
        ``None`` when *permuted_probs* is ``None`` (no-op, no second a2a).
        """
        if permuted_probs is None:
            return (None, None)
        dispatched_probs = differentiable_all_to_all_single(permuted_probs, input_splits, output_splits, group=ep_group)
        if fused:
            # The NPU mapping is source -> destination; the CPU indices are destination -> source.
            reordered = torch.empty_like(dispatched_probs).scatter(0, indices.to(torch.int64), dispatched_probs)
        else:
            reordered = dispatched_probs[indices]
        return dispatched_probs.shape, reordered

    @staticmethod
    def combine(module: Module, routed_output: object, group: Any, ctx: DispatchContext) -> object:
        """Gather expert outputs back to the originating ranks via all-to-all.

        Called explicitly after grouped-expert computation.
        Receives dispatch context from the caller (previously returned by dispatch).

        Args:
            module: The ``GroupedExperts`` module (unused, for API consistency).
            routed_output: Expert output tensor in expert-major order,
                shape ``[sum(local_counts), dim]``.
            group: EP process group (1-D).
            ctx: :class:`DispatchContext` previously returned by
                :meth:`dispatch`.

        Returns:
            Token tensor in the original token-major layout,
            shape ``[sum(input_splits), dim]``.
        """
        del module
        ep_group = group
        unpermuted = _unpermute(routed_output, ctx.input_shape, ctx.permuted_indices, fused=ctx.fused_permutation)
        combined = differentiable_all_to_all_single(unpermuted, ctx.output_splits, ctx.input_splits, group=ep_group)
        return combined


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor is stored on an Ascend NPU."""
    return tensor.device.type == "npu"


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run per-expert SwiGLU via a sequential loop (reference path).

    Args:
        w1: Shape ``[num_experts, hidden_dim, dim]``.
        w2: Shape ``[num_experts, dim, hidden_dim]``.
        w3: Shape ``[num_experts, hidden_dim, dim]``.
        x: Routed tokens in expert-major order, shape
            ``[total_routed_tokens, dim]``.
        num_tokens_per_expert: 1-D integer tensor of length ``num_experts``.
        scores: Optional 1-D tensor of shape ``[total_routed_tokens]``.
            Routing weights applied to intermediate activations (``silu(w1(x)) * w3(x)``)
            before the w2 projection. When ``None``, no weighting is applied.
            Defaults to ``None``.

    Returns:
        Expert output of shape ``[total_routed_tokens, dim]``.
    """
    # Use ``torch.cat`` instead of ``zeros_like + in-place slice assign``.
    # On some backends (notably torch_npu) in-place slice assignment onto a
    # ``requires_grad=False`` leaf tensor does not reliably upgrade it to a
    # non-leaf with a ``grad_fn`` — the forward result may end up with
    # ``grad_fn=None`` and downstream ``backward()`` fails with
    # "element 0 of tensors does not require grad and does not have a grad_fn".
    #
    # Drain ``num_tokens_per_expert`` to host **once** via ``.tolist()``
    # rather than calling ``int(n)`` per loop iteration — a single D2H
    # copy instead of ``num_local_experts`` separate ones.  Per-iter
    # ``.item()`` would stall the host between expert kernels and shrink
    # the dual-pipe overlap window.
    counts_list = num_tokens_per_expert.tolist()
    parts = []
    offset = 0
    for e, n in enumerate(counts_list):
        if n == 0:
            continue
        x_e = x[offset:offset + n]
        h = F.silu(x_e @ w1[e].T) * (x_e @ w3[e].T)
        if scores is not None:
            h = h * scores[offset:offset + n].unsqueeze(-1)
        parts.append(h @ w2[e].T)
        offset += n
    if not parts:
        # No routed tokens: return a grad-connected zero (not ``zeros_like``).
        return x * 0.0
    return torch.cat(parts, dim=0)


def _run_experts_grouped_mm_npu(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused grouped matmul path for Ascend NPU using ``torch_npu.npu_grouped_matmul``.

    Requires ``torch_npu`` to be installed.

    Args:
        w1: Shape ``[num_experts, hidden_dim, dim]``.
        w2: Shape ``[num_experts, dim, hidden_dim]``.
        w3: Shape ``[num_experts, hidden_dim, dim]``.
        x: Shape ``[total_routed_tokens, dim]``.
        num_tokens_per_expert: 1-D integer tensor of length ``num_experts``.
        scores: Optional 1-D tensor of shape ``[total_routed_tokens]``.
            Routing weights applied after the bias-free w2 projection.
            When ``None``, no weighting is applied. Defaults to ``None``.

    Returns:
        Expert output of shape ``[total_routed_tokens, dim]``.
    """
    if x.shape[0] == 0:
        parameter_zero = (w1.sum() + w2.sum() + w3.sum()).to(x.dtype) * 0.0
        return x + parameter_zero

    gate_up_weight = torch.cat((w1, w3), dim=1)
    output = npu_grouped_swiglu(
        x,
        gate_up_weight,
        w2,
        num_tokens_per_expert,
    )
    if scores is not None:
        output = output * scores.to(output.dtype).unsqueeze(-1)
    return output


class GroupedExperts(nn.Module):
    """Grouped experts with plain local parameters and optional all-to-all EP.

    Construct global [expert, output, input] weights, then call shard_experts
    before use. The borrowed process group owns rank-ordered expert slices;
    this module never creates or destroys that group. NPU computation uses
    grouped GEMM; CPU computation uses the per-expert equations.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = False,
    ) -> None:
        """Initialize GroupedExperts with stacked expert weight matrices.

        Args:
            dim: Input embedding dimension.
            hidden_dim: Intermediate hidden dimension.
            num_experts: Number of experts.
            use_grouped_mm: Whether to use grouped matrix multiplication.
        """
        super().__init__()
        # Weight layout: [num_experts, out_dim, in_dim] so that the standard
        # linear operation is x @ w[e].T.
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.num_experts = num_experts
        self.use_grouped_mm = use_grouped_mm
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming-uniform initialisation for all expert weight tensors."""
        for weight in (self.w1, self.w2, self.w3):
            nn.init.kaiming_uniform_(weight.view(weight.shape[0], -1), a=math.sqrt(5))

    def _local_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all experts on their assigned tokens.

        Args:
            x: Routed tokens in expert-major order,
                shape ``[total_routed_tokens, dim]``.
            num_tokens_per_expert: 1-D integer tensor of length
                ``num_local_experts`` with the token count per expert.
            scores: Optional 1-D tensor of shape ``[total_routed_tokens]``.
                Routing weights for each token's expert output.
                When ``None``, no weighting is applied. Defaults to ``None``.

        Returns:
            Expert output of shape ``[total_routed_tokens, dim]``.
        """
        # Parameters already contain the local expert shard.
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3

        if not self.use_grouped_mm:
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert, scores)
        if _is_npu_tensor(x):
            return _run_experts_grouped_mm_npu(w1, w2, w3, x, num_tokens_per_expert, scores)
        return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert, scores)

    def shard_experts(self, group: Any) -> None:
        """Keep the rank-ordered local expert shard and borrow its EP group."""
        self.ep_group = group
        world = dist.get_world_size(group) if dist.is_initialized() else 1
        rank = dist.get_rank(group) if dist.is_initialized() else 0
        if self.num_experts % world:
            raise ValueError("Expert count must be divisible by EP size")
        local = self.num_experts // world
        for name in ("w1", "w2", "w3"):
            parameter = getattr(self, name)
            setattr(self, name, nn.Parameter(torch.detach(parameter).narrow(0, rank * local, local).clone(),
                                            requires_grad=parameter.requires_grad))

    def forward(self, x: Tensor, num_tokens_per_expert: Tensor, scores: Optional[Tensor] = None) -> Tensor:
        """Dispatch, evaluate local experts and return source-ordered routed rows."""
        group = self.ep_group
        if not dist.is_initialized() or dist.get_world_size(group) == 1:
            return self._local_forward(x, num_tokens_per_expert, scores)
        inputs = (x, num_tokens_per_expert) if scores is None else (x, num_tokens_per_expert, scores)
        dispatched = AllToAllTokenDispatcher.dispatch(self, inputs, group)
        local_output = self._local_forward(*dispatched[:-1])
        return AllToAllTokenDispatcher.combine(self, local_output, group, dispatched[-1])
