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
"""Expert placement shared by ordinary EP and the MegaMoE executor."""
from __future__ import annotations

from dataclasses import dataclass

import torch



@dataclass(frozen=True)
class ExpertPlacement:
    """Keep the virtual routing and weight-copy ownership of one invocation.

    Attributes:
        virtual_ids: Selected virtual expert IDs, retaining original token and Top-K order.
        counts_by_source: Counts from every source for all home and borrowed expert slots.
        copies: Device plan whose rows identify the global experts borrowed by each rank.
        copies_by_rank: The same copy ownership on the host for communication split construction.
    """

    virtual_ids: torch.Tensor
    counts_by_source: torch.Tensor
    copies: torch.Tensor
    copies_by_rank: tuple[tuple[int, ...], ...] | None
    borrowed_capacity: int
    receive_capacity: int

    def __post_init__(self) -> None:
        """Validate the shared layout without reading any device tensor values."""
        ranks = self.counts_by_source.shape[0] if self.counts_by_source.ndim == 2 else 0
        if (not isinstance(self.borrowed_capacity, int) or isinstance(self.borrowed_capacity, bool)
                or self.borrowed_capacity < 0 or not isinstance(self.receive_capacity, int)
                or isinstance(self.receive_capacity, bool) or self.receive_capacity < 0):
            raise ValueError("MoonEP placement capacities must be nonnegative integers")
        if (ranks < 1 or self.counts_by_source.ndim != 2 or self.counts_by_source.shape[0] != ranks
                or self.counts_by_source.shape[1] % ranks):
            raise ValueError("MoonEP placement counts must describe the ordered EP group")
        local = self.counts_by_source.shape[1] // ranks - self.borrowed_capacity
        if local < 1 or self.borrowed_capacity > local:
            raise ValueError("MoonEP borrowed capacity exceeds its home expert bank")
        if (tuple(self.copies.shape) != (ranks, self.borrowed_capacity)
                or self.copies_by_rank is not None and (len(self.copies_by_rank) != ranks
                    or any(len(row) != self.borrowed_capacity for row in self.copies_by_rank))):
            raise ValueError("MoonEP copy tables must match the logical borrowed capacity")
        if (self.virtual_ids.dtype not in (torch.int32, torch.int64)
                or self.counts_by_source.dtype not in (torch.int32, torch.int64)
                or self.copies.dtype != torch.int32):
            raise TypeError("MoonEP placement requires integer routes/counts and INT32 copies")
        if self.virtual_ids.device != self.counts_by_source.device or self.copies.device != self.virtual_ids.device:
            raise ValueError("MoonEP placement tensors must share the source device")


def has_transfer_work(token_capacity: int, top_k: int, local_experts: int) -> bool:
    """Avoid planning when there is at most one 128-row tile per local expert."""
    return token_capacity * top_k > 128 * local_experts


def placement_is_profitable(
    maximum_received: int,
    target_load: int,
    hidden_size: int,
    intermediate_size: int,
    cube_cores: int,
    weight_bytes: int,
    *,
    projection_flops: int = 6,
    ep_size: int = 1,
    local_experts: int = 1,
    gradient_bytes: int | None = None,
) -> bool:
    """Estimate critical compute saved against dtype-aware bank and peer traffic.

    Weight publication reads/writes the home shard. Gradient publication
    reads/writes the wire shard, followed by the returned parameter-dtype store:
    three weight-byte passes plus two gradient-byte passes. With BF16 gradients
    this retains the original five-pass staging estimate; FP32 uses seven.

    Conservatively budget one full expert per peer on the critical owner rather
    than treating a split expert as proportional to its token chunk. The A3
    effective rates are estimates, not peak hardware specifications: 4 TFLOP/s
    per cube, 400 GB/s staging and 200 GB/s one-way peer transfer. The peer rate
    reflects the separately profiled prefetch/return kernels. The fixed placement
    budget remains 0.8 ms. No observation, allocation or collective occurs here.

    Args:
        maximum_received: Maximum routed expert rows before placement.
        target_load: Average rank rows, or a conservative capacity bound.
        hidden_size: Model hidden width.
        intermediate_size: Single gate/up projection width.
        cube_cores: Common device cube count for the ordered EP group.
        weight_bytes: Home shard size in the parameter transfer dtype.
        projection_flops: Saved forward/input-gradient multiply-add factor.
        ep_size: Number of ranks sharing the expert bank.
        local_experts: Number of home experts within weight_bytes.
        gradient_bytes: Home shard bytes in the gradient wire dtype, zero for
            inference. None retains the historical BF16-sized gradient estimate.
    """
    if cube_cores <= 0 or ep_size <= 0 or local_experts <= 0 or weight_bytes < 0:
        raise ValueError("MoonEP cost geometry requires positive cores/ranks/experts and nonnegative bytes")
    if gradient_bytes is None:
        gradient_bytes = weight_bytes
    if gradient_bytes < 0:
        raise ValueError("MoonEP gradient traffic must be nonnegative")
    excess = max(0, maximum_received - target_load)
    saved_compute = projection_flops * excess * hidden_size * intermediate_size / (cube_cores * 4e12)
    staging = (3 * weight_bytes + 2 * gradient_bytes) / 4e11
    peer = (ep_size - 1) * (weight_bytes + gradient_bytes) / (local_experts * 2e11)
    return saved_compute > 8e-4 + staging + peer


@torch.no_grad()
def build_placement(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    counts_by_source: torch.Tensor,
    capacity: int,
    rank: int,
    *, resources: object = None, source_rows: tuple[int, ...] | None = None, source_inverse=None,
) -> ExpertPlacement | None:
    """Borrow experts and render their virtual IDs without changing the router.

    Each virtual rank owns its home expert slots followed by equally many optional
    borrowed slots. Original expert parameters and optimizer ownership stay unchanged.
    The caller supplies its existing count table and load target; this function
    performs no additional count collective or expert computation.

    Args:
        hidden: Local token rows on the active NPU.
        topk_ids: Original selected global expert IDs, shaped [tokens, Top-K].
        counts_by_source: Current [EP, global experts] source-count table.
        capacity: Positive target routed rows per rank for the allocator.
        rank: Rank within this ordered EP group.

    Returns:
        A per-invocation placement, or None when no expert needs to be borrowed.
    """
    # The primitive is optional and is loaded only when placement is actually used.
    import torch_npu  # pylint: disable=C0415
    from .allocator import allocate_experts
    from .render import render_route

    if counts_by_source.ndim != 2:
        raise ValueError("MoonEP counts must be a source-by-expert matrix")
    ranks, experts = counts_by_source.shape
    if ranks < 1 or experts < 1 or experts % ranks or not 0 <= rank < ranks:
        raise ValueError("MoonEP counts must describe a valid ordered EP topology")
    if counts_by_source.dtype not in (torch.int32, torch.int64):
        raise TypeError("MoonEP counts must use INT32 or INT64")
    gathered = counts_by_source.contiguous()
    if gathered.dtype == torch.int64:
        # The legacy route wire uses INT64; the planner's arithmetic is INT32.
        total = int(gathered.sum(dtype=torch.int64).item())
        if not 0 <= total <= 2**31 - 1:
            raise ValueError("MoonEP routed row count exceeds the native INT32 planner limit")
        gathered = gathered.to(torch.int32)
    cumulative, _, _, copies = allocate_experts(gathered, max(1, capacity))
    # Cumulative expert extents determine actual destination loads, including
    # cases where the allocation could not attain the average target.
    if tuple(cumulative.shape) != (experts, ranks):
        raise ValueError("MoonEP allocator returned an invalid cumulative extent")
    summary = torch.cat((copies.flatten().to(torch.int64), cumulative.sum(0, dtype=torch.int64))).cpu().tolist()
    local = experts // ranks
    copies_by_rank = tuple(tuple(summary[r * local:(r + 1) * local]) for r in range(ranks))
    prefixes = summary[ranks * local:]
    loads = [value - (prefixes[i - 1] if i else 0) for i, value in enumerate(prefixes)]
    if not any(expert >= 0 for row in copies_by_rank for expert in row):
        return None
    inverse = topk_ids.reshape(-1)
    if source_inverse is not None:
        inverse = source_inverse
    elif topk_ids.numel():
        _, inverse, _, _ = torch_npu.npu_moe_init_routing_v2(
            hidden, topk_ids.int(), active_num=topk_ids.numel(), expert_num=experts,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0, experts], row_idx_type=0,
        )
    virtual_ids, counts = render_route(
        topk_ids, inverse, gathered, cumulative, copies, rank, experts // ranks,
    )
    return ExpertPlacement(virtual_ids, counts, copies, copies_by_rank, local, max(loads))
