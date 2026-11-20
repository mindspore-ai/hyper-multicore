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
"""Expert placement for the existing MegaMoE BF16 executor."""

from __future__ import annotations

import torch

from ..base_planner import ExpertPlacement
from ..mega_moe.route import _upload_route_tensor
from ._planner.planner import build_capacity_plan
from ._planner.plan import ExpertExecutionPlan
from .render import render_plan

# Effective A3 planner estimates from the original prefetch/return cost model.
# Profiled compute saved by placement at H=5120/I=1792 exceeded the old
# 4 TFLOP/s estimate: that estimate overpredicted saved time (6.6 vs 4.7 ms).
# Use 6 TFLOP/s conservatively rather than assuming a slower ordinary backend.
_A3_FLOPS_PER_CUBE = 6e12
_A3_STAGING_BYTES_PER_SECOND = 4e11
_A3_PEER_BYTES_PER_SECOND = 2e11
_A3_PLACEMENT_SECONDS = 8e-4
_SHORT_ROUTE_PAYLOAD_BYTES = 512 * 1024**2


def placement_receive_limit(
    target_rows: int, ranks: int, local_experts: int, top_k: int, hidden_size: int, element_size: int
) -> int:
    """Rows a rank may receive without borrowing, using PR19's capacity policy.

    Dynamic callers supply the actual average routed rows, rounded up. For a
    short route, bounded fan-in may fit in the 512 MiB receive payload budget.
    """
    if target_rows < 0 or min(ranks, local_experts, top_k, hidden_size, element_size) <= 0:
        raise ValueError("Invalid HotReplica receive geometry")
    tile = 128
    # ceil(1.3 * target_rows), without floating-point rounding at tile boundaries.
    limit = min((13 * target_rows + 9) // 10, ranks * target_rows)
    if 0 < target_rows < 16 * tile:
        fan_in = (ranks * target_rows * min(local_experts, top_k) + top_k - 1) // top_k
        worst = (fan_in + tile - 1) // tile * tile
        if worst * hidden_size * element_size <= _SHORT_ROUTE_PAYLOAD_BYTES:
            limit = max(limit, min(worst, ranks * target_rows))
    return (limit + tile - 1) // tile * tile


def _bounded_transfer_seconds(
    ranks: int, local: int, weight_bytes: int, gradient_bytes: int,
) -> float:
    """Estimate the leased bank traffic for a route with a configured source bound."""
    staging = (3 * weight_bytes + 2 * gradient_bytes) / _A3_STAGING_BYTES_PER_SECOND
    peer = (ranks - 1) * (weight_bytes + gradient_bytes) / (local * _A3_PEER_BYTES_PER_SECOND)
    return _A3_PLACEMENT_SECONDS + staging + peer


def bounded_move_pays(
    maximum: int, after: int, hidden: int, intermediate: int, cube_cores: int,
    ep_size: int, local_experts: int, element_size: int, *, training: bool, replay: bool,
    apply_probs_before_fc2: bool,
) -> bool:
    """Compare placement work against the busiest rank's saved projection work."""
    weights = 3 * local_experts * hidden * intermediate
    gradient_bytes = weights * 4 if training else 0
    transfer = _bounded_transfer_seconds(ep_size, local_experts, weights * element_size, gradient_bytes)
    if replay:
        transfer += 2 * weights * element_size / _A3_STAGING_BYTES_PER_SECOND
    flops = 18 if replay and not apply_probs_before_fc2 else 12 if training else 6
    compute = flops * max(0, maximum - after) * hidden * intermediate
    return compute > transfer * cube_cores * 4e12


def _transfer_seconds(weight_bytes, gradient_bytes, ep_size, local_experts, replay_bytes, transfer_experts):
    """Time to stage the shard and fetch/return one expert per peer."""
    copies = ep_size - 1 if transfer_experts is None else transfer_experts
    # BF16 partial dW is first materialized as FP32 home gradients, then read
    # again by the owner kernel. Borrowed dW is separately copied into the wire
    # slots. These are real device-memory passes, in addition to peer traffic.
    gradient_staging = (
        weight_bytes + copies * (weight_bytes + gradient_bytes) / local_experts
        if gradient_bytes else 0
    )
    staging = (3 * weight_bytes + 2 * gradient_bytes + gradient_staging) / _A3_STAGING_BYTES_PER_SECOND
    peer = copies * (weight_bytes + replay_bytes + gradient_bytes) / (
        local_experts * _A3_PEER_BYTES_PER_SECOND
    )
    return _A3_PLACEMENT_SECONDS + staging + peer


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
    backward_prefetch_weight_bytes: int = 0,
    transfer_experts: int | None = None,
) -> bool:
    """Estimate critical compute saved against dtype-aware bank and peer traffic.

    Include weight publication, FP32 home-gradient materialization, borrowed
    gradient publication, owner accumulation and the returned parameter store.
    Borrowed publication scales with actual copy records, independently of the
    number of tokens assigned to those experts.

    Use actual incoming/outgoing copy counts after allocation; the preliminary
    gate uses one expert per peer. Charge backward weight fetch as well as
    publication, because autograd prepares the saved owner weights again. The A3
    effective rates are estimates, not peak hardware specifications: 6 TFLOP/s
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
        backward_prefetch_weight_bytes: Home weight bytes republished and fetched before backward.
        transfer_experts: Maximum actual incoming or outgoing expert copies per rank.
            None uses one expert per peer for the preliminary estimate.
    """
    if cube_cores <= 0 or ep_size <= 0 or local_experts <= 0 or weight_bytes < 0:
        raise ValueError("HotReplica cost geometry requires positive cores/ranks/experts and nonnegative bytes")
    if gradient_bytes is None:
        gradient_bytes = weight_bytes
    if transfer_experts is not None and (
        not isinstance(transfer_experts, int) or isinstance(transfer_experts, bool) or transfer_experts < 0
    ):
        raise ValueError("HotReplica transfer expert count must be a nonnegative integer")
    if gradient_bytes < 0:
        raise ValueError("HotReplica gradient traffic must be nonnegative")
    excess = max(0, maximum_received - target_load)
    saved_compute = projection_flops * excess * hidden_size * intermediate_size / (cube_cores * _A3_FLOPS_PER_CUBE)
    restore = 2 * backward_prefetch_weight_bytes / _A3_STAGING_BYTES_PER_SECOND
    return saved_compute > restore + _transfer_seconds(
        weight_bytes, gradient_bytes, ep_size, local_experts,
        backward_prefetch_weight_bytes, transfer_experts,
    )


def _token_order(hidden, ids, experts, supplied):
    """Reuse the source permutation or compute expert-major routing order."""
    if supplied is not None:
        return supplied.flatten()
    if not ids.numel():
        return ids.flatten()
    import torch_npu  # pylint: disable=import-outside-toplevel

    _, inverse, _, _ = torch_npu.npu_moe_init_routing_v2(
        hidden,
        ids.int(),
        active_num=ids.numel(),
        expert_num=experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, experts],
        row_idx_type=0,
    )
    return inverse.flatten()


@torch.no_grad()
def build_placement(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    counts_by_source: torch.Tensor,
    capacity: int,
    rank: int,
    *,
    source_inverse: torch.Tensor | None = None,
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
    if counts_by_source.ndim != 2:
        raise ValueError("HotReplica counts must be a source-by-expert matrix")
    ranks, experts = counts_by_source.shape
    if ranks < 1 or experts < 1 or experts % ranks or not 0 <= rank < ranks:
        raise ValueError("HotReplica counts must describe a valid ordered EP topology")
    if counts_by_source.dtype not in (torch.int32, torch.int64):
        raise TypeError("HotReplica counts must use INT32 or INT64")
    plan = build_capacity_plan(counts_by_source.cpu().numpy(), capacity)
    if plan is None:
        return None
    return render_placement(hidden, topk_ids, plan, rank, source_inverse=source_inverse)


def render_placement(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    plan: ExpertExecutionPlan,
    rank: int,
    *,
    source_inverse: torch.Tensor | None = None,
) -> ExpertPlacement:
    """Render the admitted source quotas using the existing device permutation."""
    experts = plan.config.num_experts
    used = max(plan.replica_slots_used)
    borrowed = min(plan.config.replica_slots_per_rank, 1 << (used - 1).bit_length())
    device = topk_ids.device
    ordered, table, owners = render_plan(plan, rank, borrowed)
    if ordered.size != topk_ids.numel():
        raise ValueError("HotReplica counts must describe the local Top-K routes")
    sorted_ids = _upload_route_tensor(torch.from_numpy(ordered), device)
    ids = sorted_ids.index_select(
        0, _token_order(hidden, topk_ids, experts, source_inverse)
    ).reshape_as(topk_ids)
    copies = _upload_route_tensor(torch.tensor(owners, dtype=torch.int32), device)
    table = _upload_route_tensor(torch.from_numpy(table), device)
    loads = plan.destination_loads
    return ExpertPlacement(ids, table, copies, borrowed, max(loads), loads[rank],
                           plan.maximum_transfer_degree, plan.replica_slots_used[rank], copies_by_rank=owners,
                           owner_rows=plan.destination_counts[rank][:plan.config.home_experts])
