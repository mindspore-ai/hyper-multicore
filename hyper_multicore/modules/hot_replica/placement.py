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
from ..mega_moe.route import _permute_topk_input, _upload_route_tensor
from ._planner.capacity import ExpertReplicaConfig
from ._planner.receive import ReplicaCost, SourceQuota, select_replica_plan
from ._planner.plan import ExpertExecutionPlan
from .render import render_plan


def replica_slot_budget(home: int, requested: int | None, dispatch_mode: str) -> int:
    """Clamp the requested guest budget to the current BF16 transport."""
    budget = home if requested is None else min(home, requested)
    return min(budget, 6) if dispatch_mode == "pull" else budget


def remap_expert_ids(hidden, ids, sorted_ids, inverse):
    """Map stable logical occurrences to their physical expert IDs."""
    if (inverse is None and ids.device.type == "npu" and
            0 < ids.numel() <= 32768 and ids.shape[1] <= 8):
        _, inverse = _permute_topk_input(hidden.new_zeros((ids.shape[0], 1)), ids.int())
    if inverse is not None:
        return sorted_ids.index_select(0, inverse.flatten()).view_as(ids)
    order = torch.argsort(ids.flatten().float(), stable=True)
    remapped = torch.empty_like(sorted_ids)
    remapped.scatter_(0, order, sorted_ids)
    return remapped.view_as(ids)


@torch.no_grad()
def build_placement(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    counts_by_source: torch.Tensor,
    capacity: int | None,
    rank: int,
    *,
    source_inverse: torch.Tensor | None = None,
    replica_slots_per_rank: int | None = None,
    prices: ReplicaCost = ReplicaCost(),
    source_quota: SourceQuota = SourceQuota.RANK_ORDER,
    receive_capacity_percent=None, nodes=(), cross_node_cost=None, resident=None,
) -> ExpertPlacement | None:
    """Borrow experts and render their virtual IDs without changing the router.

    Each virtual rank owns its home expert slots followed by optional borrowed slots.
    Original expert parameters and optimizer ownership stay unchanged.
    The caller supplies its existing count table and load target; this function
    performs no additional count collective or expert computation.

    Args:
        hidden: Local token rows on the active NPU.
        topk_ids: Original selected global expert IDs, shaped [tokens, Top-K].
        counts_by_source: Current [EP, global experts] source-count table.
        capacity: Positive target routed rows per rank for the allocator.
        rank: Rank within this ordered EP group.
        replica_slots_per_rank: Optional cap on guests, defaulting to the number of home experts.
        prices: Shared cost model for this execution path.
        source_quota: Order used to assign the selected rows to source ranks.

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
    wire = counts_by_source.cpu().numpy()
    home = experts // ranks
    budget = home if replica_slots_per_rank is None else min(home, replica_slots_per_rank)
    config = ExpertReplicaConfig(experts, ranks, budget)
    if wire.min() < 0 or wire.max() > 2**31 - 1 or wire.sum(dtype="int64") > 2**31 - 1:
        raise ValueError("HotReplica counts must be nonnegative and within the INT32 planner limit")
    plan = select_replica_plan(
        wire, config, soft_target=capacity, receive_capacity_percent=receive_capacity_percent,
        source_quota=source_quota, prices=prices, nodes=nodes, cross_node_cost=cross_node_cost, resident=resident,
    )
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
    used = max(plan.replica_slots_used)
    borrowed = min(plan.config.replica_slots_per_rank, 1 << (used - 1).bit_length())
    device = topk_ids.device
    ordered, table, owners = render_plan(plan, rank, borrowed)
    if ordered.size != topk_ids.numel():
        raise ValueError("HotReplica counts must describe the local Top-K routes")
    sorted_ids = _upload_route_tensor(torch.from_numpy(ordered), device)
    ids = remap_expert_ids(hidden, topk_ids, sorted_ids, source_inverse)
    copies = _upload_route_tensor(torch.tensor(owners, dtype=torch.int32), device)
    table = _upload_route_tensor(torch.from_numpy(table), device)
    loads = plan.destination_loads
    return ExpertPlacement(ids, table, copies, borrowed, max(loads), loads[rank],
                           plan.maximum_transfer_degree, plan.replica_slots_used[rank], copies_by_rank=owners,
                           owner_rows=plan.destination_counts[rank][:plan.config.home_experts])
