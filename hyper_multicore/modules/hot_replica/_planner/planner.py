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
"""Bounded replica planner without calibration or device dependencies."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ._balance import _capacity_copies
from ._dispatch import _home_layout, _materialize, aggregate_source_nodes, source_local_node_rows
from ._placement import _consolidate_copies, _locality_copies
from ._retained import _rebalance_retained_copies
from ._selection import _admitted_uncalibrated_copies, _optional_peak_is_useful
from .admission import reduction_basis_points
from .capacity import ExpertReplicaConfig, _integer
from .plan import ExpertExecutionPlan
from .topology import _validate_rank_nodes, charge_replica_endpoints, limiting_stage_work


def _counts_matrix(counts: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Freeze and validate integer source histograms."""
    rows = tuple(tuple(row) for row in counts)
    if not rows or not rows[0]:
        raise ValueError("counts_by_source must be a nonempty rectangular matrix")
    for row in rows:
        if len(row) != len(rows[0]):
            raise ValueError("counts_by_source must be rectangular")
        for count in row:
            if type(count) not in (int,) or count < 0:
                _integer(count, "expert count", 0)
    return rows


def _prefer_locality(
    copies: list[dict[int, int]],
    config: ExpertReplicaConfig,
    nodes: tuple[int, ...],
    loads: list[int],
    limit: int,
    receive_capacity: int | None,
    admission: int,
    saved_rows: int,
    row_credit: int,
    cross_node_cost: float | None,
    counts: tuple[tuple[int, ...], ...],
) -> list[dict[int, int]]:
    """Compare at most one whole-guest locality proposal under the original limits."""
    has_guests = any(
        expert // config.home_experts != rank and rows
        for rank, row in enumerate(copies)
        for expert, rows in row.items()
    )
    if not has_guests or not nodes or not 1 < len(set(nodes)) < config.ep_size:
        return copies
    alternative = _locality_copies(copies, config, nodes)
    if alternative is copies:
        return copies
    peak = max(sum(row.values()) for row in copies)
    ceiling = min(limit, max(receive_capacity or 0, peak))
    alternate_peak = max(sum(row.values()) for row in alternative)
    if alternate_peak > ceiling:
        return copies
    if cross_node_cost is not None:
        before = _critical_quota_work(copies, config, nodes, row_credit, cross_node_cost)
        after = _critical_quota_work(alternative, config, nodes, row_credit, cross_node_cost)
        if after > before:
            return copies
        if after == before:
            # Equal replica work can still put very different token loads on
            # the fabric. Compare the exact node-local quota rule in that tie.
            sources = aggregate_source_nodes(counts, nodes)
            if source_local_node_rows(alternative, sources) >= source_local_node_rows(copies, sources):
                return copies
    return alternative if _optional_peak_is_useful(alternate_peak, loads, ceiling, admission, saved_rows) else copies


def _validate_options(options: tuple[tuple[str, int | None], ...], reduction: float) -> int:
    """Validate integer controls without accepting booleans as row counts."""
    for name, value in options:
        if value is None and name in ("capacity_limit", "receive_capacity", "target_load"):
            continue
        _integer(value, name, 0)
    return reduction_basis_points(reduction)


def _critical_quota_work(copies, config, nodes, row_credit, cross_node_cost):
    """Charge actual incoming and owner-return matrices against saved critical work."""
    if not row_credit:
        return max(sum(row.values()) for row in copies)
    home = config.home_experts
    edges = ((expert // home, rank) for rank, row in enumerate(copies)
             for expert, rows in row.items() if rows and expert // home != rank)
    charges = charge_replica_endpoints(config.ep_size, edges, nodes, cross_node_cost)
    loads = [sum(row.values()) for row in copies]
    return limiting_stage_work(loads, charges, ((1, row_credit),))


def _traffic_work_is_useful(copies, loads, config, resident, row_credit, minimum_rows, nodes, cross_node_cost):
    """Preserve capacity-required layouts and check optional rank work."""
    if not row_credit or max(loads) > resident:
        return True
    bound = _critical_quota_work(copies, config, nodes, row_credit, cross_node_cost or 1.0)
    return max(loads) >= bound + minimum_rows


def build_capacity_plan(
    counts_by_source: np.ndarray, target_load: int, *, replica_slots_per_rank: int | None = None,
    alternatives: tuple[int, ...] = (), score: Callable[[int, int], float] | None = None,
) -> ExpertExecutionPlan | None:
    """Keep capacity-based admission stable for the MegaMoE placement policies."""
    limit = 2**31 - 1
    if counts_by_source.min() < 0:
        raise ValueError("HotReplica routed row counts must be nonnegative")
    if counts_by_source.max() > limit or counts_by_source.sum(dtype=np.int64) > limit:
        raise ValueError("HotReplica routed row count exceeds the INT32 planner limit")
    ranks, experts = counts_by_source.shape
    home = experts // ranks
    budget = home if replica_slots_per_rank is None else min(home, replica_slots_per_rank)
    config = ExpertReplicaConfig(experts, ranks, budget)
    _, loads, baseline = _home_layout(counts_by_source, config)
    target = max(1, target_load)
    if not budget:
        return None
    if score is None:
        if max(loads) <= target or min(loads) >= target:
            return None
        copies = _capacity_copies(loads, baseline, config, target)
    else:
        copies = _best_capacity_copies(loads, baseline, config, target, alternatives, score)
        if copies is None:
            return None
    return _materialize(counts_by_source, copies, config, source_locality=False)


def _best_capacity_copies(loads, baseline, config, default, alternatives, score):
    """Compare host quotas before materializing the winner's source table."""
    lightest, home_peak = min(loads), max(loads)
    best, best_score = None, 0.0
    for target in dict.fromkeys((default, *alternatives)):
        if not lightest < target < home_peak:
            continue
        copies = _capacity_copies(loads, [row.copy() for row in baseline], config, target)
        peak = max(sum(row.values()) for row in copies)
        if peak >= home_peak or (target != default and peak > target):
            continue
        outgoing = [0] * config.ep_size
        incoming = [0] * config.ep_size
        for rank, slots in enumerate(copies):
            for expert, rows in slots.items():
                owner = expert // config.home_experts
                if owner != rank and rows:
                    outgoing[owner] += 1
                    incoming[rank] += 1
        benefit = score(peak, max(max(incoming), max(outgoing)))
        if benefit > best_score:
            best, best_score = copies, benefit
    return best


def build_expert_replica_plan(
    counts_by_source: Sequence[Sequence[int]],
    replica_slots_per_rank: int,
    *,
    target_load: int | None = None,
    minimum_replica_rows: int = 0,
    capacity_limit: int | None = None,
    minimum_load_reduction: float = 0.0,
    minimum_saved_rows: int = 0,
    receive_capacity: int | None = None,
    rank_nodes: Sequence[int] = (),
    transfer_row_credit: int = 0,
    cross_node_cost: float | None = None,
) -> ExpertExecutionPlan:
    """Plan lossless replicas under capacity limits and endpoint work checks.

    Args:
        counts_by_source: Nonnegative logical expert histograms, one row per EP rank.
        replica_slots_per_rank: Physical guest slot cap on every rank; zero retains owners.
        target_load: Soft desired received rows, bounded below by the group average.
        minimum_replica_rows: Soft minimum guest rows; capacity-required copies remain.
        capacity_limit: Hard receive allowance. None uses the observed owner peak.
        minimum_load_reduction: Required fractional peak reduction for optional copies.
        minimum_saved_rows: Required absolute peak reduction for optional copies.
        receive_capacity: Agreed receive reservation, independent of the hard allowance.
        rank_nodes: Optional actual node labels in EP rank order.
        transfer_row_credit: Conservative remote-row credit per deleted matrix edge.
            Runtime adapters derive this from mandatory gradient bytes; counts-only
            callers should retain zero. Lower training gain floors also charge
            every actual incoming and owner-return matrix before admission.
        cross_node_cost: Optional relative charge for a guest on another node.

    Returns:
        An immutable plan conserving every source/logical-expert row count.

    Raises:
        ValueError: Input, topology or controls are invalid, or no candidate fits capacity.
    """
    admission = _validate_options(
        (
            ("minimum_replica_rows", minimum_replica_rows),
            ("minimum_saved_rows", minimum_saved_rows),
            ("Replica transfer row credit", transfer_row_credit),
            ("capacity_limit", capacity_limit),
            ("receive_capacity", receive_capacity),
            ("target_load", target_load),
        ),
        minimum_load_reduction,
    )
    counts = _counts_matrix(counts_by_source)
    config = ExpertReplicaConfig(len(counts[0]), len(counts), replica_slots_per_rank)
    nodes = _validate_rank_nodes(rank_nodes, config.ep_size)
    totals, loads, baseline = _home_layout(counts, config)
    average = (sum(loads) + config.ep_size - 1) // config.ep_size
    target = max(average, target_load or 0)
    limit = max(loads) if capacity_limit is None else capacity_limit
    resident = min(limit, receive_capacity) if receive_capacity is not None else limit
    if (
        not replica_slots_per_rank
        or min(loads) == max(loads)
        or not _optional_peak_is_useful(average, loads, resident, admission, minimum_saved_rows)
    ):
        return _materialize(counts, baseline, config, capacity_limit, rank_nodes=nodes)
    copies = _admitted_uncalibrated_copies(
        totals, loads, config, limit, resident, target, minimum_replica_rows, admission, minimum_saved_rows
    )
    copies = baseline if copies is None else copies
    copies = _prefer_locality(copies, config, nodes, loads, limit, receive_capacity, admission,
                              minimum_saved_rows, transfer_row_credit, cross_node_cost, counts)
    _consolidate_copies(copies, counts, config, nodes, transfer_row_credit)
    if minimum_replica_rows > 0 and admission > 0:
        copies = _rebalance_retained_copies(copies, counts, config.home_experts, nodes, minimum_replica_rows)
    if 0 < admission < reduction_basis_points(0.4) and not _traffic_work_is_useful(
        copies, loads, config, resident, transfer_row_credit, minimum_replica_rows, nodes, cross_node_cost
    ):
        copies = baseline
    return _materialize(counts, copies, config, capacity_limit, rank_nodes=nodes)
