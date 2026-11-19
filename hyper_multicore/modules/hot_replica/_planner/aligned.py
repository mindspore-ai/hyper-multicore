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
"""Bounded whole/64-row expert fragments preserving global MX quantization groups."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache

from .admission import reduction_basis_points
from .mx_host import _aligned_dispatch, _search_aligned as _search_aligned_native
from .capacity import ExpertReplicaConfig, _integer
from .plan import ExpertExecutionPlan
from .planner import _counts_matrix
from .topology import _validate_rank_nodes


@dataclass(frozen=True)
class _AlignedGeometry:
    """One placement candidate before intersecting its intervals with source counts."""

    slots: tuple[tuple[int, ...], ...]
    intervals: tuple[tuple[int, int], ...]
    loads: tuple[int, ...]


@dataclass(frozen=True)
class AlignedExpertExecutionPlan(ExpertExecutionPlan):
    """Store each physical slot's interval in its original global expert row order."""

    intervals: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Own fragment boundaries alongside the immutable source quota table."""
        super().__post_init__()
        object.__setattr__(self, "intervals", tuple(tuple(span) for span in self.intervals))

    @cached_property
    def _source_slot_order(self) -> tuple[int, ...]:
        # Physical rank order does not imply fragment order: a guest may precede its owner.
        return tuple(sorted((slot for slot, expert in enumerate(self.physical_to_logical) if expert >= 0),
                            key=lambda slot: (self.physical_to_logical[slot], self.intervals[slot][0], slot)))

    def validate(self) -> None:
        """Verify conservation and exact interval intersections independently of placement."""
        super().validate()
        if len(self.intervals) != self.config.physical_experts:
            raise ValueError("Aligned intervals must cover every physical slot")
        for start, stop in self.intervals:
            _integer(start, "Fragment start", 0)
            _integer(stop, "Fragment stop", start)
        intervals_by_expert = [[] for _ in range(self.config.num_experts)]
        for slot, expert in enumerate(self.physical_to_logical):
            start, stop = self.intervals[slot]
            if expert >= 0 and stop > start:
                intervals_by_expert[expert].append(((start, stop), slot))
        for expert, intervals in enumerate(intervals_by_expert):
            end = 0
            for (start, stop), slot in sorted(intervals):
                if start != end or start % 64 or stop <= start:
                    raise ValueError("Fragments must be contiguous with 64-row internal boundaries")
                prefix = 0
                for source, logical in enumerate(self.logical_counts):
                    count = logical[expert]
                    expected = max(0, min(prefix + count, stop) - max(prefix, start))
                    if self.dispatch_counts[source][slot] != expected:
                        raise ValueError("Fragment quotas must intersect original global source intervals")
                    prefix += count
                end = stop
            if end != sum(row[expert] for row in self.logical_counts):
                raise ValueError("Fragments must cover the original expert exactly")


def build_aligned_expert_plan(counts: Sequence[Sequence[int]], config: ExpertReplicaConfig, *,
                              capacity_limit: int, minimum_replica_rows: int = 0,
                              minimum_load_reduction: float = 0.0,
                              minimum_saved_rows: int = 0, transfer_row_credit: int = 0,
                              rank_nodes: Sequence[int] | None = None) -> AlignedExpertExecutionPlan:
    """Freeze exact count agreement before the pure deterministic placement cache."""
    matrix, threshold, nodes = _aligned_inputs(counts, config, capacity_limit, minimum_replica_rows,
                                               minimum_load_reduction, minimum_saved_rows,
                                               transfer_row_credit, rank_nodes)
    return _place_aligned(matrix, config, capacity_limit, minimum_replica_rows, threshold, minimum_saved_rows,
                          transfer_row_credit, nodes)


def _aligned_inputs(counts, config, limit, minimum_rows, reduction, minimum_saved, credit, rank_nodes):
    """Validate the placement request once before searching any alternatives."""
    matrix = _counts_matrix(counts)
    if len(matrix) != config.ep_size or len(matrix[0]) != config.num_experts:
        raise ValueError("Aligned counts must match the logical expert topology")
    _integer(limit, "Aligned expert capacity", 0)
    _integer(minimum_rows, "Aligned minimum guest rows", 0)
    _integer(minimum_saved, "Aligned minimum saved rows", 0)
    _integer(credit, "Aligned transfer row credit", 0)
    nodes = _validate_rank_nodes(() if rank_nodes is None else rank_nodes, config.ep_size) or (0,) * config.ep_size
    return matrix, reduction_basis_points(reduction), nodes


@lru_cache(maxsize=64)
def _search_aligned(totals, config, minimum_rows, nodes):
    """Search on totals; source intersections are materialized separately."""
    result = _search_aligned_native(totals, config.ep_size, config.replica_slots_per_rank, minimum_rows, nodes)
    if result is not None:
        return result
    home = config.home_experts
    loads = [sum(totals[r * home:(r + 1) * home]) for r in range(config.ep_size)]
    guests = [0] * config.ep_size
    selected = {}
    # A logical expert gets at most one guest. Each accepted move strictly improves
    # the (global peak, squared rank loads) objective, so search is bounded by E.
    while config.replica_slots_per_rank:
        score = (max(loads), sum(n * n for n in loads))
        best = None
        for expert, total in enumerate(totals):
            if expert in selected or total < minimum_rows:
                continue
            owner = expert // home
            for target in range(config.ep_size):
                if target == owner or guests[target] >= config.replica_slots_per_rank:
                    continue
                gap = loads[owner] - loads[target]
                if gap <= 0:
                    continue
                ideal_cut = total - gap // 2
                lower = max(0, min(total, ideal_cut)) // 64 * 64
                for cut in sorted({0, lower, lower + 64}):
                    moved = total - cut
                    if not 0 <= cut < total or moved < minimum_rows or (cut and cut < minimum_rows):
                        continue
                    after = loads.copy()
                    after[owner] -= moved
                    after[target] += moved
                    candidate = (max(after), sum(n * n for n in after))
                    # Owner locality breaks exact objective ties only: the
                    # fragment's source rows can reside on the opposite node.
                    key = (candidate, int(nodes[owner] != nodes[target]), expert, target, cut)
                    if candidate < score and (best is None or key < best[0]):
                        best = (key, after)
        if best is None:
            break
        (_, _, expert, target, cut), loads = best
        selected[expert] = (target, cut)
        guests[target] += 1
    return tuple(loads), tuple(selected.items())


@lru_cache(maxsize=64)
def _place_aligned(matrix, config, limit, minimum_rows, threshold, minimum_saved, credit, nodes):
    totals = tuple(map(sum, zip(*matrix)))
    geometry = _select_aligned(totals, config, limit, minimum_rows, threshold, minimum_saved, credit, nodes)
    return _materialize_aligned(matrix, config, geometry.slots, geometry.intervals)


@lru_cache(maxsize=64)
def _select_aligned(totals, config, limit, minimum_rows, threshold, minimum_saved, credit, nodes):
    """Minimize the receive peak, then discard avoidable full-matrix transfers."""
    home, width = config.home_experts, config.slots_per_rank
    loads = [sum(totals[r * home:(r + 1) * home]) for r in range(config.ep_size)]
    original_peak = max(loads)
    minimum_rows = 1 if original_peak > limit else max(64, minimum_rows)
    found_loads, found_selected = _search_aligned(totals, config, minimum_rows, nodes)
    loads, selected = list(found_loads), dict(found_selected)
    peak = max(loads)
    if peak > limit:
        raise ValueError("Aligned expert fragments cannot satisfy the common receive capacity")
    if (original_peak <= limit and (original_peak - peak < minimum_saved or
            (original_peak - peak) * 10000 < original_peak * threshold)):
        selected = {}
    if selected:
        # Delete transfers that only improve a secondary load objective. Every
        # retained guest copies full weights and returns full-shape FP32 dW.
        allowed = min(limit, peak + credit)
        if original_peak <= limit:
            allowed = min(allowed, original_peak - minimum_saved,
                          original_peak * (10000 - threshold) // 10000)
        while True:
            deletions = []
            for expert, (target, cut) in selected.items():
                after = loads.copy()
                after[expert // home] += totals[expert] - cut
                after[target] -= totals[expert] - cut
                if max(after) <= allowed:
                    deletions.append(((max(after), sum(n * n for n in after), expert), after))
            if not deletions:
                break
            (_, _, expert), loads = min(deletions)
            del selected[expert]
    # Whole-expert placement is a cheap competitor, not a split-K seed. Avoid
    # paying several full weight/dW transfers for less than the existing global
    # row allowance. This also preserves complete reductions when equally good.
    allowed_whole_peak = max(loads) + credit
    if selected and original_peak <= limit and max(totals) <= allowed_whole_peak:
        # Optional whole-expert competition must not veto mandatory fragments:
        # the latter can satisfy a bound that no complete expert can fit.
        # An indivisible expert cannot fit below its own row count.
        from .whole import _select_whole_targets
        targets, whole_peak, _ = _select_whole_targets(
            totals, config, capacity_limit=limit, minimum_load_reduction=threshold / 10000,
            minimum_saved_rows=minimum_saved, minimum_replica_rows=minimum_rows,
            transfer_row_credit=credit, nodes=nodes,
        )
        whole = {expert: (target, 0) for expert, target in enumerate(targets) if target != expert // home}
        if whole_peak <= allowed_whole_peak and len(whole) <= len(selected):
            selected = whole

    slots, spans, loads = [], [], []
    for rank in range(config.ep_size):
        logical = list(range(rank * home, (rank + 1) * home))
        for expert in logical:
            spans.append((0, selected[expert][1] if expert in selected else totals[expert]))
        for expert, (target, cut) in sorted(selected.items()):
            if target == rank:
                logical.append(expert)
                spans.append((cut, totals[expert]))
        spans.extend([(0, 0)] * (width - len(logical)))
        slots.append(tuple(logical + [-1] * (width - len(logical))))
        loads.append(sum(stop - start for start, stop in spans[-width:]))
    return _AlignedGeometry(tuple(slots), tuple(spans), tuple(loads))


def _materialize_aligned(matrix, config, slots, spans):
    """Render exact source intersections for an already chosen placement."""
    frozen_slots, frozen_spans = tuple(map(tuple, slots)), tuple(map(tuple, spans))
    native_dispatch = _aligned_dispatch(matrix, frozen_slots, frozen_spans, config.ep_size,
                                        config.home_experts, config.slots_per_rank)
    if native_dispatch is not None:
        return AlignedExpertExecutionPlan(config, matrix, frozen_slots, native_dispatch, frozen_spans)
    dispatch = [[0] * config.physical_experts for _ in matrix]
    for slot, expert in enumerate(e for row in slots for e in row):
        if expert < 0:
            continue
        start, stop = spans[slot]
        prefix = 0
        for source, row in enumerate(matrix):
            count = row[expert]
            dispatch[source][slot] = max(0, min(prefix + count, stop) - max(prefix, start))
            prefix += count
    plan = AlignedExpertExecutionPlan(config, matrix, frozen_slots, tuple(map(tuple, dispatch)), frozen_spans)
    plan.validate()
    return plan
