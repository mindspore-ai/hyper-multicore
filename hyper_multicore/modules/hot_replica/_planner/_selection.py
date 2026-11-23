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
"""Bounded replica selection without calibration or device dependencies."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ._balance import _balanced_edges, _bounded_copies, _improve_copies, _prune_small_copies
from .admission import optional_peak_gain_is_sufficient
from .capacity import ExpertReplicaConfig


@dataclass
class _QuotaSearchPrefix:
    """Keep one balancing skeleton and its unmodified single-guest seed."""

    edges: list | None = None
    single_seed: list[dict[int, int]] | None = None


def _complete_fallback(
    totals: list[int], loads: list[int], config: ExpertReplicaConfig, edges: list, cutoff: int, limit: int
) -> list[dict[int, int]]:
    """Complete a capacity-overflowing thin seed before rejecting its final layout."""
    copies = _bounded_copies(loads, totals, config, edges=edges)
    _improve_copies(copies, config, cutoff)
    if max(sum(row.values()) for row in copies) > limit:
        raise ValueError("Adaptive replica cap cannot satisfy capacity_limit")
    return copies


def _admit_copies(copies, loads, config, limit, resident, minimum_rows, admission, saved_rows):
    """Qualify the pruned layout before a soft target can terminate quota search."""
    if copies is None:
        return None
    peak = max(sum(row.values()) for row in copies)
    ceiling = min(limit, max(resident, peak))
    _prune_small_copies(copies, loads, config, minimum_rows, ceiling)
    peak = max(sum(row.values()) for row in copies)
    return copies if _optional_peak_is_useful(peak, loads, ceiling, admission, saved_rows) else None


def _uncalibrated_copies(
    totals: list[int],
    loads: list[int],
    config: ExpertReplicaConfig,
    limit: int,
    resident: int,
    target: int,
    *,
    minimum_rows: int,
    admission: int,
    saved_rows: int,
    prefix: _QuotaSearchPrefix | None = None,
) -> list[dict[int, int]] | None:
    """Select finite guest budgets in the original deterministic order."""
    cutoff = min(target, resident)
    if max(loads) <= cutoff:
        return None
    fallback = (max(loads), None) if max(loads) <= limit else None
    edges = _balanced_edges(loads, totals, config.home_experts)
    if prefix is not None:
        prefix.edges = edges
    for budget in range(1, min(config.replica_slots_per_rank, config.home_experts) + 1):
        candidate = replace(config, replica_slots_per_rank=budget)
        copies = _bounded_copies(loads, totals, candidate, edges=edges)
        if budget == 1 and prefix is not None:
            prefix.single_seed = [dict(row) for row in copies]
        peak = max(sum(row.values()) for row in copies)
        if peak <= cutoff:
            admitted = _admit_copies(copies, loads, config, limit, resident, minimum_rows, admission, saved_rows)
            if admitted is not None:
                return admitted
            continue
        if _improves_capacity_fallback(peak, limit, fallback):
            fallback = peak, copies
    if fallback is None:
        selected = _complete_fallback(totals, loads, config, edges, cutoff, limit)
    else:
        selected = fallback[1]
    if selected is not None and fallback is not None:
        _improve_copies(selected, config, cutoff)
    return _admit_copies(selected, loads, config, limit, resident, minimum_rows, admission, saved_rows)


def _quota_fanout(copies: list[dict[int, int]], home: int) -> tuple[int, int]:
    """Count matrix edges and the busiest owner's outgoing copies."""
    outgoing = [0] * len(copies)
    for rank, row in enumerate(copies):
        for expert, count in row.items():
            owner = expert // home
            if count and owner != rank:
                outgoing[owner] += 1
    return sum(outgoing), max(outgoing)


def _sparse_completed_copies(
    totals: list[int],
    loads: list[int],
    config: ExpertReplicaConfig,
    current: list[dict[int, int]] | None,
    limit: int,
    resident: int,
    target: int,
    minimum_rows: int,
    admission: int,
    saved_rows: int,
    prefix: _QuotaSearchPrefix,
) -> list[dict[int, int]] | None:
    """Compare one completed single-guest seed without latency coefficients."""
    single = replace(config, replica_slots_per_rank=1)
    proposal = prefix.single_seed
    if proposal is None:
        proposal = _bounded_copies(loads, totals, single, edges=prefix.edges)
    prefix.single_seed = None
    _improve_copies(proposal, single, min(target, resident))
    peak = max(sum(row.values()) for row in proposal)
    ceiling = min(limit, max(resident, peak))
    _prune_small_copies(proposal, loads, config, minimum_rows, ceiling)
    peak = max(sum(row.values()) for row in proposal)
    if peak > min(limit, resident) or not _optional_peak_is_useful(peak, loads, ceiling, admission, saved_rows):
        return current
    edges, fanout = _quota_fanout(proposal, config.home_experts)
    if not edges:
        return current
    average = (sum(loads) + config.ep_size - 1) // config.ep_size
    # Spend receive headroom only for a large reduction in fixed matrix work.
    # Bound each individual expert by one balanced receiver's work, and the
    # combined receiver by two, even when the owner has much more work.
    if peak > 2 * average or max(max(row.values()) for row in proposal) > average:
        return current
    if current is None:
        return proposal
    old_edges, old_fanout = _quota_fanout(current, config.home_experts)
    if 2 * edges > old_edges or 2 * fanout > old_fanout:
        return current
    return proposal


def _admitted_uncalibrated_copies(
    totals: list[int],
    loads: list[int],
    config: ExpertReplicaConfig,
    limit: int,
    resident: int,
    target: int,
    minimum_rows: int,
    admission: int,
    saved_rows: int,
) -> list[dict[int, int]] | None:
    """Apply load admission and the production sparse completed-guest rescue."""
    sparse = (
        config.replica_slots_per_rank >= 2
        and minimum_rows > 0
        and admission > 0
        and max(loads) * config.ep_size >= 3 * sum(loads)
        and min(target, resident) < max(loads) <= resident
    )
    prefix = _QuotaSearchPrefix() if sparse else None
    copies = _uncalibrated_copies(
        totals, loads, config, limit, resident, target,
        minimum_rows=minimum_rows, admission=admission, saved_rows=saved_rows, prefix=prefix,
    )
    if sparse:
        return _sparse_completed_copies(
            totals, loads, config, copies, limit, resident, target, minimum_rows, admission, saved_rows, prefix
        )
    return copies


def _optional_peak_is_useful(candidate_peak, loads, limit, admission, minimum_saved_rows):
    """Keep required placements and compare optional load reduction using integers."""
    peak = max(loads)
    if (not admission and not minimum_saved_rows) or peak > limit:
        return True
    return optional_peak_gain_is_sufficient(peak, candidate_peak, admission, minimum_saved_rows)


def _improves_capacity_fallback(peak, limit, fallback):
    """Preserve the strict best admissible fallback without changing quota traversal order."""
    return peak <= limit and (fallback is None or peak < fallback[0])
