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
"""Keep MXFP8 expert reductions complete while using bounded guest slots."""

from __future__ import annotations

from functools import lru_cache

from .admission import reduction_basis_points
from .mx_host import _search_whole as _search_whole_native


@lru_cache(maxsize=64)
def _search_whole(weights, config, minimum_replica_rows, capacity_required, nodes):
    """Share deterministic whole-expert searches across exact source tables."""
    result = _search_whole_native(weights, config.ep_size, config.replica_slots_per_rank,
                                  minimum_replica_rows, capacity_required, nodes)
    if result is not None:
        return result
    owners = [expert // config.home_experts for expert in range(config.num_experts)]
    targets = owners.copy()
    loads = [sum(weights[start:start + config.home_experts])
             for start in range(0, config.num_experts, config.home_experts)]
    guests = [0] * config.ep_size
    cross, guest_total = 0, 0

    def select_move() -> tuple[tuple[int, int, int, int], tuple[int, int, int] | None]:
        """Choose the first strict improvement using the existing deterministic score."""
        square = sum(load * load for load in loads)
        current = (max(loads), square, cross, guest_total)
        leaders = sorted(range(config.ep_size), key=loads.__getitem__, reverse=True)[:3]
        outside = tuple(tuple(next((loads[r] for r in leaders if r not in (source, target)), 0)
                              for target in range(config.ep_size)) for source in range(config.ep_size))
        best = current
        selected = None
        for expert, weight in enumerate(weights):
            if not weight:
                continue
            source, owner = targets[expert], owners[expert]
            for target in range(config.ep_size):
                if target == source or (target != owner and (guests[target] >= config.replica_slots_per_rank
                        or (weight < minimum_replica_rows and not capacity_required))):
                    continue
                remaining, arriving = loads[source] - weight, loads[target] + weight
                peak = max(remaining, arriving, outside[source][target])
                candidate_square = square - loads[source] ** 2 - loads[target] ** 2 + remaining ** 2 + arriving ** 2
                candidate_cross = cross - int(nodes[source] != nodes[owner]) + int(nodes[target] != nodes[owner])
                candidate_guests = guest_total - int(source != owner) + int(target != owner)
                score = peak, candidate_square, candidate_cross, candidate_guests
                if score < best:
                    best, selected = score, (expert, source, target)
        return best, selected

    for _ in range(config.num_experts + config.physical_experts):
        best, selected = select_move()
        if selected is None:
            break
        expert, source, target = selected
        weight, owner = weights[expert], owners[expert]
        loads[source] -= weight
        loads[target] += weight
        guests[source] -= int(source != owner)
        guests[target] += int(target != owner)
        targets[expert] = target
        cross, guest_total = best[2:]
    return tuple(loads), tuple(targets)


@lru_cache(maxsize=64, typed=True)
def _select_whole_targets(weights, config, *, capacity_limit, minimum_load_reduction,
                          minimum_saved_rows, minimum_replica_rows, nodes, transfer_row_credit):
    """Choose destinations from totals before any source-quota materialization."""
    owners = [expert // config.home_experts for expert in range(config.num_experts)]
    home_loads = [sum(weights[start:start + config.home_experts])
                  for start in range(0, config.num_experts, config.home_experts)]
    baseline_peak = max(home_loads)
    lightest_home = min(home_loads)
    capacity_required = baseline_peak > capacity_limit
    found_loads, found_targets = _search_whole(weights, config, minimum_replica_rows, capacity_required, nodes)
    loads, targets = list(found_loads), list(found_targets)

    def prune_guests() -> None:
        """Prune toward a whole-expert target instead of the admission ceiling."""
        # Allow one largest expert beside the lightest original home workload.
        # This avoids moving cold matrices merely to free a few rows for a guest.
        target = max(max(loads), max(weights) + lightest_home)
        ceiling = min(capacity_limit, target + transfer_row_credit)
        if not capacity_required:
            ceiling = min(ceiling, baseline_peak - 1, baseline_peak - minimum_saved_rows,
                          int(baseline_peak * (1 - minimum_load_reduction)))
        for _ in range(config.num_experts):
            changed = False
            order = sorted(range(config.num_experts),
                           key=lambda expert: (-int(nodes[targets[expert]] != nodes[owners[expert]]),
                                               weights[expert], expert))
            for expert in order:
                source, owner, weight = targets[expert], owners[expert], weights[expert]
                if source != owner and loads[owner] + weight <= ceiling:
                    loads[source] -= weight
                    loads[owner] += weight
                    targets[expert] = owner
                    changed = True
            if not changed:
                break

    prune_guests()
    peak = max(loads)

    def admitted() -> bool:
        """Keep necessary capacity moves independent of optional benefit thresholds."""
        return (peak <= capacity_limit if capacity_required else
                peak < baseline_peak and baseline_peak - peak >= minimum_saved_rows
                and peak <= baseline_peak * (1 - minimum_load_reduction))

    if not admitted():
        targets = owners.copy()
        peak = baseline_peak
    if peak > capacity_limit:
        raise ValueError("A complete expert cannot satisfy the common receive capacity without splitting K")
    return targets, peak, baseline_peak
