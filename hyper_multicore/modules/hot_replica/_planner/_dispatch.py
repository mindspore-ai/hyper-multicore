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
"""Bounded replica dispatch without calibration or device dependencies."""

from __future__ import annotations

from .capacity import ExpertReplicaConfig
from .plan import ExpertExecutionPlan, physical_replica_slots


class ReplicaCapacityError(ValueError):
    """A valid routing table produced no placement within its receive limit."""


def _fill_node_quotas(
    quotas: list[list[int]],
    pending: list[int],
    dispatch: list[list[int]],
    width: int,
    rank_nodes: tuple[int, ...],
    sources: dict[int, list[int]],
) -> None:
    """Saturate node-local intersections after preserving all rank-local rows.

    For fixed expert quotas, each node must export at least its positive source
    surplus. Filling every local intersection attains that bound simultaneously
    with the rank-local bound, without changing any compute or replica quota.
    """
    destinations = {}
    for quota in quotas:
        if quota[1]:
            destinations.setdefault(rank_nodes[quota[0] // width], []).append(quota)
    for node, local in destinations.items():
        index = 0
        for source in sources[node]:
            while pending[source] and index < len(local):
                quota = local[index]
                moved = min(pending[source], quota[1])
                dispatch[source][quota[0]] += moved
                pending[source] -= moved
                quota[1] -= moved
                if not quota[1]:
                    index += 1


def _capacity_checked_plan(plan: ExpertExecutionPlan, capacity_limit: int | None) -> ExpertExecutionPlan:
    """Enforce the hard receive allowance on the physical dispatch."""
    if capacity_limit is not None and max(plan.destination_loads) > capacity_limit:
        raise ReplicaCapacityError("replica plan exceeds capacity_limit; use a feasible receive bound")
    return plan


def _pad_owner_rows(counts: tuple[tuple[int, ...], ...], config: ExpertReplicaConfig) -> list[list[int]]:
    """Place zero guest columns after each owner's source quota."""
    boundaries = range(config.num_experts, 0, -config.home_experts)
    padding = [0] * config.replica_slots_per_rank
    output = []
    for source in counts:
        row = list(source)
        # Insert from the end so earlier owner boundaries keep their offsets.
        for end in boundaries:
            row[end:end] = padding
        output.append(row)
    return output


def _slot_quotas(copies: list[dict[int, int]], config: ExpertReplicaConfig) -> tuple:
    """Freeze the physical slot order and collect positive guest quotas."""
    slot_map, by_expert = [], {}
    home, width = config.home_experts, config.slots_per_rank
    for rank, row in enumerate(copies):
        guests = [expert for expert, count in row.items() if expert // home != rank and count]
        guests.sort()
        slot_map.append(physical_replica_slots(config, rank, guests))
        for slot, expert in enumerate(guests, start=home):
            by_expert.setdefault(expert, []).append([rank * width + slot, row[expert]])
    return tuple(slot_map), by_expert


def _fill_remaining(quotas: list[list[int]], pending: list[int], dispatch: list[list[int]]) -> None:
    """Consume remaining sources and quotas in their original stable order."""
    index = 0
    for source, remaining in enumerate(pending):
        while remaining:
            while quotas[index][1] == 0:
                index += 1
            physical, available = quotas[index]
            rows = min(remaining, available)
            dispatch[source][physical] += rows
            quotas[index][1] -= rows
            remaining -= rows


def _assign_expert(
    counts: tuple[tuple[int, ...], ...],
    expert: int,
    quotas: list[list[int]],
    dispatch: list[list[int]],
    width: int,
    nodes: tuple[int, ...],
    node_sources: dict[int, list[int]],
) -> None:
    """Prefer rank-local rows, then node-local rows, then ordered remote work."""
    if len(quotas) == 1:
        for output, source in zip(dispatch, counts):
            output[quotas[0][0]] = source[expert]
        return
    pending = [row[expert] for row in counts]
    for quota in quotas:
        physical, remaining = quota
        source = physical // width
        local = min(pending[source], remaining)
        dispatch[source][physical] = local
        pending[source] -= local
        quota[1] -= local
    if node_sources:
        _fill_node_quotas(quotas, pending, dispatch, width, nodes, node_sources)
    _fill_remaining(quotas, pending, dispatch)


def _materialize(
    counts: tuple[tuple[int, ...], ...],
    copies: list[dict[int, int]],
    config: ExpertReplicaConfig,
    capacity_limit: int | None = None,
    *,
    rank_nodes: tuple[int, ...] = (),
) -> ExpertExecutionPlan:
    """Materialize only the selected layout, preserving exact source locality."""
    home, width = config.home_experts, config.slots_per_rank
    has_guests = any(
        expert // home != rank and count for rank, row in enumerate(copies) for expert, count in row.items()
    )
    if not has_guests:
        slots = tuple(physical_replica_slots(config, rank) for rank in range(config.ep_size))
        dispatch = tuple(map(tuple, _pad_owner_rows(counts, config))) if config.replica_slots_per_rank else counts
        plan = ExpertExecutionPlan(config, counts, slots, dispatch)
        return _capacity_checked_plan(plan, capacity_limit)
    node_sources = {}
    if 1 < len(set(rank_nodes)) < config.ep_size:
        for rank, node in enumerate(rank_nodes):
            node_sources.setdefault(node, []).append(rank)
    slots, by_expert = _slot_quotas(copies, config)
    dispatch = _pad_owner_rows(counts, config)
    for expert, quotas in by_expert.items():
        owner, local_expert = divmod(expert, home)
        owner_physical = owner * width + local_expert
        if copies[owner].get(expert, 0):
            quotas.append([owner_physical, copies[owner][expert]])
            quotas.sort(key=lambda quota: quota[0])
        for row in dispatch:
            row[owner_physical] = 0
        _assign_expert(counts, expert, quotas, dispatch, width, rank_nodes, node_sources)
    plan = ExpertExecutionPlan(config, counts, slots, tuple(map(tuple, dispatch)))
    return _capacity_checked_plan(plan, capacity_limit)
