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
"""Bounded replica retained without calibration or device dependencies."""

from __future__ import annotations



def _water_level(receivers: list[tuple[int, int]], participants: int, moved: int) -> tuple[int, int, int]:
    """Raise the lowest receivers together using stable integer remainders."""
    remaining, level, width = moved, receivers[0][0], 1
    for load, _target in receivers[1:participants]:
        required = (load - level) * width
        if required > remaining:
            break
        remaining -= required
        level, width = load, width + 1
    return level + remaining // width, width, remaining % width


def _fill_owner_fragment(copies: list[dict[int, int]], loads: list[int], expert: int,
                         home: int, receivers: list[tuple[int, int]]) -> int:
    """Move only this expert's home rows into its already retained guest matrices."""
    owner = expert // home
    total, participants = loads[owner], 0
    for load, _target in receivers:
        if load >= (total + participants) // (participants + 1):
            break
        total += load
        participants += 1
    if not participants:
        return 0
    moved = min(copies[owner][expert], loads[owner] - (total + participants) // (participants + 1))
    if not moved:
        return 0
    level, width, remainder = _water_level(receivers, participants, moved)
    copies[owner][expert] -= moved
    loads[owner] -= moved
    floor = 0
    for index, (load, target) in enumerate(receivers[:width]):
        rows = level - load + int(index < remainder)
        copies[target][expert] += rows
        loads[target] += rows
        floor = max(floor, copies[target][expert])
    return floor


def _fill_critical_guest_rows(copies: list[dict[int, int]], home: int) -> None:
    """Fill retained matrices only when their home fragment can reduce the expert peak."""
    loads = [sum(row.values()) for row in copies]
    guests: dict[int, list[int]] = {}
    for target, row in enumerate(copies):
        for expert, rows in row.items():
            if rows and expert // home != target:
                guests.setdefault(expert, []).append(target)
    floor = max((rows for rank, row in enumerate(copies) for expert, rows in row.items()
                 if expert not in guests or expert // home != rank), default=0)
    for expert in sorted(guests, key=lambda logical: (-copies[logical // home][logical], logical)):
        if copies[expert // home][expert] <= floor:
            continue
        receivers = sorted((loads[target], target) for target in guests[expert])
        floor = max(floor, _fill_owner_fragment(copies, loads, expert, home, receivers))


def _accept_retained_proposal(copies: list[dict[int, int]], proposal: list[dict[int, int]],
                               counts: tuple[tuple[int, ...], ...], nodes: tuple[int, ...]) -> list[dict[int, int]]:
    """Require a shorter expert fragment and nonincreasing receive and wire pressure."""
    if proposal == copies:
        return copies
    before = (max(sum(row.values()) for row in copies),
              max(rows for row in copies for rows in row.values()))
    after = (max(sum(row.values()) for row in proposal),
             max(rows for row in proposal for rows in row.values()))
    # A lower aggregate receive peak can leave the critical expert unchanged.
    if after[1] >= before[1] or after[0] > before[0]:
        return copies
    return proposal if _fits_retained_traffic(copies, proposal, counts, nodes) else copies


def _wire_limits_hold(old: list[tuple[int, int]], new: list[tuple[int, int]]) -> bool:
    """Compare total sends, peak sends and peak receives independently."""
    old_out, old_in = zip(*old)
    new_out, new_in = zip(*new)
    return sum(new_out) <= sum(old_out) and max(new_out) <= max(old_out) and max(new_in) <= max(old_in)


def _rank_quota_traffic(before: dict[int, int], after: dict[int, int], source: tuple[int, ...],
                        node_change: dict[int, int] | None) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute exact rank traffic and collect any changes to node quotas."""
    local = sum(min(source[expert], rows) for expert, rows in before.items())
    old_out, old_in = sum(source) - local, sum(before.values()) - local
    local_delta, receive_delta = 0, 0
    if before != after:
        for expert, rows in before.items():
            new_rows = after[expert]
            if new_rows == rows:
                continue
            delta = new_rows - rows
            local_delta += min(source[expert], new_rows) - min(source[expert], rows)
            receive_delta += delta
            if node_change is not None:
                node_change[expert] = node_change.get(expert, 0) + delta
    return (old_out, old_in), (old_out - local_delta, old_in + receive_delta - local_delta)


def _node_quota_traffic(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
                        ranks: list[int], changed: dict[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute the old node intersections once and adjust the changed columns."""
    quotas: dict[int, int] = {}
    for rank in ranks:
        for expert, rows in copies[rank].items():
            quotas[expert] = quotas.get(expert, 0) + rows
    node_counts = tuple(counts[rank] for rank in ranks)
    source = tuple(map(sum, zip(*node_counts)))
    local = sum(min(source[expert], rows) for expert, rows in quotas.items())
    old_out, old_in = sum(source) - local, sum(quotas.values()) - local
    local_delta = sum(min(source[expert], quotas[expert] + delta) - min(source[expert], quotas[expert])
                      for expert, delta in changed.items())
    return (old_out, old_in), (old_out - local_delta, old_in + sum(changed.values()) - local_delta)


def _fits_retained_traffic(copies: list[dict[int, int]], proposal: list[dict[int, int]],
                           counts: tuple[tuple[int, ...], ...], nodes: tuple[int, ...]) -> bool:
    """Compare exact wire bounds for a proposal with the same expert keys."""
    old, new = [], []
    node_changes: dict[int, dict[int, int]] = {}
    for rank, (before, after, source) in enumerate(zip(copies, proposal, counts)):
        changed = node_changes.setdefault(nodes[rank], {}) if nodes else None
        previous, candidate = _rank_quota_traffic(before, after, source, changed)
        old.append(previous)
        new.append(candidate)
    if not _wire_limits_hold(old, new):
        return False
    # Rank-local moves can cancel at node scope, leaving all node bounds unchanged.
    if not any(any(delta.values()) for delta in node_changes.values()):
        return True

    grouped: dict[int, list[int]] = {}
    for rank, node in enumerate(nodes):
        grouped.setdefault(node, []).append(rank)
    old, new = [], []
    for node, ranks in grouped.items():
        previous, candidate = _node_quota_traffic(copies, counts, ranks, node_changes[node])
        old.append(previous)
        new.append(candidate)
    return _wire_limits_hold(old, new)


def _minimum_quota_ceiling(limits: list[int], rows: int) -> int:
    """Invert the capacity sum of equally capped receivers with individual limits."""
    for available, limit in zip(range(len(limits), 0, -1), sorted(limits)):
        ceiling = (rows + available - 1) // available
        if ceiling <= limit:
            return ceiling
        rows -= limit
    raise ValueError('Retained expert rows exceed receiver limits')


def _fill_expert_deficit(quotas: dict[int, int], limits: dict[int, int], ranks: list[int],
                         order: list[int], rows: int) -> None:
    """Fill an already feasible deficit while preferring uncovered source rows."""
    for rank in sorted(ranks, key=order.__getitem__):
        if not rows:
            return
        moved = min(rows, limits[rank] - quotas[rank])
        quotas[rank] += moved
        rows -= moved
    if rows:
        raise ValueError('Retained expert quota allocation was incomplete')


def _fits_retained_rank_ingress(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...], expert: int,
                                quotas: dict[int, int], target: dict[int, int]) -> bool:
    """Check the only remaining wire bound when at most two nodes exchange rows."""
    increases = {}
    for rank, rows in target.items():
        local = counts[rank][expert]
        delta = max(rows - local, 0) - max(quotas[rank] - local, 0)
        if delta > 0:
            increases[rank] = delta
    if not increases:
        return True
    incoming = [sum(max(rows - counts[rank][logical], 0) for logical, rows in row.items())
                for rank, row in enumerate(copies)]
    peak = max(incoming)
    return all(incoming[rank] + delta <= peak for rank, delta in increases.items())


def _retained_critical_quotas(
    copies: list[dict[int, int]], home: int,
) -> tuple[int, int, int, dict[int, int]] | None:
    """Find one replicated expert that alone limits the largest fragment."""
    peak, expert, other = -1, -1, 0
    for row in copies:
        for logical, rows in row.items():
            if logical == expert:
                peak = max(peak, rows)
            elif rows > peak or (rows == peak and logical > expert):
                peak, expert, other = rows, logical, max(other, peak)
            elif rows > other:
                other = rows
    owner = expert // home
    quotas = {rank: row[expert] for rank, row in enumerate(copies) if rank == owner or row.get(expert, 0)}
    if len(quotas) < 2:
        return None
    # A fragment of a different expert is unchanged by this candidate.
    return None if other == peak else (peak, expert, other, quotas)


def _retained_node_floors(quotas: dict[int, int], counts: tuple[tuple[int, ...], ...],
                          expert: int, nodes: tuple[int, ...]) -> tuple[list[tuple[list[int], int]], int]:
    """Preserve the expert's current node-local row intersections."""
    if not nodes:
        return [], 0
    demand: dict[int, int] = {}
    groups: dict[int, list[int]] = {}
    for rank, node in enumerate(nodes):
        demand[node] = demand.get(node, 0) + counts[rank][expert]
    for rank in quotas:
        groups.setdefault(nodes[rank], []).append(rank)
    floors = [(ranks, min(demand[node], sum(quotas[rank] for rank in ranks))) for node, ranks in groups.items()]
    return floors, len(demand)


def _spread_retained_expert(quotas: dict[int, int], limits: dict[int, int], ceiling: int,
                            order: list[int], floors: list[tuple[list[int], int]]) -> dict[int, int]:
    """Fill node-local shortfalls before assigning rows with flexible destinations."""
    target = {rank: min(rows, ceiling) for rank, rows in quotas.items()}
    limits = {rank: min(limit, ceiling) for rank, limit in limits.items()}
    for ranks, floor in floors:
        deficit = max(0, floor - sum(target[rank] for rank in ranks))
        _fill_expert_deficit(target, limits, ranks, order, deficit)
    _fill_expert_deficit(target, limits, list(quotas), order, sum(quotas.values()) - sum(target.values()))
    return target


def _retained_floor(quotas: dict[int, int], counts: tuple[tuple[int, ...], ...],
                    expert: int, owner: int, minimum_rows: int) -> int:
    """Preserve source-local rows and each retained guest's minimum fragment."""
    return max(min(rows, max(counts[rank][expert], minimum_rows if rank != owner else 0))
               for rank, rows in quotas.items())


def _refill_critical_expert(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
                            home: int, nodes: tuple[int, ...], minimum_rows: int) -> list[dict[int, int]]:
    """Spread a critical expert across its existing matrices, including its owner."""
    selected = _retained_critical_quotas(copies, home)
    if selected is None:
        return copies
    peak, expert, other, quotas = selected
    owner = expert // home
    floor = _retained_floor(quotas, counts, expert, owner, minimum_rows)
    if floor == peak:
        return copies
    loads = [sum(row.values()) for row in copies]
    receive = max(loads)
    limits = {rank: rows + receive - loads[rank] for rank, rows in quotas.items()}
    ceiling = max(other, floor, _minimum_quota_ceiling(list(limits.values()), sum(quotas.values())))
    node_floors, num_nodes = _retained_node_floors(quotas, counts, expert, nodes)
    for ranks, node_floor in node_floors:
        ceiling = max(ceiling, _minimum_quota_ceiling([limits[rank] for rank in ranks], node_floor))
    if ceiling >= peak:
        return copies
    order = [0] * len(copies)
    for index, rank in enumerate(sorted(quotas, key=lambda rank: (
            counts[rank][expert] <= min(quotas[rank], ceiling), loads[rank], rank))):
        order[rank] = index
    target = _spread_retained_expert(quotas, limits, ceiling, order, node_floors)
    # The local floors cannot increase any rank's or node's outgoing rows.
    # With two nodes one node's outgoing rows are exactly its peer's ingress.
    small_topology = num_nodes <= 2
    if small_topology and not _fits_retained_rank_ingress(copies, counts, expert, quotas, target):
        return copies
    proposal = [dict(row) for row in copies]
    for rank, rows in target.items():
        proposal[rank][expert] = rows
    return proposal if small_topology else _accept_retained_proposal(copies, proposal, counts, nodes)


def _rebalance_retained_copies(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
                                home: int, nodes: tuple[int, ...], minimum_rows: int = 1) -> list[dict[int, int]]:
    """Keep the existing refinement, then try one alternative on a miss."""
    proposal = [dict(row) for row in copies]
    _fill_critical_guest_rows(proposal, home)
    chosen = _accept_retained_proposal(copies, proposal, counts, nodes)
    if chosen is not copies:
        return chosen
    return _refill_critical_expert(copies, counts, home, nodes, minimum_rows)
