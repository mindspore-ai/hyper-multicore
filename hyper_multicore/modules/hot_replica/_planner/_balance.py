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
"""Bounded replica balance without calibration or device dependencies."""

from __future__ import annotations

from heapq import heapify, heappop, heappush

from .capacity import ExpertReplicaConfig


def _balanced_edges(loads: list[int], expert_counts: list[int],
                    home: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Fill each receiver once, always sourcing original home experts."""
    rank_count = len(loads)
    quotient, remainder = divmod(sum(loads), rank_count)
    targets = [quotient + int(rank < remainder) for rank in range(rank_count)]
    remaining = list(expert_counts)
    excess = [load - target for load, target in zip(loads, targets)]
    donors = [(-rows, rank) for rank, rows in enumerate(excess) if rows > 0]
    receivers = [(rows, rank) for rank, rows in enumerate(excess) if rows < 0]
    heapify(donors)
    heapify(receivers)
    edges = []
    while receivers:
        donor_rows, donor = heappop(donors)
        receiver_rows, receiver = heappop(receivers)
        amount = -receiver_rows
        pending = amount
        segments = []
        experts = sorted(range(donor * home, (donor + 1) * home), key=lambda expert: (-remaining[expert], expert))
        for expert in experts:
            moved = min(pending, remaining[expert])
            if moved:
                segments.append((expert, moved))
                remaining[expert] -= moved
                pending -= moved
        if pending:
            raise RuntimeError("balanced transfer exceeds remaining home work")
        edges.append((donor, receiver, segments))
        donor_excess = -donor_rows - amount
        if donor_excess > 0:
            heappush(donors, (-donor_excess, donor))
        elif donor_excess < 0:
            # Filling a receiver can turn the donor into the next receiver.
            heappush(receivers, (donor_excess, donor))
    return edges


def _bounded_copies(loads: list[int], expert_counts: list[int], config: ExpertReplicaConfig,
                    *, edges: list | None = None) -> list[dict[int, int]]:
    """Thin balanced edges to fit B distinct guests without changing ownership."""
    home = config.home_experts
    budget = min(config.replica_slots_per_rank, home)
    copies = [{expert: expert_counts[expert] for expert in range(rank * home, (rank + 1) * home)}
              for rank in range(config.ep_size)]
    if not budget:
        return copies
    for donor, receiver, segments in (_balanced_edges(loads, expert_counts, home) if edges is None else edges):
        if len(segments) == 1:
            expert, count = segments[0]
            kept = budget * count // home
            if kept:
                copies[donor][expert] -= kept
                copies[receiver][expert] = kept
            continue
        quota = budget * sum(count for _, count in segments) // home
        for expert, count in sorted(segments, key=lambda segment: (-segment[1], segment[0]))[:budget]:
            kept = min(count, quota)
            if kept:
                copies[donor][expert] -= kept
                copies[receiver][expert] = kept
                quota -= kept
        if quota:
            raise RuntimeError("top-B experts cannot satisfy bounded transfer quota")
    return copies


def _owner_guests(copies: list[dict[int, int]], home: int) -> list[list[tuple[int, int]]]:
    """Index existing guest matrices without changing their insertion order."""
    result = [[] for _ in copies]
    for rank, row in enumerate(copies):
        for expert in row:
            if expert // home != rank:
                result[expert // home].append((rank, expert))
    return result


def _best_free_move(copies: list[dict[int, int]], donor: int, home: int,
                    excess: int, free: list[tuple[int, int]]) -> tuple | None:
    """Choose the original rank/expert tie-break over free receiver slots."""
    experts = range(donor * home, (donor + 1) * home)
    headroom = max((rows for _, rows in free), default=0)
    rows = min(max(copies[donor][expert] for expert in experts), excess, headroom)
    if not rows:
        return None
    receiver = next(rank for rank, room in free if room >= rows)
    expert = next(expert for expert in experts if copies[donor][expert] >= rows)
    return receiver, expert, rows


def _best_existing_move(copies: list[dict[int, int]], donor: int, target: int, loads: list[int],
                        guests: list[tuple[int, int]], best: tuple | None) -> tuple | None:
    """Prefer an already copied matrix on otherwise equal row gains."""
    score = (0, 0, 0, 0) if best is None else (best[2], int(best[1] in copies[best[0]]), -best[0], -best[1])
    for receiver, expert in guests:
        rows = min(copies[donor][expert], loads[donor] - target, target - loads[receiver])
        candidate = rows, 1, -receiver, -expert
        if rows > 0 and candidate > score:
            score, best = candidate, (receiver, expert, rows)
    return best


def _improve_copies(copies: list[dict[int, int]], config: ExpertReplicaConfig, target: int) -> None:
    """Use spare guest capacity only for moves that reduce donor excess."""
    loads = [sum(row.values()) for row in copies]
    donors = [(-load, rank) for rank, load in enumerate(loads) if load > target]
    heapify(donors)
    home = config.home_experts
    by_owner = _owner_guests(copies, home)
    guest_counts = [sum(expert // home != rank for expert in row) for rank, row in enumerate(copies)]
    free_ranks = list(range(config.ep_size))
    while donors:
        _, donor = heappop(donors)
        free_ranks = [rank for rank in free_ranks
                      if loads[rank] < target and guest_counts[rank] < config.replica_slots_per_rank]
        free = [(rank, target - loads[rank]) for rank in free_ranks]
        best = _best_free_move(copies, donor, home, loads[donor] - target, free)
        best = _best_existing_move(copies, donor, target, loads, by_owner[donor], best)
        if best is None:
            continue
        receiver, expert, rows = best
        copies[donor][expert] -= rows
        if expert not in copies[receiver]:
            guest_counts[receiver] += 1
            by_owner[donor].append((receiver, expert))
        copies[receiver][expert] = copies[receiver].get(expert, 0) + rows
        loads[donor] -= rows
        loads[receiver] += rows
        if loads[donor] > target:
            heappush(donors, (-loads[donor], donor))


def _index_positive_guests(copies: list[dict[int, int]], home: int,
                           output: list[list[tuple[int, int]]]) -> None:
    """Populate a lazy owner index only when fragment removal needs consolidation."""
    output.extend([] for _ in copies)
    for rank, row in enumerate(copies):
        for expert, count in row.items():
            if expert // home != rank and count > 0:
                output[expert // home].append((rank, expert))


def _replacement_rows(copies: list[dict[int, int]], loads: list[int], home: int, target: int,
                      expert: int, rows: int, limit: int,
                      copies_by_owner: list[list[tuple[int, int]]]) -> list[tuple[int, int, int]] | None:
    """Concentrate necessary work in existing copies before removing a matrix."""
    owner = expert // home
    needed = max(loads[owner] + rows - limit, 0)
    if not needed:
        return []
    if not copies_by_owner:
        _index_positive_guests(copies, home, copies_by_owner)
    available = dict(copies[owner])
    available[expert] += rows
    existing = sorted((-copies[rank][logical], rank, logical) for rank, logical in copies_by_owner[owner]
                      if copies[rank].get(logical, 0) > 0 and (rank, logical) != (target, expert))
    headroom = {rank: limit - loads[rank] + (rows if rank == target else 0) for _, rank, _ in existing}
    replacements = []
    for _, rank, logical in existing:
        if not needed:
            break
        moved = min(needed, available[logical], headroom[rank])
        if moved > 0:
            replacements.append((rank, logical, moved))
            available[logical] -= moved
            headroom[rank] -= moved
            needed -= moved
    return None if needed else replacements


def _prune_small_copies(copies: list[dict[int, int]], loads: list[int],
                        config: ExpertReplicaConfig, minimum_rows: int, capacity_limit: int) -> None:
    """Remove or consolidate small replicas without exceeding the receive bound."""
    if not minimum_rows:
        return
    home = config.home_experts
    candidates = sorted((rows, target, expert) for target, row in enumerate(copies)
                        for expert, rows in row.items() if expert // home != target and 0 < rows < minimum_rows)
    if not candidates:
        return
    limit = min(max(loads), capacity_limit)
    current = [sum(row.values()) for row in copies]
    copies_by_owner = []
    for _, target, expert in candidates:
        rows = copies[target].get(expert, 0)
        if not 0 < rows < minimum_rows:
            continue
        replacements = _replacement_rows(copies, current, home, target, expert, rows, limit,
                                         copies_by_owner)
        if replacements is None:
            continue
        owner = expert // home
        copies[owner][expert] += rows
        del copies[target][expert]
        current[owner] += rows
        current[target] -= rows
        for rank, logical, moved in replacements:
            copies[owner][logical] -= moved
            copies[rank][logical] += moved
            current[owner] -= moved
            current[rank] += moved
