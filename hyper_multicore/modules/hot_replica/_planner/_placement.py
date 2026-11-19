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
"""Bounded replica placement without calibration or device dependencies."""

from __future__ import annotations

from itertools import chain

from ._balance import _index_positive_guests, _replacement_rows
from .capacity import ExpertReplicaConfig


def _within_transfer_budget(
    copies: list[dict[int, int]],
    counts: tuple[tuple[int, ...], ...],
    changes: dict[tuple[int, int], int],
    rank_nodes: tuple[int, ...],
    node_ranks: dict[int, list[int]],
    credits: tuple[int, int],
) -> bool:
    """Bound changed token traffic by the deleted matrix edge's conservative row credits."""
    remote_delta = 0
    for (rank, expert), change in changes.items():
        deficit = counts[rank][expert] - copies[rank].get(expert, 0)
        remote_delta += max(deficit - change, 0) - max(deficit, 0)
    if remote_delta > credits[0]:
        return False
    node_changes = {}
    if node_ranks:
        for (rank, expert), change in changes.items():
            key = rank_nodes[rank], expert
            node_changes[key] = node_changes.get(key, 0) + change
    cross_delta = 0
    for (node, expert), change in node_changes.items():
        if change:
            deficit = sum(counts[rank][expert] - copies[rank].get(expert, 0) for rank in node_ranks[node])
            cross_delta += max(deficit - change, 0) - max(deficit, 0)
    return cross_delta <= credits[1]


def _deletion_changes(
    owner: int, target: int, expert: int, rows: int, replacements: list[tuple[int, int, int]]
) -> dict[tuple[int, int], int]:
    """Represent a matrix deletion as exact quota deltas before mutating any layout."""
    changes = {(owner, expert): rows, (target, expert): -rows}
    for rank, logical, moved in replacements:
        key = owner, logical
        changes[key] = changes.get(key, 0) - moved
        key = rank, logical
        changes[key] = changes.get(key, 0) + moved
    return changes


def _apply_changes(
    copies: list[dict[int, int]], loads: list[int], home: int, changes: dict[tuple[int, int], int]
) -> None:
    """Commit admitted integer deltas while retaining zero-valued home slots."""
    for (rank, expert), change in changes.items():
        count = copies[rank].get(expert, 0) + change
        if count or expert // home == rank:
            copies[rank][expert] = count
        else:
            del copies[rank][expert]
        loads[rank] += change


def _recoverable_rows(
    copies: list[dict[int, int]],
    counts: tuple[tuple[int, ...], ...],
    owner: int,
    home: int,
    guests: list[tuple[int, int]],
) -> int:
    """Bound source-local rows that the owner's existing matrices could recover."""
    retained = [(owner, expert) for expert in copies[owner]] + guests
    return sum(
        max(counts[rank][expert] - copies[rank].get(expert, 0), 0)
        for rank, expert in retained
        if expert // home == owner
    )


def _consolidate_copies(
    copies: list[dict[int, int]],
    counts: tuple[tuple[int, ...], ...],
    config: ExpertReplicaConfig,
    rank_nodes: tuple[int, ...],
    transfer_row_credit: int = 0,
) -> None:
    """Delete redundant matrices within receive and source/node traffic limits."""
    home = config.home_experts
    loads = [sum(row.values()) for row in copies]
    ceiling = max(loads)
    by_owner = []
    _index_positive_guests(copies, home, by_owner)
    candidates = sorted(
        (
            -int(bool(rank_nodes) and rank_nodes[expert // home] != rank_nodes[rank]),
            -len(by_owner[expert // home]),
            copies[rank][expert],
            rank,
            expert,
        )
        for rank, expert in chain.from_iterable(by_owner)
    )
    node_ranks = {}
    if 1 < len(set(rank_nodes)) < config.ep_size:
        for rank, node in enumerate(rank_nodes):
            node_ranks.setdefault(node, []).append(rank)
    rank_gain = {}
    for negative_remote, _, _, target, expert in candidates:
        rows, owner = copies[target][expert], expert // home
        if owner not in rank_gain:
            rank_gain[owner] = _recoverable_rows(copies, counts, owner, home, by_owner[owner])
        if min(counts[target][expert], rows) > rank_gain[owner] + transfer_row_credit:
            continue
        replacements = _replacement_rows(copies, loads, home, target, expert, rows, ceiling, by_owner)
        if replacements is None:
            continue
        changes = _deletion_changes(owner, target, expert, rows, replacements)
        credit = transfer_row_credit, -negative_remote * transfer_row_credit
        if _within_transfer_budget(copies, counts, changes, rank_nodes, node_ranks, credit):
            _apply_changes(copies, loads, home, changes)
            rank_gain.pop(owner, None)


class _LocalityPlacement:
    """Own mutable whole-guest moves for one deterministic locality proposal."""

    def __init__(self, selected: list[dict[int, int]], config: ExpertReplicaConfig, nodes: tuple[int, ...]) -> None:
        """Build immutable node membership and invocation-local placement indexes."""
        self.copies = [dict(row) for row in selected]
        self.home, self.budget, self.nodes = config.home_experts, config.replica_slots_per_rank, nodes
        self.loads = [sum(row.values()) for row in selected]
        self.ceiling = max(self.loads)
        self.guest_rows = [
            {expert: rows for expert, rows in row.items() if rows and expert // self.home != rank}
            for rank, row in enumerate(selected)
        ]
        self.guests = list(map(len, self.guest_rows))
        self.node_ranks: dict[int, list[int]] = {}
        for rank, node in enumerate(nodes):
            self.node_ranks.setdefault(node, []).append(rank)
        self.free_room: dict[int, int] = {}

    def _invalidate_room(self, source: int, target: int) -> None:
        """Invalidate only the two nodes whose receive loads changed."""
        self.free_room.pop(self.nodes[source], None)
        self.free_room.pop(self.nodes[target], None)

    def _try_direct(self, source: int, expert: int, rows: int, local: list[int]) -> bool:
        """Move or merge the whole guest into an owner's local node."""
        ordered = sorted(local, key=lambda rank: (expert not in self.copies[rank], self.loads[rank], rank))
        for target in ordered:
            if self.loads[target] + rows > self.ceiling:
                continue
            if expert not in self.copies[target] and self.guests[target] >= self.budget:
                continue
            if expert not in self.copies[target] and expert // self.home != target:
                self.guests[target] += 1
            self.copies[target][expert] = self.copies[target].get(expert, 0) + rows
            if expert // self.home != target:
                self.guest_rows[target][expert] = self.copies[target][expert]
            del self.copies[source][expert]
            del self.guest_rows[source][expert]
            self.guests[source] -= 1
            self.loads[source] -= rows
            self.loads[target] += rows
            self._invalidate_room(source, target)
            return True
        return False

    def _swap_score(self, source: int, target: int, rows: int, other: int, amount: int) -> tuple | None:
        """Score a swap only when it removes remote edges without growing receive peak."""
        lower = self.loads[target] + rows - self.ceiling
        upper = self.ceiling - self.loads[source] + rows
        if amount < lower or amount > upper or other in self.copies[source]:
            return None
        saved = (
            1
            + int(self.nodes[other // self.home] != self.nodes[target])
            - int(self.nodes[other // self.home] != self.nodes[source])
        )
        peak = max(self.loads[source] - rows + amount, self.loads[target] - amount + rows)
        return (saved, -peak, -target, -other) if saved > 0 and peak <= self.ceiling else None

    def _best_swap(self, source: int, expert: int, rows: int, local: list[int]) -> tuple | None:
        """Compare existing guest edges in their original dictionary order."""
        best_score, best = (0, 0, 0, 0), None
        for target in local:
            if expert in self.copies[target]:
                continue
            for other, amount in self.guest_rows[target].items():
                score = self._swap_score(source, target, rows, other, amount)
                if score is not None and score > best_score:
                    best_score, best = score, (target, other, amount)
        return best

    def _apply_swap(self, source: int, expert: int, rows: int, swap: tuple[int, int, int]) -> None:
        """Commit a whole-guest exchange, preserving quota insertion order."""
        target, other, amount = swap
        del self.copies[source][expert]
        del self.copies[target][other]
        self.copies[source][other], self.copies[target][expert] = amount, rows
        del self.guest_rows[source][expert]
        del self.guest_rows[target][other]
        self.guest_rows[source][other], self.guest_rows[target][expert] = amount, rows
        self.loads[source] += amount - rows
        self.loads[target] += rows - amount
        self._invalidate_room(source, target)

    def improve(self, edges: list[tuple[int, int, int]]) -> None:
        """Apply exactly one ordered traversal of the initial remote guest edges."""
        for negative_rows, source, expert in edges:
            rows = -negative_rows
            if self.copies[source].get(expert) != rows:
                continue
            node = self.nodes[expert // self.home]
            local = self.node_ranks[node]
            if node not in self.free_room:
                self.free_room[node] = max(self.ceiling - self.loads[rank] for rank in local)
            if rows <= self.free_room[node] and self._try_direct(source, expert, rows, local):
                continue
            swap = self._best_swap(source, expert, rows, local)
            if swap is not None:
                self._apply_swap(source, expert, rows, swap)


def _locality_copies(
    selected: list[dict[int, int]], config: ExpertReplicaConfig, rank_nodes: tuple[int, ...]
) -> list[dict[int, int]]:
    """Move, merge or swap whole guests without adding edges or receive peak."""
    home = config.home_experts
    edges = sorted(
        (-rows, rank, expert)
        for rank, row in enumerate(selected)
        for expert, rows in row.items()
        if rows and expert // home != rank and rank_nodes[expert // home] != rank_nodes[rank]
    )
    if not edges:
        return selected
    placement = _LocalityPlacement(selected, config, rank_nodes)
    placement.improve(edges)
    return selected if placement.copies == selected else placement.copies
