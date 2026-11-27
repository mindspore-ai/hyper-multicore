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
"""Choose expert intervals directly from the receive, slot and link budgets."""
from __future__ import annotations

from dataclasses import dataclass, replace
from heapq import heapify, heappop, heappush

import numpy as np

from ._dispatch import ReplicaCapacityError, _materialize
from .native_host import select_prefix
from .topology import _validate_rank_nodes


def receive_limit(percent, total, ranks):
    """Return the exact limit in routed rows. Automatic capacity has no ceiling."""
    # Only the built-in integer policy supplies a hard ceiling.
    # pylint: disable-next=unidiomatic-typecheck
    return (total * percent + 100 * ranks - 1) // (100 * ranks) if type(percent) is int else None


@dataclass(frozen=True)
class ReplicaCost:
    """Exposed cost in GEMM rows, with owner and guest prices for each serial transfer phase."""

    setup: float = 0.
    phases: tuple[tuple[float, float], ...] = ()
    compute: float = 1.
    search_single_slot: bool = False

    def transfer(self, sent, received):
        return sum(max(owner * sent, guest * received) for owner, guest in self.phases)

    def keep_home(self, peak, total, ranks, *, limit=None):
        """Skip the search when even perfect balance cannot repay activation."""
        if limit is not None and peak > limit:
            return False
        mean = (total + ranks - 1) // ranks
        return (peak - mean) * self.compute <= self.setup


def bf16_cost(hidden_size, intermediate_size, cube_cores, training, *, transport_engine="mte",
              search_single_slot=False):
    """Translate exposed latency and expert bytes to rows at the given GEMM geometry.

    Effective rates come from the EP16 PanGu BF16 alternating-route calibration.
    FP32 return accumulates at the owner; its guest only publishes the partial.
    """
    cube_flops, exposed_bandwidth, setup_seconds = 7.4e12, 5.9e11, .0035
    if transport_engine == "mte":
        # The exposed replica matrix rate is independent of token dispatch and topology.
        exposed_bandwidth /= 8
    area = hidden_size * intermediate_size
    rows_per_second = cube_cores * cube_flops / (6 * area)
    edge = rows_per_second * (6 * area / exposed_bandwidth)
    phases = ((edge / 2, edge / 2), (edge / 2, edge / 4)) if training else ((edge, edge),)
    return ReplicaCost(rows_per_second * setup_seconds, phases, search_single_slot=search_single_slot)


def _lower(counts, config, totals, moves, nodes, limit):
    """Render the selected shares using the executor's source-order contract."""
    home = config.home_experts
    rows = [{e: totals[e] for e in range(r * home, (r + 1) * home)} for r in range(config.ep_size)]
    for owner, rank, expert, start, stop in moves:
        rows[owner][expert] -= stop - start
        rows[rank][expert] = stop - start
    return _materialize(counts, rows, config, limit, rank_nodes=nodes)


def _transfer_charge(nodes, owner, guest, multiplier):
    """Apply the inter-node rate only when the route has known distinct nodes."""
    return multiplier if nodes and nodes[owner] != nodes[guest] else 1.


def _prefixes(totals, loads, config, target, nodes, multiplier, prices, source_counts):
    """Drain the busiest owner first. Intermediate equal-peak moves can clear a plateau."""
    after = loads.copy()
    remaining = list(totals)
    sent, received = [0.] * len(loads), [0.] * len(loads)
    guests = [set() for _ in loads]
    home = config.home_experts
    donors = [(-n, r) for r, n in enumerate(after) if n > target]
    heapify(donors)
    max_sent = max_received = 0.
    while donors:
        _, owner = heappop(donors)
        excess = after[owner] - target
        experts = range(owner * home, (owner + 1) * home)
        # A cheaper remote rank may beat a fuller local one.
        best_key = best_move = None
        for rank, load in enumerate(after):
            if load >= target or len(guests[rank]) >= config.replica_slots_per_rank:
                continue
            charge = _transfer_charge(nodes, owner, rank, multiplier)
            next_sent = max(max_sent, sent[owner] + charge)
            next_received = max(max_received, received[rank] + charge)
            transfer = prices.transfer(next_sent, next_received)
            room = target - load
            # No legal cut beats the maximum possible pairwise relief.
            if best_key is not None and min(room, (after[owner] - load) // 2) - transfer < best_key[0]:
                continue
            for expert in experts:
                end = remaining[expert]
                if not end or expert in guests[rank]:
                    continue
                cut = max(0, end - min(excess, room))
                moved = end - cut
                if moved <= 0:
                    continue
                # Marginal reduction of the two endpoints, then source locality and whole experts.
                relief = after[owner] - max(after[owner] - moved, load + moved)
                local = min(moved, source_counts[rank][expert])
                key = (relief - transfer, -charge, local,
                       cut == 0, -load, -rank, -expert)
                if best_key is None or key > best_key:
                    best_key = key
                    best_move = rank, expert, cut, end, charge, next_sent, next_received
        if best_move is None:
            continue
        rank, expert, cut, end, charge, max_sent, max_received = best_move
        moved = end - cut
        after[owner] -= moved
        after[rank] += moved
        remaining[expert] = cut
        guests[rank].add(expert)
        sent[owner] += charge
        received[rank] += charge
        if after[owner] > target:
            heappush(donors, (-after[owner], owner))
        yield (owner, rank, expert, cut, end), max(after), max_sent, max_received, charge


def _select_prefixes(totals, loads, config, levels, nodes, multiplier, prices, source_counts,
                     capacity):
    """Use the native selector for the built-in score, or evaluate the Python score."""
    # A subclass can override the cost calculation used by the Python selector.
    # pylint: disable-next=unidiomatic-typecheck
    if type(prices) is ReplicaCost:
        native = select_prefix(totals, source_counts, config, levels, nodes, multiplier,
                               prices, capacity)
        if native is not None:
            return native
    return _select_python_prefixes(totals, loads, config, levels, nodes, multiplier, prices,
                                   source_counts, capacity)


def _select_python_prefixes(totals, loads, config, levels, nodes, multiplier, prices, source_counts,
                            capacity):
    """Score prefixes from each target and legal slot budget in Python."""
    peak = max(loads)
    best_seen = peak
    best = None if capacity is not None else (0., 0, 0., 0, peak)
    selected = ()
    compare_slots = prices.search_single_slot and config.replica_slots_per_rank > 1
    budgets = (config, replace(config, replica_slots_per_rank=1)) if compare_slots else (config,)
    for target in levels:
        for budget in budgets:
            moves = []
            weighted_edges = 0.
            length = width = 0
            occupied = [0] * len(loads) if compare_slots else None
            for move, after, sent, received, edge in _prefixes(totals, loads, budget, target, nodes,
                                                              multiplier, prices, source_counts):
                moves.append(move)
                best_seen = min(best_seen, after)
                weighted_edges += edge
                if occupied is not None:
                    guest = move[1]
                    occupied[guest] += 1
                    if occupied[guest] > width:
                        width = min(config.replica_slots_per_rank, max(1, 2 * width))
                if capacity is not None and after > capacity:
                    continue
                delta = (after - peak) * prices.compute + prices.setup + prices.transfer(sent, received)
                key = (delta, width, weighted_edges, len(moves), after)
                if best is None or key < best:
                    best, length = key, len(moves)
            if length:
                selected = moves[:length]
            # If every wide move was legal with one slot, the one-slot search is identical.
            if width <= 1:
                break
    return selected, best_seen


def _fill_capacity(totals, config, limit, nodes, multiplier):
    """Fill one receiver at a time from one owner.

    Draining past a donor's own excess lets it become a receiver later. For row quotas
    and a full home-sized slot budget this constructs a feasible layout at the mean.
    """
    home, ranks = config.home_experts, config.ep_size
    left = list(totals)
    loads = [sum(left[r * home:(r + 1) * home]) for r in range(ranks)]
    if limit * ranks < sum(loads):
        return None
    moves = []
    while True:
        owner = max(range(ranks), key=loads.__getitem__)
        if loads[owner] <= limit:
            return moves
        peers = [r for r in range(ranks) if loads[r] < limit]
        rank = min(peers, key=lambda r: (_transfer_charge(nodes, owner, r, multiplier), limit - loads[r], r))
        need = limit - loads[rank]
        experts = sorted(range(owner * home, (owner + 1) * home), key=lambda e: -left[e])
        for expert in experts[:config.replica_slots_per_rank]:
            blocks = min(need, left[expert])
            if not blocks:
                break
            start = totals[expert] - left[expert]
            moves.append((owner, rank, expert, start, start + blocks))
            left[expert] -= blocks
            loads[owner] -= blocks
            loads[rank] += blocks
            need -= blocks
        if need:
            return None


def _pack_capacity(totals, config, limit):
    """Search small residual hard cases when greedy allocation cannot fit."""
    ranks, home = config.ep_size, config.home_experts
    experts = sorted((e for e, n in enumerate(totals) if n), key=lambda e: -totals[e])
    if ranks > 8 or len(experts) > 32:
        return None
    space = [limit] * ranks
    slots = [config.replica_slots_per_rank] * ranks
    moves = []
    checks = 65536

    def place(index, start, used):
        nonlocal checks
        if index == len(experts):
            return True
        expert = experts[index]
        owner, end = expert // home, totals[expert]
        ranks_left = [r for r in range(ranks) if r not in used and space[r] and (r == owner or slots[r])]
        if sum(space[r] for r in ranks_left) < end - start:
            return False
        for rank in sorted(ranks_left, key=lambda r: (r != owner, -space[r], r)):
            stop = min(end, start + space[rank])
            for cut in range(stop, start, -1):
                if checks <= 0:
                    return False
                checks -= 1
                count = cut - start
                space[rank] -= count
                if rank != owner:
                    slots[rank] -= 1
                    moves.append((owner, rank, expert, start, cut))
                if (place(index + 1, 0, set()) if cut == end else place(index, cut, used | {rank})):
                    return True
                if rank != owner:
                    moves.pop()
                    slots[rank] += 1
                space[rank] += count
        return False

    return moves if place(0, 0, set()) else None


def select_replica_plan(matrix, config, *, receive_capacity_percent="auto", limit=None,
                        nodes=(), cross_node_cost=None, prices=ReplicaCost(),
                        totals=None, soft_target=None):
    """Minimize exposed rank work subject to physical slots, legal cuts and the receive limit.

    The optional absolute limit is for callers that already own a row budget. All built-in
    planners use the percentage of this invocation's actual rows. An empty result is B0.
    """
    ranks, home = config.ep_size, config.home_experts
    nodes = _validate_rank_nodes(nodes, ranks)
    if totals is None:
        totals = (matrix.sum(0, dtype=np.int64).tolist() if isinstance(matrix, np.ndarray)
                  else tuple(map(sum, zip(*matrix))))
    loads = [sum(totals[r * home:(r + 1) * home]) for r in range(ranks)]
    total, peak = sum(loads), max(loads)
    mean = (total + ranks - 1) // ranks
    if receive_capacity_percent not in ("auto", None):
        limit = receive_limit(receive_capacity_percent, total, ranks)
    mandatory = limit is not None and peak > limit
    multiplier = 1. if cross_node_cost is None else cross_node_cost
    if not mandatory and (not config.replica_slots_per_rank or prices.keep_home(peak, total, ranks)):
        return None
    if mandatory:
        available = (ranks - 1) * config.replica_slots_per_rank
        lower = max(mean, max(load - sum(sorted(totals[r*home:(r+1)*home], reverse=True)[:available])
                              for r, load in enumerate(loads)))
        if lower > limit:
            raise ReplicaCapacityError(f"Receive limit conflict: receive limit {limit} rows, "
                                       f"{config.replica_slots_per_rank} guest slots/rank, "
                                       f"proven minimum peak >= {lower} rows")
    source_counts = matrix
    if not isinstance(matrix, tuple) or not isinstance(matrix[0], tuple):
        source_counts = tuple(map(tuple, matrix.tolist() if isinstance(matrix, np.ndarray) else matrix))
    # A looser level can remove a single straggler with fewer transfers. Prefixes at
    # the mean also compete; comparing after a whole equal-peak group avoids early stops.
    levels = (limit, mean) if mandatory else (mean, (mean + peak + 1) // 2)
    if soft_target is not None:
        levels = (max(mean, soft_target),)
    selected, best_seen = _select_prefixes(
        totals, loads, config, tuple(dict.fromkeys(levels)), nodes, multiplier, prices, source_counts,
        limit if mandatory else None,
    )
    if mandatory and not selected:
        selected = _fill_capacity(totals, config, limit, nodes, multiplier)
        if selected is None:
            selected = _pack_capacity(totals, config, limit)
        if selected is None:
            raise ReplicaCapacityError(f"No placement found within receive limit {limit} rows with "
                                       f"{config.replica_slots_per_rank} guest slots/rank. "
                                       f"Best peak visited by the balancing search: {best_seen} rows")
    if not selected:
        return None
    return _lower(source_counts, config, totals, selected, nodes, limit)
