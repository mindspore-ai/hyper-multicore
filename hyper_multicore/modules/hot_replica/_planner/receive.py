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

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from heapq import heapify, heappop, heappush
from itertools import chain
import os

import numpy as np

from ._dispatch import ReplicaCapacityError, _materialize
from .capacity import ExpertReplicaConfig, _integer
from .native_host import select_prefix
from .topology import _validate_rank_nodes


def receive_limit(percent, total, ranks):
    """Return the exact limit in routed rows. Automatic capacity has no ceiling."""
    return (total * percent + 100 * ranks - 1) // (100 * ranks) if type(percent) is int else None


class SourceQuota(Enum):
    """How source rows are assigned after choosing an expert's execution ranks and shares."""

    LOCAL = "source_local"
    RANK_ORDER = "rank_order"


@dataclass(frozen=True)
class ReplicaCost:
    """Exposed cost in GEMM rows, with owner and guest prices for each serial transfer phase."""

    setup: float = 0.
    phases: tuple[tuple[float, float], ...] = ()
    compute: float = 1.
    minimum_replica_rows: int = 1
    matrix_copy: float = 0.
    admission: ReplicaCost | None = None
    search_single_slot: bool = False

    def transfer(self, sent, received):
        return sum(max(owner * sent, guest * received) for owner, guest in self.phases)

    def keep_home(self, peak, total, ranks, *, limit=None, resident=None):
        """Skip the search when even perfect balance cannot repay activation."""
        if limit is not None and peak > limit:
            return False
        mean = (total + ranks - 1) // ranks
        if resident is not None and mean <= resident < peak:
            return False
        return (peak - mean) * self.compute <= self.setup


@dataclass(frozen=True)
class _CostCoefficients:
    """Coefficients for activation, busiest endpoints and all matrix copies."""

    setup_seconds: float = 0.
    setup_peak_fraction: float = 0.
    endpoint_per_core: float = 0.
    endpoint_per_half_inner: float = 0.
    endpoint_seconds: float = 0.
    endpoint_rows: float = 0.
    phases: tuple[tuple[float, float], ...] = ((1., 1.),)
    copy_rows: float = 0.
    minimum_replica_rows: int = 1

    def cost(self, hidden_size, intermediate_size, cube_cores, peak):
        """Convert time and whole-matrix charges into equivalent GEMM rows."""
        rows_per_second = cube_cores * 7.4e12 / (6 * hidden_size * intermediate_size)
        endpoint = (self.endpoint_per_core * cube_cores + int(self.endpoint_per_half_inner * (intermediate_size // 2))
                    + self.endpoint_seconds * rows_per_second + self.endpoint_rows)
        return ReplicaCost(
            setup=rows_per_second * self.setup_seconds + peak * self.setup_peak_fraction,
            phases=tuple((owner * endpoint, guest * endpoint) for owner, guest in self.phases),
            minimum_replica_rows=self.minimum_replica_rows,
            matrix_copy=self.copy_rows,
        )


_A3_ENDPOINT_PER_CORE = 7.4e12 / 5.9e11
_BF16_TRAINING_PHASES = ((.5, .5), (.5, .25))
_A3_BF16 = {
    (transport, training): _CostCoefficients(setup_seconds=.0035, endpoint_per_core=scale * _A3_ENDPOINT_PER_CORE,
                                             phases=_BF16_TRAINING_PHASES if training else ((1., 1.),))
    for transport, scale in (("mte", 8.), ("other", 1.)) for training in (False, True)
}
_BASE_PROFILES = {(2201, "bf16", transport, training): profile
                  for (transport, training), profile in _A3_BF16.items()}
_AUTO_TRAINING = {(2201, "bf16", "mte"): _A3_BF16["mte", True]}
_AUTO_ADMISSION = {}


def bf16_cost(hidden_size, intermediate_size, cube_cores, training, *, transport_engine="mte"):
    """Convert the A3 PanGu BF16 calibration to equivalent rows."""
    transport = "mte" if transport_engine == "mte" else "other"
    return _A3_BF16[transport, training].cost(hidden_size, intermediate_size, cube_cores, 0)


def replica_cost(architecture, precision, transport, training, automatic, hidden, inner, cores, peak,
                 *, rank_nodes=()):
    """Select coefficients for the executor and price its candidates in one unit."""
    key = (architecture, precision, transport)
    profile = _AUTO_TRAINING.get(key) if training and automatic else None
    if profile is None:
        profile = _BASE_PROFILES.get((*key, training))
    if profile is None:
        score = bf16_cost(hidden, inner, cores, training, transport_engine=transport)
    else:
        score = profile.cost(hidden, inner, cores, peak)
    score = replace(score, search_single_slot=automatic)
    validation = _AUTO_ADMISSION.get(key)
    if training and automatic and validation is not None and rank_nodes:
        expected_sizes, expected_qp, admission = validation
        sizes = tuple(sorted(Counter(rank_nodes).values()))
        if sizes == expected_sizes and os.environ.get("HYPER_MULTICORE_SHMEM_UDMA_QP_NUM") == expected_qp:
            return replace(score, admission=admission.cost(hidden, inner, cores, peak))
    return score


def _lower(matrix, config, totals, moves, nodes, source_quota, limit):
    """Render the selected shares using the executor's source-order contract."""
    ordered = source_quota is SourceQuota.RANK_ORDER
    if ordered:
        matrix = np.asarray(matrix)
    elif isinstance(matrix, np.ndarray):
        matrix = matrix.tolist()
    home = config.home_experts
    rows = [{e: totals[e] for e in range(r * home, (r + 1) * home)} for r in range(config.ep_size)]
    for owner, rank, expert, start, stop in moves:
        rows[owner][expert] -= stop - start
        rows[rank][expert] = stop - start
    return _materialize(matrix, rows, config, limit, rank_nodes=nodes, source_locality=not ordered)


def _tail(remaining, wanted, room, quantum):
    """Choose the nearest legal cut, respecting the receiver's remaining row budget."""
    cut = max(0, remaining - min(wanted, room))
    low = cut // quantum * quantum
    if remaining - low <= room and cut - low <= quantum // 2:
        return low
    return min(remaining, low + quantum)


def _prefixes(totals, loads, config, target, quantum, nodes, multiplier, prices, source_counts, peer_edges,
              minimum_rows):
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
            if (load >= target or len(guests[rank]) >= config.replica_slots_per_rank
                    or peer_edges and not peer_edges[owner][rank]):
                continue
            charge = multiplier if nodes and nodes[rank] != nodes[owner] else 1.
            next_sent = max(max_sent, sent[owner] + charge)
            next_received = max(max_received, received[rank] + charge)
            transfer = prices.transfer(next_sent, next_received) + prices.matrix_copy * charge
            room = target - load
            # No legal cut beats the maximum possible pairwise relief.
            if best_key is not None and min(room, (after[owner] - load) // 2) - transfer < best_key[0]:
                continue
            for expert in experts:
                end = remaining[expert]
                if not end or expert in guests[rank]:
                    continue
                cut = _tail(end, excess, room, quantum)
                moved = end - cut
                if moved < minimum_rows:
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


def _select_prefixes(totals, loads, config, levels, quantum, nodes, multiplier, prices, source_counts,
                     capacity, resident, resize, peer_edges):
    minimum_rows = prices.minimum_replica_rows if capacity is None else 1
    if type(prices) is ReplicaCost:
        native = select_prefix(totals, source_counts, config, quantum, levels, nodes, multiplier,
                               prices, capacity, resident, resize, minimum_rows, peer_edges)
        if native is not None:
            return native
    peak = max(loads)
    best_seen = peak
    best = None if capacity is not None else (0., 0, 0., 0, peak)
    selected = ()
    single_slot = (replace(config, replica_slots_per_rank=1)
                   if prices.search_single_slot and config.replica_slots_per_rank > 1 else None)
    budgets = (config,) if single_slot is None else (config, single_slot)
    for target in levels:
        for budget in budgets:
            moves = []
            weighted_edges = 0.
            length = widest = 0
            occupied = [0] * len(loads) if single_slot is not None else None
            for move, after, sent, received, edge in _prefixes(totals, loads, budget, target, quantum, nodes,
                                                            multiplier, prices, source_counts, peer_edges,
                                                            minimum_rows):
                moves.append(move)
                best_seen = min(best_seen, after)
                weighted_edges += edge
                if occupied is not None:
                    guest = move[1]
                    occupied[guest] += 1
                    widest = max(widest, occupied[guest])
                if capacity is not None and after > capacity:
                    continue
                delta = ((after - peak) * prices.compute + prices.setup + prices.transfer(sent, received)
                         + prices.matrix_copy * weighted_edges)
                if resize and after <= resident:
                    delta -= resize
                width = min(config.replica_slots_per_rank, 1 << (widest - 1).bit_length()) if widest else 0
                key = (delta, width, weighted_edges, len(moves), after)
                if best is None or key < best:
                    best, length = key, len(moves)
            if length:
                selected = moves[:length]
            # With the same legal choices at every selected move the single-slot path is identical.
            if widest <= 1:
                break
    return selected, best_seen


def _fill_capacity(totals, config, limit, quantum, nodes, multiplier, peer_edges):
    """Fill one receiver at a time from one owner; keep unaligned expert tails at home.

    Draining past a donor's own excess lets it become a receiver later. For row quotas
    and a full home-sized slot budget this constructs a feasible layout at the mean.
    """
    home, ranks = config.home_experts, config.ep_size
    left = [n // quantum for n in totals]
    loads = [sum(left[r * home:(r + 1) * home]) for r in range(ranks)]
    caps = [(limit - sum(n % quantum for n in totals[r * home:(r + 1) * home])) // quantum
            for r in range(ranks)]
    if min(caps) < 0 or sum(caps) < sum(loads):
        return None
    moves = []
    while True:
        owner = max(range(ranks), key=lambda r: loads[r] - caps[r])
        if loads[owner] <= caps[owner]:
            return moves
        peers = [r for r in range(ranks) if loads[r] < caps[r] and (not peer_edges or peer_edges[owner][r])]
        if not peers:
            return None
        rank = min(peers, key=lambda r: (multiplier if nodes and nodes[owner] != nodes[r] else 1.,
                                         caps[r] - loads[r], r))
        need = caps[rank] - loads[rank]
        experts = sorted(range(owner * home, (owner + 1) * home), key=lambda e: -left[e])
        for expert in experts[:config.replica_slots_per_rank]:
            blocks = min(need, left[expert])
            if not blocks:
                break
            start = (totals[expert] // quantum - left[expert]) * quantum
            moves.append((owner, rank, expert, start, start + blocks * quantum))
            left[expert] -= blocks
            loads[owner] -= blocks
            loads[rank] += blocks
            need -= blocks
        if need:
            return None


def _pack_capacity(totals, config, limit, quantum, peer_edges):
    """Search small residual hard cases, including moving a short K64 tail off its owner."""
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
        ranks_left = [r for r in range(ranks) if r not in used and space[r] and
                      (r == owner or slots[r] and (not peer_edges or peer_edges[owner][r]))]
        if sum(space[r] for r in ranks_left) < end - start:
            return False
        for rank in sorted(ranks_left, key=lambda r: (r != owner, -space[r], r)):
            stop = min(end, start + space[rank])
            first_aligned = (stop - 1) // quantum * quantum if stop == end else stop // quantum * quantum
            stops = range(first_aligned, start, -quantum)
            for cut in chain((end,) if stop == end else (), stops):
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


def select_replica_plan(matrix, config, *, quantum=1, receive_capacity_percent="auto", limit=None,
                        nodes=(), cross_node_cost=None, prices=ReplicaCost(), source_quota=SourceQuota.LOCAL,
                        totals=None, soft_target=None, resident=None, materialize=None, peer_edges=()):
    """Minimize exposed rank work subject to physical slots, legal cuts and the receive limit.

    The optional absolute limit is for callers that already own a row budget. All built-in
    planners use the percentage of this invocation's actual rows. An empty result is B0.
    The row cut quantum and source quota policy are independent.
    """
    ranks, home = config.ep_size, config.home_experts
    _integer(quantum, "Fragment quantum", 1)
    source_quota = SourceQuota(source_quota)
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
    # A shared receive layout already owns these rows. Charge at most one activation for
    # replacing it. Growing remains legal and competes with all in-reservation candidates.
    resize = max(0., prices.setup) if resident is not None and mean <= resident < peak else 0.
    if not mandatory and (not config.replica_slots_per_rank or (peak - mean) * prices.compute <= prices.setup - resize):
        return None
    if mandatory:
        lower = mean
        for rank, load in enumerate(loads):
            available = (sum(peer_edges[rank]) if peer_edges else ranks) - 1
            slots = available * config.replica_slots_per_rank
            removable = sorted(totals[rank * home:(rank + 1) * home], reverse=True)[:slots]
            lower = max(lower, load - sum(removable))
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
    if resize > 0:
        levels += (resident,)
    selected, best_seen = _select_prefixes(
        totals, loads, config, tuple(dict.fromkeys(levels)), quantum, nodes, multiplier, prices, source_counts,
        limit if mandatory else None, resident, resize, peer_edges,
    )
    if mandatory and not selected:
        selected = _fill_capacity(totals, config, limit, quantum, nodes, multiplier, peer_edges)
        if selected is None:
            selected = _pack_capacity(totals, config, limit, quantum, peer_edges)
        if selected is None:
            raise ReplicaCapacityError(f"No placement found within receive limit {limit} rows with "
                                       f"{config.replica_slots_per_rank} guest slots/rank, quantum {quantum}. "
                                       f"Best peak visited by the balancing search: {best_seen} rows")
    if not selected:
        return None
    counts = matrix if source_quota is SourceQuota.RANK_ORDER else source_counts
    plan = (_lower(counts, config, totals, selected, nodes, source_quota, limit) if materialize is None
            else materialize(source_counts, config, totals, selected))
    if limit is not None and max(plan.destination_loads) > limit:
        raise ReplicaCapacityError(f"Placement exceeds receive limit {limit} rows")
    acceptance = prices.admission
    if limit is None and acceptance is not None:
        degree = plan.maximum_transfer_degree
        delta = ((max(plan.destination_loads) - peak) * acceptance.compute + acceptance.setup
                 + acceptance.transfer(degree, degree))
        if delta > 0:
            return None
    return plan
