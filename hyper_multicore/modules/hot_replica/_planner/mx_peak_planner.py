# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare exact MX plans by their critical rank and whole-matrix transfer count."""
from functools import lru_cache

from .aligned import _AlignedGeometry, _aligned_inputs, _materialize_aligned, _select_aligned
from .topology import charge_replica_endpoints


def mx_guest_transfer_rows(intermediate_size):
    """Express three MX weight copies and one FP32 dW return in BF16 token rows."""
    # Three weight axes, one scale byte per 32 values and six BF16 token messages.
    return 3 * intermediate_size * (3 * 33 + 4 * 32) // (12 * 32)


def _transfer_counts(slots, home, nodes):
    edges, cross = 0, 0
    for target, bank in enumerate(slots):
        for expert in bank[home:]:
            if expert >= 0:
                edges += 1
                cross += nodes[expert // home] != nodes[target]
    return edges, cross


def localize_mx_guests(plan, nodes, weight_rows, *, cross_node_cost=None):
    """Keep the receive peak and save cross-node bytes across the three training phases."""
    if len(set(nodes)) == 1 or not plan.transfers:
        return plan
    width = plan.config.slots_per_rank
    limit = max(plan.destination_loads)
    loads = list(plan.destination_loads)
    slots = [list(row) for row in plan.slot_to_logical]
    spans = list(plan.intervals)
    changed = False
    cross_saving = weight_rows if cross_node_cost is None else weight_rows * (cross_node_cost - 1)
    for edge in plan.transfers:
        owner, target, expert = edge.owner_rank, edge.target_rank, edge.logical_expert
        owner_node, target_node = nodes[owner], nodes[target]
        if owner_node == target_node:
            continue
        old_slot = target * width + edge.target_slot
        start, end = spans[old_slot]
        rows = end - start
        # Token bytes change only for sources on the two affected nodes.
        extra_rows = sum(quota[old_slot] * ((node == target_node) - (node == owner_node))
                         for quota, node in zip(plan.dispatch_counts, nodes))
        if extra_rows >= cross_saving:
            continue
        candidates = ((loads[rank] + rows, rank) for rank, bank in enumerate(slots)
                      if rank != owner and nodes[rank] == owner_node and -1 in bank and expert not in bank
                      and loads[rank] + rows <= limit)
        destination = min(candidates, default=(limit, None))[1]
        if destination is None:
            continue
        slot = slots[destination].index(-1)
        slots[destination][slot] = expert
        spans[destination * width + slot] = (start, end)
        slots[target][edge.target_slot] = -1
        spans[old_slot] = (0, 0)
        loads[destination] += rows
        loads[target] -= rows
        changed = True
    return _materialize_aligned(plan.logical_counts, plan.config, slots, spans) if changed else plan


def _take_peak_split(loads, slots, spans, home, nodes, minimum):
    """Move one aligned suffix on the small host geometry, before quota materialization."""
    rank = loads.index(max(loads))
    width = len(slots[rank])
    current = tuple(sorted(loads, reverse=True))
    best = None
    for local_slot, expert in enumerate(slots[rank]):
        if expert < 0:
            continue
        source = rank * width + local_slot
        start, end = spans[source]
        if end - start < 2 * minimum:
            continue
        for target, row in enumerate(slots):
            if target == rank or expert in row or -1 not in row:
                continue
            moved = min(end - start - minimum, max(minimum, (loads[rank] - loads[target]) // 2))
            aligned = (end - moved) // 64 * 64
            for cut in (aligned, aligned + 64):
                if cut - start < minimum or end - cut < minimum:
                    continue
                after = list(loads)
                after[rank] -= end - cut
                after[target] += end - cut
                order = tuple(sorted(after, reverse=True))
                key = (order, nodes[expert // home] != nodes[target], expert, target, cut)
                if order < current and (best is None or key < best[0]):
                    best = (key, source, row.index(-1), start, end)
    if best is None:
        return False
    (_, _, expert, target, cut), source, slot, start, end = best
    slots[target][slot] = expert
    spans[source] = (start, cut)
    spans[target * width + slot] = (cut, end)
    loads[rank] -= end - cut
    loads[target] += end - cut
    return True


def _split_busiest(geometry, home, nodes, minimum):
    """Try moving one 64-row-aligned suffix into a free guest slot."""
    loads = list(geometry.loads)
    slots = [list(row) for row in geometry.slots]
    spans = list(geometry.intervals)
    if not _take_peak_split(loads, slots, spans, home, nodes, minimum):
        return geometry
    return _AlignedGeometry(tuple(map(tuple, slots)), tuple(spans), tuple(loads))


def _weighted_rank_peak(loads, slots, home, nodes, charge, cross_node_cost=None):
    """Charge full-matrix traffic at both endpoints, where it can delay rank completion."""
    if cross_node_cost is not None:
        edges = tuple((expert // home, rank) for rank, bank in enumerate(slots)
                      for expert in bank[home:] if expert >= 0)
        traffic = charge_replica_endpoints(len(loads), edges, nodes, cross_node_cost)
        # One MX axis carries 33 bytes per 32 values. Backward also
        # returns FP32 dW (128 bytes). Keep separate F and B endpoints.
        phase_bytes = 2 * 33 + 128
        forward = max(phase_bytes * rows + 3 * 33 * charge * copies for rows, copies in zip(loads, traffic))
        backward = max(2 * phase_bytes * rows + 3 * (33 + 128) * charge * copies
                       for rows, copies in zip(loads, traffic))
        return (forward + backward) / (3 * phase_bytes), len(edges), 0
    charged = list(loads)
    edges, cross = 0, 0
    for target, bank in enumerate(slots):
        for expert in bank[home:]:
            if expert < 0:
                continue
            owner = expert // home
            remote = nodes[target] != nodes[owner]
            cost = charge * (1 + remote)
            charged[owner] += cost
            charged[target] += cost
            edges += 1
            cross += remote
    return max(charged) + cross * (charge // 4), edges, cross


def _drain_peak_plateau(geometry, home, nodes, minimum, cross_node_cost=None, *, transfer_charge=None):
    """Split multiple tied busiest ranks when their weight traffic can progress independently."""
    loads = list(geometry.loads)
    peak = max(loads)
    if peak * (2 * len(loads)) < 3 * sum(loads) or sum(rows >= peak - 2 * minimum for rows in loads) < 3:
        return geometry
    slots = [list(row) for row in geometry.slots]
    spans = list(geometry.intervals)
    charge = minimum // 2 if transfer_charge is None else transfer_charge
    before = _weighted_rank_peak(loads, slots, home, nodes, charge, cross_node_cost)
    best, score = geometry, before
    for _ in range(min(8, len(loads) // 2)):
        if not _take_peak_split(loads, slots, spans, home, nodes, minimum):
            break
        current = _weighted_rank_peak(loads, slots, home, nodes, charge, cross_node_cost)
        if max(loads) <= peak - minimum and current[0] <= before[0] - charge and current < score:
            best, score = _AlignedGeometry(tuple(map(tuple, slots)), tuple(spans), tuple(loads)), current
    return best


def _refine_local_peak(geometry, home, nodes, minimum, transfer_rows, cross_node_cost=None):
    """Reduce the critical receive load while preserving node-pair token volumes."""
    peak = max(geometry.loads)
    ranks = len(nodes)
    if (peak - minimum) * ranks < sum(geometry.loads):
        return geometry

    # Reuse the established half-fragment floor, with a larger charge for wider FFNs.
    charge = max(minimum // 2, transfer_rows)
    width = len(geometry.slots[0])
    loads = list(geometry.loads)
    slots = [list(bank) for bank in geometry.slots]
    spans = list(geometry.intervals)
    charged = loads.copy()
    guests = {}
    for rank, bank in enumerate(slots):
        for local_slot in range(home, width):
            expert = bank[local_slot]
            if expert < 0:
                continue
            owner = expert // home
            remote = nodes[owner] != nodes[rank]
            cost = charge * ((2 if cross_node_cost is None else cross_node_cost) if remote else 1)
            charged[owner] += cost
            charged[rank] += cost
            guests[expert] = rank * width + local_slot
    if not guests:
        return geometry

    best, accepted_peak, accepted_charged = geometry, peak, max(charged)
    refined = set()
    for _ in range(min(8, ranks // 2)):
        rank = loads.index(max(loads))
        current_charged = max(charged)
        best_move = None
        for local_slot, expert in enumerate(slots[rank]):
            if expert < 0:
                continue
            source = rank * width + local_slot
            start, end = spans[source]
            owner = expert // home
            move_prefix = local_slot >= home
            peer = owner * width + expert % home if move_prefix else guests.get(expert)
            # The initial plan has at most one guest per expert. Once its
            # spans change, only split the exact current interval again.
            destinations = [] if peer is None or expert in refined else [(peer, 0)]
            if nodes[rank] == nodes[owner] and end - start >= 2 * minimum:
                destinations.extend(
                    (target * width + bank.index(-1), charge)
                    for target, bank in enumerate(slots)
                    if nodes[target] == nodes[rank] and target != rank and -1 in bank and expert not in bank
                )
            for destination, cost in destinations:
                target = destination // width
                if nodes[target] != nodes[rank]:
                    continue
                rows = min(end - start - minimum, (loads[rank] - loads[target]) // 2)
                if rows <= 0:
                    continue
                prefix = move_prefix and not cost
                cut = (start + rows) // 64 * 64 if prefix else (end - rows + 63) // 64 * 64
                moved = cut - start if prefix else end - cut
                if moved <= 0 or (cost and moved < minimum):
                    continue
                after = charged.copy()
                after[rank] -= moved
                after[target] += moved + cost
                after[owner] += cost
                tail = max(after)
                if tail > current_charged:
                    continue
                # The donor is the busiest rank and the receiver stays below its
                # old load. This also permits draining a group of tied donors.
                key = (max(loads[rank] - moved, loads[target] + moved), tail, cost,
                       expert, destination, cut)
                if best_move is None or key < best_move[0]:
                    best_move = key, source, prefix, moved, after
        if best_move is None:
            break
        (_, _, cost, expert, destination, cut), source, prefix, moved, charged = best_move
        start, end = spans[source]
        target, local_slot = divmod(destination, width)
        refined.add(expert)
        loads[rank] -= moved
        loads[target] += moved
        if cost:
            slots[target][local_slot] = expert
            spans[source], spans[destination] = (start, cut), (cut, end)
        elif prefix:
            spans[source], spans[destination] = (cut, end), (0, cut)
        else:
            spans[source], spans[destination] = (start, cut), (cut, spans[destination][1])
        new_peak, new_charged = max(loads), max(charged)
        if new_peak <= accepted_peak - charge and new_charged <= accepted_charged - charge:
            best = _AlignedGeometry(tuple(map(tuple, slots)), tuple(spans), tuple(loads))
            accepted_peak, accepted_charged = new_peak, new_charged
    return best


@lru_cache(maxsize=64)
def build_mx_local_peak_plan(counts, config, *, capacity, minimum_rows, minimum_reduction,
                             row_credit, nodes, transfer_rows, validated_expert_totals=None, cross_node_cost=None):
    """Refine splits, reusing totals only from a route that already validated its counts."""
    matrix, threshold, nodes = _aligned_inputs(counts, config, capacity, minimum_rows, minimum_reduction,
                                               0, row_credit, nodes,
                                               counts_validated=validated_expert_totals is not None)
    totals = validated_expert_totals if validated_expert_totals is not None else tuple(map(sum, zip(*matrix)))
    geometry = _select_aligned(totals, config, capacity, minimum_rows, threshold, 0, row_credit, nodes)
    geometry = _refine_local_peak(geometry, config.home_experts, nodes, max(64, minimum_rows), transfer_rows,
                                  cross_node_cost)
    return _materialize_aligned(matrix, config, geometry.slots, geometry.intervals)


@lru_cache(maxsize=64)
def build_mx_peak_plan(counts, config, *, capacity, minimum_rows, minimum_reduction, row_credit, nodes,
                       drain_peak_plateau=False, admit_small_transfer_plan=False, cross_node_cost=None):
    """Keep extra full weight/dW traffic only when the receive peak buys headroom."""
    matrix, threshold, nodes = _aligned_inputs(counts, config, capacity, minimum_rows, minimum_reduction,
                                               0, row_credit, nodes)
    totals = tuple(map(sum, zip(*matrix)))

    def place(minimum, reduction=threshold, saved=0):
        return _select_aligned(totals, config, capacity, minimum, reduction, saved, row_credit, nodes)

    home = config.home_experts
    initial = place(minimum_rows)
    base_edges, base_cross = _transfer_counts(initial.slots, home, nodes)
    if base_edges:
        alternate = place(minimum_rows // 2)
        split = _split_busiest(initial, home, nodes, minimum_rows)
        split_twice = _split_busiest(split, home, nodes, minimum_rows) if split is not initial else initial

        def score(geometry):
            if cross_node_cost is not None:
                return _weighted_rank_peak(geometry.loads, geometry.slots, home, nodes, minimum_rows // 2,
                                           cross_node_cost)
            edges, cross = _transfer_counts(geometry.slots, home, nodes)
            # Charge each extra guest for its full matrices, twice across nodes.
            allowance = (max(0, edges - base_edges) + max(0, cross - base_cross)) * (minimum_rows // 2)
            return max(geometry.loads) + allowance, edges, cross, sum(rows * rows for rows in geometry.loads)

        candidates = (item for item in (initial, alternate, split, split_twice)
                      if any(expert >= 0 for bank in item.slots for expert in bank[home:]))
        selected = min(candidates, key=score)
        if drain_peak_plateau:
            drained = _drain_peak_plateau(selected, home, nodes, minimum_rows, cross_node_cost)
            peak = max(drained.loads)
            charge = minimum_rows // 2
            # Very hot training plans can leave several near-peak ranks after
            # the coarse drain. Require a full coarse piece of headroom and
            # charge additional inter-node transfers before selecting smaller pieces.
            still_hot = (peak - minimum_rows) * len(drained.loads) >= sum(drained.loads)
            if row_credit and minimum_rows >= 128 and still_hot:
                narrow = _drain_peak_plateau(selected, home, nodes, charge, cross_node_cost,
                                             transfer_charge=charge if cross_node_cost is not None else None)
                edges, cross = _transfer_counts(narrow.slots, home, nodes)
                old_edges, old_cross = _transfer_counts(drained.slots, home, nodes)
                required = minimum_rows + (max(0, cross - old_cross) * (charge // 2)
                                           if cross_node_cost is None else 0)
                if (edges <= old_edges + 4 and max(narrow.loads) + required <= peak
                        and (cross_node_cost is None or score(narrow) < score(drained))):
                    drained = narrow
            selected = drained
    else:
        selected = initial
        if admit_small_transfer_plan:
            # The moderate entry budget applies to small plans after normal admission.
            lower = place(minimum_rows, 1800, minimum_rows)
            if 0 < _transfer_counts(lower.slots, home, nodes)[0] <= 3:
                selected = lower
    return _materialize_aligned(matrix, config, selected.slots, selected.intervals)
