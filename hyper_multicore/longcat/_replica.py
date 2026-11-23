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
"""Expert-replica planner and P2P weight/gradient binding for LongcatFlashExperts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist

ReplicaGranularity = Literal["expert", "token"]


@dataclass(frozen=True)
class ReplicaTransfer:
    """One owner->guest weight copy and matching guest->owner gradient return."""

    logical_expert: int
    owner_rank: int
    owner_slot: int
    target_rank: int
    target_slot: int  # rank-local physical slot index (>= home_experts)
    target_physical: int  # global physical expert index


@dataclass(frozen=True)
class LongcatReplicaPlan:
    """Immutable per-step physical routing and transfer plan for real experts."""

    ep_size: int
    num_experts: int
    home_experts: int
    replica_slots_per_rank: int
    physical_experts: int
    granularity: ReplicaGranularity
    home_only: bool
    transfers: tuple[ReplicaTransfer, ...]
    dispatch_counts: tuple[tuple[int, ...], ...]
    # For each source rank r and logical expert e: tuple of (physical_slot, rows).
    source_quotas: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]
    logical_rank_loads: tuple[int, ...]
    physical_rank_loads: tuple[int, ...]


def _global_peer_rank(group: Any, peer: int) -> int:
    """Translate EP group-local rank to global rank for P2P operations."""
    return peer if group is None else dist.get_global_rank(group, peer)


def _exchange_p2p(operations: list[dist.P2POp]) -> None:
    """Run one batched asynchronous P2P exchange and wait for completion."""
    if operations:
        for work in dist.batch_isend_irecv(operations):
            work.wait()


def _home_plan(
    counts_by_source: list[list[int]],
    ep_size: int,
    num_experts: int,
    replica_slots_per_rank: int,
    granularity: ReplicaGranularity,
    owner_loads: list[int],
) -> LongcatReplicaPlan:
    """Return a no-replica home-only plan."""
    home = num_experts // ep_size
    return LongcatReplicaPlan(
        ep_size=ep_size,
        num_experts=num_experts,
        home_experts=home,
        replica_slots_per_rank=replica_slots_per_rank,
        physical_experts=num_experts + ep_size * replica_slots_per_rank,
        granularity=granularity,
        home_only=True,
        transfers=(),
        dispatch_counts=tuple(tuple(row) for row in counts_by_source),
        source_quotas=(),
        logical_rank_loads=tuple(owner_loads),
        physical_rank_loads=tuple(owner_loads),
    )


def _materialize_dispatch(
    counts_by_source: list[list[int]],
    ep_size: int,
    num_experts: int,
    home: int,
    phys_per_rank: int,
    expert_allocs: list[list[tuple[int, int]]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[tuple[tuple[int, int], ...], ...], ...]]:
    """Assign each source rank logical-expert rows to physical slots preserving exact totals."""
    phys_experts = ep_size * phys_per_rank
    dispatch = [[0] * phys_experts for _ in range(ep_size)]
    quotas: list[list[list[tuple[int, int]]]] = [[[] for _ in range(num_experts)] for _ in range(ep_size)]

    for e in range(num_experts):
        copies = [(slot, cap) for slot, cap in expert_allocs[e] if cap > 0]
        if not copies:
            continue
        if len(copies) == 1:
            slot = copies[0][0]
            for r in range(ep_size):
                c = counts_by_source[r][e]
                if c:
                    dispatch[r][slot] += c
                    quotas[r][e].append((slot, c))
            continue

        remaining_cap = {slot: cap for slot, cap in copies}
        remaining_src = [counts_by_source[r][e] for r in range(ep_size)]
        # Pass 1: prefer local slot on the same rank to minimize cross-rank token movement.
        for slot, _ in copies:
            dst_rank = slot // phys_per_rank
            take = min(remaining_src[dst_rank], remaining_cap[slot])
            if take > 0:
                remaining_src[dst_rank] -= take
                remaining_cap[slot] -= take
                dispatch[dst_rank][slot] += take
                quotas[dst_rank][e].append((slot, take))
        # Pass 2: greedily assign remaining source rows to remaining slot capacities.
        slot_order = [slot for slot, _ in copies]
        s_idx = 0
        for r in range(ep_size):
            rem = remaining_src[r]
            while rem > 0:
                while remaining_cap[slot_order[s_idx]] == 0:
                    s_idx += 1
                slot = slot_order[s_idx]
                take = min(rem, remaining_cap[slot])
                rem -= take
                remaining_cap[slot] -= take
                dispatch[r][slot] += take
                quotas[r][e].append((slot, take))

    return (
        tuple(tuple(row) for row in dispatch),
        tuple(tuple(tuple(q) for q in row) for row in quotas),
    )


def _traffic_work_is_useful(
    loads: list[int],
    transfers: list[ReplicaTransfer],
    peak: int,
    row_credit: int,
    minimum_rows: int,
) -> bool:
    """Ensure saved critical-path rows exceed the conservative weight/gradient transfer credit."""
    if row_credit <= 0:
        return True
    ep_size = len(loads)
    outgoing = [0] * ep_size
    incoming = [0] * ep_size
    for xfer in transfers:
        outgoing[xfer.owner_rank] += 1
        incoming[xfer.target_rank] += 1
    bound = max(
        loads[r] + row_credit * (outgoing[r] + incoming[r])
        for r in range(ep_size)
    )
    return peak >= bound + minimum_rows


def plan_longcat_replicas(
    counts_by_source: list[list[int]],
    replica_slots_per_rank: int,
    *,
    granularity: ReplicaGranularity = "expert",
    minimum_replica_rows: int = 1024,
    minimum_load_reduction: float = 0.25,
    transfer_row_credit: int = 0,
) -> LongcatReplicaPlan:
    """Build a lossless whole-expert or token-split replica plan from real-expert counts."""
    ep_size = len(counts_by_source)
    num_experts = len(counts_by_source[0]) if ep_size else 0
    home = num_experts // ep_size
    expert_totals = [sum(counts_by_source[r][e] for r in range(ep_size)) for e in range(num_experts)]
    owner_loads = [sum(expert_totals[r * home : (r + 1) * home]) for r in range(ep_size)]
    peak = max(owner_loads) if owner_loads else 0
    total = sum(owner_loads)
    if replica_slots_per_rank <= 0 or ep_size <= 1 or total == 0:
        return _home_plan(counts_by_source, ep_size, num_experts, replica_slots_per_rank, granularity, owner_loads)

    avg = (total + ep_size - 1) // ep_size
    if peak <= avg or (peak - avg) < max(minimum_replica_rows, int(peak * minimum_load_reduction)):
        return _home_plan(counts_by_source, ep_size, num_experts, replica_slots_per_rank, granularity, owner_loads)

    phys_per_rank = home + replica_slots_per_rank
    loads = owner_loads.copy()
    used_guests = [0] * ep_size
    transfers: list[ReplicaTransfer] = []
    # Map logical expert -> list of [phys_slot, assigned_rows]
    expert_allocs: list[list[list[int]]] = [
        [[(e // home) * phys_per_rank + (e % home), expert_totals[e]]]
        for e in range(num_experts)
    ]

    if granularity == "expert":
        # Indivisible whole-expert placement: each logical expert runs on exactly one physical slot.
        candidates = sorted(
            (e for e in range(num_experts) if expert_totals[e] >= minimum_replica_rows),
            key=lambda e: (-expert_totals[e], e),
        )
        for e in candidates:
            owner = e // home
            rows = expert_totals[e]
            if loads[owner] <= avg:
                continue
            cur_peak = max(loads)
            cur_sq = sum(v * v for v in loads)
            best_rank = None
            best_key = (cur_peak, cur_sq)
            for dst in range(ep_size):
                if dst == owner or used_guests[dst] >= replica_slots_per_rank:
                    continue
                if loads[dst] + rows >= loads[owner]:
                    continue
                proposed = loads.copy()
                proposed[owner] -= rows
                proposed[dst] += rows
                key = (max(proposed), sum(v * v for v in proposed))
                if key < best_key:
                    best_key = key
                    best_rank = dst
            if best_rank is not None:
                target_slot = home + used_guests[best_rank]
                target_phys = best_rank * phys_per_rank + target_slot
                used_guests[best_rank] += 1
                loads[owner] -= rows
                loads[best_rank] += rows
                expert_allocs[e] = [[target_phys, rows]]
                transfers.append(
                    ReplicaTransfer(
                        logical_expert=e,
                        owner_rank=owner,
                        owner_slot=e % home,
                        target_rank=best_rank,
                        target_slot=target_slot,
                        target_physical=target_phys,
                    )
                )
    else:
        # Token-level replica placement: split hot experts across owner + idle guest slots.
        replicated_on: dict[int, set[int]] = {e: {e // home} for e in range(num_experts)}
        while True:
            cur_peak = max(loads)
            if cur_peak <= avg:
                break
            hot_rank = max(range(ep_size), key=lambda r: (loads[r], -r))
            idle_ranks = [
                r for r in range(ep_size)
                if r != hot_rank and used_guests[r] < replica_slots_per_rank and loads[r] < avg
            ]
            if not idle_ranks:
                break
            cold_rank = min(idle_ranks, key=lambda r: (loads[r], r))
            surplus = loads[hot_rank] - avg
            deficit = avg - loads[cold_rank]

            # Find hottest expert with rows still residing on hot_rank (owned by hot_rank).
            best_e = None
            best_avail = 0
            for e in range(hot_rank * home, (hot_rank + 1) * home):
                if cold_rank in replicated_on[e]:
                    continue
                avail = expert_allocs[e][0][1]
                if avail > best_avail:
                    best_avail = avail
                    best_e = e
            if best_e is None or best_avail < minimum_replica_rows:
                break
            move = min(surplus, deficit, best_avail)
            # Avoid leaving a tiny owner sliver below minimum_replica_rows; move whole expert if close.
            if 0 < best_avail - move < minimum_replica_rows and loads[cold_rank] + best_avail <= cur_peak:
                move = best_avail
            if move < minimum_replica_rows:
                break

            target_slot = home + used_guests[cold_rank]
            target_phys = cold_rank * phys_per_rank + target_slot
            used_guests[cold_rank] += 1
            replicated_on[best_e].add(cold_rank)
            expert_allocs[best_e][0][1] -= move
            expert_allocs[best_e].append([target_phys, move])
            loads[hot_rank] -= move
            loads[cold_rank] += move
            transfers.append(
                ReplicaTransfer(
                    logical_expert=best_e,
                    owner_rank=hot_rank,
                    owner_slot=best_e % home,
                    target_rank=cold_rank,
                    target_slot=target_slot,
                    target_physical=target_phys,
                )
            )

    new_peak = max(loads)
    if (
        not transfers
        or (peak - new_peak) < max(minimum_replica_rows, int(peak * minimum_load_reduction))
        or (
            0.0 < minimum_load_reduction < 0.4
            and not _traffic_work_is_useful(loads, transfers, peak, transfer_row_credit, minimum_replica_rows)
        )
    ):
        return _home_plan(counts_by_source, ep_size, num_experts, replica_slots_per_rank, granularity, owner_loads)

    alloc_tuples = [[(slot, rows) for slot, rows in entries if rows > 0] for entries in expert_allocs]
    dispatch_counts, source_quotas = _materialize_dispatch(
        counts_by_source, ep_size, num_experts, home, phys_per_rank, alloc_tuples
    )
    return LongcatReplicaPlan(
        ep_size=ep_size,
        num_experts=num_experts,
        home_experts=home,
        replica_slots_per_rank=replica_slots_per_rank,
        physical_experts=ep_size * phys_per_rank,
        granularity=granularity,
        home_only=False,
        transfers=tuple(transfers),
        dispatch_counts=dispatch_counts,
        source_quotas=source_quotas,
        logical_rank_loads=tuple(owner_loads),
        physical_rank_loads=tuple(loads),
    )


class LongcatReplicaPool:
    """Reusable rank-local physical weight staging buffer for LongcatFlashExperts."""

    def __init__(self) -> None:
        self._buffers: tuple[torch.Tensor, torch.Tensor] | None = None
        self._resident_tag: tuple[Any, ...] | None = None
        self._active_graphs: set[int] = set()

    def ensure(
        self,
        owner_weights: tuple[torch.Tensor, torch.Tensor],
        replica_slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate or reuse contiguous [home + replica_slots, ...] weight buffers."""
        w1, w2 = owner_weights
        shape1 = (w1.shape[0] + replica_slots, *w1.shape[1:])
        shape2 = (w2.shape[0] + replica_slots, *w2.shape[1:])
        if (
            self._buffers is not None
            and tuple(self._buffers[0].shape) == shape1
            and tuple(self._buffers[1].shape) == shape2
            and self._buffers[0].dtype == w1.dtype
            and self._buffers[0].device == w1.device
        ):
            return self._buffers
        self._buffers = (
            torch.empty(shape1, dtype=w1.dtype, device=w1.device),
            torch.empty(shape2, dtype=w2.dtype, device=w2.device),
        )
        self._resident_tag = None
        return self._buffers

    def close(self) -> None:
        """Release staged weight buffers."""
        self._buffers = None
        self._resident_tag = None
        self._active_graphs.clear()


class LongcatReplicaBinding:
    """Implements the _MegaMoeFunction weight_binding protocol for LongCat replicas.

    - Forward ``acquire`` copies home weights and prefetches guest weights via batched HCCL P2P.
    - Backward ``acquire`` reuses the resident guest weights from its own forward step when
      owner tensor versions are unchanged (avoiding redundant P2P weight prefetch in backward).
    - Backward ``gradients`` returns guest BF16 gradients to owners via batched HCCL P2P and
      accumulates into owner gradients (exact copy for whole-expert moves, FP32 sum for splits).
    """

    def __init__(
        self,
        pool: LongcatReplicaPool,
        plan: LongcatReplicaPlan,
        rank: int,
        group: Any,
        invocation_id: int,
    ) -> None:
        self._pool = pool
        self._plan = plan
        self._rank = rank
        self._group = group
        self._invocation_id = invocation_id
        self._acquired_in_forward = False

    def acquire(
        self,
        owner_weights: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous physical [home + replica_slots, ...] weights for kernel launch."""
        w1, w2 = owner_weights
        phys_w1, phys_w2 = self._pool.ensure(owner_weights, self._plan.replica_slots_per_rank)
        home = self._plan.home_experts
        tag = (self._invocation_id, w1._version, w2._version)  # pylint: disable=protected-access

        if self._acquired_in_forward and self._pool._resident_tag == tag:
            # Reuse weights prefetched by this exact invocation forward step (versions unchanged).
            return phys_w1, phys_w2

        phys_w1[:home].copy_(w1)
        phys_w2[:home].copy_(w2)

        ops: list[dist.P2POp] = []
        for xfer in self._plan.transfers:
            if self._rank == xfer.owner_rank:
                peer = _global_peer_rank(self._group, xfer.target_rank)
                ops.append(dist.P2POp(dist.isend, w1[xfer.owner_slot].contiguous(), peer, self._group))
                ops.append(dist.P2POp(dist.isend, w2[xfer.owner_slot].contiguous(), peer, self._group))
            elif self._rank == xfer.target_rank:
                peer = _global_peer_rank(self._group, xfer.owner_rank)
                ops.append(dist.P2POp(dist.irecv, phys_w1[xfer.target_slot], peer, self._group))
                ops.append(dist.P2POp(dist.irecv, phys_w2[xfer.target_slot], peer, self._group))
        _exchange_p2p(ops)

        self._pool._resident_tag = tag
        self._acquired_in_forward = True
        return phys_w1, phys_w2

    def release(self) -> None:
        """No-op lock release; buffers remain serialized on the active NPU stream."""

    def track_graph(self, ctx: Any) -> None:
        """Track active autograd graph reference."""
        self._pool._active_graphs.add(id(ctx))

    def release_graph(self, ctx: Any) -> None:
        """Release completed autograd graph reference."""
        self._pool._active_graphs.discard(id(ctx))

    def gradients(
        self,
        physical_gradients: tuple[torch.Tensor, torch.Tensor],
        owner_weights: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return guest weight gradients to owners and reduce onto owner home slices."""
        del owner_weights
        phys_dw1, phys_dw2 = physical_gradients
        home = self._plan.home_experts
        owner_dw1 = phys_dw1[:home].clone()
        owner_dw2 = phys_dw2[:home].clone()

        ops: list[dist.P2POp] = []
        incoming: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        outgoing: list[torch.Tensor] = []
        for xfer in self._plan.transfers:
            if self._rank == xfer.target_rank:
                peer = _global_peer_rank(self._group, xfer.owner_rank)
                g1 = phys_dw1[xfer.target_slot].contiguous()
                g2 = phys_dw2[xfer.target_slot].contiguous()
                outgoing.extend((g1, g2))
                ops.append(dist.P2POp(dist.isend, g1, peer, self._group))
                ops.append(dist.P2POp(dist.isend, g2, peer, self._group))
            elif self._rank == xfer.owner_rank:
                peer = _global_peer_rank(self._group, xfer.target_rank)
                r1 = torch.empty_like(owner_dw1[xfer.owner_slot])
                r2 = torch.empty_like(owner_dw2[xfer.owner_slot])
                incoming.append((xfer.owner_slot, r1, r2))
                ops.append(dist.P2POp(dist.irecv, r1, peer, self._group))
                ops.append(dist.P2POp(dist.irecv, r2, peer, self._group))
        _exchange_p2p(ops)

        if incoming:
            phys_per_rank = home + self._plan.replica_slots_per_rank
            grouped: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
            for slot, r1, r2 in incoming:
                grouped.setdefault(slot, []).append((r1, r2))
            for slot, parts in grouped.items():
                owner_phys = self._rank * phys_per_rank + slot
                owner_rows = sum(row[owner_phys] for row in self._plan.dispatch_counts)
                if owner_rows == 0 and len(parts) == 1:
                    # Whole-expert move: owner slot had 0 rows, so single guest gradient is exact.
                    owner_dw1[slot].copy_(parts[0][0])
                    owner_dw2[slot].copy_(parts[0][1])
                else:
                    # Split replica: accumulate owner + all guest partials in FP32.
                    acc1 = owner_dw1[slot].float() if owner_rows > 0 else torch.zeros_like(owner_dw1[slot], dtype=torch.float32)
                    acc2 = owner_dw2[slot].float() if owner_rows > 0 else torch.zeros_like(owner_dw2[slot], dtype=torch.float32)
                    for r1, r2 in parts:
                        acc1.add_(r1.float())
                        acc2.add_(r2.float())
                    owner_dw1[slot].copy_(acc1.to(owner_dw1.dtype))
                    owner_dw2[slot].copy_(acc2.to(owner_dw2.dtype))

        return owner_dw1, owner_dw2
