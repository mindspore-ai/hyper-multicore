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
"""Immutable logical ownership and physical dispatch contracts."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import cached_property
from itertools import chain

from .capacity import ExpertReplicaConfig, _integer


def _destination_counts_from_dispatch(
    dispatch: tuple[tuple[int, ...], ...], config: ExpertReplicaConfig,
) -> tuple[tuple[int, ...], ...]:
    """Reduce physical dispatch columns without relying on any cached quotas."""
    width = config.slots_per_rank
    if any(len(row) != config.physical_experts for row in dispatch):
        raise ValueError("dispatch row does not match physical expert count")
    totals = tuple(map(sum, zip(*dispatch)))
    return tuple(totals[rank * width:(rank + 1) * width] for rank in range(config.ep_size))


@dataclass(frozen=True)
class ExpertReplicaTransfer:
    """One original owner supplying one temporary physical expert slot."""

    logical_expert: int
    owner_rank: int
    owner_slot: int
    target_rank: int
    target_slot: int


@dataclass(frozen=True)
class ExpertExecutionPlan:
    """Exact source quotas and bounded physical slots for one forward graph."""

    config: ExpertReplicaConfig
    logical_counts: tuple[tuple[int, ...], ...]
    slot_to_logical: tuple[tuple[int, ...], ...]
    dispatch_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """Own immutable rows so cached derived metadata cannot become stale."""
        for name in ("logical_counts", "slot_to_logical", "dispatch_counts"):
            object.__setattr__(self, name, tuple(tuple(row) for row in getattr(self, name)))

    @cached_property
    def _dispatch_counts_bytes(self) -> bytes:
        """Encode the immutable count table once for device upload and host offsets."""
        size = self.config.ep_size * self.config.physical_experts
        return struct.pack(f"<{size}i", *chain.from_iterable(self.dispatch_counts))

    @cached_property
    def destination_counts(self) -> tuple[tuple[int, ...], ...]:
        """Return per-slot work for every destination without device readback."""
        return _destination_counts_from_dispatch(self.dispatch_counts, self.config)

    @cached_property
    def destination_loads(self) -> tuple[int, ...]:
        """Return total real work assigned to each execution rank."""
        return tuple(sum(row) for row in self.destination_counts)

    @cached_property
    def _source_slot_order(self) -> tuple[int, ...]:
        """Share stable expert/slot ordering without materializing every source row."""
        return tuple(slot for _, slot in sorted((expert, slot)
                     for slot, expert in enumerate(self.physical_to_logical) if expert >= 0))

    @cached_property
    def source_runs(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return stable logical-expert runs as physical slot and source quota."""
        return tuple(self.source_runs_for(rank) for rank in range(len(self.dispatch_counts)))

    def source_runs_for(self, rank: int) -> tuple[tuple[int, int], ...]:
        """Return one source row without materializing runs for every source rank."""
        source = self.dispatch_counts[rank]
        return tuple((slot, source[slot]) for slot in self._source_slot_order)

    @cached_property
    def physical_to_logical(self) -> tuple[int, ...]:
        """Return flattened physical-slot ownership, using -1 for empty slots."""
        return tuple(expert for row in self.slot_to_logical for expert in row)

    @cached_property
    def replica_slots_used(self) -> tuple[int, ...]:
        """Return actual guest counts per rank, independently of the reserved slot cap."""
        home = self.config.home_experts
        return tuple(sum(expert >= 0 for expert in row[home:]) for row in self.slot_to_logical)

    @cached_property
    def transfers(self) -> tuple[ExpertReplicaTransfer, ...]:
        """Return transfers in globally identical target/slot order."""
        home = self.config.home_experts
        return tuple(ExpertReplicaTransfer(expert, expert // home, expert % home, rank, slot)
                     for rank, row in enumerate(self.slot_to_logical)
                     for slot, expert in enumerate(row) if slot >= home and expert >= 0)

    def validate(self) -> None:
        """Check per-source conservation and physical ownership invariants."""
        config = self.config
        if len(self.logical_counts) != config.ep_size or len(self.slot_to_logical) != config.ep_size:
            raise ValueError("plan must contain one logical and physical row per rank")
        self._validate_slots()
        if len(self.dispatch_counts) != config.ep_size:
            raise ValueError("plan must contain one dispatch row per source rank")
        for source, row in enumerate(self.dispatch_counts):
            self._validate_source(source, row)
        if ("destination_counts" in self.__dict__
                and self.destination_counts != _destination_counts_from_dispatch(self.dispatch_counts, config)):
            raise ValueError("cached destination quotas must match the physical dispatch")

    def _validate_slots(self) -> None:
        """Validate fixed home ownership and unique nonempty slots on every rank."""
        config = self.config
        for rank, row in enumerate(self.slot_to_logical):
            expected = tuple(range(rank * config.home_experts, (rank + 1) * config.home_experts))
            active = [expert for expert in row if expert != -1]
            if (len(row) != config.slots_per_rank or row[:config.home_experts] != expected
                    or len(active) != len(set(active)) or any(not 0 <= e < config.num_experts for e in active)):
                raise ValueError("plan contains invalid or duplicate physical expert ownership")

    def _validate_source(self, source: int, row: tuple[int, ...]) -> None:
        """Reconstruct one logical row independently from the physical dispatch."""
        if len(row) != self.config.physical_experts:
            raise ValueError("dispatch row does not match physical expert count")
        reconstructed = [0] * self.config.num_experts
        for expert, count in zip(self.physical_to_logical, row):
            _integer(count, "physical dispatch count", 0)
            if expert == -1 and count:
                raise ValueError("invalid physical dispatch count")
            if count:
                reconstructed[expert] += count
        if tuple(reconstructed) != self.logical_counts[source]:
            raise ValueError("physical dispatch must conserve every source/logical-expert count")
