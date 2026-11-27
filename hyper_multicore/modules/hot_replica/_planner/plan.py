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
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import chain

from .capacity import ExpertReplicaConfig, _integer


def physical_replica_slots(config: ExpertReplicaConfig, rank: int, guests: Sequence[int] = ()) -> tuple[int, ...]:
    """Place the ordered owner slice and guest prefix in the common slot layout."""
    if len(guests) > config.replica_slots_per_rank:
        raise ValueError("replica layout exceeds the physical guest slot cap")
    home = config.home_experts
    owners = tuple(range(rank * home, (rank + 1) * home))
    return owners + tuple(guests) + (-1,) * (config.replica_slots_per_rank - len(guests))


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
    def dispatch_counts_bytes(self) -> bytes:
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
    def physical_to_logical(self) -> tuple[int, ...]:
        """Return flattened physical-slot ownership, using -1 for empty slots."""
        return tuple(expert for row in self.slot_to_logical for expert in row)

    @cached_property
    def replica_slots_used(self) -> tuple[int, ...]:
        """Return actual guest counts per rank, independently of the reserved slot cap."""
        home = self.config.home_experts
        return tuple(sum(expert >= 0 for expert in row[home:]) for row in self.slot_to_logical)

    @property
    def compact_borrowed_width(self) -> int:
        """Round active prefix slots without exceeding the configured budget."""
        used = max(self.replica_slots_used)
        return min(self.config.replica_slots_per_rank, 1 << (used - 1).bit_length()) if used else 0

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
