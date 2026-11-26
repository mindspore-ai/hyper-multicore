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
"""Bounded whole/64-row expert fragments preserving global MX quantization groups."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ._dispatch import intersect_source_intervals
from .capacity import _integer
from .native_host import aligned_dispatch
from .plan import ExpertExecutionPlan, physical_replica_slots


@dataclass(frozen=True)
class AlignedExpertExecutionPlan(ExpertExecutionPlan):
    """Store each physical slot's interval in its original global expert row order."""

    intervals: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Own fragment boundaries alongside the immutable source quota table."""
        super().__post_init__()
        object.__setattr__(self, "intervals", tuple(tuple(span) for span in self.intervals))

    @cached_property
    def _source_slot_order(self) -> tuple[int, ...]:
        # Physical rank order does not imply fragment order: a guest may precede its owner.
        return tuple(sorted((slot for slot, expert in enumerate(self.physical_to_logical) if expert >= 0),
                            key=lambda slot: (self.physical_to_logical[slot], self.intervals[slot][0], slot)))

    def validate(self) -> None:
        """Verify conservation and exact interval intersections independently of placement."""
        super().validate()
        if len(self.intervals) != self.config.physical_experts:
            raise ValueError("Aligned intervals must cover every physical slot")
        for start, stop in self.intervals:
            _integer(start, "Fragment start", 0)
            _integer(stop, "Fragment stop", start)
        intervals_by_expert = [[] for _ in range(self.config.num_experts)]
        for slot, expert in enumerate(self.physical_to_logical):
            start, stop = self.intervals[slot]
            if expert >= 0 and stop > start:
                intervals_by_expert[expert].append(((start, stop), slot))
        for expert, intervals in enumerate(intervals_by_expert):
            end = 0
            for (start, stop), slot in sorted(intervals):
                if start != end or start % 64 or stop <= start:
                    raise ValueError("Fragments must be contiguous with 64-row internal boundaries")
                prefix = 0
                for source, logical in enumerate(self.logical_counts):
                    count = logical[expert]
                    expected = max(0, min(prefix + count, stop) - max(prefix, start))
                    if self.dispatch_counts[source][slot] != expected:
                        raise ValueError("Fragment quotas must intersect original global source intervals")
                    prefix += count
                end = stop
            if end != sum(row[expert] for row in self.logical_counts):
                raise ValueError("Fragments must cover the original expert exactly")


def _materialize_aligned(matrix, config, slots, spans):
    """Render exact source intersections for an already chosen placement."""
    frozen_slots, frozen_spans = tuple(map(tuple, slots)), tuple(map(tuple, spans))
    native_dispatch = aligned_dispatch(matrix, frozen_slots, frozen_spans, config.ep_size,
                                        config.home_experts, config.slots_per_rank)
    if native_dispatch is not None:
        return AlignedExpertExecutionPlan(config, matrix, frozen_slots, native_dispatch, frozen_spans)
    if sum(map(sum, matrix)) <= np.iinfo(np.int32).max:
        counts = np.asarray(matrix, dtype=np.int32)
        owners = np.asarray(frozen_slots, dtype=np.int32).reshape(-1)
        starts, ends = np.asarray(frozen_spans, dtype=np.int32).T
        dispatch = intersect_source_intervals(counts, owners, starts, ends).tolist()
        plan = AlignedExpertExecutionPlan(config, matrix, frozen_slots, dispatch, frozen_spans)
        plan.validate()
        return plan
    # CPU-only plans can represent counts beyond the native INT32 quota ABI.
    dispatch = [[0] * config.physical_experts for _ in matrix]
    for slot, expert in enumerate(e for row in slots for e in row):
        if expert < 0:
            continue
        start, stop = spans[slot]
        prefix = 0
        for source, row in enumerate(matrix):
            count = row[expert]
            dispatch[source][slot] = max(0, min(prefix + count, stop) - max(prefix, start))
            prefix += count
    plan = AlignedExpertExecutionPlan(config, matrix, frozen_slots, tuple(map(tuple, dispatch)), frozen_spans)
    plan.validate()
    return plan


def materialize_aligned_moves(matrix, config, totals, moves):
    """Lower the shared search's fragments in original global K64 row order."""
    home = config.home_experts
    guest_rows = [{} for _ in range(config.ep_size)]
    guest_spans = {}
    for _, target, expert, start, stop in moves:
        guest_rows[target][expert] = start, stop
        guest_spans.setdefault(expert, []).append((start, stop))
    slots = tuple(physical_replica_slots(config, rank, sorted(guests)) for rank, guests in enumerate(guest_rows))
    owner_spans = {}
    for expert, guests in guest_spans.items():
        start = 0
        for left, right in sorted(guests):
            if left != start:
                break
            start = right
        owner_spans[expert] = start, start + totals[expert] - sum(right - left for left, right in guests)
    spans = []
    for rank, row in enumerate(slots):
        for slot, expert in enumerate(row):
            if slot < home:
                spans.append(owner_spans.get(expert, (0, totals[expert])))
            else:
                spans.append(guest_rows[rank].get(expert, (0, 0)))
    return _materialize_aligned(matrix, config, slots, spans)
