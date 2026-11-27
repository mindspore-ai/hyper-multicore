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
"""Inputs and quotas for experimental expert placement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementRequest:
    """Logical routed counts and execution geometry for an ordered EP group.

    `ordinary_receive_capacity` exposes the committed Push workspace rows to custom planners.
    """

    counts_by_source: tuple[tuple[int, ...], ...]
    maximum_received: int
    total_rows: int
    rank: int
    hidden_size: int
    intermediate_size: int
    cube_cores: int
    training: bool
    checkpointing: bool = False
    ordinary_receive_capacity: int | None = None
    apply_probs_before_fc2: bool = True
    rank_nodes: tuple[int, ...] = ()
    dispatch_mode: str = "push"
    transport_engine: str | None = "mte"


@dataclass(frozen=True)
class ExpertPlacement:
    """Plan physical quotas without binding tokens, weights or device resources.

    Each row of copies_by_rank has the same guest width. Entries name global
    experts or use -1 for an empty slot. In dispatch_counts, each source row
    specifies quotas in destination-rank order, with that rank's home experts
    followed by its guest slots. Empty slots must receive zero rows.
    """

    copies_by_rank: tuple[tuple[int, ...], ...]
    dispatch_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "copies_by_rank", tuple(tuple(row) for row in self.copies_by_rank))
        object.__setattr__(self, "dispatch_counts", tuple(tuple(row) for row in self.dispatch_counts))


class BasePlanner(ABC):
    """Choose expert placement from an EP group's already exchanged routing counts."""

    def configuration_key(self) -> str:
        """Return a stable key for decision settings shared across EP ranks.

        Parameterized policies override this method. Exclude per-rank and mutable execution state.
        """
        return ""

    @abstractmethod
    def plan(self, request: PlacementRequest) -> ExpertPlacement | None:
        """Return identical quotas on every EP rank, or None on every rank."""
