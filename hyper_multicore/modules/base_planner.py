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
"""Per-invocation inputs, result and policy interface for expert placement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from .mega_moe.route_metadata import RouteMetadata


@dataclass(frozen=True)
class PlacementRequest:
    """Counts and geometry shared by either EP executor."""

    hidden: torch.Tensor
    ids: torch.Tensor
    counts_by_source: torch.Tensor
    source_inverse: torch.Tensor | None
    maximum_received: int
    total_rows: int
    rank: int
    intermediate_size: int
    cube_cores: int
    training: bool
    checkpointing: bool = False
    ordinary_receive_capacity: int | None = None
    apply_probs_before_fc2: bool = True
    count_key: tuple[Any, ...] | None = None
    rank_nodes: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExpertPlacement:
    """Keep the virtual routing and weight-copy ownership of one invocation.

    Attributes:
        virtual_ids: Selected virtual expert IDs, retaining original token and Top-K order.
        counts_by_source: Counts from every source for all home and borrowed expert slots.
        copies: Device plan whose rows identify the global experts borrowed by each rank.
    """

    virtual_ids: torch.Tensor
    counts_by_source: torch.Tensor
    copies: torch.Tensor
    borrowed_capacity: int
    receive_capacity: int
    local_receive_capacity: int | None = None
    transfer_experts: int | None = None
    published_slots: int | None = None
    copies_by_rank: tuple[tuple[int, ...], ...] = ()
    route_metadata: RouteMetadata | None = None
    owner_rows: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the shared layout without reading any device tensor values."""
        self._validate_capacities()
        local = self.local_receive_capacity
        if local is not None and (type(local) is not int or not 0 <= local <= self.receive_capacity):
            raise ValueError("Local receive capacity must be an integer within the group receive capacity")
        self._validate_topology()
        self._validate_tensor_storage()

    def _validate_capacities(self) -> None:
        """Validate scalar limits without reading a device tensor."""
        if (
            not isinstance(self.borrowed_capacity, int)
            or isinstance(self.borrowed_capacity, bool)
            or self.borrowed_capacity < 0
            or not isinstance(self.receive_capacity, int)
            or isinstance(self.receive_capacity, bool)
            or self.receive_capacity < 0
        ):
            raise ValueError("Expert placement capacities must be nonnegative integers")

    def received_rows(self, rank: int) -> int:
        """Return the rows destined for this rank across all source ranks."""
        if self.local_receive_capacity is not None:
            return self.local_receive_capacity
        ranks, experts = self.counts_by_source.shape
        width = experts // ranks
        return int(self.counts_by_source.narrow(1, rank * width, width).sum().item())

    def _validate_topology(self) -> None:
        """Check the EP and borrowed-slot dimensions."""
        ranks = self.counts_by_source.shape[0] if self.counts_by_source.ndim == 2 else 0
        if (
            ranks < 1
            or self.counts_by_source.ndim != 2
            or self.counts_by_source.shape[1] % ranks
        ):
            raise ValueError("Expert placement counts must describe the ordered EP group")
        local = self.counts_by_source.shape[1] // ranks - self.borrowed_capacity
        if local < 1 or self.borrowed_capacity > local:
            raise ValueError("Expert borrowed capacity exceeds its home expert bank")
        if tuple(self.copies.shape) != (ranks, self.borrowed_capacity):
            raise ValueError("Expert copy tables must match the logical borrowed capacity")

    def _validate_tensor_storage(self) -> None:
        """Require integer metadata on the same device."""
        if (
            self.virtual_ids.dtype not in (torch.int32, torch.int64)
            or self.counts_by_source.dtype not in (torch.int32, torch.int64)
            or self.copies.dtype != torch.int32
        ):
            raise TypeError("Expert placement requires integer routes/counts and INT32 copies")
        if self.virtual_ids.device != self.counts_by_source.device or self.copies.device != self.virtual_ids.device:
            raise ValueError("Expert placement tensors must share the source device")


class BasePlanner(ABC):
    """Choose expert placement from an EP group's already exchanged routing counts."""

    def configuration_key(self) -> str:
        """Return a stable key for decision settings that must agree across EP ranks.

        Parameterized policies override this method. Exclude per-rank and mutable execution state.
        The executor records the planner type separately.
        """
        return ""

    @abstractmethod
    def plan(self, request: PlacementRequest) -> ExpertPlacement | None:
        """Return a group-consistent placement or None to keep ordinary routing."""
