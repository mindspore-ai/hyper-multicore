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
"""Plan one expert placement over validated EP counts."""
from __future__ import annotations

from sys import float_info
from typing import Literal

import torch

from ..base_planner import BasePlanner, ExpertPlacement, PlacementRequest
from ._planner.receive import SourceQuota, bf16_cost, receive_limit
from .placement import build_placement, replica_slot_budget


class HotReplicaPlanner(BasePlanner):
    """Choose execution slots without allocating either backend's workspaces."""

    def __init__(
        self, *, replica_slots_per_rank: int | None = None,
        receive_capacity_percent: int | Literal["auto"] | None = "auto",
        cross_node_cost: float | None = None,
    ) -> None:
        """Set optional guest and receive limits and a relative inter-node replica charge."""
        if replica_slots_per_rank is not None and (
            type(replica_slots_per_rank) is not int or replica_slots_per_rank <= 0
        ):
            raise ValueError("replica_slots_per_rank must be a positive integer")
        if receive_capacity_percent is not None and receive_capacity_percent != "auto" and (
            type(receive_capacity_percent) is not int or receive_capacity_percent < 100
        ):
            raise ValueError('receive_capacity_percent must be an integer at least 100 or "auto"')
        if cross_node_cost is not None:
            if type(cross_node_cost) not in (int, float) or not 0 < cross_node_cost <= float_info.max:
                raise ValueError("cross_node_cost must be a finite positive multiplier")
            cross_node_cost = float(cross_node_cost)
        self._replica_slots_per_rank = replica_slots_per_rank
        self._receive_capacity_percent = "auto" if receive_capacity_percent is None else receive_capacity_percent
        self._cross_node_cost = cross_node_cost

    @property
    def replica_slots_per_rank(self) -> int | None:
        """Return the optional cap on active guest experts on each rank."""
        return self._replica_slots_per_rank

    @property
    def receive_capacity_percent(self) -> int | Literal["auto"]:
        """Return the configured receive policy relative to the actual routed rows."""
        return self._receive_capacity_percent

    @property
    def cross_node_cost(self) -> float | None:
        """Return the optional cross-node replica transfer multiplier."""
        return self._cross_node_cost

    def configuration_key(self) -> str:
        """Include the explicit limits in the configuration agreed by EP ranks."""
        settings = []
        if self.replica_slots_per_rank is not None:
            settings.append(f"replica_slots_per_rank={self.replica_slots_per_rank}")
        if self.receive_capacity_percent != "auto":
            settings.append(f"receive_capacity_percent={self.receive_capacity_percent}")
        if self.cross_node_cost is not None:
            settings.append(f"cross_node_cost={self.cross_node_cost}")
        return ",".join(settings)

    @torch.no_grad()
    def plan(self, request: PlacementRequest) -> ExpertPlacement | None:
        """Return a placement if saved work repays movement."""
        ranks, experts = request.counts_by_source.shape
        prices = bf16_cost(
            request.hidden.shape[-1], request.intermediate_size, request.cube_cores, request.training,
            dispatch_mode=request.dispatch_mode, transport_engine=request.transport_engine,
        )
        limit = receive_limit(self.receive_capacity_percent, request.total_rows, ranks)
        resident = request.ordinary_receive_capacity
        if prices.keep_home(request.maximum_received, request.total_rows, ranks, limit=limit, resident=resident):
            return None
        budget = replica_slot_budget(experts // ranks, self.replica_slots_per_rank, request.dispatch_mode)
        return build_placement(
            request.hidden, request.ids, request.counts_by_source, None, request.rank,
            source_inverse=request.source_inverse, replica_slots_per_rank=budget, prices=prices,
            source_quota=SourceQuota.LOCAL, receive_capacity_percent=self.receive_capacity_percent,
            nodes=request.rank_nodes, cross_node_cost=self.cross_node_cost, resident=resident,
        )
