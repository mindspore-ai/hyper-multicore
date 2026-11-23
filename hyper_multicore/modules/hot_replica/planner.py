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

from collections import OrderedDict
from sys import float_info
from typing import Any

import torch

from ..base_planner import BasePlanner, ExpertPlacement, PlacementRequest
from .placement import build_placement, placement_is_profitable, placement_receive_limit


class HotReplicaPlanner(BasePlanner):
    """Choose execution slots without allocating either backend's workspaces."""

    def __init__(
        self, *, replica_slots_per_rank: int | None = None,
        cross_node_cost: float | None = None,
    ) -> None:
        """Set the guest slot budget and relative inter-node replica charge."""
        if replica_slots_per_rank is not None and (
            type(replica_slots_per_rank) is not int or replica_slots_per_rank <= 0
        ):
            raise ValueError("replica_slots_per_rank must be a positive integer")
        if cross_node_cost is not None:
            if type(cross_node_cost) not in (int, float) or not 0 < cross_node_cost <= float_info.max:
                raise ValueError("cross_node_cost must be a finite positive multiplier")
            cross_node_cost = float(cross_node_cost)
        self._replica_slots_per_rank = replica_slots_per_rank
        self._cross_node_cost = cross_node_cost
        self._rejected_routes: OrderedDict[tuple[Any, ...], None] = OrderedDict()

    @property
    def replica_slots_per_rank(self) -> int | None:
        """Return the optional cap on active guest experts on each rank."""
        return self._replica_slots_per_rank

    @property
    def cross_node_cost(self) -> float | None:
        """Return the optional cross-node replica transfer multiplier."""
        return self._cross_node_cost

    def configuration_key(self) -> str:
        """Include the explicit limits in the configuration agreed by EP ranks."""
        settings = []
        if self.replica_slots_per_rank is not None:
            settings.append(f"replica_slots_per_rank={self.replica_slots_per_rank}")
        if self.cross_node_cost is not None:
            settings.append(f"cross_node_cost={self.cross_node_cost}")
        return ",".join(settings)

    @torch.no_grad()
    def plan(self, request: PlacementRequest) -> ExpertPlacement | None:
        """Return a placement if saved work repays movement."""
        return self._plan_by_cost(request, self._build)

    def _plan_by_cost(self, request, build):
        """Keep the observed movement break-even while sharing geometry preparation."""
        counts = request.counts_by_source
        ranks, experts = counts.shape
        local = experts // ranks
        hidden = request.hidden.shape[-1]
        size = request.hidden.element_size()
        target = (request.total_rows + ranks - 1) // ranks
        maximum = request.maximum_received
        receive_limit = placement_receive_limit(target, ranks, local, request.ids.shape[-1], hidden, size)
        key = request.count_key
        if key is not None and key in self._rejected_routes:
            self._rejected_routes.move_to_end(key)
            return None
        weight_elements = 3 * local * hidden * request.intermediate_size

        def estimate(calculator, after, transfers=None):
            return calculator(
                maximum, after, hidden, request.intermediate_size, request.cube_cores, weight_elements * size,
                projection_flops=12 if request.training else 6, ep_size=ranks, local_experts=local,
                gradient_bytes=weight_elements * 4 if request.training else 0,
                backward_prefetch_weight_bytes=weight_elements * size if request.training else 0,
                transfer_experts=transfers,
            )

        if not estimate(placement_is_profitable, receive_limit):
            return None
        placement = build(request, receive_limit)
        if placement is None:
            return None
        if estimate(placement_is_profitable, placement.receive_capacity, placement.transfer_experts):
            return placement
        if key is not None:
            self._rejected_routes[key] = None
            if len(self._rejected_routes) > 32:
                self._rejected_routes.popitem(last=False)
        return None

    def _build(self, request: PlacementRequest, target: int, **options) -> ExpertPlacement | None:
        return build_placement(
            request.hidden, request.ids, request.counts_by_source,
            target, request.rank, source_inverse=request.source_inverse,
            replica_slots_per_rank=self.replica_slots_per_rank,
            **options,
        )
