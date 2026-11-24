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
from typing import Any, Literal

import torch

from ..base_planner import BasePlanner, ExpertPlacement, PlacementRequest
from ._planner.receive import ReplicaCost, receive_limit
from .placement import build_placement, placement_benefit_seconds


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
        self._rejected_routes: OrderedDict[tuple[Any, ...], None] = OrderedDict()

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
        return self._plan_by_cost(request, self._build)

    def _plan_by_cost(self, request, build):
        """Keep the observed movement break-even while sharing geometry preparation."""
        counts = request.counts_by_source
        ranks, experts = counts.shape
        local = experts // ranks
        hidden = request.hidden.shape[-1]
        size = request.hidden.element_size()
        maximum = request.maximum_received
        limit = receive_limit(self.receive_capacity_percent, request.total_rows, ranks)
        mandatory = limit is not None and maximum > limit
        resident = request.ordinary_receive_capacity
        key = (request.count_key, request.rank_nodes, resident) if request.count_key is not None else None
        if not mandatory and key is not None and key in self._rejected_routes:
            self._rejected_routes.move_to_end(key)
            return None
        weight_elements = 3 * local * hidden * request.intermediate_size

        def estimate(after, transfers):
            return placement_benefit_seconds(
                maximum, after, hidden, request.intermediate_size, request.cube_cores, weight_elements * size,
                projection_flops=12 if request.training else 6, ep_size=ranks, local_experts=local,
                gradient_bytes=weight_elements * 4 if request.training else 0,
                backward_prefetch_weight_bytes=weight_elements * size if request.training else 0,
                transfer_experts=transfers,
            )

        if ReplicaCost().keep_home(maximum, request.total_rows, ranks, self.receive_capacity_percent, score=estimate, resident=resident):
            return None
        placement = build(request, limit, score=estimate, receive_capacity_percent=self.receive_capacity_percent,
                          resident=resident)
        if placement is not None:
            return placement
        if mandatory:
            raise ValueError(f"No placement found within receive limit {limit} rows")
        if key is not None:
            self._rejected_routes[key] = None
            if len(self._rejected_routes) > 32:
                self._rejected_routes.popitem(last=False)
        return None

    def _build(self, request: PlacementRequest, target: int | None, **options) -> ExpertPlacement | None:
        return build_placement(
            request.hidden, request.ids, request.counts_by_source,
            target, request.rank, source_inverse=request.source_inverse,
            replica_slots_per_rank=self.replica_slots_per_rank,
            nodes=request.rank_nodes, cross_node_cost=self.cross_node_cost, **options,
        )
