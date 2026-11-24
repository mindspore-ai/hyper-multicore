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

import numpy as np

from ..base_planner import BasePlanner, ExpertPlacement, PlacementRequest
from ._planner.capacity import ExpertReplicaConfig
from ._planner.receive import bf16_cost, receive_limit, select_replica_plan


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

    def _policy(self, hidden_size, intermediate_size, cube_cores, training, dispatch_mode,
                transport_engine, peak, total, ranks, resident):
        """Use one cost, capacity and home-admission rule for both planner callers."""
        prices = bf16_cost(hidden_size, intermediate_size, cube_cores, training,
                           dispatch_mode=dispatch_mode, transport_engine=transport_engine)
        limit = receive_limit(self.receive_capacity_percent, total, ranks)
        return prices, limit, prices.keep_home(peak, total, ranks, limit=limit, resident=resident)

    def plan(self, request: PlacementRequest) -> ExpertPlacement | None:
        """Return host quotas if saved work repays movement."""
        counts = np.asarray(request.counts_by_source)
        if counts.ndim != 2 or counts.size == 0:
            raise ValueError("HotReplica counts must describe an ordered EP topology")
        ranks, experts = counts.shape
        if not 0 <= request.rank < ranks or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("HotReplica requires integer counts and a valid EP rank")
        if counts.min() < 0 or counts.max() > 2**31 - 1 or counts.sum(dtype=np.int64) > 2**31 - 1:
            raise ValueError("HotReplica counts must be nonnegative and within the INT32 planner limit")
        budget = replica_slot_budget(experts // ranks, self.replica_slots_per_rank, request.dispatch_mode)
        config = ExpertReplicaConfig(experts, ranks, budget)
        resident = request.ordinary_receive_capacity
        prices, limit, keep_home = self._policy(
            request.hidden_size, request.intermediate_size, request.cube_cores, request.training,
            request.dispatch_mode, request.transport_engine, request.maximum_received, request.total_rows,
            ranks, resident,
        )
        if keep_home:
            return None
        plan = select_replica_plan(counts, config, prices=prices, limit=limit, resident=resident,
                                   nodes=request.rank_nodes, cross_node_cost=self.cross_node_cost)
        if plan is None:
            return None
        return as_placement(plan)



def replica_slot_budget(home: int, requested: int | None, dispatch_mode: str) -> int:
    """Return the effective slot cap and reject out-of-range explicit requests."""
    limit = min(home, 6) if dispatch_mode == "pull" else home
    if requested is not None and requested > limit:
        raise ValueError(f"Requested {requested} guest slots, but {dispatch_mode} supports at most {limit} per rank")
    return limit if requested is None else requested


def as_placement(plan) -> ExpertPlacement:
    """Expose compact quotas without generating per-token device metadata."""
    config = plan.config
    stop = config.home_experts + plan.compact_borrowed_width
    stride = config.slots_per_rank
    copies = tuple(row[config.home_experts:stop] for row in plan.slot_to_logical)
    dispatch = tuple(
        tuple(count for offset in range(0, len(row), stride) for count in row[offset:offset + stop])
        for row in plan.dispatch_counts
    )
    return ExpertPlacement(copies, dispatch)
