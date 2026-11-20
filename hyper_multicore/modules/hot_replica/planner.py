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
from typing import Any

import torch

from ..base_planner import BasePlanner, ExpertPlacement, PlacementRequest
from .placement import (
    bounded_move_pays, build_placement, placement_is_profitable, placement_receive_limit,
)


class HotReplicaPlanner(BasePlanner):
    """Choose execution slots without allocating either backend's workspaces."""

    def __init__(self, *, source_token_bound: int | None = None) -> None:
        """Set the optional per-rank training row bound used for workspace reservation."""
        super().__init__(source_token_bound=source_token_bound)
        self._rejected_routes: OrderedDict[tuple[Any, ...], None] = OrderedDict()

    @torch.no_grad()
    def plan(self, request: PlacementRequest) -> ExpertPlacement | None:
        """Return a placement if saved work repays movement or prevents heap growth."""
        if request.ordinary_receive_capacity is not None:
            return self._plan_reserved(request, self._build)
        return self._plan_unreserved(request, self._build)

    def _plan_unreserved(self, request, build):
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

        def _pays(after, transfers=None):
            return placement_is_profitable(
                maximum, after, hidden, request.intermediate_size, request.cube_cores, weight_elements * size,
                projection_flops=12 if request.training else 6, ep_size=ranks, local_experts=local,
                gradient_bytes=weight_elements * 4 if request.training else 0,
                backward_prefetch_weight_bytes=weight_elements * size if request.training else 0,
                transfer_experts=transfers,
            )

        if not _pays(receive_limit):
            return None
        placement = build(request, receive_limit)
        if placement is None or _pays(placement.receive_capacity, placement.transfer_experts):
            return placement
        if key is not None:
            self._rejected_routes[key] = None
            if len(self._rejected_routes) > 32:
                self._rejected_routes.popitem(last=False)
        return None

    @staticmethod
    def _plan_reserved(request, build):
        """Maintain the reservation admission policy across quota implementations."""
        maximum, floor = request.maximum_received, request.ordinary_receive_capacity
        ranks, experts = request.counts_by_source.shape
        target = (request.total_rows + ranks - 1) // ranks
        if maximum <= target:
            return None
        local = experts // ranks
        hidden, size = request.hidden.shape[-1], request.hidden.element_size()
        receive_limit = placement_receive_limit(target, ranks, local, request.ids.shape[-1], hidden, size)

        def _worth_moving(after):
            return bounded_move_pays(
                maximum, after, hidden, request.intermediate_size, request.cube_cores, ranks, local, size,
                training=request.training, replay=request.checkpointing,
                apply_probs_before_fc2=request.apply_probs_before_fc2,
            )

        must_balance = maximum > floor
        if not must_balance and (maximum <= receive_limit or not _worth_moving(target)):
            return None
        placement = build(request, target)
        if placement is not None and (
            (must_balance and placement.receive_capacity <= floor) or _worth_moving(placement.receive_capacity)
        ):
            return placement
        return None

    @staticmethod
    def _build(request: PlacementRequest, target: int) -> ExpertPlacement | None:
        return build_placement(
            request.hidden, request.ids, request.counts_by_source,
            target, request.rank, source_inverse=request.source_inverse,
        )
