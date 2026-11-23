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
"""Internal placement and weight binding for the existing BF16 execution backend."""

from __future__ import annotations

from collections.abc import Mapping
import math
import weakref
from typing import Any, TYPE_CHECKING

import torch

from hyper_multicore.modules.mega_moe.heap_manager import _WorkspaceLayout, _workspace_layout
from hyper_multicore.modules.mega_moe.spec import _align_capacity

from .checkpoint import is_checkpointing
from .placement import ExpertPlacement, placement_receive_limit
from ..base_planner import BasePlanner, PlacementRequest
from .shmem_transfer import ExpertWeightBinding, SharedExpertPool

if TYPE_CHECKING:
    from hyper_multicore.modules.mega_moe.backends.bf16.resources import BF16Resources


def _bounded_receive(
    specification: Mapping[str, Any], source: int, dtype: torch.dtype, cube_cores: int,
) -> tuple[int, int]:
    """Size ordinary storage at the checkpoint movement break-even point."""
    ep = specification["ep_size"]
    local = specification["num_experts"] // ep
    hidden = specification["hidden_size"]
    intermediate = specification["intermediate_size"]
    top_k = specification["top_k"]
    average = source * top_k
    element_size = dtype.itemsize
    virtual = placement_receive_limit(average, ep, local, top_k, hidden, element_size)
    weight = 3 * local * hidden * intermediate
    weights = weight * element_size
    gradients = weight * 4

    def _excess(flops: int, gradient_bytes: int, restore_bytes: int = 0) -> int:
        staging = (3 * weights + 2 * gradient_bytes + restore_bytes) / 4e11
        peer = (ep - 1) * (weights + gradient_bytes) / (local * 2e11)
        return math.ceil((8e-4 + staging + peer) * cube_cores * 4e12 / (flops * hidden * intermediate))

    ordinary = average + max(_excess(12, gradients, 2 * weights), _excess(6, 0))
    upper = source * ep * min(local, top_k)
    return _align_capacity(min(upper, max(ordinary, virtual))), virtual


def workspace_reservations(
    specifications: tuple[Mapping[str, Any], ...], dtype: torch.dtype, cube_cores: int,
) -> tuple[tuple[tuple[Mapping[str, Any], _WorkspaceLayout], ...], int]:
    """Describe each ordinary layout and budget the later virtual binds once."""
    reservations = []
    banks = set()
    headroom = 0
    for specification in specifications:
        bound = specification.get("source_token_bound")
        if bound is None:
            reservations.append((specification, _workspace_layout(specification, dtype)))
            continue
        source = _align_capacity(bound)
        native, virtual = _bounded_receive(specification, source, dtype, cube_cores)
        reservations.append((specification, _workspace_layout(specification, dtype, native, source)))
        virtual_spec = {**specification, "num_experts": 2 * specification["num_experts"], "hot_replica": False, "source_token_bound": None, "planner_key": None}
        headroom += _workspace_layout(virtual_spec, dtype, virtual, source).required_bytes
        local = specification["num_experts"] // specification["ep_size"]
        bank = local, specification["hidden_size"], specification["intermediate_size"]
        if bank not in banks:
            banks.add(bank)
            headroom += 6 * math.prod(bank) * dtype.itemsize + 1024
    return tuple(reservations), headroom


_SHARED_BANKS = weakref.WeakKeyDictionary()


def _get_mega_bank(hidden, parent, owner):
    """Register a HotReplica allocation in the existing MegaMoE heap transaction."""
    spec = parent.spec
    manager = parent.heap_manager
    key = (spec.local_experts, spec.hidden_size, spec.intermediate_size, hidden.dtype)
    with manager.access():
        banks = _SHARED_BANKS.setdefault(manager, weakref.WeakValueDictionary())
        pool = banks.get(key)
        if pool is None or pool.closed:
            pool = SharedExpertPool(
                hidden, spec.ep_group, spec.local_experts, spec.intermediate_size,
                spec.num_cube_cores, spec.expert_weight_layout, root=manager, access=manager.access,
            )
            reservation = {
                "kind": "hot_replica_expert_bank", "layout": spec.expert_weight_layout, "local_experts": spec.local_experts,
            }
            signature = (
                "hot_replica_expert_bank", spec.local_experts, spec.hidden_size, spec.intermediate_size,
                str(hidden.dtype), spec.expert_weight_layout,
            )
            layout = _WorkspaceLayout(pool.allocation_bytes, signature, spec.local_experts)
            manager.reserve(((reservation, layout),), allow_resize=True)
            manager.bind(pool, spec, reservation)
            try:
                pool.allocate()
            except Exception:
                manager.close_workspace(pool)
                raise
            banks[key] = pool
            pool.owners = weakref.WeakSet()
        pool.owners.add(owner)
        return pool


class PlacementResources:
    """Own placement geometry and lazy banks under one BF16 resource owner."""

    def __init__(self, parent: Any, specification: dict[str, Any], shared: bool) -> None:
        """Attach lazy placement resources to one existing BF16 owner."""
        self.parent = parent
        self.specification = specification
        self.shared = shared
        self.pool = None
        self.geometries = {}
        self.source_bound = specification.get("source_token_bound")
        self.statistics = {
            "invocations": 0, "placements": 0,
            "route_replays": 0, "route_replay_misses": 0, "skipped_recomputes": 0,
        }

    def select(
        self,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        gathered: torch.Tensor,
        summary: tuple[int, ...],
        source_rows: list[int],
        inverse: torch.Tensor,
        autograd_mask: int,
        *, planner: BasePlanner, host_rows: list[list[int]] | None = None, top_k: int | None = None,
    ) -> tuple[BF16Resources, ExpertPlacement, ExpertWeightBinding] | None:
        """Choose a virtual route only after the existing count exchange validates it."""
        request = self.request(
            hidden, ids, gathered, summary, source_rows, inverse, autograd_mask, host_rows=host_rows, top_k=top_k
        )
        return self.bind(hidden, planner.plan(request))

    def request(
        self, hidden: torch.Tensor, ids: torch.Tensor, gathered: torch.Tensor, summary: tuple[int, ...],
        source_rows: list[int], inverse: torch.Tensor, autograd_mask: int, *,
        host_rows: list[list[int]] | None, top_k: int | None = None,
    ) -> PlacementRequest:
        """Expose counts and available capacity from one validated MegaMoE route."""
        self.statistics["invocations"] += 1
        spec = self.parent.spec
        bounded = self.source_bound is not None
        replay = is_checkpointing()
        key = (
            (autograd_mask, replay, tuple(tuple(row) for row in host_rows))
            if host_rows is not None and not bounded else None
        )
        return PlacementRequest(
            hidden, ids, gathered[:, 3:], inverse, summary[2], sum(source_rows), spec.rank_id,
            spec.intermediate_size, spec.num_cube_cores, replay or bool(autograd_mask & 1),
            checkpointing=replay, ordinary_receive_capacity=self.parent.workspace.capacity_floor if bounded else None,
            apply_probs_before_fc2=spec.apply_probs_before_fc2, count_key=key, top_k=top_k,
        )

    def bind(
        self, hidden: torch.Tensor, placement: ExpertPlacement | None,
    ) -> tuple[BF16Resources, ExpertPlacement, ExpertWeightBinding] | None:
        """Use a prepared placement to bind the existing BF16 runtime."""
        if placement is None:
            return None
        self.statistics["placements"] += 1
        resources = self._geometry(hidden, placement)
        binding = ExpertWeightBinding(self.pool, placement, self.parent.spec.expert_weight_layout)
        return resources, placement, binding

    def _geometry(self, hidden, placement):
        """Cache only static BF16 geometry; retain mainline scheduling choices."""
        # Imported lazily to avoid the BF16 resource / placement ownership cycle.
        from hyper_multicore.modules.mega_moe.backends.bf16.resources import (  # pylint: disable=import-outside-toplevel
            BF16Resources,
        )  # pylint: disable=import-outside-toplevel

        width = self.parent.spec.local_experts if self.source_bound is not None else placement.borrowed_capacity
        if width in self.geometries:
            return self.geometries[width]
        if self.pool is None:
            self.pool = _get_mega_bank(hidden, self.parent, self)
        specification = {
            **self.specification,
            "hot_replica": False, "source_token_bound": None, "planner_key": None,
            "num_experts": self.parent.spec.ep_size * (self.parent.spec.local_experts + width),
        }
        layout = None
        if self.source_bound is not None:
            source = _align_capacity(self.source_bound)
            _, virtual = _bounded_receive(self.specification, source, hidden.dtype, self.parent.spec.num_cube_cores)
            layout = _workspace_layout(
                specification, hidden.dtype, max(virtual, _align_capacity(placement.receive_capacity)), source
            )
        resources = BF16Resources(
            specification,
            hidden,
            expert_weight_layout=self.parent.spec.expert_weight_layout,
            shared=self.shared,
            active_specifications=(specification,),
            workspace_layout=layout,
        )
        self.geometries[width] = resources
        return resources

    def can_close(self) -> bool:
        """Include every owned child workspace and retained bank consumer."""
        return (self.pool is None or self.pool.can_close()) and all(
            resource.can_close() for resource in self.geometries.values()
        )

    def close(self) -> None:
        """Drain child resources through the existing root heap lifecycle."""
        for resource in self.geometries.values():
            resource.close()
        if self.pool is not None:
            manager = self.parent.heap_manager
            with manager.access():
                self.pool.owners.discard(self)
                if not self.pool.owners and not self.pool.closed:
                    manager.close_workspace(self.pool)
