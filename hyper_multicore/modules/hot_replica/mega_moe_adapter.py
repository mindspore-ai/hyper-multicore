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
from hyper_multicore.modules.mega_moe.spec import _COMMUNICATION_SPLIT, _align_capacity, _resolve_receive_capacity
from hyper_multicore.scheduler.config import mega_moe_event_capacity

from ..base_planner import BasePlanner, ExpertPlacement, PlacementRequest
from ._planner._dispatch import ReplicaCapacityError
from ._planner.planner import build_expert_replica_plan
from .bounded_placement import BoundedPlacement, PlacementWire, _rank_nodes, supports_bounded_placement
from .checkpoint import is_checkpointing
from .placement import placement_receive_limit, render_placement
from .pipeline import ReplicaPipeline, fused_replica_slots, supports_pipeline
from .planner import HotReplicaPlanner
from .shmem_transfer import ExpertWeightBinding, SharedExpertPool, _expert_bank_bytes

if TYPE_CHECKING:
    from hyper_multicore.modules.mega_moe.backends.bf16.resources import BF16Resources
    from hyper_multicore.modules.mega_moe.route import _RouteSummary


def _uses_builtin_planner(specification: Mapping[str, Any]) -> bool:
    """Identify the planner whose capacity checks the adapter owns."""
    planner_key = specification.get("planner_key")
    return planner_key is not None and planner_key[:2] == (HotReplicaPlanner.__module__, HotReplicaPlanner.__qualname__)


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
) -> tuple[tuple[tuple[Mapping[str, Any], _WorkspaceLayout], ...], int, int]:
    """Describe A3 ordinary layouts and headroom for automatic and fixed heaps."""
    reservations = []
    banks = set()
    fixed_banks = set()
    headroom = 0
    fixed_headroom = 0
    for specification in specifications:
        bound = specification.get("source_token_bound")
        if bound is None:
            reservations.append((specification, _workspace_layout(specification, dtype)))
            continue
        source = _align_capacity(bound)
        native, virtual = _bounded_receive(specification, source, dtype, cube_cores)
        reservations.append((specification, _workspace_layout(specification, dtype, native, source)))
        virtual_spec = {**specification, "num_experts": 2 * specification["num_experts"],
                        "hot_replica": False, "source_token_bound": None, "planner_key": None}
        virtual_bytes = _workspace_layout(virtual_spec, dtype, virtual, source).required_bytes
        headroom += virtual_bytes
        local = specification["num_experts"] // specification["ep_size"]
        bank = local, specification["hidden_size"], specification["intermediate_size"]
        if bank not in banks:
            banks.add(bank)
            headroom += _expert_bank_bytes(*bank, dtype)
        # BF16 placement capability has checked the A3 device. Its built-in
        # per-expert path budgets actual child layouts against the fixed heap.
        deferred = (
            _uses_builtin_planner(specification) and not specification.get("batch_experts", False)
            and specification.get("dispatch_mode", "push") == "push"
            and specification.get("transport_engine") in (None, "mte")
        )
        if not deferred:
            fixed_headroom += virtual_bytes
            if bank not in fixed_banks:
                fixed_banks.add(bank)
                fixed_headroom += _expert_bank_bytes(*bank, dtype)
    return tuple(reservations), headroom, fixed_headroom


_SHARED_BANKS = weakref.WeakKeyDictionary()


def _bank_key(spec, dtype, owner_id):
    """Identify an existing BF16 bank by its physical allocation shape."""
    return (spec.local_experts, spec.hidden_size, spec.intermediate_size, dtype, owner_id)


def _get_mega_bank(
    hidden, parent, owner, *, pool_type=SharedExpertPool, reservations=(), pipeline=False, workspace_growth=None,
):
    """Register a HotReplica allocation in the existing MegaMoE heap transaction."""
    spec = parent.spec
    manager = parent.heap_manager
    dedicated = pipeline and owner.source_bound is None and manager.heap_limit is None
    owner_id = id(owner) if dedicated or pool_type is not SharedExpertPool else None
    key = _bank_key(spec, hidden.dtype, owner_id)
    with manager.access():
        banks = _SHARED_BANKS.setdefault(manager, weakref.WeakValueDictionary())
        pool = banks.get(key)
        if pool is None or pool.closed:
            pool = pool_type(
                hidden, spec.ep_group, spec.local_experts, spec.intermediate_size,
                spec.num_cube_cores, spec.expert_weight_layout, root=manager, access=manager.access,
                pipeline=dedicated,
            )
            reservation = {
                "kind": pool.pool_kind, "layout": spec.expert_weight_layout, "local_experts": spec.local_experts,
            }
            signature = (
                pool.pool_kind, spec.local_experts, spec.hidden_size, spec.intermediate_size,
                str(hidden.dtype), spec.expert_weight_layout,
            )
            layout = _WorkspaceLayout(pool.allocation_bytes, signature, spec.local_experts)
            added = manager.reserve((*reservations, (reservation, layout)), allow_resize=True,
                                    workspace_growth=workspace_growth)
            try:
                manager.bind(pool, spec, reservation)
            except Exception:
                manager.rollback_reservations(added, None)
                raise
            try:
                pool.allocate()
            except Exception:
                manager.close_workspace(pool)
                manager.rollback_reservations(added, None)
                raise
            banks[key] = pool
            pool.owners = weakref.WeakSet()
        if pipeline and pool.pipeline is None:
            pool.pipeline = ReplicaPipeline(pool, inline=True)
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
        self.initial_receive_floor = parent.workspace.capacity_floor if self.source_bound is not None else None
        built_in = _uses_builtin_planner(specification)
        self.fixed_heap = False
        self.rank_nodes = None
        if built_in and supports_bounded_placement(parent.spec):
            heap_limited = parent.heap_manager.heap_limit is not None
            has_reservation = self.source_bound is not None
            self.bounded = BoundedPlacement(parent.spec) if has_reservation or not heap_limited else None
            self.fixed_heap = heap_limited
        else:
            self.bounded = None
        spec = parent.spec
        self.reuses_parent_workspace = (
            self.bounded is not None and self.source_bound is None
            and mega_moe_event_capacity(spec.num_experts, spec.ep_size)
            == mega_moe_event_capacity(2 * spec.num_experts, spec.ep_size)
        )
        self.pipeline_enabled = built_in and supports_pipeline(parent.spec, 1)
        self.statistics = {
            "invocations": 0, "placements": 0,
            "route_replays": 0, "route_replay_misses": 0, "skipped_recomputes": 0,
        }

    def uses_shared_quotas(self, checkpointing: bool) -> bool:
        """Keep unreserved checkpoint admission tied to the generic allocator."""
        return self.bounded is not None and (self.source_bound is not None or not checkpointing)

    def select(
        self,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        gathered: torch.Tensor,
        summary: _RouteSummary,
        source_rows: list[int],
        inverse: torch.Tensor | None,
        autograd_mask: int,
        *,
        planner: BasePlanner,
        host_rows: list[list[int]] | None = None,
        snapshot: PlacementWire | None = None,
        checkpointing: bool = False,
    ) -> tuple[BF16Resources, ExpertPlacement, ExpertWeightBinding] | None:
        """Choose a virtual route only after the existing count exchange validates it."""
        shared_quotas = self.uses_shared_quotas(checkpointing)
        if not shared_quotas or self.source_bound is not None:
            request = self.request(hidden, ids, gathered, summary, source_rows, inverse, autograd_mask,
                                   host_rows=host_rows)
            if not shared_quotas:
                # Keep the unreserved admission policy while reusing the
                # agreed copies for the fused bank, including replay.
                placement = planner.plan(request)
                if self.fixed_heap:
                    return self._bind_fixed(request, placement, summary.plan_tokens, host_rows)
                return self.bind(hidden, placement, tokens=summary.plan_tokens, fused=self.pipeline_enabled)

            def _build_reserved(current, _target):
                return self.bounded.select(
                    hidden, ids, current.counts_by_source, source_rows, inverse, 0, snapshot,
                    reserved=current.ordinary_receive_capacity,
                )

            # At the full home-slot budget the quota search can attain the
            # group average; a reservation below that bound requires growth.
            selected = HotReplicaPlanner._plan_reserved(request, _build_reserved)
            if self.fixed_heap:
                return self._bind_fixed(request, selected, summary.plan_tokens, host_rows)
            return self.bind(hidden, selected, tokens=summary.plan_tokens, fused=self.pipeline_enabled)
        self.statistics["invocations"] += 1
        peak = summary.maximum
        spec = self.parent.spec
        average = (sum(source_rows) + spec.ep_size - 1) // spec.ep_size
        projections = 2 * bool(autograd_mask & 4) + bool(autograd_mask & 8) if autograd_mask & 1 else 0
        saved = peak - average
        if not saved or (4 * saved < peak if projections == 3 else 5 * saved < 2 * peak):
            return None
        placement = self.bounded.select(hidden, ids, gathered[:, 3:], source_rows, inverse, projections, snapshot)
        return self.bind(hidden, placement, tokens=summary.plan_tokens, fused=self.pipeline_enabled)

    def _bind_fixed(self, request, placement, tokens, host_rows):
        """Fit the selected plan before reserving storage; try a new plan if home spills."""
        manager = self.parent.heap_manager
        spec = self.parent.spec
        hidden = request.hidden
        with manager.access():
            if placement is not None:
                width = spec.local_experts if self.source_bound is not None else placement.borrowed_capacity
                if self._virtual_fits(hidden.dtype, tokens, placement.receive_capacity, width):
                    return self.bind(hidden, placement, tokens=tokens, fused=self.pipeline_enabled)
            if manager.fits_fixed_heap(
                    workspace_growth=(self.parent.workspace, tokens, request.maximum_received)):
                return None

            if self.rank_nodes is None:
                nodes = self.bounded.nodes if self.bounded is not None else None
                self.rank_nodes = nodes if nodes is not None else _rank_nodes(spec.ep_group, spec.ep_size)
            counts = ([row[3:] for row in host_rows] if host_rows is not None else
                      request.counts_by_source.tolist())
            lower = (request.total_rows + spec.ep_size - 1) // spec.ep_size
            for width in dict.fromkeys((*sorted(self.geometries, reverse=True), spec.local_experts)):
                limit = self._fixed_receive_limit(hidden.dtype, tokens, lower, request.maximum_received, width)
                if limit < lower:
                    continue
                try:
                    plan = build_expert_replica_plan(
                        counts, width, capacity_limit=limit, receive_capacity=limit, rank_nodes=self.rank_nodes,
                    )
                except ReplicaCapacityError:
                    continue
                if plan.transfers:
                    placement = render_placement(hidden, request.ids, plan, request.rank,
                                                 source_inverse=request.source_inverse)
                    return self.bind(hidden, placement, tokens=tokens, fused=self.pipeline_enabled,
                                     workspace_width=width)
        return None

    def _fixed_receive_limit(self, dtype, tokens, lower, upper, width):
        """Find the largest feasible receive tile for the prospective virtual workspace."""
        tile = _COMMUNICATION_SPLIT
        lo, hi = _align_capacity(lower) // tile, _align_capacity(upper) // tile
        limit = 0
        while lo <= hi:
            middle = (lo + hi) // 2
            if self._virtual_fits(dtype, tokens, middle * tile, width):
                limit = middle * tile
                lo = middle + 1
            else:
                hi = middle - 1
        return limit

    def _virtual_fits(self, dtype, tokens, rows, width):
        """Count the actual virtual workspace and any still-unallocated shared bank."""
        spec = self.parent.spec
        manager = self.parent.heap_manager
        resources = self.geometries.get(width)
        if resources is not None:
            return manager.fits_fixed_heap(workspace_growth=(resources.workspace, tokens, rows))
        _, layout = self._virtual_reservation(dtype, tokens, rows, width)
        pool = _SHARED_BANKS.get(manager, {}).get(_bank_key(spec, dtype, None))
        layouts = (layout,)
        if pool is None or pool.closed:
            bank_bytes = _expert_bank_bytes(spec.local_experts, spec.hidden_size, spec.intermediate_size, dtype)
            layouts = (layout, _WorkspaceLayout(bank_bytes, (), 0))
        return manager.fits_fixed_heap(layouts)

    def request(
        self, hidden: torch.Tensor, ids: torch.Tensor, gathered: torch.Tensor, summary: _RouteSummary,
        source_rows: list[int], inverse: torch.Tensor | None, autograd_mask: int, *,
        host_rows: list[list[int]] | None,
    ) -> PlacementRequest:
        """Expose counts and available capacity from one validated MegaMoE route."""
        self.statistics["invocations"] += 1
        spec = self.parent.spec
        bounded = self.source_bound is not None
        replay = is_checkpointing()
        # Replanning during recompute must not depend on an intervening heap growth.
        floor = self.parent.workspace.capacity_floor if bounded else None
        if replay and self.bounded is not None:
            floor = self.initial_receive_floor
        key = (
            (autograd_mask, replay, tuple(tuple(row) for row in host_rows))
            if host_rows is not None and not bounded else None
        )
        return PlacementRequest(
            hidden, ids, gathered[:, 3:], inverse, summary.maximum, sum(source_rows), spec.rank_id,
            spec.intermediate_size, spec.num_cube_cores, replay or bool(autograd_mask & 1),
            checkpointing=replay, ordinary_receive_capacity=floor,
            apply_probs_before_fc2=spec.apply_probs_before_fc2, count_key=key,
        )

    def bind(
        self, hidden: torch.Tensor, placement: ExpertPlacement | None, *, tokens: int, fused: bool = False,
        workspace_width: int | None = None,
    ) -> tuple[BF16Resources, ExpertPlacement, ExpertWeightBinding] | None:
        """Use a prepared placement to bind the existing BF16 runtime."""
        if placement is None:
            return None
        self.statistics["placements"] += 1
        resources = (self._geometry(hidden, placement, tokens) if workspace_width is None else
                     self._geometry(hidden, placement, tokens, width=workspace_width))
        spec = self.parent.spec
        pipeline = self.pool.pipeline
        slots = 0
        return_in_kernel = False
        if fused and pipeline is not None and placement.copies_by_rank and self.pool.blocks == spec.num_cube_cores:
            slots = fused_replica_slots(spec, placement, slot_limit=pipeline.slot_limit)
            return_in_kernel = bool(slots)
            if not slots and not pipeline.inline:
                # A wide live prefix may overlap FP32 dW. It can still feed
                # the projection kernels when the flags have their own storage.
                slots = placement.borrowed_capacity
        binding = ExpertWeightBinding(
            self.pool, placement, spec.expert_weight_layout, fused=bool(slots), fused_slots=slots,
            return_in_kernel=return_in_kernel,
        )
        return resources, placement, binding

    def _geometry(self, hidden, placement, tokens, *, width=None):
        """Cache only static BF16 geometry; retain mainline scheduling choices."""
        # Imported lazily to avoid the BF16 resource / placement ownership cycle.
        from hyper_multicore.modules.mega_moe.backends.bf16.resources import (  # pylint: disable=import-outside-toplevel
            BF16Resources,
        )  # pylint: disable=import-outside-toplevel

        spec = self.parent.spec
        manager = self.parent.heap_manager
        if self.reuses_parent_workspace:
            if self.pool is None:
                source = max(_COMMUNICATION_SPLIT, _align_capacity(tokens))
                receive = _align_capacity(placement.receive_capacity)
                if self.shared:
                    receive = max(receive, _resolve_receive_capacity(
                        spec.initial_capacity_factor, source * spec.top_k, spec.ep_size,
                    ))
                self.pool = _get_mega_bank(
                    hidden, self.parent, self, pipeline=self.pipeline_enabled,
                    workspace_growth=(self.parent.workspace, source, receive),
                )
            return self.parent
        shared_geometry = self.shared and self.bounded is not None
        if width is None:
            width = (spec.local_experts if self.source_bound is not None or shared_geometry else
                     placement.borrowed_capacity)
        if width in self.geometries:
            return self.geometries[width]
        specification, layout = self._virtual_reservation(hidden.dtype, tokens, placement.receive_capacity, width)
        if self.pool is None:
            self.pool = _get_mega_bank(
                hidden, self.parent, self, reservations=((specification, layout),), pipeline=self.pipeline_enabled,
            )
        try:
            resources = BF16Resources(
                specification,
                hidden,
                expert_weight_layout=spec.expert_weight_layout,
                shared=self.shared,
                active_specifications=(specification,),
                workspace_layout=layout,
            )
        except Exception:
            manager.rollback_reservations((specification,), None)
            raise
        self.geometries[width] = resources
        return resources

    def _virtual_reservation(self, dtype, tokens, received, width):
        """Size a new virtual workspace with the same floors as resource binding."""
        spec = self.parent.spec
        source = max(_COMMUNICATION_SPLIT, _align_capacity(tokens))
        receive = _align_capacity(received)
        specification = {
            **self.specification,
            "hot_replica": False, "source_token_bound": None, "planner_key": None,
            "num_experts": spec.ep_size * (spec.local_experts + width),
        }
        if self.source_bound is not None:
            bound = _align_capacity(self.source_bound)
            _, virtual = _bounded_receive(self.specification, bound, dtype, spec.num_cube_cores)
            source, receive = max(bound, source), max(virtual, receive)
        elif self.shared and self.bounded is not None and self.parent.heap_manager.heap_limit is None:
            receive = max(receive, _resolve_receive_capacity(
                spec.initial_capacity_factor, source * spec.top_k, spec.ep_size,
            ))
        return specification, _workspace_layout(specification, dtype, receive, source)

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
