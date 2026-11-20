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

from dataclasses import replace

from collections.abc import Mapping
import math
import weakref
from typing import Any, TYPE_CHECKING

import torch

from hyper_multicore.modules.mega_moe.heap_manager import _WorkspaceLayout, _workspace_layout
from hyper_multicore.modules.mega_moe.spec import _COMMUNICATION_SPLIT, _align_capacity, _resolve_receive_capacity
from hyper_multicore.scheduler.config import mega_moe_event_capacity

from ..base_planner import BasePlanner, ExpertPlacement, PlacementRequest
from .bounded_placement import BoundedPlacement, PlacementWire, supports_bounded_placement
from .checkpoint import is_checkpointing
from .placement import placement_receive_limit
from .pipeline import ReplicaPipeline, fused_replica_slots, supports_pipeline, uses_wider_gradient_bands
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
) -> tuple[tuple[tuple[Mapping[str, Any], _WorkspaceLayout], ...], int]:
    """Describe initial A3 workspace layouts and default replica headroom."""
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
        reservations.append((
            specification, _workspace_layout(specification, dtype, native, source),
        ))
        virtual_spec = {**specification, "num_experts": 2 * specification["num_experts"],
                        "hot_replica": False, "source_token_bound": None, "planner_key": None}
        virtual_bytes = _workspace_layout(virtual_spec, dtype, virtual, source).required_bytes
        headroom += virtual_bytes
        local = specification["num_experts"] // specification["ep_size"]
        bank = local, specification["hidden_size"], specification["intermediate_size"]
        if bank not in banks:
            banks.add(bank)
            headroom += _expert_bank_bytes(*bank, dtype)
    return tuple(reservations), headroom


_SHARED_BANKS = weakref.WeakKeyDictionary()


def _get_mega_bank(
    hidden, parent, owner, *, pool_type=SharedExpertPool, reservations=(), pipeline=False, workspace_growth=None,
):
    """Register a HotReplica allocation in the existing MegaMoE heap transaction."""
    spec = parent.spec
    manager = parent.heap_manager
    dedicated = pipeline and owner.dedicated_pipeline
    owner_id = id(owner) if dedicated or pool_type is not SharedExpertPool else None
    extent = owner.bank_extent if pipeline else None
    key = (pool_type, spec.local_experts, spec.hidden_size, spec.intermediate_size, hidden.dtype, owner_id, extent)
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
            if pipeline:
                if not dedicated:
                    pool.pipeline = ReplicaPipeline(pool, shared_layouts=True)
                pool.bank_experts = extent
                signature += (pool.bank_experts,)
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
        pool.owners.add(owner)
        return pool


def _live_slot_extent(placement):
    """Find the end of the globally occupied copy prefix."""
    return max((slot + 1 for row in placement.copies_by_rank for slot, owner in enumerate(row) if owner >= 0),
               default=0)


def _bind_custom_copy_rows(spec, placement):
    """Read the planner's authoritative copy IDs and producers in one compact transfer."""
    home, ranks = spec.local_experts, spec.ep_size
    borrowed = placement.borrowed_capacity
    if not borrowed or tuple(placement.counts_by_source.shape) != (ranks, ranks * (home + borrowed)):
        return replace(placement, copies_by_rank=())
    loads = placement.counts_by_source.sum(dim=0).view(ranks, home + borrowed)
    rows = torch.cat((placement.copies, loads), dim=1).tolist()
    copies = tuple(tuple(row[:borrowed]) for row in rows)
    for rank, (owners, row) in enumerate(zip(copies, rows)):
        for owner, count in zip(owners, row[borrowed + home:]):
            if owner < 0 and count == 0:
                continue
            if not 0 <= owner < ranks * home or owner // home == rank or count <= 0:
                # The pipeline waits for a producer in every published guest slot.
                return replace(placement, copies_by_rank=())
    if not any(owner >= 0 for row in copies for owner in row):
        return replace(placement, copies_by_rank=())
    published = max((i + 1 for i, owner in enumerate(copies[spec.rank_id]) if owner >= 0), default=0)
    local = tuple(rows[spec.rank_id][borrowed:borrowed + home])
    return replace(placement, copies_by_rank=copies, owner_rows=local, published_slots=published)


class PlacementResources:
    """Own placement geometry and lazy banks under one BF16 resource owner."""

    def __init__(self, parent: Any, specification: dict[str, Any], shared: bool, *, bf16_partials: bool = False) -> None:
        """Attach lazy placement resources to one existing BF16 owner."""
        self.parent = parent
        self.specification = specification
        self.shared = shared
        self.pool = None
        self.geometries = {}
        self.source_bound = specification.get("source_token_bound")
        self.initial_receive_floor = parent.workspace.capacity_floor if self.source_bound is not None else None
        built_in = _uses_builtin_planner(specification)
        spec = parent.spec
        can_split_weights = supports_bounded_placement(spec)
        supported = built_in and can_split_weights
        self.bounded = BoundedPlacement(parent.spec) if supported else None
        self.bf16_partials = False
        self.prefer_bf16_partials = False
        self.pipeline_enabled = False
        self.dedicated_pipeline = False
        if can_split_weights:
            fp32_partials = self._uses_fp32_partials()
            self.bf16_partials = bf16_partials and not fp32_partials
            # Preserve the former separate executor's partial dtype on these schedules.
            self.prefer_bf16_partials = not fp32_partials and (not built_in or spec.batch_experts)
            self.pipeline_enabled = (
                self.bf16_partials and supports_pipeline(spec, 1, bank_experts=self.bank_extent, bf16_partials=True)
                or not self.prefer_bf16_partials and supports_pipeline(spec, 1, bank_experts=self.bank_extent)
            )
            # Keep the default path's resident weights and share banks for
            # optional schedules, custom plans and larger projections.
            self.dedicated_pipeline = (
                supported and not spec.batch_experts and self.source_bound is None
                and not uses_wider_gradient_bands(spec)
            )
        self.reuses_parent_workspace = (
            (self.pipeline_enabled or supported and not spec.batch_experts) and self.source_bound is None
            and mega_moe_event_capacity(spec.num_experts, spec.ep_size)
            == mega_moe_event_capacity(2 * spec.num_experts, spec.ep_size)
        )
        self.statistics = {
            "invocations": 0, "placements": 0,
            "route_replays": 0, "route_replay_misses": 0, "skipped_recomputes": 0,
        }

    def uses_placement_wire(self, checkpointing: bool) -> bool:
        """Enable the bounded count readback where it also supplies Push descriptors."""
        return (self.bounded is not None and not self.parent.spec.batch_experts
                and (self.source_bound is not None or not checkpointing))

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
        spec = self.parent.spec
        if self.bounded is None or self.source_bound is not None or checkpointing or spec.batch_experts:
            request = self.request(hidden, ids, gathered, summary, source_rows, inverse, autograd_mask,
                                   host_rows=host_rows)
            if self.bounded is None:
                placement = planner.plan(request)
                if placement is not None and self.pipeline_enabled:
                    placement = _bind_custom_copy_rows(spec, placement)
            else:
                def build(current, target):
                    reserved = None if spec.batch_experts else current.ordinary_receive_capacity
                    return self.bounded.select(
                        hidden, ids, current.counts_by_source, source_rows, inverse, 0, snapshot,
                        reserved=reserved, capacity_target=target if reserved is None else None, host_rows=host_rows,
                    )

                if request.ordinary_receive_capacity is None:
                    placement = planner._plan_unreserved(request, build)
                else:
                    placement = HotReplicaPlanner._plan_reserved(request, build)
            return self.bind(hidden, placement, tokens=summary.plan_tokens, fused=self.pipeline_enabled)
        self.statistics["invocations"] += 1
        peak = summary.maximum
        average = (sum(source_rows) + spec.ep_size - 1) // spec.ep_size
        projections = 2 * bool(autograd_mask & 4) + bool(autograd_mask & 8) if autograd_mask & 1 else 0
        saved = peak - average
        if not saved or (4 * saved < peak if projections == 3 else 5 * saved < 2 * peak):
            return None
        placement = self.bounded.select(hidden, ids, gathered[:, 3:], source_rows, inverse, projections, snapshot)
        return self.bind(hidden, placement, tokens=summary.plan_tokens, fused=self.pipeline_enabled)

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
        if replay and self.bounded is not None and not spec.batch_experts:
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
    ) -> tuple[BF16Resources, ExpertPlacement, ExpertWeightBinding] | None:
        """Use a prepared placement to bind the existing BF16 runtime."""
        if placement is None:
            return None
        self.statistics["placements"] += 1
        resources = self._geometry(hidden, placement, tokens)
        spec = self.parent.spec
        pipeline = self.pool.pipeline
        slots = 0
        return_in_kernel = False
        bf16_partials = False
        if fused and pipeline is not None and placement.copies_by_rank and self.pool.blocks == spec.num_cube_cores:
            if not self.prefer_bf16_partials:
                slots = fused_replica_slots(spec, placement, bank_experts=self.pool.bank_experts)
            if not slots and self.bf16_partials:
                slots = fused_replica_slots(spec, placement, bank_experts=self.pool.bank_experts, bf16_partials=True)
                bf16_partials = bool(slots)
            return_in_kernel = bool(slots)
            if not slots:
                slots = placement.borrowed_capacity
        binding = ExpertWeightBinding(
            self.pool, placement, spec.expert_weight_layout, fused=bool(slots), fused_slots=slots,
            return_in_kernel=return_in_kernel, bf16_partials=bf16_partials,
        )
        return resources, placement, binding

    def _uses_fp32_partials(self):
        """Use the same partial-gradient dtype as the ordinary placed launch."""
        return self.source_bound is not None or self.parent.spec.partial_weight_grad_dtype == "float32"

    @property
    def bank_extent(self):
        """Keep one weight and one partial-gradient record per possible guest."""
        return (3 if self._uses_fp32_partials() else 2) * self.parent.spec.local_experts

    def _geometry(self, hidden, placement, tokens):
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
        shared_geometry = self.shared and self.bounded is not None and not spec.batch_experts
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
        elif self.shared and self.bounded is not None and not spec.batch_experts:
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
