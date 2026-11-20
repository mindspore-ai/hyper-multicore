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
from dataclasses import replace
import weakref
from typing import Any, TYPE_CHECKING

import torch

from hyper_multicore.modules.mega_moe.heap_manager import _WorkspaceLayout, _workspace_layout
from hyper_multicore.modules.mega_moe.spec import _COMMUNICATION_SPLIT, _align_capacity, _resolve_receive_capacity
from hyper_multicore.scheduler.config import mega_moe_event_capacity

from ..base_planner import BasePlanner, ExpertPlacement, PlacementRequest
from ._planner.planner import build_expert_replica_plan
from .bounded_placement import BoundedPlacement, PlacementWire, _rank_nodes, supports_bounded_placement
from .capability import has_a5_bf16_pipeline
from .checkpoint import is_checkpointing
from .placement import render_placement
from .pipeline import ReplicaPipeline, fused_replica_slots, supports_pipeline, uses_wider_gradient_bands
from .planner import HotReplicaPlanner
from .shmem_transfer import ExpertWeightBinding, SharedExpertPool

if TYPE_CHECKING:
    from hyper_multicore.modules.mega_moe.backends.bf16.resources import BF16Resources
    from hyper_multicore.modules.mega_moe.route import _RouteSummary


def _uses_builtin_planner(specification: Mapping[str, Any]) -> bool:
    """Identify the planner whose capacity checks the adapter owns."""
    planner_key = specification.get("planner_key")
    return planner_key is not None and planner_key[:2] == (HotReplicaPlanner.__module__, HotReplicaPlanner.__qualname__)


_SHARED_BANKS = weakref.WeakKeyDictionary()


def _get_mega_bank(
    hidden, parent, owner, *, pool_type=SharedExpertPool, reservations=(), pipeline=False, workspace_growth=None,
):
    """Register a HotReplica allocation in the existing MegaMoE heap transaction."""
    spec = parent.spec
    manager = parent.heap_manager
    device_prefetch = spec.architecture == 3510
    dedicated = pipeline and owner.dedicated_pipeline
    owner_id = id(owner) if dedicated or pool_type is not SharedExpertPool else None
    extent = owner.bank_extent if pipeline else None
    cube_cores = spec.num_cube_cores if pipeline and spec.architecture == 2201 else None
    key = (pool_type, spec.local_experts, spec.hidden_size, spec.intermediate_size, hidden.dtype, owner_id, extent,
           cube_cores)
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
                    pool.pipeline = ReplicaPipeline(pool, shared_layouts=True, device_prefetch=device_prefetch)
                pool.bank_experts = extent
                signature += (pool.bank_experts,)
                if cube_cores is not None:
                    signature += (cube_cores,)
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


def _bind_custom_copy_rows(spec, placement, *, fused_return=False):
    """Read the planner's authoritative copy IDs and producers in one compact transfer."""
    home, ranks = spec.local_experts, spec.ep_size
    borrowed = placement.borrowed_capacity
    if not borrowed:
        return replace(placement, copies_by_rank=())
    if tuple(placement.counts_by_source.shape) != (ranks, ranks * (home + borrowed)):
        if fused_return:
            raise ValueError("A3 expert placement counts must match the EP group and borrowed capacity")
        return replace(placement, copies_by_rank=())
    loads = placement.counts_by_source.sum(dim=0).view(ranks, home + borrowed)
    rows = torch.cat((placement.copies, loads), dim=1).tolist()
    for rank, row in enumerate(rows):
        for slot, count in enumerate(row[borrowed + home:]):
            owner = row[slot]
            if count == 0 and (fused_return or owner < 0):
                row[slot] = -1
                continue
            valid = 0 <= owner < ranks * home and count > 0
            if fused_return:
                if not valid:
                    raise ValueError("A3 active guest must have a valid logical owner and positive routed rows")
            elif not valid or owner // home == rank:
                return replace(placement, copies_by_rank=())
    copies = tuple(tuple(row[:borrowed]) for row in rows)
    if not any(owner >= 0 for row in copies for owner in row):
        if fused_return:
            return None
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
        self.replica_slots_per_rank = specification.get("replica_slots_per_rank")
        built_in = _uses_builtin_planner(specification)
        self.uses_a5_quotas = built_in and parent.spec.architecture == 3510 and parent.spec.dispatch_mode == "push"
        self.rank_nodes = None
        spec = parent.spec
        can_split_weights = supports_bounded_placement(spec)
        self.requires_fused_return = can_split_weights and spec.expert_weight_layout == "EIO"
        supported = built_in and can_split_weights
        self.bounded = (BoundedPlacement(spec, replica_slots_per_rank=self.replica_slots_per_rank)
                        if supported else None)
        self.bf16_partials = False
        self.prefer_bf16_partials = False
        self.pipeline_enabled = (
            self.uses_a5_quotas and supports_pipeline(spec, 1, bank_experts=self.bank_extent)
            and has_a5_bf16_pipeline()
        )
        self.dedicated_pipeline = False
        if can_split_weights:
            fp32_partials = self._uses_fp32_partials()
            self.bf16_partials = bf16_partials and not fp32_partials
            # Preserve the rounding of routes that used separate BF16 partials.
            unaligned_eio = self.requires_fused_return and spec.hidden_size * spec.intermediate_size % 16 != 0
            self.prefer_bf16_partials = not fp32_partials and (not built_in or spec.batch_experts or unaligned_eio)
            self.pipeline_enabled = (
                self.bf16_partials and supports_pipeline(spec, 1, bank_experts=self.bank_extent, bf16_partials=True)
                or not self.prefer_bf16_partials and supports_pipeline(spec, 1, bank_experts=self.bank_extent)
            )
            # Keep the default path's resident weights and share banks for
            # optional schedules, custom plans and larger projections.
            self.dedicated_pipeline = (
                supported and not spec.batch_experts and not uses_wider_gradient_bands(spec)
            )
        share_push_workspace = self.pipeline_enabled or (supported and not spec.batch_experts)
        self.reuses_parent_workspace = (
            (spec.dispatch_mode == "pull" or share_push_workspace)
            and mega_moe_event_capacity(spec.num_experts, spec.ep_size)
            == mega_moe_event_capacity(2 * spec.num_experts, spec.ep_size)
        )
        self.statistics = {
            "invocations": 0, "placements": 0,
            "route_replays": 0, "route_replay_misses": 0, "skipped_recomputes": 0,
        }

    def uses_placement_wire(self, checkpointing: bool) -> bool:
        """Use the count wire for ordinary per-expert Push descriptors."""
        return self.bounded is not None and not self.parent.spec.batch_experts and not checkpointing

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
        receive_capacity_percent = self.specification.get("receive_capacity_percent")
        if self.bounded is None or checkpointing or spec.batch_experts or receive_capacity_percent is not None:
            if snapshot is not None:
                host_rows = snapshot.wire.tolist()
            request = self.request(hidden, ids, gathered, summary, source_rows, inverse, autograd_mask,
                                   host_rows=host_rows)
            if self.bounded is None:
                placement = self._plan_a5(request, host_rows) if self.uses_a5_quotas else planner.plan(request)
                if placement is not None and (self.pipeline_enabled or self.requires_fused_return):
                    placement = _bind_custom_copy_rows(spec, placement, fused_return=self.requires_fused_return)
            else:
                def build(current, target, **options):
                    return self.bounded.select(
                        hidden, ids, current.counts_by_source, source_rows, inverse, 0, snapshot,
                        capacity_target=target, host_rows=host_rows, **options,
                    )

                placement = planner._plan_by_cost(request, build)
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

    def _plan_a5(self, request, host_rows):
        """Use bounded quotas with the ordinary Ascend950 BF16 route renderer."""
        spec = self.parent.spec
        peak = request.maximum_received
        average = (request.total_rows + spec.ep_size - 1) // spec.ep_size
        saved = peak - average
        if saved <= 0 or (4 * saved < peak if request.training else 5 * saved < 2 * peak):
            return None
        if self.rank_nodes is None:
            self.rank_nodes = _rank_nodes(spec.ep_group, spec.ep_size)
        counts = ([row[3:] for row in host_rows] if host_rows is not None else
                  request.counts_by_source.tolist())
        budget = self.replica_slots_per_rank or spec.local_experts
        plan = build_expert_replica_plan(
            counts, min(budget, spec.local_experts), minimum_replica_rows=1024,
            minimum_load_reduction=0.25 if request.training else 0.4,
            capacity_limit=peak, receive_capacity=peak, rank_nodes=self.rank_nodes,
            transfer_row_credit=spec.intermediate_size // 2 if request.training else 0,
        )
        if not plan.transfers:
            return None
        return render_placement(request.hidden, request.ids, plan, request.rank,
                                source_inverse=request.source_inverse)

    def request(
        self, hidden: torch.Tensor, ids: torch.Tensor, gathered: torch.Tensor, summary: _RouteSummary,
        source_rows: list[int], inverse: torch.Tensor | None, autograd_mask: int, *,
        host_rows: list[list[int]] | None,
    ) -> PlacementRequest:
        """Expose counts and available capacity from one validated MegaMoE route."""
        self.statistics["invocations"] += 1
        spec = self.parent.spec
        replay = is_checkpointing()
        key = (
            (autograd_mask, replay, hidden.shape[-1], hidden.element_size(), ids.shape[-1],
             spec.intermediate_size, spec.num_cube_cores, tuple(tuple(row) for row in host_rows))
            if host_rows is not None else None
        )
        return PlacementRequest(
            hidden, ids, gathered[:, 3:], inverse, summary.maximum, sum(source_rows), spec.rank_id,
            spec.intermediate_size, spec.num_cube_cores, replay or bool(autograd_mask & 1),
            checkpointing=replay, apply_probs_before_fc2=spec.apply_probs_before_fc2, count_key=key,
        )

    def bind(
        self, hidden: torch.Tensor, placement: ExpertPlacement | None, *, tokens: int, fused: bool = False,
    ) -> tuple[BF16Resources, ExpertPlacement, ExpertWeightBinding] | None:
        """Use a prepared placement to bind the existing BF16 runtime."""
        if placement is None:
            return None
        if self.requires_fused_return and not placement.borrowed_capacity:
            return None
        self.statistics["placements"] += 1
        spec = self.parent.spec
        slots = 0
        bf16_partials = False
        if fused and placement.copies_by_rank:
            if not self.prefer_bf16_partials:
                slots = fused_replica_slots(spec, placement, bank_experts=self.bank_extent)
            if not slots and self.bf16_partials:
                slots = fused_replica_slots(spec, placement, bank_experts=self.bank_extent, bf16_partials=True)
                bf16_partials = bool(slots)
        if self.requires_fused_return and not slots:
            if not self.bf16_partials and not self._uses_fp32_partials():
                raise RuntimeError("A3 EIO HotReplica needs an A3 wheel with BF16 partial dW support for this placement")
            raise RuntimeError("No fused dW protocol fits the active A3 EIO HotReplica placement")
        resources = self._geometry(hidden, placement, tokens)
        return_in_kernel = False
        if fused and self.pool.pipeline is not None and placement.copies_by_rank and self.pool.blocks == spec.num_cube_cores:
            return_in_kernel = bool(slots)
            if not slots:
                slots = placement.borrowed_capacity
        else:
            slots = 0
            bf16_partials = False
        if self.requires_fused_return and not return_in_kernel:
            raise RuntimeError("A3 EIO HotReplica bank did not bind the selected fused dW plan")
        binding = ExpertWeightBinding(
            self.pool, placement, spec.expert_weight_layout, fused=bool(slots), fused_slots=slots,
            return_in_kernel=return_in_kernel, bf16_partials=bf16_partials,
        )
        return resources, placement, binding

    def _uses_fp32_partials(self):
        """Use the same partial-gradient dtype as the ordinary placed launch."""
        spec = self.parent.spec
        return spec.architecture == 3510 or spec.partial_weight_grad_dtype == "float32"

    @property
    def bank_extent(self):
        """Keep the guest partial suffix aligned without overlapping weights."""
        spec = self.parent.spec
        extent = (3 if self._uses_fp32_partials() else 2) * spec.local_experts
        if extent % 2 and self.requires_fused_return and spec.hidden_size * spec.intermediate_size % 2:
            # One BF16 expert record makes the end of both allocations a valid FP32 base.
            extent += 1
        return extent

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
        width = spec.local_experts if shared_geometry else placement.borrowed_capacity
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
            "hot_replica": False, "planner_key": None,
            "num_experts": spec.ep_size * (spec.local_experts + width),
        }
        if self.shared and self.bounded is not None and not spec.batch_experts:
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
