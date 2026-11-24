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

from hyper_multicore import _collective

from hyper_multicore.modules.mega_moe.heap_manager import _HeapCapacityError, _WorkspaceLayout
from hyper_multicore.modules.mega_moe.spec import _COMMUNICATION_SPLIT, _align_capacity, _resolve_receive_capacity

from ..base_planner import BasePlanner, ExpertPlacement, PlacementRequest
from ._planner.receive import bf16_cost, receive_limit
from .bounded_placement import BoundedPlacement, PlacementWire, supports_bounded_placement
from .checkpoint import is_checkpointing
from .pipeline import ReplicaPipeline, fused_replica_slots, supports_pipeline, uses_wider_gradient_bands
from .planner import HotReplicaPlanner
from .shmem_transfer import ExpertWeightBinding, SharedExpertPool

if TYPE_CHECKING:
    from hyper_multicore.modules.mega_moe.route import _RouteSummary


def _uses_builtin_planner(specification: Mapping[str, Any]) -> bool:
    """Identify the planner whose capacity checks the adapter owns."""
    planner_key = specification.get("planner_key")
    return planner_key is not None and planner_key[:2] == (HotReplicaPlanner.__module__, HotReplicaPlanner.__qualname__)


_SHARED_BANKS = weakref.WeakKeyDictionary()


def _get_mega_bank(
    hidden, parent, owner, *, tokens, received, preferred_receive=None, pipeline=False,
):
    """Register a HotReplica allocation in the existing MegaMoE heap transaction."""
    spec = parent.spec
    manager = parent.heap_manager
    dedicated = pipeline and owner.dedicated_pipeline
    owner_id = id(owner) if dedicated else None
    extent = owner.bank_extent if pipeline else None
    cube_cores = spec.num_cube_cores if pipeline else None
    key = (spec.local_experts, spec.hidden_size, spec.intermediate_size, hidden.dtype, owner_id, extent,
           cube_cores)
    with manager.access():
        banks = _SHARED_BANKS.setdefault(manager, weakref.WeakValueDictionary())
        pool = banks.get(key)
        if pool is None or pool.closed:
            pool = SharedExpertPool(
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
                if cube_cores is not None:
                    signature += (cube_cores,)
            layout = _WorkspaceLayout(pool.allocation_bytes, signature, spec.local_experts)
            added = manager.reserve(
                ((reservation, layout),), allow_resize=True,
                workspace_growth=(parent.workspace, tokens, received), preferred_receive=preferred_receive,
            )
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


def _release_mega_bank(parent, owner):
    """Close a bank when its last placement owner has been released."""
    pool = owner.pool
    if pool is not None:
        manager = parent.heap_manager
        with manager.access():
            pool.owners.discard(owner)
            if not pool.owners and not pool.closed:
                manager.close_workspace(pool)


def _bind_custom_copy_rows(spec, placement):
    """Validate the copy table and derive the physical loads before binding weights."""
    home, ranks = spec.local_experts, spec.ep_size
    borrowed = placement.borrowed_capacity
    if tuple(placement.counts_by_source.shape) != (ranks, ranks * (home + borrowed)):
        raise ValueError("Expert placement counts must match the EP group and borrowed capacity")
    if not borrowed:
        return None
    loads = placement.counts_by_source.sum(dim=0).view(ranks, home + borrowed)
    rows = torch.cat((placement.copies, loads), dim=1).tolist()
    received = [sum(row[borrowed:]) for row in rows]
    peak, local = max(received), received[spec.rank_id]
    if peak > min(placement.receive_capacity, 2**31 - 1):
        raise ValueError("Expert placement receive capacity is smaller than the routed rows")
    if placement.local_receive_capacity not in (None, local):
        raise ValueError("Expert placement local receive capacity disagrees with the actual routed rows")

    for row in rows:
        for slot, count in enumerate(row[borrowed + home:]):
            owner = row[slot]
            if not -1 <= owner < spec.num_experts or (count > 0 and owner < 0):
                raise ValueError("An active guest needs a valid logical expert in the copy table")
            if count == 0:
                row[slot] = -1
    copies = tuple(tuple(row[:borrowed]) for row in rows)
    if not any(owner >= 0 for row in copies for owner in row):
        return None
    published = max((i + 1 for i, owner in enumerate(copies[spec.rank_id]) if owner >= 0), default=0)
    return replace(placement, copies_by_rank=copies, published_slots=published, route_metadata=None,
                   owner_rows=tuple(rows[spec.rank_id][borrowed:borrowed + home]),
                   receive_capacity=peak, local_receive_capacity=local)


class PlacementResources:
    """Select BF16 routes and own their lazy expert bank."""

    def __init__(self, parent: Any, specification: dict[str, Any], shared: bool, *,
                 bf16_partials: bool = False, rank_nodes: tuple[int, ...] = ()) -> None:
        """Attach lazy placement resources to one existing BF16 owner."""
        self.parent = parent
        self.specification = specification
        self.shared = shared
        self.pool = None
        self.replica_slots_per_rank = specification.get("replica_slots_per_rank")
        self.cross_node_cost = specification.get("cross_node_cost")
        self.receive_capacity_percent = specification.get("receive_capacity_percent")
        self.rank_nodes = rank_nodes
        built_in = _uses_builtin_planner(specification)
        spec = parent.spec
        can_split_weights = supports_bounded_placement(spec)
        self.requires_fused_return = can_split_weights and spec.expert_weight_layout == "EIO"
        supported = built_in and can_split_weights
        self.bounded = (BoundedPlacement(spec, replica_slots_per_rank=self.replica_slots_per_rank,
                                        cross_node_cost=self.cross_node_cost,
                                        nodes=rank_nodes)
                        if supported else None)
        self.bf16_partials = False
        self.prefer_bf16_partials = False
        self.pipeline_enabled = self.dedicated_pipeline = False
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
        self.statistics = {
            "invocations": 0, "placements": 0,
            "route_replays": 0, "route_replay_misses": 0, "skipped_recomputes": 0,
        }

    def uses_placement_wire(self) -> bool:
        """Use the count wire for ordinary per-expert dispatch descriptors."""
        return self.bounded is not None and not self.parent.spec.batch_experts

    def shared_receive_rows(self) -> int | None:
        """Read the committed token workspace used by both ordinary and placed Push routes."""
        if self.parent.spec.dispatch_mode == "push":
            workspace = getattr(self.parent, "workspace", None)
            if workspace is not None:
                return workspace.capacity_floor or None
        return None

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
    ) -> ExpertWeightBinding | None:
        """Choose a virtual route only after the existing count exchange validates it."""
        self.statistics["invocations"] += 1
        spec = self.parent.spec
        total = sum(source_rows)
        limit = receive_limit(self.receive_capacity_percent, total, spec.ep_size)
        mandatory = limit is not None and summary.maximum > limit
        if self.bounded is None:
            request = self.request(hidden, ids, gathered, summary, source_rows, inverse, autograd_mask,
                                   host_rows=host_rows)
            placement = self._select_custom(planner, request, limit, mandatory)
        else:
            training = is_checkpointing() or (autograd_mask & 13) == 13
            prices = bf16_cost(
                spec.hidden_size, spec.intermediate_size, spec.num_cube_cores, training,
                dispatch_mode=spec.dispatch_mode, transport_engine=spec.transport_engine,
            )
            resident = self.shared_receive_rows()
            if prices.keep_home(summary.maximum, total, spec.ep_size, limit=limit, resident=resident):
                return None
            placement = self.bounded.select(
                hidden, ids, gathered[:, 3:], source_rows, inverse, prices, snapshot,
                host_rows=host_rows, limit=limit, resident=resident,
            )
        return self.bind(hidden, placement, tokens=summary.plan_tokens, fused=self.pipeline_enabled,
                         mandatory=mandatory)

    def _select_custom(self, planner, request, limit, mandatory):
        """Converge planner failures and agree on the resources needed by its result."""
        placement, signature, failure = None, None, None
        try:
            placement = planner.plan(request)
            if placement is not None:
                spec = self.parent.spec
                slots = min(spec.local_experts, self.replica_slots_per_rank or spec.local_experts)
                if not isinstance(placement, ExpertPlacement) or placement.borrowed_capacity > slots:
                    raise ValueError(f"Expert placement must fit the {slots} available guest slots")
                if placement.virtual_ids.shape != request.ids.shape or placement.virtual_ids.device != request.ids.device:
                    raise ValueError("Expert placement IDs must match the source Top-K shape and device")
                counts = placement.counts_by_source
                if int(counts.min()) < 0 or not torch.equal(counts.sum(1), request.counts_by_source.sum(1)):
                    raise ValueError("Expert placement must preserve each source rank's routed row count")
                ids = placement.virtual_ids
                if ids.numel() and ((ids < 0) | (ids >= counts.shape[1])).any():
                    raise ValueError("Expert placement IDs exceed the physical expert range")
                placement = _bind_custom_copy_rows(spec, placement)
                if placement is not None:
                    if limit is not None and placement.receive_capacity > limit:
                        raise ValueError(f"Expert placement exceeds receive limit {limit} rows")
                    signature = (placement.borrowed_capacity, placement.receive_capacity, placement.copies_by_rank)
            if mandatory and placement is None:
                raise ValueError(f"No expert placement found within receive limit {limit} rows")
        except Exception as error:  # pylint: disable=broad-exception-caught
            failure = error
        report = None if failure is None else _collective.failure_report(failure)
        results = _collective.exchange(
            (signature, report), group=self.parent.spec.ep_group, failure_prefix="HotReplica planner agreement",
        )
        signatures, failures = zip(*results)
        if any(failures):
            details = _collective.format_collective_failures(failures)
            raise ValueError(f"HotReplica planner rejected input on {details}") from failure
        if any(candidate != signatures[0] for candidate in signatures):
            raise ValueError("HotReplica planner placement differs across EP ranks")
        return placement

    def request(
        self, hidden: torch.Tensor, ids: torch.Tensor, gathered: torch.Tensor, summary: _RouteSummary,
        source_rows: list[int], inverse: torch.Tensor | None, autograd_mask: int, *,
        host_rows: list[list[int]] | None,
    ) -> PlacementRequest:
        """Expose counts and available capacity from one validated MegaMoE route."""
        spec = self.parent.spec
        checkpointing = is_checkpointing()
        key = (
            (autograd_mask, checkpointing, hidden.shape[-1], hidden.element_size(), ids.shape[-1],
             spec.intermediate_size, spec.num_cube_cores, tuple(tuple(row) for row in host_rows))
            if host_rows is not None else None
        )
        return PlacementRequest(
            hidden, ids, gathered[:, 3:], inverse, summary.maximum, sum(source_rows), spec.rank_id,
            spec.intermediate_size, spec.num_cube_cores, checkpointing or bool(autograd_mask & 1),
            checkpointing=checkpointing, apply_probs_before_fc2=spec.apply_probs_before_fc2,
            ordinary_receive_capacity=self.shared_receive_rows(), count_key=key, rank_nodes=self.rank_nodes,
            dispatch_mode=spec.dispatch_mode, transport_engine=spec.transport_engine,
        )

    def bind(
        self, hidden: torch.Tensor, placement: ExpertPlacement | None, *, tokens: int, fused: bool = False,
        mandatory: bool = False,
    ) -> ExpertWeightBinding | None:
        """Use a prepared placement to bind the existing BF16 runtime."""
        if placement is None:
            return None
        spec = self.parent.spec
        experts = spec.num_experts + spec.ep_size * placement.borrowed_capacity
        if tuple(placement.counts_by_source.shape) != (spec.ep_size, experts):
            raise ValueError("Expert placement counts must match the EP group and borrowed capacity")
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
        if self.pool is None:
            source = max(_COMMUNICATION_SPLIT, _align_capacity(tokens))
            preferred = None
            if self.shared and spec.dispatch_mode == "push":
                preferred = _resolve_receive_capacity(
                    spec.initial_capacity_factor, source * spec.top_k, spec.ep_size,
                )
            try:
                self.pool = _get_mega_bank(
                    hidden, self.parent, self, tokens=source, received=placement.receive_capacity,
                    preferred_receive=preferred, pipeline=self.pipeline_enabled,
                )
            except _HeapCapacityError:
                if mandatory:
                    raise
                return None
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
        self.statistics["placements"] += 1
        return ExpertWeightBinding(
            self.pool, placement, spec.expert_weight_layout, fused=bool(slots), fused_slots=slots,
            return_in_kernel=return_in_kernel, bf16_partials=bf16_partials,
        )

    def _uses_fp32_partials(self):
        """Use the same partial-gradient dtype as the ordinary placed launch."""
        spec = self.parent.spec
        return spec.partial_weight_grad_dtype == "float32"

    @property
    def bank_extent(self):
        """Keep the guest partial suffix aligned without overlapping weights."""
        spec = self.parent.spec
        extent = (3 if self._uses_fp32_partials() else 2) * spec.local_experts
        if extent % 2 and self.requires_fused_return and spec.hidden_size * spec.intermediate_size % 2:
            # One BF16 expert record makes the end of both allocations a valid FP32 base.
            extent += 1
        return extent

    def can_close(self) -> bool:
        """Allow close after retained bank users have drained."""
        return self.pool is None or self.pool.can_close()

    def close(self) -> None:
        """Release the placement's ownership of the expert bank."""
        _release_mega_bank(self.parent, self)
