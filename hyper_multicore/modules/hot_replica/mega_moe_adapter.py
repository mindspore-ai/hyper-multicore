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
from hashlib import sha256
import weakref
from typing import Any, TYPE_CHECKING

import torch

from hyper_multicore import _collective

from hyper_multicore.modules.mega_moe.heap_manager import _HeapCapacityError, _WorkspaceLayout
from hyper_multicore.modules.mega_moe.spec import _COMMUNICATION_SPLIT, _align_capacity, _resolve_receive_capacity

from ..base_planner import BasePlanner, PlacementRequest
from ._planner.receive import receive_limit
from .bounded_placement import BoundedPlacement, PlacementWire, render_geometry, supports_bounded_placement
from .checkpoint import is_checkpointing
from .pipeline import ReplicaPipeline, fused_replica_slots, supports_pipeline
from .placement import _LoweredPlacement, validate_lowered_ids, validate_quotas
from .planner import HotReplicaPlanner, replica_slot_budget
from .shmem_transfer import ExpertWeightBinding, ReplicaExecutionMode, SharedExpertPool

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
    dedicated = owner.dedicated_bank
    owner_id = id(owner) if dedicated else None
    extent = owner.bank_extent if pipeline else None
    cube_cores = spec.num_cube_cores
    key = (spec.local_experts, spec.hidden_size, spec.intermediate_size, hidden.dtype, owner_id, extent,
           cube_cores)
    with manager.access():
        banks = _SHARED_BANKS.setdefault(manager, weakref.WeakValueDictionary())
        pool = banks.get(key)
        if pool is None or pool.closed:
            pool = SharedExpertPool(
                hidden, spec.ep_group, spec.local_experts, spec.intermediate_size,
                spec.num_cube_cores, spec.expert_weight_layout, root=manager, access=manager.access,
                pipeline=pipeline and dedicated,
            )
            reservation = {
                "kind": pool.pool_kind, "layout": spec.expert_weight_layout, "local_experts": spec.local_experts,
            }
            signature = (
                pool.pool_kind, spec.local_experts, spec.hidden_size, spec.intermediate_size,
                str(hidden.dtype), spec.expert_weight_layout, cube_cores,
            )
            if pipeline:
                if not dedicated:
                    pool.pipeline = ReplicaPipeline(pool, shared_layouts=True)
                pool.bank_experts = extent
                signature += (pool.bank_experts,)
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


class PlacementResources:
    """Select BF16 routes and own their lazy expert bank."""

    def __init__(self, parent: Any, specification: dict[str, Any], shared: bool, *,
                 bf16_partials: bool = False, rank_nodes: tuple[int, ...] = ()) -> None:
        """Attach lazy placement resources to one existing BF16 owner."""
        self.parent = parent
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
        self.pipeline_enabled = False
        self.dedicated_bank = specification.get("replica_bank") == "dedicated"
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
        self.statistics = {
            "invocations": 0, "placement_candidates": 0, "placements": 0,
            "heap_rejections": 0, "protocol_rejections": 0,
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
        checkpointing = is_checkpointing()
        training = checkpointing or (autograd_mask & 13) == 13
        resident = self.shared_receive_rows()
        if self.bounded is None:
            limit = receive_limit(self.receive_capacity_percent, total, spec.ep_size)
            request = self.request(hidden, gathered, summary, total, training, checkpointing, resident, host_rows)
            placement = self._select_custom(planner, request, hidden, ids, inverse, limit,
                                            mandatory=limit is not None and summary.maximum > limit)
        else:
            prices, limit, keep_home = planner._policy(  # pylint: disable=protected-access
                spec.hidden_size, spec.intermediate_size, spec.num_cube_cores, training,
                spec.transport_engine, summary.maximum, total, spec.ep_size, resident,
            )
            if keep_home:
                return None
            placement = self.bounded.select(
                hidden, ids, gathered[:, 3:], source_rows, inverse, prices, snapshot,
                host_rows=host_rows, limit=limit, resident=resident,
            )
        mandatory = limit is not None and summary.maximum > limit
        return self.bind(hidden, placement, tokens=summary.plan_tokens,
                         mandatory=mandatory)

    def _select_custom(self, planner, request, hidden, ids, inverse, limit, *, mandatory):
        """Converge bad quotas and verify that every EP rank will execute the same dispatch."""
        placement, signature, failure = None, None, None
        try:
            result = planner.plan(request)
            if result is not None:
                spec = self.parent.spec
                slots = spec.local_experts
                if isinstance(planner, HotReplicaPlanner):
                    slots = replica_slot_budget(slots, planner.replica_slots_per_rank)
                plan = validate_quotas(result, request, slots=slots)
                if plan is not None:
                    peak = max(plan.destination_loads)
                    if limit is not None and peak > limit:
                        raise ValueError(f"Expert placement exceeds receive limit {limit} rows")
                    width = plan.config.replica_slots_per_rank
                    geometry = render_geometry(plan, spec, ids.device, 0, borrowed=width, prepare_metadata=False)
                    placement = geometry.lower(hidden, ids, inverse)
                    validate_lowered_ids(ids, placement.virtual_ids, plan, spec.rank_id)
                    counts_digest = sha256(plan._dispatch_counts_bytes).digest()  # pylint: disable=protected-access
                    signature = (width, placement.copies_by_rank, counts_digest)
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

    def request(self, hidden, gathered, summary, total, training, checkpointing, resident, host_rows):
        """Pass immutable logical counts from the count exchange to the custom planner."""
        spec = self.parent.spec
        rows = gathered[:, 3:].tolist() if host_rows is None else [row[3:] for row in host_rows]
        return PlacementRequest(
            tuple(tuple(row) for row in rows), summary.maximum, total, spec.rank_id, hidden.shape[-1],
            spec.intermediate_size, spec.num_cube_cores, training, checkpointing=checkpointing,
            apply_probs_before_fc2=spec.apply_probs_before_fc2, ordinary_receive_capacity=resident,
            rank_nodes=self.rank_nodes, dispatch_mode=spec.dispatch_mode, transport_engine=spec.transport_engine,
        )

    def bind(
        self, hidden: torch.Tensor, placement: _LoweredPlacement | None, *, tokens: int,
        mandatory: bool = False,
    ) -> ExpertWeightBinding | None:
        """Use a prepared placement to bind the existing BF16 runtime."""
        if placement is None:
            return None
        spec = self.parent.spec
        experts = spec.num_experts + spec.ep_size * placement.borrowed_capacity
        if tuple(placement.counts_by_source.shape) != (spec.ep_size, experts):
            raise ValueError("Expert placement counts must match the EP group and borrowed capacity")
        self.statistics["placement_candidates"] += 1
        mode, slots = self._execution_mode(placement)
        if self.requires_fused_return and mode not in (ReplicaExecutionMode.FP32_BAND, ReplicaExecutionMode.BF16_BAND):
            self.statistics["protocol_rejections"] += 1
            if not mandatory:
                return None
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
                self.statistics["heap_rejections"] += 1
                if mandatory:
                    raise
                return None
        if mode is not ReplicaExecutionMode.STAGED and (
                self.pool.pipeline is None or self.pool.blocks != spec.num_cube_cores):
            raise RuntimeError("The selected HotReplica protocol does not match its bank")
        return ExpertWeightBinding(
            self.pool, placement, spec.expert_weight_layout, mandatory=mandatory,
            execution_mode=mode, fused_slots=slots,
        )

    def record_submission(self, mode: ReplicaExecutionMode) -> None:
        """Count a submitted forward using the final execution protocol."""
        self.statistics["placements"] += 1
        mode_key = mode.value + "_bindings"
        self.statistics[mode_key] = self.statistics.get(mode_key, 0) + 1

    def _execution_mode(self, placement):
        """Select the protocol from the bank extent, numerics and active guest slots."""
        if not self.pipeline_enabled or not placement.copies_by_rank:
            return ReplicaExecutionMode.STAGED, None
        spec = self.parent.spec
        extent = self.bank_extent
        if not self.prefer_bf16_partials:
            slots = fused_replica_slots(spec, placement, bank_experts=extent)
            if slots:
                return ReplicaExecutionMode.FP32_BAND, slots
        if self.bf16_partials:
            slots = fused_replica_slots(spec, placement, bank_experts=extent, bf16_partials=True)
            if slots:
                return ReplicaExecutionMode.BF16_BAND, slots
        return ReplicaExecutionMode.WEIGHT_PIPELINE, placement.borrowed_capacity

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
