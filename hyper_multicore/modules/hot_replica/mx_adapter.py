# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind aligned Ascend950 routes to the shared MegaMoe MXFP8 executor."""

from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
import torch

from hyper_multicore import _collective

from ..mega_moe import route
from ..mega_moe.heap_manager import _workspace_layout
from ..mega_moe.spec import _align_capacity
from ._planner.admission import optional_peak_gain_is_sufficient, reduction_basis_points
from ._planner.aligned import build_aligned_expert_plan
from ._planner.capacity import ExpertReplicaConfig
from ._planner.mx_peak_planner import (
    build_mx_local_peak_plan, build_mx_peak_plan, localize_mx_guests, mx_guest_transfer_rows,
)
from .bounded_placement import _rank_nodes
from .mega_moe_adapter import PlacementResources, _get_mega_bank
from .mx_gradient import MXGradientPool
from .mx_prefetch import MXWeightPrefetch
from .placement import ExpertPlacement


_MINIMUM_REDUCTION = 0.25
_MINIMUM_ROWS = 8192
_TRAINING_MASK = 1 << 7
_SNAPSHOT_MAX_BYTES = 256 * 1024
_GEOMETRY_CACHE_ENTRIES = 8
_GEOMETRY_CACHE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class MXBindings:
    """Bind the invocation's exact quotas to serial weight and FP32 return storage."""

    placement: ExpertPlacement
    prefetch: MXWeightPrefetch
    gradients: MXGradientPool


@dataclass(frozen=True)
class _MXCountSnapshot:
    """Source counts and expert totals from the same validated route."""

    counts: np.ndarray
    expert_totals: np.ndarray


@dataclass(frozen=True)
class _MXGeometry:
    """Exact device route descriptors and host transfer ownership."""

    sorted_ids: torch.Tensor
    counts: torch.Tensor
    copies: torch.Tensor
    owners: tuple
    loads: tuple
    rows: tuple
    metadata: route.RouteMetadata

    def storage_bytes(self):
        """Count backing storage once for offset views and the packed quota input."""
        views = (self.metadata,) if self.metadata._elements is None else (self.metadata, self.metadata._elements[1])
        storages = {}
        for tensor in (self.sorted_ids, self.counts, self.copies,
                       *(value for view in views for value in vars(view).values() if isinstance(value, torch.Tensor))):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        return sum(storages.values())


def _snapshot_summary(gathered, spec, source_rows):
    """Use the same exact readback for count agreement and an active B2 plan."""
    wire = gathered.cpu().numpy()
    counts = wire[:, 3:]
    expert_totals = counts.sum(0, dtype=np.int64)
    expert_loads = expert_totals.reshape(spec.ep_size, spec.local_experts)
    rows = np.column_stack((wire[:, :3], counts.sum(1, dtype=np.int64), counts.min(1), counts.max(1),
                            expert_loads.sum(1), expert_loads.min(1))).tolist()
    result = route._summarize_dynamic_rows(rows, spec, all_experts_active=rows[spec.rank_id][7] > 0)
    source_rows.extend(row[3] for row in rows)
    return result, _MXCountSnapshot(counts, expert_totals)


def _source_indices(hidden, ids, spec):
    """Match the native permutation order using a single aligned payload tile."""
    if not ids.numel():
        return ids.new_empty(0, dtype=torch.int32), None
    (_, indices), counts = route._permute_and_count(hidden.new_empty((hidden.shape[0], 16)), ids, spec)
    return indices, counts


def _render_geometry(plan, spec, tokens, plan_tokens, device):
    """Materialize the selected aligned quotas in the common Push descriptor."""
    rank, config = spec.rank_id, plan.config
    home, width = config.home_experts, config.slots_per_rank
    owners = tuple(row[home:] for row in plan.slot_to_logical)
    runs = plan.source_runs_for(rank)
    slots = np.fromiter((slot for slot, _ in runs), dtype=np.int32, count=len(runs))
    lengths = np.fromiter((rows for _, rows in runs), dtype=np.int64, count=len(runs))
    if lengths.sum() != tokens * spec.top_k:
        raise ValueError("MX placement quotas do not cover the current source")
    host_counts = np.frombuffer(plan._dispatch_counts_bytes, dtype=np.int32)
    host_copies = np.asarray(owners, dtype=np.int32).reshape(-1)
    packed = np.concatenate((host_counts, host_copies, np.repeat(slots, lengths)))
    uploaded = route._upload_route_tensor(torch.from_numpy(packed), device)
    count_end = host_counts.size
    copy_end = count_end + host_copies.size
    counts = uploaded[:count_end].view(config.ep_size, config.physical_experts)
    copies = uploaded[count_end:copy_end].view(config.ep_size, config.replica_slots_per_rank)
    sorted_ids = uploaded[copy_end:]
    loads = plan.destination_loads
    virtual_spec = replace(spec, num_experts=spec.ep_size * width)
    metadata = route._compute_route_metadata(counts[rank], counts, None, virtual_spec, max(1, loads[rank]))
    metadata = replace(metadata, local_num_tokens=tokens, plan_tokens=plan_tokens)
    return _MXGeometry(sorted_ids, counts, copies, owners, loads, plan.destination_counts[rank], metadata)


class MXPlacementResources(PlacementResources):
    """Own one lazy virtual workspace and its B-slot FP32 return bank per shared layer group."""

    def __init__(self, parent, specification, shared):
        super().__init__(parent, specification, shared)
        spec = parent.spec
        self.config = ExpertReplicaConfig(spec.num_experts, spec.ep_size, spec.local_experts)
        self.nodes = None
        self.prefetch = None
        self._cache_key = None
        self._cached_geometry = None
        self._geometry_cache = OrderedDict()
        self._geometry_cache_bytes = 0
        self._previous_active = False
        self.statistics.update(mx_full_route_readbacks=0, mx_packed_route_readbacks=0,
                               mx_packed_home_routes=0, mx_geometry_hits=0,
                               mx_geometry_misses=0, mx_geometry_cache_peak_bytes=0)

    def _read_route(self, gathered, hidden, stream, source_rows, *, use_snapshot):
        spec, workspace = self.parent.spec, self.parent.workspace
        if use_snapshot:
            if self._previous_active:
                summary, snapshot = _snapshot_summary(gathered, spec, source_rows)
                self.statistics["mx_full_route_readbacks"] += 1
                return summary, None, None, snapshot
            packed = route._read_packed_dynamic_route(gathered, spec, hidden.shape[0], source_rows)
            if packed is not None:
                self.statistics["mx_packed_route_readbacks"] += 1
                return packed[0], None, packed[1], None
        summary, raw, metadata = route._read_cached_route(gathered, spec, workspace, hidden.device,
                                                          source_rows, stream=stream)
        return summary, raw, metadata, None

    def prepare_route(self, hidden, ids, probs, supplied_counts, mask, *, training):
        """Gather logical counts once and materialize only the route selected for this invocation."""
        parent = self.parent
        spec, workspace = parent.spec, parent.workspace
        flat_ids = route._validate_topk_inputs(hidden, ids, probs)
        stream = torch.npu.current_stream(hidden.device)
        inverse = None
        router_counts = supplied_counts is not None
        if hidden.shape[0] * max(spec.hidden_size, spec.top_k) <= 2**31 - 1:
            if not router_counts or self._previous_active:
                inverse, counted = _source_indices(hidden, ids, spec)
                if not router_counts:
                    supplied_counts = counted
        # Checkpoint forward and recomputation use the same training policy.
        mask |= _TRAINING_MASK if training else 0
        counts, payload, _ = route._dynamic_route_payload(
            hidden, flat_ids, spec, supplied_counts, mask, workspace, stream=stream)
        workspace.wait_for_reuse(stream=stream)
        gathered, work = route._start_dynamic_gather(payload, spec)
        if work is not None:
            work.wait()
        source_rows = []
        use_snapshot = (training and spec.auto_tokens and spec.ep_size * spec.num_experts > route._HOST_ROUTE_MAX_COUNTS
                        and gathered.numel() * gathered.element_size() <= _SNAPSHOT_MAX_BYTES)
        summary, raw, metadata, snapshot = self._read_route(gathered, hidden, stream, source_rows,
                                                            use_snapshot=use_snapshot)
        self.statistics["invocations"] += 1
        geometry = self._select(hidden, gathered, summary, raw, source_rows, mask, stream,
                                snapshot=snapshot, use_snapshot=use_snapshot)
        self._previous_active = geometry is not None
        if geometry is None:
            if metadata is not None and use_snapshot:
                self.statistics["mx_packed_home_routes"] += 1
            parent.prepare_auto_workspace(summary.plan_tokens, summary.maximum, hidden)
            permuted = route._permute_topk_input(hidden, ids)
            return route._finalize_dynamic_route(
                permuted, counts, summary, raw, metadata, None, gathered, spec, workspace,
                hidden.device, hidden.shape[0])
        self.statistics["placements"] += 1
        return self._prepare_virtual(hidden, ids, gathered, geometry, summary.plan_tokens, inverse)

    def _select(self, hidden, gathered, summary, raw, source_rows, mask, stream, *, snapshot=None, use_snapshot=False):
        spec = self.parent.spec
        tokens, peak = summary.plan_tokens, summary.maximum
        average = (sum(source_rows) + spec.ep_size - 1) // spec.ep_size
        if not optional_peak_gain_is_sufficient(peak, average, reduction_basis_points(_MINIMUM_REDUCTION), 0):
            return None
        if self.nodes is None:
            self.nodes = _rank_nodes(spec.ep_group, spec.ep_size)
        snapshot_counts = snapshot.counts if snapshot is not None else None
        if use_snapshot and snapshot_counts is None:
            snapshot_counts = gathered.cpu().numpy()[:, 3:]
            self.statistics["mx_full_route_readbacks"] += 1
        if snapshot_counts is None and raw is None:
            raw = gathered.tolist()
        counts = None if snapshot_counts is not None else tuple(tuple(row[3:]) for row in raw)
        training = bool(mask & _TRAINING_MASK)
        credit = spec.intermediate_size // 2 if training else 0
        use_peak_policy = (spec.ep_size, spec.num_experts, spec.hidden_size, spec.intermediate_size,
                           spec.top_k, tokens) == (16, 96, 5120, 1792, 8, 4096)
        key = (snapshot_counts.tobytes() if snapshot_counts is not None else counts,
               credit, use_peak_policy, tokens, hidden.shape[0], stream)
        cache = self._geometry_cache if use_snapshot else None
        if cache is not None and key in cache:
            cache.move_to_end(key)
            self.statistics["mx_geometry_hits"] += 1
            return cache[key][0]
        if self._cache_key != key:
            if counts is None:
                counts = tuple(map(tuple, snapshot_counts.tolist()))
            capacity = tokens * spec.ep_size * spec.top_k
            if use_peak_policy:
                plan = build_mx_peak_plan(counts, self.config, capacity=capacity, minimum_rows=_MINIMUM_ROWS,
                                          minimum_reduction=_MINIMUM_REDUCTION, row_credit=credit, nodes=self.nodes,
                                          drain_peak_plateau=True, admit_small_transfer_plan=training)
                if training:
                    plan = localize_mx_guests(plan, self.nodes, mx_guest_transfer_rows(spec.intermediate_size))
            elif training:
                plan = build_mx_local_peak_plan(
                    counts, self.config, capacity=capacity, minimum_rows=_MINIMUM_ROWS,
                    minimum_reduction=_MINIMUM_REDUCTION, row_credit=credit, nodes=self.nodes,
                    transfer_rows=mx_guest_transfer_rows(spec.intermediate_size),
                    validated_expert_totals=tuple(snapshot.expert_totals.tolist()) if snapshot is not None else None,
                )
            else:
                plan = build_aligned_expert_plan(
                    counts, self.config, capacity_limit=capacity, minimum_replica_rows=_MINIMUM_ROWS,
                    minimum_load_reduction=_MINIMUM_REDUCTION, transfer_row_credit=credit, rank_nodes=self.nodes)
            self._cached_geometry = (_render_geometry(plan, spec, hidden.shape[0], tokens, hidden.device)
                                     if plan.transfers else None)
            self._cache_key = key
            if cache is not None:
                self._retain_geometry(key, self._cached_geometry)
                self.statistics["mx_geometry_misses"] += 1
        return self._cached_geometry

    def _retain_geometry(self, key, geometry):
        size = geometry.storage_bytes() if geometry is not None else 0
        if size > _GEOMETRY_CACHE_BYTES:
            return
        cache = self._geometry_cache
        while len(cache) >= _GEOMETRY_CACHE_ENTRIES or self._geometry_cache_bytes + size > _GEOMETRY_CACHE_BYTES:
            _, (_, released) = cache.popitem(last=False)
            self._geometry_cache_bytes -= released
        cache[key] = geometry, size
        self._geometry_cache_bytes += size
        self.statistics["mx_geometry_cache_peak_bytes"] = max(self.statistics["mx_geometry_cache_peak_bytes"],
                                                               self._geometry_cache_bytes)

    def close(self):
        super().close()
        self._geometry_cache.clear()
        self._geometry_cache_bytes = 0
        self._cache_key = self._cached_geometry = None

    def _prepare_virtual(self, hidden, ids, gathered, geometry, tokens, inverse):
        spec = self.parent.spec
        if inverse is None:
            inverse, _ = _source_indices(hidden, ids, spec)
        virtual_ids = geometry.sorted_ids.index_select(0, inverse.flatten()).view_as(ids)
        peak, local = max(geometry.loads), geometry.loads[spec.rank_id]
        placement = ExpertPlacement(virtual_ids, geometry.counts, geometry.copies, self.config.replica_slots_per_rank,
                                    peak, local_receive_capacity=local, copies_by_rank=geometry.owners,
                                    owner_rows=geometry.rows)
        parent = self.parent
        workspace = parent.workspace
        # Materialize the child counters and FP32 bank before the shared receive allocation.
        # At first use this leaves the growing receive buffer at the end of the heap.
        resources = self._virtual_resources(hidden, tokens, peak)
        resources.prepare_auto_workspace(tokens, peak, hidden)
        receive_capacity = resources.workspace.expert_capacity
        if (workspace.expert_buffer is None or receive_capacity > workspace.expert_capacity
                or tokens * spec.top_k > workspace.routed_slots):
            parent.prepare_auto_workspace(tokens, receive_capacity, hidden)
        permuted = route._permute_topk_input(hidden, virtual_ids)
        prepared = route._finalize_dynamic_route(
            permuted, geometry.counts[spec.rank_id], route._RouteSummary(max(1, local), tokens, peak), None,
            geometry.metadata, None, gathered, resources.spec, resources.workspace, hidden.device, hidden.shape[0])
        return replace(prepared, execution_resources=resources,
                       weight_binding=MXBindings(placement, self.prefetch, self.pool))

    def _virtual_resources(self, hidden, tokens, received):
        """Reserve only the selected capacity, together with its FP32 banks."""
        from ..mega_moe.backends.mxfp8.backend import Resources  # pylint: disable=import-outside-toplevel

        width = self.config.replica_slots_per_rank
        if width in self.geometries:
            return self.geometries[width]
        parent = self.parent
        spec = parent.spec
        specification = {**self.specification, "hot_replica": False, "planner_key": None,
                         "num_experts": self.config.physical_experts, "_borrows_routed_buffer": True}
        layout = _workspace_layout(specification, hidden.dtype, _align_capacity(received), tokens,
                                   borrows_receive_buffer=True)
        manager = parent.heap_manager
        pool = _get_mega_bank(hidden, parent, self, pool_type=MXGradientPool,
                              reservations=((specification, layout),))
        resources = prefetch = None

        def allocate_prefetch():
            nonlocal prefetch
            prefetch = MXWeightPrefetch(spec.ep_group, hidden.device, self.config.home_experts, width)

        try:
            resources = Resources(specification, hidden, expert_weight_layout="EIO",
                                  shared=self.shared, active_specifications=(specification,),
                                  workspace_layout=layout, routed_buffer_owner=parent.workspace)
            _collective.collective_call(allocate_prefetch, stage="MXFP8 HotReplica weight buffers", group=spec.ep_group)
        except Exception:
            if resources is not None:
                resources.close()
            manager.close_workspace(pool)
            manager.rollback_reservations((specification,), None)
            raise
        self.pool = pool
        self.prefetch = prefetch
        self.geometries[width] = resources
        return resources
