# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare bounded replica quotas for the shared BF16 executor."""

from dataclasses import dataclass, replace
import numpy as np
import torch

from hyper_multicore.modules.mega_moe.route import (
    RouteMetadata, _RouteSummary, _pack_host_route, _summarize_dynamic_rows,
    _upload_route_tensor,
)
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec

from ._planner.capacity import ExpertReplicaConfig
from ._planner.receive import ReplicaCost, select_replica_plan
from .placement import _LoweredPlacement, remap_expert_ids
from .planner import replica_slot_budget
from .render import render_plan


_PINNED_WIRE_MAX_BYTES = 32 * 1024


def supports_bounded_placement(spec):
    """A3 MTE Push and Pull use the same agreed quota plan."""
    return spec.architecture == 2201 and spec.transport_engine == "mte"


@dataclass
class PlacementWire:
    """Validated count agreement and ordinary descriptors for one exact wire."""

    wire: np.ndarray
    summary: _RouteSummary
    source_rows: list[int]
    stream: object
    metadata: RouteMetadata | None = None

    def ordinary_metadata(self, spec, hidden):
        """Build the home descriptor only when the planner keeps the home layout."""
        if self.metadata is None:
            self.metadata = _placement_metadata(self.wire[:, 3:], spec, self.summary.capacity,
                                                self.summary.plan_tokens, hidden.shape[0], hidden.device)
        return self.metadata


def read_placement_wire(payload, spec, workspace, stream):
    """Share one bounded readback between header validation, placement and descriptors."""
    previous = getattr(workspace, "placement_wire", None)
    same_stream = previous is not None and previous.stream == stream
    if (workspace is not None and stream is not None and payload.device.type == "npu"
            and payload.numel() * payload.element_size() <= _PINNED_WIRE_MAX_BYTES):
        # One complete read supplies validation for both steady and changing routes.
        with workspace.lock:
            staged = workspace.placement_readback
            if staged is None or staged[0].shape != payload.shape or staged[0].dtype != payload.dtype:
                staged = (torch.empty(payload.shape, dtype=payload.dtype, device="cpu", pin_memory=True),
                          torch.npu.Event())
                workspace.placement_readback = staged
            host, ready = staged
            host.copy_(payload, non_blocking=True)
            ready.record(stream)
            ready.synchronize()
            wire = host.numpy()
            if same_stream and np.array_equal(previous.wire, wire):
                return previous
            wire = wire.copy()
    else:
        wire = payload.cpu().numpy()
        if same_stream and np.array_equal(previous.wire, wire):
            return previous
        if payload.device.type == "cpu":
            wire = wire.copy()
    counts = wire[:, 3:]
    loads = counts.reshape(spec.ep_size, spec.ep_size, spec.local_experts).sum(axis=(0, 2), dtype=np.int64)
    rows = np.column_stack((wire[:, :3], counts.sum(1, dtype=np.int64), counts.min(1), counts.max(1), loads)).tolist()
    summary = _summarize_dynamic_rows(rows, spec)
    snapshot = PlacementWire(wire, summary, [row[3] for row in rows], stream)
    workspace.placement_wire = snapshot
    return snapshot


def _placement_metadata(counts, spec, capacity, plan_tokens, tokens, device):
    """Build offsets and row counts in the selected dispatch direction's order."""
    rank = spec.rank_id
    ranks, experts = counts.shape
    local = experts // ranks
    columns = slice(rank * local, (rank + 1) * local)
    source = counts.cumsum(axis=1, dtype=np.int64) - counts
    destinations = counts.reshape(ranks, ranks, local).transpose(1, 2, 0).reshape(ranks, -1)
    ends = destinations.cumsum(axis=1, dtype=np.int64)
    targets = (ends - destinations).reshape(ranks, local, ranks)
    receive_source = source[:, columns].reshape(-1)
    receive_target = targets[rank].T.reshape(-1)
    receive_sizes = counts[:, columns].reshape(-1)
    if spec.dispatch_mode == "pull":
        dispatch_source, dispatch_target, dispatch_sizes = receive_source, receive_target, receive_sizes
    else:
        dispatch_source, dispatch_target, dispatch_sizes = source[rank], targets[:, :, rank].reshape(-1), counts[rank]
    wide = (dispatch_source, dispatch_target, receive_target, receive_source, ends[rank, ranks - 1::ranks])
    sizes = (dispatch_sizes, receive_sizes)
    metadata = _pack_host_route(tuple(a.tolist() for a in wide), tuple(a.tolist() for a in sizes),
                                capacity, device, spec.hidden_size)
    return replace(metadata, local_num_tokens=tokens, plan_tokens=plan_tokens)


@dataclass(frozen=True)
class _Geometry:
    """Cached routing tables and capacity shared by calls with equal source counts."""

    sorted_ids: torch.Tensor
    counts: torch.Tensor
    copies: torch.Tensor
    owners: tuple[tuple[int, ...], ...]
    width: int
    peak: int
    local: int
    metadata: RouteMetadata | None
    owner_rows: tuple[int, ...]
    published_slots: int

    def lower(self, hidden, ids, inverse):
        """Remap this call's original Top-K order against the cached geometry."""
        remapped = remap_expert_ids(hidden, ids, self.sorted_ids, inverse)
        return _LoweredPlacement(
            remapped, self.counts, self.copies, self.width, self.peak, self.local,
            self.published_slots, self.owners, self.owner_rows, self.metadata,
        )


def render_geometry(plan, spec, device, maximum, *, borrowed=None, prepare_metadata=True):
    """Pack count rows and only active guest columns in the common route format."""
    config = plan.config
    rank = spec.rank_id
    if borrowed is None:
        borrowed = plan.compact_borrowed_width
    sorted_ids, table, owners = render_plan(plan, rank, borrowed)
    items = (sorted_ids, table, np.asarray(owners, dtype=np.int32))
    ids, counts, copies = (_upload_route_tensor(torch.from_numpy(item), device) for item in items)
    receive = plan.destination_loads
    metadata = None
    if prepare_metadata:
        metadata = _placement_metadata(table, spec, max(1, receive[rank]),
                                       max(128, ((maximum + 127) // 128) * 128), len(items[0]) // spec.top_k, device)
    published = max((slot + 1 for slot, expert in enumerate(owners[rank]) if expert >= 0), default=0)
    return _Geometry(ids, counts, copies, owners, borrowed, max(receive), receive[rank], metadata,
                     plan.destination_counts[rank][:config.home_experts], published)


class BoundedPlacement:
    """Cache the last immutable geometry for one execution resource."""

    def __init__(self, spec: MegaMoeSpec, *, replica_slots_per_rank: int | None = None,
                 cross_node_cost: float | None = None, nodes: tuple[int, ...] = ()) -> None:
        self.spec = spec
        slots = replica_slot_budget(spec.local_experts, replica_slots_per_rank)
        self.config = ExpertReplicaConfig(spec.num_experts, spec.ep_size, slots)
        self.cross_node_cost = cross_node_cost
        self.nodes = nodes
        self.snapshot = None
        self.key = None
        self.geometry = None

    def select(
        self, hidden: torch.Tensor, ids: torch.Tensor, counts: torch.Tensor, source_rows: list[int],
        inverse: torch.Tensor | None, prices: ReplicaCost, snapshot: PlacementWire | None = None,
        *, host_rows: list[list[int]] | None = None, limit: int | None = None,
    ) -> _LoweredPlacement | None:
        """Choose and materialize one plan for the agreed counts and receive budget."""
        spec = self.spec
        if snapshot is not None:
            wire = snapshot.wire[:, 3:]
        elif host_rows is not None:
            wire = np.asarray(host_rows, dtype=np.int32)[:, 3:]
        else:
            wire = counts.cpu().numpy()

        stream = torch.npu.current_stream(hidden.device) if hidden.device.type == "npu" else None
        key = (prices, hidden.device, stream, torch.is_inference_mode_enabled(), limit)
        if self.key != key or not np.array_equal(self.snapshot, wire):
            plan = select_replica_plan(
                wire, self.config, nodes=self.nodes, prices=prices, cross_node_cost=self.cross_node_cost,
                limit=limit,
            )
            maximum = max(source_rows) // spec.top_k
            geometry = None if plan is None else render_geometry(
                plan, spec, hidden.device, maximum, prepare_metadata=not spec.batch_experts,
            )
            self.key, self.snapshot, self.geometry = key, wire.copy(), geometry
        return None if self.geometry is None else self.geometry.lower(hidden, ids, inverse)
