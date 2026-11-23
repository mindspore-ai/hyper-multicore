# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare bounded replica quotas for the shared BF16 executor."""

from dataclasses import dataclass, replace
from pathlib import Path
import socket

import numpy as np
import torch
import torch.distributed as dist

from hyper_multicore.modules.mega_moe.route import (
    RouteMetadata, _RouteSummary, _pack_host_route, _permute_topk_input, _summarize_dynamic_rows,
    _upload_route_tensor,
)
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec

from ._planner.planner import build_capacity_plan, build_expert_replica_plan
from .counts import route_summary
from .placement import ExpertPlacement
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
    device_wire: torch.Tensor
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
        rows = None
    else:
        rows = route_summary(payload, previous.device_wire if same_stream else None)
        if rows is not None and same_stream and all(row[-1] == 0 for row in rows):
            return previous
        wire = payload.cpu().numpy()
        if same_stream and np.array_equal(previous.wire, wire):
            return previous
    if rows is None:
        counts = wire[:, 3:]
        loads = counts.reshape(spec.ep_size, spec.ep_size, spec.local_experts).sum(axis=(0, 2), dtype=np.int64)
        rows = np.column_stack((wire[:, :3], counts.sum(1, dtype=np.int64), counts.min(1), counts.max(1), loads)).tolist()
    summary = _summarize_dynamic_rows(rows, spec)
    snapshot = PlacementWire(wire, summary, [row[3] for row in rows], stream, payload)
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


def _rank_nodes(group, ranks):
    """Use actual node identity in the agreed EP rank order."""
    try:
        node = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        node = ""
    identities = [None] * ranks
    dist.all_gather_object(identities, node or socket.gethostname(), group=group)
    labels = {identity: index for index, identity in enumerate(dict.fromkeys(identities))}
    return tuple(labels[identity] for identity in identities)


@dataclass(frozen=True)
class _Geometry:
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


def _render_geometry(plan, spec, device, maximum, *, prepare_metadata=True):
    """Pack count rows and only active guest columns in the common route format."""
    config = plan.config
    rank = spec.rank_id
    used = max(plan.replica_slots_used)
    borrowed = min(config.replica_slots_per_rank, 1 << (used - 1).bit_length())
    sorted_ids, table, owners = render_plan(plan, rank, borrowed)
    items = (sorted_ids, table, np.asarray(owners, dtype=np.int32))
    ids, counts, copies = (_upload_route_tensor(torch.from_numpy(item), device) for item in items)
    receive = plan.destination_loads
    metadata = None
    if prepare_metadata:
        metadata = _placement_metadata(table, spec, max(1, receive[rank]),
                                       max(128, ((maximum + 127) // 128) * 128), len(items[0]) // spec.top_k, device)
    return _Geometry(ids, counts, copies, owners, borrowed, max(receive), receive[rank], metadata,
                     plan.destination_counts[rank][:config.home_experts], plan.replica_slots_used[rank])


class BoundedPlacement:
    """Reuse immutable geometry on the same complete count wire and stream."""

    def __init__(self, spec: MegaMoeSpec, *, replica_slots_per_rank: int | None = None,
                 cross_node_cost: float | None = None) -> None:
        """Keep the most recent geometry for one EP execution owner."""
        self.spec = spec
        self.replica_slots_per_rank = min(spec.local_experts, replica_slots_per_rank or spec.local_experts)
        self.cross_node_cost = cross_node_cost
        self.nodes = None
        self.snapshot = None
        self.key = None
        self.geometry = None

    def select(
        self, hidden: torch.Tensor, ids: torch.Tensor, counts: torch.Tensor, source_rows: list[int],
        inverse: torch.Tensor | None, projections: int, snapshot: PlacementWire | None = None,
        *, capacity_target: int | None = None,
        host_rows: list[list[int]] | None = None,
        alternatives=(), score=None,
    ) -> ExpertPlacement | None:
        """Map the current route through reusable quotas for the agreed receive budget."""
        spec = self.spec
        maximum = max(source_rows) // spec.top_k
        if snapshot is not None:
            wire = snapshot.wire[:, 3:]
        elif host_rows is not None:
            wire = np.asarray(host_rows, dtype=np.int32)[:, 3:]
        else:
            wire = counts.cpu().numpy()

        transfers = None
        if capacity_target is not None:
            plan = build_capacity_plan(wire, capacity_target, replica_slots_per_rank=self.replica_slots_per_rank,
                                       alternatives=alternatives, score=score)
            if plan is None or not plan.transfers:
                return None
            # The checkpoint owns this invocation's descriptors. Use the usual
            # Push lowering and avoid a second persistent device geometry cache.
            geometry = _render_geometry(plan, spec, hidden.device, maximum, prepare_metadata=False)
            transfers = plan.maximum_transfer_degree
        else:
            capacity = maximum * spec.ep_size * min(spec.local_experts, spec.top_k)
            stream = torch.npu.current_stream(hidden.device) if hidden.device.type == "npu" else None
            key = (capacity, projections, hidden.device, stream, torch.is_inference_mode_enabled())
            if self.key != key or not np.array_equal(self.snapshot, wire):
                self._build(hidden.device, wire, capacity, projections, key, maximum)
            geometry = self.geometry
        if geometry is None:
            return None
        if (inverse is None and hidden.device.type == "npu" and spec.architecture == 2201
                and 0 < ids.numel() <= 32768 and ids.shape[1] <= 8):
            # Generate the stable inverse over one scalar per token. The full
            # activations are permuted only once, using their final physical IDs.
            _, inverse = _permute_topk_input(hidden.new_zeros((ids.shape[0], 1)), ids.int())
        if inverse is None:
            # Match the historical stable source occurrence numbering without
            # materializing the logical permutation's Top-K token buffer.
            order = torch.argsort(ids.flatten().float(), stable=True)
            remapped = torch.empty_like(geometry.sorted_ids)
            remapped.scatter_(0, order, geometry.sorted_ids)
        else:
            remapped = geometry.sorted_ids.index_select(0, inverse.flatten())
        remapped = remapped.view_as(ids)
        return ExpertPlacement(remapped, geometry.counts, geometry.copies, geometry.width,
                               geometry.peak, geometry.local, transfer_experts=transfers,
                               published_slots=geometry.published_slots,
                               copies_by_rank=geometry.owners,
                               route_metadata=geometry.metadata, owner_rows=geometry.owner_rows)

    def _build(self, device, wire, capacity, projections, key, maximum):
        spec = self.spec
        if self.nodes is None:
            self.nodes = _rank_nodes(spec.ep_group, spec.ep_size)
        budget = self.replica_slots_per_rank
        if spec.dispatch_mode == "pull":
            budget = min(budget, 6)
        plan = build_expert_replica_plan(
            wire.tolist(), budget, minimum_replica_rows=1024,
            minimum_load_reduction=0.25 if projections == 3 else 0.4,
            capacity_limit=capacity, receive_capacity=capacity, rank_nodes=self.nodes,
            transfer_row_credit=spec.intermediate_size * projections // 6,
            cross_node_cost=self.cross_node_cost,
        )
        geometry = _render_geometry(plan, spec, device, maximum) if plan.transfers else None
        self.key, self.snapshot, self.geometry = key, wire.copy(), geometry
