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
"""Prepare trusted Router output for managed Torch MegaMoe execution."""

from __future__ import annotations

from array import array
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist

from hyper_multicore._loader import load_torch_npu
from hyper_multicore.torch import ops as multicore_ops

from .spec import MegaMoeSpec, _align_capacity

if TYPE_CHECKING:
    from .workspace import MegaMoeWorkspace


# Avoid large Host transfers and Python boxing when the gathered count table grows.
_HOST_ROUTE_MAX_COUNTS = 2048
# Preserve the smaller descriptor budget when counts change on every call.
_HOST_DESCRIPTOR_MAX_COUNTS = 1024
_ROUTE_CACHE_REPROBE_STEPS = 8
_HOST_ROUTE_CACHE_ENTRIES = 8
_METADATA_ALIGNMENT = 512
_PULL_STAGING_MAX_BYTES = 64 * 1024 * 1024

RouteSelector = Callable[[torch.Tensor, int, int, Optional[int]], bool]


@dataclass(frozen=True)
class RouteMetadata:
    """Exact offsets and counts consumed by the low-level native ABI."""

    dispatch_src_off: torch.Tensor
    dispatch_target_off: torch.Tensor
    dispatch_size: torch.Tensor
    combine_src_off: torch.Tensor
    combine_target_off: torch.Tensor
    combine_size: torch.Tensor
    group_list: torch.Tensor
    expert_capacity: int
    local_num_tokens: int | None = None
    plan_tokens: int | None = None
    all_experts_active: bool = False
    _elements: tuple[int, RouteMetadata] | None = field(default=None, repr=False, compare=False)

    def for_kernel(self, hidden_size: int) -> RouteMetadata:
        """Convert row descriptors once; keep the public row-unit contract unchanged."""
        if hidden_size == 1:
            return self
        if self._elements is not None and self._elements[0] == hidden_size:
            return self._elements[1]
        return replace(
            self,
            dispatch_src_off=self.dispatch_src_off * hidden_size,
            dispatch_target_off=self.dispatch_target_off * hidden_size,
            dispatch_size=self.dispatch_size * hidden_size,
            combine_src_off=self.combine_src_off * hidden_size,
            combine_target_off=self.combine_target_off * hidden_size,
            combine_size=self.combine_size * hidden_size,
            _elements=None,
        )


@dataclass(frozen=True)
class PreparedTopKRoute:
    """Permuted payload and exact metadata for one Router decision."""

    routed_tokens: torch.Tensor
    unpermute_mapping: torch.Tensor
    tokens_per_expert: torch.Tensor
    received_counts: torch.Tensor
    metadata: RouteMetadata
    maximum_received_slots: int = 0


def _record_metadata_stream(metadata: RouteMetadata, stream: Any) -> None:
    """Retain row descriptors and cached element descriptors on their consumer stream."""
    views = (metadata,) if metadata._elements is None else (metadata, metadata._elements[1])
    for view in views:
        for value in vars(view).values():
            if isinstance(value, torch.Tensor):
                value.record_stream(stream)


@dataclass(frozen=True)
class _HostRouteCache:
    """One validated wire table and immutable route descriptors for a workspace."""

    signature: tuple[Any, ...]
    raw: list[list[int]]
    summary: tuple[int, int, int]
    metadata: RouteMetadata | None
    source_totals: tuple[int, ...] = ()
    stream: Any = None
    ready: Any = None

    def wait(self, device: Any, stream: Any = None) -> None:
        """Order a different consumer stream after the descriptor upload.

        Args:
            device: Device whose current stream consumes the cached descriptors.
        """
        if self.ready is not None and self.metadata is not None:
            current = stream if stream is not None else torch.npu.current_stream(device)
            if current != self.stream:
                current.wait_event(self.ready)
                _record_metadata_stream(self.metadata, current)


@dataclass(frozen=True)
class _RouteHeaderCache:
    """One immutable control header, retained across identical local invocations."""

    key: tuple[Any, ...]
    tensor: torch.Tensor | None
    stream: Any = None
    ready: Any = None

    def wait(self, device: Any, stream: Any = None) -> None:
        """Preserve upload ordering and allocation lifetime on another stream.

        Args:
            device: Device whose current stream consumes the cached header.
        """
        if self.ready is not None and self.tensor is not None:
            current = stream if stream is not None else torch.npu.current_stream(device)
            if current != self.stream:
                current.wait_event(self.ready)
                self.tensor.record_stream(current)


@dataclass(frozen=True)
class _PlannedRouteCache:
    """Native descriptors for an owned, immutable planner-count snapshot."""

    source: torch.Tensor
    local_counts: torch.Tensor
    received_counts: torch.Tensor
    metadata: RouteMetadata
    stream: Any = None
    ready: Any = None

    def wait(self, stream: Any) -> None:
        if self.ready is not None and stream != self.stream:
            stream.wait_event(self.ready)
            self.source.record_stream(stream)
            _record_metadata_stream(self.metadata, stream)


def _validate_topk_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Validate route tensor metadata without synchronizing tensor contents."""
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}.")
    if not topk_weights.is_floating_point():
        raise TypeError(
            f"topk_weights must be floating point, got {topk_weights.dtype}."
        )
    if (
        topk_ids.device != hidden_states.device
        or topk_weights.device != hidden_states.device
    ):
        raise ValueError(
            "hidden_states, topk_ids and topk_weights must be on one device."
        )
    return topk_ids.reshape(-1)


def _resolve_counts(
    flat_ids: torch.Tensor,
    supplied_counts: torch.Tensor | None,
    spec: MegaMoeSpec,
) -> torch.Tensor:
    """Use trusted Router counts or compute them only when omitted."""
    if supplied_counts is None:
        return (
            torch.bincount(flat_ids.to(torch.int64), minlength=spec.num_experts)
            .to(torch.int32)
            .contiguous()
        )
    if tuple(supplied_counts.shape) != (spec.num_experts,):
        raise ValueError(
            f"tokens_per_expert must have shape ({spec.num_experts},), "
            f"got {tuple(supplied_counts.shape)}."
        )
    if supplied_counts.device != flat_ids.device:
        raise ValueError("tokens_per_expert must be on the route tensor device.")
    if supplied_counts.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"tokens_per_expert must be int32 or int64, got {supplied_counts.dtype}."
        )
    if supplied_counts.dtype == torch.int32 and supplied_counts.is_contiguous():
        return supplied_counts
    return supplied_counts.to(torch.int32).contiguous()


def _start_count_gather(
    counts: torch.Tensor,
    spec: MegaMoeSpec,
) -> tuple[torch.Tensor, Any | None]:
    """Start the rank-major INT32 count gather used by route construction."""
    if spec.ep_size == 1:
        return counts.reshape(1, spec.num_experts), None
    gathered_counts = torch.empty(
        spec.ep_size * spec.num_experts,
        dtype=torch.int32,
        device=counts.device,
    )
    work = dist.all_gather_into_tensor(
        gathered_counts,
        counts,
        group=spec.ep_group,
        async_op=True,
    )
    return gathered_counts.reshape(spec.ep_size, spec.num_experts), work


def _finish_count_gather(
    counts_by_source: torch.Tensor,
    work: Any | None,
    spec: MegaMoeSpec,
) -> torch.Tensor:
    """Wait for gathered counts before reading route metadata."""
    if work is not None:
        work.wait()
    local_start = spec.rank_id * spec.local_experts
    local_end = local_start + spec.local_experts
    return counts_by_source[:, local_start:local_end].contiguous()


def _expert_capacity(
    counts_by_source: torch.Tensor,
    spec: MegaMoeSpec,
) -> tuple[int, int]:
    """Validate the global bound and size rank-local computation tensors."""
    destination_loads = counts_by_source.reshape(
        spec.ep_size,
        spec.ep_size,
        spec.local_experts,
    ).sum(dim=(0, 2), dtype=torch.int32)
    # One host transfer serves both the coordinated overflow check and local
    # allocation; the existing gathered counts require no extra collective.
    loads = destination_loads.tolist()
    # Keep a non-null ABI argument on ranks whose experts receive no tokens.
    # Source outputs and symmetric communication buffers retain their own sizes.
    return max(1, loads[spec.rank_id]), max(loads)


def _permute_and_count(hidden: torch.Tensor, ids: torch.Tensor, spec: MegaMoeSpec):
    """Produce the ordinary push buffer, inverse map and histogram in one NPU launch."""
    rows, inverse, counts, _ = load_torch_npu().npu_moe_init_routing_v2(
        hidden, ids.int(), active_num=ids.numel(), expert_num=spec.num_experts,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[0, spec.num_experts], row_idx_type=0,
    )
    return (rows, inverse.reshape(-1).int().contiguous()), counts.int().contiguous()


def _can_fuse_count(hidden: torch.Tensor, spec: MegaMoeSpec, supplied_counts) -> bool:
    return (hidden.device.type == "npu" and spec.dispatch_mode == "push" and
            supplied_counts is None and hidden.shape[0] > 0)


def _permute_topk_input(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand hidden rows with the native NPU MoE permutation operator."""
    if hidden_states.shape[0] == 0:
        return hidden_states, torch.empty(0, dtype=torch.int32, device=hidden_states.device)
    routed_tokens, unpermute_mapping = load_torch_npu().npu_moe_token_permute(
        hidden_states,
        topk_ids,
    )
    return (
        routed_tokens,
        unpermute_mapping.reshape(-1).to(torch.int32).contiguous(),
    )


def _permute_topk_input_out(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the inverse mapping owned while writing rows into leased SHMEM."""
    mapping = torch.empty(topk_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    multicore_ops.moe_token_permute_out(hidden_states.contiguous(), topk_ids.contiguous(), output, mapping)
    return output, mapping


def _compute_route_metadata(
    counts: torch.Tensor,
    counts_by_source: torch.Tensor,
    received_counts: torch.Tensor | None,
    spec: MegaMoeSpec,
    expert_capacity: int,
) -> RouteMetadata:
    """Build exact element offsets without padding or route truncation."""
    if counts_by_source.device.type == "npu" and counts_by_source.dtype == torch.int32:
        from hyper_multicore.moonep.allocator import _device_ub_bytes
        ub = _device_ub_bytes(counts_by_source.device.index)
        e,p,l = spec.num_experts,spec.ep_size,spec.local_experts
        stride=(e+31)//32*32
        span = (p - 1) * counts_by_source.stride(0) + e
        if (counts_by_source.stride(1) == 1 and counts_by_source.stride(0) >= e
                and ((span+7)//8*8+3*stride+2*((l+7)//8*8))*4 <= ub//4*3):
            off,sz=multicore_ops.route_offsets(counts_by_source, spec.rank_id, ub, spec.hidden_size)
            pull = spec.dispatch_mode == "pull"
            offset_count = 4 * stride + l
            # Split once in ATen instead of dispatching a Python slice for
            # every field. Both row and element descriptors remain public views
            # of this invocation's storage; no cache can outlive its contents.
            units = 1 if off.numel() == offset_count else 2
            offsets = off.split((e, stride - e) * 4 + (l,) if units == 1
                                else ((e, stride - e) * 4 + (l,)) * 2)
            sizes = sz.split((e, stride - e) * (2 * units))

            def describe(unit):
                o, s = 9 * unit, 4 * unit
                src, target, length = offsets[o + 4], offsets[o + 6], sizes[s + 2]
                return RouteMetadata(
                    target if pull else offsets[o], src if pull else offsets[o + 2],
                    length if pull else sizes[s], src, target, length,
                    offsets[o + 8], expert_capacity,
                )

            rows = describe(0)
            if spec.hidden_size == 1:
                return rows
            return replace(rows, _elements=(spec.hidden_size, describe(1)))

    if received_counts is None:
        start = spec.rank_id * spec.local_experts
        received_counts = counts_by_source[:, start:start + spec.local_experts].contiguous()
    counts_i32 = counts.to(dtype=torch.int32).contiguous()
    source_prefix = counts_by_source.cumsum(dim=1, dtype=torch.int64) - counts_by_source
    destination_layout = (
        counts_by_source.reshape(
            spec.ep_size,
            spec.ep_size,
            spec.local_experts,
        )
        .permute(1, 2, 0)
        .contiguous()
    )
    destination_flat = destination_layout.reshape(spec.ep_size, -1)
    destination_prefix = (
        destination_flat.cumsum(dim=1, dtype=torch.int64) - destination_flat
    ).reshape(spec.ep_size, spec.local_experts, spec.ep_size)
    local_destination_prefix = destination_prefix[spec.rank_id]
    local_destination_counts = destination_layout[spec.rank_id]
    local_start = spec.rank_id * spec.local_experts
    local_end = local_start + spec.local_experts
    return RouteMetadata(
        dispatch_src_off=(source_prefix[:, local_start:local_end].reshape(-1).to(torch.int64)
                          if spec.dispatch_mode == "pull" else source_prefix[spec.rank_id].to(torch.int64)),
        dispatch_target_off=(
            local_destination_prefix.T.reshape(-1).to(torch.int64) if spec.dispatch_mode == "pull"
            else destination_prefix[:, :, spec.rank_id].reshape(-1).to(torch.int64)
        ),
        dispatch_size=(received_counts.reshape(-1).to(torch.int32) if spec.dispatch_mode == "pull" else counts_i32),
        combine_src_off=local_destination_prefix.T.reshape(-1).to(torch.int64),
        combine_target_off=(
            source_prefix[:, local_start:local_end].reshape(-1).to(torch.int64)
        ),
        combine_size=received_counts.reshape(-1).to(torch.int32),
        group_list=(
            local_destination_prefix[:, -1] + local_destination_counts[:, -1]
        ).to(torch.int64),
        expert_capacity=expert_capacity,
    )


def _upload_route_tensor(tensor: torch.Tensor, device: Any) -> torch.Tensor:
    """Keep Host metadata alive through the pinned allocator during asynchronous upload."""
    if torch.device(device).type == "npu":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def _host_route_metadata(
    raw: list[list[int]],
    spec: MegaMoeSpec,
    capacity: int,
    device: Any,
) -> RouteMetadata:
    """Pack the native INT64 offsets and INT32 route sizes in one retained allocation."""
    counts = [row[3:] for row in raw]
    local, rank = spec.local_experts, spec.rank_id
    start = rank * local
    prefix, destinations, destination_prefix = _host_route_prefixes(counts, spec)
    source_offsets = [
        prefix[source][expert] for source in range(spec.ep_size) for expert in range(start, start + local)
    ]
    target_offsets = [
        destination_prefix[rank][expert * spec.ep_size + source]
        for source in range(spec.ep_size)
        for expert in range(local)
    ]
    received = [counts[source][expert] for source in range(spec.ep_size) for expert in range(start, start + local)]
    if spec.dispatch_mode == "pull":
        dispatch_source, dispatch_target, dispatch_sizes = source_offsets, target_offsets, received
    else:
        dispatch_source = prefix[rank]
        dispatch_target = [
            destination_prefix[target][expert * spec.ep_size + rank]
            for target in range(spec.ep_size)
            for expert in range(local)
        ]
        dispatch_sizes = counts[rank]
    groups = [
        destination_prefix[rank][(expert + 1) * spec.ep_size - 1] + destinations[rank][(expert + 1) * spec.ep_size - 1]
        for expert in range(local)
    ]
    wide = (dispatch_source, dispatch_target, target_offsets, source_offsets, groups)
    return _pack_host_route(wide, (dispatch_sizes, received), capacity, device, spec.hidden_size)


def _host_route_prefixes(
    counts: list[list[int]], spec: MegaMoeSpec,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Scan each source and each destination independently in native layout order."""
    local = spec.local_experts
    prefix = [list(accumulate(row, initial=0))[:-1] for row in counts]
    destinations = [
        [
            counts[source][expert]
            for expert in range(target * local, (target + 1) * local)
            for source in range(spec.ep_size)
        ]
        for target in range(spec.ep_size)
    ]
    destination_prefix = [list(accumulate(row, initial=0))[:-1] for row in destinations]
    return prefix, destinations, destination_prefix


def _pack_host_route(
    wide: tuple[list[int], ...],
    size_fields: tuple[list[int], ...],
    capacity: int,
    device: Any,
    hidden_size: int = 1,
) -> RouteMetadata:
    """Upload row and native element descriptors in one retained allocation."""
    fields = [*wide, *size_fields]
    dtypes = [torch.int64] * len(wide) + [torch.int32] * 2
    element_fields = [[value * hidden_size for value in values] for values in (*wide[:4], *size_fields)]
    fields += element_fields
    dtypes += [torch.int64] * 4 + [torch.int32] * 2
    storage = bytearray()
    slices = []
    for values, dtype in zip(fields, dtypes):
        encoded = array("q" if dtype == torch.int64 else "i", values).tobytes()
        start = len(storage)
        storage.extend(encoded)
        slices.append((start, len(storage), dtype))
        storage.extend(bytes((-len(storage)) % _METADATA_ALIGNMENT))
    buffer = _upload_route_tensor(torch.frombuffer(storage, dtype=torch.uint8), device)
    tensors = [buffer[start:end].view(dtype) for start, end, dtype in slices]
    offsets, sizes = tensors[:5], tensors[5:7]
    all_experts_active = all(end > start for start, end in zip([0, *wide[4]], wide[4]))
    native = RouteMetadata(
        dispatch_src_off=tensors[7], dispatch_target_off=tensors[8], dispatch_size=tensors[11],
        combine_src_off=tensors[9], combine_target_off=tensors[10], combine_size=tensors[12],
        group_list=offsets[4], expert_capacity=capacity, all_experts_active=all_experts_active,
    )
    return RouteMetadata(
        dispatch_src_off=offsets[0],
        dispatch_target_off=offsets[1],
        dispatch_size=sizes[0],
        combine_src_off=offsets[2],
        combine_target_off=offsets[3],
        combine_size=sizes[1],
        group_list=offsets[4],
        expert_capacity=capacity,
        all_experts_active=all_experts_active,
        _elements=(hidden_size, native),
    )


def prepare_topk_route(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    spec: MegaMoeSpec,
    tokens_per_expert: torch.Tensor | None,
    workspace: MegaMoeWorkspace | None = None,
    *,
    autograd_mask: int = 0,
    prepare_workspace: Callable[[int, int], None] | None = None,
    select_route: RouteSelector | None = None,
    discard_rejected: bool = False,
    validated_summary: tuple[int, int, int] | None = None,
    planned_counts: torch.Tensor | None = None,
    select_placement: Callable | None = None,
    with_source_permutation: bool = False,
) -> PreparedTopKRoute | None:
    """Overlap Top-K permutation with count exchange and build route metadata.

    Args:
        hidden_states: Flattened token states to route.
        topk_ids: Global experts selected for every token.
        topk_weights: Router weights paired with ``topk_ids``.
        spec: Bound shape and expert-parallel specification.
        tokens_per_expert: Optional trusted Router histogram.
        workspace: Optional workspace ordered before the count exchange. Pull
            requires a lease acquired by the caller or the automatic-capacity callback.
        autograd_mask: Group-consistent gradient participation flags in dynamic mode.
        prepare_workspace: Capacity callback, run before pull publication.
        select_route: Optional placement decision after count agreement. Returning False
            keeps the original route in ordinary memory without growing its workspace.
        discard_rejected: Return None once the selector commits to an alternative
            route, so unused native offsets and per-expert prefixes are never built.
        validated_summary: Internal lossless-placement contract (local receive bound,
            agreed plan tokens, global receive bound). The caller must already have
            validated source lengths/gradient participation and the placement bound.

    Returns:
        Permuted tokens and exact native route metadata.
    """
    flat_ids = _validate_topk_inputs(hidden_states, topk_ids, topk_weights)
    if select_route is not None and select_placement is not None:
        raise ValueError("Only one route selector may be supplied")
    with_source_rows = select_placement is not None
    if with_source_rows:
        select_route = select_placement
    if planned_counts is not None:
        if select_route is not None or validated_summary is None:
            raise ValueError("Planner counts require a validated, selected placement")
        return _prepare_planned_route(hidden_states, topk_ids, spec, tokens_per_expert,
                                      workspace, prepare_workspace, validated_summary, planned_counts)

    if spec.dynamic_tokens:
        return _prepare_dynamic_route(hidden_states, topk_ids, flat_ids, spec, tokens_per_expert,
                                      workspace, autograd_mask, prepare_workspace, select_route, discard_rejected,
                                      validated_summary, with_source_rows, with_source_permutation)
    permuted = None
    if _can_fuse_count(hidden_states, spec, tokens_per_expert):
        permuted, counts = _permute_and_count(hidden_states, topk_ids, spec)
    else:
        counts = _resolve_counts(flat_ids, tokens_per_expert, spec)
    permutation_output = None
    if workspace is not None:
        if spec.dispatch_mode == "pull" and select_route is None:
            if not workspace.in_use or workspace.source_buffer is None:
                raise RuntimeError("Pull permutation requires an initialized, claimed workspace.")
            permutation_output = workspace.source_buffer
        else:
            workspace.wait_for_reuse()
    counts_by_source, count_work = _start_count_gather(counts, spec)
    if permuted is not None:
        routed_tokens, unpermute_mapping = permuted
    elif permutation_output is None:
        routed_tokens, unpermute_mapping = _permute_topk_input(hidden_states, topk_ids)
    else:
        routed_tokens, unpermute_mapping = _permute_topk_input_out(hidden_states, topk_ids, permutation_output)
    received_counts = _finish_count_gather(counts_by_source, count_work, spec)
    expert_capacity, maximum_received_slots = _expert_capacity(counts_by_source, spec)
    if select_route is not None:
        args = (counts_by_source, spec.plan_tokens, maximum_received_slots,
                spec.local_num_tokens * spec.ep_size * spec.top_k)
        selected = (select_route(*args, (spec.local_num_tokens * spec.top_k,) * spec.ep_size, None,
                                 **({"source_inverse": unpermute_mapping} if with_source_permutation else {}))
                    if with_source_rows else select_route(*args))
        if not selected and discard_rejected:
            return None
        if selected and prepare_workspace is not None:
            prepare_workspace(spec.plan_tokens, maximum_received_slots)
            if spec.dispatch_mode == "pull":
                routed_tokens, unpermute_mapping = _publish_staged_pull(
                    (routed_tokens, unpermute_mapping), hidden_states, spec, workspace)
    metadata = _compute_route_metadata(counts, counts_by_source, received_counts, spec, expert_capacity)
    if select_route is not None:
        metadata = replace(metadata, local_num_tokens=hidden_states.shape[0], plan_tokens=spec.plan_tokens)
    return PreparedTopKRoute(
        routed_tokens=routed_tokens,
        unpermute_mapping=unpermute_mapping,
        tokens_per_expert=counts,
        received_counts=received_counts,
        maximum_received_slots=maximum_received_slots,
        metadata=metadata,
    )


def _valid_route_counts(row: list[int], top_k: int) -> bool:
    """Validate one source histogram against its actual distinct Top-K rows."""
    return row[3] == row[1] * top_k and row[4] >= 0 and row[5] <= row[1]


def _validate_dynamic_sizes(rows: list[list[int]], spec: MegaMoeSpec) -> None:
    """Keep native size and legacy bound rejection ahead of route publication."""
    flexible = spec.auto_tokens
    if flexible and any(row[0] or row[1] < 0 for row in rows):
        raise RuntimeError("MegaMoe dynamic route exceeds native INT32 size limits")
    if not flexible and any(
            row[0] or not 0 <= row[1] <= spec.max_local_num_tokens for row in rows):
        raise RuntimeError(f"MegaMoe token capacity overflow: limit={spec.max_local_num_tokens}, "
                           f"actual={[row[1] for row in rows]}")


def _validate_dynamic_rows(rows: list[list[int]], spec: MegaMoeSpec) -> None:
    """Validate every rank before any automatic allocation or SHMEM write."""
    _validate_dynamic_sizes(rows, spec)
    if any(row[2] != rows[0][2] for row in rows):
        raise RuntimeError("MegaMoe autograd participation differs across EP ranks")
    if any(not _valid_route_counts(row, spec.top_k) for row in rows):
        raise RuntimeError("MegaMoe counts must describe the actual distinct Top-K routes")


def _dynamic_summary(payload: torch.Tensor, spec: MegaMoeSpec, source_rows: list | None = None) -> tuple[int, int, int]:
    """Use one Host summary for shape, participation, receive limits, and schedule."""
    counts = payload[:, 3:]
    loads = counts.reshape(spec.ep_size, spec.ep_size, spec.local_experts).sum(dim=(0, 2))
    summary = torch.cat((payload[:, :3], counts.sum(1, keepdim=True),
                         counts.amin(1, keepdim=True), counts.amax(1, keepdim=True), loads[:, None]), dim=1)
    rows = summary.tolist()
    result = _summarize_dynamic_rows(rows, spec)
    if source_rows is not None:
        source_rows.extend(row[3] for row in rows)
    return result


def _summarize_dynamic_rows(rows: list[list[int]], spec: MegaMoeSpec) -> tuple[int, int, int]:
    """Keep overflow and participation checks common to both metadata policies."""
    _validate_dynamic_rows(rows, spec)
    maximum = max(row[6] for row in rows)
    maximum_tokens = max(row[1] for row in rows) if spec.auto_tokens else spec.local_num_tokens
    maximum_capacity = _align_capacity(spec.ep_size * maximum_tokens * spec.top_k)
    if maximum > maximum_capacity:
        raise RuntimeError(
            f"MegaMoe receive capacity overflow: configured_capacity={maximum_capacity}, actual_maximum={maximum}"
        )
    plan_tokens = max(128, _align_capacity(max(row[1] for row in rows)))
    if plan_tokens * max(spec.hidden_size, spec.top_k) > 2**31 - 1 or maximum > 2**31 - 1:
        raise RuntimeError("MegaMoe dynamic route exceeds native INT32 size limits")
    return max(1, rows[spec.rank_id][6]), plan_tokens, maximum


def _read_dynamic_route(
    payload: torch.Tensor,
    spec: MegaMoeSpec,
    source_rows: list | None = None,
) -> tuple[tuple[int, int, int], list[list[int]] | None]:
    """Reuse the required readback for small descriptors; reduce large tables on device."""
    if spec.ep_size * spec.num_experts > _HOST_ROUTE_MAX_COUNTS:
        return _dynamic_summary(payload, spec, source_rows), None
    raw = payload.tolist()
    summary = _summarize_host_route(raw, spec)
    if source_rows is not None:
        source_rows.extend(sum(row[3:]) for row in raw)
    return summary, raw


def _summarize_host_route(raw: list[list[int]], spec: MegaMoeSpec) -> tuple[int, int, int]:
    """Validate a bounded Host table before allocating or publishing source rows."""
    counts = [row[3:] for row in raw]
    local = spec.local_experts
    loads = [
        sum(sum(values[start : start + local]) for values in counts) for start in range(0, spec.num_experts, local)
    ]
    rows = [row[:3] + [sum(values), min(values), max(values), load] for row, values, load in zip(raw, counts, loads)]
    return _summarize_dynamic_rows(rows, spec)


def _route_cache_signature(spec: MegaMoeSpec, device: Any) -> tuple[Any, ...]:
    """Keep descriptor reuse specific to the route topology and size contract."""
    return (spec.ep_size, id(spec.ep_group), spec.rank_id, spec.num_experts, spec.top_k, spec.hidden_size,
            spec.dispatch_mode, spec.auto_tokens, spec.max_local_num_tokens, torch.device(device))


def _route_descriptor_key(raw: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Gradient flags participate in validation, not immutable offset geometry."""
    return tuple(tuple(row[:2] + row[3:]) for row in raw)


def _read_cached_route(
    payload: torch.Tensor, spec: MegaMoeSpec, workspace: MegaMoeWorkspace | None, device: Any,
    source_rows: list | None = None, *, stream: Any = None,
) -> tuple[tuple[int, int, int], list[list[int]] | None, RouteMetadata | None]:
    """Reuse exact validated geometry without repeating stable Host preparation."""
    if workspace is None or not spec.auto_tokens or spec.ep_size * spec.num_experts > _HOST_ROUTE_MAX_COUNTS:
        summary, raw = _read_dynamic_route(payload, spec, source_rows)
        return summary, raw, None
    cache = getattr(workspace, "route_cache", None)
    signature = _route_cache_signature(spec, device)
    previous = next(reversed(cache.values())) if isinstance(cache, OrderedDict) and cache else None
    same_topology = previous is not None and previous.signature == signature
    remaining = getattr(workspace, "route_cache_probe_after", 0)
    if same_topology and remaining > 0:
        workspace.route_cache_probe_after = remaining - 1
        return _dynamic_summary(payload, spec, source_rows), None, None
    workspace.route_cache_probe_after = 0
    raw = payload.tolist()
    if same_topology and previous.raw == raw:
        if source_rows is not None:
            source_rows.extend(previous.source_totals)
        previous.wait(device, stream)
        return previous.summary, raw, previous.metadata
    key = (signature, _route_descriptor_key(raw))
    cached = cache.get(key) if isinstance(cache, OrderedDict) else None
    if cached is not None:
        _validate_dynamic_sizes(raw, spec)
        if any(row[2] != raw[0][2] for row in raw):
            raise RuntimeError("MegaMoe autograd participation differs across EP ranks")
        if source_rows is not None:
            source_rows.extend(cached.source_totals)
        cache.move_to_end(key)
        cached.wait(device, stream)
        return cached.summary, raw, cached.metadata
    summary = _summarize_host_route(raw, spec)
    if source_rows is not None:
        source_rows.extend(sum(row[3:]) for row in raw)
    if same_topology and spec.ep_size * spec.num_experts > _HOST_DESCRIPTOR_MAX_COUNTS:
        workspace.route_cache_probe_after = _ROUTE_CACHE_REPROBE_STEPS
    return summary, raw, None


def _cache_host_route(
    workspace: MegaMoeWorkspace | None, spec: MegaMoeSpec, raw: list[list[int]],
    summary: tuple[int, int, int], metadata: RouteMetadata | None, device: Any,
) -> None:
    """Retain a bounded set of owned descriptors; live graphs own their saved views."""
    if workspace is None or not spec.auto_tokens:
        return
    signature = _route_cache_signature(spec, device)
    cache = getattr(workspace, "route_cache", None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        workspace.route_cache = cache
    stream, ready = None, None
    if metadata is not None and torch.device(device).type == "npu":
        stream = torch.npu.current_stream(device)
        ready = torch.npu.Event()
        ready.record(stream)
    key = (signature, _route_descriptor_key(raw))
    cache[key] = _HostRouteCache(signature, raw, summary, metadata,
                                 tuple(sum(row[3:]) for row in raw), stream, ready)
    cache.move_to_end(key)
    while len(cache) > _HOST_ROUTE_CACHE_ENTRIES:
        cache.popitem(last=False)


@torch.inference_mode(False)
@torch.no_grad()
def _dynamic_route_metadata(
    counts: torch.Tensor, gathered: torch.Tensor, host_rows: list[list[int]] | None,
    spec: MegaMoeSpec, capacity: int,
) -> RouteMetadata:
    """Keep wider cache misses on device instead of rebuilding large Host descriptors."""
    if counts.device.type != "npu" and host_rows is not None and spec.ep_size * spec.num_experts <= _HOST_DESCRIPTOR_MAX_COUNTS:
        return _host_route_metadata(host_rows, spec, capacity, counts.device)
    counts_by_source = gathered[:, 3:]
    # The native offsets kernel reads only the source matrix. Materialize the
    # received slice only in the CANN fallback that actually consumes it.
    received_counts = None
    # Retained dispatch sizes must not alias a caller-owned Router count buffer.
    local_counts = counts_by_source[spec.rank_id] if spec.auto_tokens else counts
    return _compute_route_metadata(local_counts, counts_by_source, received_counts, spec, capacity)


def _permute_dynamic_pull(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor, spec: MegaMoeSpec, workspace: MegaMoeWorkspace | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the source view only after automatic growth and lease acquisition."""
    if workspace is None or not workspace.in_use or workspace.source_buffer is None:
        raise RuntimeError("Pull permutation requires an initialized, claimed workspace.")
    tokens = hidden_states.shape[0]
    output = workspace.source_buffer[:tokens * spec.top_k]
    return (_permute_topk_input_out(hidden_states, topk_ids, output) if tokens
            else (output, topk_ids.reshape(-1)))


def _can_stage_pull(hidden_states: torch.Tensor, spec: MegaMoeSpec) -> bool:
    """Bound ordinary-memory permutation staging while capacity agreement is pending."""
    return (spec.auto_tokens and spec.dispatch_mode == "pull" and hidden_states.device.type == "npu"
            and hidden_states.numel() * spec.top_k * hidden_states.element_size() <= _PULL_STAGING_MAX_BYTES)


def _stage_dynamic_permutation(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor, spec: MegaMoeSpec, invalid: bool,
    workspace: MegaMoeWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Stage only valid automatic routes whose ordinary storage survives heap growth."""
    if invalid or not spec.auto_tokens:
        return None
    if (spec.dispatch_mode == "push"
            or _can_stage_pull(hidden_states, spec) and not getattr(workspace, "route_cache_probe_after", 0)):
        return _permute_topk_input(hidden_states, topk_ids)
    return None


def _publish_staged_pull(
    permuted: tuple[torch.Tensor, torch.Tensor], hidden_states: torch.Tensor,
    spec: MegaMoeSpec, workspace: MegaMoeWorkspace | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish staged rows only after every rank validated and committed its workspace."""
    if workspace is None or not workspace.in_use or workspace.source_buffer is None:
        raise RuntimeError("Pull publication requires an initialized, claimed workspace.")
    output = workspace.source_buffer[:hidden_states.shape[0] * spec.top_k]
    output.copy_(permuted[0])
    return output, permuted[1]


def _dynamic_header(
    values: tuple[int, int, int], dtype: torch.dtype, device: Any, workspace: MegaMoeWorkspace | None,
    *, stream: Any = None,
) -> torch.Tensor:
    """Retain exact control values across checkpoint modes without changing the wire."""
    key = (values, dtype, torch.device(device))
    cache = getattr(workspace, "route_header_cache", None)
    cached = cache.get(key) if isinstance(cache, OrderedDict) else None
    if cached is not None:
        cache.move_to_end(key)
        cached.wait(device, stream)
        return cached.tensor
    tensor = _upload_route_tensor(torch.tensor(values, dtype=dtype), device)
    if workspace is not None:
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            workspace.route_header_cache = cache
        ready = None
        if torch.device(device).type == "npu":
            if stream is None:
                stream = torch.npu.current_stream(device)
            ready = torch.npu.Event()
            ready.record(stream)
        cache[key] = _RouteHeaderCache(key, tensor, stream, ready)
        while len(cache) > _HOST_ROUTE_CACHE_ENTRIES:
            cache.popitem(last=False)
    return tensor


def _dynamic_route_payload(
    hidden_states: torch.Tensor, flat_ids: torch.Tensor, spec: MegaMoeSpec,
    supplied_counts: torch.Tensor | None, autograd_mask: int, workspace: MegaMoeWorkspace | None = None,
    *, stream: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Pack one group-consistent wire format, including coordinated overflow flags."""
    tokens = hidden_states.shape[0]
    flexible = spec.auto_tokens
    invalid = (tokens * max(spec.hidden_size, spec.top_k) > 2**31 - 1 if flexible
               else tokens > spec.max_local_num_tokens)
    counts = (torch.zeros(spec.num_experts, dtype=torch.int32, device=hidden_states.device) if invalid
              else _resolve_counts(flat_ids, supplied_counts, spec))
    payload_dtype = torch.int32 if flexible else torch.int64
    # An invalid source must still join the same collective before all ranks reject it.
    header_tokens = min(tokens, 2**31 - 1) if flexible else tokens
    header = _dynamic_header((int(invalid), header_tokens, autograd_mask), payload_dtype,
                             hidden_states.device, workspace if flexible else None, stream=stream)
    payload = torch.cat((header, counts.to(payload_dtype)))
    return counts, payload, invalid


def _start_dynamic_gather(payload: torch.Tensor, spec: MegaMoeSpec) -> tuple[torch.Tensor, Any | None]:
    """Match receive storage to the dynamic count wire dtype."""
    if spec.ep_size == 1:
        gathered, work = payload.unsqueeze(0), None
    else:
        gathered = torch.empty((spec.ep_size, payload.numel()), dtype=payload.dtype, device=payload.device)
        work = dist.all_gather_into_tensor(gathered.reshape(-1), payload, group=spec.ep_group, async_op=True)
    return gathered, work


def _prepare_dynamic_route(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor, flat_ids: torch.Tensor, spec: MegaMoeSpec,
    supplied_counts: torch.Tensor | None, workspace: MegaMoeWorkspace | None, autograd_mask: int,
    prepare_workspace: Callable[[int, int], None] | None = None,
    select_route: RouteSelector | None = None,
    discard_rejected: bool = False,
    validated_summary: tuple[int, int, int] | None = None,
    with_source_rows: bool = False,
    with_source_permutation: bool = False,
) -> PreparedTopKRoute | None:
    """Exchange variable-length metadata without adding a token-count collective."""
    source_rows = [] if with_source_rows else None
    tokens = hidden_states.shape[0]
    # Invalid dynamic ranks must still enter the original collective before failing.
    tokens = hidden_states.shape[0]
    within_limit = (tokens * max(spec.hidden_size, spec.top_k) <= 2**31 - 1 if spec.auto_tokens
                    else tokens <= spec.max_local_num_tokens)
    route_stream = (torch.npu.current_stream(hidden_states.device)
                    if workspace is not None and hidden_states.device.type == "npu" else None)
    permuted = None
    if within_limit and _can_fuse_count(hidden_states, spec, supplied_counts):
        permuted, supplied_counts = _permute_and_count(hidden_states, topk_ids, spec)
    counts, payload, invalid = _dynamic_route_payload(
        hidden_states, flat_ids, spec, supplied_counts, autograd_mask, workspace, stream=route_stream)
    if workspace is not None and spec.dispatch_mode != "pull":
        workspace.wait_for_reuse(stream=route_stream)
    gathered, work = _start_dynamic_gather(payload, spec)
    # Ordinary storage remains valid if count agreement triggers a SHMEM restart.
    if permuted is None:
        permuted = _stage_dynamic_permutation(hidden_states, topk_ids, spec, invalid, workspace)
    # A lossless placement can reuse the original route's source validation and
    # its proven receive bound. Counts are still exchanged below and exact
    # offsets are built from this invocation's gathered virtual counts.
    summary, host_rows, metadata = validated_summary, None, None
    selected = True
    if spec.auto_tokens or select_route is not None:
        if work is not None:
            work.wait()
            work = None
        if validated_summary is None:
            summary, host_rows, metadata = _read_cached_route(gathered, spec, workspace, hidden_states.device, source_rows, stream=route_stream)
        elif select_route is not None:
            raise ValueError("A validated placement cannot select another route")
        if select_route is not None:
            total = sum(source_rows) if source_rows is not None else (
                sum(row[1] for row in host_rows) * spec.top_k if host_rows is not None else None)
            args = (gathered[:, 3:], summary[1], summary[2], total)
            selected = (select_route(*args, tuple(source_rows),
                                     None if host_rows is None else tuple(tuple(row[3:]) for row in host_rows),
                                     **({"source_inverse": permuted[1] if permuted is not None else None}
                                        if with_source_permutation else {}))
                        if with_source_rows else select_route(*args))
            if not selected and discard_rejected:
                if host_rows is not None and spec.ep_size * spec.num_experts <= _HOST_DESCRIPTOR_MAX_COUNTS:
                    # The moving executor still reuses this validated source
                    # summary. No device descriptor or event belongs to it.
                    _cache_host_route(workspace, spec, host_rows, summary, None, hidden_states.device)
                return None
        if selected:
            if prepare_workspace is None:
                raise RuntimeError("Automatic MegaMoe routes require workspace capacity preparation")
            prepare_workspace(summary[1], summary[2])
    if invalid:
        permuted = None
    elif spec.dispatch_mode == "pull" and selected:
        permuted = (_permute_dynamic_pull(hidden_states, topk_ids, spec, workspace) if permuted is None else
                    _publish_staged_pull(permuted, hidden_states, spec, workspace))
    elif permuted is None:
        permuted = _permute_topk_input(hidden_states, topk_ids)
    if work is not None:
        work.wait()
    if summary is None:
        summary, host_rows = _read_dynamic_route(gathered, spec)
    capacity, plan_tokens, maximum = summary
    if metadata is None:
        metadata = _dynamic_route_metadata(counts, gathered, host_rows, spec, capacity)
        if host_rows is not None:
            start = spec.rank_id * spec.local_experts
            active = all(sum(row[3 + expert] for row in host_rows) > 0
                         for expert in range(start, start + spec.local_experts))
            native = metadata._elements
            metadata = replace(metadata, all_experts_active=active,
                               _elements=(native[0], replace(native[1], all_experts_active=active)) if native else None)
        metadata = replace(metadata, local_num_tokens=tokens, plan_tokens=plan_tokens)
        if host_rows is not None:
            _cache_host_route(workspace, spec, host_rows, summary, metadata, hidden_states.device)
    received_counts = metadata.combine_size.reshape(spec.ep_size, spec.local_experts)
    return PreparedTopKRoute(
        routed_tokens=permuted[0], unpermute_mapping=permuted[1], tokens_per_expert=counts,
        received_counts=received_counts,
        maximum_received_slots=maximum,
        metadata=metadata,
    )


class _EmptyTopKOutput(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Preserve distributed expert backward and the empty probability gradient."""

    @staticmethod
    def forward(ctx: Any, expert_output: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        ctx.probability_shape = probabilities.shape
        ctx.probability_dtype = probabilities.dtype
        ctx.probability_device = probabilities.device
        return expert_output.new_empty(expert_output.shape)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        grad_experts = grad_output if ctx.needs_input_grad[0] else None
        grad_probabilities = (grad_output.new_empty(ctx.probability_shape, dtype=ctx.probability_dtype,
                                                  device=ctx.probability_device)
                              if ctx.needs_input_grad[1] else None)
        return grad_experts, grad_probabilities


def restore_topk_output(
    expert_output: torch.Tensor,
    unpermute_mapping: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Restore token order and apply Router weights with the native NPU op.

    Args:
        expert_output: Expert outputs in the expanded routing order.
        unpermute_mapping: Inverse mapping retained by the source permutation.
        topk_weights: Router weights for the original token rows.

    Returns:
        Weighted expert outputs in the original token order.
    """
    if topk_weights.shape[0] == 0:
        # Keep both the expert backward and the empty routing-weight gradient connected.
        return _EmptyTopKOutput.apply(expert_output, topk_weights)
    return load_torch_npu().npu_moe_token_unpermute(
        expert_output,
        unpermute_mapping,
        probs=topk_weights.float().contiguous(),
    )


def _prepare_planned_route(hidden_states, topk_ids, spec, supplied_counts, workspace,
                           prepare_workspace, summary, counts_by_source):
    """Reuse a lossless planner histogram without a second count collective.

    Source sizes, gradient participation and the receive bound were validated
    by the original route. Counts are owned by this invocation, not scratch.
    """
    if (counts_by_source.shape != (spec.ep_size, spec.num_experts)
            or counts_by_source.dtype != torch.int32
            or counts_by_source.device != hidden_states.device
            or not counts_by_source.is_contiguous()):
        raise ValueError("Planner counts must be contiguous source-major int32 on the input device")
    capacity, plan_tokens, maximum = summary
    permuted = None
    if _can_fuse_count(hidden_states, spec, supplied_counts):
        permuted, _ = _permute_and_count(hidden_states, topk_ids, spec)
    if workspace is not None and spec.dispatch_mode != "pull":
        workspace.wait_for_reuse()
    if spec.auto_tokens:
        if prepare_workspace is None:
            raise RuntimeError("Automatic MegaMoe routes require workspace capacity preparation")
        prepare_workspace(plan_tokens, maximum)
    if spec.dispatch_mode == "pull":
        permuted = (_permute_dynamic_pull(hidden_states, topk_ids, spec, workspace) if permuted is None else
                    _publish_staged_pull(permuted, hidden_states, spec, workspace))
    elif permuted is None:
        permuted = _permute_topk_input(hidden_states, topk_ids)
    # Planner outputs are owned snapshots, not mutable SHMEM/scratch views.
    # Keep the source alive, key its version, and never cache the permutation.
    cacheable = workspace is not None and not counts_by_source.is_inference()
    cache = getattr(workspace, "planned_route_cache", None) if cacheable else None
    key = (_route_cache_signature(spec, hidden_states.device), id(counts_by_source),
           counts_by_source._version, summary, hidden_states.shape[0]) if cacheable else None
    cached = cache.get(key) if isinstance(cache, OrderedDict) else None
    stream = torch.npu.current_stream(hidden_states.device) if hidden_states.device.type == "npu" else None
    if cached is not None and cached.source is counts_by_source:
        cached.wait(stream)
        cache.move_to_end(key)
        counts, received, metadata = cached.local_counts, cached.received_counts, cached.metadata
    else:
        counts = counts_by_source[spec.rank_id]
        start = spec.rank_id * spec.local_experts
        received = counts_by_source[:, start:start + spec.local_experts]
        # Planned counts do not contain the dynamic wire's three header columns.
        with torch.inference_mode(False), torch.no_grad():
            metadata = _compute_route_metadata(counts, counts_by_source, received, spec, capacity)
        metadata = replace(metadata, local_num_tokens=hidden_states.shape[0], plan_tokens=plan_tokens)
        if cacheable:
            if not isinstance(cache, OrderedDict):
                cache = OrderedDict()
                workspace.planned_route_cache = cache
            ready = None
            if stream is not None:
                ready = torch.npu.Event()
                ready.record(stream)
            cache[key] = _PlannedRouteCache(counts_by_source, counts, received, metadata, stream, ready)
            while len(cache) > _HOST_ROUTE_CACHE_ENTRIES:
                cache.popitem(last=False)
    return PreparedTopKRoute(
        routed_tokens=permuted[0], unpermute_mapping=permuted[1], tokens_per_expert=counts,
        received_counts=received, maximum_received_slots=maximum,
        metadata=metadata,
    )
