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
"""Checkpoint-owned routing and replica state for BF16 Push."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import os
from threading import Lock
from typing import Any, ClassVar

import torch
import torch.distributed as dist

from ..mega_moe import a3_indexed


@dataclass
class ReplayRecord:
    """Save one forward's immutable route and optionally its owned activations."""

    owner: Any
    ids: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    route: Any = None
    stream: Any = None
    inputs: tuple[torch.Tensor, torch.Tensor] | None = None
    input_versions: tuple[int, int] | None = None
    activation: Any = None
    allow_activation: bool = False
    weight_binding: Any = None
    guest_snapshot_bytes: int = 0
    guest_weight_snapshot: Any = None
    allow_guest_restore: bool = False
    replay_count_fence: a3_indexed.CountFence | None = None

    @contextmanager
    def save_supplied_route(self, ids: torch.Tensor, counts: torch.Tensor) -> Iterator[None]:
        """Read the Router inputs before staging permutations on the same stream."""
        self.ids, self.counts, ready = _stage_router_inputs(ids, counts, self.stream)
        try:
            yield
        finally:
            if ready is not None:
                ready.synchronize()


@dataclass
class _CheckpointScope:
    records: list[ReplayRecord]
    replay: bool
    same_parameters: bool
    cursor: int = 0
    defer_forward_copies: bool = False
    guest_snapshot_bytes: int = 0


_ACTIVE: ContextVar[_CheckpointScope | None] = ContextVar("hmc_hot_replica_checkpoint", default=None)


class ReplayCheckpoint:
    """Hold one checkpoint invocation's tape across original and replay contexts."""

    def __init__(
        self, parameter_version: Callable[[], Any], *, prepare_replay: bool = False,
        defer_forward_copies: bool = False, guest_snapshot_bytes: int = 0,
    ) -> None:
        """Register the caller's version lookup and optional checkpoint policies.

        Copy deferral requires a non-reentrant checkpoint whose saved-tensor
        hooks replace original placeholders with materialized replay values.
        A positive guest_snapshot_bytes bounds each invocation's private guest
        weight copy per rank and requires a grad-enabled non-reentrant checkpoint.
        Use a separate tape per invocation and enter both corresponding scopes.
        """
        if type(guest_snapshot_bytes) is not int or guest_snapshot_bytes < 0:
            raise ValueError("guest_snapshot_bytes must be a nonnegative integer")
        self.parameter_version = parameter_version
        self.prepare_replay = prepare_replay
        self.defer_forward_copies = defer_forward_copies
        self.guest_snapshot_bytes = guest_snapshot_bytes
        self.records: list[ReplayRecord] = []
        self.forward_version = None

    @contextmanager
    def scope(self, replay: bool) -> Iterator[None]:
        """Enter the tape before running any operator of the checkpointed block."""
        if self.defer_forward_copies and not torch.is_grad_enabled():
            raise RuntimeError("Deferring forward copies requires a grad-enabled non-reentrant checkpoint")
        if self.guest_snapshot_bytes and not torch.is_grad_enabled():
            raise RuntimeError("Guest weight snapshots require a grad-enabled non-reentrant checkpoint")
        version = self.parameter_version()
        if not replay:
            self.records.clear()
            self.forward_version = version
        state = _CheckpointScope(
            self.records, replay, version is not None and version == self.forward_version,
            defer_forward_copies=self.defer_forward_copies,
            guest_snapshot_bytes=self.guest_snapshot_bytes,
        )
        token = _ACTIVE.set(state)
        try:
            if replay and state.same_parameters and self.prepare_replay and len(self.records) == 1:
                binding = self.records[0].weight_binding
                if binding is not None:
                    binding.prepare_replay()
            yield
        finally:
            _ACTIVE.reset(token)


def forward_copy_phase() -> bool | None:
    """Return the opted-in replay flag independently of route reuse."""
    scope = _ACTIVE.get()
    return scope.replay if scope is not None and scope.defer_forward_copies else None


def guest_weight_replay(record: ReplayRecord | None) -> tuple[ReplayRecord, bool] | None:
    """Use the original call's budget or its collectively approved snapshot."""
    scope = _ACTIVE.get()
    if record is None or not record.guest_snapshot_bytes or scope is None:
        return None
    if scope.replay and not record.allow_guest_restore:
        return None
    return record, scope.replay


def wrap_checkpoint(
    function: Callable[..., Any], parameter_version: Callable[[], Any], *, prepare_replay: bool = False,
) -> Callable[..., Any]:
    """Wrap one checkpoint call with state scoped to its original and replay forwards.

    The framework supplies a lightweight logical parameter version for its
    sharded module. A changed version disables weight and activation reuse.
    When requested, restore the saved borrowed suffix before recompute reads
    any FSDP parameters. Route agreement still happens at the consumer.
    """
    tape = ReplayCheckpoint(parameter_version, prepare_replay=prepare_replay)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Enter the checkpoint scope for this invocation."""
        with tape.scope(torch.is_grad_enabled()):
            return function(*args, **kwargs)

    return wrapped


def is_checkpointing() -> bool:
    """Report whether route selection is in a checkpoint call."""
    return _ACTIVE.get() is not None


def route_record(owner: Any) -> tuple[ReplayRecord | None, bool]:
    """Get the record corresponding to the current invocation in this checkpoint."""
    scope = _ACTIVE.get()
    if scope is None:
        return None, False
    if not scope.replay:
        record = ReplayRecord(owner, guest_snapshot_bytes=scope.guest_snapshot_bytes if scope.same_parameters else 0)
        scope.records.append(record)
        return record, False
    index = scope.cursor
    scope.cursor += 1
    if index < len(scope.records) and scope.records[index].owner is owner:
        return scope.records[index], True
    return None, True


def _stage_router_inputs(ids: torch.Tensor, counts: torch.Tensor, stream: Any):
    """Return both host tensors and the completion point for their copies."""
    pinned = ids.device.type == "npu"
    host_ids = torch.empty(ids.shape, dtype=ids.dtype, device="cpu", pin_memory=pinned)
    host_counts = torch.empty(counts.shape, dtype=counts.dtype, device="cpu", pin_memory=pinned)
    host_ids.copy_(ids, non_blocking=pinned)
    host_counts.copy_(counts, non_blocking=pinned)
    ready = torch.npu.Event() if pinned else None
    if ready is not None:
        ready.record(stream)
    return host_ids, host_counts, ready


def agree_replay(
    record: ReplayRecord | None, hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
    counts: torch.Tensor | None, spec: Any, mask: int, *, weight_error: bool,
) -> bool:
    """Agree across EP on exact routing and independent activation validity."""
    scope = _ACTIVE.get()
    if record is not None:
        record.replay_count_fence = None
    valid = (
        record is not None and record.ids is not None and record.route is not None
        and not weight_error and record.ids.shape == ids.shape and record.ids.dtype == ids.dtype
        and record.stream == torch.npu.current_stream(hidden.device)
        and scope is not None and scope.same_parameters
    )
    with torch.no_grad():
        if valid and counts is not None and record.counts is not None and (
                counts.device == ids.device and counts.shape == record.counts.shape
                and counts.dtype == record.counts.dtype):
            host_ids, host_counts, ready = _stage_router_inputs(ids, counts, record.stream)
            if ready is not None:
                ready.synchronize()
            same_route = torch.equal(host_ids, record.ids) and torch.equal(host_counts, record.counts)
            flag = torch.full((1,), mask if same_route else 0, dtype=torch.int32, device=hidden.device)
        elif valid and counts is None and record.counts is None:
            same_ids = (ids == record.ids).all()
            flag = same_ids.to(torch.int32).reshape(1) * mask
            if record.activation is not None and record.inputs is not None:
                value, score = record.inputs
                if (value.shape == hidden.shape and score.shape == probs.shape
                        and record.input_versions == (value._version, score._version)):
                    same_values = same_ids & (value == hidden).all() & (score == probs).all()
                    flag += same_values.to(torch.int32).reshape(1) * 32
        else:
            flag = torch.zeros(1, dtype=torch.int32, device=hidden.device)
        if valid and record.guest_weight_snapshot is not None:
            flag += 64
        replay_fence = None
        if (valid and counts is None
                and record.route.indexed_dispatch_offsets is not None):
            workspace = record.owner.workspace
            replay_fence = a3_indexed.CountFence.before_gather(workspace, record.stream)
            if replay_fence is not None:
                flag += a3_indexed.REPLAY_FENCE_BIT
        gathered = flag.new_empty(spec.ep_size)
        dist.all_gather_into_tensor(gathered, flag.contiguous(), group=spec.ep_group)
        flags = gathered.tolist()
    first = flags[0] % 32
    reuse = first != 0 and all(value % 32 == first for value in flags)
    if record is not None:
        record.allow_activation = reuse and all(value & 32 for value in flags)
        record.allow_guest_restore = reuse and all(value & 64 for value in flags)
        if reuse and all(value & a3_indexed.REPLAY_FENCE_BIT for value in flags):
            record.replay_count_fence = replay_fence
    return reuse


class ActivationReservation:
    """A per-device, process-wide limit on additional checkpoint activations."""

    __slots__ = ("device", "required")
    used: ClassVar[dict] = {}
    lock = Lock()
    counters: ClassVar[dict] = {}
    limits: ClassVar[dict] = {}

    @classmethod
    def reserve(
        cls, flat: torch.Tensor, required: int, group: dist.ProcessGroup,
    ) -> ActivationReservation | None:
        """Charge a local snapshot while retaining headroom for FSDP."""
        device = flat.device
        key = (device, id(group))
        with cls.lock:
            if key not in cls.limits:
                total = torch.npu.get_device_properties(device).total_memory
                free = torch.npu.mem_get_info(device)[0]
                configured = os.getenv("HYPER_MULTICORE_HOT_REPLICA_CHECKPOINT_BYTES")
                # Optional activations share HBM with FSDP and the symmetric
                # heap. Leave headroom for transient allocations. Admission can
                # differ by rank, and the existing replay consensus handles
                # a local miss without introducing a conditional collective.
                default_limit = total // 6
                limit = default_limit if configured is None else max(0, int(configured))
                local_limit = min(limit, max(0, free - total // 3))
                cls.limits[key] = local_limit, free
            limit, free = cls.limits[key]
            used = cls.used.get(device, 0)
            counts = cls.counters.get(device)
            if counts is None:
                counts = {
                    "initial_free": free, "attempts": 0, "budget_rejections": 0,
                    "memory_rejections": 0, "reservations": 0, "releases": 0, "peak_reserved": 0,
                    "pressure_events": 0, "allowed_bytes": limit, "allocator_retries": 0,
                }
                cls.counters[device] = counts
            if used == 0 and counts["attempts"] and limit:
                retries = torch.npu.memory_stats(device).get("num_alloc_retries", 0)
                if retries > counts["allocator_retries"]:
                    # A retry during previous work is evidence that the cached
                    # bytes competed with a required allocation. Back off at
                    # the next empty reservation boundary without freeing any
                    # live tensor or changing the collective execution order.
                    limit //= 2
                    cls.limits[key] = limit, free
                    counts["pressure_events"] += 1
                    counts["allowed_bytes"] = limit
                counts["allocator_retries"] = max(retries, counts["allocator_retries"])
            counts["attempts"] += 1
            if required > limit - used:
                counts["budget_rejections"] += 1
                return None
            obj = cls()
            cls.used[device] = used + required
            counts["reservations"] += 1
            counts["peak_reserved"] = max(counts["peak_reserved"], cls.used[device])
            obj.device = device
            obj.required = required
            return obj

    def __del__(self):
        # This charge owns no tensors and has no reference back to a graph.
        cls = type(self)
        with cls.lock:
            cls.used[self.device] -= self.required
            cls.counters[self.device]["releases"] += 1
