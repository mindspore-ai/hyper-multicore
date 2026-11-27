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
"""Checkpoint-owned routing and replica state for reentrant BF16 Push and Pull."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
import os
from threading import Lock
from typing import Any, ClassVar

import torch
import torch.distributed as dist


@dataclass
class ReplayRecord:
    """Save one forward's immutable route and optionally its owned activations."""

    owner: Any
    ids: torch.Tensor | None = None
    route: Any = None
    stream: Any = None
    inputs: tuple[torch.Tensor, torch.Tensor] | None = None
    input_versions: tuple[int, int] | None = None
    activation: Any = None
    allow_activation: bool = False
    guest_snapshot_bytes: int = 0
    guest_snapshot: Any = None
    allow_guest_restore: bool = False


@dataclass
class _CheckpointScope:
    records: list[ReplayRecord]
    replay: bool
    same_parameters: bool
    cursor: int = 0
    defer_forward_copies: bool = False
    guest_snapshot_bytes: int = 0


_ACTIVE: ContextVar[_CheckpointScope | None] = ContextVar("hmc_hot_replica_checkpoint", default=None)
_REPLAY_ACTIVATION_MASK = 1 << 7
_REPLAY_GUEST_MASK = 1 << 8
_REPLAY_SCOPES: ContextVar[tuple[AbstractContextManager, ...]] = ContextVar("hmc_replay_scopes", default=())


class _ReplayContext(AbstractContextManager):
    """Start a new replay cursor each time Torch re-enters its recompute context."""

    def __init__(self, tape: ReplayCheckpoint):
        self.tape = tape

    def __enter__(self):
        scope = self.tape.scope(True)
        scope.__enter__()
        _REPLAY_SCOPES.set((*_REPLAY_SCOPES.get(), scope))

    def __exit__(self, *exc):
        scopes = _REPLAY_SCOPES.get()
        _REPLAY_SCOPES.set(scopes[:-1])
        return scopes[-1].__exit__(*exc)


class ReplayCheckpoint:
    """Hold one checkpoint invocation's tape across original and replay contexts."""

    def __init__(
        self, parameter_version: Callable[[], Any], *,
        defer_forward_copies: bool = False, guest_snapshot_bytes: int = 0,
    ) -> None:
        """Register the parameter version lookup and opt in to checkpoint saves."""
        if not isinstance(defer_forward_copies, bool):
            raise TypeError("defer_forward_copies must be a boolean")
        if (not isinstance(guest_snapshot_bytes, int) or isinstance(guest_snapshot_bytes, bool)
                or guest_snapshot_bytes < 0):
            raise ValueError("guest_snapshot_bytes must be a nonnegative integer")
        self.parameter_version = parameter_version
        self.defer_forward_copies = defer_forward_copies
        self.guest_snapshot_bytes = guest_snapshot_bytes
        self.records: list[ReplayRecord] = []
        self.forward_version = None
        self.started = False

    def contexts(self) -> tuple[AbstractContextManager, AbstractContextManager]:
        """Provide the context_fn for one non-reentrant torch checkpoint call."""
        return self.scope(False), _ReplayContext(self)

    @contextmanager
    def scope(self, replay: bool) -> Iterator[None]:
        """Enter the tape before running any operator of the checkpointed block."""
        if (self.defer_forward_copies or self.guest_snapshot_bytes) and not torch.is_grad_enabled():
            raise RuntimeError("Checkpoint saves require a grad-enabled non-reentrant checkpoint")
        version = self.parameter_version()
        if not replay:
            if self.started:
                raise RuntimeError("Create a separate wrap_checkpoint wrapper or ReplayCheckpoint for each forward")
            self.started = True
            self.forward_version = version
        state = _CheckpointScope(
            self.records, replay, version is not None and version == self.forward_version,
            defer_forward_copies=self.defer_forward_copies, guest_snapshot_bytes=self.guest_snapshot_bytes,
        )
        token = _ACTIVE.set(state)
        try:
            yield
        finally:
            _ACTIVE.reset(token)


def wrap_checkpoint(
    function: Callable[..., Any], parameter_version: Callable[[], Any], *, prepare_replay: bool = False,
) -> Callable[..., Any]:
    """Wrap one checkpoint call with state scoped to its original and replay forwards.

    The framework supplies a lightweight logical parameter version for its
    sharded module. A changed version disables weight and activation reuse.
    The consumer restores saved guest weights after route agreement.
    The legacy ``prepare_replay`` argument is accepted for compatibility.
    """
    del prepare_replay
    tape = ReplayCheckpoint(parameter_version)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Enter the checkpoint scope for this invocation."""
        with tape.scope(torch.is_grad_enabled()):
            return function(*args, **kwargs)

    return wrapped


def is_checkpointing() -> bool:
    """Report whether route selection is in a checkpoint call."""
    return _ACTIVE.get() is not None


def forward_save_policy() -> tuple[bool, bool] | None:
    """Report the phase and copy deferral setting for a checkpoint with optional saves."""
    scope = _ACTIVE.get()
    if scope is None or not (scope.defer_forward_copies or scope.guest_snapshot_bytes):
        return None
    return scope.replay, scope.defer_forward_copies


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
        # Reuse the saved tensors while keeping each replay's agreement local to that invocation.
        return replace(scope.records[index]), True
    return None, True


def agree_replay(
    record: ReplayRecord | None, hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
    counts: torch.Tensor | None, spec: Any, mask: int, *, weight_error: bool,
) -> bool:
    """Agree across EP on exact routing, activation validity and guest snapshots."""
    scope = _ACTIVE.get()
    valid = (
        not weight_error and counts is None and record is not None
        and record.route is not None and record.ids is not None
        and record.ids.shape == ids.shape and record.ids.dtype == ids.dtype
        and record.stream == torch.npu.current_stream(hidden.device)
        and scope is not None and scope.same_parameters
    )
    with torch.no_grad():
        flag = (_replay_vote(record, hidden, ids, probs, mask) if valid else
                torch.zeros(1, dtype=torch.int32, device=hidden.device))
        gathered = flag.new_empty(spec.ep_size)
        dist.all_gather_into_tensor(gathered, flag.contiguous(), group=spec.ep_group)
        flags = gathered.tolist()
    route_mask = ~(_REPLAY_ACTIVATION_MASK | _REPLAY_GUEST_MASK)
    first = flags[0] & route_mask
    reuse = first != 0 and all((value & route_mask) == first for value in flags)
    if record is not None:
        record.allow_activation = reuse and all(value & _REPLAY_ACTIVATION_MASK for value in flags)
        record.allow_guest_restore = reuse and all(value & _REPLAY_GUEST_MASK for value in flags)
    return reuse


def _replay_vote(record: ReplayRecord, hidden, ids, probs, mask):
    """Vote on exact route identity and whether saved activations still match."""
    same_ids = (ids == record.ids).all()
    if record.guest_snapshot is not None:
        mask |= _REPLAY_GUEST_MASK
    flag = same_ids.to(torch.int32).reshape(1) * mask
    if record.activation is not None and record.inputs is not None:
        value, score = record.inputs
        if value.shape != hidden.shape or score.shape != probs.shape:
            return flag
        # The tensor version counters detect writes through aliases to the saved inputs.
        versions = value._version, score._version  # pylint: disable=protected-access
        if record.input_versions == versions:
            same_values = same_ids & (value == hidden).all() & (score == probs).all()
            flag += same_values.to(torch.int32).reshape(1) * _REPLAY_ACTIVATION_MASK
    return flag


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
