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
from types import TracebackType
from typing import Any, ClassVar

import torch
import torch.distributed as dist

from hyper_multicore.modules.mega_moe.route import (
    _INVALID_COUNTS_MASK, _INVALID_ROUTE_INPUT_MASK, _reject_invalid_route_inputs, _validate_supplied_counts,
)


@dataclass
class ReplayRecord:
    """Save one forward's immutable route and optionally its owned activations."""

    owner: Any
    ids: torch.Tensor | None = None
    supplied_counts: torch.Tensor | None = None
    route: Any = None
    stream: Any = None
    autograd_mask: int = 0
    inputs: tuple[torch.Tensor, torch.Tensor] | None = None
    input_versions: tuple[int, int] | None = None
    activation: Any = None
    allow_activation: bool = False


@dataclass
class _CheckpointScope:
    records: list[ReplayRecord]
    replay: bool
    same_parameters: bool
    non_reentrant: bool = False
    cursor: int = 0


_ACTIVE: ContextVar[_CheckpointScope | None] = ContextVar("hmc_hot_replica_checkpoint", default=None)
_REPLAY_ACTIVATION_MASK = 1 << 7
_REPLAY_SCOPES: ContextVar[tuple[AbstractContextManager, ...]] = ContextVar("hmc_replay_scopes", default=())


class _ReplayContext(AbstractContextManager):
    """Start a new replay cursor each time Torch re-enters its recompute context."""

    def __init__(self, tape: ReplayCheckpoint) -> None:
        """Keep the original invocation's tape alive for each recomputation."""
        self.tape = tape

    def __enter__(self) -> None:
        """Enter a fresh replay scope when Torch starts recomputation."""
        scope = self.tape.scope(True, non_reentrant=True)
        scope.__enter__()
        _REPLAY_SCOPES.set((*_REPLAY_SCOPES.get(), scope))

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None,
    ) -> bool | None:
        """Restore the previous scope on completion or error."""
        scopes = _REPLAY_SCOPES.get()
        _REPLAY_SCOPES.set(scopes[:-1])
        return scopes[-1].__exit__(exc_type, exc, traceback)


class ReplayCheckpoint:
    """Hold one checkpoint invocation's tape across original and replay contexts."""

    def __init__(self, parameter_version: Callable[[], Any]) -> None:
        """Register the caller's logical parameter version lookup."""
        self.parameter_version = parameter_version
        self.records: list[ReplayRecord] = []
        self.forward_version = None
        self.started = False

    def contexts(self) -> tuple[AbstractContextManager, AbstractContextManager]:
        """Provide the context_fn for one non-reentrant torch checkpoint call."""
        return self.scope(False, non_reentrant=True), _ReplayContext(self)

    @contextmanager
    def scope(self, replay: bool, *, non_reentrant: bool = False) -> Iterator[None]:
        """Enter the tape before running any operator of the checkpointed block."""
        version = self.parameter_version()
        if not replay:
            if self.started:
                raise RuntimeError("Create a separate wrap_checkpoint wrapper or ReplayCheckpoint for each forward")
            self.started = True
            self.forward_version = version
        state = _CheckpointScope(
            self.records, replay, version is not None and version == self.forward_version,
            non_reentrant=non_reentrant,
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


def route_record(owner: Any) -> tuple[ReplayRecord | None, bool]:
    """Get the record corresponding to the current invocation in this checkpoint."""
    scope = _ACTIVE.get()
    if scope is None:
        return None, False
    if not scope.replay:
        record = ReplayRecord(owner)
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
    counts: torch.Tensor | None, spec: Any, mask: int,
) -> bool:
    """Agree across EP on exact routing and activation validity."""
    scope = _ACTIVE.get()
    assert scope is not None
    if counts is not None:
        try:
            _validate_supplied_counts(counts, spec.num_experts, hidden.device)
        except (TypeError, ValueError):
            mask |= _INVALID_COUNTS_MASK
    invalid = mask & _INVALID_ROUTE_INPUT_MASK
    with torch.no_grad():
        if not invalid and _can_replay(record, ids, counts, mask, scope):
            flag = _replay_vote(record, hidden, ids, probs, counts, mask)
        else:
            flag = torch.full((1,), invalid, dtype=torch.int32, device=hidden.device)
        gathered = flag.new_empty(spec.ep_size)
        dist.all_gather_into_tensor(gathered, flag.contiguous(), group=spec.ep_group)
        flags = gathered.tolist()
    route_mask = ~_REPLAY_ACTIVATION_MASK
    first = flags[0] & route_mask
    reuse = first != 0 and not first & _INVALID_ROUTE_INPUT_MASK and all(
        (value & route_mask) == first for value in flags)
    if not reuse:
        _reject_invalid_route_inputs(flags)
        if scope.non_reentrant:
            raise RuntimeError("Non-reentrant MegaMoe checkpoint could not replay its original expert layout "
                               "on all EP ranks")
    if record is not None:
        record.allow_activation = reuse and all(value & _REPLAY_ACTIVATION_MASK for value in flags)
    return reuse


def _can_replay(
    record: ReplayRecord | None, ids: torch.Tensor, counts: torch.Tensor | None,
    mask: int, scope: _CheckpointScope,
) -> bool:
    """Check the saved metadata and execution mode before comparing tensors."""
    if not scope.same_parameters or record is None:
        return False
    return (
        record.route is not None and record.ids is not None
        and record.ids.shape == ids.shape and record.ids.dtype == ids.dtype and record.ids.device == ids.device
        and (counts is None) == (record.supplied_counts is None)
        and record.stream == torch.npu.current_stream(ids.device)
        and (not scope.non_reentrant or record.autograd_mask == mask)
    )


def _replay_vote(record: ReplayRecord, hidden, ids, probs, counts, mask):
    """Vote on exact route identity and whether saved activations still match."""
    same_route = (ids == record.ids).all()
    if counts is not None:
        same_route = same_route & (counts == record.supplied_counts).all()
    flag = same_route.to(torch.int32).reshape(1) * mask
    if record.activation is not None and record.inputs is not None:
        value, score = record.inputs
        if value.shape != hidden.shape or score.shape != probs.shape:
            return flag
        # The tensor version counters detect writes through aliases to the saved inputs.
        versions = value._version, score._version  # pylint: disable=protected-access
        if record.input_versions == versions:
            same_values = same_route & (value == hidden).all() & (score == probs).all()
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
