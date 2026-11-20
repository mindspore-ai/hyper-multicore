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
"""MegaMoe workspace capacity and SHMEM Heap policy."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, NoReturn

import torch

from hyper_multicore import _collective, shmem
from hyper_multicore.shmem import _lifecycle

from .spec import (
    _DEFAULT_CAPACITY_FACTOR,
    MegaMoeSpec,
    _align_capacity,
    _grow_receive_capacity,
    _resolve_receive_capacity,
)
from .workspace import MegaMoeWorkspace, _spec_workspace_bytes


_HEAP_GRANULARITY_BYTES = 2 * 1024 * 1024
_HEAP_MANAGER = None


def _round_heap_bytes(value: int) -> int:
    """Round the aggregate payload size to physical Heap pages."""
    return (value + _HEAP_GRANULARITY_BYTES - 1) // _HEAP_GRANULARITY_BYTES * _HEAP_GRANULARITY_BYTES


@dataclass(frozen=True)
class _WorkspaceLayout:
    """One reservation's bytes, collective identity and receive capacity."""

    required_bytes: int
    collective_signature: tuple[Any, ...]
    receive_capacity: int
    token_capacity: int | None = None
    borrows_receive_buffer: bool = False


@dataclass
class _HeapEntry:
    """One ordered lazy reservation and its optional bound workspace."""

    specification: Mapping[str, Any]
    layout: _WorkspaceLayout
    workspace: MegaMoeWorkspace | None = None
    bound_spec: MegaMoeSpec | None = None


@dataclass(frozen=True)
class _RestartEntryAgreement:
    """Rank-independent state for one entry in a restart transaction."""

    layout: _WorkspaceLayout
    is_bound: bool
    is_materialized: bool


@dataclass(frozen=True)
class _RestartAgreement:
    """Rank-independent entry state for one restart epoch."""

    epoch: int
    entries: tuple[_RestartEntryAgreement, ...]


@dataclass(frozen=True)
class _RestartBinding:
    """Map one materialized workspace to a contiguous restart result slice."""

    entry: _HeapEntry
    start: int
    stop: int


def _workspace_layout(
    specification: Mapping[str, Any],
    dtype: torch.dtype,
    capacity: int | None = None,
    token_capacity: int | None = None,
    *,
    borrows_receive_buffer: bool = False,
) -> _WorkspaceLayout:
    """Describe one MegaMoe symmetric workspace reservation."""
    physical = dict(specification)
    if token_capacity is not None:
        physical["local_num_tokens"] = token_capacity
    if capacity is None:
        capacity = _resolve_receive_capacity(
            specification["initial_capacity_factor"],
            physical["local_num_tokens"] * specification["top_k"],
            specification["ep_size"],
        )
    signature = tuple(sorted((key, value) for key, value in specification.items() if key != "ep_group"))
    return _WorkspaceLayout(
        required_bytes=_spec_workspace_bytes(physical, dtype, receive_capacity=capacity,
                                             borrows_receive_buffer=borrows_receive_buffer),
        collective_signature=signature,
        receive_capacity=capacity,
        token_capacity=physical["local_num_tokens"] if specification.get("auto_tokens", False) else None,
        borrows_receive_buffer=borrows_receive_buffer,
    )


class _HeapManager:
    """Own MegaMoe Heap policy and the sole SHMEM Runtime user reference."""

    def __init__(self, group: Any, tensor: torch.Tensor) -> None:
        """Freeze one Root and tensor binding without acquiring the Runtime."""
        self.root_group = group
        self.members = _lifecycle.root_members(group)
        self.device = tensor.device
        self.dtype = tensor.dtype
        self.element_size = tensor.element_size()
        self.entries: list[_HeapEntry] = []
        self.epoch = 0
        self.state = "ready"
        self.growth_records: list[dict[str, Any]] = []
        self.heap_bytes = 0
        self._configured = False
        self._runtime_attached = False
        self._lock = threading.RLock()

    @property
    def workspaces(self) -> list[MegaMoeWorkspace]:
        """Return bound workspaces in deterministic Allocation order."""
        return [entry.workspace for entry in self.entries if entry.workspace is not None]

    @property
    def runtime_attached(self) -> bool:
        """Return whether this manager owns the process Runtime reference."""
        return self._runtime_attached

    @contextmanager
    def access(self) -> Iterator[None]:
        """Exclude concurrent submissions while allowing same-thread restart callbacks."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("SHMEM root does not support concurrent Heap operations")
        try:
            if self.state == "failed":
                raise RuntimeError(
                    "SHMEM Heap is no longer usable after a failed restart; restart the process"
                )
            yield
        finally:
            self._lock.release()

    def reserve(
        self,
        reservations: tuple[tuple[Mapping[str, Any], _WorkspaceLayout], ...],
        *,
        allow_resize: bool,
        workspace_growth: tuple[MegaMoeWorkspace, int, int] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Add lazy reservations and resize an attached Runtime when required."""
        with self.access():
            previous_count = len(self.entries)
            previous_configuration = self._configured, self.heap_bytes
            added_specifications = self._reserve(reservations)
            try:
                layouts = [entry.layout for entry in self.entries]
                growth = None
                if workspace_growth is not None:
                    if not self._runtime_attached:
                        raise RuntimeError("Workspace growth requires an attached SHMEM Runtime")
                    workspace, tokens, received = workspace_growth
                    growth = self._token_growth_layouts(self._entry_for_workspace(workspace), tokens, received)
                    if growth is not None:
                        _, layouts = growth
                if not self._configured:
                    configured = os.getenv("HYPER_MULTICORE_SHMEM_HEAP_SIZE")
                    self._run_and_check_consistency(
                        lambda: self._configure(configured),
                        stage="Configuration",
                        value=(configured, str(self.dtype), tuple(layouts)),
                    )
                    self._configured = True
                else:
                    self._run_and_check_consistency(
                        lambda: None,
                        stage="LayoutAgreement",
                        value=tuple(layouts),
                    )
                    required = self._required_bytes(layouts)
                    if required > self.heap_bytes or growth is not None:
                        if self.workspaces and not allow_resize:
                            raise RuntimeError("Cannot grow SHMEM while a static Pull workspace is bound")
                        heap_bytes = max(self.heap_bytes, required)
                        if self._runtime_attached:
                            self._restart(layouts, heap_bytes)
                        else:
                            self.heap_bytes = heap_bytes
                return added_specifications
            except Exception:
                if self.state == "ready":
                    del self.entries[previous_count:]
                    self._configured, self.heap_bytes = previous_configuration
                raise

    def _run_and_check_consistency(self, operation: Callable[[], None], *, stage: str, value: Any) -> None:
        """Run a local operation, then check failures and values across the Root group.

        One result exchange checks that every rank succeeded and supplied the same value. A single-rank configuration
        ValueError is propagated unchanged.
        """
        local_error = None
        try:
            operation()
        except Exception as error:  # pylint: disable=broad-exception-caught
            local_error = error
        report = None if local_error is None else _collective.failure_report(local_error)
        prefix = f"SHMEM Heap failed during {stage}"
        peers = _collective.exchange(
            (value, report),
            group=self.root_group,
            failure_prefix=f"{prefix} while converging root_rank results",
        )
        failures = [item[1] for item in peers]
        if any(failure is not None for failure in failures):
            if len(peers) == 1 and isinstance(local_error, ValueError):
                raise local_error
            details = _collective.format_collective_failures(
                failures,
                rank_label="root_rank",
                ranks_label="root_ranks",
            )
            raise RuntimeError(f"{prefix} on {details}") from local_error
        if any(item[0] != value for item in peers):
            raise RuntimeError(f"{prefix}: values differ across Root ranks")

    def rollback_reservations(
        self,
        added_specifications: tuple[Mapping[str, Any], ...],
        workspace: MegaMoeWorkspace | None,
    ) -> None:
        """Remove only uncommitted reservations from a collectively failed construction."""
        with self._lock:
            if workspace is not None:
                for entry in self.entries:
                    if entry.workspace is workspace:
                        entry.workspace = None
                        entry.bound_spec = None
            self.entries[:] = [
                entry
                for entry in self.entries
                if entry.workspace is not None
                or not any(entry.specification is specification for specification in added_specifications)
            ]

    def attach_runtime(self) -> bool:
        """Acquire the manager's sole Runtime reference if it is not already attached."""
        with self._lock:
            if self._runtime_attached:
                return False
            if not self._configured or self.heap_bytes <= 0:
                raise RuntimeError("SHMEM Heap must reserve a valid layout before attaching the Runtime")
            shmem.acquire(self.root_group, heap_size_bytes=self.heap_bytes)
            self._runtime_attached = True
            return True

    def detach_runtime(self) -> None:
        """Release the sole Runtime reference after all workspaces are gone."""
        with self._lock:
            if not self._runtime_attached:
                return
            if self.workspaces:
                raise RuntimeError("cannot detach SHMEM Runtime while MegaMoe workspaces remain bound")
            shmem.release()
            self._runtime_attached = False

    def bind(
        self,
        workspace: MegaMoeWorkspace,
        bound_spec: MegaMoeSpec,
        specification: Mapping[str, Any],
    ) -> int:
        """Bind a workspace and its physical specification to one reservation."""
        with self._lock:
            for entry in self.entries:
                if entry.specification is specification:
                    if entry.workspace is not None or entry.bound_spec is not None:
                        raise RuntimeError("SHMEM Heap owner is already bound")
                    entry.workspace = workspace
                    entry.bound_spec = bound_spec
                    if entry.layout.token_capacity is not None and (
                            entry.layout.token_capacity != bound_spec.local_num_tokens
                            or entry.layout.receive_capacity != bound_spec.receive_capacity):
                        entry.bound_spec = self._layout_spec(entry, entry.layout)
                    return entry.layout.receive_capacity
        raise RuntimeError("SHMEM Heap owner was not reserved")

    def workspace_spec(self, workspace: MegaMoeWorkspace) -> MegaMoeSpec:
        """Return the physical specification committed for a bound workspace."""
        with self._lock:
            spec = self._entry_for_workspace(workspace).bound_spec
            if spec is None:
                raise RuntimeError("SHMEM Heap workspace has no bound MegaMoe specification")
            return spec

    def close_workspace(self, workspace: MegaMoeWorkspace) -> None:
        """Close and remove one bound workspace without releasing the Runtime user."""
        with self._lock:
            workspace.close()
            self.entries[:] = [entry for entry in self.entries if entry.workspace is not workspace]

    def ensure_capacity(self, workspace: MegaMoeWorkspace, maximum_received_slots: int) -> None:
        """Grow one workspace while preserving its committed source capacity.

        Args:
            workspace: Owner of the receive reservation.
            maximum_received_slots: Largest current owner load in the EP group.
        """
        if maximum_received_slots <= workspace.capacity_floor:
            return
        with self.access():
            entry = self._entry_for_workspace(workspace)
            if entry.bound_spec is None:
                raise RuntimeError("SHMEM Heap workspace has no bound MegaMoe specification")
            capacity = _grow_receive_capacity(
                entry.bound_spec,
                entry.layout.receive_capacity,
                maximum_received_slots,
            )
            layout = self._replacement_layout(entry, capacity, entry.layout.token_capacity)
            layouts = [layout if item is entry else item.layout for item in self.entries]
            self._restart(layouts, max(self.heap_bytes, self._required_bytes(layouts)))

    def ensure_token_capacity(self, workspace: MegaMoeWorkspace, tokens: int, received: int) -> MegaMoeSpec:
        """Grow source and destination storage in one transaction after route agreement.

        Args:
            workspace: Automatic execution resource receiving this route.
            tokens: Group-maximum actual token rows, aligned for scheduling.
            received: Group-maximum actual destination rows.

        Returns:
            Committed physical specification, independent of retained execution plans.
        """
        with self.access():
            entry = self._entry_for_workspace(workspace)
            growth = self._token_growth_layouts(entry, tokens, received)
            if growth is None:
                return entry.bound_spec
            layout, layouts = growth
            required = self._required_bytes(layouts)
            heap_bytes = max(self.heap_bytes, required)
            if not self._grow_a5_push_storage(entry, layout, required, heap_bytes):
                self._restart(layouts, heap_bytes)
            return entry.bound_spec

    def borrow_mx_receive_buffer(self, workspace: MegaMoeWorkspace) -> None:
        """Retire the virtual receive allocation after home has materialized its larger buffer."""
        with self.access():
            entry = self._entry_for_workspace(workspace)
            owner = workspace.lease_owner
            layout = _workspace_layout(
                entry.specification, self.dtype, entry.layout.receive_capacity, entry.layout.token_capacity,
                borrows_receive_buffer=True,
            )
            started = time.perf_counter()

            def prepare() -> None:
                if (owner is workspace or owner.in_use or workspace.expert_buffer is None
                        or owner.expert_buffer is None or owner.expert_capacity < layout.receive_capacity):
                    raise RuntimeError("MX Push receive sharing requires a quiescent, sufficiently large home buffer")
                if owner.used:
                    owner.completion_event.synchronize()

            owner_index = next(index for index, candidate in enumerate(self.entries) if candidate is entry)
            self._run_and_check_consistency(
                prepare, stage="MX Push receive sharing", value=(owner_index, entry.layout, layout),
            )
            try:
                shmem.free(workspace.expert_buffer)
            except Exception:
                self.state = "failed"
                raise
            workspace.expert_buffer = None
            workspace.borrows_receive_buffer = True
            entry.layout = layout
            self.epoch += 1
            self.growth_records.append({
                "epoch": self.epoch, "mode": "mx_push_shared_receive", "heap_bytes": self.heap_bytes,
                "old_heap_bytes": self.heap_bytes, "total_ms": (time.perf_counter() - started) * 1000,
                "capacities": [item.layout.receive_capacity for item in self.entries],
            })

    def _token_growth_layouts(
        self, entry: _HeapEntry, tokens: int, received: int,
    ) -> tuple[_WorkspaceLayout, list[_WorkspaceLayout]] | None:
        """Project one dynamic workspace growth against all pending reservations."""
        spec = entry.bound_spec
        if spec is None or not spec.auto_tokens:
            raise RuntimeError("Automatic token growth requires an automatic workspace")
        grow_source = tokens > spec.local_num_tokens
        grow_receive = spec.dispatch_mode == "push" and received > entry.layout.receive_capacity
        if not grow_source and not grow_receive:
            return None
        factor = spec.capacity_growth_factor or _DEFAULT_CAPACITY_FACTOR
        token_limit = ((2**31 - 1) // max(spec.hidden_size, spec.top_k) // 128) * 128
        token_capacity = (_align_capacity(max(tokens, math.ceil(min(token_limit, spec.local_num_tokens * factor))))
                          if grow_source else spec.local_num_tokens)
        receive_capacity = entry.layout.receive_capacity
        if grow_receive:
            receive_capacity = _align_capacity(max(received, math.ceil(min(2**31 - 1, receive_capacity * factor))))
        layout = self._replacement_layout(entry, receive_capacity, token_capacity)
        layouts = [layout if item is entry else item.layout for item in self.entries]
        return layout, layouts

    def _grow_a5_push_storage(
        self, entry: _HeapEntry, layout: _WorkspaceLayout, required: int, heap_bytes: int,
    ) -> bool:
        """Grow Ascend950 Push scratch inside the attached heap."""
        spec = entry.bound_spec
        workspace = entry.workspace
        if (not self._runtime_attached or heap_bytes != self.heap_bytes
                or spec.architecture != 3510 or spec.precision not in ("bf16", "mxfp8")
                or spec.dispatch_mode != "push" or spec.apply_probs_before_fc2):
            return False
        new_spec = self._layout_spec(entry, layout)
        replacements = workspace.describe_restart_allocations(new_spec, self.dtype, layout.receive_capacity)
        # Push saves owned activations and reads current scratch when backward
        # launches. Event storage and expert banks keep their allocations.
        old_layout = entry.layout
        data_fields = {id(workspace.expert_buffer): "expert_buffer", id(workspace.routed_buffer): "routed_buffer"}
        changed = tuple((data_fields[id(item.current)], item) for item in replacements
                        if id(item.current) in data_fields and tuple(item.current.shape) != item.target.shape)
        temporary_bytes = sum(item.current.numel() * self.element_size for _, item in changed)
        release_first = required + temporary_bytes > self.heap_bytes
        if release_first and (spec.precision != "mxfp8"
                              or tuple(name for name, _ in changed) != ("expert_buffer",)):
            return False

        started = time.perf_counter()

        def prepare() -> None:
            workspace.validate_restart_allocations(spec, self.dtype, old_layout.receive_capacity, replacements)
            owner = workspace.lease_owner
            if changed and owner.used:
                owner.completion_event.synchronize()

        owner_index = next(index for index, candidate in enumerate(self.entries) if candidate is entry)
        signature = (release_first, tuple((name, item.target) for name, item in changed))
        self._run_and_check_consistency(
            prepare, stage="Ascend950 Push growth preflight", value=(owner_index, old_layout, layout, signature),
        )
        try:
            if release_first:
                stages_ms = self._replace_mx_push_receive(workspace, changed[0][1])
                if stages_ms is None:
                    return False
            else:
                stages_ms = self._replace_push_buffers(workspace, changed)
        except Exception:
            self.state = "failed"
            raise
        if replacements:
            workspace.expert_capacity = layout.receive_capacity
            workspace.routed_slots = new_spec.routed_slots
        workspace.capacity_floor = layout.receive_capacity
        entry.bound_spec = new_spec
        entry.layout = layout
        self.epoch += 1
        self.growth_records.append({
            "epoch": self.epoch,
            "mode": "mx_push_storage" if spec.precision == "mxfp8" else "bf16_push_storage",
            "old_heap_bytes": self.heap_bytes,
            "heap_bytes": self.heap_bytes,
            "old_receive": old_layout.receive_capacity,
            "receive": layout.receive_capacity,
            "old_source": spec.local_num_tokens,
            "source": new_spec.local_num_tokens,
            "replaced": [name for name, _ in changed],
            "release_first": release_first,
            "total_ms": (time.perf_counter() - started) * 1000,
            "stages_ms": stages_ms,
            "capacities": [item.layout.receive_capacity for item in self.entries],
        })
        return True

    @torch.inference_mode(False)
    def _replace_mx_push_receive(
        self, workspace: MegaMoeWorkspace, replacement: _lifecycle._AllocationReplacement,
    ) -> dict[str, float] | None:
        """Retire quiescent receive scratch before allocating its larger buffer."""
        current, target = replacement.current, replacement.target
        old_shape, old_dtype = tuple(current.shape), current.dtype

        def release() -> None:
            shmem.free(current)
            workspace.expert_buffer = None

        def allocate(shape: tuple[int, ...], dtype: torch.dtype) -> None:
            workspace.expert_buffer = shmem.empty(shape, dtype=dtype, alignment=target.alignment_bytes)

        started = time.perf_counter()
        _collective.collective_call(release, stage="MX Push receive release", group=self.root_group)
        released = time.perf_counter()
        failure = None
        recoverable = False
        local_error = None
        try:
            allocate(target.shape, target.dtype)
        except Exception as error:  # pylint: disable=broad-exception-caught
            local_error = error
            failure = _collective.failure_report(error)
            # A null result from the native aligned allocator did not register a new Allocation.
            recoverable = (failure[1].startswith(
                "[HYPER_MULTICORE_SHMEM_ERROR] error_code=CANN_ERROR operation=Allocate phase=CannCall "
            ) and _lifecycle._runtime_is_healthy())
        peers = _collective.exchange((failure, recoverable), group=self.root_group,
                                     failure_prefix="MX Push receive allocation failed while converging Root ranks")
        if all(item[0] is None for item in peers):
            return {"release_ms": (released - started) * 1000,
                    "allocate_ms": (time.perf_counter() - released) * 1000}
        if all(item[1] for item in peers):
            # Restore the committed scratch shape before the usual whole-heap fallback.
            _collective.collective_call(lambda: allocate(old_shape, old_dtype),
                                        stage="MX Push receive restoration", group=self.root_group)
            return None
        failures = _collective.format_collective_failures([item[0] for item in peers])
        raise RuntimeError(f"MX Push receive allocation failed on {failures}") from local_error

    @staticmethod
    def _replace_push_buffers(
        workspace: MegaMoeWorkspace, changed: tuple[tuple[str, _lifecycle._AllocationReplacement], ...],
    ) -> dict[str, float]:
        """Allocate replacements before releasing quiescent scratch tensors."""
        started = time.perf_counter()
        staged = []
        try:
            for name, item in changed:
                target = item.target
                tensor = shmem.empty(target.shape, dtype=target.dtype, alignment=target.alignment_bytes)
                staged.append((name, tensor))
        except Exception:
            for _, tensor in reversed(staged):
                shmem.free(tensor)
            raise
        bound = time.perf_counter()
        # Publish before freeing so a failed collective free cannot strand
        # the new Allocation outside its sole workspace owner.
        for name, tensor in staged:
            setattr(workspace, name, tensor)
        for _, item in changed:
            shmem.free(item.current)
        return {"allocate_ms": (bound - started) * 1000, "release_ms": (time.perf_counter() - bound) * 1000}

    def _replacement_layout(self, entry: _HeapEntry, received: int, tokens: int | None) -> _WorkspaceLayout:
        """Retain the event stride across capacity growth."""
        return _workspace_layout(entry.specification, self.dtype, received, tokens,
                                 borrows_receive_buffer=entry.layout.borrows_receive_buffer)

    @staticmethod
    def _layout_spec(entry: _HeapEntry, layout: _WorkspaceLayout) -> MegaMoeSpec:
        """Project candidate capacities without modifying a committed specification."""
        spec = entry.bound_spec
        if layout.token_capacity is None:
            return spec
        return replace(spec, local_num_tokens=layout.token_capacity,
                       receive_capacity=layout.receive_capacity)

    @staticmethod
    def _required_bytes(layouts: list[_WorkspaceLayout]) -> int:
        """Return the aggregate payload rounded once to physical Heap pages."""
        return _round_heap_bytes(sum(layout.required_bytes for layout in layouts))

    def _configure(self, configured: str | None) -> None:
        """Size the initial heap for the reserved layouts and any larger user seed."""
        required = sum(entry.layout.required_bytes for entry in self.entries)
        if configured is None:
            self.heap_bytes = _round_heap_bytes(required)
            return
        try:
            initial_bytes = int(configured)
        except ValueError as error:
            raise ValueError(
                "HYPER_MULTICORE_SHMEM_HEAP_SIZE must be a positive integer number of bytes"
            ) from error
        if initial_bytes <= 0:
            raise ValueError("HYPER_MULTICORE_SHMEM_HEAP_SIZE must be positive")
        if initial_bytes % _HEAP_GRANULARITY_BYTES:
            raise ValueError(
                "HYPER_MULTICORE_SHMEM_HEAP_SIZE must be a multiple of the 2 MiB physical page size"
            )
        self.heap_bytes = max(initial_bytes, _round_heap_bytes(required))

    def _reserve(
        self,
        reservations: tuple[tuple[Mapping[str, Any], _WorkspaceLayout], ...],
    ) -> tuple[Mapping[str, Any], ...]:
        """Append specifications by identity without disturbing existing order."""
        added = []
        for specification, layout in reservations:
            if not any(entry.specification is specification for entry in self.entries):
                self.entries.append(_HeapEntry(specification, layout))
                added.append(specification)
        return tuple(added)

    def _entry_for_workspace(self, workspace: MegaMoeWorkspace) -> _HeapEntry:
        """Find one bound workspace by identity."""
        for entry in self.entries:
            if entry.workspace is workspace:
                return entry
        raise RuntimeError("SHMEM Heap owner was not bound")

    def _describe_restart(
        self,
        layouts: list[_WorkspaceLayout],
    ) -> tuple[
        tuple[_lifecycle._AllocationReplacement, ...],
        tuple[_RestartBinding, ...],
        _RestartAgreement,
    ]:
        """Describe physical replacements and result slices without validating business state."""
        replacements = []
        bindings = []
        agreements = []
        for entry, layout in zip(self.entries, layouts):
            is_bound = entry.workspace is not None
            materialized = False
            if is_bound:
                start = len(replacements)
                workspace_replacements = entry.workspace.describe_restart_allocations(
                    self._layout_spec(entry, layout),
                    self.dtype,
                    layout.receive_capacity,
                )
                replacements.extend(workspace_replacements)
                materialized = bool(workspace_replacements)
                if materialized:
                    bindings.append(_RestartBinding(entry, start, len(replacements)))
            agreements.append(_RestartEntryAgreement(layout, is_bound, materialized))
        return (
            tuple(replacements),
            tuple(bindings),
            _RestartAgreement(self.epoch, tuple(agreements)),
        )

    def _restart_preflight(
        self,
        layouts: list[_WorkspaceLayout],
        replacements: tuple[_lifecycle._AllocationReplacement, ...],
        bindings: tuple[_RestartBinding, ...],
        agreement: _RestartAgreement,
    ) -> _RestartAgreement:
        """Validate the frozen business snapshot and return its rank-independent agreement."""
        if self.state != "ready":
            raise RuntimeError("SHMEM Heap is not ready for restart")
        if len(layouts) != len(self.entries):
            raise RuntimeError("SHMEM Heap entries changed during restart preflight")
        binding_by_entry = {id(binding.entry): binding for binding in bindings}
        for entry, entry_agreement in zip(self.entries, agreement.entries):
            if entry.workspace is None:
                continue
            if entry.bound_spec is None:
                raise RuntimeError("SHMEM Heap workspace has no bound MegaMoe specification")
            binding = binding_by_entry.get(id(entry))
            workspace_replacements = () if binding is None else replacements[binding.start:binding.stop]
            materialized = entry.workspace.validate_restart_allocations(
                entry.bound_spec,
                self.dtype,
                entry.layout.receive_capacity,
                workspace_replacements,
            )
            if materialized != entry_agreement.is_materialized:
                raise RuntimeError("SHMEM Heap materialization state changed during preflight")
        return agreement

    def _validate_restart_result(
        self,
        result: _lifecycle._RestartResult,
        bindings: tuple[_RestartBinding, ...],
        layouts: list[_WorkspaceLayout],
    ) -> None:
        """Validate every result slice before assigning any workspace field."""
        expected_count = sum(binding.stop - binding.start for binding in bindings)
        if len(result.tensors) != expected_count:
            raise RuntimeError("SHMEM restart returned an unexpected Allocation count")
        layout_by_entry = {id(entry): layout for entry, layout in zip(self.entries, layouts)}
        for binding in bindings:
            entry = binding.entry
            if entry.workspace is None or entry.bound_spec is None:
                raise RuntimeError("SHMEM Heap workspace binding changed after restart")
            entry.workspace.validate_restarted_allocations(
                self._layout_spec(entry, layout_by_entry[id(entry)]),
                self.dtype,
                layout_by_entry[id(entry)].receive_capacity,
                result.tensors[binding.start:binding.stop],
            )

    def _bind_restart_result(
        self,
        result: _lifecycle._RestartResult,
        bindings: tuple[_RestartBinding, ...],
        layouts: list[_WorkspaceLayout],
    ) -> None:
        """Bind validated result slices back to their original workspace objects."""
        layout_by_entry = {id(entry): layout for entry, layout in zip(self.entries, layouts)}
        for binding in bindings:
            entry = binding.entry
            entry.workspace.bind_restarted_allocations(
                self._layout_spec(entry, layout_by_entry[id(entry)]),
                self.dtype,
                self.device,
                layout_by_entry[id(entry)].receive_capacity,
                result.tensors[binding.start:binding.stop],
            )

    def _restart(self, layouts: list[_WorkspaceLayout], heap_bytes: int) -> None:
        """Restart SHMEM, validate and bind local results, then publish business state."""
        started = time.perf_counter()
        replacements, bindings, agreement = self._describe_restart(layouts)
        old_bytes = self.heap_bytes

        def raise_post_restart_failure(error: Exception, *, context: str) -> NoReturn:
            """Invalidate the manager and report a post-restart business-state failure."""
            self.state = "failed"
            raise RuntimeError(
                f"{context}: {error}. The restarted SHMEM Allocations cannot be rolled back; "
                "terminate the process and let the launcher restart the job."
            ) from error

        try:
            result = shmem._restart(  # pylint: disable=protected-access
                heap_size_bytes=heap_bytes,
                replacements=replacements,
                preflight_hook=lambda: self._restart_preflight(layouts, replacements, bindings, agreement),
            )
        except Exception:
            if not _lifecycle._runtime_is_healthy():  # pylint: disable=protected-access
                self.state = "failed"
            raise

        context = "MegaMoe restart result validation failed due to a local implementation error"
        try:
            self._validate_restart_result(result, bindings, layouts)
            context = "MegaMoe workspace binding failed after SHMEM restart due to a local implementation error"
            self._bind_restart_result(result, bindings, layouts)
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise_post_restart_failure(error, context=context)

        stages_ms = dict(result.stages_ms)
        for entry, layout in zip(self.entries, layouts):
            if entry.bound_spec is not None:
                entry.bound_spec = self._layout_spec(entry, layout)
            entry.layout = layout
            if entry.workspace is not None:
                entry.workspace.capacity_floor = layout.receive_capacity
        self.heap_bytes = heap_bytes
        self.epoch += 1
        self.growth_records.append({
            "epoch": self.epoch,
            "old_heap_bytes": old_bytes,
            "heap_bytes": heap_bytes,
            "stages_ms": stages_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
            "capacities": [entry.layout.receive_capacity for entry in self.entries],
        })


def _get_heap_manager(group: Any, tensor: torch.Tensor) -> _HeapManager:
    """Return the compatible process MegaMoe Heap Manager."""
    global _HEAP_MANAGER  # pylint: disable=global-statement
    if (_HEAP_MANAGER is None
            or not _HEAP_MANAGER.workspaces and not _HEAP_MANAGER.runtime_attached):
        _HEAP_MANAGER = _HeapManager(group, tensor)
        return _HEAP_MANAGER
    if _HEAP_MANAGER.members != _lifecycle.root_members(group) or _HEAP_MANAGER.device != tensor.device:
        raise RuntimeError("SHMEM Heap requires one device and ordered Root per process")
    if _HEAP_MANAGER.dtype != tensor.dtype or _HEAP_MANAGER.element_size != tensor.element_size():
        raise ValueError("SHMEM Heap owners must use the same dtype and element size")
    return _HEAP_MANAGER
