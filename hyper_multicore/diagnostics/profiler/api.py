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
"""Standalone Torch MegaKernel profiling API and capture lifecycle."""

from __future__ import annotations

__all__ = ["TorchMegaKernelProfiler", "mega_kernel_profile"]

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch

from hyper_multicore.diagnostics.common.session import (
    _activate_session,
    _deactivate_session,
)
from hyper_multicore.diagnostics.profiler.cycle_trace import _PreparedProfileRuntime
from hyper_multicore.diagnostics.common.call import DiagnosticCall, LaunchArguments, LaunchContext
from hyper_multicore.diagnostics.common.schedule import DiagnosticAction


@dataclass
class _RawInvocation:
    """One completed invocation copied from Device to Host."""

    runtime: _PreparedProfileRuntime
    data: bytes
    direction: str
    step: int
    invocation_id: int
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class _TraceAccumulator:
    """Aggregate parsed invocations into one capture window."""

    global_anchor: int
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    metadata_keys: set[tuple[Any, Any, Any]] = field(default_factory=set)
    dropped_count: int = 0
    record_count: int = 0
    invocation_metadata: list[dict[str, Any]] = field(default_factory=list)

    def append(self, invocation: _RawInvocation, trace: dict[str, Any]) -> None:
        """Append one parsed invocation and shift it to the window anchor.

        Args:
            invocation: Invocation identity and captured bytes.
            trace: Parsed standalone trace for the invocation.
        """
        metadata = trace["megaKernelCycleTrace"]
        shift_us = (metadata["anchorCycle"] - self.global_anchor) / metadata["cycleFrequencyMHz"]
        self.dropped_count += metadata["droppedRecordCount"]
        self.record_count += metadata["recordCount"]
        invocation_metadata = {
            "invocationId": invocation.invocation_id,
            "step": invocation.step,
            "direction": invocation.direction,
            "kernelName": invocation.runtime.kernel_name,
            "rank": metadata.get("rank", invocation.runtime.rank),
            "deviceId": metadata.get("deviceId", invocation.runtime.device_id),
            "anchorCycle": metadata["anchorCycle"],
            "recordCount": metadata["recordCount"],
            "droppedRecordCount": metadata["droppedRecordCount"],
        }
        for key in ('workerCapacities', 'profileBufferBytes'):
            if key in metadata:
                invocation_metadata[key] = metadata[key]
        if invocation.tags:
            invocation_metadata['tags'] = dict(invocation.tags)
        self.invocation_metadata.append(invocation_metadata)
        self._append_events(trace["traceEvents"], invocation, shift_us)

    def _append_events(
        self,
        events: list[dict[str, Any]],
        invocation: _RawInvocation,
        shift_us: float,
    ) -> None:
        """Deduplicate metadata events and shift duration events."""
        for event in events:
            if event["ph"] == "M":
                key = (event["name"], event["pid"], event["tid"])
                if key not in self.metadata_keys:
                    self.metadata_keys.add(key)
                    self.trace_events.append(event)
                continue
            copied_event = dict(event)
            copied_event["ts"] += shift_us
            copied_event["args"] = {
                **event.get("args", {}),
                "invocation_id": invocation.invocation_id,
                "step": invocation.step,
                "direction": invocation.direction,
                "scope": dict(invocation.tags),
            }
            self.trace_events.append(copied_event)


class _ProfileCall(DiagnosticCall):
    """Internal launch arguments and profiling-buffer ownership for one kernel call."""

    def __init__(
        self,
        *,
        runtime_config: Any,
        event_counters: Any,
        diagnostic_buffer: Any,
        clear_event_counters: Any | None = None,
        profiler: TorchMegaKernelProfiler | None = None,
        runtime: _PreparedProfileRuntime | None = None,
        profile_buffer_key: tuple[int, int] | None = None,
        direction: str = "",
        step: int = 0,
        invocation_id: int = 0, tags: dict[str, str] | None = None,
    ) -> None:
        """Store launch arguments and optional active-session bookkeeping."""
        super().__init__(LaunchArguments(
            runtime_config, event_counters, diagnostic_buffer,
            event_counters if clear_event_counters is None else clear_event_counters,
        ))
        self._profiler = profiler
        self.runtime = runtime
        self.profile_buffer_key = profile_buffer_key
        self.direction = direction
        self.step = step
        self.invocation_id = invocation_id
        self.tags = dict(tags or {})
        self.finished = False

    def complete(self) -> None:
        """Mark the direct-write profile buffer as complete after a successful launch."""
        if self.finished:
            return
        if self._profiler is not None:
            self._profiler.complete_call(self)
        self.finished = True

    def cancel(self) -> None:
        """Discard an unlaunched call and release its private profile buffer."""
        if self.finished:
            return
        if self._profiler is not None:
            self._profiler.cancel_call(self)
        self.finished = True


class TorchMegaKernelProfiler:
    """Standalone schedule-driven MegaKernel internal profiler for Torch."""

    def __init__(
        self,
        *,
        schedule: Callable[[int], DiagnosticAction] | None,
        on_trace_ready: Callable[[Any], None] | None,
        detailed_task_names: bool,
        max_pending_calls: int,
    ) -> None:
        # pylint: disable=redefined-outer-name
        """Validate options and initialize the schedule-driven capture state."""
        if schedule is not None and not callable(schedule):
            raise TypeError("schedule must be callable or None")
        if on_trace_ready is not None and not callable(on_trace_ready):
            raise TypeError("on_trace_ready must be callable or None")
        if not isinstance(detailed_task_names, bool):
            raise TypeError("detailed_task_names must be bool")
        if not isinstance(max_pending_calls, int) or isinstance(max_pending_calls, bool) or max_pending_calls <= 0:
            raise ValueError("max_pending_calls must be a positive integer, " f"got {max_pending_calls!r}")
        self._schedule = schedule
        self._on_trace_ready = on_trace_ready
        self._detailed_task_names = detailed_task_names
        self._max_pending_calls = max_pending_calls
        self._step = 0
        self._action = self._action_for_step(0)
        self._entered = False
        self._next_invocation_id = 0
        self._pending: list[_ProfileCall] = []
        self._profile_buffer_pool: dict[tuple[int, int], list[Any]] = {}
        self._profile_runtimes: dict[Any, _PreparedProfileRuntime] = {}
        self._raw_invocations: list[_RawInvocation] = []
        self._last_trace: dict[str, Any] | None = None
        self._window_index = 0

    @property
    def step_num(self) -> int:
        """Return the zero-based step currently being recorded."""
        return self._step

    @property
    def current_action(self) -> DiagnosticAction:
        """Return the current schedule action."""
        return self._action

    def __enter__(self) -> TorchMegaKernelProfiler:
        """Activate this profiler as the process-local MegaKernel session."""
        _activate_session("profiler", self)
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Finalize successful captures or discard pending data on failure."""
        try:
            if exc_type is None:
                self._finish_context()
            else:
                self._drain(keep=False)
        finally:
            self._raw_invocations.clear()
            for call in self._pending:
                self._release_profile_buffer(call, cache=False)
            self._pending.clear()
            self._release_profile_buffer_pool()
            _deactivate_session(self)
            self._profile_runtimes.clear()
            self._entered = False

    def step(self) -> None:
        """Finish the current schedule step and advance to the next one."""
        if not self._entered:
            raise RuntimeError("profiler.step() must be called inside its context")
        next_action = self._action_for_step(self._step + 1)
        if self._action is DiagnosticAction.WARMUP:
            self._drain(keep=False)
        elif self._action is DiagnosticAction.RECORD_AND_SAVE:
            self._drain(keep=True)
            self._finalize_window()
        elif self._action is DiagnosticAction.RECORD and next_action not in (
            DiagnosticAction.RECORD,
            DiagnosticAction.RECORD_AND_SAVE,
        ):
            self._drain(keep=True)
            self._finalize_window()
        self._step += 1
        self._action = next_action

    def export_chrome_trace(self, path: str | Path) -> dict[str, Any]:
        """Write the most recently completed internal capture window.

        Args:
            path: Destination path for the Chrome Trace JSON object.
        """
        if self._last_trace is None:
            raise RuntimeError(
                "no completed MegaKernel capture window is available; "
                "call step() at the window boundary or leave the context first"
            )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self._last_trace, ensure_ascii=False),
            encoding="utf-8",
        )
        return self._last_trace

    def _action_for_step(self, step: int) -> DiagnosticAction:
        if self._schedule is None:
            return DiagnosticAction.RECORD
        action = self._schedule(step)
        if not isinstance(action, DiagnosticAction):
            raise TypeError("MegaKernel schedule must return DiagnosticAction, " f"got {type(action).__name__}")
        return action

    def is_capture_enabled(self) -> bool:
        """Return whether the current schedule action records Device cycles."""
        return self._action is not DiagnosticAction.NONE

    def prepare_call(self, context: LaunchContext) -> DiagnosticCall:
        """Select profiling independently, without constructing any dump state.

        Args:
            context: Selected invocation or operand context supplied by the diagnostic framework.
        """
        if not self.is_capture_enabled():
            return DiagnosticCall(context.ordinary_arguments())
        runtime = self._profile_runtimes.get(context.runtime)
        if runtime is None:
            runtime = _PreparedProfileRuntime(context.runtime)
            self._profile_runtimes[context.runtime] = runtime
        return self.acquire_call(runtime, context.fallback_event_counters, context.direction, tags=context.tags)

    def acquire_call(
        self,
        runtime: _PreparedProfileRuntime,
        event_counters: Any,
        direction: str, tags: dict[str, str] | None = None,
    ) -> _ProfileCall:
        """Reserve a profiling buffer and register one pending kernel call.

        Args:
            runtime: Prepared runtime configuration and profiling metadata.
            event_counters: Device event-counter storage.
            direction: Kernel direction label included in trace metadata.
        """
        if len(self._pending) >= self._max_pending_calls:
            self._drain(keep=self._action is not DiagnosticAction.WARMUP)
        invocation_id = self._next_invocation_id
        profile_buffer, profile_buffer_key = self._take_profile_buffer(runtime)
        self._next_invocation_id += 1
        call = _ProfileCall(
            runtime_config=runtime.profile_tensor,
            event_counters=event_counters,
            clear_event_counters=event_counters,
            profiler=self,
            diagnostic_buffer=profile_buffer,
            runtime=runtime,
            direction=direction,
            step=self._step,
            profile_buffer_key=profile_buffer_key,
            invocation_id=invocation_id, tags=tags,
        )
        self._pending.append(call)
        return call

    def complete_call(self, call: _ProfileCall) -> None:
        """Validate that a successfully launched call belongs to this session.

        Args:
            call: Pending kernel call that completed launch.
        """
        if call not in self._pending:
            raise RuntimeError("completed MegaKernel profiling call is not pending")

    def _take_profile_buffer(
        self,
        runtime: _PreparedProfileRuntime,
    ) -> tuple[Any, tuple[int, int]]:
        """Take one ordinary NPU buffer that is private to the next invocation."""
        profile_buffer_key = (runtime.device_id, runtime.buffer_size)
        available_buffers = self._profile_buffer_pool.get(profile_buffer_key)
        if available_buffers:
            profile_buffer = available_buffers.pop()
        else:
            profile_buffer = torch.empty(
                (runtime.buffer_size,),
                dtype=torch.uint8,
                device=runtime.device,
            )
        profile_buffer.zero_()
        return profile_buffer, profile_buffer_key

    def cancel_call(self, call: _ProfileCall) -> None:
        """Cancel a pending call and release its profiling buffer.

        Args:
            call: Pending kernel call to cancel.
        """
        if call in self._pending:
            self._pending.remove(call)
        self._release_profile_buffer(call, cache=False)

    def _release_profile_buffer(
        self,
        call: _ProfileCall,
        *,
        cache: bool,
    ) -> None:
        """Return a completed buffer to the session pool or release its storage."""
        profile_buffer = call.diagnostic_buffer
        profile_buffer_key = call.profile_buffer_key
        if profile_buffer is None or profile_buffer_key is None:
            return
        if cache:
            self._profile_buffer_pool.setdefault(profile_buffer_key, []).append(profile_buffer)
        else:
            profile_buffer.untyped_storage().resize_(0)
        call.arguments.diagnostic_buffer = None
        call.profile_buffer_key = None

    def _release_profile_buffer_pool(self) -> None:
        """Release ordinary NPU buffers retained for reuse by this session."""
        for available_buffers in self._profile_buffer_pool.values():
            for profile_buffer in available_buffers:
                profile_buffer.untyped_storage().resize_(0)
        self._profile_buffer_pool.clear()

    def _drain(self, *, keep: bool) -> None:
        """Synchronize pending calls, optionally copy their records to Host."""
        if not self._pending:
            if not keep:
                self._raw_invocations.clear()
            return
        unfinished = [call for call in self._pending if not call.finished]
        if unfinished:
            raise RuntimeError("cannot copy MegaKernel profiling buffers before launches complete")

        device_ids = {call.runtime.device_id for call in self._pending if call.runtime is not None}
        for device_id in device_ids:
            torch.npu.synchronize(device_id)

        for call in self._pending:
            runtime = call.runtime
            profile_buffer = call.diagnostic_buffer
            if keep and runtime is not None and profile_buffer is not None:
                host_data = profile_buffer.cpu().numpy().tobytes()
                self._raw_invocations.append(
                    _RawInvocation(
                        runtime=runtime,
                        data=host_data,
                        direction=call.direction,
                        step=call.step,
                        invocation_id=call.invocation_id, tags=dict(call.tags),
                    )
                )
            self._release_profile_buffer(call, cache=True)
        self._pending.clear()
        if not keep:
            self._raw_invocations.clear()

    def _finish_context(self) -> None:
        if self._action is DiagnosticAction.WARMUP:
            self._drain(keep=False)
            return
        if self._pending or self._raw_invocations:
            self._drain(keep=True)
            if self._raw_invocations:
                self._finalize_window()

    def _finalize_window(self) -> None:
        try:
            self._last_trace = self._build_trace(self._raw_invocations)
        finally:
            self._raw_invocations.clear()
        self._window_index += 1
        if self._on_trace_ready is not None:
            self._on_trace_ready(self)

    def _build_trace(self, invocations: list[_RawInvocation]) -> dict[str, Any]:
        """Combine retained invocation data into one standalone Chrome Trace."""
        if not invocations:
            return {
                "traceEvents": [],
                "megaKernelCycleTrace": {
                    "schemaVersion": 1,
                    "windowIndex": self._window_index,
                    "invocationCount": 0,
                    "recordCount": 0,
                    "droppedRecordCount": 0,
                    "detailedTaskNames": self._detailed_task_names,
                    "invocations": [],
                    "warnings": [],
                },
            }

        parsed_invocations = [
            (
                invocation,
                invocation.runtime.parse(
                    invocation.data,
                    detailed_task_names=self._detailed_task_names,
                ),
            )
            for invocation in invocations
        ]
        global_anchor = min(trace["megaKernelCycleTrace"]["anchorCycle"] for _, trace in parsed_invocations)
        accumulator = _TraceAccumulator(global_anchor)
        for invocation, trace in parsed_invocations:
            accumulator.append(invocation, trace)

        metadata_events = [event for event in accumulator.trace_events if event["ph"] == "M"]
        duration_events = sorted(
            (event for event in accumulator.trace_events if event["ph"] != "M"),
            key=lambda event: (event["ts"], event["pid"], event["tid"]),
        )
        return {
            "traceEvents": metadata_events + duration_events,
            "megaKernelCycleTrace": {
                "schemaVersion": 1,
                "windowIndex": self._window_index,
                "anchorCycle": global_anchor,
                "invocationCount": len(invocations),
                "recordCount": accumulator.record_count,
                "droppedRecordCount": accumulator.dropped_count,
                "detailedTaskNames": self._detailed_task_names,
                "invocations": accumulator.invocation_metadata,
                "warnings": (
                    []
                    if accumulator.dropped_count == 0
                    else [f"Device dropped {accumulator.dropped_count} cycle trace records"]
                ),
            },
        }


# The argument name intentionally mirrors torch.profiler.profile.
def mega_kernel_profile(
    *,
    schedule: Callable[[int], DiagnosticAction] | None = None,
    on_trace_ready: Callable[[Any], None] | None = None,
    detailed_task_names: bool = False,
    max_pending_calls: int = 16,
) -> TorchMegaKernelProfiler:
    # pylint: disable=redefined-outer-name
    """Create the standalone MegaKernel profiler.

    The returned context manager records only MegaKernel-internal events. It
    does not start an external framework profiler, and its export operation
    does not merge with an external trace.

    Args:
        schedule: Optional step schedule. Without one, all calls are retained.
        on_trace_ready: Optional callback invoked after each completed window.
        detailed_task_names: Include owner and task numbers in event names.
        max_pending_calls: Maximum in-flight invocation buffers before an
            intermediate D2H drain and reuse.

    Returns:
        Standalone MegaKernel profiler context manager.
    """
    return TorchMegaKernelProfiler(
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        detailed_task_names=detailed_task_names,
        max_pending_calls=max_pending_calls,
    )
