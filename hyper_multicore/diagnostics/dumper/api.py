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
"""Schedule-driven Tensor dump for Torch MegaKernel invocations."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import torch

from hyper_multicore.diagnostics.common.session import (
    _activate_session,
    _deactivate_session,
)
from hyper_multicore.diagnostics.common.call import DiagnosticCall, LaunchArguments, LaunchContext
from hyper_multicore.diagnostics.common.schedule import DiagnosticAction
from hyper_multicore.diagnostics.dumper._layout import _SAFE_FILE_NAME, _stage_names
from hyper_multicore.diagnostics.dumper._kernel import (
    _GraphTensor, _KernelDumpTensorSnapshot, _describe_graph_tensors, _snapshot_tensor, _write_kernel_stage_files,
)
from hyper_multicore.diagnostics.dumper._task import (
    _TaskDumpTensorSnapshot,
    _PreparedTaskDump,
    _decode_task_dump,
    _prepare_task_dump,
)


__all__ = [
    "TensorDumpLevel",
    "TensorDumpConfig",
    "TensorDumpSession",
    "tensor_dump",
]


_VALID_DIRECTIONS = frozenset(("forward", "backward"))
_VALID_ROLES = frozenset(("external_input", "stage_output", "final_output"))
_VALID_IO = frozenset(("input", "output"))
_VALID_CORE_TYPES = frozenset(("AIC", "AIV"))


def _normalize_string_filter(
    value: str | Collection[str] | None,
    *,
    name: str,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...] | None:
    """Normalize a scalar or collection of named selection values."""
    if value is None:
        return None
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    for item in values:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} entries must be non-empty strings, got {item!r}")
        if allowed is not None and item not in allowed:
            raise ValueError(f"unsupported {name} entry {item!r}; expected one of {sorted(allowed)}")
    return values


def _normalize_non_negative_int_filter(
    value: int | Collection[int] | None,
    *,
    name: str,
) -> tuple[int, ...] | None:
    """Normalize integer selections while rejecting booleans and negative IDs."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} entries must be non-negative integers, got {value!r}")
    values = (value,) if isinstance(value, int) else tuple(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    for item in values:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{name} entries must be non-negative integers, got {item!r}")
    return values


def _normalize_ranks(value: int | Collection[int] | None) -> tuple[int, ...] | None:
    return _normalize_non_negative_int_filter(value, name="ranks")


class TensorDumpLevel(str, Enum):
    """Execution scope at whose before/after points Tensor values are dumped."""

    KERNEL = "kernel"
    TASK = "task"


@dataclass(frozen=True)
class TensorDumpConfig:
    """Configuration for MegaKernel Tensor value collection.

    Args:
        output_dir: Root directory for per-rank dump windows.
        level: Kernel or Task dump level. Accepts TensorDumpLevel or a
            case-insensitive ``"kernel"`` / ``"task"`` string. Defaults to Kernel.
        kernel_names: Optional exact MegaKernel names whose invocations should
            be retained.
        tensor_names: Optional exact logical Tensor names selected from the
            MegaKernel compute graph.
        roles: Optional Tensor roles for Kernel dump. Supported roles are
            ``external_input``, ``stage_output`` and ``final_output``.
        directions: Forward and/or backward invocations to retain.
        ranks: Optional non-negative distributed ranks to retain.
        stages: Optional exact Task stage names.
        task_ids: Optional task IDs in this invocation's diagnostic RuntimeConfig.
            Diagnostic adaptation may renumber them relative to standard execution.
        stage_task_indices: Optional task indices local to a stage.
        owner_ids: Optional business diagnostic owner IDs.
        core_types: Optional Task Device worker types, ``AIC`` and/or ``AIV``.
        core_ids: Optional core-local worker IDs.
        io: Task operator ports to capture: ``input``, ``output``, or both.
        device_memory_budget_mb: Maximum Device dump-buffer memory per Kernel
            invocation when using Task dump.
        max_bytes_per_tensor: Optional per-record byte cap. A capped Tensor is
            exported as a flattened prefix and marked as truncated.
    """

    output_dir: str | Path
    level: TensorDumpLevel | str = TensorDumpLevel.KERNEL
    tensor_names: str | Collection[str] | None = None
    roles: str | Collection[str] | None = None
    directions: str | Collection[str] | None = None
    ranks: int | Collection[int] | None = None
    kernel_names: str | Collection[str] | None = None
    stages: str | Collection[str] | None = None
    task_ids: int | Collection[int] | None = None
    stage_task_indices: int | Collection[int] | None = None
    owner_ids: int | Collection[int] | None = None
    core_types: str | Collection[str] | None = None
    core_ids: int | Collection[int] | None = None
    io: str | Collection[str] | None = None
    device_memory_budget_mb: int = 512
    max_bytes_per_tensor: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize user-facing collection filters."""
        if not isinstance(self.output_dir, (str, Path)):
            raise TypeError(f"output_dir must be str or Path, got {type(self.output_dir).__name__}")
        try:
            level = TensorDumpLevel(self.level.lower() if isinstance(self.level, str) else self.level)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported Tensor dump level {self.level!r}; expected 'kernel' or 'task'"
            ) from error
        output_dir = Path(self.output_dir).expanduser()
        kernel_names = _normalize_string_filter(self.kernel_names, name="kernel_names")
        tensor_names = _normalize_string_filter(self.tensor_names, name="tensor_names")
        roles = _normalize_string_filter(self.roles, name="roles", allowed=_VALID_ROLES)
        directions = _normalize_string_filter(
            self.directions,
            name="directions",
            allowed=_VALID_DIRECTIONS,
        )
        stages = _normalize_string_filter(self.stages, name="stages")
        task_ids = _normalize_non_negative_int_filter(self.task_ids, name="task_ids")
        stage_task_indices = _normalize_non_negative_int_filter(
            self.stage_task_indices,
            name="stage_task_indices",
        )
        owner_ids = _normalize_non_negative_int_filter(self.owner_ids, name="owner_ids")
        core_types = _normalize_string_filter(
            self.core_types,
            name="core_types",
            allowed=_VALID_CORE_TYPES,
        )
        core_ids = _normalize_non_negative_int_filter(self.core_ids, name="core_ids")
        io = _normalize_string_filter(self.io, name="io", allowed=_VALID_IO)
        if not isinstance(self.device_memory_budget_mb, int) or isinstance(
            self.device_memory_budget_mb,
            bool,
        ):
            raise TypeError("device_memory_budget_mb must be a positive integer")
        if self.device_memory_budget_mb <= 0:
            raise ValueError("device_memory_budget_mb must be positive")
        if self.max_bytes_per_tensor is not None:
            if not isinstance(self.max_bytes_per_tensor, int) or isinstance(
                self.max_bytes_per_tensor,
                bool,
            ):
                raise TypeError("max_bytes_per_tensor must be a positive integer or None")
            if self.max_bytes_per_tensor <= 0:
                raise ValueError("max_bytes_per_tensor must be positive")
        if level is TensorDumpLevel.TASK and all(
            value is None
            for value in (stages, task_ids, stage_task_indices, owner_ids, tensor_names)
        ):
            raise ValueError("Task Tensor dump requires at least one stage, task, owner, or Tensor filter")

        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "kernel_names", kernel_names)
        object.__setattr__(self, "tensor_names", tensor_names)
        object.__setattr__(self, "roles", roles or tuple(sorted(_VALID_ROLES)))
        object.__setattr__(self, "directions", directions or tuple(sorted(_VALID_DIRECTIONS)))
        object.__setattr__(self, "ranks", _normalize_ranks(self.ranks))
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "stage_task_indices", stage_task_indices)
        object.__setattr__(self, "owner_ids", owner_ids)
        object.__setattr__(self, "core_types", core_types)
        object.__setattr__(self, "core_ids", core_ids)
        object.__setattr__(self, "io", io or tuple(sorted(_VALID_IO)))

    def selects_invocation(self, direction: str, rank: int, kernel_name: str | None = None) -> bool:
        """Return whether one invocation passes kernel, direction and rank filters.

        Args:
            direction: Forward or backward invocation direction.
            rank: Job-global distributed rank used for selection and export.
            kernel_name: Stable diagnostic name of the invoked kernel.
        """
        if self.kernel_names is not None and kernel_name not in self.kernel_names:
            return False
        if direction not in self.directions:
            return False
        return self.ranks is None or rank in self.ranks

    def selects_tensor(self, tensor_name: str, role: str) -> bool:
        """Return whether one logical Tensor passes Kernel name and role filters.

        Args:
            tensor_name: Logical tensor name recorded by the graph.
            role: Graph-derived external input, stage output or final output role.
        """
        if role not in self.roles:
            return False
        return self.tensor_names is None or tensor_name in self.tensor_names


@dataclass
class _DumpInvocation:
    """A completed invocation retained until its dump window is finalized."""

    invocation_id: int
    step: int
    kernel_name: str
    direction: str
    rank: int
    device: str
    level: TensorDumpLevel
    snapshots: list[_KernelDumpTensorSnapshot]
    task_snapshots: list[_TaskDumpTensorSnapshot]
    device_buffer_bytes: int = 0


class _TensorDumpCall(DiagnosticCall):
    """Capture handle surrounding one native MegaKernel launch."""

    def __init__(
        self,
        on_complete: Callable[[_DumpInvocation], None] | None = None,
        invocation: _DumpInvocation | None = None,
        graph_tensors: tuple[_GraphTensor, ...] = (),
        parameters: Sequence[Any] = (),
        prepared_task_dump: _PreparedTaskDump | None = None,
        *,
        arguments: LaunchArguments,
    ) -> None:
        """Store complete launch arguments and Kernel or Task capture state."""
        super().__init__(arguments)
        self._on_complete = on_complete
        self._invocation = invocation
        self._graph_tensors = graph_tensors
        self._parameters = parameters
        self._prepared_task = prepared_task_dump
        self.finished = False

    def complete(self) -> None:
        """Capture post-launch values after a successful native launch."""
        if self.finished:
            return
        if self._on_complete is not None and self._invocation is not None:
            if self._prepared_task is not None:
                self._invocation.task_snapshots.extend(_decode_task_dump(self._prepared_task))
            else:
                for graph_tensor in self._graph_tensors:
                    if graph_tensor.role != "external_input":
                        self._invocation.snapshots.append(
                            _snapshot_tensor(self._parameters[graph_tensor.param_position], graph_tensor, "after")
                        )
            self._on_complete(self._invocation)
        self.finished = True

    def cancel(self) -> None:
        """Discard retained snapshots when the native launch fails."""
        if self.finished:
            return
        if self._invocation is not None:
            self._invocation.snapshots.clear()
            self._invocation.task_snapshots.clear()
        self._prepared_task = None
        self.finished = True


class TensorDumpSession:
    """Standalone schedule-driven Kernel/Task Tensor dumper for Torch MegaKernels."""

    def __init__(
        self,
        config: TensorDumpConfig,
        *,
        schedule: Callable[[int], DiagnosticAction] | None,
        on_dump_ready: Callable[[Any], None] | None,
    ) -> None:
        # pylint: disable=redefined-outer-name
        """Validate options and initialize dump window state."""
        if not isinstance(config, TensorDumpConfig):
            raise TypeError(f"config must be TensorDumpConfig, got {type(config).__name__}")
        if schedule is not None and not callable(schedule):
            raise TypeError("schedule must be callable or None")
        if on_dump_ready is not None and not callable(on_dump_ready):
            raise TypeError("on_dump_ready must be callable or None")
        self.config = config
        self._schedule = schedule
        self._on_dump_ready = on_dump_ready
        self._step = 0
        self._action = self._action_for_step(0)
        self._entered = False
        self._next_invocation_id = 0
        self._window_index = 0
        self._invocations: list[_DumpInvocation] = []
        self._last_manifest_paths: tuple[Path, ...] = ()

    @property
    def step_num(self) -> int:
        """Return the zero-based user step currently selected."""
        return self._step

    @property
    def current_action(self) -> DiagnosticAction:
        """Return the current shared diagnostic schedule action."""
        return self._action

    @property
    def last_manifest_paths(self) -> tuple[Path, ...]:
        """Return manifest paths from the most recently finalized window."""
        return self._last_manifest_paths

    def __enter__(self) -> TensorDumpSession:
        """Activate this dumper as the process-local diagnostic session."""
        _activate_session("tensor_dump", self)
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Finalize successful snapshots or discard them after an exception."""
        try:
            if exc_type is None and self._invocations:
                self._finalize_window()
            elif exc_type is not None:
                self._invocations.clear()
        finally:
            _deactivate_session(self)
            self._entered = False

    def step(self) -> None:
        """Finish the current user step and advance the dump schedule."""
        if not self._entered:
            raise RuntimeError("dumper.step() must be called inside its context")
        next_action = self._action_for_step(self._step + 1)
        if self._action is DiagnosticAction.RECORD_AND_SAVE:
            self._finalize_window()
        elif self._action is DiagnosticAction.RECORD and next_action not in (
            DiagnosticAction.RECORD,
            DiagnosticAction.RECORD_AND_SAVE,
        ):
            self._finalize_window()
        self._step += 1
        self._action = next_action

    def _action_for_step(self, step: int) -> DiagnosticAction:
        if self._schedule is None:
            return DiagnosticAction.RECORD
        action = self._schedule(step)
        if not isinstance(action, DiagnosticAction):
            raise TypeError(f"MegaKernel schedule must return DiagnosticAction, got {type(action).__name__}")
        return action

    def should_prepare_call(self) -> bool:
        """Return whether this step prepares Tensor collection; warmup is excluded."""
        return self._action in (DiagnosticAction.RECORD, DiagnosticAction.RECORD_AND_SAVE)

    def prepare_call(self, context: LaunchContext) -> DiagnosticCall:
        """Prepare only this dumper's selected invocation, with complete arguments.

        Args:
            context: Selected invocation or operand context supplied by the diagnostic framework.
        """
        runtime = context.runtime
        if not self.should_prepare_call() or not self.config.selects_invocation(
            context.direction, runtime.rank, runtime.kernel_name,
        ):
            return DiagnosticCall(context.ordinary_arguments())
        # Graph/owner work is lazy and never part of ordinary MegaKernel execution.
        _ = runtime.metadata
        _stage_names(runtime.graph.topological_sort())
        return self._acquire_call(
            runtime.graph, context.parameters, direction=context.direction, rank=runtime.rank,
            kernel_name=runtime.kernel_name, device=runtime.runtime_config_tensor.device,
            runtime=runtime, event_counters=context.event_counters,
        )

    def _acquire_call(
        self,
        graph: Any,
        parameters: Sequence[Any],
        *,
        direction: str,
        rank: int,
        kernel_name: str,
        device: Any,
        runtime: Any,
        event_counters: Any,
    ) -> DiagnosticCall:
        """Plan a selected invocation and retain independent input snapshots before launch."""
        arguments = LaunchArguments(
            runtime.runtime_config_tensor, event_counters,
            runtime.runtime_config_tensor if event_counters is None else event_counters, event_counters,
        )
        if self.config.level is TensorDumpLevel.TASK:
            prepared_task_dump = _prepare_task_dump(
                runtime,
                graph,
                parameters,
                self.config,
                rank=rank,
                device=device,
            )
            invocation = _DumpInvocation(
                invocation_id=self._next_invocation_id,
                step=self._step,
                direction=direction,
                kernel_name=kernel_name,
                rank=rank,
                device=str(device),
                level=self.config.level,
                snapshots=[],
                task_snapshots=[],
                device_buffer_bytes=prepared_task_dump.total_bytes,
            )
            self._next_invocation_id += 1
            arguments.runtime_config_tensor = prepared_task_dump.runtime_config_tensor
            arguments.diagnostic_buffer = prepared_task_dump.buffer
            return _TensorDumpCall(
                self._invocations.append, invocation, prepared_task_dump=prepared_task_dump, arguments=arguments,
            )

        graph_tensors = tuple(
            tensor
            for tensor in _describe_graph_tensors(graph)
            if self.config.selects_tensor(tensor.name, tensor.role)
        )
        if not graph_tensors:
            return DiagnosticCall(arguments)
        max_position = max(tensor.param_position for tensor in graph_tensors)
        if max_position >= len(parameters):
            raise ValueError(
                f"MegaKernel dump requires param_position {max_position}, but only {len(parameters)} parameters exist"
            )
        invocation = _DumpInvocation(
            invocation_id=self._next_invocation_id,
            step=self._step,
            direction=direction,
            kernel_name=kernel_name,
            rank=rank,
            device=str(device),
            level=self.config.level,
            snapshots=[],
            task_snapshots=[],
        )
        self._next_invocation_id += 1
        for graph_tensor in graph_tensors:
            if graph_tensor.role == "external_input":
                invocation.snapshots.append(
                    _snapshot_tensor(parameters[graph_tensor.param_position], graph_tensor, "before")
                )
        return _TensorDumpCall(self._invocations.append, invocation, graph_tensors, parameters, arguments=arguments)

    def _finalize_window(self) -> None:
        """Export completed invocations by rank and release the window's CPU snapshots."""
        if not self._invocations:
            self._last_manifest_paths = ()
            self._window_index += 1
            return
        manifest_paths = []
        ranks = sorted({invocation.rank for invocation in self._invocations})
        for rank in ranks:
            rank_invocations = [invocation for invocation in self._invocations if invocation.rank == rank]
            manifest_paths.append(self._write_rank_window(rank, rank_invocations))
        self._last_manifest_paths = tuple(manifest_paths)
        self._invocations.clear()
        self._window_index += 1
        if self._on_dump_ready is not None:
            self._on_dump_ready(self)

    def _write_kernel_invocation(
        self,
        invocation: _DumpInvocation,
        invocation_dir: Path,
        window_dir: Path,
    ) -> dict[str, Any]:
        """Write logical tensor snapshots and their stage-port metadata."""
        tensor_metadata = []
        for snapshot in invocation.snapshots:
            safe_name = _SAFE_FILE_NAME.sub("_", snapshot.graph_tensor.name)
            output_path, port_metadata = _write_kernel_stage_files(
                snapshot,
                invocation_dir,
                window_dir,
                safe_name,
            )
            tensor_metadata.append(
                {
                    "name": snapshot.graph_tensor.name,
                    "paramPosition": snapshot.graph_tensor.param_position,
                    "role": snapshot.graph_tensor.role,
                    "capturePhase": snapshot.phase,
                    "producers": list(snapshot.graph_tensor.producers),
                    "consumers": list(snapshot.graph_tensor.consumers),
                    "ports": port_metadata,
                    "declaredShape": list(snapshot.graph_tensor.declared_shape),
                    "actualShape": list(snapshot.shape),
                    "originalDtype": snapshot.original_dtype,
                    "storedDtype": snapshot.stored_dtype,
                    "file": output_path.relative_to(window_dir).as_posix(),
                }
            )
        return {
            "invocationId": invocation.invocation_id,
            "step": invocation.step,
            "direction": invocation.direction,
            "kernelName": invocation.kernel_name,
            "device": invocation.device,
            "tensorCount": len(tensor_metadata),
            "tensors": tensor_metadata,
        }

    def _write_task_invocation(
        self,
        invocation: _DumpInvocation,
        invocation_dir: Path,
        window_dir: Path,
    ) -> dict[str, Any]:
        """Write task-local tensor snapshots with their native copy metadata."""
        snapshots_by_task: dict[tuple[int, str, int], list[_TaskDumpTensorSnapshot]] = {}
        for snapshot in invocation.task_snapshots:
            record = snapshot.record
            snapshots_by_task.setdefault((record.task_id, record.core_type, record.core_id), []).append(snapshot)

        task_metadata = []
        for key in sorted(snapshots_by_task):
            task_id = key[0]
            snapshots = snapshots_by_task[key]
            first = snapshots[0].record
            safe_stage = _SAFE_FILE_NAME.sub("_", first.stage)
            task_dir = invocation_dir / "stages" / safe_stage / f"task_{task_id:06d}"
            if first.worker_id != 0xFFFFFFFF:
                task_dir = task_dir.with_name(f"{task_dir.name}_worker_{first.core_id:03d}")
            task_dir.mkdir(parents=True, exist_ok=True)
            tensor_metadata = []
            for snapshot in snapshots:
                record = snapshot.record
                safe_name = _SAFE_FILE_NAME.sub("_", record.tensor_name)
                file_name = f"{record.io}_{record.index:02d}_{safe_name}.pt"
                output_path = task_dir / file_name
                torch.save(snapshot.data, output_path)
                tensor_metadata.append(
                    {
                        "name": record.tensor_name,
                        "paramPosition": record.parameter_position,
                        "io": record.io,
                        "index": record.index,
                        "capturePhase": record.phase,
                        "sourceOffsetBytes": record.source_offset_bytes,
                        "actualShape": list(record.actual_shape),
                        "storedShape": list(snapshot.stored_shape),
                        "originalDtype": snapshot.original_dtype,
                        "storedDtype": snapshot.stored_dtype,
                        "fullBytes": record.full_bytes,
                        "storedBytes": record.payload_bytes,
                        "truncated": record.truncated,
                        "note": record.note,
                        "file": output_path.relative_to(window_dir).as_posix(),
                    }
                )
            metadata_path = task_dir / "task.json"
            metadata = {
                "taskId": task_id,
                "stageTaskIndex": first.task_index,
                "stage": first.stage,
                "owner": {"label": first.owner_label, "id": first.owner_id},
                "coreType": first.core_type,
                "coreId": first.core_id,
                "tensorCount": len(tensor_metadata),
                "tensors": tensor_metadata,
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            metadata["metadataFile"] = metadata_path.relative_to(window_dir).as_posix()
            task_metadata.append(metadata)
        return {
            "invocationId": invocation.invocation_id,
            "step": invocation.step,
            "direction": invocation.direction,
            "kernelName": invocation.kernel_name,
            "device": invocation.device,
            "deviceBufferBytes": invocation.device_buffer_bytes,
            "taskCount": len(task_metadata),
            "tasks": task_metadata,
        }

    def _write_rank_window(self, rank: int, invocations: list[_DumpInvocation]) -> Path:
        """Export one rank's window without overwriting an existing directory."""
        window_dir = self.config.output_dir / f"rank_{rank}" / f"window_{self._window_index:04d}"
        if window_dir.exists():
            raise FileExistsError(f"Tensor dump window already exists: {window_dir}")
        window_dir.mkdir(parents=True)
        invocation_metadata = []
        for invocation in invocations:
            invocation_dir = window_dir / f"invocation_{invocation.invocation_id:06d}_{invocation.direction}"
            if invocation.level is TensorDumpLevel.TASK:
                metadata = self._write_task_invocation(invocation, invocation_dir, window_dir)
            else:
                metadata = self._write_kernel_invocation(invocation, invocation_dir, window_dir)
            invocation_metadata.append(metadata)
        is_task_dump = self.config.level is TensorDumpLevel.TASK
        manifest = {
            "schemaVersion": 5,
            "level": self.config.level.value,
            "tensorSerialization": "torch.save",
            "tensorFileLayout": "stage_only",
            "windowIndex": self._window_index,
            "capturePhaseSemantics": "task_boundary" if is_task_dump else "mega_kernel_boundary",
            "stageViewStorage": "per_task_files" if is_task_dump else "primary_file_with_hard_links",
            "rank": rank,
            "invocationCount": len(invocation_metadata),
            "invocations": invocation_metadata,
        }
        manifest_path = window_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path


def tensor_dump(
    config: TensorDumpConfig,
    *,
    schedule: Callable[[int], DiagnosticAction] | None = None,
    on_dump_ready: Callable[[Any], None] | None = None,
) -> TensorDumpSession:
    # pylint: disable=redefined-outer-name
    """Create the standalone tensor MegaKernel Tensor dumper.

    Args:
        config: Output and logical Tensor selection configuration.
        schedule: Optional shared diagnostic step schedule. Warmup actions do
            not copy Tensor values because Kernel collection needs no warmup.
        on_dump_ready: Optional callback invoked after a window is exported.

    Returns:
        tensor MegaKernel Tensor dump context manager.

    Note:
        Tensor dumping synchronizes Device-to-Host copies and changes execution
        timing. It is mutually exclusive with MegaKernel cycle profiling.
    """
    return TensorDumpSession(
        config,
        schedule=schedule,
        on_dump_ready=on_dump_ready,
    )
