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

from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from enum import Enum
import json
import logging
from os import rename
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from threading import Lock
from typing import Any

import torch

from hyper_multicore.diagnostics.common.session import (
    _activate_session,
    _deactivate_session,
)
from hyper_multicore.diagnostics.common.call import DiagnosticCall, KernelDumpRowGroups, LaunchArguments, LaunchContext
from hyper_multicore.diagnostics.common.schedule import DiagnosticAction
from hyper_multicore.diagnostics.dumper._layout import (
    _KERNEL_MANIFEST_VERSION, _TASK_MANIFEST_VERSION,
)
from hyper_multicore.diagnostics.dumper._kernel import (
    _GraphTensor, _KernelDumpTensorSnapshot, _describe_graph_tensors, _snapshot_tensor, _write_kernel_snapshots,
)
from hyper_multicore.diagnostics.dumper._summary import (
    _LOG_ROWS, _print_summary, _resolve_expert_rows, _summarize_invocation, _write_summary_csv,
)
from hyper_multicore.diagnostics.dumper._task import (
    _TaskDumpCapacityExceeded,
    _TaskDumpTensorSnapshot,
    _PreparedTaskDump,
    _decode_task_dump,
    _prepare_task_dump,
    _write_task_snapshots,
    UnsupportedTaskDumpPort,
)


__all__ = [
    "DumpLevel",
    "DumpConfig",
    "DumperSession",
    "dump",
]

_LOGGER = logging.getLogger(__name__)


_VALID_DIRECTIONS = frozenset(("forward", "backward"))
_VALID_ROLES = frozenset(("external_input", "stage_output", "final_output"))
_VALID_IO = frozenset(("input", "output"))
_VALID_CORE_TYPES = frozenset(("AIC", "AIV"))
_VALID_OUTPUTS = frozenset(("tensor", "csv", "log"))
_DEFAULT_DEVICE_MEMORY_BUDGET_MB = 512


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
    values = (value,) if isinstance(value, int) else tuple(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    for item in values:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{name} entries must be non-negative integers, got {item!r}")
    return values


class DumpLevel(str, Enum):
    """Execution scope at whose before/after points Tensor values are dumped."""

    KERNEL = "kernel"
    TASK = "task"


@dataclass(frozen=True)
class DumpConfig:
    """Configuration for MegaKernel Tensor value collection.

    Args:
        output_dir: Root directory for per-rank dump windows.
        level: Kernel or Task dump level. Accepts DumpLevel or a
            case-insensitive ``"kernel"`` / ``"task"`` string. Defaults to Kernel.
        kernel_names: Optional exact MegaKernel names whose invocations should
            be retained.
        tensor_names: Optional exact logical Tensor names selected from the
            MegaKernel compute graph.
        roles: Optional Tensor roles for Kernel dump. Supported roles are
            ``external_input``, ``stage_output`` and ``final_output``.
        directions: Forward and/or backward invocations to retain.
        ranks: Optional non-negative distributed ranks to retain.
        stages: Optional exact stage names published by the diagnostic graph.
        task_ids: Optional task IDs in this invocation's diagnostic RuntimeConfig.
            Diagnostic adaptation may renumber them relative to standard execution.
        stage_task_indices: Optional task indices local to a stage.
        owner_ids: Optional business diagnostic owner IDs.
        core_types: Optional Task Device worker types, ``AIC`` and/or ``AIV``.
        core_ids: Optional core-local worker IDs.
        io: Stage ports to capture: ``input``, ``output``, or both by default.
        device_memory_budget_mb: Maximum Device dump-buffer memory per Kernel
            invocation when using Task dump. Defaults to 512 MiB.
        max_bytes_per_tensor: Optional Task dump per-record byte cap. A capped Tensor is
            exported as a flattened prefix and marked as truncated.
        outputs: Artifacts to generate. The default is the Tensor files and manifest.
            Select csv or log to summarize snapshots as the job runs.
        tags: Training identifiers shared by the calls captured in this session.
    """

    output_dir: str | Path
    level: DumpLevel | str = DumpLevel.KERNEL
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
    device_memory_budget_mb: int = _DEFAULT_DEVICE_MEMORY_BUDGET_MB
    max_bytes_per_tensor: int | None = None
    outputs: str | Collection[str] = ("tensor",)
    tags: Mapping[str, str | int] | None = None

    def __post_init__(self) -> None:
        """Validate and normalize user-facing collection filters."""
        if not isinstance(self.output_dir, (str, Path)):
            raise TypeError(f"output_dir must be str or Path, got {type(self.output_dir).__name__}")
        try:
            level = DumpLevel(self.level.lower() if isinstance(self.level, str) else self.level)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported Tensor dump level {self.level!r}; expected 'kernel' or 'task'"
            ) from error
        self._validate_level_options(level)
        outputs, tags = self._normalize_output_options()
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
        if level is DumpLevel.TASK and all(
            value is None
            for value in (stages, task_ids, stage_task_indices, owner_ids, tensor_names)
        ):
            raise ValueError("Task Tensor dump requires at least one stage, task, owner, or Tensor filter")

        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "kernel_names", kernel_names)
        object.__setattr__(self, "tensor_names", tensor_names)
        object.__setattr__(self, "roles", (roles or tuple(sorted(_VALID_ROLES))) if level is DumpLevel.KERNEL else None)
        object.__setattr__(self, "directions", directions or tuple(sorted(_VALID_DIRECTIONS)))
        object.__setattr__(self, "ranks", _normalize_non_negative_int_filter(self.ranks, name="ranks"))
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "stage_task_indices", stage_task_indices)
        object.__setattr__(self, "owner_ids", owner_ids)
        object.__setattr__(self, "core_types", core_types)
        object.__setattr__(self, "core_ids", core_ids)
        object.__setattr__(self, "io", (io or tuple(sorted(_VALID_IO))) if level is DumpLevel.TASK else io)

    def _validate_level_options(self, level: DumpLevel) -> None:
        """Check the applicability and values of level-specific options."""
        budget_mb = self.device_memory_budget_mb
        if not isinstance(budget_mb, int) or isinstance(budget_mb, bool):
            raise TypeError("device_memory_budget_mb must be a positive integer")
        if budget_mb <= 0:
            raise ValueError("device_memory_budget_mb must be positive")

        max_bytes = self.max_bytes_per_tensor
        if level is DumpLevel.KERNEL:
            for name in (
                "task_ids", "stage_task_indices", "owner_ids", "core_types", "core_ids",
                "max_bytes_per_tensor",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} is only supported for Task dump")
            if budget_mb != _DEFAULT_DEVICE_MEMORY_BUDGET_MB:
                raise ValueError("device_memory_budget_mb is only configurable for Task dump")
        elif self.roles is not None:
            raise ValueError("roles is only supported for Kernel dump")
        if max_bytes is not None:
            if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
                raise TypeError("max_bytes_per_tensor must be a positive integer or None")
            if max_bytes <= 0:
                raise ValueError("max_bytes_per_tensor must be positive")

    def _normalize_output_options(self) -> tuple[tuple[str, ...], Mapping[str, str | int]]:
        """Normalize artifact selections and copy the supplied job tags."""
        outputs = _normalize_string_filter(self.outputs, name="outputs", allowed=_VALID_OUTPUTS)
        if outputs is None:
            raise TypeError("outputs must name at least one artifact")
        if self.tags is not None and not isinstance(self.tags, Mapping):
            raise TypeError("tags must be a mapping")
        tags = dict(self.tags or {})
        for key, value in tags.items():
            if not isinstance(key, str) or not key or type(value) not in (str, int):
                raise TypeError("tags must map non-empty strings to strings or integers")
        return tuple(dict.fromkeys(outputs)), tags

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
        if self.roles is None or role not in self.roles:
            return False
        return self.tensor_names is None or tensor_name in self.tensor_names


@dataclass
class _DumpInvocation:
    """The snapshots and metadata owned by a single call."""

    invocation_id: int
    window_token: object
    step: int
    kernel_name: str
    direction: str
    rank: int
    device: str
    level: DumpLevel
    tags: dict[str, str | int] = field(default_factory=dict)
    row_groups: KernelDumpRowGroups | None = None
    kernel_snapshots: list[_KernelDumpTensorSnapshot] = field(default_factory=list)
    task_snapshots: list[_TaskDumpTensorSnapshot] = field(default_factory=list)
    device_buffer_bytes: int = 0


@dataclass
class _DumpWindow:
    """Staged files and manifest entries for one rank's current window."""

    staging_dir: Path | None = None
    invocation_metadata: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    preview_total: int = 0
    preview_lock: Lock = field(default_factory=Lock)


class _DumpCall(DiagnosticCall):
    """Capture handle surrounding one native MegaKernel launch."""

    def __init__(
        self,
        session: DumperSession,
        invocation: _DumpInvocation,
        graph_tensors: tuple[_GraphTensor, ...] = (),
        parameters: Sequence[Any] = (),
        prepared_task_dump: _PreparedTaskDump | None = None,
        *,
        ordinary_arguments: LaunchArguments,
    ) -> None:
        """Store complete launch arguments and Kernel or Task capture state."""
        arguments = ordinary_arguments
        if prepared_task_dump is not None:
            arguments = replace(
                ordinary_arguments,
                runtime_config_tensor=prepared_task_dump.runtime_config_tensor,
                diagnostic_buffer=prepared_task_dump.buffer,
            )
        super().__init__(arguments)
        self._ordinary_arguments = ordinary_arguments
        self._session = session
        self._invocation = invocation
        self._graph_tensors = graph_tensors
        self._parameters = parameters
        self._prepared_task = prepared_task_dump

    def complete(self) -> None:
        """Capture and write this call's snapshots after a successful native launch."""
        if self.finished:
            return
        invocation = self._invocation
        try:
            if self._prepared_task is not None:
                invocation.task_snapshots.extend(_decode_task_dump(self._prepared_task))
            else:
                for graph_tensor in self._graph_tensors:
                    if graph_tensor.role != "external_input":
                        invocation.kernel_snapshots.append(
                            _snapshot_tensor(self._parameters[graph_tensor.param_position], graph_tensor, "after")
                        )
            self._session._write_invocation(invocation)
        except (MemoryError, torch.OutOfMemoryError) as error:
            self._session._report_dump_skip(invocation, "allocation_oom", str(error))
        finally:
            self.cancel()

    def cancel(self) -> None:
        """Release resources owned by the capture."""
        if self.finished:
            return
        self._invocation.kernel_snapshots.clear()
        self._invocation.task_snapshots.clear()
        self._invocation.row_groups = None
        self._parameters = ()
        self._prepared_task = None
        self.arguments = self._ordinary_arguments
        self.finished = True


def _remove_temporary_directory(directory: Path) -> None:
    """Clean up unpublished files while preserving the original failure."""
    try:
        rmtree(directory)
    except FileNotFoundError:
        pass
    except OSError as error:
        _LOGGER.warning("Cannot remove temporary dump directory %s: %s", directory, error)


class DumperSession:
    """Standalone schedule-driven Kernel/Task Tensor dumper for Torch MegaKernels."""

    def __init__(
        self,
        config: DumpConfig,
        *,
        schedule: Callable[[int], DiagnosticAction] | None,
        on_dump_ready: Callable[[Any], None] | None,
    ) -> None:
        # pylint: disable=redefined-outer-name
        """Validate options and initialize dump window state."""
        if not isinstance(config, DumpConfig):
            raise TypeError(f"config must be DumpConfig, got {type(config).__name__}")
        if schedule is not None and not callable(schedule):
            raise TypeError("schedule must be callable or None")
        if on_dump_ready is not None and not callable(on_dump_ready):
            raise TypeError("on_dump_ready must be callable or None")
        self.config = config
        self._output_dir: Path
        self._schedule = schedule
        self._on_dump_ready = on_dump_ready
        self._step = 0
        self._action = self._action_for_step(0)
        self._window_token: object | None = None
        self._next_invocation_id = 0
        self._window_index = 0
        self._window_lock = Lock()
        self._rank_windows: dict[int, _DumpWindow] = {}
        self._window_has_skips = False
        self._last_manifest_paths: tuple[Path, ...] = ()
        self._reported_skips: set[tuple[int, str]] = set()

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
        """Return manifest paths from the most recently finalized window, or () for an empty window."""
        return self._last_manifest_paths

    def __enter__(self) -> DumperSession:
        """Activate this dumper as the process-local diagnostic session."""
        with self._window_lock:
            output_dir = self.config.output_dir.absolute()
            _activate_session("dumper", self)
            self._output_dir = output_dir
            self._window_token = object()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Publish completed windows or discard their temporary files on error."""
        try:
            with self._window_lock:
                self._window_token = None
                if exc_type is not None:
                    self._discard_windows()
                    return
            if self._rank_windows or self._window_has_skips:
                self._finalize_window()
        finally:
            _deactivate_session(self)

    def step(self) -> None:
        """Finish the current user step and advance the dump schedule."""
        if self._window_token is None:
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

    def _selects_invocation(self, kernel_name: str, direction: str, rank: int) -> bool:
        """Combine the current schedule action with the call filters."""
        return self.should_prepare_call() and self.config.selects_invocation(direction, rank, kernel_name)

    def prepare_call(self, context: LaunchContext) -> DiagnosticCall:
        """Prepare only this dumper's selected invocation, with complete arguments.

        Args:
            context: Selected invocation or operand context supplied by the diagnostic framework.
        """
        runtime = context.runtime
        if not self._selects_invocation(runtime.kernel_name, context.direction, runtime.rank):
            return DiagnosticCall(context.ordinary_arguments())
        is_kernel_dump = self.config.level is DumpLevel.KERNEL
        try:
            # Resolve and validate the adapted graph before assigning a capture ID.
            _ = runtime.metadata
            graph_tensors = tuple(
                tensor for tensor in _describe_graph_tensors(
                    runtime.graph, stages=self.config.stages, io=self.config.io,
                )
                if self.config.selects_tensor(tensor.name, tensor.role)
            ) if is_kernel_dump else ()
        except (MemoryError, torch.OutOfMemoryError) as error:
            self._record_dump_skip(
                "allocation_oom", str(error), kernel_name=runtime.kernel_name,
                direction=context.direction, rank=runtime.rank,
            )
            return DiagnosticCall(context.ordinary_arguments())
        if is_kernel_dump and not graph_tensors:
            self._record_dump_skip(
                "empty_selection", "no Tensor snapshots to export for this call",
                kernel_name=runtime.kernel_name, direction=context.direction, rank=runtime.rank,
            )
            return DiagnosticCall(context.ordinary_arguments())
        return self._acquire_call(context, graph_tensors)

    def _report_unsupported_kernel(self, *, kernel_name: str, direction: str, rank: int) -> None:
        """Report a selected call whose native kernel does not support Tensor dumps."""
        if self._selects_invocation(kernel_name, direction, rank):
            self._record_dump_skip(
                "unsupported_kernel", "this kernel does not support Tensor dumps",
                kernel_name=kernel_name, direction=direction, rank=rank,
            )

    def _report_dump_skip(self, invocation: _DumpInvocation, reason: str, detail: str) -> None:
        """Report a skip after an invocation has been assigned a capture ID."""
        self._record_dump_skip(
            reason, detail, kernel_name=invocation.kernel_name, direction=invocation.direction,
            rank=invocation.rank, invocation=invocation,
        )

    def _record_dump_skip(
        self, reason: str, detail: str, *, kernel_name: str, direction: str, rank: int,
        invocation: _DumpInvocation | None = None,
    ) -> None:
        """Report each skip reason once per rank in this session."""
        key = (rank, reason)
        with self._window_lock:
            if self._window_token is None or (
                invocation is not None and invocation.window_token is not self._window_token
            ):
                return
            step = self._step if invocation is None else invocation.step
            self._window_has_skips = True
            if key in self._reported_skips:
                return
            self._reported_skips.add(key)
        _LOGGER.warning(
            "%s dump skipped for kernel=%s rank=%d direction=%s step=%d reason=%s%s: %s",
            self.config.level.value.capitalize(), kernel_name, rank, direction, step, reason,
            "" if invocation is None else f" invocation={invocation.invocation_id}", detail,
        )

    def _acquire_call(self, context: LaunchContext, graph_tensors: tuple[_GraphTensor, ...]) -> DiagnosticCall:
        """Allocate resources for the selected invocation and retain inputs before launch."""
        runtime = context.runtime
        parameters = context.parameters
        arguments = context.ordinary_arguments()
        device = runtime.runtime_config_tensor.device
        with self._window_lock:
            if self._window_token is None:
                return DiagnosticCall(arguments)
            invocation = _DumpInvocation(
                invocation_id=self._next_invocation_id,
                window_token=self._window_token,
                step=self._step,
                kernel_name=runtime.kernel_name,
                direction=context.direction,
                rank=runtime.rank,
                device=str(device),
                level=self.config.level,
                tags={**self.config.tags, **(context.tags or {})},
                row_groups=context.dump_row_groups,
            )
            self._next_invocation_id += 1
        if invocation.level is DumpLevel.TASK:
            try:
                owner_ids = context.resolve_owner_ids() if context.resolve_owner_ids is not None else None
                prepared_task_dump = _prepare_task_dump(runtime, parameters, self.config, owner_ids=owner_ids)
            except _TaskDumpCapacityExceeded as error:
                self._report_dump_skip(invocation, error.reason, str(error))
                return DiagnosticCall(arguments)
            except UnsupportedTaskDumpPort as error:
                self._report_dump_skip(invocation, "unsupported_port", str(error))
                return DiagnosticCall(arguments)
            except (MemoryError, torch.OutOfMemoryError) as error:
                self._report_dump_skip(invocation, "allocation_oom", str(error))
                return DiagnosticCall(arguments)
            if prepared_task_dump is None:
                self._report_dump_skip(invocation, "empty_selection", "no Tensor snapshots to export for this call")
                return DiagnosticCall(arguments)
            invocation.device_buffer_bytes = prepared_task_dump.total_bytes
            return _DumpCall(
                self, invocation, prepared_task_dump=prepared_task_dump, ordinary_arguments=arguments,
            )

        try:
            for graph_tensor in graph_tensors:
                if graph_tensor.role == "external_input":
                    invocation.kernel_snapshots.append(
                        _snapshot_tensor(parameters[graph_tensor.param_position], graph_tensor, "before")
                    )
        except (MemoryError, torch.OutOfMemoryError) as error:
            self._report_dump_skip(invocation, "allocation_oom", str(error))
            return DiagnosticCall(arguments)
        return _DumpCall(self, invocation, graph_tensors, parameters, ordinary_arguments=arguments)

    def _window_directory(self, rank: int) -> Path:
        """Return the public path of a rank's current window."""
        return self._output_dir / f"rank_{rank}" / f"window_{self._window_index:04d}"

    def _stage_rank_window(self, rank: int, window: _DumpWindow) -> Path:
        """Allocate a staging directory only when a file output is requested."""
        if window.staging_dir is None:
            target = self._window_directory(rank)
            target.parent.mkdir(parents=True, exist_ok=True)
            window.staging_dir = Path(mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        return window.staging_dir

    def _write_invocation(self, invocation: _DumpInvocation) -> None:
        """Collect the requested file outputs and bounded log previews."""
        outputs = self.config.outputs
        preview_only = "log" in outputs and "csv" not in outputs
        with self._window_lock:
            if invocation.window_token is not self._window_token:
                return
            window = self._rank_windows.setdefault(invocation.rank, _DumpWindow())
            limit = _LOG_ROWS - len(window.summaries) if preview_only else None

        rows = []
        with ExitStack() as locks:
            try:
                if limit != 0 and (preview_only or "csv" in outputs):
                    expert_rows = _resolve_expert_rows(invocation, limit)
                    if preview_only:
                        # Another call may have filled the preview while the component resolved its rows.
                        locks.enter_context(window.preview_lock)
                        if invocation.window_token is not self._window_token:
                            return
                        limit = _LOG_ROWS - len(window.summaries)
                    rows = _summarize_invocation(invocation, limit, expert_rows)
            except (MemoryError, torch.OutOfMemoryError, ValueError) as error:
                _LOGGER.warning(
                    "Dump summary skipped for kernel=%s rank=%d step=%d invocation=%d: %s",
                    invocation.kernel_name, invocation.rank, invocation.step, invocation.invocation_id, error,
                )
            self._stage_invocation(invocation, window, rows)

    def _stage_invocation(
        self, invocation: _DumpInvocation, window: _DumpWindow, rows: list[dict[str, Any]],
    ) -> None:
        """Publish a completed call into the window to which it still belongs."""
        try:
            with self._window_lock:
                if invocation.window_token is not self._window_token:
                    return
                try:
                    if "tensor" in self.config.outputs:
                        self._write_tensor_invocation(invocation, window)
                finally:
                    # A failed Tensor write cannot invalidate an already computed summary.
                    window.summaries.extend(rows)
                    if "log" in self.config.outputs:
                        window.preview_total += len(invocation.kernel_snapshots) + len(invocation.task_snapshots)
        except OSError as error:
            self._report_dump_skip(invocation, "write_failed", str(error))

    def _write_tensor_invocation(self, invocation: _DumpInvocation, window: _DumpWindow) -> None:
        """Stage a completed call and register it after every Tensor file is written."""
        staging_dir = self._stage_rank_window(invocation.rank, window)
        invocation_name = f"invocation_{invocation.invocation_id:06d}_{invocation.direction}"
        # Keep incomplete writes outside the directory eligible for publication.
        scratch_dir = Path(mkdtemp(prefix=f".{invocation_name}.", dir=staging_dir.parent))
        try:
            invocation_dir = scratch_dir / invocation_name
            metadata = {
                "invocationId": invocation.invocation_id,
                "step": invocation.step,
                "direction": invocation.direction,
                "kernelName": invocation.kernel_name,
                "device": invocation.device,
                **({"tags": invocation.tags} if invocation.tags else {}),
            }
            if invocation.level is DumpLevel.TASK:
                metadata.update(_write_task_snapshots(
                    invocation.task_snapshots, invocation_dir, scratch_dir, invocation.device_buffer_bytes,
                ))
            else:
                metadata.update(_write_kernel_snapshots(invocation.kernel_snapshots, invocation_dir, scratch_dir))
            window.invocation_metadata.append(metadata)
            try:
                rename(invocation_dir, staging_dir / invocation_name)
            except (OSError, MemoryError):
                window.invocation_metadata.pop()
                raise
        finally:
            _remove_temporary_directory(scratch_dir)

    def _finalize_window(self) -> None:
        """Publish each rank's staged window and release temporary state."""
        previews = []
        with self._window_lock:
            if self._window_token is not None:
                self._window_token = object()
            manifest_paths = []
            try:
                for rank, window in sorted(self._rank_windows.items()):
                    number = self._window_index
                    manifest_written = self._stage_tensor_manifest(rank, window)
                    csv_written = self._stage_summary_csv(rank, window) if "csv" in self.config.outputs else False
                    if window.preview_total and "log" in self.config.outputs:
                        previews.append((rank, number, window.preview_total, window.summaries[:_LOG_ROWS]))
                    if not manifest_written and not csv_written:
                        continue
                    try:
                        published = self._publish_rank_window(rank, window.staging_dir)
                    except (OSError, MemoryError) as error:
                        _LOGGER.warning(
                            "Dump window publication failed for rank=%d window=%d reason=publish_failed: %s",
                            rank, self._window_index, error,
                        )
                    else:
                        if manifest_written:
                            manifest_paths.append(published / "manifest.json")
                        del self._rank_windows[rank]
            finally:
                self._discard_windows()
            self._last_manifest_paths = tuple(manifest_paths)
            self._window_index += 1
        for rank, number, count, rows in previews:
            _print_summary(rank, number, count, rows)
        if manifest_paths and self._on_dump_ready is not None:
            self._on_dump_ready(self)

    def _stage_summary_csv(self, rank: int, window: _DumpWindow) -> bool:
        """Stage a complete CSV without affecting publication of Tensor files on failure."""
        if not window.summaries:
            return False
        try:
            directory = self._stage_rank_window(rank, window)
            _write_summary_csv(directory / "summary.csv", window.summaries, self._window_index)
        except (OSError, MemoryError) as error:
            _LOGGER.warning("Dump summary write failed for rank=%d window=%d: %s", rank, self._window_index, error)
            return False
        return True

    def _discard_windows(self) -> None:
        """Clear unpublished windows while the session holds the window lock."""
        for window in self._rank_windows.values():
            if window.staging_dir is not None:
                _remove_temporary_directory(window.staging_dir)
        self._rank_windows.clear()
        self._window_has_skips = False

    def _stage_tensor_manifest(self, rank: int, window: _DumpWindow) -> bool:
        """Keep the raw snapshots only if their manifest can also be written."""
        if not window.invocation_metadata:
            return False
        try:
            invocations = sorted(window.invocation_metadata, key=lambda item: item["invocationId"])
            is_task_dump = self.config.level is DumpLevel.TASK
            manifest = {
                "schemaVersion": _TASK_MANIFEST_VERSION if is_task_dump else _KERNEL_MANIFEST_VERSION,
                "level": self.config.level.value,
                "tensorSerialization": "torch.save",
                "tensorFileLayout": "stage_only",
                "windowIndex": self._window_index,
                "capturePhaseSemantics": "task_boundary" if is_task_dump else "mega_kernel_boundary",
                "stageViewStorage": "per_task_files" if is_task_dump else "primary_file_with_hard_links",
                "rank": rank,
                "invocationCount": len(invocations),
                "invocations": invocations,
            }
            path = window.staging_dir / "manifest.json"
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except (OSError, MemoryError) as error:
            _LOGGER.warning(
                "Dump manifest write failed for rank=%d window=%d: %s", rank, self._window_index, error,
            )
            _remove_temporary_directory(window.staging_dir)
            window.staging_dir = None
            window.invocation_metadata.clear()
            return False
        return True

    def _publish_rank_window(self, rank: int, staging_dir: Path) -> Path:
        """Atomically publish one rank's staged files."""
        target = self._window_directory(rank)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Tensor dump window already exists: {target}")
        return staging_dir.rename(target)


def dump(
    config: DumpConfig,
    *,
    schedule: Callable[[int], DiagnosticAction] | None = None,
    on_dump_ready: Callable[[Any], None] | None = None,
) -> DumperSession:
    # pylint: disable=redefined-outer-name
    """Create a MegaKernel tensor dump session.

    Args:
        config: Output and logical Tensor selection configuration.
        schedule: Optional shared diagnostic step schedule. Collection starts
            on recording actions.
        on_dump_ready: Optional callback invoked after the window publishes any
            Tensor manifests. CSV- and log-only windows do not trigger it.

    Returns:
        MegaKernel tensor dump context manager.

    Note:
        Tensor dumping synchronizes Device-to-Host copies and changes execution
        timing. It is mutually exclusive with MegaKernel cycle profiling.
    """
    return DumperSession(
        config,
        schedule=schedule,
        on_dump_ready=on_dump_ready,
    )
