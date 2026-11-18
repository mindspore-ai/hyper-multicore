# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""MX task profiling bindings; tensor dumps require a separate physical-view contract."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from hyper_multicore.diagnostics.api import prepare_mega_kernel_diagnostics
from hyper_multicore.diagnostics.common.session import _get_active_session

_SCOPE = ContextVar('mx_profile_scope', default=None)


@contextmanager
def mx_profile_scope(*, layer: str, phase: str = 'forward') -> Iterator[None]:
    """Optional caller labels; no model identity or filtering inside the component."""
    if not isinstance(layer, str) or not layer or phase not in ('forward', 'recompute', 'backward'):
        raise ValueError('MX profile scope requires a layer and forward/recompute/backward phase')
    token = _SCOPE.set({'layer': layer, 'phase': phase})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def current_mx_profile_scope():
    """Return the current layer and phase used for profiling."""
    return _SCOPE.get()


@dataclass(frozen=True)
class MxProfileOwnerContext:
    """Retain the expert topology for task attribution."""
    spec: Any


def resolve_mx_profile_owner(operator: Any, task: Any, context: MxProfileOwnerContext) -> int:
    """Map the profiled native task to its global expert."""
    spec = context.spec
    base = spec.rank_id * spec.local_experts
    if operator.name in ('dispatch', 'combine'):
        tasks_per_expert, remainder = divmod(task.task_split_num, spec.num_experts)
        if remainder or not tasks_per_expert:
            raise ValueError('MX transport profiling requires a whole task count per global expert.')
        expert = task.task_index // tasks_per_expert
        if not 0 <= expert < spec.num_experts:
            raise ValueError('MX transport profiling expert index is out of range.')
        if operator.name == 'dispatch' and spec.dispatch_mode == 'push':
            return expert
        return base + expert % spec.local_experts
    workers, remainder = divmod(operator.task_num, spec.local_experts)
    if remainder or not workers:
        raise ValueError('MX compute profiling requires a whole worker count per local expert.')
    expert = task.task_index // workers
    if not 0 <= expert < spec.local_experts:
        raise ValueError('MX compute profiling expert index is out of range.')
    return base + expert


def mx_profile_bindings(buffers: Sequence[Any], runtime: Any, events: Any) -> list[Any]:
    """Expand the logical operand list for the profiler."""
    if not 1 <= len(buffers) <= 64:
        raise ValueError('MX profiling requires 1..64 business buffers.')
    # This is the expanded device input_list, not the compact Torch TensorList ABI.
    return list(buffers) + [None] * (64 - len(buffers)) + [runtime, events, None, None]


def mx_profile_capture_enabled() -> bool:
    """Select profiling and reject unsupported diagnostic modes."""
    session = _get_active_session()
    if session is None or not session.is_capture_enabled():
        return False
    if _get_active_session('profiler') is not session:
        raise RuntimeError('MX tensor dumps are unsupported; only the task profiler has a physical ABI contract.')
    return True


def prepare_mx_diagnostics(host: Any, spec: Any, device: Any) -> Any:
    """Create the profiler binding for this native graph."""
    return prepare_mega_kernel_diagnostics(
        host.graph, host.config, kernel_name=host.native_operator, parameter_count=68, device=device,
        rank=spec.rank_id, owner_label='Expert', owner_resolver=resolve_mx_profile_owner,
        owner_context=MxProfileOwnerContext(spec), execution_mode='event_driven',
        vector_worker_stride=1,
    )


def profile_mx_launch(buffers: Sequence[Any], runtime: Any, events: Any, spec: Any,
                      direction: str, prepare: Callable, launch: Callable, *, scope=None) -> None:
    """Submit the graph through the shared diagnostic lifecycle."""
    parameters = mx_profile_bindings(buffers, runtime.runtime, events)
    with runtime.diagnostics.launch(
            parameters, direction=direction, fallback_event_counters=events, tags=scope) as call:
        # Use master's diagnostic lifecycle; normal launches bypass it entirely.
        descriptor = prepare(buffers, call.runtime_config, call.event_counters, spec)
        launch(buffers, call.runtime_config, call.event_counters, call.diagnostic_buffer, descriptor)
