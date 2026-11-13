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
"""Complete launch arguments and a tool-neutral per-invocation lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class LaunchArguments:
    """Complete native arguments selected by exactly one diagnostic tool."""

    runtime_config: Any
    event_counters: Any
    diagnostic_buffer: Any
    clear_event_counters: Any


@dataclass(frozen=True)
class LaunchContext:
    """Business-independent inputs available to a selected diagnostic session."""

    runtime: Any
    parameters: Sequence[Any]
    direction: str
    fallback_event_counters: Any
    tags: dict[str, str] | None = None

    def ordinary_arguments(self) -> LaunchArguments:
        """Reuse the original runtime and counters without allocating device storage."""
        return LaunchArguments(
            self.runtime.normal_tensor, self.fallback_event_counters,
            self.runtime.normal_tensor if self.fallback_event_counters is None else self.fallback_event_counters,
            self.fallback_event_counters,
        )


class DiagnosticCall:
    """Context boundary shared by ordinary, profiling and Tensor-dump calls."""

    def __init__(self, arguments: LaunchArguments) -> None:
        """Store the full argument selection and idempotent completion state."""
        self.arguments = arguments
        self.finished = False

    @property
    def runtime_config(self) -> Any:
        """Return this call's complete runtime configuration."""
        return self.arguments.runtime_config

    @property
    def event_counters(self) -> Any:
        """Return the existing execution event counters."""
        return self.arguments.event_counters

    @property
    def diagnostic_buffer(self) -> Any:
        """Return this call's shared diagnostic-buffer argument."""
        return self.arguments.diagnostic_buffer

    @property
    def clear_event_counters(self) -> Any:
        """Return the event-counter region cleared by the execution owner."""
        return self.arguments.clear_event_counters

    def __enter__(self) -> DiagnosticCall:
        """Enter the native launch boundary."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Complete successful calls and cancel failed preparation or capture."""
        if exc_type is not None:
            self.cancel()
            return
        try:
            self.complete()
        except BaseException:
            self.cancel()
            raise

    def complete(self) -> None:
        """Finish an ordinary call without any diagnostic work."""
        self.finished = True

    def cancel(self) -> None:
        """Discard an ordinary call without touching execution-owned resources."""
        self.finished = True
