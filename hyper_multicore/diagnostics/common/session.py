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
"""Process-local ownership for mutually exclusive MegaKernel diagnostics."""

from __future__ import annotations

from threading import Lock
from typing import Protocol

from hyper_multicore.diagnostics.common.call import DiagnosticCall, LaunchContext


class DiagnosticSession(Protocol):
    """Minimal contract implemented independently by profiler and Tensor dumper."""

    def should_prepare_call(self) -> bool:
        """Return whether this step's schedule calls for diagnostic preparation.

        Profiling includes warmup; Tensor dump only enables recording actions.
        This does not check active-session ownership. Invocation-specific
        selection is performed by prepare_call().
        """
        raise NotImplementedError

    def prepare_call(self, context: LaunchContext) -> DiagnosticCall:
        """Return complete launch arguments and tool-owned invocation state.

        Args:
            context: Selected invocation or operand context supplied by the diagnostic framework.
        """
        raise NotImplementedError


_ACTIVE_LOCK = Lock()
_ACTIVE_TOOL_NAME: str | None = None
_ACTIVE_SESSION: DiagnosticSession | None = None


def _activate_session(tool_name: str, session: DiagnosticSession) -> None:
    """Claim the single process-local MegaKernel diagnostic session."""
    # pylint: disable=global-statement
    global _ACTIVE_TOOL_NAME, _ACTIVE_SESSION
    with _ACTIVE_LOCK:
        if _ACTIVE_SESSION is not None:
            raise RuntimeError(
                "nested MegaKernel diagnostic contexts are not supported: "
                f"active={_ACTIVE_TOOL_NAME}, requested={tool_name}"
            )
        _ACTIVE_TOOL_NAME = tool_name
        _ACTIVE_SESSION = session


def _deactivate_session(session: DiagnosticSession) -> None:
    """Release ownership when ``session`` is still the active owner."""
    # pylint: disable=global-statement
    global _ACTIVE_TOOL_NAME, _ACTIVE_SESSION
    with _ACTIVE_LOCK:
        if _ACTIVE_SESSION is session:
            _ACTIVE_TOOL_NAME = None
            _ACTIVE_SESSION = None


def _get_active_session(tool_name: str | None = None) -> DiagnosticSession | None:
    """Return the active session, optionally restricted to one diagnostic tool."""
    with _ACTIVE_LOCK:
        if tool_name is not None and _ACTIVE_TOOL_NAME != tool_name:
            return None
        return _ACTIVE_SESSION


def _reset_active_session_for_test() -> None:
    """Clear leaked diagnostic state after a unit-test failure."""
    # pylint: disable=global-statement
    global _ACTIVE_TOOL_NAME, _ACTIVE_SESSION
    with _ACTIVE_LOCK:
        _ACTIVE_TOOL_NAME = None
        _ACTIVE_SESSION = None
