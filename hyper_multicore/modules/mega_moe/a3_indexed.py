# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""A3 BF16 indexed dispatch and the count exchange's readiness contract."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import os
from typing import TYPE_CHECKING, Any

import torch

from hyper_multicore import _collective
from hyper_multicore._loader import require_multicore_environment

if TYPE_CHECKING:
    from .spec import MegaMoeSpec
    from .workspace import MegaMoeWorkspace


MAX_INDEXED_TOKENS = 16384
COUNT_FENCE_BIT = 1 << 10
REPLAY_FENCE_BIT = 1 << 7
_SOURCE_BITS = 17
_FENCED = 1 << 16


def requested() -> bool:
    """Capture the opt-in as part of the execution group's configuration."""
    return os.environ.get("HMC_A3_INDEXED_PUSH") == "1"


@lru_cache(maxsize=1)
def _ordered_operator():
    from hyper_multicore.modules.hot_replica.counts import _operators  # pylint: disable=import-outside-toplevel

    operators = _operators()
    if operators is None:
        return None
    namespace, library = operators
    available = getattr(namespace, "a3_ordered_replica_route_available", None)
    if available is None or not available(library):
        return None
    vendor = require_multicore_environment() / "op_api" / "lib" / "libcust_opapi.so"
    api = ctypes.CDLL(str(vendor), mode=ctypes.RTLD_LOCAL)
    version = getattr(api, "HmcA3IndexedPushVersion", None)
    if version is None or version() != 1:
        return None
    return namespace.a3_ordered_replica_route, library


def qualify(spec: MegaMoeSpec, enabled: bool) -> bool:
    """Agree on the supported geometry and bundled code before preparing routes."""
    if (not enabled or spec.architecture != 2201 or spec.precision != "bf16"
            or not spec.auto_tokens or spec.dispatch_mode != "push" or spec.transport_engine != "mte"
            or spec.expert_weight_layout != "EOI" or spec.apply_probs_before_fc2 or spec.batch_experts
            or (spec.ep_size, spec.num_experts, spec.top_k, spec.hidden_size, spec.intermediate_size)
            != (16, 128, 8, 4096, 1408)):
        return False
    from hyper_multicore.modules.hot_replica.bounded_placement import _node_identity  # pylint: disable=import-outside-toplevel

    node = _node_identity()
    local_error = None
    try:
        supported = _ordered_operator() is not None
    except Exception as error:  # pylint: disable=broad-exception-caught
        local_error = error
        supported = False
    failure = None if local_error is None else _collective.failure_report(local_error)
    reports = _collective.exchange((node, supported, failure), group=spec.ep_group,
                                   failure_prefix="A3 indexed dispatch capability")
    nodes, support, failures = zip(*reports)
    if any(failures):
        raise RuntimeError(f"A3 indexed dispatch capability failed on "
                           f"{_collective.format_collective_failures(failures)}") from local_error
    return len(set(nodes)) == 1 and all(support)


def ordered_count(ids: torch.Tensor, mask: int):
    """Return the validated local wire, stable inverse and stable source rows."""
    operator, library = _ordered_operator()
    return operator(ids, mask, library)


def note_source_ready(ids: torch.Tensor) -> None:
    """Mark final Router IDs on their producer stream before expert weights materialize."""
    if (requested() and ids.device.type == "npu" and ids.ndim == 2 and ids.shape[1] == 8
            and 0 < ids.shape[0] <= MAX_INDEXED_TOKENS
            and not ids.is_inference() and not hasattr(ids, "_hmc_indexed_source_ready")):
        ready = torch.npu.Event()
        ready.record(torch.npu.current_stream(ids.device))
        ids._hmc_indexed_source_ready = (ids._version, ready)


def take_source_ready(ids: torch.Tensor):
    """Consume the event only for the original, unmodified tensor."""
    marker = getattr(ids, "_hmc_indexed_source_ready", None)
    if marker is not None:
        del ids._hmc_indexed_source_ready
        version, ready = marker
        if version == ids._version:
            return ready
    return None


def pack_dispatch_offsets(offsets: torch.Tensor, source_rows: torch.Tensor) -> torch.Tensor:
    """Append the int32 source rows after the ordinary int64 offsets."""
    return torch.cat((offsets, source_rows.view(torch.int64)))


def launch_sequence(plan_tokens: int, source_rows: int, fenced: bool) -> int:
    """Encode the source extent and readiness in the private launch attribute."""
    return -((plan_tokens << _SOURCE_BITS) | source_rows | (_FENCED if fenced else 0))


@dataclass(frozen=True)
class CountFence:
    """Track the exact SHMEM buffers cleared before the EP count exchange."""

    events: torch.Tensor
    receive: torch.Tensor
    combine: torch.Tensor
    lease_generation: int
    stream: Any

    @classmethod
    def before_gather(cls, workspace: MegaMoeWorkspace, stream: Any, *, launch_stream: Any = None):
        """Prepare counters when already bound and order them after the last lease."""
        if (not workspace.forward_ready_initialized or workspace.forward_event_counters is None
                or workspace.expert_buffer is None or workspace.routed_buffer is None or workspace.in_use):
            return None
        workspace.wait_for_reuse(stream=stream)
        events = workspace.prepare_event_counters(forward=True)
        return cls(events, workspace.expert_buffer, workspace.routed_buffer, workspace.lease_generation,
                   stream if launch_stream is None else launch_stream)

    def matches(self, workspace: MegaMoeWorkspace) -> bool:
        """A capacity restart replaces these symmetric allocations on every EP rank."""
        return (workspace.forward_event_counters is self.events and workspace.expert_buffer is self.receive
                and workspace.routed_buffer is self.combine and workspace.forward_ready_initialized)

    def claim_events(self, workspace: MegaMoeWorkspace, counter_bytes: int | None) -> torch.Tensor:
        """Reuse the collective's counters for the immediately following lease."""
        if (not self.matches(workspace) or workspace.lease_generation != self.lease_generation + 1
                or workspace.lease_stream != self.stream):
            raise RuntimeError("A3 indexed dispatch changed its workspace lease after route agreement")
        offset = 0 if counter_bytes is None else workspace.event_counter_bytes - counter_bytes
        if not 0 <= offset < workspace.event_counter_bytes:
            raise ValueError("MegaMoE event counters are outside the reserved workspace")
        return self.events[offset:] if offset else self.events
