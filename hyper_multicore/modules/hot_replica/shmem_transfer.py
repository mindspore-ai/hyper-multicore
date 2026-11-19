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
"""Managed HotReplica expert banks over HyperMulticore's private SHMEM heap.

The bank completion event includes native compute and gradient return. It is
independent of the native token workspace event, whose lease can end earlier.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import InitVar, dataclass, field
import threading
import weakref
from typing import Any

import torch

from hyper_multicore import _collective, shmem
from hyper_multicore.shmem import _lifecycle

from .bank_ops import prefetch_banks, reduce_banks
from .placement import ExpertPlacement
from .pipeline import ReplicaPipeline, projection_shapes

@contextmanager
def _ordered_transfers(pool: Any):
    """Order same-root collective transfers without holding native compute."""
    with pool.access():
        previous, event = _ROOT_TRANSFER_EVENTS.get(pool.root, (None, None))
        # The bank lease already waits for all transfers from the same pool.
        if previous is not None and previous() is not pool:
            torch.npu.current_stream(pool.device).wait_event(event)
        try:
            yield
        finally:
            if event is None:
                event = torch.npu.Event()
            event.record(torch.npu.current_stream(pool.device))
            _ROOT_TRANSFER_EVENTS[pool.root] = (weakref.ref(pool), event)


_ROOT_TRANSFER_EVENTS = weakref.WeakKeyDictionary()


class SharedExpertPool:
    """Lease expert storage under an active SHMEM lifecycle."""

    pool_kind = "hot_replica_expert_bank"

    def __init__(
        self, hidden: torch.Tensor, group: torch.distributed.ProcessGroup,
        local_experts: int, intermediate_size: int, blocks: int, layout: str, *,
        root: Any, access: Callable[[], AbstractContextManager], pipeline: bool = False,
    ) -> None:
        """Describe a bank that either EP backend can allocate and lease."""
        self.group = group
        self.rank, self.ranks = torch.distributed.get_rank(group), torch.distributed.get_world_size(group)
        self.local_experts = local_experts
        self.hidden, self.intermediate = hidden.shape[-1], intermediate_size
        self.blocks = blocks
        self.device, self.dtype = hidden.device, hidden.dtype
        self.root, self.access = root, access
        self.weights: tuple[torch.Tensor, ...] = ()
        self._alternate_banks = ()
        self.leased_banks = ()
        self.completion_event = None
        self.completion_stream = None
        self._replay_stream = None
        self.used = False
        self.generation = 0
        self.in_use = False
        self.closed = False
        self.lock = threading.RLock()
        self.graphs = weakref.WeakSet()
        self.statistics = {
            "prefetches": 0, "gradient_returns": 0, "remote_prefetches": 0, "remote_gradient_returns": 0,
            "checkpoint_restores": 0, "backward_cache_hits": 0,
            "early_restores": 0, "early_restore_hits": 0,
        }
        self.capacity_floor = local_experts
        self.expert_weight_layout = layout
        self.pipeline = ReplicaPipeline(self) if pipeline else None

    @property
    def allocation_bytes(self) -> int:
        """Report the aligned combined bank and optional peer-event requirement."""
        weights = 6 * self.local_experts * self.hidden * self.intermediate * self.dtype.itemsize + 1024
        signal_bytes = 0 if self.pipeline is None or self.pipeline.inline else 4 * self.pipeline.signal_words + 512
        return weights + signal_bytes

    def allocate(self) -> None:
        """Allocate after the caller has acquired the runtime and reserved capacity."""
        _collective.collective_call(self._allocate, stage="HotReplica bank construction", group=self.group)

    def _allocate(self) -> None:
        """Allocate persistent symmetric weight banks outside inference mode."""
        allocated = []
        # Persistent buffers must remain writable after an inference-only first call.
        with torch.inference_mode(False), torch.no_grad():
            try:
                for shape in self._bank_shapes():
                    allocated.append(shmem.empty(shape, dtype=self.dtype, alignment=512))
            finally:
                self.weights = tuple(allocated)
            if self.pipeline is not None and not self.pipeline.inline:
                self.pipeline.allocate()
            if self.completion_event is None:
                self.completion_event = torch.npu.Event()

    def _free(self) -> None:
        self._alternate_banks = ()
        if self.pipeline is not None:
            self.pipeline.free()
        # A later collective free can fail. Retire each successful allocation
        # immediately so a retry does not free the same SHMEM storage twice.
        while self.weights:
            shmem.free(self.weights[0])
            self.weights = self.weights[1:]
        self.completion_stream = None

    def _bank_shapes(self, layout=None):
        return tuple(
            (2 * self.local_experts, *shape)
            for shape in projection_shapes(self.hidden, self.intermediate, layout or self.expert_weight_layout)
        )

    def _allocations(self, dtype):
        specs = tuple((shape, dtype) for shape in self._bank_shapes())
        if self.pipeline is None or self.pipeline.inline:
            return self.weights, specs
        return (*self.weights, self.pipeline.storage), (*specs, ((self.pipeline.signal_words,), torch.int32))

    def describe_restart_allocations(
        self,
        spec: Any,
        dtype: torch.dtype,
        receive_capacity: int,
    ) -> tuple[Any, ...]:
        """Participate in HyperMulticore's atomic allocation replacement, without rebuilding runtime."""
        del spec, receive_capacity
        tensors, specs = self._allocations(dtype)
        return tuple(
            _lifecycle._AllocationReplacement(  # pylint: disable=protected-access
                tensor, _lifecycle._AllocationSpec(shape, tensor_dtype, 512)  # pylint: disable=protected-access
            )
            for tensor, (shape, tensor_dtype) in zip(tensors, specs)
        )

    def validate_restart_allocations(
        self,
        spec: Any,
        dtype: torch.dtype,
        current_capacity: int,
        replacements: tuple[Any, ...],
    ) -> bool:
        """Keep bank leases independent of the token workspace's native completion."""
        del spec, current_capacity
        if self.in_use:
            raise RuntimeError("Active HotReplica bank prevents heap growth")
        tensors, specs = self._allocations(dtype)
        if len(replacements) != len(tensors):
            raise RuntimeError("HotReplica bank restart allocation mapping changed")
        for tensor, replacement, (shape, tensor_dtype) in zip(tensors, replacements, specs):
            if replacement.current is not tensor or tuple(tensor.shape) != shape or tensor.dtype != tensor_dtype:
                raise RuntimeError("HotReplica bank restart mapping does not match committed storage")
        return bool(self.weights)

    def validate_restarted_allocations(
        self,
        spec: Any,
        dtype: torch.dtype,
        receive_capacity: int,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Validate every replacement before the HeapManager commits any owner."""
        del spec, receive_capacity
        _, specs = self._allocations(dtype)
        if len(tensors) != len(specs):
            raise RuntimeError("HotReplica bank restart returned incomplete storage")
        if any(
            tuple(tensor.shape) != shape or tensor.dtype != tensor_dtype
            for tensor, (shape, tensor_dtype) in zip(tensors, specs)
        ):
            raise RuntimeError("HotReplica bank restart returned an incompatible allocation")

    def bind_restarted_allocations(
        self,
        spec: Any,
        dtype: torch.dtype,
        device: Any,
        receive_capacity: int,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Bind new bank tensors after HMC replaces the symmetric heap."""
        del spec, dtype, device, receive_capacity
        self._alternate_banks = ()
        if self.pipeline is None or self.pipeline.inline:
            self.weights = tensors
            if self.pipeline is not None:
                self.pipeline.free()
        else:
            self.weights = tensors[:-1]
            self.pipeline.bind(tensors[-1])
        self.generation += 1
        # The heap transaction retired the previous bank. Retain its completion
        # event as a conservative first-use dependency, but forget its stream.
        self.completion_stream = None

    def claim(self, layout: str) -> None:
        """Order bank reuse after compute and all peer gradient reads."""
        with self.access(), self.lock:
            if self.closed or self.in_use:
                raise RuntimeError("HotReplica expert banks require serial, open execution")
            stream = torch.npu.current_stream(self.device)
            # FIFO already orders prior readers on this exact device stream.
            # Cross-stream reuse still waits on the last recorded completion.
            if self.used and (
                self.completion_stream is None
                or stream.device != self.completion_stream.device
                or stream != self.completion_stream
            ):
                stream.wait_event(self.completion_event)
            if layout == self.expert_weight_layout:
                self.leased_banks = self.weights
            else:
                # Both layouts transfer equal-length contiguous expert records.
                # The lease orders their reuse across streams and autograd graphs.
                if not self._alternate_banks:
                    with torch.inference_mode(False), torch.no_grad():
                        self._alternate_banks = tuple(
                            bank.view(shape) for bank, shape in zip(self.weights, self._bank_shapes(layout))
                        )
                self.leased_banks = self._alternate_banks
            self.in_use = True

    def release(self) -> None:
        """Record the complete weight/gradient lease on the current stream."""
        with self.access(), self.lock:
            if not self.in_use:
                raise RuntimeError("HotReplica expert banks released without a lease")
            stream = torch.npu.current_stream(self.device)
            self.completion_event.record(stream)
            self.completion_stream = stream
            self.used = True
            self.in_use = False
            self.leased_banks = ()

    def active_weights(self, extra: int) -> tuple[torch.Tensor, ...]:
        """Return the logical prefix of the leased bank."""
        return tuple(bank[:self.local_experts + extra] for bank in self.leased_banks)

    def stage_snapshot(self, layout: str, snapshot: tuple[tuple[torch.Tensor, ...], Any]) -> int | None:
        """Submit a local suffix restore and retire its lease on the copy stream."""
        tails, _ = snapshot
        if not tails[0].shape[0]:
            return None
        with self.access(), self.lock:
            if self.closed or self.in_use:
                return None
            if self._replay_stream is None:
                self._replay_stream = torch.npu.Stream(device=self.device)
            with torch.npu.stream(self._replay_stream):
                self.claim(layout)
                try:
                    # Invalidate old tickets before a write, including a failed copy.
                    self.generation += 1
                    self.restore_snapshot(snapshot, self.local_experts)
                    self.statistics["early_restores"] += 1
                    return self.generation
                finally:
                    self.release()

    def restore_snapshot(self, snapshot: tuple[tuple[torch.Tensor, ...], Any], start: int) -> None:
        """Copy saved guest operands into the selected prefix or suffix."""
        tails, ready = snapshot
        stream = torch.npu.current_stream(self.device)
        stream.wait_event(ready)
        for bank, tail in zip(self.leased_banks, tails):
            bank[start:start + tail.shape[0]].copy_(tail)
            tail.record_stream(stream)

    def prefetch(
        self, weights: tuple[torch.Tensor, ...], copies: torch.Tensor, extra: int,
        *, snapshot: tuple[tuple[torch.Tensor, ...], Any] | None = None,
        staged_generation: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Publish current owner weights and fill borrowed slots on the lease stream."""
        if not self.in_use:
            raise RuntimeError("HotReplica prefetch requires an active bank lease")
        if self.pipeline is not None:
            self.pipeline.invalidate()
        with _ordered_transfers(self):
            if snapshot is None:
                prefetch_banks(weights, self.leased_banks, copies, self.rank, self.blocks)
                self.statistics["remote_prefetches"] += int(bool(extra))
            else:
                for bank, home in zip(self.leased_banks, weights):
                    bank[:self.local_experts].copy_(home)
                if staged_generation is not None and staged_generation == self.generation:
                    self.statistics["early_restore_hits"] += 1
                else:
                    self.restore_snapshot(snapshot, self.local_experts)
                self.statistics["checkpoint_restores"] += 1
        self.generation += 1
        self.statistics["prefetches"] += 1
        return self.active_weights(extra)

    def can_close(self) -> bool:
        """Reject cleanup while a call or retained backward needs the bank."""
        return not self.in_use and not self.graphs

    def return_generated_gradients(
        self,
        gradients: tuple[torch.Tensor, ...],
        copies: torch.Tensor,
        active: bool = True,
        *,
        output_dtypes: tuple[torch.dtype, ...] | None = None,
        published_slots: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Sum borrowed gradients into the expert producer's home gradients."""
        if self.pipeline is not None:
            self.pipeline.invalidate()
        with _ordered_transfers(self):
            home = reduce_banks(
                gradients,
                self.leased_banks,
                copies,
                self.rank,
                self.blocks,
                output_dtypes,
                published_slots=published_slots,
            )
        self.generation += 1
        self.statistics["gradient_returns"] += 1
        self.statistics["remote_gradient_returns"] += int(active)
        return tuple(home)

    def track_graph(self, ctx: Any) -> None:
        """Keep the bank's owner alive while its autograd graph can replay."""
        self.graphs.add(ctx)

    def release_graph(self, ctx: Any) -> None:
        """Retain explicit retained graphs and release completed graph ownership."""
        keep = getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None)  # pylint: disable=protected-access
        if keep is not None and not keep():
            self.graphs.discard(ctx)

    def _wait_for_close(self):
        if self.in_use or self.graphs:
            raise RuntimeError("Cannot close HotReplica banks while a call or graph owns them")
        if self.used:
            self.completion_event.synchronize()

    def close(self) -> None:
        """Converge local stream errors before peers enter the SHMEM barrier."""
        if self.closed:
            return
        with self.access():
            _collective.collective_call(self._wait_for_close, stage="HotReplica bank drain", group=self.group)
            shmem.host_barrier()
            self._free()
            self.closed = True


@dataclass
class ExpertWeightBinding:
    """Per-call ownership; the autograd function saves current owner parameters."""

    pool: SharedExpertPool
    placement: ExpertPlacement
    layout: str
    snapshot: Any = None
    generation: int | None = None
    staged_generation: int | None = None
    fused: bool = False
    token: object = field(default_factory=object)
    transfers: tuple = ()
    gradient_storage: tuple = ()
    fused_slots: InitVar[int | None] = None

    def __post_init__(self, fused_slots):
        self.edges = self.pool.pipeline.plan(self.placement, fused_slots) if self.fused else None

    @property
    def owns_returns(self):
        return self.fused and bool(self.edges[2])

    def prepare_replay(self) -> None:
        """Restore the saved suffix while the framework replays attention and routing."""
        # Only the agreed replay can overwrite the fused prefix. Early staging
        # can run locally before EP agrees that parameter versions still match.
        if self.snapshot is not None and not self.fused:
            self.staged_generation = self.pool.stage_snapshot(self.layout, self.snapshot)

    def acquire(
        self, weights: tuple[torch.Tensor, ...], *, backward: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Lease reusable banks and publish this graph's saved owner weights."""
        self.pool.claim(self.layout)
        try:
            if self.fused:
                with _ordered_transfers(self.pool):
                    result = self.pool.pipeline.begin(weights, self, backward)
                stats = self.pool.statistics
                stats["fused_prefetches"] = stats.get("fused_prefetches", 0) + 1
                return result
            if self.generation is not None and self.generation == self.pool.generation:
                self.pool.statistics["backward_cache_hits"] += 1
                return self.pool.active_weights(self.placement.borrowed_capacity)
            staged_generation, self.staged_generation = self.staged_generation, None
            borrowed = self.pool.prefetch(
                weights, self.placement.copies, self.placement.borrowed_capacity,
                snapshot=self.snapshot, staged_generation=staged_generation,
            )
            self.generation = self.pool.generation
            return borrowed
        except BaseException:
            try:
                if self.fused:
                    self.pool.pipeline.resident = None
                    self.pool.pipeline.complete(self)
            finally:
                self.pool.release()
            raise

    def runtime(self, ordinary, gradients=None):
        """Select split weights and optional in-kernel dW for this launch."""
        return self.pool.pipeline.runtime(ordinary, self, gradients) if self.fused else None

    def save_snapshot(self, weights: tuple[torch.Tensor, ...]) -> None:
        """Save the used guests from the completed origin of the checkpoint."""
        start = 0 if self.fused else self.pool.local_experts
        used = self.placement.published_slots
        stop = start + (self.placement.borrowed_capacity if used is None else used)
        tails = tuple(weight[start:stop].detach().clone() for weight in weights)
        ready = torch.npu.Event()
        ready.record(torch.npu.current_stream(tails[0].device))
        self.snapshot = tails, ready
        self.generation = None

    def gradients(
        self, gradients: tuple[torch.Tensor, ...], weights: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Return partial expert gradients to owners before casting parameter gradients."""
        if self.fused:
            stats = self.pool.statistics
            stats["fused_gradient_returns"] = stats.get("fused_gradient_returns", 0) + 1
            return gradients
        return self.pool.return_generated_gradients(
            gradients, self.placement.copies, output_dtypes=tuple(w.dtype for w in weights),
            published_slots=self.placement.published_slots,
        )

    def release(self) -> None:
        """Record completion after both expert compute and peer gradient reads."""
        if self.fused:
            self.pool.pipeline.complete(self)
        self.pool.release()

    def track_graph(self, ctx: Any) -> None:
        """Prevent resource cleanup while a backward graph can still execute."""
        self.pool.track_graph(ctx)

    def release_graph(self, ctx: Any) -> None:
        """Release only completed, non-retained graphs."""
        self.pool.release_graph(ctx)
