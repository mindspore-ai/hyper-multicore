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
"""HotReplica planning and weight transfers for an external expert executor."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore import _automatic, shmem
from hyper_multicore._device import device_core_limit

from .capability import require_capability
from .checkpoint import ReplayRecord, agree_replay, is_checkpointing, route_record
from .placement import ExpertPlacement
from .planner import HotReplicaPlanner, PlacementRequest
from .shmem_transfer import ExpertWeightBinding, SharedExpertPool

WeightPair = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class HotReplicaRoute:
    """An agreed route and placement for one independent EP invocation."""

    counts: torch.Tensor
    placement: ExpertPlacement | None
    record: ReplayRecord | None
    replay: bool = False
    received: int = 0


class HotReplica:
    """Plan and lend experts without owning their router or grouped GEMMs."""

    def __init__(
        self, ep_group: dist.ProcessGroup, geometries: tuple[tuple[int, int, int], ...],
    ) -> None:
        """Register (home experts, hidden size, intermediate size) for one model."""
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        if self.ep_size <= 1:
            raise ValueError("HotReplica requires an EP group with more than one rank")
        self.rank = dist.get_rank(ep_group)
        self._geometries = tuple(dict.fromkeys(geometries))
        self._planner = HotReplicaPlanner()
        self._banks: dict[tuple[int, int, int], SharedExpertPool] = {}
        self._lock = RLock()
        self._cores: int | None = None
        self._acquired = False

    def prepare(
        self, hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
        counts: Callable[[], torch.Tensor], intermediate_size: int,
    ) -> HotReplicaRoute:
        """Plan from the framework's count exchange or recover the original route."""
        if self._cores is None:
            require_capability(hidden, self.ep_group)
            self._cores = device_core_limit(hidden.device, "cube_core_num")
        flat = hidden.reshape(-1, hidden.shape[-1])
        record, replay = route_record(self)
        if replay and agree_replay(record, flat, ids, probs, None, self, 1, weight_error=False):
            placement, table = record.route
            return HotReplicaRoute(table, placement, record, replay=True)
        if replay:
            record = None
        table = counts()
        # Read one small aggregated vector. The dispatcher uses the exact same
        # table for all-to-all input/output splits instead of gathering again.
        ranks, experts = table.shape
        loads = table.reshape(ranks, ranks, experts // ranks).sum(dim=(0, 2)).tolist()
        request = PlacementRequest(
            flat, ids, table, None, max(loads, default=0), sum(loads), self.rank,
            intermediate_size, self._cores, torch.is_grad_enabled() or is_checkpointing(),
        )
        placement = self._planner.plan_native(request)
        if record is not None:
            record.ids = ids.detach().clone()
            record.route = placement, table
            record.stream = torch.npu.current_stream(hidden.device)
        return HotReplicaRoute(table, placement, record, received=loads[self.rank])

    def borrow(self, weights: WeightPair, route: HotReplicaRoute) -> WeightPair:
        """Materialize owned operands suitable for ordinary autograd GEMMs."""
        if route.placement is None:
            return weights
        return self._borrow(weights, route, weights[0].dtype, fp32_gradients=False)

    def borrow_for_fp32_gradients(
        self, weights: WeightPair, route: HotReplicaRoute, *, compute_dtype: torch.dtype,
    ) -> tuple[WeightPair, WeightPair]:
        """Return compute operands and FP32 gradient targets for an active placement.

        Supply each target as the differentiable input of its expert operator.
        That operator must return an FP32 weight gradient for the target. The
        operands are detached; use them for the GEMM without reading the values
        of the targets. Only the owner gradient uses the original parameter dtype.
        """
        if route.placement is None:
            raise ValueError("FP32 gradient targets require an active HotReplica placement")
        first, second, grad_first, grad_second = self._borrow(
            weights, route, compute_dtype, fp32_gradients=True,
        )
        return (first, second), (grad_first, grad_second)

    def _borrow(
        self, weights: WeightPair, route: HotReplicaRoute, compute_dtype: torch.dtype, *, fp32_gradients: bool,
    ) -> tuple[torch.Tensor, ...]:
        first, second = weights
        geometry = first.shape[0], first.shape[1], second.shape[1]
        if geometry not in self._geometries:
            raise ValueError("HotReplica owner weight geometry was not registered")
        record = route.record
        if route.replay and record is not None:
            binding = record.weight_binding
        else:
            pool = self._bank(first, geometry, compute_dtype)
            binding = ExpertWeightBinding(pool, route.placement, "EIO")
        return _BorrowedExpertWeights.apply(
            first, second, binding, record if not route.replay else None, compute_dtype, fp32_gradients,
        )

    def _bank(
        self, weight: torch.Tensor, geometry: tuple[int, int, int], compute_dtype: torch.dtype,
    ) -> SharedExpertPool:
        with self._lock:
            pool = self._banks.get(geometry)
            if pool is not None:
                return pool
            local, hidden, intermediate = geometry
            anchor = weight.new_empty((0, hidden), dtype=compute_dtype)
            pool = SharedExpertPool(
                anchor, self.ep_group, local, intermediate, self._cores, "EIO",
                root=self, access=lambda: self._lock,
            )
            if not self._acquired:
                page = 2 * 1024**2
                needed = sum(
                    6 * experts * width * inner * anchor.element_size() + 1024
                    for experts, width, inner in self._geometries
                )
                shmem.acquire(self.ep_group, heap_size_bytes=((needed + page - 1) // page) * page)
                self._acquired = True
                _automatic.register(self.close, self.ep_group)
            try:
                pool.allocate()
            except Exception:
                pool.close()
                raise
            self._banks[geometry] = pool
            return pool

    def close(self) -> None:
        """Release banks and the runtime before the framework destroys the EP group."""
        with self._lock:
            if any(not pool.can_close() for pool in self._banks.values()):
                raise RuntimeError("HotReplica cannot free banks held by an active call or graph")
            for pool in self._banks.values():
                pool.close()
            self._banks.clear()
            if self._acquired:
                shmem.release()
                self._acquired = False


class _BorrowedExpertWeights(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Return owned weight operands and map their gradients to parameter owners."""

    @staticmethod
    def forward(
        ctx: Any, first: torch.Tensor, second: torch.Tensor,
        binding: ExpertWeightBinding, record: ReplayRecord | None,
        compute_dtype: torch.dtype, fp32_gradients: bool,
    ) -> tuple[torch.Tensor, ...]:
        """Clone only the operands needed by the ordinary autograd caller."""
        owners = (first.to(compute_dtype), second.to(compute_dtype)) if fp32_gradients else (first, second)
        operands = binding.acquire(owners)
        try:
            local = binding.pool.local_experts
            result = tuple(torch.cat((weight, bank[local:]), dim=0) for weight, bank in zip(owners, operands))
            if record is not None:
                binding.save_snapshot(operands)
                record.weight_binding = binding
        finally:
            binding.release()
        ctx.binding = binding
        ctx.dtypes = first.dtype, second.dtype
        binding.track_graph(ctx)
        if not fp32_gradients:
            return result

        # Shape-only targets keep the autograd edges in FP32 with scalar storage.
        ctx.mark_non_differentiable(*result)
        ctx.set_materialize_grads(False)
        ctx.gradient_shapes = tuple(weight.shape for weight in result)
        ctx.gradient_device = first.device
        target = first.new_empty((), dtype=torch.float32)
        return (*result, *(target.expand(shape) for shape in ctx.gradient_shapes))

    @staticmethod
    def backward(ctx: Any, grad_first: torch.Tensor | None, grad_second: torch.Tensor | None,
                 *fp32_gradients: torch.Tensor | None) -> tuple[Any, ...]:
        """Return borrowed gradients in FP32, then cast to the owner dtype."""
        if fp32_gradients:
            grad_first, grad_second = (
                grad if grad is not None else torch.zeros(shape, device=ctx.gradient_device, dtype=torch.float32)
                for grad, shape in zip(fp32_gradients, ctx.gradient_shapes)
            )
        binding = ctx.binding
        binding.pool.claim("EIO")
        try:
            grads = binding.pool.return_generated_gradients(
                (grad_first, grad_second), binding.placement.copies, output_dtypes=ctx.dtypes,
                published_slots=binding.placement.published_slots,
            )
            binding.release_graph(ctx)
            return grads[0], grads[1], None, None, None, None
        finally:
            binding.release()
