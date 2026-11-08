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
"""Official MoonEP provider for the HMC execution seam.

The caller owns the HMC runtime, native schedule and autograd. This provider
adds only official planning/transfer calls and a managed planning workspace.
The logical carrier preserves the official local-base/remote-owner contract;
its metadata is consumed only by the official prefetch launcher.
"""
import torch

from hyper_multicore import shmem
from hyper_multicore.shmem import _lifecycle
from hyper_multicore.modules.mega_moe.heap_manager import _WorkspaceLayout

from npu_patch.launchers import (
    planning_workspace_sizes, _env_use_udma,
)


class PlanningWorkspace:
    """Register official symmetric scratch in the existing heap transaction."""

    def __init__(self, engine, tokens, top_k, experts, device):
        self.manager, self.group = engine.heap_manager, engine.spec.ep_group
        self.capacity = 1 << max(0, (tokens - 1).bit_length())
        local, symmetric = planning_workspace_sizes(
            self.capacity, top_k, experts, self.capacity * top_k,
            engine.spec.ep_size, engine.spec.num_cube_cores,
        )
        self.shape = (symmetric,)
        self.closed, self.in_use, self.closing = False, False, False
        self.local = torch.empty(local, device=device, dtype=torch.int32)
        self.key = {"kind": "official_moonep_planning", "capacity": self.capacity, "experts": experts, "top_k": top_k}
        signature = ("official_moonep_planning", self.capacity, experts, top_k, symmetric)
        layout = _WorkspaceLayout(symmetric + 512, signature, 1)
        self.manager.reserve(((self.key, layout),), allow_resize=True)
        self.manager.bind(self, engine.spec, self.key)
        with torch.inference_mode(False), torch.no_grad():
            self.symmetric = shmem.empty(self.shape, dtype=torch.uint8, alignment=512)
        self.symmetric.zero_()
        shmem.barrier()

    def describe_restart_allocations(self, *_args):
        return (_lifecycle._AllocationReplacement(
            self.symmetric, _lifecycle._AllocationSpec(self.shape, torch.uint8, 512)),)

    def validate_restart_allocations(self, _spec, _dtype, _capacity, replacements):
        if self.in_use or len(replacements) != 1 or replacements[0].current is not self.symmetric:
            raise RuntimeError("Official planner scratch is busy or has an invalid replacement")
        return True

    def validate_restarted_allocations(self, _spec, _dtype, _capacity, tensors):
        if len(tensors) != 1 or tensors[0].dtype != torch.uint8 or tuple(tensors[0].shape) != self.shape:
            raise RuntimeError("Official planner scratch replacement changed geometry")

    def bind_restarted_allocations(self, _spec, _dtype, _device, _capacity, tensors):
        self.symmetric = tensors[0]
        self.symmetric.zero_()
        shmem.barrier()

    def close(self):
        if not self.closed:
            if self.in_use:
                raise RuntimeError("Cannot close active official planning workspace")
            shmem.host_barrier()
            shmem.free(self.symmetric)
            self.closed = True


def build_official_placement(hidden, ids, counts, target, rank, *, resources, source_rows,
                             source_inverse=None, source_counts=None):
    """Bind managed scratch to the execution-independent official planner."""
    from .official_planner import OfficialPlanner

    engine = resources
    ranks, experts = counts.shape
    tokens, top_k = ids.shape
    workspace = getattr(engine, "_official_planning", None)
    if workspace is None or workspace.capacity < tokens:
        if workspace is not None:
            workspace.manager.close_workspace(workspace)
        workspace = PlanningWorkspace(engine, tokens, top_k, experts, hidden.device)
        engine._official_planning = workspace
    planner = getattr(workspace, "planner", None)
    if planner is None:
        planner = OfficialPlanner(experts, top_k, ranks, rank, engine.spec.num_cube_cores)
        workspace.planner = planner
    workspace.in_use = True
    try:
        return planner(hidden, ids, counts, target, rank,
                       resources=(workspace.local, workspace.symmetric.data_ptr(), workspace.symmetric.numel()),
                       source_rows=source_rows, source_inverse=source_inverse, source_counts=source_counts)
    finally:
        workspace.in_use = False


def prefetch(pool, weights, copies, extra, token=None):
    """Use official prefetch on the existing shared weight-bank lease."""
    from hyper_multicore.moonep.shmem_transfer import _ordered_transfers
    if not pool.in_use:
        raise RuntimeError("Official prefetch requires an active upstream bank lease")
    pool.cache_token = pool.cache_generation = None
    from npu_patch import _C
    if extra:
        with _ordered_transfers(pool):
            udma = _env_use_udma('prefetch')
            blocks = pool.blocks if udma else getattr(pool, "transfer_blocks", pool.blocks)
            prefetched = tuple(_C.prefetch_banks(
                weights, pool.weights, copies[:, :extra].contiguous(), pool.local_experts,
                pool.rank, pool.ranks, blocks, udma))
    else:
        for bank, weight in zip(pool.weights, weights):
            bank[:pool.local_experts].copy_(weight)
        prefetched = tuple(bank[:pool.local_experts].detach() for bank in pool.weights)
    pool.generation += 1
    pool.cache_generation, pool.cache_token = pool.generation, token
    pool.statistics["prefetches"] += 1
    pool.statistics["remote_prefetches"] += int(bool(extra))
    return prefetched


def return_gradients(pool, gradients, copies, active=True, *, output_dtypes=None):
    """Publish FP32/BF16 partial gradients and reduce with FP32 accumulation."""
    from hyper_multicore.moonep.shmem_transfer import _ordered_transfers
    from npu_patch import _C
    if active:
        with _ordered_transfers(pool):
            udma = _env_use_udma('grad_reduce')
            blocks = pool.blocks if udma else getattr(pool, "transfer_blocks", pool.blocks)
            home = _C.reduce_banks(gradients, pool.weights, copies, pool.local_experts,
                                   pool.rank, blocks, udma,
                                   **({'output_dtypes': output_dtypes} if output_dtypes is not None else {}))
    else:
        home = tuple(gradient[:pool.local_experts] for gradient in gradients)
    pool.statistics["gradient_returns"] += 1
    pool.statistics["remote_gradient_returns"] += int(active)
    pool.cache_token = pool.cache_generation = None
    return tuple(home)


class OfficialTransferBackend:
    """Select official transfers without changing other MoonEP executors globally."""

    prefetch = staticmethod(prefetch)
    return_gradients = staticmethod(return_gradients)
    return_gradients_as = return_gradients
