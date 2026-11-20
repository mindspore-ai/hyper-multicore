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
"""Two-rank Ascend 910_93 HotReplica placement and checkpoint regression."""
from dataclasses import replace
import gc
import os
import weakref

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_multicore import BasePlanner, MegaMoeExperts, MegaMoeNumericsOptions, MegaMoePlacementOptions
from hyper_multicore.modules.hot_replica.checkpoint import wrap_checkpoint
from hyper_multicore.modules.hot_replica.placement import build_placement
from hyper_multicore.modules.hot_replica.planner import HotReplicaPlanner


class SkewPlanner(BasePlanner):
    """Replicate when peak load reaches 125% of average load."""

    def plan(self, request):
        ranks = request.counts_by_source.shape[0]
        if request.total_rows == 0 or 4 * ranks * request.maximum_received < 5 * request.total_rows:
            return None
        target = (request.total_rows + ranks - 1) // ranks
        placement = build_placement(
            request.hidden, request.ids, request.counts_by_source, target, request.rank,
            source_inverse=request.source_inverse,
        )
        return replace(placement, local_receive_capacity=None) if placement is not None else None


def test_placement_checkpoint_and_close():
    """Verify default-cost placement, checkpoint gradients and resource cleanup."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device("npu", rank)
    try:
        torch.manual_seed(437 + rank)
        # This large hot route crosses the default cost gate without overriding policy.
        tokens, hidden, intermediate = 8192, 5120, 1792
        def sample(*dims: int) -> torch.Tensor:
            """Create deterministic BF16 samples on the local NPU."""
            return torch.randn(*dims, dtype=torch.bfloat16, device=device) * 0.02
        x = sample(tokens, hidden)
        probs = torch.rand(tokens, 2, device=device).softmax(-1)
        weights = (sample(4, hidden, 2 * intermediate), sample(4, intermediate, hidden))
        ids = torch.arange(2, dtype=torch.int32, device=device).expand(tokens, 2).contiguous()
        upstream = sample(tokens, hidden)
        reference = None
        for planner in (None, HotReplicaPlanner(), SkewPlanner()):
            replica = planner is not None
            op = MegaMoeExperts(hidden_size=hidden, intermediate_size=intermediate, num_experts=8, top_k=2,
                                ep_group=dist.group.WORLD,
                                expert_placement=(MegaMoePlacementOptions("adaptive", planner)
                                                  if replica else None),
                                numerics_options=MegaMoeNumericsOptions(apply_probs_before_fc2=True))
            def run(h: torch.Tensor, p: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
                """Pass current owner parameters through the ordinary expert API."""
                return op(h, ids, p, expert_weights=(w1, w2))
            leaves = [v.detach().requires_grad_() for v in (x, probs, *weights)]
            # Route and replica reuse are scoped to the checkpoint and parameter generation.
            wrapped = wrap_checkpoint(run, lambda: tuple(w._version for w in weights), prepare_replay=True) if replica else run
            out = checkpoint(wrapped, *leaves, use_reentrant=True) if replica else run(*leaves)
            out.backward(upstream)
            observed = [v.detach().cpu() for v in (out, *(t.grad for t in leaves))]
            if replica:
                owner = op._resource_group.resources.placement  # pylint: disable=protected-access
                assert owner.statistics["route_replays"] == 1
                counts = owner.pool.statistics
                assert counts["remote_prefetches"] + counts.get("fused_prefetches", 0) > 0, counts
                assert counts["remote_gradient_returns"] + counts.get("fused_gradient_returns", 0) > 0, counts
                for a, b in zip(observed, reference):
                    torch.testing.assert_close(a, b, rtol=.02, atol=.002)
            else:
                reference = observed
            if replica:
                placements = owner.statistics["placements"]
                small_ids = torch.arange(32, device=device, dtype=torch.int32).reshape(16, 2) % 8
                with torch.no_grad():
                    small = op(x[:16], small_ids, probs[:16], expert_weights=weights)
                assert small.shape == x[:16].shape and torch.isfinite(small).all()
                assert owner.statistics["placements"] == placements
            del out, leaves
            parent = weakref.ref(op._resource_group)  # pylint: disable=protected-access
            op.close()
            del op
            gc.collect()
            assert parent() is None
            if replica:
                assert owner.pool.closed
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_placement_checkpoint_and_close()
