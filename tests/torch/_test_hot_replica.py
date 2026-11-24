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
import gc
import json
import os
from pathlib import Path
import weakref

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions
from hyper_multicore.experimental import BasePlanner, HotReplicaPlanner, MegaMoeTuningOptions, wrap_checkpoint
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica._planner.receive import SourceQuota, select_replica_plan
from hyper_multicore.modules.hot_replica.planner import as_placement


class SkewPlanner(BasePlanner):
    """Replicate when peak load reaches 125% of average load."""

    def plan(self, request):
        counts = request.counts_by_source
        ranks, experts = len(counts), len(counts[0])
        if request.total_rows == 0 or 4 * ranks * request.maximum_received < 5 * request.total_rows:
            return None
        target = (request.total_rows + ranks - 1) // ranks
        plan = select_replica_plan(counts, ExpertReplicaConfig(experts, ranks, experts // ranks),
                                   soft_target=target, source_quota=SourceQuota.RANK_ORDER)
        self.last_placement = None if plan is None else as_placement(plan)
        return self.last_placement


def test_placement_checkpoint_and_close():
    """Verify default-cost placement, checkpoint gradients and resource cleanup."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device("npu", rank)
    dispatch_mode = os.environ.get("HOT_REPLICA_DISPATCH_MODE", "push")
    layout = os.environ.get("HOT_REPLICA_WEIGHT_LAYOUT", "EIO")
    placement_records = []
    try:
        torch.manual_seed(437 + rank)
        # This large hot route crosses the default cost gate without overriding policy.
        tokens, hidden, intermediate = 12288, 5120, 1792
        def sample(*dims: int) -> torch.Tensor:
            """Create deterministic BF16 samples on the local NPU."""
            return torch.randn(*dims, dtype=torch.bfloat16, device=device) * 0.02
        x = sample(tokens, hidden)
        probs = torch.rand(tokens, 2, device=device).softmax(-1)
        weights = (sample(4, hidden, 2 * intermediate), sample(4, intermediate, hidden))
        if layout == "EOI":
            weights = tuple(weight.transpose(1, 2).contiguous() for weight in weights)
        ids = torch.arange(2, dtype=torch.int32, device=device).expand(tokens, 2).contiguous()
        upstream = sample(tokens, hidden)
        reference = None
        for planner in (None, HotReplicaPlanner(), SkewPlanner()):
            replica = planner is not None
            op = MegaMoeExperts(hidden_size=hidden, intermediate_size=intermediate, num_experts=8, top_k=2,
                                ep_group=dist.group.WORLD,
                                tuning_options=MegaMoeTuningOptions(
                                    dispatch_mode=dispatch_mode if replica else "push", expert_placement=planner,
                                ),
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
                assert owner.statistics["skipped_recomputes"] == 1
                assert owner.pool is not None, "The hot route must admit a replica"
                counts = owner.pool.statistics
                assert counts["remote_prefetches"] + counts.get("fused_prefetches", 0) > 0, counts
                assert counts["remote_gradient_returns"] + counts.get("fused_gradient_returns", 0) > 0, counts
                spec = owner.parent.spec
                assert (spec.dispatch_mode, spec.transport_engine, spec.expert_weight_layout) == (
                    dispatch_mode, "mte", layout,
                )
                assert spec.compact_pull == (dispatch_mode == "pull")
                placement_records.append({
                    "planner": type(planner).__name__, "route_replays": owner.statistics["route_replays"],
                    "skipped_recomputes": owner.statistics["skipped_recomputes"],
                    "bank": dict(counts),
                })
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
        if result_dir := os.environ.get("HOT_REPLICA_TEST_RESULTS"):
            (Path(result_dir) / f"rank{rank}.json").write_text(json.dumps({
                "rank": rank, "dispatch_mode": dispatch_mode, "layout": layout, "placements": placement_records,
            }))
    finally:
        dist.destroy_process_group()


def test_wide_placement_and_hard_receive_limit():
    """Force eight guest slots, then reject an impossible seven-slot plan."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device("npu", rank)
    tokens, hidden, intermediate, experts, top_k = 2048, 512, 256, 32, 8
    receipts = []
    try:
        torch.manual_seed(713 + rank)
        positions = torch.arange(tokens * top_k, device=device, dtype=torch.int32)
        ids = (positions % (experts // 2)).reshape(tokens, top_k)
        x = (torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16) * .02).requires_grad_()
        probs = torch.rand(tokens, top_k, device=device).softmax(-1).requires_grad_()
        weights = tuple((torch.randn(shape, device=device, dtype=torch.bfloat16) * .02).requires_grad_()
                        for shape in ((experts // 2, hidden, 2 * intermediate),
                                      (experts // 2, intermediate, hidden)))
        upstream = torch.randn_like(x)
        reference = None
        for mode in ("push", "pull"):
            for slots in (7, 8):
                planner = HotReplicaPlanner(replica_slots_per_rank=slots, receive_capacity_percent=100,
                                            cross_node_cost=1.5)
                op = MegaMoeExperts(
                    hidden_size=hidden, intermediate_size=intermediate, num_experts=experts, top_k=top_k,
                    ep_group=dist.group.WORLD,
                    tuning_options=MegaMoeTuningOptions(
                        dispatch_mode=mode, capacity_growth_factor=1.0, initial_capacity_factor=2.5,
                        expert_placement=planner,
                    ),
                    numerics_options=MegaMoeNumericsOptions(apply_probs_before_fc2=True),
                )
                try:
                    if slots == 7:
                        with pytest.raises(ValueError, match="receive limit"):
                            op(x, ids, probs, expert_weights=weights)
                        continue
                    out = op(x, ids, probs, expert_weights=weights)
                    gradients = torch.autograd.grad(out, (x, probs, *weights), upstream)
                    owner = op._resource_group.resources.placement  # pylint: disable=protected-access
                    geometry = owner.bounded.geometry
                    assert geometry.width == 8 and geometry.peak == tokens * top_k
                    assert owner.statistics["placements"] == 1
                    assert owner.pool.statistics["fused_gradient_returns"] == 1
                    received = tuple(t.detach().cpu() for t in (out, *gradients))
                    if reference is None:
                        reference = received
                    else:
                        for observed, expected in zip(received, reference, strict=True):
                            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                    receipts.append({"mode": mode, "actual_slots": geometry.width,
                                     "peak_rows": geometry.peak, "fused_returns": 1})
                finally:
                    op.close()
        if result_dir := os.environ.get("HOT_REPLICA_TEST_RESULTS"):
            (Path(result_dir) / f"rank{rank}.json").write_text(json.dumps({"rank": rank, "placements": receipts}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_placement_checkpoint_and_close()
