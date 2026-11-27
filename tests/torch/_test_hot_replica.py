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
from collections import defaultdict
from contextlib import nullcontext
import gc
from itertools import product
import json
import os
from pathlib import Path
import weakref

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions
from hyper_multicore.diagnostics import dump, profiler
from hyper_multicore.experimental import (
    BasePlanner, HotReplicaPlanner, MegaMoeTuningOptions, ReplayCheckpoint, wrap_checkpoint,
)
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica._planner.receive import select_replica_plan
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
                                   soft_target=target)
        self.last_placement = None if plan is None else as_placement(plan)
        return self.last_placement


def _check_guest_weight_waits(events, projection, gmm, home):
    waits = events["forward", f"ReplicaW{projection}WeightWait"]
    assert waits
    tasks = defaultdict(list)
    for task in events["forward", gmm]:
        tasks[task["block_id"], task["owner_id"]].append(task)
    for wait in waits:
        assert wait["rank"] == 0
        assert wait["owner_id"] == home + wait["stage_task_index"]
        assert any(task["start_cycle"] <= wait["start_cycle"] <= wait["end_cycle"] <= task["end_cycle"]
                   for task in tasks[wait["block_id"], wait["owner_id"]])


def _gradient_tail_cycles(events):
    task_end = max(args["end_cycle"] for (direction, name), group in events.items()
                   if direction == "backward" and not name.startswith("Replica") for args in group)
    replica_end = max(args["end_cycle"] for (direction, name), group in events.items()
                      if direction == "backward" and name.startswith("ReplicaDW") for args in group)
    return replica_end - task_end


def _check_replica_trace(trace, rank, copies_by_rank, home):
    """Check the trace's stage coverage and physical owner identifiers."""
    metadata = trace["megaKernelCycleTrace"]
    assert metadata["invocationCount"] == 2 and metadata["droppedRecordCount"] == 0
    events = defaultdict(list)
    for event in trace["traceEvents"]:
        if event["ph"] == "X":
            args = event["args"]
            events[args["direction"], event["name"]].append(args)
    width = home + len(copies_by_rank[rank])
    physical_owners = {rank * width + expert % home for expert in copies_by_rank[0] if expert >= 0}
    for projection, gmm in (("13", "GMM1"), ("2", "GMM2")):
        if rank == 0:
            _check_guest_weight_waits(events, projection, gmm, home)
            assert events["backward", f"ReplicaDW{projection}GuestAckWait"]
        else:
            waiting = events["backward", f"ReplicaDW{projection}LongestBandReadyWait"]
            assert waiting and all(args["rank"] == rank and args["owner_id"] in physical_owners for args in waiting)
            assert events["backward", f"ReplicaDW{projection}AckPublish"]
    return _gradient_tail_cycles(events)


def test_placement_diagnostics():
    """Check group agreement, staged snapshots, and both projections of the fused trace."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device("npu", rank)
    result_dir = Path(os.environ["HOT_REPLICA_TEST_RESULTS"])
    torch.manual_seed(519 + rank)
    tokens, hidden, intermediate, home = 256, 512, 256, 2
    ids = torch.arange(home, 2 * home, dtype=torch.int32, device=device).expand(tokens, -1).contiguous()
    inputs = tuple((torch.randn(shape, dtype=torch.bfloat16, device=device) * .02).requires_grad_()
                   for shape in ((tokens, hidden), (home, hidden, 2 * intermediate), (home, intermediate, hidden)))
    probs = torch.full((tokens, home), .5, dtype=torch.float32, device=device)
    upstream = torch.randn_like(inputs[0])

    def _train(op):
        output = op(inputs[0], ids, probs, expert_weights=inputs[1:])
        return (output, *torch.autograd.grad(output, inputs, upstream))

    records = []
    try:
        for mode in ("push", "pull"):
            planner = SkewPlanner()
            op = MegaMoeExperts(
                hidden_size=hidden, intermediate_size=intermediate, num_experts=2 * home, top_k=home,
                ep_group=dist.group.WORLD,
                tuning_options=MegaMoeTuningOptions(dispatch_mode=mode, expert_placement=planner),
                numerics_options=MegaMoeNumericsOptions(partial_weight_grad_dtype="float32"),
            )

            try:
                reference = _train(op)
                owner = op._resource_group.resources.placement  # pylint: disable=protected-access
                config = dump.TensorDumpConfig(result_dir / f"{mode}-dump", ranks=(0,), max_bytes_per_tensor=2048)
                with (dump.tensor_dump(config) if rank == 0 else nullcontext()):
                    with pytest.raises(RuntimeError, match="Tensor Dump mode differs across EP ranks"):
                        _train(op)
                assert owner.statistics["placements"] == 1
                with dump.tensor_dump(config) as dumper:
                    observed = _train(op)
                assert bool(dumper.last_manifest_paths) == (rank == 0)
                assert owner.statistics["staged_bindings"] == 1
                for got, expected in zip(observed, reference, strict=True):
                    torch.testing.assert_close(got, expected, rtol=.02, atol=.002)
                with profiler.mega_kernel_profile() as capture:
                    observed = _train(op)
                for got, expected in zip(observed, reference, strict=True):
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)
                trace = capture.export_chrome_trace(result_dir / f"{mode}-rank{rank}-trace.json")
                tail_cycles = _check_replica_trace(trace, rank, planner.last_placement.copies_by_rank, home)
                assert owner.statistics["placements"] == 3
                assert owner.statistics["fp32_band_bindings"] == 2
                records.append({"mode": mode, "replica_tail_cycles": tail_cycles})
            finally:
                op.close()
        (result_dir / f"rank{rank}-diagnostics.json").write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


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
            wrapped = wrap_checkpoint(
                run, lambda: tuple(w._version for w in weights),  # pylint: disable=protected-access
                prepare_replay=True,
            ) if replica else run
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


def test_nonreentrant_checkpoint_replay():
    """Compare non-reentrant replay with and without a supplied logical histogram."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        torch.manual_seed(817 + rank)
        device = torch.device("npu", rank)
        tokens, hidden, intermediate = 1024, 512, 256
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * .02
        ids = torch.arange(2, dtype=torch.int32, device=device).expand(tokens, 2).contiguous()
        counts = torch.tensor([tokens, tokens] + [0] * 6, dtype=torch.int32, device=device)
        probs = torch.rand(tokens, 2, device=device).softmax(-1)
        weights = tuple(torch.randn(shape, dtype=torch.bfloat16, device=device) * .02
                        for shape in ((4, hidden, 2 * intermediate), (4, intermediate, hidden)))
        upstream = torch.randn_like(x)
        reference = None
        receipts = []
        for supplied in (False, True):
            op = MegaMoeExperts(
                hidden_size=hidden, intermediate_size=intermediate, num_experts=8, top_k=2,
                ep_group=dist.group.WORLD, tuning_options=MegaMoeTuningOptions(expert_placement=SkewPlanner()),
            )

            def run(h, p, w1, w2, execute=op, use_counts=supplied):
                return execute(h, ids, p, expert_weights=(w1, w2),
                               tokens_per_expert=counts.clone() if use_counts else None)

            leaves = tuple(t.detach().requires_grad_() for t in (x, probs, *weights))
            tape = ReplayCheckpoint(lambda: 1)
            out = checkpoint(run, *leaves, use_reentrant=False, context_fn=tape.contexts)
            gradients = torch.autograd.grad(out, leaves, upstream)
            observed = tuple(t.detach().cpu() for t in (out, *gradients))
            owner = op._resource_group.resources.placement  # pylint: disable=protected-access
            assert owner.statistics["route_replays"] == 1 and owner.statistics["skipped_recomputes"] == 0
            assert owner.statistics["placements"] >= 1 and owner.pool is not None
            if reference is None:
                reference = observed
            else:
                for value, expected in zip(observed, reference, strict=True):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
            receipts.append({"route_replays": 1, "supplied_counts": supplied})
            del out, gradients, leaves, tape
            if supplied:
                inputs = x, probs, *weights
                receipts.extend(_check_checkpoint_count_rejections(op, inputs, ids, counts, rank))
            op.close()
        output = Path(os.environ["HOT_REPLICA_TEST_RESULTS"])
        (output / f"rank{rank}.json").write_text(json.dumps({"rank": rank, "cases": receipts}))
    finally:
        dist.destroy_process_group()


def _check_checkpoint_count_rejections(op, inputs, ids, counts, rank):
    """Reject rank-local counts errors before planning a replacement route."""
    placement = op._resource_group.resources.placement  # pylint: disable=protected-access
    receipts = []
    for name, in_replay, invalid_counts, error, message in (
        ("forward_shape", False, counts[:-1], ValueError, "tokens_per_expert"),
        ("replay_shape", True, counts[:-1], ValueError, "tokens_per_expert"),
        ("replay_values", True, counts.roll(1), RuntimeError, "original expert layout"),
    ):
        calls = 0
        before = placement.statistics["invocations"]
        leaves = tuple(value.detach().requires_grad_() for value in inputs)
        tape = ReplayCheckpoint(lambda: 1)

        def _run(hidden, probs, w1, w2, *, replay_error=in_replay, malformed_counts=invalid_counts):
            nonlocal calls
            replaying = calls > 0
            calls += 1
            local_counts = malformed_counts if rank == 1 and replaying == replay_error else counts
            return op(hidden, ids, probs, expert_weights=(w1, w2), tokens_per_expert=local_counts)

        if in_replay:
            output = checkpoint(_run, *leaves, use_reentrant=False, context_fn=tape.contexts)
            with pytest.raises(error, match=message):
                torch.autograd.grad(output, leaves, torch.ones_like(output))
            del output
        else:
            with pytest.raises(error, match=message):
                checkpoint(_run, *leaves, use_reentrant=False, context_fn=tape.contexts)
        assert calls == 1 + int(in_replay)
        assert placement.statistics["invocations"] == before + int(in_replay)
        receipts.append({"mode": name, "error": error.__name__})
    return receipts


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
        for mode, slots in product(("push", "pull"), (7, 8)):
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
