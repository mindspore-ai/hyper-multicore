# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Standalone MoE forward/backward example and rank-max latency measurement."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time


def _check_result(torch_module, output, hidden, probabilities, gate_up, down):
    """Check output and gradients after the timed device work."""
    if (not bool(torch_module.isfinite(output).all()) or hidden.grad is None or probabilities.grad is None
            or gate_up.grad is None or down.grad is None):
        raise RuntimeError("Invalid output or missing gradients")


def main(default_warmup: int = 0, default_iterations: int = 1) -> None:
    """Run under torchrun; defaults choose smoke versus benchmark iteration counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=default_warmup)
    parser.add_argument("--iterations", type=int, default=default_iterations)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if min(args.tokens, args.hidden_size, args.intermediate_size, args.experts, args.top_k, args.iterations) <= 0:
        parser.error("dimensions and iterations must be positive")
    if args.warmup < 0 or args.top_k > args.experts:
        parser.error("warmup must be nonnegative and top-k must not exceed experts")
    # Framework imports follow argument parsing so --help needs no NPU installation.
    import torch  # pylint: disable=import-outside-toplevel
    import torch.distributed as dist  # pylint: disable=import-outside-toplevel
    import torch_npu  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    from hyper_multicore import MegaMoeExperts, shmem  # pylint: disable=import-outside-toplevel
    from hyper_multicore._loader import get_multicore_paths  # pylint: disable=import-outside-toplevel

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.experts % world:
        raise ValueError("experts must be divisible by world size")
    device = torch.device("npu", local_rank)
    torch.manual_seed(2026)
    layer = None
    shmem.acquire()
    try:
        layer = MegaMoeExperts(
            hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
            num_experts=args.experts, top_k=args.top_k, ep_group=dist.group.WORLD,
        ).to(device=device, dtype=torch.bfloat16)
        local_experts = args.experts // world
        gate_up = torch.nn.Parameter(torch.empty(local_experts, args.hidden_size, args.intermediate_size * 2,
                                                device=device, dtype=torch.bfloat16))
        down = torch.nn.Parameter(torch.empty(local_experts, args.intermediate_size, args.hidden_size,
                                             device=device, dtype=torch.bfloat16))
        for expert in range(local_experts):
            torch.nn.init.xavier_uniform_(gate_up[expert])
            torch.nn.init.xavier_uniform_(down[expert])
        hidden = torch.randn(args.tokens, args.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
        ids = (torch.arange(args.tokens * args.top_k, device=device).view(args.tokens, args.top_k)
               .remainder(args.experts).to(torch.int32))
        probabilities = torch.full((args.tokens, args.top_k), 1 / args.top_k, device=device, requires_grad=True)
        samples = []
        for iteration in range(args.warmup + args.iterations):
            gate_up.grad = down.grad = None
            hidden.grad = None
            probabilities.grad = None
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            output = layer(hidden, ids, probabilities, expert_weights=(gate_up, down))
            output.float().square().mean().backward()
            torch.npu.synchronize()
            elapsed = torch.tensor((time.perf_counter() - start) * 1000, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            _check_result(torch, output, hidden, probabilities, gate_up, down)
            if iteration >= args.warmup:
                samples.append(float(elapsed.cpu()))
        if rank == 0:
            vendor, _ = get_multicore_paths()
            info = json.loads((vendor.parents[2] / "build_info.json").read_text(encoding="utf-8"))
            result = {"world_size": world,
                      "shape": {key: value for key, value in vars(args).items() if key != "result"},
                      "rank_max_fwd_bwd_ms": samples, "median_ms": statistics.median(samples),
                      "build_info": info, "performance_acceptance": False}
            print(json.dumps(result, indent=2))
            if args.result:
                args.result.parent.mkdir(parents=True, exist_ok=True)
                args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        if layer is not None:
            layer.close()
        shmem.release()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
