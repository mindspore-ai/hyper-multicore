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
"""Standalone LongCat Flash expert accuracy, benchmark, and profile validation worker."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from _precision import STRICT_GATE, compare_tensors
from native_reference import NativeLongcatExperts

import hyper_multicore
from hyper_multicore import shmem
from hyper_multicore.diagnostics import profiler as multicore_profiler
from hyper_multicore.longcat import LongcatFlashExperts


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject inconsistent dimensions before distributed startup."""
    args.real = 8 * args.ep if args.real is None else args.real
    args.zero = 4 * args.ep if args.zero is None else args.zero
    if min(args.ep, args.tokens, args.hidden, args.intermediate, args.real, args.top_k,
           args.updates, args.samples) <= 0 or min(args.zero, args.warmup) < 0:
        parser.error("invalid non-positive dimension, count or negative warmup")
    if args.real % args.ep or args.zero % args.ep or args.top_k > args.real + args.zero:
        parser.error("expert counts must divide EP and top-k must fit total experts")
    if args.initial_token_capacity is not None and args.initial_token_capacity <= 0:
        parser.error("initial-token-capacity must be positive")
    if args.route_fixture == "hot" and (args.real < 8 * args.ep or args.zero < 4 * args.ep or args.top_k != 12):
        parser.error("the hot fixture requires at least 8 real and 4 zero experts per rank and K=12")
    if args.check in ("accuracy", "profile") and args.backend == "native":
        parser.error(f"{args.check} requires push or pull backend")


def arguments() -> argparse.Namespace:
    """Parse a single-node EP LongCat validation configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("accuracy", "benchmark", "profile"), default="accuracy")
    parser.add_argument("--backend", choices=("native", "push", "pull"), default="push")
    parser.add_argument("--swiglu-precision", choices=("fused", "hf"), default="fused")
    parser.add_argument("--operator-timeout", type=int, default=60)
    parser.add_argument("--exercise-edge-routes", action="store_true",
                        help="Include hot owners, empty sources, and all-identity steps")
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--initial-token-capacity", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--real", type=int)
    parser.add_argument("--zero", type=int)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--no-gate-up-padded", dest="gate_up_padded", action="store_false", default=True)
    parser.add_argument("--route-fixture", choices=("router", "hot"), default="router")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _validate_arguments(parser, args)
    return args


def start_shmem_lifetime() -> None:
    """Choose a local SHMEM bootstrap endpoint after HCCL is ready."""
    endpoint = [None]
    if dist.get_rank() == 0:
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            endpoint[0] = f"tcp://127.0.0.1:{server.getsockname()[1]}"
    dist.broadcast_object_list(endpoint, src=0)
    os.environ["HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]


def hash_tensor(value: torch.Tensor) -> str:
    """Hash tensor bytes outside timed regions."""
    digest = hashlib.sha256()
    flattened = value.detach().reshape(-1)
    for start in range(0, flattened.numel(), 16 * 1024 * 1024):
        part = flattened[start : start + 16 * 1024 * 1024].cpu().contiguous().view(torch.uint8)
        digest.update(part.numpy().tobytes())
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    """Capture runtime environment and loaded native payload hashes."""
    source = Path(hyper_multicore.__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"],
                                         text=True, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, []
    mapped = sorted({
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
        if "/hyper_multicore/" in line and line.split()[-1].startswith("/")
    })
    libraries = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in mapped if Path(name).is_file()
    }
    return {
        "source": str(source),
        "git_revision": revision,
        "git_status": status,
        "python": platform.python_version(),
        "host_arch": platform.machine(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "chip": torch.npu.get_device_name(),
        "cann_root": os.environ.get("ASCEND_HOME_PATH"),
        "native_sha256": libraries,
    }


def fixture(args: argparse.Namespace, rank: int) -> dict[str, torch.Tensor]:
    """Generate deterministic owner-local EOI weights and router operands."""
    torch.npu.manual_seed(22000 + rank)
    local_real = args.real // args.ep
    local_zero = args.zero // args.ep if args.gate_up_padded else 0
    shape_gate = (local_real + local_zero, 2 * args.intermediate, args.hidden)
    shape_down = (local_real, args.hidden, args.intermediate)
    gate = torch.randn(shape_gate, device="npu", dtype=torch.bfloat16).mul_(0.02)
    down = torch.randn(shape_down, device="npu", dtype=torch.bfloat16).mul_(0.02)
    torch.manual_seed(12000)
    router = torch.nn.Linear(args.hidden, args.real + args.zero, bias=False).to("npu", torch.bfloat16)
    torch.npu.manual_seed(32000 + rank)
    hidden = torch.randn(args.tokens, args.hidden, device="npu", dtype=torch.bfloat16)
    with torch.no_grad():
        scores = F.linear(hidden.float(), router.weight.float()).softmax(-1)  # pylint: disable=not-callable
        ids = torch.topk(scores, k=args.top_k, dim=-1, sorted=False)[1]
        probability = scores.gather(1, ids) * 6.0
        if args.route_fixture == "hot":
            real = torch.arange(8, device="npu")
            identity = torch.arange(4, device="npu") + args.real
            ids = torch.cat((real, identity)).expand(args.tokens, -1).clone()
    upstream = torch.randn_like(hidden)
    return {
        "gate": gate.requires_grad_(),
        "down": down.requires_grad_(),
        "hidden": hidden.requires_grad_(),
        "ids": ids,
        "probability": probability.detach().requires_grad_(),
        "upstream": upstream,
    }


def candidate(args: argparse.Namespace) -> LongcatFlashExperts:
    """Construct the isolated LongcatFlashExperts executor."""
    return LongcatFlashExperts(
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_experts=args.real,
        zero_experts=args.zero,
        top_k=args.top_k,
        ep_size=args.ep,
        ep_group=dist.group.WORLD,
        dispatch_mode=args.backend,
        swiglu_precision=args.swiglu_precision,
        gate_up_padded=args.gate_up_padded,
        initial_token_capacity=args.initial_token_capacity,
    )


def routes(args: argparse.Namespace, data: dict[str, torch.Tensor], step: int, rank: int) -> tuple[str, torch.Tensor]:
    """Vary route shapes across steps including hot, empty_source, and all-identity steps."""
    if not args.exercise_edge_routes:
        return args.route_fixture, data["ids"]
    kinds = ("normal", "hot", "empty_source", "identity", "normal", "hot")
    kind = kinds[step % len(kinds)]
    if kind == "normal":
        return kind, data["ids"]
    local_real = args.real // args.ep
    real_take = min(local_real, args.top_k)
    zero_take = args.top_k - real_take
    owner = (step // 2) % args.ep
    real = torch.arange(real_take, device="npu") + owner * local_real
    identity = torch.arange(zero_take, device="npu") + args.real
    ids = torch.cat((real, identity)).expand(args.tokens, -1).clone()
    if (kind == "identity" or (kind == "empty_source" and rank == 0)) and args.zero >= args.top_k:
        ids = (torch.arange(args.top_k, device="npu") + args.real).expand(args.tokens, -1).clone()
    return kind, ids.long()


def execute(
    executor: Any,
    data: dict[str, torch.Tensor],
    ids: torch.Tensor,
    *,
    backward: bool,
) -> torch.Tensor:
    """Execute one forward (+ optional backward) pass with fresh gradients."""
    for key in ("gate", "down", "hidden", "probability"):
        data[key].grad = None
    with torch.set_grad_enabled(backward):
        output = executor(data["hidden"], ids, data["probability"], expert_weights=(data["gate"], data["down"]))
        if backward:
            output.backward(data["upstream"])
    return output


def values(data: dict[str, torch.Tensor], output: torch.Tensor) -> dict[str, torch.Tensor | None]:
    """Collect forward output and all input, router, and expert weight gradients."""
    return {
        "output": output,
        "input_gradient": data["hidden"].grad,
        "probability_gradient": data["probability"].grad,
        "gate_up_gradient": data["gate"].grad,
        "down_gradient": data["down"].grad,
    }


def compare(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Compare all tensors while preserving globally-unused None gradient semantics."""
    result = {}
    for name, value in actual.items():
        reference = expected[name]
        if value is None or reference is None:
            result[name] = {
                "strict_pass": value is None and reference is None,
                "both_unused": value is None and reference is None,
            }
        else:
            result[name] = compare_tensors(value, reference)
    return result


def _update_parameters(data: dict, other: dict, masters: list | None) -> bool:
    """Apply independent FP32 master updates and verify BF16 parameter hashes match."""
    if masters is None:
        return True
    with torch.no_grad():
        for index, item in enumerate((data, other)):
            for key in ("gate", "down"):
                if item[key].grad is not None:
                    masters[index][key].add_(item[key].grad.float(), alpha=-1e-5)
                    item[key].copy_(masters[index][key])
    return all(hash_tensor(data[key]) == hash_tensor(other[key]) for key in ("gate", "down"))


def accuracy(
    args: argparse.Namespace,
    data: dict[str, torch.Tensor],
    layer: LongcatFlashExperts,
    rank: int,
) -> dict[str, Any]:
    """Compare isolated LongcatFlashExperts against NativeLongcatExperts over updates."""
    reference = NativeLongcatExperts(
        args.real,
        args.zero,
        args.ep,
        dist.group.WORLD,
        args.swiglu_precision,
        gate_up_padded=args.gate_up_padded,
    )
    other = {
        key: value.detach().clone().requires_grad_(True) if key in ("gate", "down") else value
        for key, value in data.items()
    }
    masters = (
        [{key: item[key].detach().float().clone() for key in ("gate", "down")} for item in (data, other)]
        if args.updates > 1
        else None
    )
    records = []
    for step in range(args.updates):
        kind, ids = routes(args, data, step, rank)
        ref_output = execute(reference, data, ids, backward=True)
        expected = {
            name: None if value is None else value.detach().cpu()
            for name, value in values(data, ref_output).items()
        }
        out = execute(layer, other, ids, backward=True)
        comparison = compare(values(other, out), expected)
        match = _update_parameters(data, other, masters)
        local_pass = all(item["strict_pass"] for item in comparison.values()) and match
        state = torch.tensor([local_pass], dtype=torch.int32, device="npu")
        dist.all_reduce(state, op=dist.ReduceOp.MIN)
        records.append({
            "step": step,
            "routes": kind,
            "comparison": comparison,
            "parameters_match": match,
            "all_ranks_strict_pass": bool(state.item()),
        })
        if not records[-1]["all_ranks_strict_pass"]:
            break
    return {
        "records": records,
        "strict_pass": len(records) == args.updates and all(r["all_ranks_strict_pass"] for r in records),
    }


def benchmark(args: argparse.Namespace, data: dict[str, torch.Tensor], executor: Any) -> dict[str, Any]:
    """Measure synchronized forward+backward latency."""
    for _ in range(args.warmup):
        execute(executor, data, data["ids"], backward=True)
    torch.npu.synchronize()
    dist.barrier()
    local, maximum = [], []
    for _ in range(args.samples):
        start = time.perf_counter()
        out = execute(executor, data, data["ids"], backward=True)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        rank_max = torch.tensor([elapsed], device="npu")
        dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
        local.append(elapsed)
        maximum.append(float(rank_max.item()))
    finite = bool(torch.isfinite(out).all()) and all(
        data[key].grad is not None and bool(torch.isfinite(data[key].grad).all())
        for key in ("gate", "down", "hidden", "probability")
    )
    return {
        "local_ms": local,
        "rank_max_ms": maximum,
        "median_rank_max_ms": sorted(maximum)[len(maximum) // 2],
        "finite": finite,
    }


def profile(args: argparse.Namespace, data: dict[str, torch.Tensor], executor: Any) -> dict[str, Any]:
    """Capture internal megakernel Chrome trace."""
    for _ in range(args.warmup):
        execute(executor, data, data["ids"], backward=True)
    torch.npu.synchronize()
    dist.barrier()
    with multicore_profiler.mega_kernel_profile() as capture:
        for _ in range(args.samples):
            output = execute(executor, data, data["ids"], backward=True)
            capture.step()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = capture.export_chrome_trace(args.output / f"trace-rank{dist.get_rank()}.json")
    metadata = trace["megaKernelCycleTrace"]
    return {"finite": bool(torch.isfinite(output).all()), "profile": metadata}


def set_operator_timeout(seconds: int) -> None:
    """Bound per-process NPU operator timeout."""
    if seconds <= 0:
        raise ValueError("--operator-timeout must be positive")
    runtime = ctypes.CDLL("libascendcl.so")
    runtime.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
    runtime.aclrtSetOpExecuteTimeOut.restype = ctypes.c_int
    status = runtime.aclrtSetOpExecuteTimeOut(seconds)
    if status:
        raise RuntimeError(f"aclrtSetOpExecuteTimeOut failed: {status}")


def main() -> None:
    """Run one local EP worker and emit rank JSON evidence."""
    args = arguments()
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    set_operator_timeout(args.operator_timeout)
    dist.init_process_group("hccl")
    if dist.get_world_size() != args.ep or int(os.environ["LOCAL_WORLD_SIZE"]) != args.ep:
        raise ValueError("this worker requires single-node WORLD_SIZE == LOCAL_WORLD_SIZE == EP")
    start_shmem_lifetime()
    data = fixture(args, rank)
    layer = None if args.backend == "native" else candidate(args)
    executor = (
        layer
        if layer is not None
        else NativeLongcatExperts(
            args.real,
            args.zero,
            args.ep,
            dist.group.WORLD,
            args.swiglu_precision,
            gate_up_padded=args.gate_up_padded,
        )
    )
    if args.check == "accuracy":
        result = accuracy(args, data, layer, rank)
    elif args.check == "profile":
        result = profile(args, data, executor)
    else:
        result = benchmark(args, data, executor)

    if layer is not None:
        layer.close()
    torch.npu.synchronize()
    runtime_env = environment()
    final_shmem = shmem.debug_state()
    runtime_ok = final_shmem["state"] == "Uninitialized" and (
        layer is None or bool(runtime_env["native_sha256"])
    )
    report = {
        "rank": rank,
        "check": args.check,
        "backend": args.backend,
        "swiglu_precision": args.swiglu_precision,
        "gate_up_padded": args.gate_up_padded,
        "shape": {
            key: getattr(args, key)
            for key in ("ep", "tokens", "hidden", "intermediate", "real", "zero", "top_k")
        },
        "strict_gate": STRICT_GATE,
        "result": result,
        "shmem_after_close": final_shmem,
        "runtime_ok": runtime_ok,
        "environment": runtime_env,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()
    if not runtime_ok:
        raise RuntimeError("SHMEM cleanup or native library check failed")
    if not result.get("strict_pass", result.get("finite", False)):
        raise RuntimeError("LongCat Flash validation failed; inspect rank JSON reports")


if __name__ == "__main__":
    main()
