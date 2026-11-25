# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
"""Whole-layer precision & performance benchmark:
MegaKda (CP=1, ulysses, allgather, prefix_sum) vs MindSpeed-MM native (eager & fla_npu ascendc).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import fla_npu.ops.ascendc as fa

from hyper_multicore.modules.mega_kda import MegaKda


def _load_chunk_kda_naive():
    path = "/home/qinhaibo/code/MindSpeed-MM/mindspeed_mm/fsdp/ops/kda/chunk_kda_naive.py"
    spec = importlib.util.spec_from_file_location("chunk_kda_naive", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.chunk_kda_naive


CHUNK_KDA_NAIVE = _load_chunk_kda_naive()


def ulysses_all_to_all(x: torch.Tensor, group, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    cp = dist.get_world_size(group)
    if cp == 1:
        return x
    chunks = [t.contiguous() for t in torch.chunk(x, cp, dim=scatter_dim)]
    in_buf = torch.stack(chunks, dim=0).contiguous()
    out_buf = torch.empty_like(in_buf)
    dist.all_to_all_single(out_buf, in_buf, group=group)
    return torch.cat(list(out_buf.unbind(0)), dim=gather_dim).contiguous()


def short_conv1d_native(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    x_t = x.transpose(1, 2)
    y = F.conv1d(x_t, weight, bias=None, padding=weight.shape[-1] - 1, groups=D)[..., :T]
    return F.silu(y.float()).to(x.dtype).transpose(1, 2).contiguous()


def mindspeed_mm_kda_layer_forward(
    module: MegaKda,
    hidden_states: torch.Tensor,
    *,
    backend: str = "ascendc",
    cp_group=None,
) -> torch.Tensor:
    cp_size = module.cp_size
    cp_rank = module.cp_rank
    B, S_local, _ = hidden_states.shape
    H = module.num_heads
    H_local = H // cp_size
    d = module.head_dim
    d_off = cp_rank * H_local * d
    d_len = H_local * d

    q = module.q_proj(hidden_states)
    k = module.k_proj(hidden_states)
    v = module.v_proj(hidden_states)
    g = module.f_b_proj(module.f_a_proj(hidden_states))
    beta = module.b_proj(hidden_states)

    if cp_size > 1:
        q = ulysses_all_to_all(q, cp_group, scatter_dim=2, gather_dim=1)
        k = ulysses_all_to_all(k, cp_group, scatter_dim=2, gather_dim=1)
        v = ulysses_all_to_all(v, cp_group, scatter_dim=2, gather_dim=1)
        g = ulysses_all_to_all(g, cp_group, scatter_dim=2, gather_dim=1)
        beta = ulysses_all_to_all(beta, cp_group, scatter_dim=2, gather_dim=1)

    w_conv_q = module.q_conv1d_weight[d_off : d_off + d_len]
    w_conv_k = module.k_conv1d_weight[d_off : d_off + d_len]
    w_conv_v = module.v_conv1d_weight[d_off : d_off + d_len]
    a_log = module.A_log[cp_rank * H_local : (cp_rank + 1) * H_local]
    dt_bias = module.dt_bias[d_off : d_off + d_len]

    q = short_conv1d_native(q, w_conv_q).view(B, -1, H_local, d)
    k = short_conv1d_native(k, w_conv_k).view(B, -1, H_local, d)
    v = short_conv1d_native(v, w_conv_v).view(B, -1, H_local, d)
    g = g.view(B, -1, H_local, d)
    beta = beta.float()

    if backend == "eager":
        o, _ = CHUNK_KDA_NAIVE(
            q=q, k=k, v=v, g=g, beta=beta,
            A_log=a_log, dt_bias=dt_bias,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=module.gate_lower_bound,
            chunk_size=module.chunk_size,
        )
    else:
        o = fa.chunk_kda_fwd(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=d ** -0.5,
            chunk_size=module.chunk_size,
            A_log=a_log, dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=(module.gate_lower_bound is not None),
            lower_bound=module.gate_lower_bound,
        )[0]

    if cp_size > 1:
        o = ulysses_all_to_all(o, cp_group, scatter_dim=1, gather_dim=2)

    if module.use_full_rank_gate:
        g_out = module.g_proj(hidden_states)
    else:
        g_out = module.g_b_proj(module.g_a_proj(hidden_states))
    g_out = g_out.view(B, S_local, H, d)
    o = torch_npu.npu_rms_norm(o, module.o_norm_weight, module.rms_norm_eps)[0]
    o = o * F.sigmoid(g_out)
    return module.o_proj(o.reshape(B, S_local, H * d))


def _measure_ms(fn, warmup: int, iters: int, sync_dist: bool) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    if sync_dist:
        dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    if sync_dist:
        dist.barrier()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _err_str(out: torch.Tensor, ref: torch.Tensor) -> str:
    diff = (out.float() - ref.float()).abs()
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    rel = diff.norm().item() / (ref.float().norm().item() + 1e-12)
    return f"max_abs={diff.max().item():.3e}  mean_abs={diff.mean().item():.3e}  rel_l2={rel:.3e}  cos={cos:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-full", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--no-lower-bound", action="store_true")
    parser.add_argument("--full-rank-gate", action="store_true")
    parser.add_argument("--cp-modes", type=str, default="auto",
                        help="Comma-separated cp modes to test (e.g. ulysses,allgather,prefix_sum)")
    parser.add_argument("--skip-eager", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)

    cp_group = None
    if world_size > 1:
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
        cp_group = dist.group.WORLD

    lower_bound = None if args.no_lower_bound else args.lower_bound
    torch.manual_seed(1234)
    module = MegaKda(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        use_full_rank_gate=args.full_rank_gate,
        gate_lower_bound=lower_bound,
        cp_size=world_size,
        cp_rank=rank,
        cp_group=cp_group,
        cp_mode="auto",
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.no_grad():
        for p in module.parameters():
            if p.dtype == torch.bfloat16:
                p.normal_(0.0, 0.02)
        module.o_norm_weight.fill_(1.0)
        module.A_log.copy_(torch.log(torch.empty_like(module.A_log).uniform_(1.0, 4.0)))
        module.dt_bias.normal_(0.0, 0.1)

    S_local = args.seq_full // world_size
    x_full = torch.randn(1, args.seq_full, args.hidden_size, device=device, dtype=torch.bfloat16) * 0.5
    x_local = x_full[:, rank * S_local : (rank + 1) * S_local, :].contiguous()

    modes = ["ulysses"] if world_size == 1 else [m.strip() for m in args.cp_modes.split(",") if m.strip()]

    with torch.no_grad():
        out_ascendc = mindspeed_mm_kda_layer_forward(module, x_local, backend="ascendc", cp_group=cp_group)
        out_eager = None if args.skip_eager else mindspeed_mm_kda_layer_forward(
            module, x_local, backend="eager", cp_group=cp_group
        )
        t_ascendc = _measure_ms(
            lambda: mindspeed_mm_kda_layer_forward(module, x_local, backend="ascendc", cp_group=cp_group),
            args.warmup, args.iters, sync_dist=(world_size > 1),
        )

    if rank == 0:
        print(
            f"\n=== MegaKda Whole-Layer Eval (CP={world_size}, S_full={args.seq_full}, "
            f"S_local={S_local}, D={args.hidden_size}, H={args.num_heads}, d={args.head_dim}, "
            f"lb={lower_bound}, full_gate={args.full_rank_gate}) ===",
            flush=True,
        )
        if out_eager is not None:
            print(f"  [Ref: fla_npu vs MindSpeed-MM eager] {_err_str(out_ascendc, out_eager)}", flush=True)
        print(f"  [Baseline: MindSpeed-MM fla_npu]     {t_ascendc:7.3f} ms", flush=True)

    with torch.no_grad():
        for mode in modes:
            out_mega = module(x_local, cp_mode=mode)
            torch.npu.synchronize()
            if world_size > 1:
                dist.barrier()
            t_mega = _measure_ms(
                lambda m=mode: module(x_local, cp_mode=m),
                args.warmup, args.iters, sync_dist=(world_size > 1),
            )
            if rank == 0:
                sp = t_ascendc / t_mega
                gain = (sp - 1.0) * 100.0
                print(f"  [MegaKda ({mode:10s})] {t_mega:7.3f} ms  (speedup = {sp:.2f}x, {gain:+.1f}% vs fla_npu)", flush=True)
                if out_eager is not None:
                    print(f"    vs eager  : {_err_str(out_mega, out_eager)}", flush=True)
                print(f"    vs fla_npu: {_err_str(out_mega, out_ascendc)}", flush=True)

    module.release_resources()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
