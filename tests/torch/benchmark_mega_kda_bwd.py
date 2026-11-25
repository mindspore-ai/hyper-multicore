# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
"""Whole-layer Forward + Backward precision & performance benchmark:
MegaKda + HyperMegaKdaGrad (CP=1, ulysses, allgather, prefix_sum, auto)
vs MindSpeed-MM native eager (chunk_kda_naive autograd) & fla_npu AscendC (chunk_kda_fwd + chunk_kda_bwd).
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


class _UlyssesA2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group, scatter_dim: int, gather_dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        cp = dist.get_world_size(group)
        if cp == 1:
            return x
        chunks = [t.contiguous() for t in torch.chunk(x, cp, dim=scatter_dim)]
        in_buf = torch.stack(chunks, dim=0).contiguous()
        out_buf = torch.empty_like(in_buf)
        dist.all_to_all_single(out_buf, in_buf, group=group)
        return torch.cat(list(out_buf.unbind(0)), dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _UlyssesA2A.apply(grad_out, ctx.group, ctx.gather_dim, ctx.scatter_dim), None, None, None


class _FlaNpuChunkKdaFn(torch.autograd.Function):
    """Fastest fla_npu AscendC chunk_kda_fwd + chunk_kda_bwd (disable_recompute=True)."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, chunk_size):
        B, S, H, d = q.shape
        device = q.device
        q_hat = torch.empty(B, H, S, d, device=device, dtype=torch.bfloat16)
        k_hat = torch.empty(B, H, S, d, device=device, dtype=torch.bfloat16)
        q_rstd = torch.empty(B, H, S, device=device, dtype=torch.float32)
        k_rstd = torch.empty(B, H, S, device=device, dtype=torch.float32)
        beta_eff = torch.empty(B, H, S, device=device, dtype=torch.float32)

        res = fa.chunk_kda_fwd(
            q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), g=g.contiguous(), beta=beta.contiguous(),
            scale=scale, chunk_size=chunk_size, layout="BSND",
            A_log=A_log.contiguous(), dt_bias=dt_bias.contiguous(),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=(lower_bound is not None),
            lower_bound=(lower_bound if lower_bound is not None else -5.0),
            disable_recompute=True,
            q_hat_out=q_hat, k_hat_out=k_hat,
            q_rstd_out=q_rstd, k_rstd_out=k_rstd,
            beta_eff_out=beta_eff,
        )
        o, _, gk, Aqk, Akk, w, _, qg, kg, v_new, h, _ = res
        ctx.save_for_backward(
            q_hat, k_hat, v.contiguous(), beta.contiguous(), g.contiguous(),
            A_log.contiguous(), dt_bias.contiguous(),
            gk, Aqk, Akk, w, qg, kg, v_new, h, q_rstd, k_rstd, beta_eff,
        )
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.chunk_size = chunk_size
        return o

    @staticmethod
    def backward(ctx, do):
        (
            q_hat, k_hat, v, beta, raw_g, A_log, dt_bias,
            gk, Aqk, Akk, w, qg, kg, v_new, h, q_rstd, k_rstd, beta_eff,
        ) = ctx.saved_tensors
        dq, dk, dv, db, dg, _, dA, dbias = fa.chunk_kda_bwd(
            q_hat, k_hat, v, beta,
            gk, Aqk, Akk, w, qg, kg, v_new, h,
            do.contiguous(), float(ctx.scale),
            raw_g=raw_g, A_log=A_log, dt_bias=dt_bias,
            chunk_size=ctx.chunk_size,
            safe_gate=(ctx.lower_bound is not None),
            lower_bound=(ctx.lower_bound if ctx.lower_bound is not None else -5.0),
            use_gate_in_kernel=True,
            disable_recompute=True,
            use_exp2=True,
            state_v_first=False,
            q_rstd=q_rstd, k_rstd=k_rstd,
            layout="BSND",
            use_beta_sigmoid_in_kernel=True,
            beta_eff=beta_eff,
        )
        return dq, dk, dv, dg, db, dA, dbias, None, None, None


def _short_conv1d_diff(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    y = F.conv1d(x.transpose(1, 2), weight, bias=None, padding=weight.shape[-1] - 1, groups=D)[..., :T]
    return F.silu(y.float()).to(x.dtype).transpose(1, 2).contiguous()


class _ConvTailExchange(torch.autograd.Function):
    """Standard MindSpeed-MM Conv1d CP causal-boundary exchange via all_gather
    (matches mindspeed_mm/fsdp/distributed/context_parallel/pipeline_comm.py)."""

    @staticmethod
    def forward(ctx, tail_to_send: torch.Tensor, group, is_first: bool, is_last: bool):
        ctx.group = group
        ctx.is_first = is_first
        ctx.is_last = is_last
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        gathered = torch.empty(
            (world_size,) + tuple(tail_to_send.shape),
            dtype=tail_to_send.dtype,
            device=tail_to_send.device,
        )
        dist.all_gather_into_tensor(gathered, tail_to_send.contiguous(), group=group)
        if is_first:
            return torch.zeros_like(tail_to_send)
        return gathered[rank - 1].clone()

    @staticmethod
    def backward(ctx, grad_recv_tail: torch.Tensor):
        rank = dist.get_rank(ctx.group)
        world_size = dist.get_world_size(ctx.group)
        gathered_grad = torch.empty(
            (world_size,) + tuple(grad_recv_tail.shape),
            dtype=grad_recv_tail.dtype,
            device=grad_recv_tail.device,
        )
        dist.all_gather_into_tensor(gathered_grad, grad_recv_tail.contiguous(), group=ctx.group)
        if ctx.is_last:
            return torch.zeros_like(grad_recv_tail), None, None, None
        return gathered_grad[rank + 1].clone(), None, None, None


_EYE_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _get_eye(k: int, device: torch.device) -> torch.Tensor:
    key = (k, device)
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(k, device=device, dtype=torch.float32)
        _EYE_CACHE[key] = eye
    return eye


def _kda_compute_a_p_b_p(
    kg: torch.Tensor, w: torch.Tensor, u: torch.Tensor, gk: torch.Tensor, chunk_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard MindSpeed-MM parallel_scan.py compute_a_p_b_p adapted for KDA (vector gk, exp2).
    Inputs are head-major [B, H, T, D] from fla_npu chunk_kda_fwd_prepare (zero recompute of kg)."""
    B, H, T, K = kg.shape
    V = u.shape[-1]
    BT = chunk_size
    assert T % BT == 0
    NT = T // BT
    eye_K = _get_eye(K, kg.device)

    kg_c = kg.float().reshape(B, H, NT, BT, K)
    w_c = w.float().reshape(B, H, NT, BT, K)
    u_c = u.float().reshape(B, H, NT, BT, V)
    alpha = torch.exp2(gk[:, :, BT - 1 :: BT, :].float())  # [B, H, NT, K]

    At = torch.diag_embed(alpha) - torch.matmul(kg_c.transpose(-1, -2), w_c)  # [B, H, NT, K, K]
    Bt = torch.matmul(kg_c.transpose(-1, -2), u_c)                             # [B, H, NT, K, V]
    del kg_c, w_c, u_c, alpha

    A_p = eye_K.expand(B, H, K, K).clone()
    B_p = torch.zeros(B, H, K, V, device=kg.device, dtype=torch.float32)
    for t in range(NT):
        At_t = At[:, :, t]
        Bt_t = Bt[:, :, t]
        B_p = torch.baddbmm(
            Bt_t.reshape(-1, K, V), At_t.reshape(-1, K, K), B_p.reshape(-1, K, V)
        ).view(B, H, K, V)
        A_p = torch.matmul(At_t, A_p)
    return A_p, B_p


def _kda_compute_da_p_db_p(
    qg_scaled: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    do_bnsd: torch.Tensor,
    dv_local: torch.Tensor,
    gk: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard MindSpeed-MM parallel_scan.py compute_da_p_db_p adapted for KDA (vector gk, exp2).
    Inputs are head-major [B, H, T, D] from fwd_prepare & bwd_prepare."""
    B, H, T, K = kg.shape
    V = do_bnsd.shape[-1]
    BT = chunk_size
    assert T % BT == 0
    NT = T // BT
    eye_K = _get_eye(K, kg.device)

    qg_c = qg_scaled.float().reshape(B, H, NT, BT, K)
    kg_c = kg.float().reshape(B, H, NT, BT, K)
    w_c = w.float().reshape(B, H, NT, BT, K)
    do_c = do_bnsd.float().reshape(B, H, NT, BT, V)
    dv_c = dv_local.float().reshape(B, H, NT, BT, V)
    alpha = torch.exp2(gk[:, :, BT - 1 :: BT, :].float())  # [B, H, NT, K]

    dM = torch.diag_embed(alpha) - torch.matmul(w_c.transpose(-1, -2), kg_c)  # [B, H, NT, K, K]
    db = torch.matmul(qg_c.transpose(-1, -2), do_c) - torch.matmul(w_c.transpose(-1, -2), dv_c)
    del qg_c, kg_c, w_c, do_c, dv_c, alpha

    dA_p = eye_K.expand(B, H, K, K).clone()
    dB_p = torch.zeros(B, H, K, V, device=kg.device, dtype=torch.float32)
    for t in range(NT - 1, -1, -1):
        dM_t = dM[:, :, t]
        db_t = db[:, :, t]
        dB_p = torch.baddbmm(
            db_t.reshape(-1, K, V), dM_t.reshape(-1, K, K), dB_p.reshape(-1, K, V)
        ).view(B, H, K, V)
        dA_p = torch.matmul(dM_t, dA_p)
    return dA_p, dB_p


def _kda_prefix_scan_compose_fwd(all_hm: torch.Tensor, cp_rank: int) -> torch.Tensor | None:
    if cp_rank == 0:
        return None
    _, B, H, K, KV = all_hm.shape
    V = KV - K
    h = all_hm[0, ..., K:].contiguous()
    for j in range(1, cp_rank):
        A_j = all_hm[j, ..., :K].contiguous()
        B_j = all_hm[j, ..., K:].contiguous()
        h = torch.baddbmm(
            B_j.reshape(-1, K, V), A_j.reshape(-1, K, K), h.reshape(-1, K, V)
        ).view(B, H, K, V)
    return h


def _kda_prefix_scan_compose_bwd(all_dhm: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor | None:
    if cp_rank == cp_size - 1:
        return None
    _, B, H, K, KV = all_dhm.shape
    V = KV - K
    dht = all_dhm[cp_size - 1, ..., K:].contiguous()
    for j in range(cp_size - 2, cp_rank, -1):
        dA_j = all_dhm[j, ..., :K].contiguous()
        dB_j = all_dhm[j, ..., K:].contiguous()
        dht = torch.baddbmm(
            dB_j.reshape(-1, K, V), dA_j.reshape(-1, K, K), dht.reshape(-1, K, V)
        ).view(B, H, K, V)
    return dht


class _FlaNpuChunkKdaAllGatherFn(torch.autograd.Function):
    """Standard MindSpeed-MM kvallgather scan CP + fla_npu 3-stage AscendC KDA."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, chunk_size, cp_group, cp_rank, cp_size):
        B, S, H, d = q.shape
        device = q.device
        safe_gate = lower_bound is not None
        lb = lower_bound if safe_gate else -5.0

        gk, Aqk, Akk, w, u, qg, kg, qg_scaled, q_hat, k_hat, q_rstd, k_rstd, beta_eff = fa.chunk_kda_fwd_prepare(
            q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(),
            float(scale), chunk_size,
            layout="BSND", safe_gate=safe_gate, lower_bound=lb,
            use_gate_in_kernel=True, A_log=A_log.contiguous(), dt_bias=dt_bias.contiguous(),
            use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            use_exp2=True, backward_mode="save",
        )

        h_init = None
        if cp_size > 1:
            if cp_rank < cp_size - 1:
                A_p, B_p = _kda_compute_a_p_b_p(kg, w, u, gk, chunk_size=chunk_size)
                hm = torch.cat([A_p, B_p], dim=-1).contiguous()
            else:
                hm = torch.zeros(B, H, d, 2 * d, device=device, dtype=torch.float32)
            all_hm = torch.empty((cp_size, B, H, d, 2 * d), device=device, dtype=torch.float32)
            dist.all_gather_into_tensor(all_hm, hm, group=cp_group)
            h_init = _kda_prefix_scan_compose_fwd(all_hm, cp_rank)

        h, v_new, _ = fa.chunk_fwd_h(
            kg, w, u, gk=gk, initial_state=h_init,
            output_final_state=False, chunk_size=chunk_size, save_new_value=True,
            use_exp2=True, state_v_first=False,
        )
        o = fa.chunk_kda_fwd_finalize(
            qg_scaled, Aqk, v_new, h, output_layout="BSND", state_v_first=False,
        )

        ctx.save_for_backward(
            q_hat, k_hat, v.contiguous(), g.contiguous(), A_log.contiguous(), dt_bias.contiguous(),
            gk, Aqk, Akk, w, qg, kg, qg_scaled, v_new, h, q_rstd, k_rstd, beta_eff,
            h_init if h_init is not None else torch.empty(0, device=device),
        )
        ctx.has_h_init = h_init is not None
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.chunk_size = chunk_size
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        return o

    @staticmethod
    def backward(ctx, do):
        (
            q_hat, k_hat, v, raw_g, A_log, dt_bias,
            gk, Aqk, Akk, w, qg, kg, qg_scaled, v_new, h, q_rstd, k_rstd, beta_eff,
            h_init_buf,
        ) = ctx.saved_tensors
        h_init = h_init_buf if ctx.has_h_init else None
        B, S, H, d = v.shape
        device = v.device
        safe_gate = ctx.lower_bound is not None
        lb = ctx.lower_bound if safe_gate else -5.0

        do_bnsd = do.transpose(1, 2).contiguous()
        dAqk, dv_local, dq_raw = fa.chunk_kda_bwd_prepare(
            Aqk, v_new, do_bnsd, h,
            scale=float(ctx.scale), layout="BNSD", chunk_size=ctx.chunk_size, state_v_first=False,
        )

        dht = None
        if ctx.cp_size > 1:
            if ctx.cp_rank > 0:
                dA_p, dB_p = _kda_compute_da_p_db_p(
                    qg_scaled, kg, w, do_bnsd, dv_local, gk, chunk_size=ctx.chunk_size,
                )
                dhm = torch.cat([dA_p, dB_p], dim=-1).contiguous()
            else:
                dhm = torch.zeros(B, H, d, 2 * d, device=device, dtype=torch.float32)
            all_dhm = torch.empty((ctx.cp_size, B, H, d, 2 * d), device=device, dtype=torch.float32)
            dist.all_gather_into_tensor(all_dhm, dhm, group=ctx.cp_group)
            dht = _kda_prefix_scan_compose_bwd(all_dhm, ctx.cp_rank, ctx.cp_size)

        dh, _, dv_scan = fa.chunk_gated_delta_rule_bwd_dhu(
            qg, kg, w, do_bnsd, dv_local, float(ctx.scale), ctx.chunk_size,
            gK=gk, h0=None, dht=(dht.to(qg.dtype) if dht is not None else None), use_exp2=True, transpose_state_layout=False,
        )
        dq, dk, dv, db, dg, dA, dbias = fa.chunk_kda_bwd_finalize(
            q_hat, k_hat, v.transpose(1, 2).contiguous(),
            gk, raw_g.transpose(1, 2).contiguous(),
            beta_eff, A_log, dt_bias, Akk, v_new, h, dh,
            dv_scan, dAqk, dq_raw,
            scale=float(ctx.scale), lower_bound=lb, chunk_size=ctx.chunk_size,
            safe_gate=safe_gate, use_gate_in_kernel=True, use_exp2=True, state_v_first=False,
            q_rstd=q_rstd, k_rstd=k_rstd,
            use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False, output_layout="BSND",
        )
        return dq, dk, dv, dg, db, dA, dbias.view_as(dt_bias), None, None, None, None, None, None


def _short_conv1d_with_halo_diff(x: torch.Tensor, recv_tail: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # recv_tail: [B, W-1, D], x: [B, T, D] -> pad=0 gives exact causal conv of length T
    x_pad = torch.cat([recv_tail, x], dim=1)
    y = F.conv1d(x_pad.transpose(1, 2), weight, bias=None, padding=0, groups=x.shape[-1])
    return F.silu(y.float()).to(x.dtype).transpose(1, 2).contiguous()


def mindspeed_mm_kda_layer_allgather_diff(
    module: MegaKda,
    hidden_states: torch.Tensor,
    *,
    cp_group=None,
) -> torch.Tensor:
    """Standard MindSpeed-MM kvallgather scan CP whole-layer baseline (no Ulysses)."""
    cp_size, cp_rank = module.cp_size, module.cp_rank
    B, S_local, _ = hidden_states.shape
    H, d = module.num_heads, module.head_dim
    D = H * d
    W = module.q_conv1d_weight.shape[-1]

    q = module.q_proj(hidden_states)
    k = module.k_proj(hidden_states)
    v = module.v_proj(hidden_states)
    g = module.f_b_proj(module.f_a_proj(hidden_states))
    beta = module.b_proj(hidden_states)

    if cp_size > 1:
        tail_send = torch.cat([q[:, -(W - 1):, :], k[:, -(W - 1):, :], v[:, -(W - 1):, :]], dim=-1).contiguous()
        tail_recv = _ConvTailExchange.apply(tail_send, cp_group, cp_rank == 0, cp_rank == cp_size - 1)
        q_recv, k_recv, v_recv = tail_recv.split(D, dim=-1)
        q = _short_conv1d_with_halo_diff(q, q_recv, module.q_conv1d_weight).view(B, S_local, H, d)
        k = _short_conv1d_with_halo_diff(k, k_recv, module.k_conv1d_weight).view(B, S_local, H, d)
        v = _short_conv1d_with_halo_diff(v, v_recv, module.v_conv1d_weight).view(B, S_local, H, d)
    else:
        q = _short_conv1d_diff(q, module.q_conv1d_weight).view(B, S_local, H, d)
        k = _short_conv1d_diff(k, module.k_conv1d_weight).view(B, S_local, H, d)
        v = _short_conv1d_diff(v, module.v_conv1d_weight).view(B, S_local, H, d)

    g = g.view(B, S_local, H, d)
    beta = beta.float()

    o = _FlaNpuChunkKdaAllGatherFn.apply(
        q, k, v, g, beta, module.A_log, module.dt_bias,
        module.scale, module.gate_lower_bound, module.chunk_size,
        cp_group, cp_rank, cp_size,
    )

    if module.use_full_rank_gate:
        g_out = module.g_proj(hidden_states)
    else:
        g_out = module.g_b_proj(module.g_a_proj(hidden_states))
    g_out = g_out.view(B, S_local, H, d)
    o = torch_npu.npu_rms_norm(o, module.o_norm_weight, module.rms_norm_eps)[0]
    o = o * F.sigmoid(g_out)
    return module.o_proj(o.reshape(B, S_local, D))


def mindspeed_mm_kda_layer_diff(
    module: MegaKda,
    hidden_states: torch.Tensor,
    *,
    backend: str = "ascendc",
    cp_group=None,
) -> torch.Tensor:
    cp_size, cp_rank = module.cp_size, module.cp_rank
    B, S_local, _ = hidden_states.shape
    H, d = module.num_heads, module.head_dim
    H_local = H // cp_size
    d_off, d_len = cp_rank * H_local * d, H_local * d

    q = module.q_proj(hidden_states)
    k = module.k_proj(hidden_states)
    v = module.v_proj(hidden_states)
    g = module.f_b_proj(module.f_a_proj(hidden_states))
    beta = module.b_proj(hidden_states)

    if cp_size > 1:
        q = _UlyssesA2A.apply(q, cp_group, 2, 1)
        k = _UlyssesA2A.apply(k, cp_group, 2, 1)
        v = _UlyssesA2A.apply(v, cp_group, 2, 1)
        g = _UlyssesA2A.apply(g, cp_group, 2, 1)
        beta = _UlyssesA2A.apply(beta, cp_group, 2, 1)

    w_cq = module.q_conv1d_weight[d_off : d_off + d_len]
    w_ck = module.k_conv1d_weight[d_off : d_off + d_len]
    w_cv = module.v_conv1d_weight[d_off : d_off + d_len]
    a_log = module.A_log[cp_rank * H_local : (cp_rank + 1) * H_local]
    dt_bias = module.dt_bias[d_off : d_off + d_len]

    q = _short_conv1d_diff(q, w_cq).view(B, -1, H_local, d)
    k = _short_conv1d_diff(k, w_ck).view(B, -1, H_local, d)
    v = _short_conv1d_diff(v, w_cv).view(B, -1, H_local, d)
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
        o = _FlaNpuChunkKdaFn.apply(
            q, k, v, g, beta, a_log, dt_bias,
            module.scale, module.gate_lower_bound, module.chunk_size,
        )

    if cp_size > 1:
        o = _UlyssesA2A.apply(o, cp_group, 1, 2)

    if module.use_full_rank_gate:
        g_out = module.g_proj(hidden_states)
    else:
        g_out = module.g_b_proj(module.g_a_proj(hidden_states))
    g_out = g_out.view(B, S_local, H, d)
    o = torch_npu.npu_rms_norm(o, module.o_norm_weight, module.rms_norm_eps)[0]
    o = o * F.sigmoid(g_out)
    return module.o_proj(o.reshape(B, S_local, H * d))


def _err_str(out: torch.Tensor, ref: torch.Tensor) -> str:
    diff = (out.float() - ref.float()).abs()
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    rel = diff.norm().item() / (ref.float().norm().item() + 1e-12)
    return f"max_abs={diff.max().item():.3e}  rel_l2={rel:.3e}  cos={cos:.8f}"


def _run_fwd_bwd_and_collect(fn_fwd, module: MegaKda, x_local: torch.Tensor, dy_local: torch.Tensor, world_size: int):
    x = x_local.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    y = fn_fwd(x)
    y.backward(dy_local)
    grads = {"dx": x.grad.detach().clone()}
    for name, p in module.named_parameters():
        g = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        if world_size > 1:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
        grads[name] = g
    return y.detach().clone(), grads


def _measure_fwd_bwd_ms(fn_fwd, module: MegaKda, x_local: torch.Tensor, dy_local: torch.Tensor,
                        warmup: int, iters: int, sync_dist: bool) -> tuple[float, float, float]:
    # Warmup
    for _ in range(warmup):
        x = x_local.detach().requires_grad_(True)
        module.zero_grad(set_to_none=True)
        y = fn_fwd(x)
        y.backward(dy_local)
    torch.npu.synchronize()
    if sync_dist:
        dist.barrier()

    t_fwd_acc = 0.0
    t_bwd_acc = 0.0
    for _ in range(iters):
        x = x_local.detach().requires_grad_(True)
        module.zero_grad(set_to_none=True)
        torch.npu.synchronize()
        if sync_dist:
            dist.barrier()
        t0 = time.perf_counter()
        y = fn_fwd(x)
        torch.npu.synchronize()
        if sync_dist:
            dist.barrier()
        t1 = time.perf_counter()
        y.backward(dy_local)
        torch.npu.synchronize()
        if sync_dist:
            dist.barrier()
        t2 = time.perf_counter()
        t_fwd_acc += (t1 - t0) * 1000.0
        t_bwd_acc += (t2 - t1) * 1000.0

    tf = t_fwd_acc / iters
    tb = t_bwd_acc / iters
    return tf, tb, tf + tb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-full", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--no-lower-bound", action="store_true")
    parser.add_argument("--full-rank-gate", action="store_true")
    parser.add_argument("--cp-modes", type=str, default="auto")
    parser.add_argument("--baseline-cp", type=str, default="ulysses", choices=["ulysses", "allgather"])
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
    dy_full = torch.randn(1, args.seq_full, args.hidden_size, device=device, dtype=torch.bfloat16) * 0.5
    x_local = x_full[:, rank * S_local : (rank + 1) * S_local, :].contiguous()
    dy_local = dy_full[:, rank * S_local : (rank + 1) * S_local, :].contiguous()

    modes = ["auto"] if world_size == 1 else [m.strip() for m in args.cp_modes.split(",") if m.strip()]

    # Reference 1: MindSpeed-MM Eager (chunk_kda_naive autograd)
    y_eager, g_eager = (None, None)
    if not args.skip_eager:
        y_eager, g_eager = _run_fwd_bwd_and_collect(
            lambda inp: mindspeed_mm_kda_layer_diff(module, inp, backend="eager", cp_group=cp_group),
            module, x_local, dy_local, world_size,
        )

    # Reference 2: MindSpeed-MM fla_npu AscendC (AllGather scan CP or Ulysses CP)
    if args.baseline_cp == "allgather" and world_size > 1:
        base_fn = lambda inp: mindspeed_mm_kda_layer_allgather_diff(module, inp, cp_group=cp_group)
        base_label = "MindSpeed-MM + fla_npu (allgather CP)"
    else:
        base_fn = lambda inp: mindspeed_mm_kda_layer_diff(module, inp, backend="ascendc", cp_group=cp_group)
        base_label = "MindSpeed-MM + fla_npu (ulysses CP)" if world_size > 1 else "fla_npu (no_recomp)"

    y_fla, g_fla = _run_fwd_bwd_and_collect(
        base_fn,
        module, x_local, dy_local, world_size,
    )
    tf_fla, tb_fla, tt_fla = _measure_fwd_bwd_ms(
        base_fn,
        module, x_local, dy_local, args.warmup, args.iters, sync_dist=(world_size > 1),
    )

    if rank == 0:
        print(
            f"\n=== MegaKda Whole-Layer FWD+BWD Eval (CP={world_size}, S_full={args.seq_full}, "
            f"S_local={S_local}, D={args.hidden_size}, H={args.num_heads}, d={args.head_dim}, "
            f"lb={lower_bound}, full_gate={args.full_rank_gate}) ===",
            flush=True,
        )
        print(
            f"  [Baseline: {base_label}] fwd={tf_fla:6.3f} ms | bwd={tb_fla:6.3f} ms | tot={tt_fla:6.3f} ms",
            flush=True,
        )
        if g_eager is not None:
            print(f"  [Ref fla vs eager] out: {_err_str(y_fla, y_eager)} | dx: {_err_str(g_fla['dx'], g_eager['dx'])}", flush=True)

    ref_y = y_eager if y_eager is not None else y_fla
    ref_g = g_eager if g_eager is not None else g_fla
    ref_tag = "eager" if g_eager is not None else "fla_npu"

    for mode in modes:
        y_mega, g_mega = _run_fwd_bwd_and_collect(
            lambda inp, m=mode: module(inp, cp_mode=m),
            module, x_local, dy_local, world_size,
        )
        tf_m, tb_m, tt_m = _measure_fwd_bwd_ms(
            lambda inp, m=mode: module(inp, cp_mode=m),
            module, x_local, dy_local, args.warmup, args.iters, sync_dist=(world_size > 1),
        )
        if rank == 0:
            sp_f = tf_fla / tf_m
            sp_b = tb_fla / tb_m
            sp_t = tt_fla / tt_m
            print(
                f"  [MegaKda ({mode:10s})] fwd={tf_m:6.3f} ms ({sp_f:.2f}x) | "
                f"bwd={tb_m:6.3f} ms ({sp_b:.2f}x, {(sp_b-1)*100:+.1f}%) | "
                f"tot={tt_m:6.3f} ms ({sp_t:.2f}x, {(sp_t-1)*100:+.1f}%)",
                flush=True,
            )
            print(f"    vs {ref_tag:7s} out           : {_err_str(y_mega, ref_y)}", flush=True)
            for k in [
                "dx", "q_proj.weight", "k_proj.weight", "v_proj.weight",
                "f_b_proj.weight", "b_proj.weight", "o_proj.weight",
                "q_conv1d_weight", "k_conv1d_weight", "v_conv1d_weight",
                "A_log", "dt_bias", "o_norm_weight",
            ]:
                if k in g_mega and k in ref_g:
                    print(f"    vs {ref_tag:7s} {k:15s}: {_err_str(g_mega[k], ref_g[k])}", flush=True)

    module.release_resources()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
