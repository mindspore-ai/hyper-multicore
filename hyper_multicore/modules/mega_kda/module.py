# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
"""Whole-layer MegaKda PyTorch nn.Module with fused Forward & Backward (CP=1, ulysses, allgather, prefix_sum)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from hyper_multicore.scheduler.runtime import serialize_runtime_config
from hyper_multicore.modules.mega_kda.graph import (
    CP_MODE_ALLGATHER,
    CP_MODE_PREFIX_SUM,
    CP_MODE_ULYSSES,
    MegaKdaScheduleSpec,
    build_mega_kda_runtime_config,
    resolve_schedule_spec,
)


def _detect_num_cube_cores(device: torch.device) -> int:
    try:
        from hyper_multicore.runtime.device import query_core_spec
        return int(query_core_spec(device).num_cube_cores)
    except Exception:
        return 20
from hyper_multicore.modules.mega_kda.resources import (
    MegaKdaBuffers,
    MegaKdaResourcePool,
    compute_bwd_cache_bytes,
)
from hyper_multicore.torch.ops import mega_kda as mega_kda_op
from hyper_multicore.torch.ops import mega_kda_grad as mega_kda_grad_op


class _UlyssesAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: Any, scatter_dim: int, gather_dim: int) -> torch.Tensor:
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
        grad_in = _UlyssesAllToAll.apply(grad_out, ctx.group, ctx.gather_dim, ctx.scatter_dim)
        return grad_in, None, None, None


class _MegaKdaCoreFunction(torch.autograd.Function):
    """Differentiable fused core: (q/k/v/g/beta/g_out + conv/gate/norm weights) -> o_normed."""

    @staticmethod
    def forward(
        ctx,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        fb_in: torch.Tensor,
        beta_in: torch.Tensor,
        g_in: torch.Tensor,
        w_conv_q: torch.Tensor,
        w_conv_k: torch.Tensor,
        w_conv_v: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        o_norm_weight: torch.Tensor,
        x_ref: torch.Tensor,
        w_fa: torch.Tensor,
        w_ga: torch.Tensor,
        w_o: torch.Tensor,
        bufs: MegaKdaBuffers,
        cfg_tensor: torch.Tensor,
        spec: MegaKdaScheduleSpec,
        lower_bound: float | None,
        norm_eps: float,
        scale: float,
    ) -> torch.Tensor:
        S_local = q_in.shape[0]
        q_c = q_in.contiguous()
        k_c = k_in.contiguous()
        v_c = v_in.contiguous()
        fb_c = fb_in.contiguous()
        beta_c = beta_in.contiguous()
        g_c = g_in.contiguous()
        wcq_2d = w_conv_q.squeeze(1).contiguous()
        wck_2d = w_conv_k.squeeze(1).contiguous()
        wcv_2d = w_conv_v.squeeze(1).contiguous()
        alog_c = a_log.contiguous()
        dt_c = dt_bias.contiguous()
        onorm_c = o_norm_weight.contiguous()

        cache_bytes = compute_bwd_cache_bytes(S_local, spec.num_heads, spec.head_dim, spec.chunk_size)
        bwd_cache = torch.empty((cache_bytes,), dtype=torch.uint8, device=q_c.device)
        o_normed = torch.empty((1, S_local, spec.num_heads * spec.head_dim), device=q_c.device, dtype=q_c.dtype)

        mega_kda_op(
            x_ref,
            q_c, k_c, v_c,
            w_fa, fb_c, beta_c,
            w_ga, g_c,
            wcq_2d, wck_2d, wcv_2d,
            alog_c, dt_c, onorm_c, w_o,
            bufs.comm_send_buf, bufs.comm_recv_buf,
            cfg_tensor, bufs.event_counters, bufs.diagnostic_buffer,
            bwd_cache,
            o_normed,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
            chunk_size=spec.chunk_size,
            cp_size=spec.cp_size,
            cp_rank=spec.cp_rank,
            head_groups=spec.head_groups,
            num_segs=spec.num_segs,
            proj_m_tile=0,
            o_proj_m_tile=spec.o_proj_m_tile,
            use_full_rank_gate=spec.use_full_rank_gate,
            has_lower_bound=(lower_bound is not None),
            lower_bound=(lower_bound if lower_bound is not None else 0.0),
            norm_eps=norm_eps,
            scale=scale,
            cp_mode=spec.cp_mode,
            save_for_bwd=True,
        )

        ctx.save_for_backward(
            q_c, k_c, v_c, fb_c, beta_c, g_c,
            wcq_2d, wck_2d, wcv_2d, alog_c, dt_c, onorm_c, bwd_cache,
        )
        ctx.bufs = bufs
        ctx.spec = spec
        ctx.lower_bound = lower_bound
        ctx.norm_eps = norm_eps
        ctx.scale = scale
        ctx.w_conv_dtype = w_conv_q.dtype
        ctx.o_norm_dtype = o_norm_weight.dtype
        return o_normed

    @staticmethod
    def backward(ctx, d_o_normed: torch.Tensor):
        (
            q_c, k_c, v_c, fb_c, beta_c, g_c,
            wcq_2d, wck_2d, wcv_2d, alog_c, dt_c, onorm_c, bwd_cache,
        ) = ctx.saved_tensors
        spec: MegaKdaScheduleSpec = ctx.spec
        bufs: MegaKdaBuffers = ctx.bufs

        (
            d_q_in, d_k_in, d_v_in, d_g_in, d_beta_in, d_g_out,
            d_w_conv, d_a_log, d_dt_bias, d_o_norm_weight,
        ) = mega_kda_grad_op(
            d_o_normed.contiguous(),
            q_c, k_c, v_c, fb_c, beta_c, g_c,
            wcq_2d, wck_2d, wcv_2d, alog_c, dt_c, onorm_c,
            bwd_cache,
            bufs.comm_send_buf, bufs.comm_recv_buf, bufs.event_counters,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
            chunk_size=spec.chunk_size,
            cp_size=spec.cp_size,
            cp_rank=spec.cp_rank,
            cp_mode=spec.cp_mode,
            has_lower_bound=(ctx.lower_bound is not None),
            lower_bound=(ctx.lower_bound if ctx.lower_bound is not None else 0.0),
            norm_eps=ctx.norm_eps,
            scale=ctx.scale,
        )

        d_wcq = d_w_conv[0].unsqueeze(1).to(ctx.w_conv_dtype)
        d_wck = d_w_conv[1].unsqueeze(1).to(ctx.w_conv_dtype)
        d_wcv = d_w_conv[2].unsqueeze(1).to(ctx.w_conv_dtype)
        d_onorm = d_o_norm_weight.to(ctx.o_norm_dtype)

        return (
            d_q_in, d_k_in, d_v_in, d_g_in, d_beta_in, d_g_out,
            d_wcq, d_wck, d_wcv, d_a_log, d_dt_bias, d_onorm,
            None, None, None, None, None, None, None, None, None, None,
        )


class MegaKda(nn.Module):
    """Whole-layer Kimi Delta Attention (KDA) megakernel module with Forward & Backward.

    Supports:
    - CP=1 (single NPU, 3-phase L2-resident fused forward & backward)
    - CP>1 with cp_mode in {"auto", "ulysses", "allgather", "prefix_sum"}
    - Both low-rank output gate (g_a_proj -> g_b_proj) and full-rank g_proj
    - Both safe lower-bounded decay gate (lower_bound=-5.0) and standard softplus gate (lower_bound=None)
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        head_dim: int = 128,
        chunk_size: int = 64,
        conv_size: int = 4,
        use_full_rank_gate: bool = False,
        gate_lower_bound: float | None = -5.0,
        rms_norm_eps: float = 1e-6,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_group: Any = None,
        cp_mode: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if conv_size != 4:
            raise ValueError(f"MegaKda currently supports conv_size=4, got {conv_size}")
        if head_dim != 128 or chunk_size != 64:
            raise ValueError(f"MegaKda currently supports head_dim=128 and chunk_size=64, got {head_dim}, {chunk_size}")
        if num_heads % cp_size != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by cp_size ({cp_size})")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.conv_size = conv_size
        self.use_full_rank_gate = use_full_rank_gate
        self.gate_lower_bound = gate_lower_bound
        self.rms_norm_eps = rms_norm_eps
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_group = cp_group
        self.cp_mode = cp_mode
        self.scale = 1.0 / math.sqrt(head_dim)

        D_proj = num_heads * head_dim
        kw = {"dtype": dtype, "device": device}

        self.q_proj = nn.Linear(hidden_size, D_proj, bias=False, **kw)
        self.k_proj = nn.Linear(hidden_size, D_proj, bias=False, **kw)
        self.v_proj = nn.Linear(hidden_size, D_proj, bias=False, **kw)
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **kw)
        self.f_b_proj = nn.Linear(head_dim, D_proj, bias=False, **kw)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False, **kw)
        if use_full_rank_gate:
            self.g_proj = nn.Linear(hidden_size, D_proj, bias=False, **kw)
            self.g_a_proj = None
            self.g_b_proj = None
        else:
            self.g_proj = None
            self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **kw)
            self.g_b_proj = nn.Linear(head_dim, D_proj, bias=False, **kw)

        self.q_conv1d_weight = nn.Parameter(torch.empty((D_proj, 1, conv_size), **kw))
        self.k_conv1d_weight = nn.Parameter(torch.empty((D_proj, 1, conv_size), **kw))
        self.v_conv1d_weight = nn.Parameter(torch.empty((D_proj, 1, conv_size), **kw))
        self.A_log = nn.Parameter(torch.zeros((num_heads,), dtype=torch.float32, device=device))
        self.dt_bias = nn.Parameter(torch.zeros((D_proj,), dtype=torch.float32, device=device))
        self.o_norm_weight = nn.Parameter(torch.ones((head_dim,), **kw))
        self.o_proj = nn.Linear(D_proj, hidden_size, bias=False, **kw)

        self.register_buffer("_dummy_gb", torch.zeros((1, 1), **kw), persistent=False)
        self.register_buffer("_dummy_bwd_cache", torch.zeros((512,), dtype=torch.uint8, device=device), persistent=False)

        dev = torch.device(device) if device is not None else torch.device("npu", torch.npu.current_device())
        self._pool = MegaKdaResourcePool(dev, cp_group=cp_group, cp_size=cp_size)
        self._sched_cache: dict[tuple[int, str, int, int], tuple[MegaKdaScheduleSpec, torch.Tensor, int]] = {}
        self._ulysses_cp1_pool: MegaKdaResourcePool | None = None

    def _get_schedule(
        self,
        seq_local: int,
        cp_mode_override: str | None = None,
        cp_size_override: int | None = None,
        num_heads_override: int | None = None,
    ) -> tuple[MegaKdaScheduleSpec, torch.Tensor, int]:
        mode_str = cp_mode_override if cp_mode_override is not None else self.cp_mode
        cp_s = cp_size_override if cp_size_override is not None else self.cp_size
        cp_r = 0 if cp_s == 1 else self.cp_rank
        n_h = num_heads_override if num_heads_override is not None else self.num_heads
        key = (seq_local, mode_str, cp_s, n_h)
        if key in self._sched_cache:
            return self._sched_cache[key]
        device = self.q_proj.weight.device
        num_cube_cores = _detect_num_cube_cores(device)
        spec = resolve_schedule_spec(
            seq_local=seq_local,
            hidden_size=self.hidden_size,
            num_heads=n_h,
            head_dim=self.head_dim,
            cp_size=cp_s,
            cp_rank=cp_r,
            cp_mode=mode_str,
            chunk_size=self.chunk_size,
            use_full_rank_gate=self.use_full_rank_gate,
            num_cube_cores=num_cube_cores,
        )
        cfg = build_mega_kda_runtime_config(spec)
        raw_bytes = serialize_runtime_config(cfg)
        cfg_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8).to(device).contiguous()
        res = (spec, cfg_tensor, int(cfg.event_capacity))
        self._sched_cache[key] = res
        return res

    def forward(self, hidden_states: torch.Tensor, *, cp_mode: str | None = None) -> torch.Tensor:
        if hidden_states.dim() == 3:
            B, S_local, D = hidden_states.shape
            if B != 1:
                raise ValueError(f"MegaKda currently supports B=1, got shape {hidden_states.shape}")
            x = hidden_states.contiguous()
        elif hidden_states.dim() == 2:
            S_local, D = hidden_states.shape
            x = hidden_states.unsqueeze(0).contiguous()
        else:
            raise ValueError(f"Expected 2D or 3D hidden_states, got {hidden_states.shape}")

        spec, cfg_tensor, ev_cap = self._get_schedule(S_local, cp_mode_override=cp_mode)
        w_ga = self.g_proj.weight if self.use_full_rank_gate else self.g_a_proj.weight
        w_gb = self._dummy_gb if self.use_full_rank_gate else self.g_b_proj.weight

        need_grad = torch.is_grad_enabled() and any(
            t.requires_grad
            for t in (
                x, self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                self.f_a_proj.weight, self.f_b_proj.weight, self.b_proj.weight,
                w_ga, self.q_conv1d_weight, self.k_conv1d_weight, self.v_conv1d_weight,
                self.A_log, self.dt_bias, self.o_norm_weight, self.o_proj.weight,
            )
        )

        # ---- Training Path (Differentiable Forward + Fused HyperMegaKdaGrad Backward) ----
        if need_grad:
            x2d = x.squeeze(0)
            q_in = F.linear(x2d, self.q_proj.weight)
            k_in = F.linear(x2d, self.k_proj.weight)
            v_in = F.linear(x2d, self.v_proj.weight)
            fb_in = F.linear(F.linear(x2d, self.f_a_proj.weight), self.f_b_proj.weight)
            beta_in = F.linear(x2d, self.b_proj.weight)
            if self.use_full_rank_gate:
                g_in = F.linear(x2d, self.g_proj.weight)
            else:
                g_in = F.linear(F.linear(x2d, self.g_a_proj.weight), self.g_b_proj.weight)

            if spec.cp_mode == CP_MODE_ULYSSES and spec.cp_size > 1:
                # Differentiable Ulysses A2A around CP=1 head-local fused Forward+Backward!
                h_l = spec.num_local_heads
                h0 = spec.cp_rank * h_l
                d0, d1 = h0 * self.head_dim, (h0 + h_l) * self.head_dim
                S_full = S_local * spec.cp_size
                q_u = _UlyssesAllToAll.apply(q_in.unsqueeze(0), self.cp_group, 2, 1).squeeze(0)
                k_u = _UlyssesAllToAll.apply(k_in.unsqueeze(0), self.cp_group, 2, 1).squeeze(0)
                v_u = _UlyssesAllToAll.apply(v_in.unsqueeze(0), self.cp_group, 2, 1).squeeze(0)
                fb_u = _UlyssesAllToAll.apply(fb_in.unsqueeze(0), self.cp_group, 2, 1).squeeze(0)
                b_u = _UlyssesAllToAll.apply(beta_in.unsqueeze(0), self.cp_group, 2, 1).squeeze(0)
                g_u = _UlyssesAllToAll.apply(g_in.unsqueeze(0), self.cp_group, 2, 1).squeeze(0)

                spec_u, cfg_u, ev_u = self._get_schedule(
                    S_full, cp_mode_override="allgather", cp_size_override=1, num_heads_override=h_l
                )
                if self._ulysses_cp1_pool is None:
                    self._ulysses_cp1_pool = MegaKdaResourcePool(x.device, cp_group=None, cp_size=1)
                bufs_u = self._ulysses_cp1_pool.get(spec_u, ev_u)
                x_u_ref = torch.empty((1, S_full, self.hidden_size), device=x.device, dtype=x.dtype)
                o_norm_u = _MegaKdaCoreFunction.apply(
                    q_u, k_u, v_u, fb_u, b_u, g_u,
                    self.q_conv1d_weight[d0:d1],
                    self.k_conv1d_weight[d0:d1],
                    self.v_conv1d_weight[d0:d1],
                    self.A_log[h0 : h0 + h_l],
                    self.dt_bias[d0:d1],
                    self.o_norm_weight,
                    x_u_ref, self.f_a_proj.weight, w_ga, self.o_proj.weight,
                    bufs_u, cfg_u, spec_u,
                    self.gate_lower_bound, self.rms_norm_eps, self.scale,
                )
                o_normed = _UlyssesAllToAll.apply(o_norm_u.view(1, S_full, h_l, self.head_dim), self.cp_group, 1, 2)
                out_2d = F.linear(o_normed.reshape(S_local, self.num_heads * self.head_dim), self.o_proj.weight)
                return out_2d.unsqueeze(0) if hidden_states.dim() == 3 else out_2d

            bufs = self._pool.get(spec, ev_cap)
            o_normed = _MegaKdaCoreFunction.apply(
                q_in, k_in, v_in, fb_in, beta_in, g_in,
                self.q_conv1d_weight, self.k_conv1d_weight, self.v_conv1d_weight,
                self.A_log, self.dt_bias, self.o_norm_weight,
                x.detach(), self.f_a_proj.weight, w_ga, self.o_proj.weight,
                bufs, cfg_tensor, spec,
                self.gate_lower_bound, self.rms_norm_eps, self.scale,
            )
            out_2d = F.linear(o_normed.squeeze(0), self.o_proj.weight)
            return out_2d.unsqueeze(0) if hidden_states.dim() == 3 else out_2d

        # ---- Zero-Overhead Inference Path (100% identical to MegaKDA-v7!) ----
        bufs = self._pool.get(spec, ev_cap)
        if spec.cp_mode == CP_MODE_ULYSSES and spec.cp_size > 1:
            h_l = spec.num_local_heads
            h0 = spec.cp_rank * h_l
            d0 = h0 * self.head_dim
            d1 = (h0 + h_l) * self.head_dim
            w_cq = self.q_conv1d_weight[d0:d1].squeeze(1).contiguous()
            w_ck = self.k_conv1d_weight[d0:d1].squeeze(1).contiguous()
            w_cv = self.v_conv1d_weight[d0:d1].squeeze(1).contiguous()
            a_log = self.A_log[h0 : h0 + h_l].contiguous()
            dt_bias = self.dt_bias[d0:d1].contiguous()
        else:
            w_cq = self.q_conv1d_weight.squeeze(1).contiguous()
            w_ck = self.k_conv1d_weight.squeeze(1).contiguous()
            w_cv = self.v_conv1d_weight.squeeze(1).contiguous()
            a_log = self.A_log.contiguous()
            dt_bias = self.dt_bias.contiguous()

        if spec.proj_m_tile == 0:
            x2d = x.squeeze(0)
            q_in = F.linear(x2d, self.q_proj.weight)
            k_in = F.linear(x2d, self.k_proj.weight)
            v_in = F.linear(x2d, self.v_proj.weight)
            fb_in = F.linear(F.linear(x2d, self.f_a_proj.weight), self.f_b_proj.weight)
            beta_in = F.linear(x2d, self.b_proj.weight)
            if self.use_full_rank_gate:
                g_in = F.linear(x2d, self.g_proj.weight)
            else:
                g_in = F.linear(F.linear(x2d, self.g_a_proj.weight), self.g_b_proj.weight)
            o_normed = torch.empty((1, S_local, self.num_heads * self.head_dim), device=x.device, dtype=x.dtype)
            mega_kda_op(
                x,
                q_in, k_in, v_in,
                self.f_a_proj.weight, fb_in, beta_in,
                w_ga, g_in,
                w_cq, w_ck, w_cv,
                a_log, dt_bias, self.o_norm_weight, self.o_proj.weight,
                bufs.comm_send_buf, bufs.comm_recv_buf,
                cfg_tensor, bufs.event_counters, bufs.diagnostic_buffer,
                self._dummy_bwd_cache,
                o_normed,
                num_heads=spec.num_heads,
                head_dim=spec.head_dim,
                chunk_size=spec.chunk_size,
                cp_size=spec.cp_size,
                cp_rank=spec.cp_rank,
                head_groups=spec.head_groups,
                num_segs=spec.num_segs,
                proj_m_tile=0,
                o_proj_m_tile=spec.o_proj_m_tile,
                use_full_rank_gate=spec.use_full_rank_gate,
                has_lower_bound=(self.gate_lower_bound is not None),
                lower_bound=(self.gate_lower_bound if self.gate_lower_bound is not None else 0.0),
                norm_eps=self.rms_norm_eps,
                scale=self.scale,
                cp_mode=spec.cp_mode,
                save_for_bwd=False,
            )
            out_2d = F.linear(o_normed.squeeze(0), self.o_proj.weight)
            return out_2d.unsqueeze(0) if hidden_states.dim() == 3 else out_2d

        out = torch.empty_like(x)
        mega_kda_op(
            x,
            self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
            self.f_a_proj.weight, self.f_b_proj.weight, self.b_proj.weight,
            w_ga, w_gb,
            w_cq, w_ck, w_cv,
            a_log, dt_bias, self.o_norm_weight, self.o_proj.weight,
            bufs.comm_send_buf, bufs.comm_recv_buf,
            cfg_tensor, bufs.event_counters, bufs.diagnostic_buffer,
            self._dummy_bwd_cache,
            out,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
            chunk_size=spec.chunk_size,
            cp_size=spec.cp_size,
            cp_rank=spec.cp_rank,
            head_groups=spec.head_groups,
            num_segs=spec.num_segs,
            proj_m_tile=spec.proj_m_tile,
            o_proj_m_tile=spec.o_proj_m_tile,
            use_full_rank_gate=spec.use_full_rank_gate,
            has_lower_bound=(self.gate_lower_bound is not None),
            lower_bound=(self.gate_lower_bound if self.gate_lower_bound is not None else 0.0),
            norm_eps=self.rms_norm_eps,
            scale=self.scale,
            cp_mode=spec.cp_mode,
            save_for_bwd=False,
        )
        return out if hidden_states.dim() == 3 else out.squeeze(0)

    def release_resources(self) -> None:
        self._pool.release()
        if self._ulysses_cp1_pool is not None:
            self._ulysses_cp1_pool.release()
            self._ulysses_cp1_pool = None
