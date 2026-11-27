# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
"""Symmetric SHMEM and workspace resource manager for HyperMegaKda (all 3 CP modes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hyper_multicore.modules.mega_kda.graph import (
    CP_MODE_ALLGATHER,
    CP_MODE_PREFIX_SUM,
    CP_MODE_ULYSSES,
    MegaKdaScheduleSpec,
)

ALIGN_BYTES = 512


def _align_up(v: int, a: int = ALIGN_BYTES) -> int:
    return ((v + a - 1) // a) * a


def compute_bwd_cache_bytes(seq_local: int, num_heads: int, head_dim: int, chunk_size: int = 64) -> int:
    """Exact 512B-aligned byte size matching common/mega_kda/bwd_cache_layout.h."""
    S, H, d, C = seq_local, num_heads, head_dim, chunk_size
    NT = S // C
    hc = H * NT
    off = 0
    for nbytes in (
        H * S * d * 2,           # qNorm
        H * S * d * 2,           # kNorm
        H * S * d * 2,           # vConv
        H * S * 4,               # qRstd
        H * S * 4,               # kRstd
        hc * C * d * 4,          # gCum
        hc * C * C * 2,          # mInv
        hc * C * C * 2,          # aqk
        hc * C * d * 2,          # w
        hc * C * d * 2,          # u
        hc * C * d * 2,          # qg
        hc * C * d * 2,          # kg
        hc * d * 4,              # expGLast
        hc * d * d * 2,          # hChunks
        hc * C * d * 2,          # vNew
        hc * C * d * 2,          # oAttn
        H * S * 4,               # oRstd
        3 * 3 * H * d * 2,       # haloQkv
        H * d * d * 4,           # aRank
    ):
        off = _align_up(off + nbytes)
    return max(512, off)


def compute_comm_buffer_bytes(spec: MegaKdaScheduleSpec) -> tuple[int, int]:
    """Return exact (send_bytes, recv_bytes) required for comm_send_buf and comm_recv_buf."""
    group_dim = spec.heads_per_group * spec.head_dim
    G = spec.head_groups
    cp = spec.cp_size
    d = spec.head_dim

    if spec.cp_mode == CP_MODE_ULYSSES:
        send_qkv = _align_up(spec.seq_local * group_dim * 2)
        send_beta = _align_up(spec.seq_local * spec.heads_per_group * 2)
        send_slot = _align_up(4 * send_qkv + send_beta)
        recv_o_slot = _align_up(spec.seq_local * group_dim * 2)
        # comm_send_buf holds: [cp*G*send_slot] for A2A #1 + [cp*G*recv_o_slot] for A2A #2!
        send_bytes = _align_up(cp * G * send_slot + cp * G * recv_o_slot)

        recv_qkv = _align_up(cp * spec.seq_local * group_dim * 2)
        recv_beta = _align_up(cp * spec.seq_local * spec.heads_per_group * 2)
        recv_group = _align_up(4 * recv_qkv + recv_beta)
        recv_o_off = _align_up(G * recv_group)
        recv_bytes = _align_up(recv_o_off + cp * G * recv_o_slot)
        return max(4096, send_bytes), max(4096, recv_bytes)

    # ALLGATHER / PREFIX_SUM: tiny constant SHMEM buffers (independent of seq_local!)
    halo_grp = _align_up(3 * 3 * group_dim * 2)
    scan_off = _align_up(G * halo_grp)
    scan_grp_half = _align_up(spec.heads_per_group * d * d * 4)
    scan_slot = _align_up(G * 2 * scan_grp_half)
    send_bytes = _align_up(scan_off + 2 * scan_slot)  # ping-pong [2] slots
    recv_slots = cp if spec.cp_mode == CP_MODE_ALLGATHER else max(1, spec.scan_rounds)
    recv_bytes = _align_up(scan_off + recv_slots * scan_slot)
    bwd_halo = _align_up(3 * 3 * spec.num_heads * d * 2)
    bwd_half = _align_up(spec.num_heads * d * d * 4)
    bwd_slot = 2 * bwd_half
    send_bytes = max(send_bytes, _align_up(bwd_halo + bwd_slot))
    recv_bytes = max(recv_bytes, _align_up(bwd_halo + recv_slots * bwd_slot))
    return max(4096, send_bytes), max(4096, recv_bytes)


@dataclass
class MegaKdaBuffers:
    comm_send_buf: torch.Tensor
    comm_recv_buf: torch.Tensor
    event_counters: torch.Tensor
    diagnostic_buffer: torch.Tensor
    shmem_handle: Any = None


class MegaKdaResourcePool:
    """Manages reusable NPU / SHMEM symmetric buffers per schedule key."""

    def __init__(self, device: torch.device, cp_group: Any = None, cp_size: int = 1) -> None:
        self.device = device
        self.cp_group = cp_group
        self.cp_size = cp_size
        self._shmem_acquired = False
        self._shmem_heap_bytes = 0
        self._cache: dict[tuple[int, int, int], MegaKdaBuffers] = {}

    def get(self, spec: MegaKdaScheduleSpec, event_capacity: int) -> MegaKdaBuffers:
        send_bytes, recv_bytes = compute_comm_buffer_bytes(spec)
        ev_elems = max(256, ((event_capacity + 32 * (spec.cp_size + 2) + 15) // 16) * 16)
        key = (send_bytes, recv_bytes, ev_elems)
        if key in self._cache:
            bufs = self._cache[key]
            if not (spec.proj_m_tile == 0 and (spec.num_heads % 2 == 0) and (spec.cp_size == 1 or spec.cp_mode in (1, 2))):
                bufs.event_counters[:event_capacity].zero_()
            return bufs

        if self.cp_size > 1:
            from hyper_multicore import shmem

            needed = _align_up(send_bytes + recv_bytes + ev_elems * 4 + 32 * 1024 * 1024, 64 * 1024 * 1024)
            if not self._shmem_acquired:
                heap_bytes = max(needed, 512 * 1024 * 1024)
                shmem.acquire(self.cp_group, heap_size_bytes=heap_bytes)
                self._shmem_acquired = True
                self._shmem_heap_bytes = heap_bytes
            send_buf = shmem.empty((send_bytes,), dtype=torch.uint8)
            recv_buf = shmem.empty((recv_bytes,), dtype=torch.uint8)
            ev_buf = shmem.empty((ev_elems,), dtype=torch.int32)
            ev_buf.zero_()
            torch.npu.synchronize()
            shmem.host_barrier()
        else:
            send_buf = torch.empty((send_bytes,), dtype=torch.uint8, device=self.device)
            recv_buf = torch.empty((recv_bytes,), dtype=torch.uint8, device=self.device)
            ev_buf = torch.zeros((ev_elems,), dtype=torch.int32, device=self.device)

        diag_buf = torch.zeros((256,), dtype=torch.int32, device=self.device)
        bufs = MegaKdaBuffers(
            comm_send_buf=send_buf,
            comm_recv_buf=recv_buf,
            event_counters=ev_buf,
            diagnostic_buffer=diag_buf,
        )
        self._cache[key] = bufs
        return bufs

    def release(self) -> None:
        cached = list(self._cache.values())
        self._cache.clear()
        if self._shmem_acquired:
            import gc
            from hyper_multicore import shmem

            torch.npu.synchronize()
            for bufs in cached:
                shmem.free(bufs.comm_send_buf)
                shmem.free(bufs.comm_recv_buf)
                shmem.free(bufs.event_counters)
            cached.clear()
            gc.collect()
            shmem.release()
            self._shmem_acquired = False
