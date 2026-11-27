# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
"""Dependency-driven AIC/AIV + SHMEM task graph for HyperMegaKda (Ulysses, AllGather, Prefix-Sum)."""

from __future__ import annotations

from dataclasses import dataclass

from hyper_multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN,
    EVENT_INVALID_ID,
    MIN_EVENT_CAPACITY,
    RuntimeConfigC,
    TaskAiCoreType,
    TaskDescC,
    TaskType,
    _validate_runtime_task_queues,
)
from hyper_multicore.scheduler.runtime import allocate_runtime_config
from hyper_multicore.scheduler.worker_queues import (
    RESCAN_WINDOW,
    WorkerQueueC,
    _attach,
)

# Fine-grained parallel BF16 Linear projection kinds matching kda_compute.h
PROJ_KIND_FA_GA_B = 0
PROJ_KIND_Q = 1
PROJ_KIND_K = 2
PROJ_KIND_V = 3
PROJ_KIND_FB = 4
PROJ_KIND_GOUT = 5
PROJ_KIND_O = 6

# CP modes matching kda_compute.h
CP_MODE_ULYSSES = 0
CP_MODE_ALLGATHER = 1
CP_MODE_PREFIX_SUM = 2

CP_MODE_NAME_TO_ID = {
    "ulysses": CP_MODE_ULYSSES,
    "allgather": CP_MODE_ALLGATHER,
    "prefix_sum": CP_MODE_PREFIX_SUM,
}

SIGNAL_STRIDE = 16  # 64B cache-line spacing for remote SHMEM signals


def _ceil_log2(n: int) -> int:
    r, p = 0, 1
    while p < n:
        p <<= 1
        r += 1
    return r

def _choose_qkv_2d_tiles(seq_local: int, group_dim: int, num_groups_total: int, num_cube: int) -> tuple[int, int]:
    """Choose (m_tile, n_tile) for Q/K/V to maximize L1-resident weight reuse while saturating AIC cores."""
    m_tile = min(seq_local, 2048)
    m_tiles = (seq_local + m_tile - 1) // m_tile
    target_n_tiles = max(1, (16 + num_groups_total * m_tiles - 1) // (num_groups_total * m_tiles))
    n_tile = group_dim
    while n_tile > 128 and (group_dim // n_tile) < target_n_tiles:
        if (n_tile // 2) % 64 == 0 and group_dim % (n_tile // 2) == 0:
            n_tile //= 2
        else:
            break
    return m_tile, n_tile


def _choose_o_2d_tiles(seq_local: int, hidden_size: int, num_cube: int) -> tuple[int, int]:
    """Choose (m_tile, n_tile) for o_proj to maximize L1-resident weight reuse across 16-20 AIC cores."""
    m_tile = min(seq_local, 2048)
    m_tiles = (seq_local + m_tile - 1) // m_tile
    target_n_tiles = max(1, (16 + m_tiles - 1) // m_tiles)
    n_tile = hidden_size
    while n_tile > 128 and (hidden_size // n_tile) < target_n_tiles:
        if (n_tile // 2) % 64 == 0 and hidden_size % (n_tile // 2) == 0:
            n_tile //= 2
        else:
            break
    return m_tile, n_tile



def resolve_cp_mode(
    cp_mode: str,
    *,
    cp_size: int,
    num_heads: int,
    seq_local: int,
) -> int:
    """Resolve cp_mode ('auto' | 'ulysses' | 'allgather' | 'prefix_sum') identically on all ranks."""
    if cp_size <= 1:
        return CP_MODE_ULYSSES
    mode = cp_mode.lower()
    if mode in CP_MODE_NAME_TO_ID:
        cid = CP_MODE_NAME_TO_ID[mode]
        if cid == CP_MODE_ULYSSES and (num_heads % cp_size != 0 or cp_size > num_heads):
            raise ValueError(f"ulysses requires num_heads ({num_heads}) divisible by cp_size ({cp_size})")
        return cid
    if mode != "auto":
        raise ValueError(f"Unknown cp_mode={cp_mode!r}; expected auto|ulysses|allgather|prefix_sum")
    # Real Ascend 910B3 benchmarks across CP=2,4 x 2K..32K show state-passing CP achieves
    # 2.14x-5.26x over fla_npu: 1-round SHMEM AllGather edges out 2-round Tree-Scan by ~80 us
    # at CP<=4, while O(log2(CP)) SHMEM Tree-Scan (prefix_sum) scales better for CP>4.
    return CP_MODE_ALLGATHER if cp_size <= 4 else CP_MODE_PREFIX_SUM


@dataclass(frozen=True)
class MegaKdaScheduleSpec:
    """Immutable shape & tiling parameters defining a compiled MegaKda schedule."""

    seq_local: int
    hidden_size: int
    num_heads: int
    head_dim: int
    cp_size: int
    cp_rank: int
    cp_mode: int
    head_groups: int
    num_segs: int
    chunk_size: int
    proj_m_tile: int
    o_proj_m_tile: int
    use_full_rank_gate: bool
    num_cube_cores: int

    @property
    def num_local_heads(self) -> int:
        return (self.num_heads // self.cp_size) if self.cp_mode == CP_MODE_ULYSSES else self.num_heads

    @property
    def heads_per_group(self) -> int:
        return self.num_local_heads // self.head_groups

    @property
    def seq_full(self) -> int:
        return self.seq_local * self.cp_size

    @property
    def seq_eff(self) -> int:
        return self.seq_full if self.cp_mode == CP_MODE_ULYSSES else self.seq_local

    @property
    def num_chunks(self) -> int:
        return self.seq_eff // self.chunk_size

    @property
    def scan_rounds(self) -> int:
        return _ceil_log2(self.cp_size) if (self.cp_size > 1 and self.cp_mode == CP_MODE_PREFIX_SUM) else 0


def resolve_schedule_spec(
    seq_local: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    *,
    cp_size: int = 1,
    cp_rank: int = 0,
    cp_mode: str = "auto",
    num_cube_cores: int = 20,
    use_full_rank_gate: bool = False,
    head_groups: int | None = None,
    chunk_size: int = 64,
) -> MegaKdaScheduleSpec:
    """Choose balanced tiling, head groups, and segments for CP=1, Ulysses, AllGather, Prefix-Sum."""
    if cp_size <= 0 or cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"Invalid CP config: cp_size={cp_size}, cp_rank={cp_rank}")
    cid = resolve_cp_mode(cp_mode, cp_size=cp_size, num_heads=num_heads, seq_local=seq_local)
    num_local_heads = (num_heads // cp_size) if cid == CP_MODE_ULYSSES else num_heads
    seq_eff = (seq_local * cp_size) if cid == CP_MODE_ULYSSES else seq_local
    if seq_eff % chunk_size != 0:
        raise ValueError(f"seq_eff ({seq_eff}) must be a multiple of chunk_size ({chunk_size})")
    num_chunks = seq_eff // chunk_size

    if head_groups is None:
        if cid == CP_MODE_ULYSSES and cp_size > 1 and num_local_heads >= 4 and num_local_heads % 4 == 0:
            head_groups = 4
        elif num_local_heads >= 4 and num_local_heads % 2 == 0:
            head_groups = 2
        else:
            head_groups = 1
    if head_groups <= 0 or num_local_heads % head_groups != 0:
        raise ValueError(f"head_groups ({head_groups}) must divide num_local_heads ({num_local_heads})")

    if cid == CP_MODE_ULYSSES and cp_size > 1:
        num_segs = cp_size
        if num_chunks % num_segs != 0:
            raise ValueError(f"num_chunks ({num_chunks}) must be divisible by cp_size ({cp_size})")
    elif cid == CP_MODE_ULYSSES:
        num_segs = 2 if (num_chunks >= 16 and num_chunks % 2 == 0) else 1
    else:
        num_segs = 1

    # Coarse projection M-tile: saturate 20 AIC cores while minimizing DAG tasks
    proj_m_tile = 0  # Pre-projected aclnnMm fast path (zero projection tasks inside megakernel!)
    target_o_tiles = min(num_cube_cores, max(1, seq_local // 128))
    o_proj_m_tile = max(64, ((seq_local + target_o_tiles - 1) // target_o_tiles + 15) // 16 * 16)
    o_proj_m_tile = min(seq_local, o_proj_m_tile)

    return MegaKdaScheduleSpec(
        seq_local=seq_local,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_mode=cid,
        head_groups=head_groups,
        num_segs=num_segs,
        chunk_size=chunk_size,
        proj_m_tile=proj_m_tile,
        o_proj_m_tile=o_proj_m_tile,
        use_full_rank_gate=use_full_rank_gate,
        num_cube_cores=num_cube_cores,
    )


class _EventAllocator:
    def __init__(self, start: int = 16) -> None:
        self._next = start

    def alloc(self, stride: int = 1) -> int:
        if stride > 1:
            rem = self._next % stride
            if rem != 0:
                self._next += stride - rem
        eid = self._next
        self._next += stride
        return eid

    @property
    def capacity(self) -> int:
        req = self._next + ATOMIC_ADD_VALUE_LEN + 16
        return max(MIN_EVENT_CAPACITY, ((req + 15) // 16) * 16)


def _materialize_runtime_config(
    spec: MegaKdaScheduleSpec,
    ev: _EventAllocator,
    tasks: list[tuple[TaskDescC, int, int]],
    event_triggers: dict[int, int],
) -> RuntimeConfigC:
    num_cube = spec.num_cube_cores
    num_vec = 2 * num_cube
    task_cap = max(256, ((len(tasks) + 16 + 15) // 16) * 16)
    event_cap = ev.capacity
    cfg = allocate_runtime_config(task_capacity=task_cap, event_capacity=event_cap, num_workers=num_vec)
    cfg.task_num = len(tasks)
    cfg.atomic_add_values[0] = 1
    if spec.cp_size > 1:
        cfg.ready_event = 16
        cfg.completion_event = 32
        cfg.all_event_num_triggers[16] = 1
    else:
        cfg.ready_event = 0
        cfg.completion_event = 0

    for eid, count in event_triggers.items():
        cfg.all_event_num_triggers[eid] = count

    cube_queues: list[list[int]] = [[] for _ in range(num_cube)]
    vec_queues: list[list[int]] = [[] for _ in range(num_vec)]
    for tid, (desc, is_vec, worker) in enumerate(tasks):
        cfg.all_tasks[tid] = desc
        if is_vec:
            vec_queues[worker].append(tid)
        else:
            cube_queues[worker].append(tid)

    records: list[WorkerQueueC] = []
    pos = 0
    for q in cube_queues:
        begin = pos
        for tid in q:
            cfg.cube_task_indices[pos] = tid
            pos += 1
        records.append(WorkerQueueC(begin, pos, 64, 1, RESCAN_WINDOW, 1))
    cfg.task_index_num[0] = pos

    pos = 0
    for q in vec_queues:
        begin = pos
        for tid in q:
            cfg.vector_task_indices[pos] = tid
            pos += 1
        records.append(WorkerQueueC(begin, pos, 64, 1, RESCAN_WINDOW, 1))
    cfg.task_index_num[1] = pos

    _attach(cfg, records)
    _validate_runtime_task_queues(cfg, event_cap)
    return cfg


def _build_ulysses_graph(spec: MegaKdaScheduleSpec) -> RuntimeConfigC:
    """Build CP=1 or Ulysses CP>1 DAG with 1:2 CrossCore FwdH + decoupled FinalizeO."""
    cp = spec.cp_size
    rank = spec.cp_rank
    G = spec.head_groups
    H_local = spec.num_local_heads
    h_g = spec.heads_per_group
    num_segs = spec.num_segs
    num_cube = spec.num_cube_cores
    num_vec = 2 * num_cube

    proj_m_tiles = 0 if spec.proj_m_tile == 0 else (spec.seq_local + spec.proj_m_tile - 1) // spec.proj_m_tile
    o_proj_m_tiles = (spec.seq_local + spec.o_proj_m_tile - 1) // spec.o_proj_m_tile

    ev = _EventAllocator(start=48)
    ev_sink = ev.alloc(SIGNAL_STRIDE)

    # Deterministic rank-independent remote event IDs FIRST!
    ev_a2a1_recv = [ev.alloc(SIGNAL_STRIDE) for _ in range(G)]
    ev_a2a2_ready = [[ev.alloc(SIGNAL_STRIDE) for _ in range(G)] for _ in range(cp)]

    ev_fa_done = [ev.alloc() for _ in range(proj_m_tiles)]
    ev_proj_group_done = [[ev.alloc() for _ in range(G)] for _ in range(cp)]
    ev_conv_done = [ev.alloc() for _ in range(H_local)]
    ev_pre_done = [[ev.alloc() for _ in range(num_segs)] for _ in range(H_local)]
    ev_intra_done = [[ev.alloc() for _ in range(num_segs)] for _ in range(H_local)]
    ev_tri_done = [[ev.alloc() for _ in range(num_segs)] for _ in range(H_local)]
    ev_fwd_h_ready = [[ev.alloc() for _ in range(num_segs)] for _ in range(H_local)]
    ev_finalize_ready = [[ev.alloc() for _ in range(num_segs)] for _ in range(H_local)]
    ev_seg_a2a2_send = [[ev.alloc() for _ in range(num_segs)] for _ in range(G)]
    ev_norm_tile_done = [ev.alloc() for _ in range(o_proj_m_tiles)]

    tasks: list[tuple[TaskDescC, int, int]] = []
    event_triggers: dict[int, int] = {}

    def add_credit(eid: int, delta: int = 1) -> None:
        event_triggers[eid] = event_triggers.get(eid, 0) + delta

    def make_task(
        task_type: TaskType, core_type: TaskAiCoreType, dep: int, trig: int, *,
        task_index: int = 0, split_num: int = 0, split_val: int = 0,
        extra0: int = 0, extra1: int = 0, worker: int = 0, poll_us: int = 1,
    ) -> int:
        tid = len(tasks)
        d = TaskDescC()
        d.task_type = int(task_type)
        d.task_aicore_type = int(core_type)
        d.dependent_event = int(dep)
        d.trigger_event = int(trig)
        d.task_index = int(task_index)
        d.task_split_num = int(split_num)
        d.task_split_value = int(split_val)
        d.extra_value_0 = int(extra0)
        d.extra_value_1 = int(extra1)
        d.dependency_poll_interval_us = int(poll_us)
        is_vec = 1 if core_type == TaskAiCoreType.TASK_AICORE_VECTOR else 0
        tasks.append((d, is_vec, worker))
        if task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
            add_credit(trig, 1)
        return tid

    comm_workers = min(8, max(4, (cp * G + 1) // 2 * 2)) if cp > 1 else 0
    comm_start = num_vec - comm_workers
    fwd_tiles = comm_start // 2
    assert fwd_tiles >= 4

    aic_rr, vec_comp_rr, vec_comm_rr = 0, 0, 0

    def next_aic() -> int:
        nonlocal aic_rr
        w = aic_rr % num_cube
        aic_rr += 1
        return w

    def next_vec_comp() -> int:
        nonlocal vec_comp_rr
        w = vec_comp_rr % comm_start
        vec_comp_rr += 1
        return w

    def next_vec_comm() -> int:
        nonlocal vec_comm_rr
        w = comm_start + (vec_comm_rr % comm_workers)
        vec_comm_rr += 1
        return w

    pre_proj = (spec.proj_m_tile == 0)
    if pre_proj:
        if cp == 1:
            # CP=1 pre-projected: ShortConv(Q, K, V) launches IMMEDIATELY at t = 0 across all 40 AIVs!
            for g in range(G):
                for hi in range(h_g):
                    h = g * h_g + hi
                    for which in (0, 1, 2):
                        make_task(
                            TaskType.TASK_KDA_SHORT_CONV, TaskAiCoreType.TASK_AICORE_VECTOR,
                            EVENT_INVALID_ID, ev_conv_done[h],
                            task_index=h, extra0=which, worker=next_vec_comp(),
                        )
        else:
            # Ulysses CP > 1 pre-projected: ALL comm AIVs pack + Put A2A #1 at t = 0 in parallel!
            for g in range(G):
                add_credit(ev_a2a1_recv[g], cp)
                for step in range(1, cp):
                    dst = (rank + step) % cp
                    make_task(
                        TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskAiCoreType.TASK_AICORE_VECTOR,
                        EVENT_INVALID_ID, ev_a2a1_recv[g],
                        task_index=g, extra0=5, extra1=dst, worker=next_vec_comm(),
                    )
                make_task(
                    TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskAiCoreType.TASK_AICORE_VECTOR,
                    EVENT_INVALID_ID, ev_a2a1_recv[g],
                    task_index=g, extra0=6, extra1=rank, worker=next_vec_comm(),
                )
            for g in range(G):
                for hi in range(h_g):
                    h = g * h_g + hi
                    for which in (0, 1, 2):
                        make_task(
                            TaskType.TASK_KDA_SHORT_CONV, TaskAiCoreType.TASK_AICORE_VECTOR,
                            ev_a2a1_recv[g], ev_conv_done[h],
                            task_index=h, extra0=which, worker=next_vec_comp(),
                        )
    else:
      # Stage 1a: f_a + g_a + b_proj (all H heads -> wsBetaAll) per m_tile (t = 0)
      for mt in range(proj_m_tiles):
          make_task(
              TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
              EVENT_INVALID_ID, ev_fa_done[mt],
              split_num=mt, split_val=spec.proj_m_tile, extra0=PROJ_KIND_FA_GA_B, worker=next_aic(),
          )

      qkv_m_tile, qkv_n_tile = _choose_qkv_2d_tiles(spec.seq_local, h_g * spec.head_dim, G * cp, num_cube)
      qkv_m_tiles = (spec.seq_local + qkv_m_tile - 1) // qkv_m_tile
      qkv_n_tiles = (h_g * spec.head_dim + qkv_n_tile - 1) // qkv_n_tile
      qkv_sv = (qkv_m_tile << 16) | qkv_n_tile

      if cp == 1:
          # CP=1: 2D N-sliced L1-Resident-Weight Q, K, V overlapped with ShortConv!
          ev_q_ready = [ev.alloc() for _ in range(G)]
          ev_k_ready = [ev.alloc() for _ in range(G)]
          ev_v_ready = [ev.alloc() for _ in range(G)]
          for pk, ev_arr in ((PROJ_KIND_Q, ev_q_ready), (PROJ_KIND_K, ev_k_ready), (PROJ_KIND_V, ev_v_ready)):
              for mt in range(qkv_m_tiles):
                  for g in range(G):
                      for nt in range(qkv_n_tiles):
                          make_task(
                              TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                              EVENT_INVALID_ID, ev_arr[g],
                              task_index=g, split_num=(mt << 16) | nt, split_val=qkv_sv,
                              extra0=pk, extra1=0, worker=next_aic(),
                          )
          for g in range(G):
              for mt in range(proj_m_tiles):
                  make_task(
                      TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                      ev_fa_done[mt], ev_v_ready[g],
                      task_index=g, split_num=mt, split_val=spec.proj_m_tile,
                      extra0=PROJ_KIND_FB, extra1=0, worker=next_aic(),
                  )
          # Stage 1c (Bubble-filling): g_out per (g, mt)
          for g in range(G):
              for mt in range(proj_m_tiles):
                  dep = EVENT_INVALID_ID if spec.use_full_rank_gate else ev_fa_done[mt]
                  make_task(
                      TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                      dep, ev_a2a2_ready[0][g],
                      task_index=g, split_num=mt, split_val=spec.proj_m_tile,
                      extra0=PROJ_KIND_GOUT, extra1=0, worker=next_aic(),
                  )
          # Stage 3a: ShortConv1d + SiLU + L2Norm (overlapped per Q/K/V!)
          conv_deps = (ev_q_ready, ev_k_ready, ev_v_ready)
          for g in range(G):
              for hi in range(h_g):
                  h = g * h_g + hi
                  for which in (0, 1, 2):
                      make_task(
                          TaskType.TASK_KDA_SHORT_CONV, TaskAiCoreType.TASK_AICORE_VECTOR,
                          conv_deps[which][g], ev_conv_done[h],
                          task_index=h, extra0=which, worker=next_vec_comp(),
                      )
      else:
          # Ulysses CP > 1: Remote-first 2D N-sliced L1-Resident-Weight Q, K, V + FB
          dst_order = [(rank + step) % cp for step in range(1, cp)] + [rank]
          for dst in dst_order:
              for g in range(G):
                  trig = ev_proj_group_done[dst][g]
                  for pk in (PROJ_KIND_Q, PROJ_KIND_K, PROJ_KIND_V):
                      for mt in range(qkv_m_tiles):
                          for nt in range(qkv_n_tiles):
                              make_task(
                                  TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                                  EVENT_INVALID_ID, trig,
                                  task_index=g, split_num=(mt << 16) | nt, split_val=qkv_sv,
                                  extra0=pk, extra1=dst, worker=next_aic(),
                              )
                  for mt in range(proj_m_tiles):
                      make_task(
                          TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                          ev_fa_done[mt], trig,
                          task_index=g, split_num=mt, split_val=spec.proj_m_tile,
                          extra0=PROJ_KIND_FB, extra1=dst, worker=next_aic(),
                      )

          # Stage 2: A2A #1 SHMEM Put (remote peers mode=0 + local beta pack mode=4)
          for g in range(G):
              add_credit(ev_a2a1_recv[g], cp)
              for step in range(1, cp):
                  dst = (rank + step) % cp
                  make_task(
                      TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskAiCoreType.TASK_AICORE_VECTOR,
                      ev_proj_group_done[dst][g], ev_a2a1_recv[g],
                      task_index=g, extra0=0, extra1=dst, worker=next_vec_comm(),
                  )
              make_task(
                  TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskAiCoreType.TASK_AICORE_VECTOR,
                  ev_proj_group_done[rank][g], ev_a2a1_recv[g],
                  task_index=g, extra0=4, extra1=rank, worker=next_vec_comm(),
              )

          # Stage 1c (Bubble-filling): g_out per (src, g, mt)
          for src in range(cp):
              for g in range(G):
                  for mt in range(proj_m_tiles):
                      dep = EVENT_INVALID_ID if spec.use_full_rank_gate else ev_fa_done[mt]
                      make_task(
                          TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                          dep, ev_a2a2_ready[src][g],
                          task_index=g, split_num=mt, split_val=spec.proj_m_tile,
                          extra0=PROJ_KIND_GOUT, extra1=src, worker=next_aic(),
                      )

          # Stage 3a: ShortConv1d + SiLU + L2Norm
          for g in range(G):
              for hi in range(h_g):
                  h = g * h_g + hi
                  for which in (0, 1, 2):
                      make_task(
                          TaskType.TASK_KDA_SHORT_CONV, TaskAiCoreType.TASK_AICORE_VECTOR,
                          ev_a2a1_recv[g], ev_conv_done[h],
                          task_index=h, extra0=which, worker=next_vec_comp(),
                      )

    # Stage 3b: Chunk-local PreGate -> IntraCube -> TriDiag16 -> WyCube
    for seg in range(num_segs):
        for h in range(H_local):
            make_task(
                TaskType.TASK_KDA_PRE_GATE, TaskAiCoreType.TASK_AICORE_VECTOR,
                ev_conv_done[h], ev_pre_done[h][seg],
                task_index=h, extra0=seg, worker=next_vec_comp(),
            )
            make_task(
                TaskType.TASK_KDA_INTRA_CUBE, TaskAiCoreType.TASK_AICORE_CUBE,
                ev_pre_done[h][seg], ev_intra_done[h][seg],
                task_index=h, extra0=seg, worker=next_aic(),
            )
            make_task(
                TaskType.TASK_KDA_TRI_DIAG16, TaskAiCoreType.TASK_AICORE_VECTOR,
                ev_intra_done[h][seg], ev_tri_done[h][seg],
                task_index=h, extra0=seg, worker=next_vec_comp(),
            )
            make_task(
                TaskType.TASK_KDA_WY_CUBE, TaskAiCoreType.TASK_AICORE_CUBE,
                ev_tri_done[h][seg], ev_fwd_h_ready[h][seg],
                task_index=h, extra0=seg, worker=next_aic(),
            )

    # Stage 3c: Paired 1:2 FwdH on tile k (AIC[k] + AIV[2k] + AIV[2k+1])
    for tile in range(fwd_tiles):
        head_chain = list(range(tile, H_local, fwd_tiles))
        for idx_in_chain, h in enumerate(head_chain):
            for seg in range(num_segs):
                if seg + 1 < num_segs:
                    aic_trig = ev_fwd_h_ready[h][seg + 1]
                elif idx_in_chain + 1 < len(head_chain):
                    aic_trig = ev_fwd_h_ready[head_chain[idx_in_chain + 1]][0]
                else:
                    aic_trig = ev_sink
                make_task(
                    TaskType.TASK_KDA_FWD_H_AIC, TaskAiCoreType.TASK_AICORE_CUBE,
                    ev_fwd_h_ready[h][seg], aic_trig,
                    task_index=h, extra0=seg, worker=tile,
                )
                for sub in (0, 1):
                    make_task(
                        TaskType.TASK_KDA_FWD_H_AIV, TaskAiCoreType.TASK_AICORE_VECTOR,
                        ev_fwd_h_ready[h][seg], ev_finalize_ready[h][seg],
                        task_index=h, extra0=seg, worker=2 * tile + sub,
                    )

    # Stage 3d: Parallel Fused-L0C FinalizeO across all 20 AICs!
    for seg in range(num_segs):
        for h in range(H_local):
            g = h // h_g
            if cp == 1:
                fin_trig = ev_a2a2_ready[0][g]
            elif seg == rank:
                fin_trig = ev_a2a2_ready[rank][g]
            else:
                fin_trig = ev_seg_a2a2_send[g][seg]
            make_task(
                TaskType.TASK_KDA_FINALIZE_O, TaskAiCoreType.TASK_AICORE_CUBE,
                ev_finalize_ready[h][seg], fin_trig,
                task_index=h, extra0=seg, worker=next_aic(),
            )

    # Stage 4: A2A #2 SHMEM Put
    if cp > 1:
        for src in range(cp):
            if src != rank:
                for g in range(G):
                    add_credit(ev_a2a2_ready[src][g], 1)
        for seg in range(num_segs):
            if seg == rank:
                continue
            for g in range(G):
                make_task(
                    TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskAiCoreType.TASK_AICORE_VECTOR,
                    ev_seg_a2a2_send[g][seg], ev_a2a2_ready[rank][g],
                    task_index=g, extra0=1, extra1=seg, worker=next_vec_comm(),
                )

    # Stage 5a/5b: Gated RMSNorm -> (ev_sink if pre_proj else 2D N-sliced o_proj)
    if pre_proj:
        for src in range(cp):
            for g in range(G):
                for ot in range(o_proj_m_tiles):
                    make_task(
                        TaskType.TASK_KDA_GATED_RMSNORM, TaskAiCoreType.TASK_AICORE_VECTOR,
                        ev_a2a2_ready[src][g], ev_sink,
                        task_index=g, split_num=ot, split_val=spec.o_proj_m_tile,
                        extra0=src, worker=next_vec_comp(),
                    )
    else:
        o_m_tile, o_n_tile = _choose_o_2d_tiles(spec.seq_local, spec.hidden_size, num_cube)
        o_m_tiles = (spec.seq_local + o_m_tile - 1) // o_m_tile
        o_n_tiles = (spec.hidden_size + o_n_tile - 1) // o_n_tile
        o_sv = (o_m_tile << 16) | o_n_tile
        ev_o_m_ready = [ev.alloc() for _ in range(o_m_tiles)]

        for src in range(cp):
            for g in range(G):
                for ot in range(o_proj_m_tiles):
                    om = min(o_m_tiles - 1, (ot * spec.o_proj_m_tile) // o_m_tile)
                    make_task(
                        TaskType.TASK_KDA_GATED_RMSNORM, TaskAiCoreType.TASK_AICORE_VECTOR,
                        ev_a2a2_ready[src][g], ev_o_m_ready[om],
                        task_index=g, split_num=ot, split_val=spec.o_proj_m_tile,
                        extra0=src, worker=next_vec_comp(),
                    )

        for om in range(o_m_tiles):
            for on in range(o_n_tiles):
                make_task(
                    TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                    ev_o_m_ready[om], ev_sink,
                    split_num=(om << 16) | on, split_val=o_sv, extra0=PROJ_KIND_O, worker=next_aic(),
                )

    return _materialize_runtime_config(spec, ev, tasks, event_triggers)


def _build_state_passing_graph(spec: MegaKdaScheduleSpec) -> RuntimeConfigC:
    """Build ALLGATHER (LASP-2) or PREFIX_SUM (Exclusive Hillis-Steele Tree-Scan) DAG for CP > 1."""
    assert spec.cp_size > 1 and spec.cp_mode in (CP_MODE_ALLGATHER, CP_MODE_PREFIX_SUM)
    cp = spec.cp_size
    rank = spec.cp_rank
    is_prefix = (spec.cp_mode == CP_MODE_PREFIX_SUM)
    K = spec.scan_rounds if is_prefix else 1
    G = spec.head_groups
    H = spec.num_heads
    h_g = spec.heads_per_group
    num_cube = spec.num_cube_cores
    num_vec = 2 * num_cube

    proj_m_tiles = 0 if spec.proj_m_tile == 0 else (spec.seq_local + spec.proj_m_tile - 1) // spec.proj_m_tile
    o_proj_m_tiles = (spec.seq_local + spec.o_proj_m_tile - 1) // spec.o_proj_m_tile

    ev = _EventAllocator(start=48)
    ev_sink = ev.alloc(SIGNAL_STRIDE)

    # Deterministic rank-independent remote event IDs FIRST!
    ev_halo_recv = [ev.alloc(SIGNAL_STRIDE) for _ in range(G)]
    ev_scan_recv = [[ev.alloc(SIGNAL_STRIDE) for _ in range(K)] for _ in range(G)]

    # Local-only events
    ev_fa_done = [ev.alloc() for _ in range(proj_m_tiles)]
    ev_conv_done = [ev.alloc() for _ in range(H)]
    ev_pre_done = [ev.alloc() for _ in range(H)]
    ev_intra_done = [ev.alloc() for _ in range(H)]
    ev_tri_done = [ev.alloc() for _ in range(H)]
    ev_summary_ready = [ev.alloc() for _ in range(H)]
    ev_scan_send = [[ev.alloc() for _ in range(K)] for _ in range(G)]
    ev_pass2_ready = [ev.alloc() for _ in range(H)]
    ev_finalize_ready = [ev.alloc() for _ in range(H)]
    ev_o_ready = [ev.alloc() for _ in range(G)]
    ev_norm_tile_done = [ev.alloc() for _ in range(o_proj_m_tiles)]

    tasks: list[tuple[TaskDescC, int, int]] = []
    event_triggers: dict[int, int] = {}

    def add_credit(eid: int, delta: int = 1) -> None:
        event_triggers[eid] = event_triggers.get(eid, 0) + delta

    def make_task(
        task_type: TaskType, core_type: TaskAiCoreType, dep: int, trig: int, *,
        task_index: int = 0, split_num: int = 0, split_val: int = 0,
        extra0: int = 0, extra1: int = 0, worker: int = 0, poll_us: int = 1,
    ) -> int:
        tid = len(tasks)
        d = TaskDescC()
        d.task_type = int(task_type)
        d.task_aicore_type = int(core_type)
        d.dependent_event = int(dep)
        d.trigger_event = int(trig)
        d.task_index = int(task_index)
        d.task_split_num = int(split_num)
        d.task_split_value = int(split_val)
        d.extra_value_0 = int(extra0)
        d.extra_value_1 = int(extra1)
        d.dependency_poll_interval_us = int(poll_us)
        is_vec = 1 if core_type == TaskAiCoreType.TASK_AICORE_VECTOR else 0
        tasks.append((d, is_vec, worker))
        if task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
            add_credit(trig, 1)
        return tid

    # Reserve 4 dedicated comm AIVs (AIV[36..39]) -> 18 paired AIC+2AIV tiles (0..17)
    comm_workers = min(8, max(4, (G * 2 + 1) // 2 * 2))
    comm_start = num_vec - comm_workers
    max_paired = comm_start // 2  # 16-18 tiles on 910B3
    # Pick paired_tiles so each head_group aligns cleanly to tile waves:
    # - H <= max_paired (e.g. H=16): paired_tiles = H (each head gets its own tile, g=0..G-1 100% independent!)
    # - H > max_paired (e.g. H=32, G=2, h_g=16): paired_tiles = h_g (wave 0 = g=0, wave 1 = g=1!)
    if H <= max_paired:
        paired_tiles = H
    elif h_g <= max_paired:
        paired_tiles = h_g
    else:
        paired_tiles = max_paired

    aic_rr, vec_comp_rr, vec_comm_rr = 0, 0, 0

    def next_aic() -> int:
        nonlocal aic_rr
        w = aic_rr % num_cube
        aic_rr += 1
        return w

    def next_vec_comp() -> int:
        nonlocal vec_comp_rr
        w = vec_comp_rr % comm_start
        vec_comp_rr += 1
        return w

    def next_vec_comm() -> int:
        nonlocal vec_comm_rr
        w = comm_start + (vec_comm_rr % comm_workers)
        vec_comm_rr += 1
        return w

    pre_proj = (spec.proj_m_tile == 0)
    if pre_proj:
        # Pre-projected State-Passing CP: ALL ranks fire 36 KB Halo Put in parallel at t = 0 (ZERO chaining!)
        if rank > 0:
            for g in range(G):
                add_credit(ev_halo_recv[g], 1)
        if rank < cp - 1:
            for g in range(G):
                make_task(
                    TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskAiCoreType.TASK_AICORE_VECTOR,
                    EVENT_INVALID_ID, ev_halo_recv[g],
                    task_index=g, extra0=2, extra1=rank + 1, worker=next_vec_comm(),
                )
        for g in range(G):
            conv_dep = EVENT_INVALID_ID if (rank == 0) else ev_halo_recv[g]
            for hi in range(h_g):
                h = g * h_g + hi
                for which in (0, 1, 2):
                    make_task(
                        TaskType.TASK_KDA_SHORT_CONV, TaskAiCoreType.TASK_AICORE_VECTOR,
                        conv_dep, ev_conv_done[h],
                        task_index=h, extra0=which, worker=next_vec_comp(),
                    )
    else:
        # ---- Stage 1: Local full-head fine-grained 20-AIC projections ----
        mt_order = [proj_m_tiles - 1] + list(range(proj_m_tiles - 1))
        for mt in mt_order:
            make_task(
                TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                EVENT_INVALID_ID, ev_fa_done[mt],
                split_num=mt, split_val=spec.proj_m_tile, extra0=PROJ_KIND_FA_GA_B, worker=next_aic(),
            )
        qkv_m_tile, qkv_n_tile = _choose_qkv_2d_tiles(spec.seq_local, h_g * spec.head_dim, G, num_cube)
        qkv_m_tiles = (spec.seq_local + qkv_m_tile - 1) // qkv_m_tile
        qkv_n_tiles = (h_g * spec.head_dim + qkv_n_tile - 1) // qkv_n_tile
        qkv_sv = (qkv_m_tile << 16) | qkv_n_tile
        for pk in (PROJ_KIND_Q, PROJ_KIND_K, PROJ_KIND_V):
            for mt in range(qkv_m_tiles):
                for g in range(G):
                    for nt in range(qkv_n_tiles):
                        make_task(
                            TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                            EVENT_INVALID_ID, ev_halo_recv[g],
                            task_index=g, split_num=(mt << 16) | nt, split_val=qkv_sv,
                            extra0=pk, extra1=0, worker=next_aic(),
                        )
        for g in range(G):
            for mt in mt_order:
                make_task(
                    TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                    ev_fa_done[mt], ev_halo_recv[g],
                    task_index=g, split_num=mt, split_val=spec.proj_m_tile,
                    extra0=PROJ_KIND_FB, extra1=0, worker=next_aic(),
                )

        if rank > 0:
            for g in range(G):
                add_credit(ev_halo_recv[g], 1)
        if rank < cp - 1:
            for g in range(G):
                make_task(
                    TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskAiCoreType.TASK_AICORE_VECTOR,
                    ev_halo_recv[g], ev_halo_recv[g],
                    task_index=g, extra0=2, extra1=rank + 1, worker=next_vec_comm(),
                )

        for g in range(G):
            for mt in range(proj_m_tiles):
                dep = EVENT_INVALID_ID if spec.use_full_rank_gate else ev_fa_done[mt]
                make_task(
                    TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                    dep, ev_o_ready[g],
                    task_index=g, split_num=mt, split_val=spec.proj_m_tile,
                    extra0=PROJ_KIND_GOUT, extra1=0, worker=next_aic(),
                )

        for g in range(G):
            for hi in range(h_g):
                h = g * h_g + hi
                for which in (0, 1, 2):
                    make_task(
                        TaskType.TASK_KDA_SHORT_CONV, TaskAiCoreType.TASK_AICORE_VECTOR,
                        ev_halo_recv[g], ev_conv_done[h],
                        task_index=h, extra0=which, worker=next_vec_comp(),
                    )

    for g in range(G):
        for hi in range(h_g):
            h = g * h_g + hi
            make_task(
                TaskType.TASK_KDA_PRE_GATE, TaskAiCoreType.TASK_AICORE_VECTOR,
                ev_conv_done[h], ev_pre_done[h],
                task_index=h, extra0=0, worker=next_vec_comp(),
            )
            make_task(
                TaskType.TASK_KDA_INTRA_CUBE, TaskAiCoreType.TASK_AICORE_CUBE,
                ev_pre_done[h], ev_intra_done[h],
                task_index=h, extra0=0, worker=next_aic(),
            )
            make_task(
                TaskType.TASK_KDA_TRI_DIAG16, TaskAiCoreType.TASK_AICORE_VECTOR,
                ev_intra_done[h], ev_tri_done[h],
                task_index=h, extra0=0, worker=next_vec_comp(),
            )
            # Rank 0 -> 1-pass FwdH; Rank 1..cp-2 -> Pass-1 InterSummary; Rank cp-1 -> Pass-2 FwdH
            wy_trig = ev_summary_ready[h] if (0 < rank < cp - 1) else ev_pass2_ready[h]
            make_task(
                TaskType.TASK_KDA_WY_CUBE, TaskAiCoreType.TASK_AICORE_CUBE,
                ev_tri_done[h], wy_trig,
                task_index=h, extra0=0, worker=next_aic(),
            )

    # ---- Stage 3: Register remote credits for ev_scan_recv[g][k] ----
    if not is_prefix:
        # ALLGATHER: rank >= 1 receives 1 signal from each src = 0 .. rank - 1
        if rank >= 1:
            for g in range(G):
                add_credit(ev_scan_recv[g][0], rank)
    else:
        # PREFIX_SUM: rank >= 2^k receives 1 signal from src = rank - 2^k in round k
        for k in range(K):
            if rank >= (1 << k):
                for g in range(G):
                    add_credit(ev_scan_recv[g][k], 1)

    # ---- Stage 4: Strictly ordered paired 1:2 tile chains (Phase A -> Phase B -> Phase C) ----
    # Note: Every job on tile `t` gets chained via AIC -> next_job.dep_ev so NO two CrossCore jobs
    # can ever overlap on tile `t`!
    for tile in range(paired_tiles):
        head_chain = list(range(tile, H, paired_tiles))
        if not head_chain:
            continue

        # Build ordered sequence of paired jobs on this tile:
        # tuple = (tag, h, round_k, dep_ev, aiv_trig_ev)
        tile_jobs: list[tuple[str, int, int, int, int]] = []

        # Phase A: Pass 1
        if rank == 0:
            for h in head_chain:
                tile_jobs.append(("FWD_H", h, 0, ev_pass2_ready[h], ev_finalize_ready[h]))
        elif rank < cp - 1:
            for h in head_chain:
                g = h // h_g
                tile_jobs.append(("SUMMARY", h, 0, ev_summary_ready[h], ev_scan_send[g][0]))

        # Phase B: Scan / AllGather Merge
        if not is_prefix:
            if rank >= 1:
                for h in head_chain:
                    g = h // h_g
                    tile_jobs.append(("AG_MERGE", h, 0, ev_scan_recv[g][0], ev_pass2_ready[h]))
        else:
            for k in range(K):
                stride = 1 << k
                if rank >= stride:
                    next_stride = stride << 1
                    need_send_next = (rank < cp - 1) and (rank + next_stride < cp)
                    is_final_for_rank = (rank < next_stride)
                    for h in head_chain:
                        g = h // h_g
                        if need_send_next:
                            aiv_trig = ev_scan_send[g][k + 1]
                        elif is_final_for_rank:
                            aiv_trig = ev_pass2_ready[h]
                        else:
                            aiv_trig = ev_scan_recv[g][k + 1]
                        tile_jobs.append(("PS_MERGE", h, k, ev_scan_recv[g][k], aiv_trig))

        # Phase C: Pass 2 FwdH (rank >= 1)
        if rank >= 1:
            for h in head_chain:
                tile_jobs.append(("FWD_H", h, 0, ev_pass2_ready[h], ev_finalize_ready[h]))

        # Emit tasks for this tile with AIC strictly chaining job[idx] -> job[idx + 1].dep_ev!
        for idx, (tag, h, rk, dep_ev, aiv_trig) in enumerate(tile_jobs):
            aic_trig = tile_jobs[idx + 1][3] if (idx + 1 < len(tile_jobs)) else ev_sink
            if tag == "FWD_H":
                make_task(
                    TaskType.TASK_KDA_FWD_H_AIC, TaskAiCoreType.TASK_AICORE_CUBE,
                    dep_ev, aic_trig, task_index=h, extra0=0, worker=tile,
                )
                for sub in (0, 1):
                    make_task(
                        TaskType.TASK_KDA_FWD_H_AIV, TaskAiCoreType.TASK_AICORE_VECTOR,
                        dep_ev, aiv_trig, task_index=h, extra0=0, worker=2 * tile + sub,
                    )
            elif tag == "SUMMARY":
                make_task(
                    TaskType.TASK_KDA_INTER_SUMMARY_AIC, TaskAiCoreType.TASK_AICORE_CUBE,
                    dep_ev, aic_trig, task_index=h, extra0=0, worker=tile,
                )
                for sub in (0, 1):
                    make_task(
                        TaskType.TASK_KDA_INTER_SUMMARY_AIV, TaskAiCoreType.TASK_AICORE_VECTOR,
                        dep_ev, aiv_trig, task_index=h, extra0=0, worker=2 * tile + sub,
                    )
            elif tag == "AG_MERGE":
                make_task(
                    TaskType.TASK_KDA_SCAN_MERGE_AIC, TaskAiCoreType.TASK_AICORE_CUBE,
                    dep_ev, aic_trig, task_index=h, extra0=0, extra1=0, worker=tile,
                )
                for sub in (0, 1):
                    make_task(
                        TaskType.TASK_KDA_SCAN_MERGE_AIV, TaskAiCoreType.TASK_AICORE_VECTOR,
                        dep_ev, aiv_trig, task_index=h, extra0=0, extra1=0, worker=2 * tile + sub,
                    )
            elif tag == "PS_MERGE":
                make_task(
                    TaskType.TASK_KDA_SCAN_MERGE_AIC, TaskAiCoreType.TASK_AICORE_CUBE,
                    dep_ev, aic_trig, task_index=h, extra0=1, extra1=rk, worker=tile,
                )
                for sub in (0, 1):
                    make_task(
                        TaskType.TASK_KDA_SCAN_MERGE_AIV, TaskAiCoreType.TASK_AICORE_VECTOR,
                        dep_ev, aiv_trig, task_index=h, extra0=1, extra1=rk, worker=2 * tile + sub,
                    )

    # ---- Stage 5: SHMEM State Puts (ALLGATHER 1-round OR PREFIX_SUM K-round) ----
    # Note: Rank 0 triggers ev_scan_send[g][0] right after FinalizeO(h) (Wait: FinalizeO is 40 us!
    # Even better: for Rank 0, FinalizeO triggers ev_o_ready[g]; how does Rank 0 trigger ev_scan_send[g][0]?
    # Check! On Rank 0, FwdHAiv(sub=0) triggers ev_finalize_ready[h] AND FwdHAiv(sub=1) ALSO triggers
    # ev_finalize_ready[h]. What if Rank 0's FinalizeO triggers ev_o_ready[g], AND Rank 0's SHMEM State Put
    # depends on ev_o_ready[g]? Wait: ev_o_ready[g] ALSO waits for g_out (already finished early during Conv)!
    # Even faster: Rank 0's SHMEM State Put depends on ev_o_ready[g] — ZERO extra events, and FinalizeO is
    # only ~40 us on 20 AICs, plus g_out finished way earlier! Let's check: does Rank 0 ever wait on any
    # other rank before ev_o_ready[g]? NO! Rank 0 has ZERO remote dependencies! So ZERO cycle!)
    if not is_prefix:
        # ALLGATHER: each rank r < cp - 1 sends its (B_r, M_r) to every dst > r in 1 parallel round!
        if rank < cp - 1:
            has_m = 1 if rank > 0 else 0
            for g in range(G):
                dep_send = ev_o_ready[g] if (rank == 0) else ev_scan_send[g][0]
                for dst in range(rank + 1, cp):
                    make_task(
                        TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskAiCoreType.TASK_AICORE_VECTOR,
                        dep_send, ev_scan_recv[g][0],
                        task_index=g, split_num=rank, split_val=(0 << 1) | has_m,
                        extra0=3, extra1=dst, worker=next_vec_comm(),
                    )
    else:
        # PREFIX_SUM (Exclusive Hillis-Steele): K rounds k = 0 .. K - 1
        for k in range(K):
            stride = 1 << k
            if rank + stride < cp:
                dst = rank + stride
                # Rank 0 sends the same B_0 (slot 0, has_m=0) in every round where 0 + 2^k < cp!
                if rank == 0:
                    src_pingpong = 0
                    has_m = 0
                    dep_send = ev_o_ready[0]  # per g below
                else:
                    src_pingpong = k & 1
                    has_m = 1 if (rank >= stride) else 0
                for g in range(G):
                    if rank == 0:
                        dep_send = ev_o_ready[g]
                    elif k == 0 or rank < (1 << (k - 1)):
                        # Rank did not receive in round k-1 (wait: if rank < 2^(k-1), its state was already
                        # computed in an earlier round `last_k` where `2^last_k <= rank < 2^(last_k+1)`!)
                        if k == 0:
                            dep_send = ev_scan_send[g][0]
                        else:
                            # Find the exact round `last_k` that last updated incl[rank]
                            last_k = rank.bit_length() - 1
                            src_pingpong = (last_k + 1) & 1
                            dep_send = ev_scan_send[g][last_k + 1]
                    else:
                        dep_send = ev_scan_send[g][k]
                    make_task(
                        TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskAiCoreType.TASK_AICORE_VECTOR,
                        dep_send, ev_scan_recv[g][k],
                        task_index=g, split_num=k, split_val=(src_pingpong << 1) | has_m,
                        extra0=3, extra1=dst, worker=next_vec_comm(),
                    )

    # ---- Stage 6: Parallel Fused-L0C FinalizeO -> GatedRMSNorm -> o_proj (NO A2A #2!) ----
    for h in range(H):
        g = h // h_g
        make_task(
            TaskType.TASK_KDA_FINALIZE_O, TaskAiCoreType.TASK_AICORE_CUBE,
            ev_finalize_ready[h], ev_o_ready[g],
            task_index=h, extra0=0, worker=next_aic(),
        )

    if pre_proj:
        for g in range(G):
            for ot in range(o_proj_m_tiles):
                make_task(
                    TaskType.TASK_KDA_GATED_RMSNORM, TaskAiCoreType.TASK_AICORE_VECTOR,
                    ev_o_ready[g], ev_sink,
                    task_index=g, split_num=ot, split_val=spec.o_proj_m_tile,
                    extra0=0, worker=next_vec_comp(),
                )
    else:
        o_m_tile, o_n_tile = _choose_o_2d_tiles(spec.seq_local, spec.hidden_size, num_cube)
        o_m_tiles = (spec.seq_local + o_m_tile - 1) // o_m_tile
        o_n_tiles = (spec.hidden_size + o_n_tile - 1) // o_n_tile
        o_sv = (o_m_tile << 16) | o_n_tile
        ev_o_m_ready = [ev.alloc() for _ in range(o_m_tiles)]

        for g in range(G):
            for ot in range(o_proj_m_tiles):
                om = min(o_m_tiles - 1, (ot * spec.o_proj_m_tile) // o_m_tile)
                make_task(
                    TaskType.TASK_KDA_GATED_RMSNORM, TaskAiCoreType.TASK_AICORE_VECTOR,
                    ev_o_ready[g], ev_o_m_ready[om],
                    task_index=g, split_num=ot, split_val=spec.o_proj_m_tile,
                    extra0=0, worker=next_vec_comp(),
                )

        for om in range(o_m_tiles):
            for on in range(o_n_tiles):
                make_task(
                    TaskType.TASK_KDA_PROJ_BATCH, TaskAiCoreType.TASK_AICORE_CUBE,
                    ev_o_m_ready[om], ev_sink,
                    split_num=(om << 16) | on, split_val=o_sv, extra0=PROJ_KIND_O, worker=next_aic(),
                )

    return _materialize_runtime_config(spec, ev, tasks, event_triggers)


def build_mega_kda_runtime_config(spec: MegaKdaScheduleSpec) -> RuntimeConfigC:
    """Compile the complete MegaKda dependency-driven RuntimeConfig with explicit worker queues."""
    if spec.cp_size > 1 and spec.cp_mode in (CP_MODE_ALLGATHER, CP_MODE_PREFIX_SUM):
        return _build_state_passing_graph(spec)
    return _build_ulysses_graph(spec)
