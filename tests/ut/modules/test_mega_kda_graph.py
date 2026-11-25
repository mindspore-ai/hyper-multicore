# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
"""Multi-rank discrete-event simulator UT verifying deadlock-freedom across CP=1/2/3/4/8/16 and all 3 CP modes."""

from __future__ import annotations

import pytest

from hyper_multicore.modules.mega_kda.graph import (
    CP_MODE_ALLGATHER,
    CP_MODE_PREFIX_SUM,
    CP_MODE_ULYSSES,
    build_mega_kda_runtime_config,
    resolve_schedule_spec,
)
from hyper_multicore.scheduler.config import EVENT_INVALID_ID, TaskType


def _simulate_distributed(cp_size: int, seq_full: int, hidden_size: int, num_heads: int, cp_mode: str,
                          full_gate: bool = False) -> None:
    seq_local = seq_full // cp_size
    cfgs = []
    specs = []
    for r in range(cp_size):
        sp = resolve_schedule_spec(
            seq_local=seq_local, hidden_size=hidden_size, num_heads=num_heads, head_dim=128,
            cp_size=cp_size, cp_rank=r, cp_mode=cp_mode, use_full_rank_gate=full_gate,
        )
        specs.append(sp)
        cfgs.append(build_mega_kda_runtime_config(sp))

    # Build per-rank state
    num_ranks = cp_size
    counters = [[0] * cfg.event_capacity for cfg in cfgs]
    done = [[False] * cfg.task_num for cfg in cfgs]

    # Validate explicit queue coverage
    for r, cfg in enumerate(cfgs):
        seen = set()
        for i in range(cfg.task_index_num[0]):
            seen.add(cfg.cube_task_indices[i])
        for i in range(cfg.task_index_num[1]):
            seen.add(cfg.vector_task_indices[i])
        assert len(seen) == cfg.task_num, f"rank {r}: missing tasks in queues"

    # Event-driven progress loop (with ready_depth=64 out-of-order rescan)
    progress = True
    while progress:
        progress = False
        for r, cfg in enumerate(cfgs):
            sp = specs[r]
            for tid in range(cfg.task_num):
                if done[r][tid]:
                    continue
                t = cfg.all_tasks[tid]
                dep = int(t.dependent_event)
                if dep != EVENT_INVALID_ID:
                    req = int(cfg.all_event_num_triggers[dep])
                    if counters[r][dep] < req:
                        continue
                # Fire task!
                done[r][tid] = True
                progress = True
                trig = int(t.trigger_event)
                if int(t.task_type) == int(TaskType.TASK_SHMEM_PUT_MEM_SIGNAL):
                    dst = int(t.extra_value_1)
                    assert 0 <= dst < num_ranks and (dst != r or int(t.extra_value_0) in (4, 6)), f"bad dst={dst} on rank={r}"
                    counters[dst][trig] += 1
                else:
                    counters[r][trig] += 1

    for r, cfg in enumerate(cfgs):
        unfinished = [tid for tid in range(cfg.task_num) if not done[r][tid]]
        if unfinished:
            sample = [(tid, TaskType(cfg.all_tasks[tid].task_type).name,
                       cfg.all_tasks[tid].dependent_event,
                       counters[r][cfg.all_tasks[tid].dependent_event],
                       cfg.all_event_num_triggers[cfg.all_tasks[tid].dependent_event])
                      for tid in unfinished[:8]]
            pytest.fail(f"Deadlock on rank {r} (cp={cp_size}, mode={cp_mode}, H={num_heads}): {sample}")


@pytest.mark.parametrize("seq_full,D,H,full_gate", [
    (2048, 2048, 16, False),
    (4096, 2048, 16, True),
    (8192, 4096, 32, False),
])
def test_cp1(seq_full, D, H, full_gate):
    _simulate_distributed(1, seq_full, D, H, "ulysses", full_gate)


@pytest.mark.parametrize("cp_size,seq_full,D,H,cp_mode", [
    (2, 2048, 2048, 16, "ulysses"),
    (4, 4096, 2048, 16, "ulysses"),
    (4, 8192, 4096, 32, "ulysses"),
    (2, 2048, 2048, 16, "allgather"),
    (4, 4096, 2048, 16, "allgather"),
    (4, 8192, 4096, 32, "allgather"),
    (2, 2048, 2048, 16, "prefix_sum"),
    (3, 3072, 2048, 16, "prefix_sum"),   # non-power-of-2 CP!
    (4, 4096, 2048, 16, "prefix_sum"),
    (4, 8192, 4096, 32, "prefix_sum"),
    (8, 8192, 2048, 16, "prefix_sum"),
    (16, 16384, 2048, 16, "prefix_sum"),
    (32, 32768, 2048, 16, "prefix_sum"), # CP > H!
    (32, 32768, 2048, 16, "auto"),       # auto selects prefix_sum when CP > H!
])
def test_multi_cp_modes(cp_size, seq_full, D, H, cp_mode):
    _simulate_distributed(cp_size, seq_full, D, H, cp_mode)
