# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Select the qualified A5 BF16 policy once when constructing a static plan."""
import os

from hyper_multicore.scheduler.config import RuntimeConfigC, TaskType
from hyper_multicore.scheduler.graph import ComputeGraph

from .bf16_gmm_tiling import retile_a5_bf16


def _use_a5_policy() -> bool:
    policy = os.environ.get("HYPER_MULTICORE_BF16_GMM_POLICY", "generic")
    if policy not in ("generic", "blaze_bwd_lanes_a5"):
        raise ValueError(f"HYPER_MULTICORE_BF16_GMM_POLICY must be generic or blaze_bwd_lanes_a5, got {policy!r}")
    return policy == "blaze_bwd_lanes_a5"


def configure_gmm_tasks(graph: ComputeGraph, config: RuntimeConfigC) -> None:
    """Preserve task boundaries while selecting Blaze and balanced backward lanes."""
    if not _use_a5_policy():
        return
    for index in range(config.task_num):
        task = config.all_tasks[index]
        if task.task_type == TaskType.TASK_SWI_GLU_GRAD:
            if os.environ.get('HYPER_MULTICORE_BF16_ACTIVATION_POLICY', 'legacy') == 'a5':
                task.extra_value_1 = 1
        if task.task_type == TaskType.TASK_GROUPED_MATMUL:
            task.extra_value_1 = 2
    offset = 0
    for op in graph.topological_sort():
        if op.name in ("act_grad", "gate_grad", "w1_grad", "w2_grad"):
            for index in range(offset, offset + op.task_num):
                task = config.all_tasks[index]
        if task.task_type == TaskType.TASK_SWI_GLU_GRAD:
            if os.environ.get('HYPER_MULTICORE_BF16_ACTIVATION_POLICY', 'legacy') == 'a5':
                task.extra_value_1 = 1
                if task.task_type != TaskType.TASK_GROUPED_MATMUL:
                    raise RuntimeError(f"Backward GMM descriptor type differs for {op.name} at {index}")
                task.extra_value_1 = 3
        offset += op.task_num


def select_gmm_tiling(data: bytes, cube_cores: int) -> bytes:
    """Pair the A5 descriptor selector with its qualified resource geometry."""
    return retile_a5_bf16(data, cube_cores) if _use_a5_policy() else data
