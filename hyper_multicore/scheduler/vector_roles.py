# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Compact role-owned schedules reuse the existing task-index storage."""
from typing import Any

VECTOR_ROLE_BOUNDARY_SLOT = 3
RULE_POOL_WORDS = 10


def configure_vector_roles(cfg: Any, graph: Any, *, ready: bool, forward: bool) -> None:
    """Keep physical communication owners while moving math to the even AIVs."""
    original = list(cfg.vector_task_indices)[:cfg.task_index_num[1]]
    communication = [index for index in original if cfg.all_tasks[index].task_type in (0, 105, 141)]
    mathematics = [index for index in original if cfg.all_tasks[index].task_type not in (0, 105, 141)]
    if len(communication) % 2 != 1 or cfg.all_tasks[communication[-1]].task_type != 0:
        raise ValueError('Vector roles require dispatch/combine followed by one terminal task.')
    for position, index in enumerate(communication + mathematics):
        cfg.vector_task_indices[position] = index
    cfg.task_index_num[VECTOR_ROLE_BOUNDARY_SLOT] = len(communication)
    if not ready or forward:
        return
    stages, offset = {}, 0
    for op in graph.topological_sort():
        stages[op.name] = (offset, op.task_num)
        offset += op.task_num
    for target, source in (('da', 'dw2'), ('dx', 'dw1')):
        target_start, count = stages[target]
        source_start, source_count = stages[source]
        if count != source_count:
            raise ValueError('Backward role stages require matching per-expert logical cube counts.')
        for lane in range(count):
            cfg.all_tasks[target_start + lane].dependent_event = cfg.all_tasks[source_start + lane].dependent_event


def configure_rule_stages(cfg: Any, graph: Any, *, forward: bool) -> None:
    """Version7 uses the inactive mix queue for explicit ranges, retaining every owner."""
    stages, offset = {}, 0
    for op in graph.topological_sort():
        stages[op.name] = (offset, offset + op.task_num)
        offset += op.task_num
    boundary = cfg.task_index_num[VECTOR_ROLE_BOUNDARY_SLOT]
    vector = list(cfg.vector_task_indices[:cfg.task_index_num[1]])
    pools = (list(cfg.cube_task_indices[:cfg.task_index_num[0]]), vector[:boundary], vector[boundary:])
    names = (('down', 'up') if forward else ('dx', 'da', 'dw1', 'dw2'),
             ('combine', 'dispatch'),
             ('swiglu_quant', 'quant_x') if forward else ('swiglu_grad_quant', 'quant_dy'))
    for pool, (indices, order) in enumerate(zip(pools, names)):
        metadata = [len(order), len(indices) - 1 if pool == 1 else -1]
        for name in order:
            first, last = stages[name]
            positions = [position for position, task in enumerate(indices) if first <= task < last]
            begin, end = positions[0], positions[-1] + 1
            if positions != list(range(begin, end)):
                raise ValueError(f'Rule stage {name} is not contiguous in pool {pool}')
            metadata.extend((begin, end))
        metadata.extend([0] * (RULE_POOL_WORDS - len(metadata)))
        for index, value in enumerate(metadata):
            cfg.mix_task_indices[pool * RULE_POOL_WORDS + index] = value
