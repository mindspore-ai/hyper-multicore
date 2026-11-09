# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Finite MX expert windows that preserve original tasks and signal contracts."""
from typing import Any


def resolve_mx_expert_window(value: str) -> int:
    if value not in ('0', '2', '3'):
        raise ValueError('HYPER_MULTICORE_MX_EXPERT_WINDOW must be 0, 2 or 3.')
    return int(value)


def reorder_mx_expert_windows(cfg: Any, graph: Any, local_experts: int,
                              window: int, forward: bool) -> None:
    if window not in (0, 2, 3):
        raise ValueError('MX expert window must be 0, 2 or 3.')
    if window == 0:
        return
    stages = graph.topological_sort()
    ranges, offset = {}, 0
    for stage in stages:
        ranges[stage.name] = range(offset, offset + stage.task_num)
        offset += stage.task_num
    cube_stages = ('up', 'down') if forward else ('dw2', 'da', 'dw1', 'dx')
    vector_stages = ('dispatch', 'quant_x', 'swiglu_quant') if forward else (
        'dispatch', 'quant_dy', 'swiglu_grad_quant')
    for field, count, stage_order in (
        ('cube_task_indices', cfg.task_index_num[0], cube_stages),
        ('vector_task_indices', cfg.task_index_num[1], vector_stages),
    ):
        queue = getattr(cfg, field)
        original = list(queue)[:count]
        grouped = {name: [[] for _ in range(local_experts)] for name in stage_order}
        for name in stage_order:
            stage = graph.get_op(name)
            for index in original:
                if index not in ranges[name]:
                    continue
                task = cfg.all_tasks[index]
                if name == 'dispatch':
                    global_experts = graph.get_op('dispatch').inputs[0].shape[0]
                    per_expert = stage.task_num // global_experts
                    expert = (task.task_index // per_expert) % local_experts
                else:
                    expert = task.task_index // (stage.task_num // local_experts)
                grouped[name][expert].append(index)
        replacement = []
        for start in range(0, local_experts, window):
            for name in stage_order:
                for expert in range(start, min(start + window, local_experts)):
                    replacement.extend(grouped[name][expert])
        # Combine and termination remain after all windows, preserving their RATR order.
        selected = {index for values in grouped.values() for expert in values for index in expert}
        replacement.extend(index for index in original if index not in selected)
        if sorted(replacement) != sorted(original):
            raise ValueError('MX expert window changed the queue task multiset.')
        for position, index in enumerate(replacement):
            queue[position] = index
