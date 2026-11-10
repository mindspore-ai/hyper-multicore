# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Bounded ALL transport permutation for a half-period balanced payload alias."""
from typing import Any


def balance_all_transport(cfg: Any, graph: Any, spec: Any) -> str:
    if spec.mx_vector_policy != 'all':
        return ('single_vector: original transport mapping'
                if spec.mx_vector_policy == 'single_vector' else 'legacy_odd')
    active = 2 * spec.num_cube_cores
    ep = spec.ep_size
    if active != 64 or ep <= 0 or active % (2 * ep) or spec.local_experts % 2:
        return 'unchanged: unsupported worker/EP/expert geometry'
    period = active // ep
    rotation = period // 2
    slots = spec.local_num_tokens * spec.top_k
    if slots % spec.num_experts:
        return 'unchanged: fractional balanced expert rows'
    balanced_rows = slots // spec.num_experts
    stages = graph.topological_sort()
    plans = []
    offset = 0
    for stage in stages:
        if stage.name in ('dispatch', 'combine'):
            tiles, remainder = divmod(stage.task_num, spec.num_experts)
            split = stage.split_value
            if remainder or tiles == 0 or tiles % period or balanced_rows % split:
                return 'unchanged: unsupported transport tile geometry'
            if (balanced_rows // split) % period != rotation:
                return 'unchanged: payload is not a half-period alias'
            plans.append((offset, offset + stage.task_num, tiles))
        offset += stage.task_num
    queue = cfg.vector_task_indices
    count = cfg.task_index_num[1]
    for start, end, tiles in plans:
        positions = [[] for _ in range(spec.local_experts)]
        for position in range(count):
            task_id = queue[position]
            if start <= task_id < end:
                task = cfg.all_tasks[task_id]
                expert = (task.task_index // tiles) % spec.local_experts
                positions[expert].append(position)
        for expert in range(1, spec.local_experts, 2):
            target = positions[expert]
            original = [queue[position] for position in target]
            shift = rotation * ep
            rotated = original[shift:] + original[:shift]
            for position, task_id in zip(target, rotated):
                queue[position] = task_id
    return f'rotated: odd-expert tile shift {rotation}, active {active}, EP {ep}'
