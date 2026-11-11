"""Build BF16 vector roles and complete logical-Cube partitions without padding."""
from typing import Any

from .transport_queue import build_transport_queues


BF16_CSR_QUEUE_TAG = -2
CUBE_OFFSETS_START = 65


def configure_bf16_queue(cfg: Any, graph: Any, spec: Any, *, forward: bool) -> None:
    """Preserve dependencies while selecting window2 or disjoint half-Cube experts."""
    if spec.num_cube_cores != 32 or spec.bf16_schedule not in ('window2', 'half16'):
        raise ValueError('BF16 scheduled execution requires A5 Cube32 and window2 or half16')
    half = forward and spec.bf16_schedule == 'half16'
    lanes = 16 if half else 32
    stages = {}
    offset = 0
    for op in graph.topological_sort():
        if op.task_num % spec.local_experts:
            raise ValueError(f'BF16 stage cannot partition tasks over experts: {op.name}')
        stages[op.name] = (offset, op.task_num // spec.local_experts)
        offset += op.task_num

    def tasks(stage: str, expert: int, lane: int, stride: int) -> list[int]:
        start, width = stages[stage]
        return list(range(start + expert * width + lane, start + (expert + 1) * width, stride))

    cubes = [[] for _ in range(32)]
    vectors = [[] for _ in range(64)]
    cube_stages = ('up_proj', 'down_proj') if forward else ('w2_grad', 'act_grad', 'gate_grad', 'w1_grad')
    for worker, queue in enumerate(cubes):
        if half:
            for expert in range(worker // 16, spec.local_experts, 2):
                for stage in cube_stages:
                    queue.extend(tasks(stage, expert, worker % 16, 16))
        else:
            front = 1 if forward else 2
            for stage in cube_stages[:front]:
                queue.extend(tasks(stage, 0, worker, 32))
            for expert in range(1, spec.local_experts):
                for stage in cube_stages[:front]:
                    queue.extend(tasks(stage, expert, worker, 32))
                for stage in cube_stages[front:]:
                    queue.extend(tasks(stage, expert - 1, worker, 32))
            for stage in cube_stages[front:]:
                queue.extend(tasks(stage, spec.local_experts - 1, worker, 32))
    math_stage = 'swiglu' if forward else 'swiglu_grad'
    for expert in range(spec.local_experts):
        for lane in range(32):
            vectors[2 * lane].extend(tasks(math_stage, expert, lane, 32))

    old_vector = list(cfg.vector_task_indices[:cfg.task_index_num[1]])
    transport = build_transport_queues(cfg, spec, old_vector, 1)
    for worker in range(64):
        vectors[worker].extend(transport[worker])
    old_cube = list(cfg.cube_task_indices[:cfg.task_index_num[0]])
    flat_cube = [tid for queue in cubes for tid in queue]
    flat_vector = [tid for queue in vectors for tid in queue]
    if sorted(flat_cube) != sorted(old_cube) or sorted(flat_vector) != sorted(old_vector):
        raise ValueError('BF16 scheduling must preserve every task exactly once')
    if len(cfg.mix_task_indices) < CUBE_OFFSETS_START + 33:
        raise ValueError('BF16 CSR queue requires at least98 runtime index slots')
    for index, tid in enumerate(flat_cube):
        cfg.cube_task_indices[index] = tid
    for index, tid in enumerate(flat_vector):
        cfg.vector_task_indices[index] = tid
    for start, queues in ((0, vectors), (CUBE_OFFSETS_START, cubes)):
        offset = 0
        for worker, queue in enumerate(queues):
            cfg.mix_task_indices[start + worker] = offset
            offset += len(queue)
        cfg.mix_task_indices[start + len(queues)] = offset
    if spec.ready_depth not in (0, 1, 8):
        raise ValueError('BF16 ready depth must be0,1 or8')
    if spec.ready_depth:
        for queue in ([] if half else cubes) + vectors[::2]:
            for tid in queue:
                cfg.all_tasks[tid].extra_value_1 |= 32 if spec.ready_depth == 8 else 64
    cfg.task_index_num[3] = BF16_CSR_QUEUE_TAG
    cfg.task_num = max(flat_cube + flat_vector) + 1
    if cfg.completion_event:
        cfg.all_event_num_triggers[cfg.completion_event] = 96
