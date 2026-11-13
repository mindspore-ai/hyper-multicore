"""Expert-stage ordering shared by BF16 and MX; execution stays in generic runtime."""
from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.scheduler.worker_queues import install_worker_queues

from .transport import TRANSPORT_TASKS, transport_queues


def configure_expert_queues(cfg, graph, spec, *, forward: bool, mxfp8: bool = False):
    """Build the adopted two-expert pipeline with FIFO QP1 communication owners."""
    if spec.transport_engine != "udma" and not mxfp8:
        return cfg
    cores = spec.num_cube_cores
    stages = {}
    offset = 0
    for op in graph.topological_sort():
        stages[op.name] = (offset, op.task_num)
        offset += op.task_num

    def tasks(stage, expert, lane):
        begin, count = stages[stage]
        if count % spec.local_experts:
            raise ValueError(f"Stage {stage} does not partition over local experts")
        width = count // spec.local_experts
        return list(range(begin + expert * width + lane, begin + (expert + 1) * width, cores))

    if mxfp8:
        cube_stages = ('up', 'down') if forward else ('dw2', 'da', 'dw1', 'dx')
        vector_stages = ('quant_x', 'swiglu_quant') if forward else ('quant_dy', 'swiglu_grad_quant')
    else:
        cube_stages = ('up_proj', 'down_proj') if forward else ('w2_grad', 'act_grad', 'gate_grad', 'w1_grad')
        vector_stages = ('swiglu',) if forward else ('swiglu_grad',)
    communication = [tid for tid in cfg.vector_task_indices[:cfg.task_index_num[1]]
                     if cfg.all_tasks[tid].task_type in (*TRANSPORT_TASKS, TaskType.TASK_TERMINATE)]
    vectors = transport_queues(cfg, spec, communication)
    cubes = [[] for _ in range(cores)]
    front = 1 if forward else 2
    for lane in range(cores):
        for stage in cube_stages[:front]:
            cubes[lane].extend(tasks(stage, 0, lane))
        for expert in range(1, spec.local_experts):
            for stage in cube_stages[:front]:
                cubes[lane].extend(tasks(stage, expert, lane))
            for stage in cube_stages[front:]:
                cubes[lane].extend(tasks(stage, expert - 1, lane))
        for stage in cube_stages[front:]:
            cubes[lane].extend(tasks(stage, spec.local_experts - 1, lane))
        if len(vector_stages) == 1:
            for expert in range(spec.local_experts):
                vectors[2 * lane].extend(tasks(vector_stages[0], expert, lane))
        else:
            vectors[2 * lane].extend(tasks(vector_stages[0], 0, lane))
            for expert in range(spec.local_experts):
                if expert + 1 < spec.local_experts:
                    vectors[2 * lane].extend(tasks(vector_stages[0], expert + 1, lane))
                vectors[2 * lane].extend(tasks(vector_stages[1], expert, lane))
    # Communication and termination never use speculative ready selection.
    return install_worker_queues(cfg, cubes, vectors, cube_ready_depth=spec.ready_depth,
                                 vector_ready_depths=[spec.ready_depth if i % 2 == 0 else 0
                                                      for i in range(2 * cores)])
