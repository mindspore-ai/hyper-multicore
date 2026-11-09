# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Independent MXFP8 expert task graphs sharing the established BF16 transport."""

from dataclasses import dataclass
from typing import Any

from hyper_multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN, EVENT_INVALID_ID, MIN_EVENT_CAPACITY, TaskAiCoreType, TaskDescC, TaskSplitValue, TaskType, configure_ready_handshake,
    init_task_split_value, validate_runtime_config,
)
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_multicore.scheduler.mxfp8 import (
    MX_GMM_TASK_TYPE, make_mx_gmm_descriptor, serialize_mx_gmm_descriptors,
)
from hyper_multicore.scheduler.mxfp8_quant import (
    MX_QUANT_TASK, MX_SWIGLU_TASK, MX_SWIGLU_GRAD_TASK, MX_SWIGLU_ROW_TASK,
    MxQuantDescriptor, serialize_mx_quant_descriptors,
)
from hyper_multicore.scheduler.runtime import allocate_runtime_config
from hyper_multicore.scheduler.scheduler import revise_task_queue
from hyper_multicore.scheduler.expert_window import reorder_mx_expert_windows
from hyper_multicore.scheduler.all_transport import balance_all_transport
from hyper_multicore.scheduler.vector_roles import configure_vector_roles, configure_rule_stages
from hyper_multicore.scheduler.dispatch_quant import DispatchQuantStage, configure_dispatch_quant_queue
from hyper_multicore.tasks.alltoall import AllToAllFillConfig, AllToAllType
from hyper_multicore.tasks.utils import (
    add_dynamic_data, add_terminate, advance_tsv_cube, advance_tsv_vector, make_tensor_desc,
)

from .spec import MegaMoeSpec


@dataclass
class _MXTaskValues(TaskSplitValue):
    compute_stages: int = 4

    @property
    def all_event_num(self) -> int:
        return 1 + self.all_expert_num + self.compute_stages * self.single_rank_expert_num


@dataclass(frozen=True)
class MXFP8Runtime:
    graph: ComputeGraph
    config: Any
    gmm_descriptors: bytes
    quant_descriptors: bytes
    buffer_count: int
    swiglu_row_only: bool = False
    transport_schedule: str = 'legacy_odd'
    native_operator: str = 'HyperMegaMoeMxfp8'


def _fill_compute_stage(cfg: Any, op: OperatorNode, tsv: TaskSplitValue,
                        kind: int, workers: int, first: bool, indices: list[int]) -> None:
    cube = kind == MX_GMM_TASK_TYPE
    for index, record_index in enumerate(indices):
        expert = index // workers
        task = TaskDescC()
        task.task_type = kind
        task.task_aicore_type = (TaskAiCoreType.TASK_AICORE_CUBE if cube else TaskAiCoreType.TASK_AICORE_VECTOR)
        task.num_inputs, task.num_outputs = len(op.inputs), len(op.outputs)
        for slot, spec in enumerate(op.inputs):
            task.inputs[slot] = make_tensor_desc(input_position=spec.param_position,
                                                data_type=spec.dtype_size, shape=spec.shape)
        for slot, spec in enumerate(op.outputs):
            task.outputs[slot] = make_tensor_desc(input_position=spec.param_position,
                                                 data_type=spec.dtype_size, shape=spec.shape)
        rank_offset = tsv.rank_id * tsv.single_rank_expert_num if first else 0
        task.dependent_event = tsv.pre_pre_event_num + rank_offset + expert + 1
        task.trigger_event = tsv.pre_event_num + expert + 1
        cfg.all_event_num_triggers[task.trigger_event] = workers
        task.task_index, task.task_split_num = index, len(indices)
        task.task_split_value, task.extra_value_0 = 1, record_index
        task.tiling_data_position = 0xFFFFFFFF
        task_index = tsv.pre_task_num + index
        cfg.all_tasks[task_index] = task
        queue = cfg.cube_task_indices if cube else cfg.vector_task_indices
        offset = tsv.pre_cube_task_num if cube else tsv.pre_vector_task_num
        queue[offset + index] = task_index
    cfg.task_index_num[0 if cube else 1] += len(indices)
    advance = advance_tsv_cube if cube else advance_tsv_vector
    advance(tsv, len(indices), tsv.single_rank_expert_num)


@dataclass
class _GmmStage:
    layout: str
    m: int
    n: int
    k: int
    x_scale: int
    w_scale: int
    workers: int
    records: list

    def fill(self, cfg: Any, op: OperatorNode, tsv: TaskSplitValue) -> None:
        indices = []
        for expert in range(tsv.single_rank_expert_num):
            for worker in range(self.workers):
                indices.append(len(self.records))
                self.records.append(make_mx_gmm_descriptor(
                    layout=self.layout, m=self.m, n=self.n, k=self.k,
                    x_scale_position=self.x_scale, w_scale_position=self.w_scale,
                    cube_count=self.workers, cube_index=worker, group_list_offset=expert * 8,
                    w_offset=0 if self.layout == 'TN' else expert * self.n * self.k,
                    w_scale_offset=0 if self.layout == 'TN' else expert * self.n * self.k // 32,
                    y_offset=expert * self.m * self.n * 2 if self.layout == 'TN' else 0,
                ))
        _fill_compute_stage(cfg, op, tsv, MX_GMM_TASK_TYPE, self.workers, False, indices)


@dataclass
class _QuantStage:
    kind: int
    columns: int
    workers: int
    first: bool
    records: list

    def fill(self, cfg: Any, op: OperatorNode, tsv: TaskSplitValue) -> None:
        indices = []
        for expert in range(tsv.single_rank_expert_num):
            for worker in range(self.workers):
                indices.append(len(self.records))
                self.records.append(MxQuantDescriptor(
                    columns=self.columns, expert_index=expert, worker_index=worker,
                    worker_count=self.workers, task_type=self.kind,
                ))
        _fill_compute_stage(cfg, op, tsv, self.kind, self.workers, self.first, indices)


def _add_compute(graph: ComputeGraph, name: str, specs: dict[int, TensorSpec],
                 inputs: list[int], outputs: list[int], fill: Any, workers: int,
                 op_type: OpType = OpType.GMM) -> None:
    graph.add_op(OperatorNode(
        name=name, op_type=op_type, inputs=[specs[index] for index in inputs],
        outputs=[specs[index] for index in outputs], param_positions=inputs + outputs,
        split_value=1, split_spec=SplitSpec(
            split_inputs=None, split_output_dims=[0] * len(outputs),
            task_num_fn=lambda tsv: tsv.single_rank_expert_num * workers,
        ), tiling_position=-1, fill_config=fill,
    ))


def _add_transport(graph: ComputeGraph, specs: dict[int, TensorSpec], *, forward: bool,
                   dispatch_split: int, combine_split: int) -> None:
    source, target, size, destination = (13, 14, 15, 12) if forward else (14, 15, 16, 13)
    payload = 11 if forward else 12
    for name, inputs, output, split, kind, group in (
        ('dispatch', [1, 2, 3, 4], 0, dispatch_split, AllToAllType.DISPATCH, 'dispatch'),
        ('combine', [source, payload, target, size], destination, combine_split, AllToAllType.COMBINE, 'combine'),
    ):
        fill = AllToAllFillConfig(moe_type=kind, advance='vector', event_group=1)
        graph.add_op(OperatorNode(
            name=name, op_type=OpType.ALLTOALL, inputs=[specs[index] for index in inputs],
            outputs=[specs[output]], param_positions=inputs + [output], split_value=split,
            split_spec=SplitSpec(split_inputs=None, split_output_dims=[0],
                                 task_num_fn=lambda tsv, tile=split: tsv.all_expert_num * (tsv.seq_size // tile)),
            tiling_position=-1, fill_config=fill,
        ))
        if group == 'dispatch':
            fill.event_group = specs[1].shape[0]


def _buffer_specs(spec: MegaMoeSpec, forward: bool) -> dict[int, TensorSpec]:
    h, width, rows, experts = spec.hidden_size, spec.intermediate_size, spec.routed_slots, spec.local_experts
    count = 27 if forward else 33
    result = {index: TensorSpec(f'buffer_{index}', [rows, h], index, dtype_size=1) for index in range(count)}
    ordinary = ({0: h, 2: h, 7: 2 * width, 11: h, 12: h} if forward
                else {0: h, 2: h, 6: h, 8: width, 9: 2 * width, 12: h, 13: h, 18: 2 * width})
    for index, columns in ordinary.items():
        result[index].shape, result[index].dtype_size = [rows, columns], 2
    metadata = ([1, 3, 13, 14] if forward else [1, 3, 14, 15])
    sizes = [4, 15] if forward else [4, 16]
    for index in metadata + sizes:
        result[index].shape = [spec.num_experts]
        result[index].dtype_size = 4 if index in sizes else 8
    for index in ([6, 10] if forward else [19]):
        result[index].shape, result[index].dtype_size = [experts], 8
    # Only the transport worker uses legacy dynamic tensor offsets. MX tasks
    # resolve token and packed-scale ranges directly from the live group list.
    result[0].is_dynamic = True
    payload = 11 if forward else 12
    result[payload].is_dynamic, result[payload].tensor_type = True, 1
    return result


def _apply_mx_poll_policy(cfg: Any, policy: str) -> None:
    """Use queue roles because transport descriptor core tags do not define scheduling."""
    if policy not in ('0', 'fast'):
        raise ValueError('mx_poll_policy must be exactly 0 or fast.')
    if policy == '0':
        return
    seen = set()
    for indices, count, interval in zip((cfg.cube_task_indices, cfg.vector_task_indices),
                                         cfg.task_index_num[:2], (10, 30)):
        for task_id in indices[:count]:
            if task_id in seen:
                raise ValueError(f'MX poll policy requires unique queue membership: task {task_id}.')
            seen.add(task_id)
            task = cfg.all_tasks[task_id]
            if task.dependent_event != EVENT_INVALID_ID:
                task.extra_value_2 = interval


def build_mxfp8_runtime(spec: MegaMoeSpec, *, forward: bool, swiglu_row_only: bool = False,
                       expert_window: int = 0, dispatch_quant: bool = False,
                       aggregate_empty_credits: bool = False, ready_depth: int = 0) -> MXFP8Runtime:
    """Serialize phase-specific MX tasks with per-expert producer/consumer events."""
    if swiglu_row_only and not forward:
        raise ValueError('Row-only SwiGLU is valid only for a forward without saved backward state.')
    if spec.dispatch_mode not in ('push', 'pull') or spec.swiglu_limit is not None:
        raise ValueError('MXFP8 task graphs require push/pull transport and unclamped SwiGLU.')
    if spec.dispatch_mode == 'pull' and spec.mx_vector_policy != 'window2':
        raise ValueError('MX Pull requires window2 source-completion queues')
    if ready_depth not in (0, 1, 8) or (ready_depth and spec.mx_vector_policy != 'window2'):
        raise ValueError('Ready depth must be 0, 1 or 8 on window2')
    role_policy = spec.mx_vector_policy in ('roles', 'roles_ready', 'roles_rule', 'planned', 'window2')
    ready_policy = spec.mx_vector_policy == 'roles_ready'
    if role_policy and (spec.num_cube_cores != 32
                        or (swiglu_row_only and spec.mx_vector_policy != 'window2') or expert_window
                        or spec.dispatch_split != spec.combine_split):
        raise ValueError('Vector roles require Cube32, dual-axis, equal transport splits and no expert window.')
    values = _MXTaskValues(tp=1, ep=spec.ep_size, seq_size=spec.local_num_tokens,
                           all_expert_num=spec.num_experts, top_k=spec.top_k,
                           compute_stages=4 if forward else 6, dispatch_mode=spec.dispatch_mode)
    specs, graph, gmms, quants = _buffer_specs(spec, forward), ComputeGraph(), [], []
    _add_transport(graph, specs, forward=forward, dispatch_split=spec.dispatch_split, combine_split=spec.combine_split)
    h, width, capacity, cubes = spec.hidden_size, spec.intermediate_size, spec.receive_capacity, spec.num_cube_cores
    vectors = 2 * cubes
    glist, gtable, qtable = (6, 25, 26) if forward else (19, 31, 32)
    dispatch_quant_stage = None
    fragments = spec.local_num_tokens // spec.dispatch_split
    event_base = (values.all_event_num + 3 + 2 * ATOMIC_ADD_VALUE_LEN + 15) // 16 * 16
    if dispatch_quant:
        if swiglu_row_only:
            raise ValueError('Dispatch quant prototype requires the rebuilt dual-axis entry; row-only is not built.')
        if expert_window or spec.mx_vector_policy not in ('roles', 'roles_rule', 'planned') or spec.dispatch_split % 64:
            raise ValueError('Dispatch quant requires roles, roles_rule or planned, no expert window '
                             'and a 64-row aligned split.')
        if event_base + spec.local_experts * (fragments + 1) + ATOMIC_ADD_VALUE_LEN > MIN_EVENT_CAPACITY:
            raise ValueError('Dispatch quant exceeds the existing event budget; SHMEM expansion is forbidden.')

    def quant(name: str, kind: int, columns: int, inputs: list[int], outputs: list[int], first: bool = False) -> None:
        nonlocal dispatch_quant_stage
        if first and dispatch_quant:
            dispatch_quant_stage = DispatchQuantStage(
                columns, vectors, fragments, spec.dispatch_split, event_base,
                spec.ep_size, spec.local_experts, 14 if forward else 15, 15 if forward else 16,
                quants, _fill_compute_stage)
            _add_compute(graph, name, specs, inputs + [glist, qtable], outputs,
                         dispatch_quant_stage, fragments * 4 + vectors, OpType.SWIGLU)
            return
        _add_compute(graph, name, specs, inputs + [glist, qtable], outputs,
                     _QuantStage(kind, columns, vectors, first, quants), vectors, OpType.SWIGLU)

    def gmm(name: str, layout: str, dims: tuple[int, int, int], inputs: list[int], output: int,
            scales: tuple[int, int]) -> None:
        _add_compute(graph, name, specs, inputs + [glist, gtable], [output],
                     _GmmStage(layout, *dims, *scales, cubes, gmms), cubes)

    if forward:
        quant('quant_x', MX_QUANT_TASK, h, [0], [16, 17, 18, 19], first=True)
        gmm('up', 'NN', (capacity, 2 * width, h), [16, 5], 7, (17, 23))
        quant('swiglu_quant', MX_SWIGLU_ROW_TASK if swiglu_row_only else MX_SWIGLU_TASK,
              width, [7], [8, 20] if swiglu_row_only else [8, 20, 21, 22])
        gmm('down', 'NN', (capacity, h, width), [8, 9], 11, (20, 24))
        order = ['dispatch', 'quant_x', 'up', 'swiglu_quant', 'down', 'combine']
    else:
        quant('quant_dy', MX_QUANT_TASK, h, [0], [20, 21, 22, 23], first=True)
        gmm('dw2', 'TN', (width, h, capacity), [5, 22], 6, (24, 23))
        gmm('da', 'NT', (capacity, width, h), [20, 7], 8, (21, 25))
        quant('swiglu_grad_quant', MX_SWIGLU_GRAD_TASK, 2 * width, [9, 8], [10, 26, 27, 28])
        gmm('dw1', 'TN', (h, 2 * width, capacity), [17, 27], 18, (30, 28))
        gmm('dx', 'NT', (capacity, h, 2 * width), [10, 11], 12, (26, 29))
        order = ['dispatch', 'quant_dy', 'dw2', 'da', 'swiglu_grad_quant', 'dw1', 'dx', 'combine']
    edges = list(zip(order, order[1:]))
    if ready_policy and not forward:
        edges = [('dispatch', 'quant_dy'), ('quant_dy', 'dw2'), ('quant_dy', 'da'),
                 ('da', 'swiglu_grad_quant'), ('swiglu_grad_quant', 'dw1'),
                 ('swiglu_grad_quant', 'dx'), ('dw2', 'combine'), ('dw1', 'combine'), ('dx', 'combine')]
    for left, right in edges:
        graph.add_edge(left, right)
    graph.propagate_splits(values)
    tasks = 1 + sum(op.task_num for op in graph.topological_sort())
    required_events = values.all_event_num + 3 + 2 * ATOMIC_ADD_VALUE_LEN
    event_capacity = max(MIN_EVENT_CAPACITY, (required_events + 15) // 16 * 16)
    cfg = allocate_runtime_config(tasks, event_capacity, spec.local_experts, num_workers=2 * cubes)
    init_task_split_value(values)
    values.rank_id = spec.rank_id
    for op in graph.topological_sort():
        op.fill_config.fill(cfg, op, values)
    combine = graph.get_op('combine')
    add_terminate(cfg, values, combine.task_num)
    middle_vectors = sum(graph.get_op(name).task_num for name in order
                         if name not in ('dispatch', 'combine') and graph.get_op(name).op_type == OpType.SWIGLU)
    revise_task_queue(cfg, values, graph.get_op('dispatch').task_num, middle_vectors, combine.task_num)
    reorder_mx_expert_windows(cfg, graph, spec.local_experts, expert_window, forward)
    transport_schedule = balance_all_transport(cfg, graph, spec)
    if dispatch_quant_stage is not None:
        configure_dispatch_quant_queue(cfg, graph, dispatch_quant_stage, spec.rank_id)
        transport_schedule = 'dispatch_quant: source-fragment waves, no padding'
    add_dynamic_data(cfg, values, glist)
    cfg.task_num = sum(op.task_num for op in graph.topological_sort())
    cfg.atomic_add_values[0] = 1
    configure_ready_handshake(cfg, values)
    if spec.dispatch_mode == 'pull':
        cfg.all_event_num_triggers[cfg.completion_event] = 3 * cubes
    validate_runtime_config(cfg, values, cubes)
    if aggregate_empty_credits:
        if spec.mx_vector_policy != 'window2':
            raise ValueError('Empty credit aggregation requires the qualified window2 entry.')
        for task in cfg.all_tasks:
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                task.extra_value_1 = 1
    _apply_mx_poll_policy(cfg, spec.mx_poll_policy)
    if ready_depth:
        for task in cfg.all_tasks:
            if int(task.task_type) in (142, 143, 145, 146, 147):
                task.extra_value_1 |= 32 if ready_depth == 8 else 64
    if role_policy:
        configure_vector_roles(cfg, graph, ready=ready_policy, forward=forward)
        transport_schedule = spec.mx_vector_policy + ': odd communication, even mathematics, no padding'
    if spec.mx_vector_policy == 'roles_rule':
        configure_rule_stages(cfg, graph, forward=forward)
        transport_schedule = 'roles_rule: explicit ready heads, fair dispatch/combine'
    if spec.mx_vector_policy == 'planned':
        from hyper_multicore.scheduler.planned_queue import configure_planned_queue
        configure_planned_queue(cfg, spec, forward=forward)
        transport_schedule = 'planned CSR: fixed odd communication, arbitrary AIV mathematics'
    if spec.mx_vector_policy == 'window2':
        from hyper_multicore.scheduler.window_queue import configure_window_queue
        configure_window_queue(cfg, graph, spec, forward=forward)
        if ready_depth and (cfg.task_index_num[0] // 32 > 64 or any(
                cfg.mix_task_indices[w + 1] - cfg.mix_task_indices[w] > 64 for w in range(0, 64, 2))):
            raise ValueError('Ready math queues support at most 64 tasks per worker')
        transport_schedule = 'window2 CSR: odd communication, even math, two-expert Cube lookahead'
    native_operator = 'HyperMegaMoeMxfp8' if forward else 'HyperMegaMoeMxfp8Grad'
    return MXFP8Runtime(graph, cfg, serialize_mx_gmm_descriptors(gmms),
                        serialize_mx_quant_descriptors(quants), 27 if forward else 33, swiglu_row_only, transport_schedule,
                        native_operator)
