# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Compose forward/backward expert DAGs from standard MX mathematical tasks."""
from dataclasses import dataclass
from typing import Any
from hyper_multicore.scheduler.config import (TaskSplitValue, ATOMIC_ADD_VALUE_LEN, MIN_EVENT_CAPACITY,
    init_task_split_value, configure_ready_handshake, validate_runtime_config, mega_moe_event_capacity)
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, TensorSpec, SplitSpec
from hyper_multicore.scheduler.runtime import allocate_runtime_config
from hyper_multicore.scheduler.scheduler import revise_task_queue
from hyper_multicore.tasks.alltoall import AllToAllFillConfig, AllToAllType
from hyper_multicore.tasks.gmm_mxfp8 import serialize_mx_gmm_descriptors
from hyper_multicore.tasks.quantization_mxfp8 import (MX_QUANT_TASK, MX_SWIGLU_TASK, MX_SWIGLU_GRAD_TASK,
    serialize_mx_quant_descriptors)
from hyper_multicore.tasks.gmm_mxfp8 import Mxfp8GmmFillConfig
from hyper_multicore.tasks.quantization_mxfp8 import Mxfp8QuantFillConfig
from hyper_multicore.tasks.swiglu_mxfp8 import Mxfp8SwiGLUFillConfig
from hyper_multicore.tasks.utils import add_dynamic_data, add_terminate
from ...spec import MegaMoeSpec
from ...schedule import configure_expert_queues


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
    transport_schedule: str = 'QP1'
    native_operator: str = 'HyperMegaMoeMxfp8'


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
    if not forward and spec.partial_weight_grad_dtype == "float32":
        for index in (6, 18):
            result[index].dtype_size = 4
    metadata = ([1, 3, 13, 14] if forward else [1, 3, 14, 15])
    sizes = [4, 15] if forward else [4, 16]
    for index in metadata + sizes:
        result[index].shape = [spec.num_experts]
        result[index].dtype_size = 4 if index in sizes else 8
    for index in ([6, 10] if forward else [19]):
        result[index].shape, result[index].dtype_size = [experts], 8
    # Only the transport worker uses dynamic tensor offsets. MX tasks
    # resolve token and packed-scale ranges directly from the live group list.
    result[0].is_dynamic = True
    payload = 11 if forward else 12
    result[payload].is_dynamic, result[payload].tensor_type = True, 1
    return result


def build_mxfp8_runtime(spec: MegaMoeSpec, *, forward: bool) -> MXFP8Runtime:
    """Wire per-expert dependencies; preserve adopted math, transport and task boundaries."""
    if spec.swiglu_limit is not None:
        raise ValueError("MX tasks currently require unclamped activation")
    if spec.local_num_tokens % spec.dispatch_split or spec.local_num_tokens % spec.combine_split:
        raise ValueError("MX communication splits must divide the fixed local token bound")
    values = _MXTaskValues(tp=1, ep=spec.ep_size, seq_size=spec.local_num_tokens,
                           all_expert_num=spec.num_experts, top_k=spec.top_k,
                           compute_stages=4 if forward else 6, dispatch_mode=spec.dispatch_mode)
    specs, graph, gmms, quants = _buffer_specs(spec, forward), ComputeGraph(), [], []
    _add_transport(graph, specs, forward=forward, dispatch_split=spec.dispatch_split, combine_split=spec.combine_split)
    h, width, capacity, cubes = spec.hidden_size, spec.intermediate_size, spec.receive_capacity, spec.num_cube_cores
    vectors = 2 * cubes
    glist, gtable, qtable = (6, 25, 26) if forward else (19, 31, 32)

    def quant(name, kind, columns, inputs, outputs, first=False):
        _add_compute(graph, name, specs, inputs + [glist, qtable], outputs,
                     (Mxfp8QuantFillConfig if kind == MX_QUANT_TASK else Mxfp8SwiGLUFillConfig)(
                         kind, columns, vectors, first, quants), vectors,
                     {MX_QUANT_TASK: OpType.QUANTIZE, MX_SWIGLU_TASK: OpType.SWIGLU_QUANT,
                      MX_SWIGLU_GRAD_TASK: OpType.SWIGLU_GRAD_QUANT}[kind])

    def gmm(name, layout, dims, inputs, output, scales):
        _add_compute(graph, name, specs, inputs + [glist, gtable], [output],
                     Mxfp8GmmFillConfig(layout, *dims, *scales, cubes, gmms), cubes)

    if forward:
        quant('quant_x', MX_QUANT_TASK, h, [0], [16, 17, 18, 19], first=True)
        gmm('up', 'NN', (capacity, 2 * width, h), [16, 5], 7, (17, 23))
        quant('swiglu_quant', MX_SWIGLU_TASK,
              width, [7], [8, 20, 21, 22])
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
    for left, right in zip(order, order[1:]):
        graph.add_edge(left, right)
    graph.propagate_splits(values)
    tasks = 1 + sum(op.task_num for op in graph.topological_sort())
    event_capacity = mega_moe_event_capacity(spec.num_experts, spec.ep_size, compute_stages=6)
    cfg = allocate_runtime_config(tasks, event_capacity, spec.local_experts, num_workers=2 * cubes)
    init_task_split_value(values)
    values.rank_id = spec.rank_id
    for op in graph.topological_sort():
        op.fill_config.fill(cfg, op, values)
    combine = graph.get_op('combine')
    add_terminate(cfg, values, combine.task_num)
    middle_vectors = sum(graph.get_op(name).task_num for name in order
                         if name not in ('dispatch', 'combine') and graph.get_op(name).op_type in (OpType.QUANTIZE, OpType.SWIGLU_QUANT, OpType.SWIGLU_GRAD_QUANT))
    revise_task_queue(cfg, values, graph.get_op('dispatch').task_num, middle_vectors, combine.task_num)
    add_dynamic_data(cfg, values, glist)
    cfg.task_num = sum(op.task_num for op in graph.topological_sort())
    cfg.atomic_add_values[0] = 1
    configure_ready_handshake(cfg, values)
    validate_runtime_config(cfg, values, cubes)
    cfg = configure_expert_queues(cfg, graph, spec, forward=forward)
    native_operator = 'HyperMegaMoeMxfp8' if forward else 'HyperMegaMoeMxfp8Grad'
    return MXFP8Runtime(graph, cfg, serialize_mx_gmm_descriptors(gmms),
                        serialize_mx_quant_descriptors(quants), 27 if forward else 33,
                        f'{spec.schedule_policy}/QP1', native_operator)
