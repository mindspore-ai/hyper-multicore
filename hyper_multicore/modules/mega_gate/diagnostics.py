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
"""MegaGate broadcast task identities and native GM operand bindings."""

from typing import Any

from hyper_multicore._core_spec import get_device_core_spec
from hyper_multicore.diagnostics.common.graph import bind_graph_tensor_ports
from hyper_multicore.diagnostics.dump.l1 import L1TensorView, L1TensorViewContext
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec


MEGA_GATE_OWNER_LABEL = "TokenRowShard"
MEGA_GATE_ROUTE_STAGE_NAMES = (
    "Softplus",
    "Sqrt",
    "AddBias",
    "TopK",
    "Gather",
    "ReduceSum",
    "AddEpsilon",
    "Div",
    "MulScale",
    "CastIndex",
)

MEGA_GATE_ROUTE_GRAD_STAGE_NAMES = (
    "MulsScaleGrad",
    "BroadcastDenominator",
    "Neg",
    "DivSelected",
    "DivSelectedRatio",
    "MulCross",
    "DivDirect",
    "ReduceSum",
    "BroadcastRowSum",
    "AddSelectedGrad",
    "ZerosLike",
)

MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES = (
    "MulsScaleGrad",
    "ZerosLike",
)



def _port(name: str, position: int, dtype_size: int = 4) -> TensorSpec:
    """Bind one native GM operand; shapes are invocation-owned."""
    return TensorSpec(name, [], position, dtype_size=dtype_size, is_dynamic=True)


def _row_view(context: L1TensorViewContext) -> L1TensorView:
    """Mirror Route/RouteGrad tiling without running a Device-side resolver."""
    tensor = context.parameters[context.tensor_spec.param_position]
    shape = tuple(tensor.shape)
    tokens = int(context.parameters[0].shape[0])
    if context.tensor_spec.name in ("text_bias", "vision_bias") or (
        context.tensor_spec.name == "image_mask" and tensor.numel() == 1
    ):
        return context.make_view(shape, source_offset_bytes=0)
    cache_key = -1
    if cache_key not in context.parameter_cache:
        cores = int(context.runtime_config.num_workers)
        if tensor.device.type == "npu":
            cores = min(cores, get_device_core_spec(tensor.device.index).num_vector_cores)
        workers = min(tokens, cores)
        context.parameter_cache[cache_key] = (tokens + workers - 1) // workers
    rows_per_worker = context.parameter_cache[cache_key]
    offset = context.core_id * rows_per_worker
    rows = max(0, min(rows_per_worker, tokens - offset))
    if rows == 0:
        return context.make_view((0, *shape[1:]), source_offset_bytes=0)
    row_elements = tensor.numel() // tokens
    return context.make_view((rows, *shape[1:]),
                             source_offset_bytes=offset * row_elements * tensor.element_size())


def bind_pipeline_graph(graph: Any, config: Any, *, backward: bool = False) -> ComputeGraph:
    """Build diagnostic ports only; native stages still use their original ABI.

    RouteGrad's scatter/sqrt/softplus tail is launched by ACLNN after the MegaKernel.
    A zero-task boundary node describes its L0 result, not an L1 Device capture.
    Internal CANN workspace arrays are deliberately not advertised as ABI Tensors.

    Args:
        graph: Graph placeholder replaced by the lazily constructed broadcast graph.
        config: Configuration consumed by this callback.
        backward: Whether to prepare the RouteGrad pipeline instead of Route.
    """
    p = _port
    if backward:
        names = MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES if config.task_num == 2 else MEGA_GATE_ROUTE_GRAD_STAGE_NAMES
        bindings = {name: ([], []) for name in names}
        bindings["MulsScaleGrad"] = ([p("grad_routing_weights", 2)], [])
        if config.task_num != 2:
            bindings["BroadcastDenominator"] = ([p("normalization_denominator", 1)], [])
            bindings["DivSelected"] = ([p("selected_scores", 0)], [])
        bindings["RouteGradPost"] = (
            [p("logits", 10), p("route_scores", 3), p("expert_indices", 4, 8),
             p("selected_scores", 0), p("normalization_denominator", 1)], [p("grad_logits", 9)],
        )
        names = (*names, "RouteGradPost")
    else:
        names = MEGA_GATE_ROUTE_STAGE_NAMES
        score, selected, denominator = p("route_scores", 8), p("selected_scores", 9), p("normalization_denominator", 10)
        bindings = {
            "Softplus": ([p("logits", 0)], []),
            "Sqrt": ([], [score]),
            "AddBias": ([score, p("text_bias", 1), p("vision_bias", 2), p("image_mask", 3, 1)], []),
            "TopK": ([], []), "Gather": ([score], [selected]), "ReduceSum": ([selected], []),
            "AddEpsilon": ([], [denominator]), "Div": ([selected, denominator], []),
            "MulScale": ([], [p("routing_weights", 6)]), "CastIndex": ([], [p("expert_indices", 7, 8)]),
        }
    graph = ComputeGraph()
    previous = None
    for index, name in enumerate(names):
        node = OperatorNode(name, OpType.AIV_PIPELINE, [], [], [], 0, SplitSpec(None, lambda _: 1), 0, None,
                            diagnostic_name=name)
        node.task_num = 1 if index < config.task_num else 0
        graph.add_op(node)
        if previous is not None:
            graph.add_edge(previous, node)
        previous = node
    return bind_graph_tensor_ports(graph, bindings, _row_view)
