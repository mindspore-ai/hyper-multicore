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
"""Real GM operand bindings for the shifted mHC forward kernel."""

from math import prod
from typing import Any

from hyper_multicore.diagnostics.common.graph import bind_graph_tensor_ports
from hyper_multicore.diagnostics.dumper import TaskDumpTensorView, TaskDumpTensorViewContext
from hyper_multicore.scheduler.graph import TensorSpec


def tensor_port(name: str, position: int, dtype_size: int = 4) -> TensorSpec:
    """Describe a native operand whose actual shape is supplied at launch.

    Args:
        name: Logical diagnostic name of the native operand.
        position: Zero-based position in the native ABI parameter list.
        dtype_size: Expected element size in bytes.
    """
    return TensorSpec(name, [], position, dtype_size=dtype_size, is_dynamic=True)


def resolve_token_view(context: TaskDumpTensorViewContext) -> TaskDumpTensorView:
    """Resolve a contiguous token tile, not a native CANN workspace alias.

    Args:
        context: Selected invocation or operand context supplied by the diagnostic framework.
    """
    tensor = context.parameters[context.tensor_spec.param_position]
    shape = tuple(tensor.shape)
    # Weights and scalar parameters are shared by every task.
    if context.tensor_spec.name in ("phi", "alpha", "bias", "norm_weight"):
        return context.make_view(shape, source_offset_bytes=0)
    token_count = int(context.parameters[1].shape[1])
    offset = int(context.task_desc.task_index) * int(context.task_desc.task_split_value)
    rows = max(0, min(int(context.task_desc.task_split_value), token_count - offset))
    if len(shape) >= 2 and shape[0] == 1 and shape[1] == token_count:
        row_shape = shape[2:]
    elif shape and shape[0] == token_count:
        row_shape = shape[1:]
    else:
        raise ValueError(f"mHC operand {context.tensor_spec.name} has no contiguous token axis: {shape}")
    return context.make_view((rows, *row_shape),
                             source_offset_bytes=offset * prod(row_shape) * tensor.element_size())


def bind_forward_graph(graph: Any, config: Any) -> Any:
    """Keep scheduling edges but replace placeholder operands by real ABI ports.

    The ring X-cast workspace is CANN-owned, not hc_before_norm. It is deliberately
    not exposed as a Tensor. Iteration-major Sinkhorn caches are also omitted:
    a token tile there is strided, whereas Task currently copies contiguous ranges.

    Args:
        graph: Business compute graph whose scheduling storage remains unchanged.
        config: Private scheduling configuration; port binding does not rewrite tasks.
    """
    del config
    p = tensor_port
    previous = p("previous_output", 0, 2)
    residual = p("residual", 1, 2)
    updated = p("new_residual", 12, 2)
    projection = p("hc_before_norm", 17)
    mixed = p("mixed_input", 21, 2)
    ports = {
        "mhc_post": ([previous, residual, p("previous_post_mix", 3), p("previous_residual_mix", 4)], [updated]),
        "mhc_norm_cast": ([updated], [p("inv_rms", 18)]),
        "mhc_projection": ([p("phi", 5)], [projection]),
        "mhc_input_mix": ([updated, p("previous_pre_mix", 2)], [mixed]),
        "mhc_rms_norm": ([mixed, p("norm_weight", 8, 2)], [p("block_input", 16, 2), p("rms_rstd", 22)]),
        "mhc_mapping": ([projection, p("inv_rms", 18), p("alpha", 6), p("bias", 7)],
                        [p("next_pre_mix", 13), p("next_post_mix", 14), p("next_residual_mix", 15)]),
    }
    return bind_graph_tensor_ports(graph, ports, resolve_token_view)
