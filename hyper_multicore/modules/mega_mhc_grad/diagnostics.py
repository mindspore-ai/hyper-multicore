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
"""Native backward operand bindings; reduction outputs are Kernel-only."""

from math import prod
from typing import Any

from hyper_multicore.diagnostics.common.graph import bind_graph_tensor_ports
from hyper_multicore.diagnostics.dumper import TaskDumpTensorView, TaskDumpTensorViewContext
from hyper_multicore.scheduler.graph import TensorSpec


def resolve_backward_view(context: TaskDumpTensorViewContext) -> TaskDumpTensorView:
    """Capture local token ranges, never a concurrently accumulated parameter.

    Args:
        context: Selected invocation or operand context supplied by the diagnostic framework.
    """
    position = context.tensor_spec.param_position
    tensor = context.parameters[position]
    if context.io == "output" and position in (25, 26, 27, 32):
        raise ValueError(
            "mHC parameter gradients are multi-core reductions; use Kernel dump or select token-local Task dump outputs"
        )
    shape = tuple(tensor.shape)
    token_count = int(context.parameters[3].shape[1])
    if position in (4, 5, 6, 10, 11, 15):
        return context.make_view(shape, source_offset_bytes=0, note="shared read-only parameter or forward cache")
    if len(shape) >= 2 and shape[0] == 1 and shape[1] == token_count:
        row_shape = shape[2:]
    elif shape and shape[0] == token_count:
        row_shape = shape[1:]
    else:
        raise ValueError(f"mHC backward operand {context.tensor_spec.name} has no contiguous token axis: {shape}")
    offset = int(context.task_desc.task_index) * int(context.task_desc.task_split_value)
    rows = max(0, min(int(context.task_desc.task_split_value), token_count - offset))
    return context.make_view((rows, *row_shape),
                             source_offset_bytes=offset * prod(row_shape) * tensor.element_size())


def bind_backward_graph(graph: Any, config: Any) -> Any:
    """Describe true ABI Tensors, leaving native workspace-only edges unexposed.

    Args:
        graph: Business compute graph whose scheduling storage remains unchanged.
        config: Private scheduling configuration; port binding does not rewrite tasks.
    """
    del config
    names = (
        "grad_block_input", "grad_next_post_mix", "grad_next_residual_mix", "new_residual",
        "phi", "alpha", "bias", "previous_pre_mix", "hc_before_norm", "inv_rms", "sum_out", "norm_out",
        "grad_next_pre_mix", "mixed_input", "rms_rstd", "norm_weight", "grad_new_residual", "residual",
        "previous_output", "previous_post_mix", "previous_residual_mix", "runtime", "events", "diagnostic",
        "grad_residual", "grad_phi", "grad_alpha", "grad_bias", "grad_previous_output", "grad_previous_pre_mix",
        "grad_previous_post_mix", "grad_previous_residual_mix", "grad_norm_weight",
    )
    bf16 = {0, 3, 13, 15, 16, 17, 18, 24, 28}

    def ports(indices: tuple[int, ...]) -> list[Any]:
        """Map native positions to actual operand dtypes without workspace aliases.

        Args:
            indices: ABI positions selected from the backward operand table.
        """
        return [
            TensorSpec(names[index], [], index, dtype_size=2 if index in bf16 else 4, is_dynamic=True)
            for index in indices
        ]

    bindings = {
        "grad_output_init": ([], []),
        "rms_norm_grad": (ports((0, 13, 14, 15)), ports((32,))),
        "mhc_grad_prev_a_and_mapping": (ports((3, 7, 12, 1, 2, 5, 6, 8, 9, 10, 11)), ports((29,))),
        "mhc_grad_phi_rms": (ports((3, 4, 8, 9)), ports((25, 26, 27))),
        "mhc_grad_prev_x_and_post": (ports((3, 4, 16, 17, 18, 19, 20)), ports((24, 28, 30, 31))),
    }
    return bind_graph_tensor_ports(graph, bindings, resolve_backward_view)
