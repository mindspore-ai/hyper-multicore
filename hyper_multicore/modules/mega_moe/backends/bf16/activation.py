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
"""Build BF16 activation nodes and prepare their online tiling payloads."""

from __future__ import annotations

from hyper_multicore.modules.mega_moe.numerics import SiTUGLUParameters, SwiGLUParameters
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.graph import OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_multicore.tasks.swiglu import SwiGLUFillConfig

from .backward.tiling_tables import get_swiglu_grad_tiling_bytes
from .forward.tiling_tables import get_swiglu_tiling_bytes


def build_activation_node(
    activation_parameters: SwiGLUParameters | SiTUGLUParameters, *, backward: bool,
    inputs: list[TensorSpec], outputs: list[TensorSpec], param_positions: list[int],
    split_value: int, tiling_position: int,
) -> OperatorNode:
    """Build the concrete BF16 operator node for the selected activation."""
    if not isinstance(activation_parameters, SwiGLUParameters):
        raise NotImplementedError("BF16 currently builds only SwiGLU activation nodes")
    return OperatorNode(
        name="swiglu_grad" if backward else "swiglu",
        op_type=OpType.SWIGLU_GRAD if backward else OpType.SWIGLU,
        diagnostic_stage_name="SwiGLUGrad" if backward else "SwiGLU",
        inputs=inputs, outputs=outputs, param_positions=param_positions,
        split_value=split_value,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv: (tsv.per_expert_seq // split_value) * tsv.single_rank_expert_num,
        ),
        tiling_position=tiling_position,
        fill_config=SwiGLUFillConfig(clamp_limit=activation_parameters.clamp_limit,
                                    clamp_after_activation=activation_parameters.clamp_after_silu),
    )


def prepare_activation_tiling(spec: MegaMoeSpec, *, backward: bool, rows_per_task: int) -> bytes:
    """Build the activation payload for the active workers in the online plan."""
    if not isinstance(spec.activation_parameters, SwiGLUParameters):
        raise NotImplementedError("BF16 currently prepares only SwiGLU activation tiling")
    build_tiling = get_swiglu_grad_tiling_bytes if backward else get_swiglu_tiling_bytes
    tiling_bytes = build_tiling(rows_per_task, intermediate_size=spec.intermediate_size)
    # The standalone tables contain one default record and 48 worker records.
    baseline_records = 49
    if len(tiling_bytes) % baseline_records:
        raise RuntimeError("invalid baseline SwiGLU tiling byte length.")
    record_size = len(tiling_bytes) // baseline_records
    return tiling_bytes[:record_size] * (2 * spec.num_cube_cores + 1)
