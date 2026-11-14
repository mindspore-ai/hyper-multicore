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
"""Stable model numerical options owned by the MegaMoE component."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MegaMoeNumericsOptions:
    """Define the model's MegaMoe activation and probability semantics.

    Import from ``hyper_multicore`` and pass to
    ``MegaMoeExperts(numerics_options=...)``. The executor validates these
    immutable options during construction. Defaults preserve ordinary SwiGLU
    and after-FC2 probability weighting. These are stable model semantics;
    tuning policies cannot override them.

    Args:
        swiglu_limit: Optional positive, finite float32-representable clamp
            limit. The gate branch is clipped only above the limit; the up
            branch is clipped to ``[-limit, limit]``. None disables clipping.
        clamp_after_silu: Clip SiLU(gate) before multiplying by up when True;
            otherwise clip gate before applying SiLU. Must be a bool and has
            no effect when swiglu_limit is None.
        apply_probs_before_fc2: Weight activated features before FC2 when True;
            otherwise weight expert outputs after FC2. Must be a bool. The
            before-FC2 path first rounds activations to BF16, multiplies FP32
            probabilities, and rounds to BF16 again before FC2.
    """

    swiglu_limit: float | None = None
    clamp_after_silu: bool = False
    apply_probs_before_fc2: bool = False
