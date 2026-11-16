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
"""
Backward tiling lookup tables.
Each dict maps split_value -> CSV tiling string (for GMM ops) or
struct fields (for SwiGLU-grad).

C++ source functions:
  GMM1  (pos 20) → get_tiling_data_first_gmm(split_value)
  GMM2  (pos 21) → get_tiling_data_second_gmm(split_value)
  GMM3  (pos 22) → get_tiling_data_third_gmm(split_value)
  GMM4  (pos 23) → get_tiling_data_fourth_gmm(split_value)
  SwiGLU-grad (pos 24) → get_tiling_data_swiglu_grad(split_value)
"""
# pylint: disable=line-too-long
import copy

from hyper_multicore._core_spec import DEFAULT_CORE_SPEC
from hyper_multicore.scheduler.config import (
    SwiGluTilingDataC,
)

from .._gmm_tiling import _expand_gmm_string, _patch_gmm_dims


# ── helpers ───────────────────────────────────────────────────────────────────

def _expand_swiglu_struct(td: SwiGluTilingDataC) -> bytes:
    """Tile SwiGluTilingDataC 49 times (= 2 * 24 AI Cube cores + 1)."""
    return bytes(td) * 49


def _make_swiglu(row_len, col_len, base_row_len, base_col_len):
    """Build a SwiGluTilingDataC for SwiGLU-grad with the given tile dimensions."""
    # These member names match the CANN tiling ABI and must retain their native spellings.
    # pylint: disable=invalid-name
    td = SwiGluTilingDataC()
    td.is32BAligned         = 1
    td.isDoubleBuffer       = 1
    td.rowLen               = row_len
    td.colLen               = col_len
    td.baseRowLen           = base_row_len
    td.baseColLen           = base_col_len
    td.activateLeft         = 0
    td.biasIsEmpty          = 0
    td.quantScaleIsEmpty    = 0
    td.activateScaleIsEmpty = 0
    td.swiColLen            = 0
    td.perRowLen            = 0
    td.modRowLen            = 0
    td.usedCoreNum          = 0
    return td


# ── Backward GMM1: x=[per_rank_seq,7168], weight=[E,2048,7168] ───────────────
# C++: get_tiling_data_first_gmm(split_value)
# x input [*, 7168] → y output [*, 2048]

FIRST_GMM_TABLE = {
    4096:  "8,24,0,0,0,0,0,1,1,1,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,4096,2048,7168,7168,4096,256,7168,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Backward GMM2: x=[per_rank_seq,4096], weight=[E,7168,4096] ───────────────
# C++: get_tiling_data_second_gmm(split_value)
# x input [*, 4096] → y output [*, 7168]

SECOND_GMM_TABLE = {
    4096:  "8,24,0,0,0,0,0,1,1,1,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,4096,7168,4096,4096,4096,256,4096,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Backward GMM3: x1^T=[sre,7168], x2=[sre,4096], weight grad ───────────────
# C++: get_tiling_data_third_gmm(split_value)

THIRD_GMM_TABLE = {
    4096:  "8,24,0,0,0,0,0,1,1,1,2,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,7168,4096,4096,4096,7168,256,4096,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Backward GMM4: x1^T=[sre,2048], x2=[sre,7168], weight grad ───────────────
# C++: get_tiling_data_fourth_gmm(split_value)

FOURTH_GMM_TABLE = {
    4096:  "8,24,0,0,0,0,0,1,1,1,2,0,0,0,2048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,2048,7168,4096,4096,2048,256,4096,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Backward SwiGLU-grad tiling ───────────────────────────────────────────────
# C++: get_tiling_data_swiglu_grad(split_value)  → baseColLen=256

SWIGLU_GRAD_TABLE = {
    32:  _make_swiglu(row_len=32,  col_len=2048, base_row_len=19, base_col_len=256),
    64:  _make_swiglu(row_len=64,  col_len=2048, base_row_len=19, base_col_len=256),
    128: _make_swiglu(row_len=128, col_len=2048, base_row_len=19, base_col_len=256),
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_act_grad_tiling_bytes(split_value: int, *,
                                hidden_size: int = 7168,
                                intermediate_size: int = 2048,
                                num_groups: int = 8,
                                num_cube_cores: int = DEFAULT_CORE_SPEC.num_cube_cores) -> bytes:
    """Backward GMM1 [x,hidden]×[E,intermediate,hidden] → y=[x,intermediate]."""
    csv = FIRST_GMM_TABLE.get(split_value)
    if csv is None:
        raise KeyError(f"No first_gmm tiling for split_value={split_value}")
    csv = _patch_gmm_dims(csv,
                          {7168: hidden_size, 2048: intermediate_size},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_gate_grad_tiling_bytes(split_value: int, *,
                                 hidden_size: int = 7168,
                                 intermediate_size: int = 2048,
                                 num_groups: int = 8,
                                 num_cube_cores: int = DEFAULT_CORE_SPEC.num_cube_cores) -> bytes:
    """Backward GMM2 [x,intermediate*2]×[E,hidden,intermediate*2] → y=[x,hidden]."""
    csv = SECOND_GMM_TABLE.get(split_value)
    if csv is None:
        raise KeyError(f"No second_gmm tiling for split_value={split_value}")
    csv = _patch_gmm_dims(csv,
                          {4096: intermediate_size * 2, 7168: hidden_size},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_w1_grad_tiling_bytes(split_value: int, *,
                                hidden_size: int = 7168,
                                intermediate_size: int = 2048,
                                num_groups: int = 8,
                                num_cube_cores: int = DEFAULT_CORE_SPEC.num_cube_cores,
                                expert_weight_layout: str = "EIO") -> bytes:
    """Build W13-gradient tiling with caller-layout output axes and dynamic reduction K."""
    csv = THIRD_GMM_TABLE.get(split_value)
    if csv is None:
        raise KeyError(f"No third_gmm tiling for split_value={split_value}")
    rows, columns = ((intermediate_size * 2, hidden_size) if expert_weight_layout == "EOI"
                     else (hidden_size, intermediate_size * 2))
    csv = _patch_gmm_dims(csv,
                          {7168: rows, 4096: columns},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_w2_grad_tiling_bytes(split_value: int, *,
                                 hidden_size: int = 7168,
                                 intermediate_size: int = 2048,
                                 num_groups: int = 8,
                                 num_cube_cores: int = DEFAULT_CORE_SPEC.num_cube_cores,
                                 expert_weight_layout: str = "EIO") -> bytes:
    """Build W2-gradient tiling with caller-layout output axes and dynamic reduction K."""
    csv = FOURTH_GMM_TABLE.get(split_value)
    if csv is None:
        raise KeyError(f"No fourth_gmm tiling for split_value={split_value}")
    rows, columns = ((hidden_size, intermediate_size) if expert_weight_layout == "EOI"
                     else (intermediate_size, hidden_size))
    csv = _patch_gmm_dims(csv,
                          {2048: rows, 7168: columns,
                           4096: intermediate_size * 2},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_swiglu_grad_tiling_bytes(split_value: int, *,
                                  intermediate_size: int = 2048) -> bytes:
    """Backward SwiGLU-grad tiling (colLen=intermediate_size)."""
    td = SWIGLU_GRAD_TABLE.get(split_value)
    if td is None:
        raise KeyError(f"No swiglu_grad tiling for split_value={split_value}")
    td = copy.copy(td)
    td.colLen = intermediate_size
    return _expand_swiglu_struct(td)
