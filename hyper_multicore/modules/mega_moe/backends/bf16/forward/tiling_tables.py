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
Forward tiling lookup tables.
Each function mirrors the C++ get_tiling_data_* functions.
String-based GMM tiling is expanded num_cube_cores times (default 24);
struct-based SwiGLU tiling is expanded 49 times (= 2 * 24 + 1 for Ascend 910B).
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
    one = bytes(td)
    return one * 49


# ── Forward GMM1: x=[per_rank_seq,7168], weight=[E,7168,4096] ─────────────────
# C++: get_tiling_data_gmm_x_7168_g2(split_value)

GMM1_TABLE = {
    4096: "8,24,0,0,0,0,0,1,1,1,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,4096,4096,7168,7168,4096,256,7168,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Forward GMM2: x=[per_rank_seq,2048], weight=[E,2048,7168] ─────────────────
# C++: get_tiling_data_gmm_x_2048_g2(split_value)

GMM2_TABLE = {
    4096: "8,24,0,0,0,0,0,1,1,1,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,4096,7168,2048,2048,4096,256,2048,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Forward SwiGLU: colLen=2048 ───────────────────────────────────────────────
# C++: get_tiling_data_swiglu(split_value)  →  SwiGluTilingData struct

def _make_swiglu(is32b, idbl, row_len, col_len, base_row_len, base_col_len,
                 act_left, bias_empty, qs_empty, as_empty,
                 swi_col_len, per_row_len, mod_row_len, used_core_num):
    """Build a SwiGluTilingDataC struct with the given tile parameters."""
    td = SwiGluTilingDataC()
    td.is32BAligned         = is32b
    td.isDoubleBuffer       = idbl
    td.rowLen               = row_len
    td.colLen               = col_len
    td.baseRowLen           = base_row_len
    td.baseColLen           = base_col_len
    td.activateLeft         = act_left
    td.biasIsEmpty          = bias_empty
    td.quantScaleIsEmpty    = qs_empty
    td.activateScaleIsEmpty = as_empty
    td.swiColLen            = swi_col_len
    td.perRowLen            = per_row_len
    td.modRowLen            = mod_row_len
    td.usedCoreNum          = used_core_num
    return td


SWIGLU_TABLE = {
    16:   _make_swiglu(1,1,16,  2048, 16, 512, 0,0,0,0, 0,0,0,0),
    32:   _make_swiglu(1,1,32,  2048, 32, 256, 0,0,0,0, 0,0,0,0),
    64:   _make_swiglu(1,1,64,  2048, 19, 512, 0,0,0,0, 0,0,0,0),
    128:  _make_swiglu(1,1,128, 2048, 19, 512, 0,0,0,0, 0,0,0,0),
    256:  _make_swiglu(1,1,256, 2048, 19, 512, 0,0,0,0, 0,0,0,0),
    512:  _make_swiglu(1,1,512, 2048, 19, 512, 0,0,0,0, 0,0,0,0),
    1024: _make_swiglu(1,1,1024,2048, 19, 512, 0,0,0,0, 0,0,0,0),
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_up_proj_tiling_bytes(split_value: int, *,
                           hidden_size: int = 7168,
                           intermediate_size: int = 2048,
                           num_groups: int = 8,
                           num_cube_cores: int = DEFAULT_CORE_SPEC.num_cube_cores) -> bytes:
    """Forward GMM1 tiling [x,hidden]×[E,hidden,intermediate*2], expanded num_cube_cores times."""
    if split_value not in GMM1_TABLE:
        raise KeyError(f"GMM1 tiling: split_value={split_value} not found")
    csv = _patch_gmm_dims(GMM1_TABLE[split_value],
                          {7168: hidden_size, 4096: intermediate_size * 2},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_down_proj_tiling_bytes(split_value: int, *,
                           hidden_size: int = 7168,
                           intermediate_size: int = 2048,
                           num_groups: int = 8,
                           num_cube_cores: int = DEFAULT_CORE_SPEC.num_cube_cores) -> bytes:
    """Forward GMM2 tiling [x,intermediate]×[E,intermediate,hidden], expanded num_cube_cores times."""
    if split_value not in GMM2_TABLE:
        raise KeyError(f"GMM2 tiling: split_value={split_value} not found")
    csv = _patch_gmm_dims(GMM2_TABLE[split_value],
                          {2048: intermediate_size, 7168: hidden_size},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_swiglu_tiling_bytes(split_value: int, *,
                             intermediate_size: int = 2048) -> bytes:
    """Forward SwiGLU tiling (colLen=intermediate_size), expanded 49 times."""
    if split_value not in SWIGLU_TABLE:
        raise KeyError(f"SwiGLU tiling: split_value={split_value} not found")
    td = copy.copy(SWIGLU_TABLE[split_value])
    td.colLen = intermediate_size
    return _expand_swiglu_struct(td)
