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
"""GMM tiling template expansion and dimension patching."""

import ctypes

from hyper_multicore._core_spec import DEFAULT_CORE_SPEC


def _expand_gmm_string(csv: str, repeat: int = DEFAULT_CORE_SPEC.num_cube_cores) -> bytes:
    """Parse comma-separated uint32 string and tile it `repeat` times."""
    vals = [int(x) for x in csv.split(',')]
    arr = (ctypes.c_uint32 * (len(vals) * repeat))()
    for i in range(len(vals) * repeat):
        arr[i] = vals[i % len(vals)]
    return bytes(arr)


def _find_protected_positions(vals, active_map, split_value, is_weight_grad):
    """Return index positions that must not be replaced by dim_map substitution.

    When split_value collides with a K/N dim in active_map, the M-tile or
    K-tile positions (depending on GMM type) must be shielded.
    T is located via the 393216 anchor which sits at T+19.
    """
    if split_value is None or split_value not in active_map:
        return set()
    for i in range(350, len(vals)):
        if vals[i] == 393216:          # anchor always at t_base+19
            t_base = i - 19
            if is_weight_grad:
                return {t_base + 3, t_base + 4, t_base + 7}
            return {14, t_base + 1, t_base + 5}
    return set()


def _patch_gmm_dims(csv: str, dim_map: dict, num_groups: int,
                    default_groups: int = 8,
                    split_value: int = None,
                    num_cube_cores: int = DEFAULT_CORE_SPEC.num_cube_cores,
                    default_num_cores: int = 24) -> str:
    """Replace matrix dimension values in a GMM tiling CSV string.

    dim_map:         {old_int_value: new_int_value} for K/N dimension replacements.
                     Entries where old == new are no-ops and can be omitted.
    num_groups:      replacement for default_groups; only applied at index > 350.
    split_value:     M tile size (table key). When it equals a K/N value in dim_map,
                     we must protect M-tile positions from accidental replacement.
                     Two GMM types detected via vals[10]:
                       0 → activation GMM: M tile at {14, T+1, T+5}
                       2 → weight-grad GMM: split (K-tile) at {T+3, T+4, T+7}
                     T is found via the 393216 anchor fixed at T+19.
    num_cube_cores:  number of Cube cores used by the operator.
                     Replaces the hardcoded usedCoreNum/blockDim fields in the CSV:
                       vals[1]    — header blockDim field
                       i > 350    — tail usedCoreNum and any derived parallelism fields
    """
    active_map = {k: v for k, v in dim_map.items() if k != v}
    groups_changed = num_groups != default_groups
    cores_changed  = num_cube_cores != default_num_cores
    if not active_map and not groups_changed and not cores_changed:
        return csv  # fast path: nothing to do

    vals = list(map(int, csv.split(',')))
    is_weight_grad = vals[10] == 2

    # Patch groupNum and blockDim before dimension substitution.
    for index, old_value, new_value in ((0, default_groups, num_groups),
                                       (1, default_num_cores, num_cube_cores)):
        if vals[index] == old_value:
            vals[index] = new_value

    protected = _find_protected_positions(vals, active_map, split_value, is_weight_grad)

    for i, v in enumerate(vals):
        if i in protected:
            continue
        if v in active_map:
            vals[i] = active_map[v]
        elif cores_changed and i > 350 and v == default_num_cores:
            vals[i] = num_cube_cores   # covers T+0 (usedCoreNum) and derived fields
    return ','.join(map(str, vals))
