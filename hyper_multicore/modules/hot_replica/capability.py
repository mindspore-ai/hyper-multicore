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
"""Validate the actual input device and selected HotReplica artifacts before allocation."""

import os
from pathlib import Path

import torch
from hyper_multicore import _collective
from hyper_multicore._loader import load_torch_npu, query_npu_arch, validate_native_compatibility


def _require_a3(tensor):
    """Reject devices sharing arch22 unless their actual SOC is Atlas A3."""
    if tensor.device.type != "npu" or tensor.dtype != torch.bfloat16:
        raise ValueError("HotReplica requires BF16 input on an Atlas A3 NPU")
    load_torch_npu()
    index = tensor.device.index
    if index is None:
        index = torch.npu.current_device()
    name = torch.npu.get_device_name(index).lower()
    if query_npu_arch(index) != 2201 or not any(part in name for part in ("910_93", "910c")):
        raise ValueError(f"HotReplica supports Atlas A3 only, input device {index} is {name}")
    selected = validate_native_compatibility()
    artifacts = selected["variant"].get("hot_replica")
    if not artifacts or artifacts.get("soc") != "ascend910_93":
        raise RuntimeError("The selected Torch/PTA adapter has no Atlas A3 HotReplica artifacts")
    for key in ("bridge", "kernels"):
        if not (selected["root"] / artifacts[key]).is_file():
            raise RuntimeError(f"Selected HotReplica artifact is missing: {key}")
    return bool(selected["build_info"].get("a3_replica_bf16_partials", False))


def _require_mx(tensor, cube_cores: int):
    """Validate the A5 transport contract before starting SHMEM."""
    if tensor.device.type != "npu" or tensor.dtype != torch.bfloat16:
        raise ValueError("MXFP8 HotReplica requires BF16 input on Ascend950")
    load_torch_npu()
    index = tensor.device.index
    if index is None:
        index = torch.npu.current_device()
    if query_npu_arch(index) != 3510:
        raise ValueError("MXFP8 HotReplica requires Ascend950")
    from ..mega_moe.schedule import MX_GRADIENT_CUBE_CORES  # pylint: disable=import-outside-toplevel

    if cube_cores != MX_GRADIENT_CUBE_CORES:
        raise ValueError(f"MXFP8 HotReplica requires {MX_GRADIENT_CUBE_CORES} Cube cores, got {cube_cores}")
    if os.environ.get("HCCL_OP_EXPANSION_MODE") != "AICPU_CacheDisable":
        raise ValueError("MXFP8 HotReplica prefetch requires HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable")
    selected = validate_native_compatibility()
    adapter = selected["variant"]["shmem_so"].get("arch35")
    if adapter is None:
        raise RuntimeError("The selected Torch/PTA adapter has no Ascend950 support")
    planner = selected["root"] / Path(adapter).parent / "hot_replica/_replica_host.so"
    from ._planner.mx_host import load_mx_planner  # pylint: disable=import-outside-toplevel

    load_mx_planner(planner)


def require_capability(
    tensor: torch.Tensor, group: torch.distributed.ProcessGroup, *, mx_cube_cores: int | None = None,
) -> bool:
    """Validate artifacts and agree on the A3 BF16 partial protocol before transfer."""
    local_error = None
    bf16_partials = False
    try:
        if mx_cube_cores is None:
            bf16_partials = _require_a3(tensor)
        else:
            _require_mx(tensor, mx_cube_cores)
    except Exception as error:  # pylint: disable=broad-exception-caught
        local_error = error
    failure = None if local_error is None else _collective.failure_report(local_error)
    reports = _collective.exchange(
        (bf16_partials, failure), group=group, failure_prefix="HotReplica capability failed while converging ranks",
    )
    capabilities, failures = zip(*reports)
    if any(failure is not None for failure in failures):
        details = _collective.format_collective_failures(failures)
        raise RuntimeError(f"HotReplica capability failed on {details}") from local_error
    return all(capabilities)
