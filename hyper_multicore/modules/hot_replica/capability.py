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
    for key in ("allocator", "bridge", "kernels"):
        if not (selected["root"] / artifacts[key]).is_file():
            raise RuntimeError(f"Selected HotReplica artifact is missing: {key}")


def require_capability(tensor: torch.Tensor, group: torch.distributed.ProcessGroup) -> None:
    """Converge capability failures before any rank allocates a bank or enters transfer."""
    _collective.collective_call(lambda: _require_a3(tensor), stage="HotReplica capability", group=group)
