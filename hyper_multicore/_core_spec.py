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
"""Core specifications and Device resource queries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CoreSpec:
    """Cube and Vector core counts."""

    num_cube_cores: int
    num_vector_cores: int

    def __post_init__(self) -> None:
        for name in ("num_cube_cores", "num_vector_cores"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")


DEFAULT_CORE_SPEC = CoreSpec(num_cube_cores=24, num_vector_cores=48)


def get_device_core_spec(device_index: int) -> CoreSpec:
    """Query the NPU device's current Cube and Vector limits on every call."""
    import torch  # pylint: disable=import-outside-toplevel

    limits = torch.npu.get_device_limit(device_index)
    return CoreSpec(
        num_cube_cores=limits["cube_core_num"],
        num_vector_cores=limits["vector_core_num"],
    )
