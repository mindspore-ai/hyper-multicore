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
"""Physical scalar workspace ABI shared by lowering, validation and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from hyper_multicore.scheduler.config import TensorDescC
from hyper_multicore.scheduler.graph import TensorSpec

PROBABILITY_PLANES = 4
PROBABILITY_DTYPE_BYTES = 4
PROBABILITY_POSITION_FORWARD = 16
PROBABILITY_POSITION_BACKWARD = 25


class ProbabilityPlane(IntEnum):
    """FP32 scalar-row planes in one symmetric MegaMoe allocation."""

    SOURCE = 0
    RECEIVED = 1
    RECEIVED_GRADIENT = 2
    RETURNED_GRADIENT = 3


@dataclass(frozen=True)
class ProbabilityLayout:
    """Describe the same probability operand in native tasks and diagnostic graphs."""

    stride: int
    parameter_position: int

    def descriptor(self) -> TensorDescC:
        """Create the auxiliary dense FP32 descriptor without changing task events."""
        result = TensorDescC()
        result.tensor_type = 0
        result.data_type = PROBABILITY_DTYPE_BYTES
        result.input_position = self.parameter_position
        result.dim[0] = PROBABILITY_PLANES
        result.dim[1] = self.stride
        return result

    def tensor_spec(self) -> TensorSpec:
        """Describe the identical operand for diagnostic graph validation."""
        return TensorSpec("probability_workspace", [PROBABILITY_PLANES, self.stride],
                          self.parameter_position, dtype_size=PROBABILITY_DTYPE_BYTES)

    @classmethod
    def from_descriptor(cls, descriptor: TensorDescC) -> ProbabilityLayout:
        """Reject an incompatible probability descriptor before interpreting its units."""
        if (descriptor.tensor_type != 0 or descriptor.data_type != PROBABILITY_DTYPE_BYTES
                or descriptor.dim[0] != PROBABILITY_PLANES or descriptor.dim[1] <= 0 or descriptor.base_ptr_offset != 0
                or descriptor.input_position not in (PROBABILITY_POSITION_FORWARD, PROBABILITY_POSITION_BACKWARD)):
            raise ValueError("Invalid MegaMoe probability workspace descriptor")
        return cls(int(descriptor.dim[1]), int(descriptor.input_position))
