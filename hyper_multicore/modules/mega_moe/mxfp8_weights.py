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
"""Resolve current physical expert weights after each MXFP8 FSDP gather phase."""

from dataclasses import dataclass
from typing import Literal

import torch



@dataclass(frozen=True)
class MXFP8PhysicalWeight:
    """Borrow one phase's physical data and packed E8M0 scales without conversion."""

    logical_shape: tuple[int, int, int]
    col_data: torch.Tensor
    col_scale: torch.Tensor
    row_data: torch.Tensor | None = None
    row_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class MXFP8ExpertWeightBinding:
    """Retain the original unsharded root, never a phase-specific quantized view.

    HyperParallel refreshes the root's physical fields in place during backward
    all-gather. A 3-D MXFP8 wrapper constructed during forward does not receive
    those replacements. Autograd contexts keep this binding and resolve it again
    after pre-backward hooks have materialized the current root.
    """

    root: torch.Tensor
    logical_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.root, torch.Tensor):
            raise TypeError('MegaMoE MXFP8 weights require an original tensor root with physical MX storage.')
        if len(self.logical_shape) != 3 or any(size <= 0 for size in self.logical_shape):
            raise ValueError(f'MegaMoE MXFP8 expert shape must be positive [E,M,N], got {self.logical_shape}.')
        experts, rows, columns = self.logical_shape
        if rows % 64 or columns % 64:
            raise ValueError(f'MegaMoE MXFP8 expert shape requires M/N divisible by 64, got {self.logical_shape}.')
        accepted = ((experts * rows, columns), self.logical_shape)
        if tuple(self.root.shape) not in accepted:
            raise ValueError(
                f'MegaMoE MXFP8 root shape {tuple(self.root.shape)} does not match expert shape {self.logical_shape}.')

    def _physical_pair(self, direction: Literal['row', 'col']) -> tuple[torch.Tensor, torch.Tensor]:
        data = getattr(self.root, f'{direction}_data')
        scale = getattr(self.root, f'{direction}_scale')
        if data is None or scale is None:
            raise ValueError(f'MegaMoE MXFP8 current root is missing required {direction} data/scale.')
        if data.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8:
            raise TypeError('MegaMoE MXFP8 physical storage requires E4M3FN data and packed uint8 E8M0 scales.')
        experts, rows, columns = self.logical_shape
        scale_shape = ((experts, rows, columns // 64, 2) if direction == 'row'
                       else (experts, rows // 64, columns, 2))
        return data.view(self.logical_shape), scale.view(scale_shape)

    def resolve(self, phase: Literal['forward', 'backward']) -> MXFP8PhysicalWeight:
        """Read current roots at execution time; backward requires both gathered axes."""
        if phase not in ('forward', 'backward'):
            raise ValueError(f'MegaMoE MXFP8 phase must be forward/backward, got {phase!r}.')
        col_data, col_scale = self._physical_pair('col')
        row_data = row_scale = None
        if phase == 'backward':
            row_data, row_scale = self._physical_pair('row')
        return MXFP8PhysicalWeight(self.logical_shape, col_data, col_scale, row_data, row_scale)
