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
"""Versioned side records for group-aware BF16 MX quantization tasks."""

from dataclasses import dataclass
import struct
from typing import Any, Iterable

from hyper_multicore.scheduler.config import TaskSplitValue
from hyper_multicore.scheduler.graph import OperatorNode
from hyper_multicore.tasks.utils import fill_expert_compute_tasks

MX_QUANT_TASK = 143
MX_SWIGLU_TASK = 145
MX_SWIGLU_GRAD_TASK = 146
MX_QUANT_MAGIC = 0x4D585154
_RECORD = struct.Struct('<IHH10I2Q')


@dataclass(frozen=True)
class MxQuantDescriptor:
    """Describe one logical vector worker processing one expert's tokens.

    ``columns`` is always the quantized output width: H for plain quantization,
    I for SwiGLU forward, and 2I for SwiGLU backward. The cumulative group list
    supplies the actual token range at execution time. Scale storage retains the
    original packed64*2 grouped-column convention, including empty-group gaps.
    Zero clamp bits select the unclamped activation specialization.
    """

    columns: int
    expert_index: int
    worker_index: int = 0
    worker_count: int = 1
    task_type: int = MX_QUANT_TASK
    clamp_limit_bits: int = 0

    def __post_init__(self) -> None:
        fields = (self.columns, self.expert_index, self.worker_index, self.worker_count,
                  self.task_type, self.clamp_limit_bits)
        if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFFFFFF
               for value in fields):
            raise ValueError('MX quant descriptor fields must be nonnegative uint32 integers')
        if self.columns == 0 or self.columns % 64:
            raise ValueError('MX quant columns must be a positive multiple of 64')
        if self.worker_count == 0 or self.worker_index >= self.worker_count:
            raise ValueError('MX quant worker_index must be smaller than positive worker_count')
        if self.task_type not in (MX_QUANT_TASK, MX_SWIGLU_TASK, MX_SWIGLU_GRAD_TASK):
            raise ValueError('Unsupported MX quant task type')
        if self.task_type == MX_SWIGLU_GRAD_TASK and self.columns % 128:
            raise ValueError('SwiGLU backward output columns must be a multiple of 128')
        if self.clamp_limit_bits >= 0x7F800000 or (self.task_type == MX_QUANT_TASK and self.clamp_limit_bits):
            raise ValueError('MX clamp bits require an activation task and a finite nonnegative float32')

    @property
    def block_rows(self) -> int:
        """Match the upstream packed-column quantization block height."""
        return 64

    @property
    def block_columns(self) -> int:
        """Keep the backward three-input scratch footprint within the native UB."""
        return 128 if self.task_type == MX_SWIGLU_GRAD_TASK else 256

    def serialize(self) -> bytes:
        """Return the 64-byte little-endian native record."""
        return _RECORD.pack(MX_QUANT_MAGIC, 2, 2, self.columns, self.expert_index,
                            self.worker_index, self.worker_count, self.block_rows, self.block_columns,
                            self.task_type, _RECORD.size, self.clamp_limit_bits, 0, 0, 0)


def serialize_mx_quant_descriptors(records: Iterable[MxQuantDescriptor]) -> bytes:
    """Concatenate records for a contiguous uint8 side-table buffer."""
    return b''.join(record.serialize() for record in records)



@dataclass
class Mxfp8QuantFillConfig:
    """Describe the per-expert MX quantization tasks."""
    kind: int
    columns: int
    workers: int
    first: bool
    records: list
    clamp_limit_bits: int = 0

    def fill(self, cfg: Any, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Populate the runtime task table and MX quantization records."""
        indices = []
        for expert in range(tsv.single_rank_expert_num):
            for worker in range(self.workers):
                indices.append(len(self.records))
                self.records.append(MxQuantDescriptor(
                    columns=self.columns, expert_index=expert, worker_index=worker,
                    worker_count=self.workers, task_type=self.kind, clamp_limit_bits=self.clamp_limit_bits,
                ))
        fill_expert_compute_tasks(cfg, op, tsv, self.kind, self.workers, self.first, indices, cube=False)
