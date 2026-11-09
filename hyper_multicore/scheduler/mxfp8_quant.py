# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Versioned side records for group-aware BF16 MX quantization tasks."""

from dataclasses import dataclass
import struct
from typing import Iterable

MX_QUANT_TASK = 143
MX_SWIGLU_TASK = 145
MX_SWIGLU_GRAD_TASK = 146
MX_SWIGLU_ROW_TASK = 147
MX_DISPATCH_QUANT_TASK = 153
MX_QUANT_MAGIC = 0x4D585154
_RECORD = struct.Struct('<IHH8I3Q')


@dataclass(frozen=True)
class MxQuantDescriptor:
    """Describe one logical vector worker processing one expert's tokens.

    ``columns`` is always the quantized output width: H for plain quantization,
    I for SwiGLU forward, and 2I for SwiGLU backward. The cumulative group list
    supplies the actual token range at execution time. Scale storage retains the
    original packed64*2 grouped-column convention, including empty-group gaps.
    """

    columns: int
    expert_index: int
    worker_index: int = 0
    worker_count: int = 1
    task_type: int = MX_QUANT_TASK

    def __post_init__(self) -> None:
        fields = (self.columns, self.expert_index, self.worker_index, self.worker_count, self.task_type)
        if any(type(value) is not int or value < 0 or value > 0xFFFFFFFF for value in fields):
            raise ValueError('MX quant descriptor fields must be nonnegative uint32 integers')
        if self.columns == 0 or self.columns % 64:
            raise ValueError('MX quant columns must be a positive multiple of 64')
        if self.worker_count == 0 or self.worker_index >= self.worker_count:
            raise ValueError('MX quant worker_index must be smaller than positive worker_count')
        if self.task_type not in (MX_QUANT_TASK, MX_SWIGLU_TASK, MX_SWIGLU_GRAD_TASK, MX_SWIGLU_ROW_TASK):
            raise ValueError('Unsupported MX quant task type')
        if self.task_type == MX_SWIGLU_GRAD_TASK and self.columns % 128:
            raise ValueError('SwiGLU backward output columns must be a multiple of 128')

    @property
    def block_rows(self) -> int:
        """Match the upstream packed-column quantization block height."""
        return 16 if self.task_type == MX_SWIGLU_ROW_TASK else 64

    @property
    def block_columns(self) -> int:
        """Keep the backward three-input scratch footprint within the native UB."""
        return 128 if self.task_type == MX_SWIGLU_GRAD_TASK else 256

    def serialize(self) -> bytes:
        """Return the 64-byte little-endian native record."""
        return _RECORD.pack(MX_QUANT_MAGIC, 1, 2, self.columns, self.expert_index,
                            self.worker_index, self.worker_count, self.block_rows,
                            self.block_columns, self.task_type, _RECORD.size, 0, 0, 0)


def serialize_mx_quant_descriptors(records: Iterable[MxQuantDescriptor]) -> bytes:
    """Concatenate records for a contiguous uint8 side-table buffer."""
    return b''.join(record.serialize() for record in records)


@dataclass(frozen=True)
class MxDispatchQuantDescriptor:
    """One input-fragment wave or the final cross-fragment boundary pass."""

    columns: int
    expert_index: int
    worker_index: int
    worker_count: int
    fragment_index: int
    fragment_rows: int
    offset_position: int
    size_position: int
    ep: int
    local_experts: int

    def serialize(self) -> bytes:
        MxQuantDescriptor(self.columns, self.expert_index, self.worker_index, self.worker_count)
        if self.fragment_rows <= 0 or self.fragment_rows % 64 or self.fragment_index < -1:
            raise ValueError('Dispatch quant requires aligned positive fragment_rows and fragment_index >= -1.')
        if not (0 <= self.offset_position < 64 and 0 <= self.size_position < 64
                and 0 < self.ep < 65536 and 0 < self.local_experts < 65536):
            raise ValueError('Dispatch quant metadata roles or topology exceed their serialized range.')
        roles = self.offset_position | (self.size_position << 8) | (self.ep << 16) | (self.local_experts << 32)
        return _RECORD.pack(MX_QUANT_MAGIC, 2, 2, self.columns, self.expert_index,
                            self.worker_index, self.worker_count, 64, 256, MX_DISPATCH_QUANT_TASK,
                            _RECORD.size, self.fragment_index & 0xFFFFFFFFFFFFFFFF, self.fragment_rows, roles)
