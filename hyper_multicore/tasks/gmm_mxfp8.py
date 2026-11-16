# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Versioned MX GMM side records; the existing runtime task ABI stays unchanged."""

from dataclasses import dataclass
from typing import Any
from hyper_multicore.scheduler.config import TaskSplitValue
from hyper_multicore.scheduler.graph import OperatorNode
from hyper_multicore.tasks.utils import fill_expert_compute_tasks
import ctypes
import os
from collections.abc import Sequence

MX_GMM_MAGIC = 0x4D58474D
MX_GMM_VERSION = 1
MX_GMM_TASK_TYPE = 142
MX_GMM_MAX_BUFFERS = 64
MX_GMM_DYNAMIC_GROUP = 1
MX_GMM_LAYOUTS = {'NN': 0, 'NT': 1, 'TN': 2}
MX_GMM_OUTPUT_BF16 = 0
MX_GMM_OUTPUT_FP32 = 1
MX_GMM_TILES = {
    'baseline': (64, 128, 128, 256),
    't1': (128, 128, 128, 256),
    't2': (64, 256, 128, 256),
    't3': (128, 128, 128, 512),
    't4': (256, 128, 128, 256),
    't5': (128, 256, 128, 256),
    't6': (256, 256, 128, 256),
}


class MxGmmDescriptor(ctypes.LittleEndianStructure):
    """One pointer-free, per-expert/per-worker MX task record."""

    _fields_ = [(name, ctypes.c_uint32) for name in (
        'magic', 'version', 'size', 'layout', 'flags',
        'x_scale_position', 'w_scale_position', 'cube_index', 'cube_count',
        'm', 'n', 'k', 'base_m', 'base_n', 'base_k',
        'k_al1', 'k_bl1', 'scale_k', 'db_l0c', 'output_dtype',
    )] + [(name, ctypes.c_uint64) for name in (
        'x_offset', 'w_offset', 'y_offset', 'x_scale_offset', 'w_scale_offset', 'group_list_offset',
    )]


def make_mx_gmm_descriptor(
    *, layout: str, m: int, n: int, k: int,
    x_scale_position: int, w_scale_position: int,
    cube_count: int, cube_index: int = 0,
    x_offset: int = 0, w_offset: int = 0, y_offset: int = 0,
    x_scale_offset: int = 0, w_scale_offset: int = 0,
    group_list_offset: int = 0, output_dtype: int = MX_GMM_OUTPUT_BF16, tile: str | None = None,
) -> MxGmmDescriptor:
    """Create conservative A5 tiling and offsets measured in bytes.

    Group-list offset selects an expert endpoint in the original cumulative list.
    The worker replaces M (NN/NT) or K (TN) with that expert's actual received rows.
    Set HYPER_MULTICORE_MX_GMM_TILE before plan construction, or pass an explicit tile choice.
    """
    if layout not in MX_GMM_LAYOUTS:
        raise ValueError(f'Unsupported MX GMM layout: {layout}')
    tile = os.environ.get('HYPER_MULTICORE_MX_GMM_TILE', 't6') if tile is None else tile
    if tile not in MX_GMM_TILES:
        raise ValueError(f'Unsupported MX GMM tile: {tile}; expected one of {tuple(MX_GMM_TILES)}')
    if cube_count <= 0 or not 0 <= cube_index < cube_count:
        raise ValueError(f'Invalid MX cube worker {cube_index}/{cube_count}')
    values = dict(m=m, n=n, k=k, x_scale_position=x_scale_position, w_scale_position=w_scale_position,
                  x_offset=x_offset, w_offset=w_offset, y_offset=y_offset,
                  x_scale_offset=x_scale_offset, w_scale_offset=w_scale_offset,
                  group_list_offset=group_list_offset)
    if any(value < 0 for value in values.values()) or n == 0:
        raise ValueError(f'Invalid MX GMM dimensions/offsets: {values}')
    if output_dtype not in (MX_GMM_OUTPUT_BF16, MX_GMM_OUTPUT_FP32):
        raise ValueError(f'Unsupported MX output dtype: {output_dtype}')
    if group_list_offset % 8:
        raise ValueError('MX group_list_offset must address an INT64 endpoint')
    record = MxGmmDescriptor()
    if max(x_scale_position, w_scale_position) >= MX_GMM_MAX_BUFFERS:
        raise ValueError('MX scale positions must be within the 64-slot buffer list')
    record.flags = MX_GMM_DYNAMIC_GROUP
    record.magic = MX_GMM_MAGIC
    record.version = MX_GMM_VERSION
    record.size = ctypes.sizeof(MxGmmDescriptor)
    record.layout = MX_GMM_LAYOUTS[layout]
    record.cube_index = cube_index
    record.cube_count = cube_count
    record.base_m, record.base_n, record.base_k, record.k_al1 = MX_GMM_TILES[tile]
    record.k_bl1 = record.scale_k = record.k_al1
    record.db_l0c = 1 if tile == 't6' else 2
    record.output_dtype = output_dtype
    for name, value in values.items():
        field = dict(MxGmmDescriptor._fields_)[name]
        if value >= 1 << (8 * ctypes.sizeof(field)):
            raise ValueError(f'MX field {name} exceeds its serialized range: {value}')
        setattr(record, name, value)
    return record


def mx_gmm_l1_bytes(record: MxGmmDescriptor) -> int:
    """Return conservative double-buffered data and scale L1 storage."""
    data = 2 * (record.base_m * record.k_al1 + record.base_n * record.k_bl1)
    scales = 2 * (record.base_m + record.base_n) * ((record.scale_k + 63) // 64) * 2
    return data + scales


def serialize_mx_gmm_descriptors(records: Sequence[MxGmmDescriptor]) -> bytes:
    """Serialize records without embedding live gathered-weight pointers."""
    return b''.join(bytes(record) for record in records)



@dataclass
class Mxfp8GmmFillConfig:
    layout: str
    m: int
    n: int
    k: int
    x_scale: int
    w_scale: int
    workers: int
    records: list

    def fill(self, cfg: Any, op: OperatorNode, tsv: TaskSplitValue) -> None:
        indices = []
        for expert in range(tsv.single_rank_expert_num):
            for worker in range(self.workers):
                indices.append(len(self.records))
                self.records.append(make_mx_gmm_descriptor(
                    layout=self.layout, m=self.m, n=self.n, k=self.k,
                    x_scale_position=self.x_scale, w_scale_position=self.w_scale,
                    cube_count=self.workers, cube_index=worker, group_list_offset=expert * 8,
                    w_offset=0 if self.layout == 'TN' else expert * self.n * self.k,
                    w_scale_offset=0 if self.layout == 'TN' else expert * self.n * self.k // 32,
                    y_offset=expert * self.m * self.n * op.outputs[0].dtype_size if self.layout == 'TN' else 0,
                    output_dtype=MX_GMM_OUTPUT_FP32 if op.outputs[0].dtype_size == 4 else MX_GMM_OUTPUT_BF16,
                ))
        fill_expert_compute_tasks(cfg, op, tsv, MX_GMM_TASK_TYPE, self.workers, False, indices, cube=True)
