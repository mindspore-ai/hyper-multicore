# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Versioned MX GMM side records; the existing runtime task ABI stays unchanged."""

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
    tile = os.environ.get('HYPER_MULTICORE_MX_GMM_TILE', 'baseline') if tile is None else tile
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


class MxLaunchDescriptor(ctypes.LittleEndianStructure):
    """Launch bounds and the legacy rank/topology prefix consumed by SHMEM workers."""

    _fields_ = [(name, ctypes.c_int64) for name in (
        'rank', 'ep', 'experts', 'hidden', 'local_tokens', 'cube_count',
    )] + [('runtime_bytes', ctypes.c_uint64), ('event_bytes', ctypes.c_uint64)] + [
        ('magic', ctypes.c_uint32), ('version', ctypes.c_uint32),
        ('buffer_count', ctypes.c_uint32), ('reserved', ctypes.c_uint32),
        ('buffer_bytes', ctypes.c_uint64 * MX_GMM_MAX_BUFFERS),
    ]


def serialize_mx_launch_metadata(
    buffer_sizes: Sequence[int], *, rank: int, ep: int, experts: int, hidden: int,
    local_tokens: int, cube_count: int, runtime_bytes: int, event_bytes: int,
    vector_policy: str = 'legacy_odd',
) -> bytes:
    """Build the byte-list launch descriptor without reading any device data."""
    if not 0 < len(buffer_sizes) <= MX_GMM_MAX_BUFFERS or any(size < 0 for size in buffer_sizes):
        raise ValueError('MX task graph requires 1..64 nonnegative byte-buffer sizes')
    if min(ep, experts, hidden, local_tokens, cube_count, runtime_bytes, event_bytes) <= 0:
        raise ValueError('MX launch topology and runtime bounds must be positive')
    if experts % ep or rank < 0:
        raise ValueError('MX launch requires nonnegative rank and experts divisible by EP')
    record = MxLaunchDescriptor()
    for name, value in dict(rank=rank, ep=ep, experts=experts, hidden=hidden, local_tokens=local_tokens,
                            cube_count=cube_count, runtime_bytes=runtime_bytes, event_bytes=event_bytes).items():
        setattr(record, name, value)
    if vector_policy not in (
            'legacy_odd', 'all', 'single_vector', 'roles', 'roles_ready', 'roles_rule', 'planned', 'window2'):
        raise ValueError(f'Unsupported MX vector policy: {vector_policy}')
    record.magic = 0x4D584C47
    record.version, record.reserved = {'legacy_odd': (1, 0), 'all': (2, 1),
                                       'single_vector': (3, 2), 'roles': (4, 3), 'roles_ready': (5, 4),
                                       'planned': (6, 5), 'window2': (6, 5), 'roles_rule': (7, 6)}[vector_policy]
    record.buffer_count = len(buffer_sizes)
    for index, size in enumerate(buffer_sizes):
        record.buffer_bytes[index] = size
    return bytes(record)
