# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""MoE byte-list launch bounds; independent of the GMM task contract."""
import ctypes
from collections.abc import Sequence
MAX_GRAPH_BUFFERS = 64

class MxLaunchDescriptor(ctypes.LittleEndianStructure):
    """Launch bounds and the legacy rank/topology prefix consumed by SHMEM workers."""

    _fields_ = [(name, ctypes.c_int64) for name in (
        'rank', 'ep', 'experts', 'hidden', 'local_tokens', 'cube_count',
    )] + [('runtime_bytes', ctypes.c_uint64), ('event_bytes', ctypes.c_uint64)] + [
        ('magic', ctypes.c_uint32), ('version', ctypes.c_uint32),
        ('buffer_count', ctypes.c_uint32), ('reserved', ctypes.c_uint32),
        ('buffer_bytes', ctypes.c_uint64 * MAX_GRAPH_BUFFERS),
    ]


def serialize_mx_launch_metadata(
    buffer_sizes: Sequence[int], *, rank: int, ep: int, experts: int, hidden: int,
    local_tokens: int, cube_count: int, runtime_bytes: int, event_bytes: int,
) -> bytes:
    """Build the byte-list launch descriptor without reading any device data."""
    if not 0 < len(buffer_sizes) <= MAX_GRAPH_BUFFERS or any(size < 0 for size in buffer_sizes):
        raise ValueError('MX task graph requires 1..64 nonnegative byte-buffer sizes')
    if min(ep, experts, hidden, local_tokens, cube_count, runtime_bytes, event_bytes) <= 0:
        raise ValueError('MX launch topology and runtime bounds must be positive')
    if experts % ep or rank < 0:
        raise ValueError('MX launch requires nonnegative rank and experts divisible by EP')
    record = MxLaunchDescriptor()
    for name, value in dict(rank=rank, ep=ep, experts=experts, hidden=hidden, local_tokens=local_tokens,
                            cube_count=cube_count, runtime_bytes=runtime_bytes, event_bytes=event_bytes).items():
        setattr(record, name, value)
    record.magic = 0x4D584C47
    record.version, record.reserved = 1, 0
    record.buffer_count = len(buffer_sizes)
    for index, size in enumerate(buffer_sizes):
        record.buffer_bytes[index] = size
    return bytes(record)
