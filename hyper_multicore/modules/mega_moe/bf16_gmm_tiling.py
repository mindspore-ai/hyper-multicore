# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Isolated arch35 BF16 geometry for the two experimental logical-Cube workers."""
import struct

_RECORD_BYTES = 2016
_A5_FIELDS = {6: 256, 8: 256, 9: 256, 10: 64, 11: 8, 12: 8, 13: 1, 14: 1,
              19: 512*1024, 20: 256*1024, 26: 4, 27: 4, 30: 2, 31: 2, 32: 1}


def retile_a5_bf16(data: bytes, cube_cores: int) -> bytes:
    """Retain tensor/group metadata while using BM256/BN256/BK64 and single L0C.

    Each ping-pong A/B L0 half consumes 32KiB, C consumes 256KiB, and two
    A/B L1 slots consume 512KiB. Only the validated A5 Cube32 layout is exposed.
    The marker and field offsets match the legacy TCubeTiling producer and the
    pinned CANN9.2 kernel_tiling.h; the arch35 worker consumes individual fields.
    """
    if cube_cores != 32:
        raise ValueError('A5 BF16 retile requires 32 Cube cores')
    if len(data) != _RECORD_BYTES*cube_cores:
        raise ValueError('A5 BF16 retile requires 2016 bytes per Cube record')
    records = []
    for offset in range(0, len(data), _RECORD_BYTES):
        values = list(struct.unpack('<504I', data[offset:offset+_RECORD_BYTES]))
        markers = [index for index in range(350, len(values)) if values[index] == 393216]
        if len(markers) != 1:
            raise ValueError('Expected one legacy TCubeTiling shared-L1 marker per record')
        start = markers[0]-19
        for field, value in _A5_FIELDS.items():
            values[start+field] = value
        records.append(struct.pack('<504I', *values))
    return b''.join(records)
