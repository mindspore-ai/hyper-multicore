# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Byte-alias launch adapter for the native MXFP8 MegaMoE ABI."""

from functools import lru_cache
from collections.abc import Sequence

import torch

from hyper_multicore.torch.ops import _load_native

from ...spec import MegaMoeSpec
from .launch_contract import serialize_mx_launch_metadata


@lru_cache(maxsize=1)
def _load_mxfp8_native() -> None:
    _load_native()
    if torch.ops.hyper_multicore.mega_moe_mxfp8_abi_version() != 4:
        raise RuntimeError('MXFP8 MegaMoE adapter ABI mismatch; rebuild the local multicore component.')


def serialize_mxfp8_launch(buffers: Sequence[torch.Tensor], runtime: torch.Tensor,
                         events: torch.Tensor, spec: MegaMoeSpec) -> bytes:
    """Serialize exact current byte bounds without reading device tensor contents."""
    return serialize_mx_launch_metadata(
        [tensor.numel() * tensor.element_size() for tensor in buffers],
        rank=spec.rank_id, ep=spec.ep_size, experts=spec.num_experts, hidden=spec.hidden_size,
        local_tokens=spec.local_num_tokens, cube_count=spec.num_cube_cores,
        runtime_bytes=runtime.numel(), event_bytes=events.numel(),
    )


def prepare_mxfp8_launch(buffers: Sequence[torch.Tensor], runtime: torch.Tensor,
                         events: torch.Tensor, spec: MegaMoeSpec, *, cache=None) -> torch.Tensor:
    """Reuse exact immutable bounds while retaining cross-stream upload ordering."""
    metadata = serialize_mxfp8_launch(buffers, runtime, events, spec)
    cached = None if cache is None else cache.get(metadata)
    stream = torch.npu.current_stream(runtime.device) if runtime.device.type == 'npu' else None
    if cached is not None:
        tensor, producer, ready = cached
        if stream is not None:
            if producer != stream:
                stream.wait_event(ready)
            tensor.record_stream(stream)
        cache.move_to_end(metadata)
        return tensor
    tensor = torch.frombuffer(bytearray(metadata), dtype=torch.uint8).to(device=runtime.device)
    if cache is not None:
        ready = None
        if stream is not None:
            ready = torch.npu.Event()
            ready.record(stream)
        cache[metadata] = tensor, stream, ready
        while len(cache) > 8:
            cache.popitem(last=False)
    return tensor


def launch_mxfp8_experts(buffers: Sequence[torch.Tensor], runtime: torch.Tensor,
                           events: torch.Tensor, diagnostic: torch.Tensor, launch: torch.Tensor,
                           *, direction: str = 'forward') -> None:
    """Run the native fused MoE with alias-preserving physical arguments."""
    _load_mxfp8_native()
    if direction not in ('forward', 'backward'):
        raise ValueError('MX native direction must be forward or backward')
    primary = (torch.ops.hyper_multicore.mega_moe_mxfp8_grad if direction == 'backward'
               else torch.ops.hyper_multicore.mega_moe_mxfp8)
    primary(buffers, runtime, events, diagnostic, launch)
