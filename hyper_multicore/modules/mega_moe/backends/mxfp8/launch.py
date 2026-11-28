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
"""Byte-alias launch adapter for the native MXFP8 MegaMoE ABI."""

from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache

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
                         events: torch.Tensor, spec: MegaMoeSpec, *,
                         cache: OrderedDict | None = None) -> torch.Tensor:
    """Reuse exact immutable bounds while retaining cross-stream upload ordering."""
    # Bounds describe storage sizes, never a previous invocation's addresses.
    sizes = tuple(tensor.numel() * tensor.element_size() for tensor in buffers)
    geometry = (spec.rank_id, spec.ep_size, spec.num_experts, spec.hidden_size,
                spec.local_num_tokens, spec.num_cube_cores, runtime.numel(), events.numel())
    key = (runtime.device, sizes, geometry)
    cached = None if cache is None else cache.get(key)
    stream = torch.npu.current_stream(runtime.device) if runtime.device.type == 'npu' else None
    if cached is not None:
        tensor, producer, ready = cached
        if stream is not None:
            if producer != stream:
                stream.wait_event(ready)
            tensor.record_stream(stream)
        cache.move_to_end(key)
        return tensor
    metadata = serialize_mx_launch_metadata(
        sizes, rank=geometry[0], ep=geometry[1], experts=geometry[2], hidden=geometry[3],
        local_tokens=geometry[4], cube_count=geometry[5], runtime_bytes=geometry[6], event_bytes=geometry[7])
    tensor = torch.frombuffer(bytearray(metadata), dtype=torch.uint8).to(device=runtime.device)
    if cache is not None:
        ready = None
        if stream is not None:
            ready = torch.npu.Event()
            ready.record(stream)
        cache[key] = tensor, stream, ready
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
