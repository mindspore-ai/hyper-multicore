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
"""Device buffers and static tiling for the local shared SwiGLU branch."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import torch

from hyper_multicore.scheduler.config import SwiGluTilingDataC

from .backward.tiling_tables import (
    get_act_grad_tiling_bytes, get_gate_grad_tiling_bytes,
    get_swiglu_grad_tiling_bytes, get_w1_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
)
from .forward.tiling_tables import (
    get_down_proj_tiling_bytes, get_swiglu_tiling_bytes,
    get_up_proj_tiling_bytes,
)


GMM_TILING_BYTES = 2016


def _static_gmm(data: bytes, tokens: int, *, weight_gradient: bool) -> bytes:
    """Patch the row dimensions in one static GMM tiling table."""
    words = np.frombuffer(data, dtype="<u4").copy().reshape(-1, GMM_TILING_BYTES // 4)
    anchors = np.flatnonzero(words[0] == 393216)
    if len(anchors) != 1:
        raise ValueError("shared GMM tiling has no unique matrix anchor")
    base = int(anchors[0]) - 19
    positions = (base + 3, base + 4, base + 7) if weight_gradient else (14, base + 1, base + 5)
    words[:, positions] = tokens
    return words.tobytes()


def shared_tiling_bytes(spec: Any, *, backward: bool) -> bytes:
    """Serialize shared GMM and SwiGLU tiling records."""
    options = {
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.shared_intermediate_size,
        "num_groups": 1,
        "num_cube_cores": spec.num_cube_cores,
    }
    if backward:
        builders = (
            get_act_grad_tiling_bytes, get_gate_grad_tiling_bytes,
            get_w1_grad_tiling_bytes, get_w2_grad_tiling_bytes,
        )
    else:
        builders = (get_up_proj_tiling_bytes, get_down_proj_tiling_bytes)
    records = [struct.pack("<q", spec.local_num_tokens)]
    for index, builder in enumerate(builders):
        layout = {"expert_weight_layout": "EOI"} if backward and index >= 2 else {}
        data = builder(4096, **options, **layout)
        if len(data) != GMM_TILING_BYTES * spec.num_cube_cores:
            raise ValueError("shared GMM tiling stride differs from the device ABI")
        records.append(_static_gmm(data, spec.local_num_tokens, weight_gradient=backward and index >= 2))
    swiglu = get_swiglu_grad_tiling_bytes if backward else get_swiglu_tiling_bytes
    swiglu_record = swiglu(128, intermediate_size=spec.shared_intermediate_size)[:80]
    records.append(swiglu_record)
    tail = spec.local_num_tokens % 128
    if tail:
        tail_record = SwiGluTilingDataC.from_buffer_copy(swiglu_record)
        tail_record.rowLen = tail
        tail_record.baseRowLen = min(tail, tail_record.baseRowLen)
        records.append(bytes(tail_record))
    return b"".join(records)


def pointer_table(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Create a pinned host pointer table and copy it to the target device."""
    addresses = torch.tensor(
        [0] + [tensor.data_ptr() for tensor in tensors],
        dtype=torch.int64,
        pin_memory=True,
    )
    return addresses.to(device=tensors[0].device, non_blocking=True).view(torch.uint8)


def shared_forward_buffers(
    x: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, intermediate: int,
    *, output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate shared forward intermediates and output."""
    up = x.new_empty((x.shape[0], 2 * intermediate))
    activation = x.new_empty((x.shape[0], intermediate))
    output = torch.empty_like(x) if output is None else output
    return up, activation, output, pointer_table((x, weight1, weight2, up, activation, output))


def shared_backward_buffers(
    dy: torch.Tensor, saved: tuple[torch.Tensor, ...],
    *, scratch: tuple[torch.Tensor, torch.Tensor] | None = None, integrated_output: bool = False,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, bool]:
    """Allocate shared backward outputs and its indirect address table."""
    x, weight1, weight2, up, activation = saved
    # Shared tasks start after all routed readers have finished these scratch buffers.
    def temporary(reference: torch.Tensor, storage: torch.Tensor | None, offset: int = 0) -> torch.Tensor:
        """Use a sufficiently large dead scratch slice or allocate owned storage."""
        if storage is not None and storage.numel() >= offset + reference.numel():
            return storage.view(-1).narrow(0, offset, reference.numel()).view_as(reference)
        return torch.empty_like(reference)

    act_scratch, up_scratch = scratch if scratch is not None else (None, None)
    reuse_weights = (x.shape[0] > 0 and up_scratch is not None
                     and up_scratch.numel() >= up.numel() + weight1.numel() + weight2.numel())
    dw2 = temporary(weight2, up_scratch, up.numel()) if reuse_weights else torch.empty_like(weight2)
    dactivation = temporary(activation, act_scratch)
    dup = temporary(up, up_scratch)
    dx = temporary(x, act_scratch, activation.numel()) if integrated_output else torch.empty_like(x)
    dw1 = (temporary(weight1, up_scratch, up.numel() + weight2.numel())
           if reuse_weights else torch.empty_like(weight1))
    if x.shape[0] == 0:
        # No shared task writes these gradients; routed scratch is still live.
        dw1.zero_()
        dw2.zero_()
    tensors = (dy.contiguous(), activation, dw2, weight2, dactivation, up, dup,
               weight1, dx, x, dw1)
    if integrated_output:
        tensors += (torch.empty_like(x),)
    return tensors, pointer_table(tensors), reuse_weights
