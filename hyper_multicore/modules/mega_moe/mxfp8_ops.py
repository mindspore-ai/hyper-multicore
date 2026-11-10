# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Byte-alias launch adapter for the separately versioned native MX task graph."""

from functools import lru_cache
from collections.abc import Sequence

import torch

from hyper_multicore.scheduler.mxfp8 import serialize_mx_launch_metadata
from hyper_multicore.torch.ops import _load_native

from .spec import MegaMoeSpec


@lru_cache(maxsize=1)
def _load_mxfp8_native() -> None:
    _load_native()
    if torch.ops.hyper_multicore.mxfp8_task_graph_version() != 2:
        raise RuntimeError('MXFP8 task-graph adapter ABI mismatch; rebuild the local multicore component.')


@lru_cache(maxsize=None)
def require_mxfp8_window_feature(feature: str) -> None:
    """Require the primary ABI with the adopted A5 task extensions."""
    _load_mxfp8_native()
    supported = {'pull_dispatch', 'ready_math', 'nograd_row_only', 'dispatch_quant'}
    if feature not in supported:
        raise ValueError(f'Unsupported MX task extension: {feature}')


def _byte_aliases(buffers: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    aliases = []
    for tensor in buffers:
        if not tensor.is_contiguous():
            raise ValueError('MXFP8 task buffers must be contiguous; implicit physical weight copies are forbidden.')
        aliases.append(tensor.view(torch.uint8).reshape(-1))
    return aliases


def serialize_mxfp8_launch(buffers: Sequence[torch.Tensor], runtime: torch.Tensor,
                         events: torch.Tensor, spec: MegaMoeSpec) -> bytes:
    """Serialize exact current byte bounds without reading device tensor contents."""
    return serialize_mx_launch_metadata(
        [tensor.numel() * tensor.element_size() for tensor in buffers],
        rank=spec.rank_id, ep=spec.ep_size, experts=spec.num_experts, hidden=spec.hidden_size,
        local_tokens=spec.local_num_tokens, cube_count=spec.num_cube_cores,
        runtime_bytes=runtime.numel(), event_bytes=events.numel(), vector_policy=spec.mx_vector_policy,
    )


def prepare_mxfp8_launch(buffers: Sequence[torch.Tensor], runtime: torch.Tensor,
                         events: torch.Tensor, spec: MegaMoeSpec) -> torch.Tensor:
    metadata = serialize_mxfp8_launch(buffers, runtime, events, spec)
    return torch.frombuffer(bytearray(metadata), dtype=torch.uint8).to(device=runtime.device)


def launch_mxfp8_task_graph(buffers: Sequence[torch.Tensor], runtime: torch.Tensor,
                           events: torch.Tensor, diagnostic: torch.Tensor, launch: torch.Tensor,
                           *, vector_policy: str = 'legacy_odd', swiglu_row_only: bool = False,
                           direction: str = 'forward') -> None:
    """Run the native fused task graph with alias-preserving physical arguments."""
    aliases = _byte_aliases(buffers)
    _load_mxfp8_native()
    if direction not in ('forward', 'backward'):
        raise ValueError('MX native direction must be forward or backward')
    primary = (torch.ops.hyper_multicore.mxfp8_task_graph_grad if direction == 'backward'
               else torch.ops.hyper_multicore.mxfp8_task_graph)
    if vector_policy not in ('legacy_odd', 'window2'):
        raise ValueError('MXFP8 supports the primary legacy or window2 task queue')
    primary(aliases, runtime, events, diagnostic, launch)
