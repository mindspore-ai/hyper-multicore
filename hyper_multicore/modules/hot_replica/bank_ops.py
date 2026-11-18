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
"""Publish expert weights and return borrowed gradients in FP32."""

from functools import lru_cache
from types import ModuleType

import torch

from .allocator import load_native_extension


@lru_cache(maxsize=1)
def load_bridge() -> ModuleType:
    """Load the selected Torch/PTA adapter's expert-transfer bridge."""
    return load_native_extension("_hot_replica_shared_bridge")


def prefetch_banks(
    weights: tuple[torch.Tensor, ...],
    banks: tuple[torch.Tensor, ...],
    copies: torch.Tensor,
    rank: int,
    blocks: int,
) -> None:
    """Publish home weights before peers fetch their selected borrowed records."""
    local = banks[0].shape[0] // 2
    extra = copies.shape[1]
    for weight, bank in zip(weights, banks):
        bank[:local].copy_(weight.reshape_as(bank[:local]))
    for i, bank in enumerate(banks):
        load_bridge().prefetch(
            bank, bank[local : local + extra], copies[rank], rank, copies.shape[0], blocks, i == 0, i + 1 == len(banks)
        )


def reduce_banks(
    gradients: tuple[torch.Tensor, ...],
    banks: tuple[torch.Tensor, ...],
    copies: torch.Tensor,
    rank: int,
    blocks: int,
    output_dtypes: tuple[torch.dtype, ...] | None = None,
    *,
    published_slots: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Publish FP32 borrowed gradients, sum at owners, then cast to parameter dtype."""
    local = banks[0].shape[0] // 2
    extra = copies.shape[1]
    slots = [bank.view(-1).view(torch.float32).view(local, *bank.shape[1:])[:extra] for bank in banks]
    # Peers read only slots assigned by this invocation's copy table.
    count = extra if published_slots is None else published_slots
    if count:
        for slot, gradient in zip(slots, gradients):
            slot[:count].copy_(gradient[local : local + count])
    owners = []
    for i, gradient in enumerate(gradients):
        home = gradient[:local].float()
        result = (
            torch.empty_like(home, dtype=torch.bfloat16)
            if output_dtypes is not None and output_dtypes[i] == torch.bfloat16
            else home
        )
        load_bridge().reduce(home, result, slots[i], copies, rank, blocks, i == 0, i + 1 == len(banks))
        owners.append(result)
    return tuple(owners)
