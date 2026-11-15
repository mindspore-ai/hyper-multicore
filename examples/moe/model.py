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

"""MoE Transformer with ordinary residual connections and no mHC modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from examples._layers import RMSNorm
from examples._transformer import (
    TransformerAttention, TransformerConfig, language_model_output,
)
from examples.moe.experts import MoeTransformerBlock
from hyper_multicore import MegaMoeExperts


@dataclass(frozen=True)
class MoeTransformerConfig(TransformerConfig):
    """Routed/shared experts for a TP1 model with configurable expert parallelism."""

    num_experts: int = 48
    top_k: int = 8
    intermediate_size: int = 1792
    shared_expert_intermediate_size: int = 1792
    routed_scaling_factor: float = 1.0
    ep_size: int = 8

    def __post_init__(self) -> None:
        """Validate shared dimensions and the expert-parallel configuration."""
        super().__post_init__()
        for name in ("num_experts", "top_k", "intermediate_size", "shared_expert_intermediate_size", "ep_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_experts % self.ep_size:
            raise ValueError("num_experts must be divisible by ep_size")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")


class MoeTransformerDecoderLayer(nn.Module):
    """Pre-norm attention and MoE with two ordinary residual additions."""

    def __init__(self, config: MoeTransformerConfig, ep_group: Any | None, mega_kernels: tuple[str, ...]) -> None:
        """Construct attention, routed/shared experts and their ordinary norms."""
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = TransformerAttention(config)
        self.mlp = MoeTransformerBlock(config, ep_group, mega_kernels)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply attention and experts without four-stream residual state.

        Args:
            hidden_states: Ordinary residual activations shaped [batch, sequence, hidden].
            position_ids: Token positions used by rotary attention.
        """
        hidden_states = hidden_states + self.self_attn(self.input_norm(hidden_states), position_ids)
        return hidden_states + self.mlp(self.post_attention_norm(hidden_states))


class MoeTransformerModel(nn.Module):
    """Random-weight MoE model; only Router/expert implementations are selectable."""

    def __init__(
        self, config: MoeTransformerConfig, *, ep_group: Any | None = None,
        mega_kernels: tuple[str, ...] = ("moe", "gate"),
    ) -> None:
        """Build one MoE architecture shared by common and Mega backends."""
        super().__init__()
        unknown = set(mega_kernels) - {"moe", "gate"}
        if unknown:
            raise ValueError(f"Unsupported Mega kernels: {sorted(unknown)}")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(MoeTransformerDecoderLayer(config, ep_group, mega_kernels)
                                    for _ in range(config.num_layers))
        MegaMoeExperts.share_execution_resources(layer.mlp.experts.executor for layer in self.layers)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        """Return vocabulary logits and optional next-token loss.

        Args:
            input_ids: Rank-local token IDs shaped [batch, sequence].
            labels: Optional labels with matching leading dimensions.
        """
        hidden_states = self.embed_tokens(input_ids)
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.int64)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        return language_model_output(hidden_states, labels, self.norm, self.lm_head)

    def expert_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return expert shards excluded from replicated dense gradient averaging."""
        return tuple(parameter for layer in self.layers for parameter in layer.mlp.experts.parameters())

    def close(self) -> None:
        """Release managed expert resources after outstanding work completes."""
        for layer in self.layers:
            layer.mlp.experts.close()
