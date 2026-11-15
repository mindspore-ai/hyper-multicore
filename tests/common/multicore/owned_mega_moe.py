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
"""Test-only owner for exercising MegaMoe under an ordinary model boundary."""

from typing import Any

import torch

from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions
from hyper_multicore.experimental import MegaMoeTuningOptions


class OwnedMegaMoeExperts(MegaMoeExperts):
    """Keep test parameters outside the executor's implementation contract."""

    def __init__(self, *, create_parameters: bool = True, expert_weight_layout: str = "EIO", **kwargs: Any) -> None:
        """Create the executor and optional model-owned test weights."""
        tuning = {name: kwargs.pop(name) for name in (
            "dispatch_mode", "initial_capacity_factor", "capacity_growth_factor", "dispatch_split", "combine_split",
        ) if name in kwargs}
        if tuning:
            if "tuning_options" in kwargs:
                raise TypeError("Pass test tuning fields or tuning_options, not both")
            kwargs["tuning_options"] = MegaMoeTuningOptions(**tuning)
        numerics = {name: kwargs.pop(name) for name in (
            "swiglu_limit", "clamp_after_silu", "apply_probs_before_fc2",
        ) if name in kwargs}
        if numerics:
            if "numerics_options" in kwargs:
                raise TypeError("Pass test numerical fields or numerics_options, not both")
            kwargs["numerics_options"] = MegaMoeNumericsOptions(**numerics)
        super().__init__(**kwargs)
        self.expert_weight_layout = expert_weight_layout
        if create_parameters:
            hidden, intermediate, local = self.hidden_size, self.intermediate_size, self.local_experts
            shapes = (((local, hidden, intermediate * 2), (local, intermediate, hidden))
                      if expert_weight_layout == "EIO" else
                      ((local, intermediate * 2, hidden), (local, hidden, intermediate)))
            self.gate_up_weight = torch.nn.Parameter(torch.empty(shapes[0]))
            self.down_weight = torch.nn.Parameter(torch.empty(shapes[1]))
            for expert in range(local):
                torch.nn.init.xavier_uniform_(self.gate_up_weight[expert])
                torch.nn.init.xavier_uniform_(self.down_weight[expert])
        else:
            self.register_parameter("gate_up_weight", None)
            self.register_parameter("down_weight", None)

    def forward(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, *,
                expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
                tokens_per_expert: torch.Tensor | None = None) -> torch.Tensor:
        """Supply current test parameters to the parameterless executor."""
        weights = (self.gate_up_weight, self.down_weight) if expert_weights is None else expert_weights
        return super().forward(hidden_states, topk_ids, topk_weights,
                               expert_weights=weights, tokens_per_expert=tokens_per_expert)

    def _get_execution_resources(self, tensor: Any) -> Any:
        """Support direct plan inspection after validating the test-owned weights."""
        if self._resource_group.resources is None and self._pending_layout is None:
            self._validate_tensors(tensor, (self.gate_up_weight, self.down_weight))
        return super()._get_execution_resources(tensor)
