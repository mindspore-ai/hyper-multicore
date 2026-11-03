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

"""Local expert compute using the borrowed parameter views of an active placement."""
import torch
from torch.nn import functional as F

from .precision import expert_linear
from .npu_grouped_swiglu import npu_grouped_swiglu


def run_local_experts(experts, dispatched_states, local_expert_indices, placement, *,
                      sort_keys, count_tokens, resolve_swiglu_weights):
    """Compute after the calling model's FSDP hooks have unsharded the local experts."""
    expert_parameters, expert_gradient_targets = placement.parameters(experts)
    local_expert_count = 2 * experts.local_expert_count
    token_order = sort_keys(
        local_expert_indices, bound=local_expert_count)
    sorted_states = dispatched_states[token_order]
    local_expert_counts = count_tokens(
        local_expert_indices, local_expert_count)
    if getattr(experts, "_ep_use_grouped_gemm", False):
        grouped_forward = getattr(experts, "forward_expert_major", None)
        if callable(grouped_forward):
            options = {"expert_parameters": expert_parameters}
            if expert_gradient_targets is not None:
                options["expert_gradient_targets"] = expert_gradient_targets
            sorted_output = grouped_forward(sorted_states, local_expert_counts, **options)
        else:
            gate_weight, up_weight, down_weight = resolve_swiglu_weights(experts, expert_parameters)
            if up_weight is not None:
                raise ValueError(
                    "EP grouped GEMM currently requires packed gate_up_proj weights"
                )
            grouped_options = {}
            if expert_gradient_targets is not None:
                targets_by_id = {id(expert_parameters[name]): target
                                 for name, target in expert_gradient_targets.items()}
                gate_target = targets_by_id.get(id(gate_weight))
                down_target = targets_by_id.get(id(down_weight))
                if gate_target is not None:
                    gate_target.configure(tuple(gate_weight.transpose(1, 2).shape), True)
                if down_target is not None:
                    down_target.configure(tuple(down_weight.transpose(1, 2).shape), True)
                grouped_options['gradient_targets'] = (gate_target, down_target)
            sorted_output = npu_grouped_swiglu(
                sorted_states, gate_weight, down_weight, local_expert_counts, **grouped_options)
        if sorted_states.shape[0] == 0:
            control = next((parameter for parameter in expert_parameters.values() if parameter.requires_grad), None)
            if control is not None:
                sorted_output = sorted_output + control.reshape(-1)[:0].sum().to(sorted_output.dtype)
        output = torch.empty_like(sorted_output)
        output[token_order] = sorted_output
        return output

    gate_weight, up_weight, down_weight = resolve_swiglu_weights(experts, expert_parameters)
    # The eager path needs host ints: read them in one transfer rather than
    # draining the queue once per local expert.
    gate_target = up_target = down_target = None
    if expert_gradient_targets is not None:
        targets_by_id = {id(expert_parameters[name]): target for name, target in expert_gradient_targets.items()}
        gate_target = targets_by_id.get(id(gate_weight))
        up_target = targets_by_id.get(id(up_weight))
        down_target = targets_by_id.get(id(down_weight))
    expert_token_counts = local_expert_counts.tolist()
    sorted_outputs = []
    token_start = 0
    for local_expert_index in range(local_expert_count):
        expert_token_count = expert_token_counts[local_expert_index]
        expert_states = sorted_states[token_start:token_start + expert_token_count]
        if up_weight is None:
            gate_up_states = (F.linear(expert_states, gate_weight[local_expert_index]) if gate_target is None
                              else expert_linear(expert_states, gate_weight, local_expert_index, gate_target))
            apply_gate = getattr(experts, "_ep_apply_gate", None)
            if apply_gate is None:
                gate_states, up_states = gate_up_states.chunk(2, dim=-1)
                activation = getattr(experts, "_ep_act_fn", F.silu)
                activated_states = activation(gate_states) * up_states
            else:
                activated_states = apply_gate(gate_up_states)
        else:
            gate_states = (F.linear(expert_states, gate_weight[local_expert_index]) if gate_target is None
                           else expert_linear(expert_states, gate_weight, local_expert_index, gate_target))
            up_states = (F.linear(expert_states, up_weight[local_expert_index]) if up_target is None
                         else expert_linear(expert_states, up_weight, local_expert_index, up_target))
            activation = getattr(experts, "_ep_act_fn", F.silu)
            activated_states = activation(gate_states) * up_states
        sorted_outputs.append(
            F.linear(activated_states, down_weight[local_expert_index]) if down_target is None
            else expert_linear(activated_states, down_weight, local_expert_index, down_target))
        token_start += expert_token_count

    sorted_output = torch.cat(sorted_outputs)
    output = torch.empty_like(sorted_output)
    output[token_order] = sorted_output
    return output
