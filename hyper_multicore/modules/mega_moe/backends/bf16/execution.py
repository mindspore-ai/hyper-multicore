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
"""BF16 autograd launch and restoration over the common invocation lifecycle."""
from typing import Any, TYPE_CHECKING
import torch
from ...route import PreparedTopKRoute, restore_topk_output

from .autograd import _CheckpointedPushFunction, execute_mega_moe_with_permutation, _execute_mega_moe_without_autograd

if TYPE_CHECKING:
    from .resources import BF16Resources


def forward(
    module: Any, hidden_states: torch.Tensor, hidden_flat: torch.Tensor,
    topk_ids: torch.Tensor, topk_weights: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor], resources: "BF16Resources",
    route: PreparedTopKRoute,
) -> torch.Tensor:
    """Launch BF16 experts over the route prepared by the frontend."""
    del module
    module_resources = resources
    resources = route.execution_resources or resources
    binding = route.weight_binding
    ordinary_options = {}
    if binding is not None and module_resources.placement.source_bound is not None:
        home = module_resources.spec.local_experts
        nominal = (home >= 2 * resources.spec.num_cube_cores and resources.spec.intermediate_size < 2048
                   and resources.spec.lowering != "dense" and resources.spec.ready_depth is None)
        ordinary_options = dict(
            weight_grad_dtype=torch.float32, route_experts=binding.placement.counts_by_source.shape[1],
            nominal_schedule=nominal,
            batch_experts=nominal and route.metadata.plan_tokens * resources.spec.top_k <= 128 * home,
        )
        if ordinary_options["batch_experts"]:
            binding.fused = False
    if binding is not None and binding.fused:
        plan = resources.plan_for_tokens(
            route.metadata.plan_tokens, hidden_flat.device,
            weight_grad_dtype=torch.float32, route_experts=ordinary_options.get("route_experts"),
            replica_pipeline=True, reserve_merge_workers=binding.owns_returns,
        )
        if plan.fwd_diagnostics.is_active or plan.bwd_diagnostics.is_active:
            binding.fused = False
    if binding is None or not binding.fused:
        plan = resources.plan_for_tokens(route.metadata.plan_tokens, hidden_flat.device, **ordinary_options)
    pull = resources.spec.dispatch_mode == "pull"
    before_fc2 = resources.spec.apply_probs_before_fc2
    options = {} if route.weight_binding is None else {"weight_binding": route.weight_binding}
    record = route.checkpoint_record
    if not torch.is_grad_enabled():
        save = None
        if record is not None and before_fc2 and hidden_flat.shape[0]:
            save = _save_replay_activations(record, plan.spec, route, hidden_flat, topk_weights)
        return _execute_mega_moe_without_autograd(
            topk_weights, *weights, route, plan, resources.workspace,
            workspace_claimed=pull, save_activation=save, **options).reshape_as(hidden_states)
    if record is not None and record.allow_activation:
        module_resources.placement.statistics["skipped_recomputes"] += 1
        return _CheckpointedPushFunction.apply(
            hidden_flat, *weights, route.metadata, plan, resources.workspace,
            record.activation, topk_weights, False, route.weight_binding,
        ).reshape_as(hidden_states)
    output = execute_mega_moe_with_permutation(
        hidden_flat, topk_ids, *weights, route, plan, resources.workspace,
        topk_weights=topk_weights if pull or before_fc2 else None,
        workspace_claimed=pull, **options)
    if not pull and not before_fc2:
        output = restore_topk_output(output, route.unpermute_mapping, topk_weights)
    return output.reshape_as(hidden_states)


def _save_replay_activations(record, spec, route, hidden: torch.Tensor, probs: torch.Tensor):
    """Reserve optional copies within the same per-device checkpoint budget."""
    from hyper_multicore.modules.hot_replica.checkpoint import ActivationReservation  # pylint: disable=import-outside-toplevel

    rows, hidden_size = hidden.shape
    required = (
        (route.metadata.expert_capacity * (hidden_size + 3 * spec.intermediate_size) + 2 * rows * hidden_size)
        * hidden.element_size() + rows * spec.top_k * (4 + probs.element_size())
    )
    reservation = ActivationReservation.reserve(hidden, required, spec.ep_group)
    if reservation is None:
        return None

    def save(
        output: torch.Tensor, dispatch: torch.Tensor, up: torch.Tensor,
        activation: torch.Tensor, mapping: torch.Tensor,
    ) -> None:
        """Retain owned tensors beyond the forward workspace lease."""
        record.inputs = hidden.detach(), probs.detach()
        record.input_versions = hidden._version, probs._version  # pylint: disable=protected-access
        record.activation = output.detach().clone(), dispatch, up, activation, mapping, reservation

    return save
