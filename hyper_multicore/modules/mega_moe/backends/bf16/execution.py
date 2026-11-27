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

from hyper_multicore.modules.hot_replica.checkpoint import forward_save_policy
from hyper_multicore.modules.hot_replica.shmem_transfer import ReplicaExecutionMode

from ...route import PreparedTopKRoute, restore_topk_output

from .autograd import _CheckpointedMegaMoeFunction, execute_mega_moe_with_permutation, _execute_mega_moe_without_autograd

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
    pull = resources.spec.dispatch_mode == "pull"
    before_fc2 = resources.spec.apply_probs_before_fc2
    binding = route.weight_binding
    plan_options = {}
    if binding is not None:
        binding, plan_options = _prepare_replica_binding(binding, route.tensor_dump, resources.spec.num_experts)
    plan = resources.plan_for_tokens(route.metadata.plan_tokens, hidden_flat.device, **plan_options)
    options = {} if binding is None else {"weight_binding": binding}
    record = route.checkpoint_record
    if not torch.is_grad_enabled():
        save = None
        if record is not None and before_fc2 and hidden_flat.shape[0]:
            save = _save_replay_activations(record, plan.spec, route, hidden_flat, topk_weights)
        output = _execute_mega_moe_without_autograd(
            topk_weights, *weights, route, plan, resources.workspace,
            workspace_claimed=pull, save_activation=save, **options)
        if record is not None and binding is not route.weight_binding:
            # Both bank layouts save the same owned guest records. The
            # original binding still selects its own replay protocol.
            route.weight_binding.snapshot = binding.snapshot
    elif record is not None and record.allow_activation:
        resources.placement.statistics["skipped_recomputes"] += 1
        return _CheckpointedMegaMoeFunction.apply(
            hidden_flat, *weights, route.metadata, plan, resources.workspace,
            record.activation, topk_weights, binding,
        ).reshape_as(hidden_states)
    else:
        integrated_unpermute = pull or before_fc2
        if not integrated_unpermute and resources.placement is not None:
            integrated_unpermute = _apply_checkpoint_saves(plan.spec, binding, record, options)
        output = execute_mega_moe_with_permutation(
            hidden_flat, topk_ids, *weights, route, plan, resources.workspace,
            topk_weights=topk_weights if integrated_unpermute else None,
            workspace_claimed=pull, **options)
        if not integrated_unpermute:
            output = restore_topk_output(output, route.unpermute_mapping, topk_weights)
    if binding is not None:
        resources.placement.record_submission(binding.mode)
    return output.reshape_as(hidden_states)


def _apply_checkpoint_saves(spec, binding, record, options):
    """Configure after-FC2 A3 Push replay and report whether it owns unpermutation."""
    policy = forward_save_policy()
    if policy is None or spec.architecture != 2201 or spec.transport_engine != "mte":
        return False
    replay, defer = policy
    if defer:
        options["checkpoint_replay"] = replay
    if (record is not None and record.guest_snapshot_bytes and binding is not None and binding.fused
            and (not replay or record.allow_guest_restore)):
        options["checkpoint_record"] = record
    return defer


def _prepare_replica_binding(binding, tensor_dump, num_experts):
    """Match the plan to the execution slots, gradient precision and diagnostic mode."""
    options = {}
    route_experts = binding.placement.counts_by_source.shape[1]
    if route_experts != num_experts:
        options["route_experts"] = route_experts
    if binding.mode is ReplicaExecutionMode.FP32_BAND:
        options["weight_grad_dtype"] = torch.float32
    if binding.fused and tensor_dump:
        binding = binding.for_tensor_dump()
    if binding.fused:
        options.update(replica_pipeline=True, reserve_merge_workers=binding.owns_returns)
    return binding, options


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
