# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""BF16 autograd launch and restoration over the common invocation lifecycle."""
import torch
from ...route import restore_topk_output
from .autograd import execute_mega_moe_with_permutation, _execute_mega_moe_without_autograd


def forward(module, hidden_states, hidden_flat, topk_ids, topk_weights, weights, resources, route):
    """Launch BF16 experts over the route prepared by the frontend."""
    del module
    plan = resources.plan_for_tokens(route.metadata.plan_tokens, hidden_flat.device)
    pull = resources.spec.dispatch_mode == "pull"
    before_fc2 = resources.spec.apply_probs_before_fc2
    if not torch.is_grad_enabled():
        return _execute_mega_moe_without_autograd(
            topk_weights, *weights, route, plan, resources.workspace,
            workspace_claimed=pull).reshape_as(hidden_states)
    output = execute_mega_moe_with_permutation(
        hidden_flat, topk_ids, *weights, route, plan, resources.workspace,
        topk_weights=topk_weights if pull or before_fc2 else None,
        workspace_claimed=pull)
    if not pull and not before_fc2:
        output = restore_topk_output(output, route.unpermute_mapping, topk_weights)
    return output.reshape_as(hidden_states)
