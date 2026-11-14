# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""BF16 route and execution orchestration with the existing lease ordering."""
import torch
from ...route import prepare_topk_route, restore_topk_output
from .autograd import execute_mega_moe_with_permutation

def forward(self, hidden_states, hidden_flat, topk_ids, topk_weights,
            tokens_per_expert, weights, resources):
    route_options = {"autograd_mask": (
        int(torch.is_grad_enabled()) | (int(hidden_flat.requires_grad) << 1)
        | (int(weights[0].requires_grad) << 2) | (int(weights[1].requires_grad) << 3)
        | (int(topk_weights.requires_grad) << 4)
    )}
    pull = self.dispatch_mode == "pull"
    automatic = resources.spec.auto_tokens
    fast_no_grad = self.nograd_no_copy and not torch.is_grad_enabled() and resources.spec.transport_engine == "udma"
    claimed = False
    def prepare_workspace(tokens: int, received: int) -> None:
        """Grow before publishing pull source rows into symmetric memory.

        Args:
            tokens: Current local source-token count.
            received: Current routed receive-row count.
        """
        nonlocal claimed
        resources.prepare_auto_workspace(tokens, received, hidden_flat)
        claimed = pull

    if automatic:
        route_options["prepare_workspace"] = prepare_workspace
    elif pull:
        resources.workspace.ensure(resources.spec, hidden_flat.dtype, hidden_flat.device)
        resources.workspace.claim()
        claimed = True
    try:
        # Pull source writes and execution must share one workspace lease.
        with torch.no_grad():
            route = prepare_topk_route(
                hidden_flat,
                topk_ids,
                topk_weights,
                resources.spec,
                tokens_per_expert,
                workspace=resources.workspace,
                **route_options,
            )
        if not pull and not automatic:
            resources.ensure_capacity(route.maximum_received_slots)
        plan = (resources.plan_for_tokens(route.metadata.plan_tokens, hidden_flat.device)
                if automatic else resources.plan)
        expert_output = execute_mega_moe_with_permutation(
            hidden_flat,
            topk_ids,
            weights[0],
            weights[1],
            route,
            plan,
            resources.workspace,
            topk_weights=topk_weights if pull or fast_no_grad else None,
            workspace_claimed=pull,
        )
    finally:
        if claimed:
            resources.workspace.release()
    if pull or fast_no_grad:
        return expert_output.reshape_as(hidden_states)
    output = restore_topk_output(
        expert_output,
        route.unpermute_mapping,
        topk_weights,
    )
    return output.reshape(hidden_states.shape)
