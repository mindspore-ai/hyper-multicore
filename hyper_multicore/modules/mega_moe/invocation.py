# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Per-invocation routing and leases shared by all expert precision backends."""
from contextlib import contextmanager
import torch
from .route import _INVALID_WEIGHT_MASK, prepare_topk_route


@contextmanager
def prepare_expert_invocation(hidden, ids, probs, counts, weights, resources, *, weight_error=False):
    """Converge weight errors on the existing route collective before physical binding.

    Pull source publication and consumption share one lease. Automatic capacity
    changes happen before claiming that lease. No precision implementation may
    repeat route preparation or add a second validation collective.
    """
    mask = _INVALID_WEIGHT_MASK if weight_error else (
        int(torch.is_grad_enabled()) | (int(hidden.requires_grad) << 1)
        | (int(weights[0].requires_grad) << 2) | (int(weights[1].requires_grad) << 3)
        | (int(probs.requires_grad) << 4))
    pull = resources.spec.dispatch_mode == "pull"
    claimed = False

    def prepare_workspace(tokens, received):
        nonlocal claimed
        resources.prepare_auto_workspace(tokens, received, hidden)
        claimed = pull

    options = {"autograd_mask": mask}
    if resources.spec.auto_tokens:
        options["prepare_workspace"] = prepare_workspace
    elif pull:
        resources.workspace.ensure(resources.spec, hidden.dtype, hidden.device)
        resources.workspace.claim()
        claimed = True
    try:
        with torch.no_grad():
            route = prepare_topk_route(hidden, ids, probs, resources.spec, counts,
                                       workspace=resources.workspace, **options)
        if not pull and not resources.spec.auto_tokens:
            resources.ensure_capacity(route.maximum_received_slots)
        yield route
    finally:
        if claimed:
            resources.workspace.release()
