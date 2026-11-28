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
"""Per-invocation routing and leases shared by all expert precision backends."""
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from collections.abc import Iterator
from typing import TYPE_CHECKING
import torch
from .route import (
    _INVALID_WEIGHT_MASK, PreparedTopKRoute, _permute_topk_input, prepare_topk_route,
)

if TYPE_CHECKING:
    from ..base_planner import BasePlanner
    from .resources import _MegaMoeExecutionResources


@contextmanager
def prepare_expert_invocation(
    hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
    counts: torch.Tensor | None, weights: tuple[torch.Tensor, torch.Tensor],
    resources: "_MegaMoeExecutionResources", *, weight_error: bool = False,
    planner: "BasePlanner | None" = None,
) -> Iterator[PreparedTopKRoute]:
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

    def prepare_workspace(tokens: int, received: int) -> None:
        """Prepare the capacity before claiming a source publication lease."""
        nonlocal claimed
        resources.prepare_auto_workspace(tokens, received, hidden)
        claimed = pull

    record = None
    if not pull and resources.placement is not None:
        replay, record = _try_checkpoint_replay(resources, hidden, ids, probs, counts, mask, weight_error)
        if replay is not None:
            yield replay
            return

    options = {"autograd_mask": mask}
    if planner is not None:
        options["select_placement"] = partial(resources.placement.select, planner=planner)
    if resources.spec.auto_tokens:
        options["prepare_workspace"] = prepare_workspace
        if resources.spec.device_route_capacity_factor is not None:
            options["prepare_execution"] = resources.prepare_forward
    elif pull:
        with torch.inference_mode(False):
            resources.workspace.ensure(resources.spec, hidden.dtype, hidden.device)
        resources.workspace.claim()
        claimed = True
    try:
        # Routing may allocate or replace persistent SHMEM owners.
        with torch.inference_mode(False), torch.no_grad():
            route = prepare_topk_route(hidden, ids, probs, resources.spec, counts,
                                       workspace=resources.workspace, **options)
        if not pull and not resources.spec.auto_tokens:
            with torch.inference_mode(False):
                resources.ensure_capacity(route.maximum_received_slots)
        yield _record_checkpoint_route(route, record, counts, ids, hidden)
    finally:
        if claimed:
            resources.workspace.release()


def _try_checkpoint_replay(resources, hidden, ids, probs, counts, mask, weight_error):
    """Resolve recorded placement and reconstruct the permutation when needed."""
    from hyper_multicore.modules.hot_replica.checkpoint import agree_replay, route_record  # pylint: disable=import-outside-toplevel

    record, replay = route_record(resources)
    if replay and agree_replay(record, hidden, ids, probs, counts, resources.spec, mask, weight_error=weight_error):
        resources.placement.statistics["route_replays"] += 1
        route = record.route
        selected = route.execution_resources or resources
        selected.workspace.wait_for_reuse()
        if not record.allow_activation:
            with torch.no_grad():
                binding = route.weight_binding
                virtual_ids = binding.placement.virtual_ids if binding is not None else ids
                rows, mapping = _permute_topk_input(hidden, virtual_ids)
            route = replace(route, routed_tokens=rows, unpermute_mapping=mapping)
        return replace(route, checkpoint_record=record), record
    if replay:
        resources.placement.statistics["route_replay_misses"] += 1
        record = None

    return None, record


def _record_checkpoint_route(route, record, counts, ids, hidden):
    """Keep the route descriptors and placement with the checkpoint invocation."""
    if record is None or counts is not None:
        return route
    record.ids = ids.detach().clone()
    record.route = replace(route, routed_tokens=None, unpermute_mapping=None)
    record.weight_binding = route.weight_binding
    record.stream = torch.npu.current_stream(hidden.device)
    return replace(route, checkpoint_record=record)
