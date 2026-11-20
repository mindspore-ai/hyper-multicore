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
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from functools import partial
from collections.abc import Iterator
from typing import TYPE_CHECKING
import torch
from .route import (
    _INVALID_WEIGHT_MASK, PreparedTopKRoute, _materialize_route_permutation, prepare_topk_route,
)

if TYPE_CHECKING:
    from ..base_planner import BasePlanner
    from .resources import _MegaMoeExecutionResources


@contextmanager
def prepare_expert_invocation(
    hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
    counts: torch.Tensor | None, weights: tuple[torch.Tensor, torch.Tensor],
    resources: "_MegaMoeExecutionResources", *, weight_error: bool = False,
    planner: "BasePlanner | None" = None, training: bool = True,
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
    leased_workspace = None

    def prepare_workspace(tokens: int, received: int, *, selected=resources) -> None:
        """Prepare the capacity before claiming a source publication lease."""
        nonlocal leased_workspace
        selected.prepare_auto_workspace(tokens, received, hidden)
        if pull:
            leased_workspace = selected.workspace

    placement = getattr(resources, "placement", None)
    mx_placement = placement is not None and resources.spec.precision == "mxfp8"
    record, replay = None, False
    try:
        if placement is not None and not mx_placement:
            from hyper_multicore.modules.hot_replica.checkpoint import agree_replay, route_record  # pylint: disable=import-outside-toplevel

            record, replay = route_record(resources)
            if replay and agree_replay(
                    record, hidden, ids, probs, counts, resources.spec, mask, weight_error=weight_error):
                placement.statistics["route_replays"] += 1
                route = record.route
                selected = route.execution_resources or resources
                selected.workspace.wait_for_reuse()
                if pull or not record.allow_activation:
                    with torch.inference_mode(False), torch.no_grad():
                        if pull:
                            prepare_workspace(
                                route.metadata.plan_tokens, route.maximum_received_slots, selected=selected,
                            )
                        binding = route.weight_binding
                        virtual_ids = binding.placement.virtual_ids if binding is not None else ids
                        rows, mapping = _materialize_route_permutation(
                            hidden, virtual_ids, selected.spec, selected.workspace,
                        )
                    route = replace(route, routed_tokens=rows, unpermute_mapping=mapping)
                yield replace(route, checkpoint_record=record) if counts is None else route
                return
            if replay:
                placement.statistics["route_replay_misses"] += 1
                record = None

        options = {"autograd_mask": mask}
        if planner is not None and not mx_placement:
            checkpointing = record is not None or replay
            options["select_placement"] = partial(
                placement.select, planner=planner, checkpointing=checkpointing,
            )
            if placement.uses_placement_wire(checkpointing):
                options["reuse_placement_geometry"] = resources.spec.ep_size * resources.spec.num_experts <= 16384
        if resources.spec.auto_tokens:
            options["prepare_workspace"] = prepare_workspace
        elif pull:
            with torch.inference_mode(False):
                resources.workspace.ensure(resources.spec, hidden.dtype, hidden.device)
            resources.workspace.claim()
            leased_workspace = resources.workspace
        snapshot = nullcontext()
        if record is not None and not weight_error:
            record.stream = torch.npu.current_stream(hidden.device)
            if counts is not None:
                snapshot = record.save_supplied_route(ids, counts)
        # Routing may allocate or replace persistent SHMEM owners.
        with torch.inference_mode(False), torch.no_grad(), snapshot:
            if mx_placement:
                route = placement.prepare_route(hidden, ids, probs, counts, mask, training=training)
            else:
                route = prepare_topk_route(hidden, ids, probs, resources.spec, counts,
                                           workspace=resources.workspace, **options)
        if not pull and not resources.spec.auto_tokens:
            with torch.inference_mode(False):
                resources.ensure_capacity(route.maximum_received_slots)
        yield _record_checkpoint_route(route, record, counts, ids)
    finally:
        if leased_workspace is not None:
            leased_workspace.release()


def _record_checkpoint_route(route, record, counts, ids):
    """Keep the route descriptors and placement with the checkpoint invocation."""
    if record is None:
        return route
    record.route = replace(route, routed_tokens=None, unpermute_mapping=None)
    if counts is not None:
        return route
    record.ids = ids.detach().clone()
    record.weight_binding = route.weight_binding
    return replace(route, checkpoint_record=record)
