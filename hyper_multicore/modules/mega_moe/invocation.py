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
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from functools import partial
from collections.abc import Iterator
from typing import TYPE_CHECKING
import torch
from hyper_multicore.diagnostics.api import is_tensor_dump_active
from .heap_manager import _HeapCapacityError
from .route import (
    _INVALID_WEIGHT_MASK, _TENSOR_DUMP_MASK, PreparedTopKRoute, _materialize_route_permutation,
    _prepare_placed_route, prepare_topk_route,
)

if TYPE_CHECKING:
    from ..base_planner import BasePlanner
    from .resources import _MegaMoeExecutionResources


def prepare_expert_invocation(
    hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
    counts: torch.Tensor | None, weights: tuple[torch.Tensor, torch.Tensor],
    resources: "_MegaMoeExecutionResources", *, weight_error: bool = False,
    planner: "BasePlanner | None" = None, scheduled: bool = False,
) -> AbstractContextManager[PreparedTopKRoute]:
    """Converge weight errors on the existing route collective before physical binding.

    Scheduled calls queue the agreement now and wait on context entry.
    Pull source publication and consumption share one lease. Automatic capacity
    changes happen before claiming that lease. No precision implementation may
    repeat route preparation or add a second validation collective.
    """
    placement = getattr(resources, "placement", None)
    tensor_dump = placement is not None and is_tensor_dump_active()
    mask = _INVALID_WEIGHT_MASK if weight_error else (
        int(torch.is_grad_enabled()) | (int(hidden.requires_grad) << 1)
        | (int(weights[0].requires_grad) << 2) | (int(weights[1].requires_grad) << 3)
        | (int(probs.requires_grad) << 4))
    if tensor_dump:
        mask |= _TENSOR_DUMP_MASK
    pull = resources.spec.dispatch_mode == "pull"
    leased_workspace = None

    def prepare_workspace(tokens: int, received: int, *, placed: bool = False) -> None:
        """Prepare the capacity before claiming a source publication lease."""
        nonlocal leased_workspace
        try:
            resources.prepare_auto_workspace(tokens, received, hidden)
        except _HeapCapacityError:
            if placed:
                placement.statistics["heap_rejections"] += 1
            raise
        if pull:
            leased_workspace = resources.workspace

    record, replay = None, False
    finish_vote, pending = None, None
    if placement is not None:
        from ..hot_replica.checkpoint import agree_replay, route_record  # pylint: disable=import-outside-toplevel

        record, replay = route_record(resources)
    if scheduled:
        from ..hot_replica.prefetch import start_replay_vote, start_wire  # pylint: disable=import-outside-toplevel

        if replay:
            finish_vote = start_replay_vote(record, hidden, ids, probs, counts, resources, mask, weight_error)
        else:
            pending = start_wire(hidden, ids, probs, counts, resources, mask)
    elif replay:
        finish_vote = partial(agree_replay, record, hidden, ids, probs, counts, resources.spec,
                              mask, weight_error=weight_error)

    @contextmanager
    def consume() -> Iterator[PreparedTopKRoute]:
        nonlocal leased_workspace, record
        try:
            if finish_vote is not None and finish_vote():
                placement.statistics["route_replays"] += 1
                yield _restore_checkpoint_route(record, hidden, ids, resources, prepare_workspace, tensor_dump)
                return
            if replay:
                placement.statistics["route_replay_misses"] += 1
                record = None
            options = {"autograd_mask": mask}
            if planner is not None:
                options["select_placement"] = partial(placement.select, planner=planner)
                if placement.uses_placement_wire():
                    options["reuse_placement_geometry"] = resources.spec.ep_size * resources.spec.num_experts <= 16384
            if resources.spec.auto_tokens:
                options["prepare_workspace"] = prepare_workspace
            elif pull:
                with torch.inference_mode(False):
                    resources.workspace.ensure(resources.spec, hidden.dtype, hidden.device)
                resources.workspace.claim()
                leased_workspace = resources.workspace
            if record is not None and counts is None and not weight_error:
                record.stream = torch.npu.current_stream(hidden.device)
            with torch.inference_mode(False), torch.no_grad():
                if pending is None:
                    route = prepare_topk_route(hidden, ids, probs, resources.spec, counts,
                                               workspace=resources.workspace, **options)
                else:
                    stream = torch.npu.current_stream(hidden.device)
                    if not pull:
                        resources.workspace.wait_for_reuse(stream=stream)
                    snapshot = pending.snapshot(resources.spec, resources.workspace, stream)
                    route = _prepare_placed_route(
                        hidden, ids, resources.spec, resources.workspace, None, pending.payload[3:], pending.gathered,
                        None, stream, prepare_workspace, options["select_placement"], mask,
                        reuse_placement_geometry=True, snapshot=snapshot,
                    )
            if not pull and not resources.spec.auto_tokens:
                with torch.inference_mode(False):
                    resources.ensure_capacity(route.maximum_received_slots)
            yield _record_checkpoint_route(route, record, counts, ids)
        finally:
            if leased_workspace is not None:
                leased_workspace.release()

    return consume()


def _restore_checkpoint_route(record, hidden, ids, resources, prepare_workspace, tensor_dump):
    """Reuse agreed descriptors, publishing a fresh permutation if recompute is required."""
    route = record.route
    if tensor_dump:
        record.allow_activation = False
    resources.workspace.wait_for_reuse()
    if not record.allow_activation:
        with torch.inference_mode(False), torch.no_grad():
            if resources.spec.dispatch_mode == "pull":
                prepare_workspace(route.metadata.plan_tokens, route.maximum_received_slots)
            binding = route.weight_binding
            virtual_ids = binding.placement.virtual_ids if binding is not None else ids
            rows, mapping = _materialize_route_permutation(hidden, virtual_ids, resources.spec, resources.workspace)
        route = replace(route, routed_tokens=rows, unpermute_mapping=mapping)
    return replace(route, checkpoint_record=record, tensor_dump=tensor_dump)


def _record_checkpoint_route(route, record, counts, ids):
    """Keep the route descriptors and placement with the checkpoint invocation."""
    if record is None or counts is not None:
        return route
    record.route = replace(route, routed_tokens=None, unpermute_mapping=None)
    record.ids = ids.detach().clone()
    return replace(route, checkpoint_record=record)
