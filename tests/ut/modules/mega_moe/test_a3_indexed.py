# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Check compact route order and EP-wide readiness for A3 indexed dispatch."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hyper_multicore.modules.hot_replica.bounded_placement import _indexed_maps
from hyper_multicore.modules.mega_moe import a3_indexed, route
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_indexed_maps_preserve_both_orders():
    slots = np.array([4, 1, 3, 0, 2], dtype=np.int32)
    counts = np.array([1, 3, 0, 2, 2], dtype=np.int32)
    logical_to_physical, physical_to_logical = _indexed_maps(slots, counts)
    expanded = np.repeat(slots, counts[slots])
    expected = np.argsort(expanded, kind="stable")
    np.testing.assert_array_equal(physical_to_logical, expected)
    np.testing.assert_array_equal(logical_to_physical[expected], np.arange(expected.size))


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("all_ready,rebound", [(True, False), (False, False), (True, True)])
def test_placed_route_reuses_count_fence_only_for_same_allocation(all_ready, rebound):
    spec = MegaMoeSpec(
        local_num_tokens=128, hidden_size=4, intermediate_size=2, num_experts=4, top_k=2,
        initial_capacity_factor=1.0, receive_capacity=128, ep_size=2, ep_group=None,
        rank_id=0, num_cube_cores=24,
    )
    workspace = SimpleNamespace(
        forward_event_counters=torch.zeros(12, dtype=torch.int32),
        expert_buffer=object(), routed_buffer=object(), forward_ready_initialized=True,
    )
    fence = a3_indexed.CountFence(
        workspace.forward_event_counters, workspace.expert_buffer, workspace.routed_buffer, 3, "compute")
    mask = 15
    flags = [mask | a3_indexed.COUNT_FENCE_BIT, mask | (a3_indexed.COUNT_FENCE_BIT if all_ready else 0)]
    wire = [[0, 2, flag, 1, 1, 1, 1] for flag in flags]
    hidden = torch.zeros(2, 4)
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    order = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    placement = SimpleNamespace(
        counts_by_source=torch.ones(2, 4, dtype=torch.int32),
        local_receive_capacity=4, receive_capacity=4,
        physical_inverse=torch.tensor([1, 0, 3, 2], dtype=torch.int32),
        physical_to_logical=torch.tensor([2, 1, 3, 0], dtype=torch.int32),
    )
    binding = SimpleNamespace(placement=placement)

    def prepare(tokens, maximum):
        assert (tokens, maximum) == (128, 4)
        if rebound:
            workspace.forward_event_counters = torch.zeros_like(workspace.forward_event_counters)
            workspace.expert_buffer = object()
            workspace.routed_buffer = object()

    result = route._prepare_placed_route(
        hidden, ids, spec, workspace, None, placement.counts_by_source[0],
        torch.tensor(wire, dtype=torch.int32), None, "compute", prepare,
        lambda *_args, **kwargs: binding if kwargs["indexed"] else None,
        logical_inverse=order, logical_order=order, count_fence=fence,
        indexed_readback=(route._RouteSummary(4, 128, 4), wire, [4, 4]),
    )
    assert result.routed_tokens is hidden
    assert result.count_fence is (fence if all_ready and not rebound else None)
    assert result.metadata.local_num_tokens == hidden.shape[0]
    assert result.metadata.plan_tokens == 128
    native_offsets = torch.arange(4, dtype=torch.int64) * spec.hidden_size
    torch.testing.assert_close(result.indexed_dispatch_offsets[:4], native_offsets)
    torch.testing.assert_close(result.indexed_dispatch_offsets[4:].view(torch.int32),
                               order[placement.physical_to_logical])
