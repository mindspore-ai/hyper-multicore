"""Immutable launch bounds must not retain a previous invocation's operands."""
from collections import OrderedDict
from dataclasses import replace
import ctypes

import torch

from hyper_multicore.modules.mega_moe.backends.mxfp8.launch import prepare_mxfp8_launch
from hyper_multicore.modules.mega_moe.backends.mxfp8.launch_contract import MxLaunchDescriptor
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


def specification():
    return MegaMoeSpec(local_num_tokens=4096,hidden_size=5120,intermediate_size=1792,
                       num_experts=384,top_k=8,initial_capacity_factor=None,
                       receive_capacity=32768,ep_size=64,ep_group=None,rank_id=48,
                       num_cube_cores=32,precision='mxfp8',architecture=3510)


def test_launch_reuses_bounds_for_fresh_storage_and_rejects_stale_sizes():
    cache=OrderedDict()
    runtime=torch.empty(128,dtype=torch.uint8)
    events=torch.empty(256,dtype=torch.uint8)
    spec=specification()
    first=prepare_mxfp8_launch([torch.empty(64)],runtime,events,spec,cache=cache)
    same=prepare_mxfp8_launch([torch.ones(64)],runtime,events,spec,cache=cache)
    assert first is same
    larger=prepare_mxfp8_launch([torch.empty(65)],runtime,events,spec,cache=cache)
    assert larger is not first
    record=MxLaunchDescriptor.from_buffer_copy(larger.numpy().tobytes())
    assert record.buffer_bytes[0]==65*4
    changed=prepare_mxfp8_launch([torch.empty(65)],runtime,events,replace(spec,rank_id=49),cache=cache)
    assert changed is not larger
    assert len(cache)==3


def test_launch_cache_is_bounded_and_eviction_preserves_live_descriptors():
    cache=OrderedDict()
    runtime=torch.empty(128,dtype=torch.uint8)
    events=torch.empty(256,dtype=torch.uint8)
    spec=specification()
    retained=prepare_mxfp8_launch([torch.empty(1)],runtime,events,spec,cache=cache)
    for count in range(2,16):
        prepare_mxfp8_launch([torch.empty(count)],runtime,events,spec,cache=cache)
    assert len(cache)==8
    assert retained.numel()==ctypes.sizeof(MxLaunchDescriptor)
    assert MxLaunchDescriptor.from_buffer_copy(retained.numpy().tobytes()).buffer_bytes[0]==4


def test_cache_hit_avoids_reserializing_but_tracks_changed_bounds():
    from unittest.mock import patch
    from hyper_multicore.modules.mega_moe.backends.mxfp8 import launch
    cache = OrderedDict()
    runtime, events = torch.empty(128, dtype=torch.uint8), torch.empty(256, dtype=torch.uint8)
    spec = specification()
    with patch.object(launch, "serialize_mx_launch_metadata", wraps=launch.serialize_mx_launch_metadata) as serialize:
        first = launch.prepare_mxfp8_launch([torch.empty(64)], runtime, events, spec, cache=cache)
        same = launch.prepare_mxfp8_launch([torch.ones(64)], runtime, events, spec, cache=cache)
        assert same is first and serialize.call_count == 1
        other = launch.prepare_mxfp8_launch([torch.ones(64)], runtime, events[:128], spec, cache=cache)
        assert other is not first and serialize.call_count == 2
        record = MxLaunchDescriptor.from_buffer_copy(other.numpy().tobytes())
        assert record.event_bytes == 128
        changed = launch.prepare_mxfp8_launch([torch.ones(64)], runtime, events, replace(spec,local_num_tokens=8192), cache=cache)
        assert changed is not first and serialize.call_count == 3
