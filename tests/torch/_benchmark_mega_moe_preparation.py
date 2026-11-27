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
"""Real EP BF16 preparation worker; use the repository's locked launcher."""
import json
import hashlib
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch_npu
from hyper_multicore import MegaMoeExperts
from hyper_multicore.experimental import MegaMoeTuningOptions

from tests.common.multicore.preparation_metrics import preparation_windows


def native_forward(hidden, ids, probabilities, weight1, weight2, experts):
    """Standalone PanGu-style BF16 control, not the original PanGu job binary."""
    world, rank = dist.get_world_size(), dist.get_rank()
    local = experts // world
    counts = torch.bincount(ids.long().flatten(), minlength=experts).int()
    gathered = torch.empty((world, experts), dtype=counts.dtype, device=hidden.device)
    dist.all_gather_into_tensor(gathered.flatten(), counts)
    received = gathered[:, rank * local:(rank + 1) * local]
    recv = received.sum(1).cpu().tolist()
    send = counts.reshape(world, local).sum(1).cpu().tolist()
    routed, mapping = torch_npu.npu_moe_token_permute(hidden, ids)
    dispatch = hidden.new_empty((sum(recv), hidden.shape[1]))
    dist.all_to_all_single(dispatch, routed, recv, send)
    if dispatch.shape[0]:
        labels = torch.arange(local, dtype=torch.int32, device=hidden.device).repeat(world)
        labels = torch.repeat_interleave(labels, received.flatten().long(), output_size=dispatch.shape[0])
        expert_input, local_mapping = torch_npu.npu_moe_token_permute(dispatch, labels[:, None])
        groups = received.sum(0).cumsum(0).long()
        up = torch_npu.npu_grouped_matmul(
            x=[expert_input], weight=[weight1], bias=[], group_list=groups,
            split_item=3, group_type=0, group_list_type=0)[0]
        activation = torch_npu.npu_swiglu(up, dim=-1)
        down = torch_npu.npu_grouped_matmul(
            x=[activation], weight=[weight2], bias=[], group_list=groups,
            split_item=3, group_type=0, group_list_type=0)[0]
        dispatch_output = torch_npu.npu_moe_token_unpermute(down, local_mapping)
    else:
        dispatch_output = dispatch
    combined = hidden.new_empty(routed.shape)
    dist.all_to_all_single(combined, dispatch_output, send, recv)
    return torch_npu.npu_moe_token_unpermute(combined, mapping, probs=probabilities)


def test_preparation_window():
    """Capture count-AllGather device end to Dispatch/core device start per call."""
    device_id = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(device_id)
    torch.set_num_threads(1)
    dist.init_process_group('hccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('npu', device_id)
    root = Path(os.environ['HMC_PREP_OUTPUT'])
    root.mkdir(parents=True, exist_ok=True)
    n, h, intermediate = (int(os.getenv(key, value)) for key, value in (
        ('HMC_PREP_TOKENS', '4096'), ('HMC_PREP_HIDDEN', '5120'), ('HMC_PREP_INTERMEDIATE', '1792')))
    e, k = int(os.getenv('HMC_PREP_EXPERTS', str(world * 6))), int(os.getenv('HMC_PREP_TOPK', '8'))
    warmup, steps = int(os.getenv('HMC_PREP_WARMUP', '8')), int(os.getenv('HMC_PREP_STEPS', '10'))
    assert e % world == 0 and 0 < k <= e and warmup >= 4 and steps > 0
    mode = os.getenv('HP_MEGAMOE_DISPATCH', 'pull')
    mega = MegaMoeExperts(hidden_size=h, intermediate_size=intermediate, num_experts=e,
                          top_k=k, ep_group=dist.group.WORLD, precision='bf16',
                          tuning_options=MegaMoeTuningOptions(dispatch_mode=mode))
    generator = torch.Generator().manual_seed(731 + rank)
    hidden = (torch.randn(n, h, generator=generator) * 0.1).to(device=device, dtype=torch.bfloat16)
    weights = ((torch.randn(e // world, h, 2 * intermediate, generator=generator) * 0.01).to(
        device=device, dtype=torch.bfloat16),
        (torch.randn(e // world, intermediate, h, generator=generator) * 0.01).to(
            device=device, dtype=torch.bfloat16))
    probabilities = torch.rand(n, k, generator=generator)
    probabilities = (probabilities / probabilities.sum(1, keepdim=True)).to(device)
    results = {}
    try:
        for routing in ('fixed', 'random', 'empty'):
            fixtures = []
            for index in range(4):
                if routing == 'random':
                    ids = torch.rand(n, e, generator=generator).topk(k, dim=1).indices
                else:
                    limit = e if routing == 'fixed' else e - e // world
                    assert k <= limit
                    ids = (torch.arange(n * k) + rank * k).remainder(limit).reshape(n, k)
                fixtures.append(ids.to(device=device, dtype=torch.int32))
            for variant in ('native', 'mega'):
                def invoke(index):
                    ids = fixtures[index % len(fixtures)]
                    if variant == 'native':
                        return native_forward(hidden, ids, probabilities, *weights, e)
                    return mega(hidden, ids, probabilities, expert_weights=weights)

                with torch.no_grad():
                    for ids in fixtures if routing == 'random' else fixtures[:1]:
                        reference = native_forward(hidden, ids, probabilities, *weights, e)
                        actual = mega(hidden, ids, probabilities, expert_weights=weights)
                        error = None
                        try:
                            torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)
                        except AssertionError as exc:
                            error = str(exc)
                        errors = [None] * world
                        dist.all_gather_object(errors, error)
                        assert not any(errors), f'Numerical parity failed by rank: {errors}'
                        del reference, actual
                    for index in range(warmup):
                        invoke(index)
                    torch.npu.synchronize()
                    dist.barrier()
                    torch.npu.synchronize()
                    unprofiled = []
                    torch.npu.reset_peak_memory_stats()
                    for index in range(steps):
                        start = time.perf_counter_ns()
                        output = invoke(index)
                        torch.npu.synchronize()
                        unprofiled.append((time.perf_counter_ns() - start) / 1e6)
                        del output
                    peaks = dict(allocated=torch.npu.max_memory_allocated(),
                                 reserved=torch.npu.max_memory_reserved())
                    trace = root / f'{routing}-{variant}-rank{rank}.json'
                    config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level0)
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                    torch_npu.profiler.ProfilerActivity.NPU],
                        record_shapes=True, profile_memory=False, with_stack=False,
                        experimental_config=config,
                    ) as profiler:
                        for index in range(steps):
                            dist.barrier()
                            torch.npu.synchronize()
                            with torch.autograd.profiler.record_function(
                                    f'hmc_prelude/{variant}/{routing}/{index}'):
                                output = invoke(index)
                                torch.npu.synchronize()
                                del output
                    profiler.export_chrome_trace(str(trace))
                    windows = preparation_windows(json.loads(trace.read_text()), expected_steps=steps)
                    results[f'{routing}-{variant}'] = dict(
                        scope='count AllGather device end -> Dispatch AllToAllV/MegaKernel device start',
                        windows=windows, preparation_p50_ms=statistics.median(x['prepare_ms'] for x in windows),
                        unprofiled_forward_p50_ms=statistics.median(unprofiled),
                        unprofiled_forward_samples_ms=unprofiled, peak_memory=peaks,
                        caveat='Standalone BF16 native-style control; EP4 is not EP64 communication',
                        numerical_parity=True,
                        route_sha256=[hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                                      for ids in fixtures])
        (root / f'summary-rank{rank}.json').write_text(json.dumps(dict(
            shape=dict(n=n, h=h, intermediate=intermediate, experts=e, top_k=k, ep=world),
            rank=rank, dispatch_mode=mode, results=results,
            implementation_file=str(Path(sys.modules[MegaMoeExperts.__module__].__file__).resolve())),
            indent=2) + '\n')
    finally:
        mega.close()
        dist.destroy_process_group()
