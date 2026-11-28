# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Real-device bounded metadata and equal-input BF16/MXFP8 training checks."""
import os
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import

from hyper_multicore import MegaMoeExperts
from hyper_multicore.experimental import MegaMoeTuningOptions
from hyper_multicore._loader import query_npu_arch
from hyper_multicore._device import device_ub_bytes
from hyper_multicore.modules.mega_moe import route
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec, _device_route_capacity
from hyper_multicore.torch import ops
from tests.common.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence


def test_shared_metadata_entries():
    """Compare exact, summary and bounded entries with the same count oracle."""
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('npu', torch.npu.current_device())
    checked = []
    for ep, experts, tokens, topk in ((1, 3, 17, 2), (2, 2, 17, 2), (2, 32, 17, 2),
                                      (3, 3, 17, 2), (2, 6, 129, 2), (3, 21, 127, 3),
                                      (2, 12, 4096, 8), (4, 24, 4096, 8),
                                      (8, 48, 257, 8), (9, 54, 17, 3), (17, 17, 17, 3),
                                      (64, 384, 4096, 8)):
        for unequal in (False, True):
            lengths = [tokens + (peer % 3 if unequal else 0) for peer in range(ep)]
            counts = torch.stack([torch.bincount((torch.arange(n * topk) + peer * 13) % experts,
                                                minlength=experts).int()
                                  for peer, n in enumerate(lengths)])
            headers = torch.tensor([[0, n, 15] for n in lengths], dtype=torch.int32)
            # Exercise the actual row stride rather than assuming a packed table.
            padded = torch.full((ep, experts + 12), -99, dtype=torch.int32, device=device)
            padded[:, :experts + 3] = torch.cat((headers, counts), 1).to(device)
            payload = padded[:, :experts + 3]
            local = experts // ep
            received = counts.long().sum(0).reshape(ep, local)
            capacity = max(128, ((int(received.sum(1).max()) + 127) // 128) * 128)
            summary_expected = torch.tensor([
                [0, lengths[i], 15, int(counts[i].sum()), int(counts[i].min()), int(counts[i].max()),
                 int(received[i].sum()), int(bool((received[i] > 0).all()))] for i in range(ep)
            ], dtype=torch.int64)
            for hidden in (1, 5120):
                for rank in sorted({0, ep // 2, ep - 1}):
                    spec = MegaMoeSpec(local_num_tokens=lengths[rank], hidden_size=hidden,
                                       intermediate_size=1792, num_experts=experts, top_k=topk,
                                       initial_capacity_factor=None, receive_capacity=0,
                                       ep_size=ep, ep_group=None, rank_id=rank, num_cube_cores=24,
                                       auto_tokens=True, dispatch_mode='pull')
                    expected = route._compute_route_metadata(counts[rank], counts, None, spec, capacity)
                    offsets, sizes = ops.route_offsets(payload[:, 3:], rank, device_ub_bytes(device.index), hidden)
                    exact = route._unpack_route_metadata(offsets, sizes, spec, capacity)
                    offsets, sizes = ops.route_offsets(counts.to(device), rank, device_ub_bytes(device.index), hidden)
                    compact = route._unpack_route_metadata(offsets, sizes, spec, capacity)
                    offsets, sizes, summary = ops.prepare_route(payload, rank, device_ub_bytes(device.index), hidden)
                    prepared = route._unpack_route_metadata(offsets, sizes, spec, capacity)
                    torch.testing.assert_close(summary.cpu(), summary_expected, rtol=0, atol=0)
                    actual = [exact, compact, prepared]
                    if not unequal:
                        views = ops.bounded_route(payload, rank, device_ub_bytes(device.index), hidden,
                                                  capacity, tokens, topk)
                        elements = None if hidden == 1 else (hidden, route.RouteMetadata(*views[7:], capacity))
                        actual.append(route.RouteMetadata(*views[:7], capacity, _elements=elements))
                    for value in actual:
                        for unit in sorted({1, hidden}):
                            for name in ('dispatch_src_off', 'dispatch_target_off', 'dispatch_size',
                                         'combine_src_off', 'combine_target_off', 'combine_size', 'group_list'):
                                torch.testing.assert_close(getattr(value.for_kernel(unit), name).cpu(),
                                                           getattr(expected.for_kernel(unit), name), rtol=0, atol=0)
                    checked.append(dict(ep=ep, hidden=hidden, unequal=unequal, rank=rank))
    output = Path(os.environ['HP_MEGA_MOE_EVIDENCE_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'shared-metadata.json').write_text(json.dumps(checked, indent=2))


def test_bounded_metadata():
    """Exercise actual count tables up to EP64 without needing 64 physical cards."""
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('npu', torch.npu.current_device())
    checked = []
    for ep, experts, tokens, topk in ((2, 4, 257, 2), (8, 48, 257, 8), (64, 384, 4096, 8)):
        for changed in (False, True):
            counts = []
            for peer in range(ep):
                ids = (torch.arange(tokens * topk) + (peer * 13 if changed else 0)) % experts
                counts.append(torch.bincount(ids, minlength=experts).int())
            counts = torch.stack(counts)
            payload = torch.cat((torch.tensor([[0, tokens, 15]] * ep).int(), counts), 1).to(device)
            for rank in (0, ep - 1):
                spec = MegaMoeSpec(local_num_tokens=tokens, hidden_size=5120, intermediate_size=1792,
                                   num_experts=experts, top_k=topk, initial_capacity_factor=None,
                                   receive_capacity=0, ep_size=ep, ep_group=None, rank_id=rank,
                                   num_cube_cores=24, auto_tokens=True, dispatch_mode='pull',
                                   device_route_capacity_factor=1.05)
                capacity = _device_route_capacity(spec, tokens)
                views = ops.bounded_route(payload, rank, device_ub_bytes(device.index), 5120,
                                          capacity, tokens, topk)
                elements = route.RouteMetadata(*views[7:], capacity)
                actual = route.RouteMetadata(*views[:7], capacity, _elements=(5120, elements))
                expected = route._compute_route_metadata(counts[rank], counts, None, spec, capacity)
                for unit in (1, 5120):
                    for name in ('dispatch_src_off', 'dispatch_target_off', 'dispatch_size',
                                 'combine_src_off', 'combine_target_off', 'combine_size', 'group_list'):
                        torch.testing.assert_close(getattr(actual.for_kernel(unit), name).cpu(),
                                                   getattr(expected.for_kernel(unit), name), rtol=0, atol=0)
                checked.append(dict(ep=ep, changed=changed, rank=rank, capacity=capacity))
    output = Path(os.environ['HP_MEGA_MOE_EVIDENCE_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'bounded-metadata.json').write_text(json.dumps({'bounded_metadata': checked}, indent=2))


def test_bounded_training():
    """Retain actual per-call counts and autograd outputs with no NPU-to-CPU reads."""
    from tests.torch import _test_mega_moe_dynamic_tokens as reference
    from tests.common.multicore.owned_mega_moe import OwnedMegaMoeExperts
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(1)
    dist.init_process_group('hccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('npu', torch.npu.current_device())
    start_shmem_lifetime()
    experts, topk = 6 * world, 2
    weights = reference._weights(experts)
    mega = OwnedMegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=experts,
                               top_k=topk, dispatch_mode='pull', ep_group=dist.group.WORLD).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        mega.gate_up_weight.copy_(weights[0][rank*6:(rank+1)*6])
        mega.down_weight.copy_(weights[1][rank*6:(rank+1)*6])
    evidence = []
    try:
        # Full, changed, cold-expert and empty local expert work; fixed input
        # shape per group, including retained ctx across changing input sizes.
        for tokens, distribution in ((128, 'balanced'), (257, 'balanced'), (257, 'cold'),
                                     (257, 'concentrated'), (0, 'balanced')):
            inputs = reference._inputs([tokens] * world, experts, topk)
            if distribution == 'cold':
                # Each destination receives its usual total but only two of its
                # six local experts are active. This is not a cached fix route.
                indices = [((x // 6) * 6 + x % 2).int() for x in inputs[1]]
                inputs = (inputs[0], indices, *inputs[2:])
            elif distribution == 'concentrated':
                # Distinct expert IDs for each token, one active expert per
                # destination. Rank totals still fit, but a single expert's M
                # exceeds local T: cached plans must cover this legal skew.
                selected = torch.arange(topk, dtype=torch.int32) * 6
                indices = [selected.repeat(tokens, 1) for _ in range(world)]
                inputs = (inputs[0], indices, *inputs[2:])
            expected = reference._rounded_oracle(weights, inputs)
            mega.zero_grad(set_to_none=True)
            # First use and source growth use the framework's cold-path
            # configuration collectives. Exclude these from steady-state D2H.
            warm = reference._forward(mega, inputs, rank, device)
            (warm[2] * warm[3]).sum().backward()
            mega.zero_grad(set_to_none=True)
            # Any regression into the exact-count Host path must fail this test.
            original_cpu, original_tolist = torch.Tensor.cpu, torch.Tensor.tolist
            def cpu(value, *args, **kwargs):
                assert value.device.type != 'npu', 'unexpected NPU D2H during MegaMoE'
                return original_cpu(value, *args, **kwargs)
            def tolist(value, *args, **kwargs):
                assert value.device.type != 'npu', 'unexpected count D2H during MegaMoE'
                return original_tolist(value, *args, **kwargs)
            with (patch.object(torch.Tensor, 'cpu', cpu), patch.object(torch.Tensor, 'tolist', tolist),
                  patch.object(route._PendingDeviceRoute, 'finish', side_effect=AssertionError('count D2H'))):
                invocation = reference._forward(mega, inputs, rank, device)
                x, probs, output, dy = invocation
                (output * dy).sum().backward()
            begin, end = rank * tokens, (rank + 1) * tokens
            actual = (output, x.grad, probs.grad, mega.gate_up_weight.grad, mega.down_weight.grad)
            expected = (expected[0][begin:end], expected[1][begin:end], expected[2][begin:end],
                        expected[3][rank*6:(rank+1)*6], expected[4][rank*6:(rank+1)*6])
            errors = [reference._assert_close(str(index), found, want)
                      for index, (found, want) in enumerate(zip(actual, expected))]
            evidence.append(dict(tokens=tokens, distribution=distribution, max_errors=errors))
        # New early-prepared buffers must survive several later source writes.
        batches = [reference._inputs([n] * world, experts, topk) for n in (128, 257, 128)]
        expected_batches = [reference._rounded_oracle(weights, batch) for batch in batches]
        pending = [reference._forward(mega, batch, rank, device) for batch in batches]
        for index in (2, 0, 1):
            mega.zero_grad(set_to_none=True)
            errors = reference._check_gradients(mega, pending[index], expected_batches[index],
                                                [pending[index][0].shape[0]] * world, rank)
            evidence.append(dict(pending_invocation=index, errors=errors))
    finally:
        mega.close()
    write_evidence({'device_route_training': evidence})
    dist.destroy_process_group()


def test_bounded_rejection():
    """Run each fatal Device assertion in its own disposable process."""
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('npu', torch.npu.current_device())
    kind = os.environ['HMC_BOUNDED_REJECTION']
    payload = torch.tensor([[0, 256, 15, 128, 128, 128, 128]] * 2, dtype=torch.int32, device=device)
    capacity = 128 if kind == 'capacity' else 640
    topk = 2
    if kind in ('remote_source', 'remote_capacity'):
        counts = torch.full((64, 384), 5, dtype=torch.int32, device=device)
        counts[:, :128] = 6
        capacity, topk = 4096, 8
        if kind == 'remote_capacity':
            counts.zero_()
            counts[:, -8:] = 256
            capacity = 65536
        payload = torch.cat((torch.tensor([[0, 256, 15]] * 64, dtype=torch.int32, device=device), counts), 1)
        if kind == 'remote_source': payload[-1, 1] = 255
    if kind == 'negative': payload[0, 3] = -1
    if kind == 'tokens': payload[1, 1] = 255
    if kind == 'weights': payload[1, 2] = route._INVALID_WEIGHT_MASK
    if kind == 'participation': payload[1, 2] = 7
    try:
        ops.bounded_route(payload, 0, device_ub_bytes(device.index), 512, capacity, 256, topk)
        torch.npu.synchronize()
    except RuntimeError as error:
        # A trapping context cannot be reused. The launcher owns process and
        # device cleanup; leave evidence before skipping Python finalizers.
        Path(os.environ['HMC_REJECTION_RESULT']).write_text(json.dumps(
            {'kind': kind, 'rejected': True, 'error': str(error)}))
        os._exit(0)
    raise AssertionError(f'Device accepted invalid bounded metadata: {kind}')


def test_mxfp8_bounded_training():
    """Compare exact and budget F/R/B with changing counts and fresh weight _axes."""
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    if query_npu_arch(torch.npu.current_device()) != 3510:
        pytest.skip('MXFP8 requires arch35')
    torch.set_num_threads(1)
    dist.init_process_group('hccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('npu', torch.npu.current_device())
    start_shmem_lifetime()
    h, intermediate, local, topk = 512, 128, 6, 2
    experts = local * world
    generator = torch.Generator().manual_seed(2971 + rank)
    roots, physical = [], []
    for shape in ((local, h, 2 * intermediate), (local, intermediate, h)):
        weight = (torch.randn(shape, generator=generator) * 0.02).to(device=device, dtype=torch.bfloat16)
        roots.append(torch.nn.Parameter(weight))
        physical.append(torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            weight, dst_type=torch.float8_e4m3fn, scale_alg=1))
    def _axes(backward=False):
        for root, data in zip(roots, physical):
            # Resolve current physical roots on every launch, including backward
            # after subsequent forwards have replaced the Python attributes.
            root.row_data, root.row_scale = [t.clone() for t in data[:2]] if backward else (None, None)
            root.col_data, root.col_scale = [t.clone() for t in data[2:]]
    modules = []
    for flag in ('0', '1'):
        with patch.dict(os.environ, {'HYPER_MULTICORE_MEGAMOE_DEVICE_ROUTE': flag}):
            modules.append(MegaMoeExperts(hidden_size=h, intermediate_size=intermediate,
                num_experts=experts, top_k=topk, precision='mxfp8', ep_group=dist.group.WORLD,
                tuning_options=MegaMoeTuningOptions(dispatch_mode='pull')))
    def _invoke(module, fixture):
        x, probabilities, ids, dy = fixture
        x, probabilities = x.detach().clone().requires_grad_(), probabilities.detach().clone().requires_grad_()
        _axes()
        output = module(x, ids, probabilities, expert_weights=tuple(roots))
        return x, probabilities, output, dy
    def _gradients(invocation):
        x, probabilities, output, dy = invocation
        _axes(True)
        return (output.detach(), *torch.autograd.grad(output, (x, probabilities, *roots), grad_outputs=dy))
    def _compare(actual, expected):
        errors = []
        for found, want in zip(actual, expected):
            torch.testing.assert_close(found, want, rtol=0.03, atol=0.003)
            assert bool(found.isfinite().all().item())
            errors.append(float((found.float() - want.float()).abs().max().item()) if found.numel() else 0.)
        return errors
    fixtures = []
    for tokens, cold in ((128, False), (257, False), (257, True), (128, False), (0, False)):
        x = (torch.randn(tokens, h, generator=generator) * 0.1).to(device=device, dtype=torch.bfloat16)
        probabilities = torch.softmax(torch.randn(tokens, topk, generator=generator), -1).to(device)
        ids = ((torch.arange(tokens * topk).reshape(tokens, topk) + rank * 13) % experts).int()
        if cold:
            ids = (ids // local) * local + ids % topk
        fixtures.append((x, probabilities, ids.to(device), torch.full_like(x, 0.01)))
    evidence = []
    try:
        for fixture in fixtures:
            expected = _gradients(_invoke(modules[0], fixture))
            # Warm source growth and collectives separately from count-D2H audit.
            _gradients(_invoke(modules[1], fixture))
            original_cpu, original_tolist = torch.Tensor.cpu, torch.Tensor.tolist
            def _cpu(value, *args, **kwargs):
                assert value.device.type != 'npu', 'unexpected NPU D2H during MX experts'
                return original_cpu(value, *args, **kwargs)
            def _tolist(value, *args, **kwargs):
                assert value.device.type != 'npu', 'unexpected count D2H during MX experts'
                return original_tolist(value, *args, **kwargs)
            with (patch.object(torch.Tensor, 'cpu', _cpu), patch.object(torch.Tensor, 'tolist', _tolist),
                  patch.object(route._PendingDeviceRoute, 'finish', side_effect=AssertionError('count D2H'))):
                _axes()
                with torch.no_grad():
                    first = modules[1](fixture[0], fixture[2], fixture[1], expert_weights=tuple(roots))
                actual = _gradients(_invoke(modules[1], fixture))
            torch.testing.assert_close(first, actual[0], rtol=0, atol=0)
            evidence.append({'tokens': len(fixture[0]), 'errors': _compare(actual, expected)})
        expected = [_gradients(_invoke(modules[0], f)) for f in fixtures[:3]]
        pending = [_invoke(modules[1], f) for f in fixtures[:3]]
        for index in (2, 0, 1):
            evidence.append({'pending': index, 'errors': _compare(_gradients(pending[index]), expected[index])})
    finally:
        for module in reversed(modules):
            module.close()
    write_evidence({'mxfp8_device_route_training': evidence})
    dist.destroy_process_group()


def test_repeated_metadata_launches():
    """Stress descriptor lifetime and compact protocol strides without profiling."""
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('npu', torch.npu.current_device())
    for peers, experts in ((2, 12), (4, 24), (8, 48), (64, 384)):
        tokens, topk = 4096, 8
        counts = torch.stack([torch.bincount((torch.arange(tokens * topk) + rank * 13) % experts,
                                            minlength=experts).int() for rank in range(peers)])
        payload = torch.cat((torch.tensor([[0, tokens, 15]] * peers).int(), counts), 1).to(device)
        capacity = ((int(1.05 * tokens * topk) + 127) // 128) * 128
        for entry in ('offsets', 'summary', 'bounded'):
            for _ in range(128):
                if entry == 'offsets':
                    value = ops.route_offsets(payload[:, 3:], 0, device_ub_bytes(device.index), 5120)
                elif entry == 'summary':
                    value = ops.prepare_route(payload, 0, device_ub_bytes(device.index), 5120)
                else:
                    value = ops.bounded_route(payload, 0, device_ub_bytes(device.index), 5120,
                                              capacity, tokens, topk)
            torch.npu.synchronize()
            assert value[0].device == device
