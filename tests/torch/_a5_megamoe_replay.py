# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Component replay: F/R/B, empty experts, fresh roots, streams and optional trace.

Run under torchrun after activating the source-built payload. Normal correctness
replay is separate from performance measurement. No downstream framework needed.
"""
import argparse, json, hashlib, os
import time
from pathlib import Path
from datetime import timedelta
import torch, torch.distributed as dist, torch_npu
from hyper_multicore import MegaMoeExperts, MXFP8MegaMoeExperts

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--precision', choices=['bf16', 'mxfp8'], default='mxfp8')
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--intermediate', type=int, default=128)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['push', 'pull'], default='push')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--empty', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--save-tensors', action='store_true')
    p.add_argument('--benchmark', action='store_true')
    a = p.parse_args()
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local)
    dist.init_process_group('hccl', timeout=timedelta(seconds=120))
    world = dist.get_world_size()
    dev = torch.device('npu', local)
    torch.manual_seed(1729 + rank)
    torch.npu.manual_seed(1729 + rank)
    h, i, e, k = (a.hidden, a.intermediate, 6, 8)
    shapes = ((e, h, 2 * i), (e, i, h))
    physical = []
    roots = []
    for shape in shapes:
        w = torch.randn(shape, dtype=torch.bfloat16, device=dev) * 0.02
        quant = torch_npu.npu_dynamic_mx_quant_with_dual_axis(w, dst_type=torch.float8_e4m3fn, scale_alg=1) if a.precision == 'mxfp8' else None
        root = torch.nn.Parameter(w)
        roots.append(root)
        physical.append(quant)
    module = (MXFP8MegaMoeExperts if a.precision == 'mxfp8' else MegaMoeExperts)(local_num_tokens=a.tokens, hidden_size=h, intermediate_size=i, num_experts=e * world, top_k=k, ep_size=world, ep_group=dist.group.WORLD, create_parameters=False, dispatch_mode=a.mode, initial_capacity_factor=2 if a.mode == 'push' else None)

    def axes(backward=False, fresh=False):
        if a.precision == 'bf16':
            return
        for root, data in zip(roots, physical):
            data = tuple((t.clone() for t in data)) if fresh else data
            root.row_data, root.row_scale = (data[0], data[1]) if backward else (None, None)
            root.col_data, root.col_scale = data[2:]

    def inputs(index):
        x = (torch.randn(a.tokens, h, dtype=torch.bfloat16, device=dev) * 0.1).requires_grad_()
        probs = torch.softmax(torch.randn(a.tokens, k, device=dev), -1).requires_grad_()
        ids = (torch.arange(a.tokens * k, device=dev).reshape(-1, k) + index) % (k if a.empty else e * world)
        return (x, probs, ids, torch.randn_like(x) * 0.1)
    records = []

    def sig(t):
        data = t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return dict(shape=list(t.shape), sha256=hashlib.sha256(data).hexdigest())
    prof = None
    if a.profile:
        from hyper_multicore.diagnostics.profiler.api import TorchMegaKernelProfiler
        from hyper_multicore.diagnostics.common.schedule import DiagnosticAction
        prof = TorchMegaKernelProfiler(schedule=lambda _: DiagnosticAction.RECORD_AND_SAVE, on_trace_ready=lambda _: None, detailed_task_names=True, max_pending_calls=3)
        prof.__enter__()
    for micro in range(1 if a.profile else 2):
        x, probs, ids, dy = inputs(micro)
        axes()
        with torch.no_grad():
            first = module(x, ids, probs, expert_weights=tuple(roots))
        axes()
        out = module(x, ids, probs, expert_weights=tuple(roots))
        torch.testing.assert_close(first, out, rtol=0, atol=0)
        axes(True, True)
        grads = torch.autograd.grad(out, (x, probs, *roots), grad_outputs=dy)
        torch.npu.synchronize()
        if not all((bool(t.float().isfinite().all().item()) for t in (out, *grads))):
            raise RuntimeError('nonfinite output/gradient')
        if a.empty:
            inactive = torch.arange(e, device=dev) + rank * e >= k
            for grad in grads[2:]:
                if torch.count_nonzero(grad.reshape(e, -1)[inactive]).item():
                    raise RuntimeError('empty expert gradient')
        records.append(dict(micro=micro, input=sig(x), results=[sig(t) for t in (out, *grads)]))
        a.output.mkdir(parents=True, exist_ok=True)
        if a.save_tensors:
            torch.save([t.detach().cpu() for t in (out, *grads)], a.output / f'rank{rank:05d}-micro{micro}.pt')
    if prof:
        prof.__exit__(None, None, None)
        prof.export_chrome_trace(a.output / f'rank{rank:05d}-trace.json')
    else:
        streams = [torch.npu.Stream(), torch.npu.Stream()]
        contexts = []
        for index in range(2):
            x, probs, ids, dy = inputs(index + 2)
            axes()
            streams[index].wait_stream(torch.npu.current_stream())
            with torch.npu.stream(streams[index]):
                out = module(x, ids, probs, expert_weights=tuple(roots))
            contexts.append((x, probs, out, dy))
        for index in (1, 0):
            x, probs, out, dy = contexts[index]
            axes(True, True)
            streams[1 - index].wait_stream(torch.npu.current_stream())
            with torch.npu.stream(streams[1 - index]):
                grads = torch.autograd.grad(out, (x, probs, *roots), grad_outputs=dy)
            torch.npu.synchronize()
            if not all((bool(t.float().isfinite().all().item()) for t in grads)):
                raise RuntimeError('old-context stream gradient')
    timings = []
    if a.benchmark:
        if a.profile:
            raise ValueError('Benchmark and profiling must be separate launches')
        axes(True)
        x, probs, ids, dy = inputs(0)
        torch.npu.synchronize()
        for iteration in range(10):
            start, end = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            wall = time.perf_counter()
            start.record()
            out = module(x, ids, probs, expert_weights=tuple(roots))
            grads = torch.autograd.grad(out, (x, probs, *roots), grad_outputs=dy)
            end.record()
            end.synchronize()
            if iteration >= 3:
                timings.append(dict(event_ms=start.elapsed_time(end), wall_ms=(time.perf_counter() - wall) * 1000))
    module.close()
    dist.barrier()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / f'rank{rank:05d}.json').write_text(json.dumps(dict(
        passed=True, rank=rank, world=world, precision=a.precision, mode=a.mode, empty=a.empty,
        shape=dict(tokens=a.tokens, hidden=h, intermediate=i, experts_per_rank=e, top_k=k),
        records=records, timings=timings), indent=2))
    print(json.dumps(dict(rank=rank, passed=True)), flush=True)
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
