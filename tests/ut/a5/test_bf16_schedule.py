"""Qualify complete BF16 work ownership and reduced-QP scheduling on the host."""
from collections import Counter
from dataclasses import replace

import pytest

from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import TaskType


def specification(ep: int, experts: int, schedule: str, rank: int = 0) -> MegaMoeSpec:
    return MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                       num_experts=ep * experts, top_k=8, initial_capacity_factor=1.25,
                       receive_capacity=40960, ep_size=ep, ep_group=None, rank_id=rank,
                       num_cube_cores=32, dispatch_split=512, combine_split=512,
                       bf16_schedule=schedule)


def queues(cfg):
    assert cfg.task_index_num[3] == -2
    offsets = cfg.mix_task_indices
    cubes = [list(cfg.cube_task_indices[offsets[65 + w]:offsets[66 + w]]) for w in range(32)]
    vectors = [list(cfg.vector_task_indices[offsets[w]:offsets[w + 1]]) for w in range(64)]
    return cubes, vectors


@pytest.mark.parametrize('ep,experts', [(4, 6), (8, 6), (64, 6), (4, 5)])
@pytest.mark.parametrize('schedule', ['window2', 'half16'])
def test_bf16_complete_unique_work_and_qp1_owner(monkeypatch, ep, experts, schedule):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    monkeypatch.setenv('HYPER_MULTICORE_BF16_GMM_POLICY', 'blaze_bwd_lanes_a5')
    spec = specification(ep, experts, schedule)
    forward, fcfg, backward, bcfg = _build_runtime_artifacts(spec)
    for graph, cfg, is_forward in ((forward, fcfg, True), (backward, bcfg, False)):
        cubes, vectors = queues(cfg)
        work = [tid for q in cubes + vectors for tid in q]
        assert len(work) == len(set(work)) == cfg.task_num
        assert set(work) == set(range(cfg.task_num))
        owners = {}
        for worker, queue in enumerate(vectors):
            for tid in queue:
                task = cfg.all_tasks[tid]
                if task.task_type in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
                    assert worker % 2
                    peer = task.task_index // (task.task_split_num // ep)
                    if peer != spec.rank_id:
                        assert owners.setdefault(peer, worker) == worker
                elif task.task_type != TaskType.TASK_TERMINATE:
                    assert worker % 2 == 0
        expected_lanes = 16 if schedule == 'half16' and is_forward else 32
        per_expert = Counter()
        for worker, queue in enumerate(cubes):
            for tid in queue:
                task = cfg.all_tasks[tid]
                assert task.task_type == TaskType.TASK_GROUPED_MATMUL
                expert, lane = divmod(task.task_index, expected_lanes)
                assert task.extra_value_0 == expected_lanes
                assert cfg.all_event_num_triggers[task.trigger_event] >= expected_lanes
                if expected_lanes == 16:
                    assert worker // 16 == expert % 2
                    assert worker % 16 == lane
                else:
                    assert worker == lane
                per_expert[expert] += 1
        assert all(count == expected_lanes * (2 if is_forward else 4) for count in per_expert.values())
        assert len(per_expert) == experts


def test_half16_forward_order_and_unchanged_backward(monkeypatch):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    monkeypatch.setenv('HYPER_MULTICORE_BF16_GMM_POLICY', 'blaze_bwd_lanes_a5')
    spec = specification(4, 6, 'half16')
    _, fcfg, _, bcfg = _build_runtime_artifacts(spec)
    cubes, _ = queues(fcfg)
    for worker, queue in enumerate(cubes):
        assert [fcfg.all_tasks[tid].task_index // 16 for tid in queue] == [
            expert for expert in range(worker // 16, 6, 2) for _ in range(2)]
    _, _, _, control = _build_runtime_artifacts(replace(spec, bf16_schedule='window2'))
    assert bytes(bcfg) == bytes(control)


@pytest.mark.parametrize('policy', ['window2', 'half16'])
@pytest.mark.parametrize('mode', ['push', 'pull'])
@pytest.mark.parametrize('depth', [0, 8])
def test_multi_rank_queues_can_complete_without_dependency_cycle(monkeypatch, policy, mode, depth):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    monkeypatch.setenv('HYPER_MULTICORE_BF16_GMM_POLICY', 'blaze_bwd_lanes_a5')
    configs = []
    for rank in range(4):
        spec = replace(specification(4, 6, policy, rank), dispatch_mode=mode, ready_depth=depth)
        _, fcfg, _, bcfg = _build_runtime_artifacts(spec)
        configs.append((fcfg, bcfg))
    for phase in (0, 1):
        counters = [Counter() for _ in configs]
        workers = []
        expected = 0
        for rank, pair in enumerate(configs):
            cfg = pair[phase]
            cubes, vectors = queues(cfg)
            for queue in cubes + vectors:
                if queue:
                    lookahead = 8 if cfg.all_tasks[queue[0]].extra_value_1 & 32 else 1
                    workers.append([rank, list(queue), lookahead])
                    expected += len(queue)
        completed = 0
        while completed < expected:
            progress = False
            for rank, queue, depth in workers:
                cfg = configs[rank][phase]
                for index, tid in enumerate(queue[:depth]):
                    task = cfg.all_tasks[tid]
                    dep = task.dependent_event
                    if dep != 0xffffffff and counters[rank][dep] < cfg.all_event_num_triggers[dep]:
                        continue
                    target = rank
                    if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                        target = task.task_index // (task.task_split_num // 4)
                    counters[target][task.trigger_event] += 1
                    queue.pop(index)
                    completed += 1
                    progress = True
                    break
            assert progress, (phase, [(rank, q[:2],
                configs[rank][phase].all_tasks[q[0]].dependent_event,
                counters[rank][configs[rank][phase].all_tasks[q[0]].dependent_event],
                configs[rank][phase].all_event_num_triggers[configs[rank][phase].all_tasks[q[0]].dependent_event])
                for rank, q, _ in workers if q][:4])
        assert completed == expected


def test_ep64_ready_depth_preserves_long_vector_queues(monkeypatch):
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    monkeypatch.setenv('HYPER_MULTICORE_BF16_GMM_POLICY', 'blaze_bwd_lanes_a5')
    for policy in ('window2','half16'):
        spec = replace(specification(64,6,policy),ready_depth=8)
        _,fcfg,_,bcfg = _build_runtime_artifacts(spec)
        for cfg in (fcfg,bcfg):
            _,vectors=queues(cfg)
            assert max(map(len,vectors[::2])) > 64
            assert all(cfg.all_tasks[q[0]].extra_value_1 & 32 for q in vectors[::2] if q)


def test_serialized_tag_is_read_by_production_cpp(monkeypatch,tmp_path):
    import subprocess
    from pathlib import Path
    from hyper_multicore.scheduler.runtime import serialize_runtime_config
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM','1')
    monkeypatch.setenv('HYPER_MULTICORE_BF16_GMM_POLICY','blaze_bwd_lanes_a5')
    root=Path(__file__).resolve().parents[3]
    cpp=tmp_path/'read.cpp'
    cpp.write_text(r'''#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>
#define __aicore__
#define __gm__
namespace AscendC {
enum class HardEvent {};
struct Pipe { uint32_t FetchEventID(HardEvent) { return0; } };
inline Pipe* GetTPipePtr(){return nullptr;}
template<HardEvent E> inline void SetFlag(uint32_t){}
template<HardEvent E> inline void WaitFlag(uint32_t){}
}
using AscendC::GetTPipePtr;
#include "hyper_multicore/ops/runtime/runtime_config.hpp"
int main(int argc,char**argv){
 std::ifstream f(argv[1],std::ios::binary);
 std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)),{});
 using namespace MulticoreRuntime;
 if(getTaskQueueLayoutTag(b.data())!=-2)return1;
 if(getTaskIndexNumByTaskType(b.data(),TaskAiCoreType::TASK_AICORE_MIX)!=0)return2;
 auto offsets=reinterpret_cast<int32_t*>(b.data()+getMixTaskIndexsOffset(b.data()));
 return offsets[65+32]==getTaskIndexNumByTaskType(b.data(),TaskAiCoreType::TASK_AICORE_CUBE)?0:3;
}
'''.replace('return0','return 0').replace('return1','return 1').replace('return2','return 2'))
    subprocess.run(['g++','-std=c++17','-I',str(root),str(cpp),'-o',str(tmp_path/'read')],check=True)
    for policy in ('window2','half16'):
        _,fcfg,_,bcfg=_build_runtime_artifacts(specification(4,6,policy))
        for cfg in (fcfg,bcfg):
            data=tmp_path/'runtime.bin';data.write_bytes(serialize_runtime_config(cfg))
            subprocess.run([str(tmp_path/'read'),str(data)],check=True)
