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
"""Check progress of shared forward/backward worker queues and events."""

import itertools
import unittest

from hyper_multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_multicore.modules.mega_moe.backward.graph import (
    build_backward_graph, build_route_backward_graph,
)
from hyper_multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank,
)
from hyper_multicore.modules.mega_moe.forward.graph import (
    build_forward_graph, build_route_forward_graph,
)
from hyper_multicore.scheduler.config import (
    EVENT_INVALID_ID, TaskSplitValue, TaskType,
)


def _worker_queues(configs, task_num_includes_termination):
    """Validate and split serialized task queues into fixed worker lanes."""
    events = [[0] * cfg.event_capacity for cfg in configs]
    workers = []
    for rank, cfg in enumerate(configs):
        if cfg.ready_event:
            events[rank][cfg.ready_event] = 1
        lanes = cfg.num_workers // 2
        for index, task_id in enumerate(cfg.cube_task_indices[:cfg.task_index_num[0]]):
            assert cfg.all_tasks[task_id].task_index % lanes == index % lanes, (
                "changed GMM core assignment"
            )
        queued = []
        for queue, count in ((cfg.cube_task_indices, cfg.task_index_num[0]),
                             (cfg.vector_task_indices, cfg.task_index_num[1])):
            task_ids = list(queue[:count])
            queued.extend(task_ids)
            workers.extend([rank, task_ids[lane::lanes], 0] for lane in range(lanes))
        count = cfg.task_num + int(not task_num_includes_termination)
        assert sorted(queued) == list(range(count)), "lost or duplicated task"
    return events, workers


def _dependencies_ready(task, cfg, events):
    """Include the extra branch join consumed inside weighted unpermute."""
    dependencies = [task.dependent_event]
    if task.task_type == TaskType.TASK_WEIGHTED_UNPERMUTE and task.inputs[3].dim[3]:
        dependencies.append(task.inputs[3].dim[3])
    return all(event == EVENT_INVALID_ID or events[event] >= cfg.all_event_num_triggers[event]
               for event in dependencies)


def _signal_task(task, events, rank):
    """Model remote PUT completion and the local indexed-dispatch reader join."""
    if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
        peer = task.task_index // (task.task_split_num // len(events))
        events[peer][task.trigger_event] += 1
    else:
        events[rank][task.trigger_event] += 1
    if task.task_type in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM) and task.extra_value_2:
        events[rank][task.extra_value_2] += 1


def _assert_worker_progress(configs, *, task_num_includes_termination=False):
    """Interpret event signaling and fixed per-worker queues across all ranks."""
    events, workers = _worker_queues(configs, task_num_includes_termination)
    while True:
        pending = progress = 0
        for worker in workers:
            rank, task_ids, index = worker
            if index == len(task_ids):
                continue
            pending += 1
            cfg = configs[rank]
            task = cfg.all_tasks[task_ids[index]]
            if not _dependencies_ready(task, cfg, events[rank]):
                continue
            _signal_task(task, events, rank)
            worker[2] += 1
            progress += 1
        if not pending:
            return
        assert progress, "worker queues contain an event dependency cycle"


class TestSharedForwardSchedule(unittest.TestCase):
    """Exercise event joins with partial worker rounds and multiple EP ranks."""

    def test_worker_progress_with_unequal_and_empty_source_ranks(self):
        """Per-rank shared task counts must not break remote completion joins."""
        lengths = (0, 1, 257, 512)
        for cores, builder in itertools.product((1, 20, 24), (build_forward_graph, build_route_forward_graph)):
            with self.subTest(cores=cores, builder=builder.__name__):
                configs = []
                for rank, tokens in enumerate(lengths):
                    values = TaskSplitValue(tp=1, ep=4, seq_size=512, all_expert_num=8, top_k=2)
                    values.local_num_tokens = tokens
                    graph = builder(values, hidden_size=5120, intermediate_size=128,
                                    num_cube_cores=cores, integrated_unpermute=True,
                                    shared_intermediate_size=256, fuse_shared_expert=True)
                    graph.propagate_splits(values)
                    configs.append(build_config_for_rank(graph, values, rank, cores))
                _assert_worker_progress(configs)

    def test_backward_progress_with_unequal_and_empty_source_ranks(self):
        """Ragged source endpoints join routed and shared gradients in both transports."""
        lengths = (0, 1, 257, 512)
        for cores, transport, mode, shared in itertools.product(
                (1, 20, 24), ("push", "pull"), ("legacy", "permute", "unpermute", "both"), (False, True)):
            with self.subTest(cores=cores, transport=transport, mode=mode, shared=shared):
                configs = []
                for rank, tokens in enumerate(lengths):
                    values = TaskSplitValue(tp=1, ep=4, seq_size=512, all_expert_num=8,
                                            top_k=2, dispatch_mode=transport)
                    values.local_num_tokens = tokens
                    builder = build_backward_graph if mode == "legacy" else build_route_backward_graph
                    endpoints = {} if mode == "legacy" else {
                        "integrated_permute": mode in ("permute", "both"),
                        "integrated_unpermute": mode in ("unpermute", "both"),
                    }
                    graph = builder(values, hidden_size=5120, intermediate_size=128,
                                    num_cube_cores=cores, shared_intermediate_size=256,
                                    fuse_shared_expert=shared, **endpoints)
                    graph.propagate_splits(values)
                    configs.append(build_backward_config(graph, values, rank, cores))
                _assert_worker_progress(configs, task_num_includes_termination=True)

    def test_worker_progress_and_completion_join(self):
        """Every scheduled task runs and reduction waits on both branches."""
        shapes = ((128, 4, 2, 2), (4096, 32, 4, 8), (128, 32, 4, 8),
                  (512, 4, 2, 2), (128, 4, 1, 2), (384, 6, 2, 2))
        for shape, cores, builder in itertools.product(
                shapes, (1, 20, 24), (build_forward_graph, build_route_forward_graph)):
            tokens, experts, ep, top_k = shape
            with self.subTest(shape=shape, cores=cores, builder=builder.__name__):
                values = TaskSplitValue(tp=1, ep=ep, seq_size=tokens,
                                       all_expert_num=experts, top_k=top_k)
                graph = builder(values, hidden_size=5120, intermediate_size=1792,
                                num_cube_cores=cores, integrated_unpermute=True,
                                shared_intermediate_size=1792, fuse_shared_expert=True)
                graph.propagate_splits(values)
                configs = [build_config_for_rank(graph, values, rank, cores) for rank in range(ep)]
                join = graph.get_op("weighted_unpermute").fill_config.dependent_event
                self.assertEqual(graph.get_op("shared_up_proj").fill_config.dependency,
                                 EVENT_INVALID_ID)
                self.assertEqual(graph.get_op("shared_down_proj").fill_config.trigger, join)
                for cfg in configs:
                    self.assertEqual(cfg.all_event_num_triggers[join],
                                     graph.get_op("combine").task_num + cores)
                _assert_worker_progress(configs)
