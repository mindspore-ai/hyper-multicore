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
"""Exercise the real Python/C++ queue ABI, coverage and uneven per-worker schedules."""
import ctypes
import struct

import pytest

from hyper_multicore.scheduler.runtime import allocate_runtime_config, serialize_runtime_config, copy_runtime_config
from hyper_multicore.scheduler.worker_queues import install_worker_queues
from tests.ut.scheduler import test_runtime_serialization as abi


def configuration():
    cfg = allocate_runtime_config(16, num_workers=4)
    cfg.task_num = 6
    cfg.task_index_num[0] = 2
    cfg.task_index_num[1] = 4
    cfg.cube_task_indices[:2] = [0, 1]
    cfg.vector_task_indices[:4] = [2, 3, 4, 5]
    return cfg


def test_explicit_intervals_preserve_original_work():
    cfg = configuration()
    result = install_worker_queues(cfg, [[0, 1], []], [[2], [], [3, 4, 5], []],
                                   cube_ready_depth=8, vector_ready_depths=[0, 0, 8, 0])
    assert cfg.queue_layout == 0
    assert [(q.begin, q.end, q.ready_depth) for q in result.worker_queues] == [
        (0, 2, 8), (2, 2, 8), (0, 1, 0), (1, 1, 0), (1, 4, 8), (4, 4, 0)]
    raw = serialize_runtime_config(result)
    clone = copy_runtime_config(result)
    assert serialize_runtime_config(clone) == raw
    offset = struct.unpack_from('<I', raw, 44)[0]
    assert len(raw) == offset + 6 * 24
    assert struct.unpack_from('<6I', raw, offset + 4 * 24) == (1, 4, 8, 1, 1, 10)


@pytest.mark.parametrize('cubes,vectors', [([[0], [0]], [[2], [3], [4], [5]]),
                                         ([[0], [1]], [[2], [3], [4], []])])
def test_missing_or_duplicate_tasks_rejected(cubes, vectors):
    with pytest.raises(ValueError, match='exactly once'):
        install_worker_queues(configuration(), cubes, vectors)


def test_extended_image_is_accepted_by_production_cpp_reader():
    # Compile and run the same production header used by the independent ABI suite.
    abi.TestRuntimeCppReader.setUpClass()
    reader = abi.TestRuntimeCppReader.reader
    cfg = install_worker_queues(configuration(), [[0], [1]], [[2, 3], [], [4, 5], []])
    raw = serialize_runtime_config(cfg)
    assert reader.valid_runtime(ctypes.create_string_buffer(raw), len(raw), 4096)
    assert not reader.valid_runtime(ctypes.create_string_buffer(raw), len(raw) - 1, 4096)
    damaged = bytearray(raw)
    offset = struct.unpack_from('<I', damaged, 44)[0]
    struct.pack_into('<I', damaged, offset + 4, 99)
    assert not reader.valid_runtime(ctypes.create_string_buffer(bytes(damaged)), len(raw), 4096)


def test_strided_views_preserve_master_placement_and_wait_contract():
    from hyper_multicore.scheduler.worker_queues import install_strided_worker_queues
    cfg = configuration()
    install_strided_worker_queues(cfg)
    assert cfg.queue_layout == 2
    assert [(q.begin, q.end, q.stride, q.ready_depth, q.wait_mode, q.poll_interval_us)
            for q in cfg.worker_queues] == [
        (0, 2, 2, 0, 0, 5), (1, 2, 2, 0, 0, 5),
        (0, 4, 4, 256, 0, 5), (1, 4, 4, 256, 0, 5),
        (2, 4, 4, 256, 0, 5), (3, 4, 4, 256, 0, 5)]
    raw = serialize_runtime_config(cfg)
    abi.TestRuntimeCppReader.setUpClass()
    assert abi.TestRuntimeCppReader.reader.valid_runtime(ctypes.create_string_buffer(raw), len(raw), 4096)
    old = bytearray(raw)
    struct.pack_into('<I', old, 40, 1)
    assert not abi.TestRuntimeCppReader.reader.valid_runtime(ctypes.create_string_buffer(bytes(old)), len(raw), 4096)


def test_copy_keeps_queues_independent():
    cfg = install_worker_queues(configuration(), [[0], [1]], [[2, 3], [], [4, 5], []])
    clone = copy_runtime_config(cfg)
    clone.worker_queues[0].ready_depth = 8
    assert cfg.worker_queues[0].ready_depth == 0
    assert clone.worker_queues[0].ready_depth == 8
