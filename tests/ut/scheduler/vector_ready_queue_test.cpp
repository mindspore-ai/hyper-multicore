// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#include "hyper_multicore/ops/runtime/vector_ready_queue.h"

struct Queue {
  std::vector<uint32_t> executed;
  uint32_t unlocked = 0;
  uint32_t records = 0;
  bool profile = false;
  uint32_t TaskAt(uint32_t index) { return index; }
  bool IsReady(uint32_t task) { return task % 13 != 0 || task <= unlocked; }
  void Execute(uint32_t task) {
    executed.push_back(task);
    if (profile) ++records;
  }
  void Wait(uint32_t task) { unlocked = task; }
};

int main() {
  for (uint32_t tasks : {0U, 1U, 255U, 256U, 257U, 18001U}) {
    for (uint32_t stride : {1U, 20U, 40U}) {
      std::vector<uint32_t> covered(tasks);
      for (uint32_t first = 0; first < stride; ++first) {
        Queue normal, profile;
        profile.profile = true;
        MulticoreRuntime::RunVectorReadyQueue(normal, first, tasks, stride);
        MulticoreRuntime::RunVectorReadyQueue(profile, first, tasks, stride);
        assert(normal.executed == profile.executed);
        assert(normal.records == 0 && profile.records == profile.executed.size());
        for (auto task : normal.executed) {
          assert(task % stride == first);
          ++covered[task];
        }
      }
      for (auto count : covered) assert(count == 1);
    }
  }
}
