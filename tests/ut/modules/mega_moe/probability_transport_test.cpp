/*
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>
#include <cassert>
#include <cstring>
#include <vector>

#define __aicore__
#define __gm__

// Only the unused synchronization primitive needs a host stand-in. Layout and
// task decoding below compile the production header itself.
namespace AscendC {
enum class HardEvent {};
struct Pipe {
  uint32_t FetchEventID(HardEvent) { return 0; }
};
inline Pipe *GetTPipePtr() { return nullptr; }
template <HardEvent event>
inline void SetFlag(uint32_t) {}
template <HardEvent event>
inline void WaitFlag(uint32_t) {}
}  // namespace AscendC
using AscendC::GetTPipePtr;
#include "hyper_multicore/ops/runtime/runtime_config.hpp"
#include "hyper_multicore/ops/runtime/runtime_storage_bounds_host.h"

#include "hyper_multicore/ops/common/swiglu_task_options.h"
#include "hyper_multicore/ops/common/mega_moe/probability_abi.h"


using GM_ADDR = uint8_t *;
struct CopyCall {
  int64_t elements;
  int peer;
};
std::vector<CopyCall> calls;

template <typename T>
void PullToLocal(GM_ADDR destination, GM_ADDR source, int64_t elements, int peer) {
  calls.push_back({elements, peer});
  std::memcpy(destination, source, elements * sizeof(T));
}
#define HYPER_MULTICORE_MULTICORE_GET_MEM_H
#include "hyper_multicore/ops/common/mega_moe/transport.h"

void CheckTransfer(bool weighted, int32_t size) {
  using namespace MulticoreRuntime;  // NOLINT(build/namespaces)
  uint16_t source[64], destination[64];
  float probabilities[4 * 32];
  for (int index = 0; index < 64; ++index) {
    source[index] = index + 100;
    destination[index] = 0;
  }
  for (int index = 0; index < 4 * 32; ++index) {
    probabilities[index] = index < 32 ? index + 0.25f : -1.0f;
  }
  int64_t destination_offset = 12, source_offset = 4;
  int64_t tiling[] = {0, 2, 4, 4};
  GM_ADDR parameters[26] = {};
  parameters[0] = reinterpret_cast<GM_ADDR>(destination);
  parameters[1] = reinterpret_cast<GM_ADDR>(source);
  parameters[2] = reinterpret_cast<GM_ADDR>(&destination_offset);
  parameters[3] = reinterpret_cast<GM_ADDR>(&source_offset);
  parameters[4] = reinterpret_cast<GM_ADDR>(&size);
  parameters[ProbabilityLayout::FORWARD_POSITION] = reinterpret_cast<GM_ADDR>(probabilities);
  TaskDesc task = {};
  task.task_index = 5;  // Peer 1, second two-row tile.
  task.task_split_num = 8;
  task.task_split_value = 2;
  task.inputs[0].input_position = 2;
  task.inputs[1].input_position = 1;
  task.inputs[2].input_position = 3;
  task.inputs[3].input_position = 4;
  task.outputs[0].input_position = 0;
  task.outputs[1].input_position = ProbabilityLayout::FORWARD_POSITION;
  task.outputs[1].dim[1] = 32;
  task.num_outputs = weighted ? 2 : 1;
  task.extra_value_1 = weighted ? static_cast<uint32_t>(MegaMoeTransportPayload::TOKENS_WITH_PROBABILITIES) : 0;
  calls.clear();
  ExecuteMoeShmemGetMem<uint16_t>(task, parameters, reinterpret_cast<GM_ADDR>(tiling));
  if (size <= 8) {
    assert(calls.empty());
    assert(probabilities[32 + 5] == -1.0f);
    assert(destination[20] == 0);
    return;
  }
  const int elements = size - 8 < 8 ? size - 8 : 8;
  assert(calls.size() == (weighted ? 2U : 1U));
  assert(calls.back().peer == 1);
  assert(calls.back().elements == elements);
  for (int index = 0; index < elements; ++index) {
    assert(destination[20 + index] == source[12 + index]);
  }
  for (int row = 0; row < 32; ++row) {
    const bool copied = weighted && row >= 5 && row < 5 + elements / 4;
    assert(probabilities[32 + row] == (copied ? probabilities[row - 2] : -1.0f));
    assert(probabilities[64 + row] == -1.0f);
    assert(probabilities[96 + row] == -1.0f);
  }
  if (weighted) {
    assert(calls[0].peer == 1);
    assert(calls[0].elements == elements / 4);
  }
}

int main() {
  for (bool weighted : {false, true}) {
    for (int32_t size : {0, 8, 12, 20}) {
      CheckTransfer(weighted, size);
    }
  }
}
