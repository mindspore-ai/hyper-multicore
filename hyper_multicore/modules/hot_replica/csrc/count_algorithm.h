// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#ifndef HYPER_MULTICORE_CORE_EXPERT_PARALLEL_HOT_REPLICA__DEVICE_KERNEL_COUNT_ALGORITHM_H_
#define HYPER_MULTICORE_CORE_EXPERT_PARALLEL_HOT_REPLICA__DEVICE_KERNEL_COUNT_ALGORITHM_H_

#include <cstdint>

#ifndef HOT_COUNT_DEVICE
#define HOT_COUNT_DEVICE inline
#define HOT_COUNT_LOCAL
#endif

namespace hot {
template <int TopK, typename InputT, typename CountT>
HOT_COUNT_DEVICE void CountFixedRow(const HOT_COUNT_LOCAL InputT *ids, HOT_COUNT_LOCAL CountT *counts,
                                    int experts) {
  static_assert(TopK >= 1 && TopK <= 8, "fixed replica count width must be between one and eight");
  InputT selected[TopK];
#pragma unroll
  for (int index = 0; index < TopK; ++index) selected[index] = ids[index];
#pragma unroll
  for (int index = 0; index < TopK; ++index) {
    const int64_t expert = selected[index];
    const int slot = expert < 0 ? 0 : (expert >= experts ? experts - 1 : static_cast<int>(expert));
    if (expert != slot) counts[experts] = 1;
    ++counts[slot];
#pragma unroll
    for (int previous = 0; previous < index; ++previous) {
      if (selected[previous] == expert) counts[experts] = 1;
    }
  }
}

template <typename InputT>
HOT_COUNT_DEVICE void CountRow(const HOT_COUNT_LOCAL InputT *ids, HOT_COUNT_LOCAL int64_t *counts, int top_k,
                               int experts) {
  for (int index = 0; index < top_k; ++index) {
    const int64_t expert = ids[index];
    const int slot = expert < 0 ? 0 : (expert >= experts ? experts - 1 : static_cast<int>(expert));
    if (expert != slot) counts[experts] = 1;
    ++counts[slot];
    for (int previous = 0; previous < index; ++previous) {
      if (ids[previous] == expert) counts[experts] = 1;
    }
  }
}
}  // namespace hot

#endif  // HYPER_MULTICORE_CORE_EXPERT_PARALLEL_HOT_REPLICA__DEVICE_KERNEL_COUNT_ALGORITHM_H_
