// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_TRANSPORT_H_
#define HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_TRANSPORT_H_
#include "task_options.h"
#include "probability_workspace.h"
#include "../../runtime/get_mem.h"

namespace MulticoreRuntime {
// The serialized offsets below count token elements; probability offsets count rows.
template <typename T>
__aicore__ inline void ExecuteMoeShmemGetMem(const TaskDesc &task, GM_ADDR *input_list, GM_ADDR tiling) {
  int64_t ep = getExtraValueFromTiling(tiling, 1);
  int64_t experts = getExtraValueFromTiling(tiling, 2);
  int64_t hidden = getExtraValueFromTiling(tiling, 3);
  int64_t tiles = task.task_split_num / experts;
  int64_t source_pe = task.task_index / ((experts / ep) * tiles);
  int64_t size =
    reinterpret_cast<__gm__ int32_t *>(input_list[task.inputs[3].input_position])[task.inputs[3].base_ptr_offset];
  const bool runtime_rows = GetShmemTaskPartition(task) == ShmemTaskPartition::RUNTIME_ROWS;
  const int64_t partition_rows = runtime_rows ? (size / hidden + tiles - 1) / tiles : task.task_split_value;
  const int64_t tile_elements = partition_rows * hidden;
  const int64_t start = (task.task_index % tiles) * tile_elements;
  if (start >= size) {
    return;
  }
  int64_t elements = size - start < tile_elements ? size - start : tile_elements;
  int64_t destination_offset =
    reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position])[task.inputs[0].base_ptr_offset];
  int64_t source_offset =
    reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[2].input_position])[task.inputs[2].base_ptr_offset];
  if (GetMegaMoeTransportPayload(task) == MegaMoeTransportPayload::TOKENS_WITH_PROBABILITIES) {
    const auto &aux = task.outputs[task.num_outputs - 1];
    GM_ADDR buffer = input_list[aux.input_position];
    PullToLocal<float>(
        ProbabilityPlane(buffer, aux.dim[1], ProbabilityLayout::RECEIVED) +
            (destination_offset + start) / hidden * sizeof(float),
        ProbabilityPlane(buffer, aux.dim[1], ProbabilityLayout::SOURCE) +
            (source_offset + start) / hidden * sizeof(float), elements / hidden, source_pe);
  }
  // Long compact GET tiles amortize the MTE loop with a larger burst.
  constexpr uint64_t long_read_bytes = 4UL * 1024 * 1024;
  const bool long_read = runtime_rows && static_cast<uint64_t>(elements) * sizeof(T) >= long_read_bytes;
  const uint64_t burst_bytes = long_read ? 32 * 1024 : 4 * 1024;
  PullToLocal<T>(input_list[task.outputs[0].input_position] + (destination_offset + start) * sizeof(T),
                 input_list[task.inputs[1].input_position] + (source_offset + start) * sizeof(T), elements,
                 source_pe, burst_bytes);
}
}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_TRANSPORT_H_
