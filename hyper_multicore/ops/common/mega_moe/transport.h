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
  int64_t start = (task.task_index % tiles) * static_cast<int64_t>(task.task_split_value) * hidden;
  if (start >= size) {
    return;
  }
  int64_t tile_elements = static_cast<int64_t>(task.task_split_value) * hidden;
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
  PullToLocal<T>(input_list[task.outputs[0].input_position] + (destination_offset + start) * sizeof(T),
                 input_list[task.inputs[1].input_position] + (source_offset + start) * sizeof(T), elements,
                 source_pe);
}
}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_TRANSPORT_H_
