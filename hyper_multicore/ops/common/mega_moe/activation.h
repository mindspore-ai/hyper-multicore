/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 1.0.
 */
#ifndef HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_ACTIVATION_H_
#define HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_ACTIVATION_H_

// The source assembler supplies the locked SwiGLU implementation at this path.
#include "../../swi_glu/swi_glu.cpp"  // NOLINT(build/include)
#include "situ_glu.h"
#include "task_options.h"

namespace MulticoreRuntime {

// Each launch exposes one read-only parameter record at its activation input.
__aicore__ inline void ValidateSituLaunch(const TaskDesc &task, TaskType expectedType, uint32_t port,
                                         GM_ADDR config, GM_ADDR launchTiling) {
  if (!IsSituGluTask(task)) return;
  const auto *header = reinterpret_cast<__gm__ const RuntimeHeader *>(config);
  if (!AscendC::IsSameType<DTYPE_DISPATCH_TARGET, bfloat16_t>::value ||
      header->protocol_version < SITU_RUNTIME_PROTOCOL_VERSION || task.task_type != expectedType ||
      task.tiling_data_position != port || task.tiling_data_offset != 0 ||
      getExtraValueFromTiling(launchTiling, 8) != sizeof(SituParameters)) AscendC::Trap();
}

// A tiled task owns one row interval. A persistent lane also owns each subsequent
// interval at the same lane index, up to the end of this expert.
struct ActivationRowRange {
  int64_t first;
  int64_t end;
  int64_t stride;
};

__aicore__ inline ActivationRowRange GetActivationRows(const TaskDesc &task, const TensorDesc &anchor,
                                                       GM_ADDR *inputs, GM_ADDR config) {
  const int64_t tileRows = task.task_split_value;
  if (tileRows <= 0) AscendC::Trap();
  if (!task.inputs[0].dynamic_shape && !task.outputs[0].dynamic_shape) {
    const int64_t first = anchor.base_ptr_offset / anchor.dim[1];
    return {first, first + tileRows, tileRows};
  }
  DynamicData data;
  getDynamicData(config, &data);
  if (data.dynamic_type != DynamicType::DYNAMIC_DSV3_MOE) return {0, 0, 1};
  if (data.dynamic_group_size == 0 || task.task_split_num == 0 ||
      task.task_split_num % data.dynamic_group_size != 0 || task.task_index >= task.task_split_num) AscendC::Trap();
  const uint32_t lanes = task.task_split_num / data.dynamic_group_size;
  const uint32_t expert = task.task_index / lanes;
  AscendC::GlobalTensor<int64_t> groups;
  groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(inputs[data.dynamic_input_position]),
                         data.dynamic_group_size);
  const int64_t start = expert == 0 ? 0 : groups.GetValue(expert - 1);
  int64_t end = groups.GetValue(expert);
  const int64_t first = start + (task.task_index % lanes) * tileRows;
  if (GetSwiGluTaskIteration(task) == SwiGluTaskIteration::TILED && first + tileRows < end) end = first + tileRows;
  return {first, end, lanes * tileRows};
}

__aicore__ inline GM_ADDR ActivationOperand(GM_ADDR *inputs, const TensorDesc &tensor, int64_t row) {
  const int64_t offset = tensor.dynamic_shape == 1 ? row * tensor.dim[1] : tensor.base_ptr_offset;
  return inputs[tensor.input_position] + offset * tensor.data_type;
}

__aicore__ inline float DecodeActivationParameter(uint32_t bits) {
  union {
    uint32_t bits;
    float value;
  } encoded = {bits};
  return encoded.value;
}

__aicore__ inline float GetSwiGluClampLimit(const TaskDesc &task) {
  return DecodeActivationParameter(task.extra_value_0);
}

__aicore__ inline void RunMoeActivationForward(const TaskDesc &task, GM_ADDR input, GM_ADDR output, int64_t rows,
                                               int64_t width, GM_ADDR tiling, const SituParameters &params) {
  if (IsSituGluTask(task)) {
    RunSiTUGlu<false>(input, nullptr, output, rows, width, params);
  } else {
    swi_glu(input, output, nullptr, tiling, GetSwiGluClampLimit(task),
            GetSwiGluClampPosition(task) == SwiGluClampPosition::AFTER_SILU);
  }
}

}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_ACTIVATION_H_
