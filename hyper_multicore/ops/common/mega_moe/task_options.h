// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_TASK_OPTIONS_H_
#define HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_TASK_OPTIONS_H_
#include "../../runtime/runtime_config.hpp"
#include "probability_abi.h"
#include "../grouped_matmul_task_options.h"
#include "../swiglu_task_options.h"
#include "../shmem_task_options.h"

namespace MulticoreRuntime {
__aicore__ inline GroupedMatmulTaskMode GetGroupedMatmulTaskMode(const TaskDesc &task) {
  return static_cast<GroupedMatmulTaskMode>(task.extra_value_1);
}
__aicore__ inline SwiGluTaskIteration GetSwiGluTaskIteration(const TaskDesc &task) {
  return static_cast<SwiGluTaskIteration>(task.extra_value_1 & 2U);
}
__aicore__ inline ShmemTaskPartition GetShmemTaskPartition(const TaskDesc &task) {
  return static_cast<ShmemTaskPartition>(task.extra_value_1 & 2U);
}
__aicore__ inline SwiGluClampPosition GetSwiGluClampPosition(const TaskDesc &task) {
  return static_cast<SwiGluClampPosition>(task.extra_value_1 & 1U);
}

enum class MegaMoeActivationWeighting : uint32_t { UNWEIGHTED = 0, BEFORE_FC2 = 4 };
enum class MegaMoeTransportPayload : uint32_t {
  TOKENS = 0, TOKENS_WITH_PROBABILITIES = 4, TOKENS_WITH_PROBABILITY_GRADIENTS = 8
};

__aicore__ inline bool IsBatchedGroupedMatmulTask(const TaskDesc &task) {
  const auto mode = GetGroupedMatmulTaskMode(task);
  return mode == GroupedMatmulTaskMode::BATCHED || mode == GroupedMatmulTaskMode::BATCHED_READY;
}

__aicore__ inline int64_t GmmRowOffset(const TensorDesc &tensor, int64_t start) {
  return tensor.dynamic_shape == 1 ? start * tensor.dim[1] * tensor.data_type
                                  : tensor.base_ptr_offset * tensor.data_type;
}

__aicore__ inline MegaMoeActivationWeighting GetMegaMoeActivationWeighting(const TaskDesc &task) {
  return static_cast<MegaMoeActivationWeighting>(task.extra_value_1 & 4U);
}
__aicore__ inline MegaMoeTransportPayload GetMegaMoeTransportPayload(const TaskDesc &task) {
  return static_cast<MegaMoeTransportPayload>(task.extra_value_1 & 12U);
}

// Scalar offsets are rows; only this business adapter decodes the workspace.
__aicore__ inline bool IsProbabilityOperandValid(const TaskDesc &task, bool weighted,
                                                 uint32_t inputs, uint32_t probability_position) {
  if (task.num_inputs != inputs || task.num_outputs != (weighted ? 2U : 1U)) return false;
  if (!weighted) return true;
  const auto &aux = task.outputs[1];
  return aux.tensor_type == 0 && aux.data_type == ProbabilityLayout::ELEMENT_BYTES &&
         aux.dim[0] == ProbabilityLayout::PLANES && aux.dim[1] > 0 && aux.base_ptr_offset == 0 &&
         aux.input_position == probability_position;
}

__aicore__ inline bool IsActivationTaskOptionsValid(const TaskDesc &task, uint32_t probability_position) {
  constexpr uint32_t allowed = static_cast<uint32_t>(SwiGluTaskIteration::PERSISTENT_LANES) |
      static_cast<uint32_t>(SwiGluClampPosition::AFTER_SILU) |
      static_cast<uint32_t>(MegaMoeActivationWeighting::BEFORE_FC2);
  if ((task.extra_value_1 & ~allowed) != 0) return false;
  if ((GetSwiGluClampPosition(task) == SwiGluClampPosition::AFTER_SILU) && task.extra_value_0 == 0) return false;
  const bool weighted = GetMegaMoeActivationWeighting(task) == MegaMoeActivationWeighting::BEFORE_FC2;
  const uint32_t inputs = task.task_type == TaskType::TASK_SWI_GLU ? 1U : 2U;
  return IsProbabilityOperandValid(task, weighted, inputs, probability_position);
}

__aicore__ inline bool IsTransportTaskOptionsValid(const TaskDesc &task, uint32_t probability_position) {
  constexpr uint32_t scalar_bits = static_cast<uint32_t>(MegaMoeTransportPayload::TOKENS_WITH_PROBABILITIES) |
      static_cast<uint32_t>(MegaMoeTransportPayload::TOKENS_WITH_PROBABILITY_GRADIENTS);
  constexpr uint32_t allowed = static_cast<uint32_t>(ShmemTaskPartition::RUNTIME_ROWS) | scalar_bits;
  if ((task.extra_value_1 & ~allowed) != 0) return false;
  const auto payload = GetMegaMoeTransportPayload(task);
  if (static_cast<uint32_t>(payload) == scalar_bits) return false;
  if (task.task_type == TaskType::TASK_SHMEM_GET_MEM &&
      payload == MegaMoeTransportPayload::TOKENS_WITH_PROBABILITY_GRADIENTS) return false;
  return IsProbabilityOperandValid(task, payload != MegaMoeTransportPayload::TOKENS, 4, probability_position);
}

// Host validation is repeated before any native auxiliary operand is decoded.
__aicore__ inline bool IsMoeTaskOptionsValid(const TaskDesc &task, uint32_t probability_position) {
  switch (task.task_type) {
    case TaskType::TASK_GROUPED_MATMUL:
      return task.extra_value_1 <= static_cast<uint32_t>(GroupedMatmulTaskMode::BATCHED_READY) &&
             task.num_inputs == 3 && task.num_outputs == 1;
    case TaskType::TASK_SWI_GLU:
    case TaskType::TASK_SWI_GLU_GRAD:
      return IsActivationTaskOptionsValid(task, probability_position);
    case TaskType::TASK_SHMEM_PUT_MEM_SIGNAL:
    case TaskType::TASK_SHMEM_GET_MEM:
      return IsTransportTaskOptionsValid(task, probability_position);
    default:
      return true;
  }
}
}  // namespace MulticoreRuntime
#endif
