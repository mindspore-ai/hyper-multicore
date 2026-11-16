// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once
// MXFP8 task dispatch shared by forward and backward native entries.
#include "kernel_operator.h"
#include "common/mega_moe/contracts/mx_launch_descriptor.h"
#include "common/mega_moe/contracts/mx_byte_tensor_list.h"
#include "runtime/worker_kernel.h"
#include "common/mega_moe/invocation_protocol.h"
#include "common/mega_moe/transport_task.h"
#ifdef __DAV_CUBE__
#include "common/mega_moe/adapters/arch35/gmm/mxfp8_worker.h"
#else
#include "common/mega_moe/adapters/arch35/quantization/dual_axis_mxfp8_worker.h"
#endif

namespace MulticoreRuntime {
class MxTaskWorker : public KernelWorkerBase<MxTaskWorker, VectorWorkerPolicy::ALL,
                                           RuntimeStorageBoundsPolicy::NATIVE_TILING,
                                           MulticoreMoE::InvocationProtocol> {
 public:
  static constexpr uint32_t EVENT_IDX = 65;
  static constexpr uint32_t DIAGNOSTIC_IDX = 66;
  static constexpr uint32_t TILING_IDX = 67;

  __aicore__ inline void ExecuteComputeKernel(TaskDesc task) {
    auto launch = reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor*>(input_list[TILING_IDX]);
    if (task.num_inputs > 4 || task.num_outputs > 4) AscendC::Trap();
    for (uint32_t i = 0; i < task.num_inputs; ++i)
      if (task.inputs[i].input_position >= launch->buffer_count) AscendC::Trap();
    for (uint32_t i = 0; i < task.num_outputs; ++i)
      if (task.outputs[i].input_position >= launch->buffer_count) AscendC::Trap();
    switch (static_cast<uint32_t>(task.task_type)) {
      case 0:
      case 10: break;
#ifdef __DAV_CUBE__
      case 142: MulticoreMx::ExecuteMxGmmTask(task, input_list, launch); break;
#else
      case 143:
      case 145:
      case 146:
        MulticoreMx::ExecuteMxQuantTask(task, input_list); break;
      case 105:
      case 141: MulticoreMoE::ExecuteTransfer<bfloat16_t>(task, input_list,
                                                        input_list[TILING_IDX], input_list[EVENT_IDX]); break;
#endif
      default: AscendC::Trap();
    }
  }
};
}  // namespace MulticoreRuntime

__aicore__ inline void RunMegaMoeMxfp8(GM_ADDR buffers, GM_ADDR runtime, GM_ADDR events,
                                     GM_ADDR diagnostic, GM_ADDR launch_metadata) {
  auto launch = reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor*>(launch_metadata);
  if (launch->magic != MulticoreMx::MX_LAUNCH_MAGIC || launch->version != 1 || launch->reserved != 0 ||
      launch->buffer_count == 0 || launch->buffer_count > 64 ||
      !MulticoreMx::IsMxLaunchTopologyValid(launch->ep, launch->experts) ||
      !MulticoreRuntime::usesExplicitWorkerQueues(runtime)) AscendC::Trap();
  auto words = reinterpret_cast<__gm__ uint64_t*>(buffers);
  const uint64_t offset = words[0];
  if (!MulticoreMx::IsMxByteListHeaderValid(offset, launch->buffer_count)) AscendC::Trap();
  auto pointers = words + offset / sizeof(uint64_t);
  GM_ADDR inputs[68] = {};
  for (uint32_t i = 0; i < launch->buffer_count; ++i) {
    if (!MulticoreMx::IsMxByteVectorShapeValid(words[1 + 2 * i], words[2 + 2 * i], launch->buffer_bytes[i]))
      AscendC::Trap();
    inputs[i] = reinterpret_cast<GM_ADDR>(pointers[i]);
  }
  inputs[64] = runtime;
  inputs[65] = events;
  inputs[66] = diagnostic;
  inputs[67] = launch_metadata;
  MulticoreRuntime::MxTaskWorker worker;
  worker.Init(AscendC::GetBlockIdx(), runtime, inputs);
  worker.Process();
}
