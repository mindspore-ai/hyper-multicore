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
#include "runtime/replica_gradient_udma.h"
#endif

namespace MulticoreRuntime {
class MxTaskWorker : public KernelWorkerBase<MxTaskWorker, VectorWorkerPolicy::ALL,
                                           RuntimeStorageBoundsPolicy::NATIVE_TILING,
                                           MulticoreMoE::InvocationProtocol> {
 public:
  static constexpr uint32_t EVENT_IDX = 65;
  static constexpr uint32_t DIAGNOSTIC_IDX = 66;
  static constexpr uint32_t TILING_IDX = 67;

  __aicore__ inline void ReturnMxGradients() {
#if !defined(__DAV_CUBE__)
    constexpr uint32_t return_position = 40;
    auto launch = reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor *>(input_list[TILING_IDX]);
    if (launch->buffer_count != return_position + 1 || worker_id_ % 2 != 0 || worker_id_ >= 8) return;
    const uint64_t words = launch->buffer_bytes[return_position] / sizeof(uint64_t);
    const auto table = reinterpret_cast<__gm__ const uint64_t *>(input_list[return_position]);
    if (words < 35 || (table[0] != 2 && table[0] != 3) ||
        table[1] != 3 || table[2] < 19 || table[2] > words - 16) {
      AscendC::Trap();
    }
    // Both projections use the same edges. Each helper finishes its W2 reads
    // before W13; W13's final ACK therefore releases both guest banks.
    ReturnMxProjection(table + table[1], table[2] - table[1], launch->ep,
                       launch->experts / launch->ep, table[0] == 2);
    ReturnMxProjection(table + table[2], words - table[2], launch->ep,
                       launch->experts / launch->ep, true);
#endif
  }

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
      case 164:
      case 165:
        MulticoreMx::ExecuteMxQuantTask(task, input_list); break;
      case 105:
      case 141: MulticoreMoE::ExecuteTransfer<bfloat16_t>(task, input_list,
                                                        input_list[TILING_IDX], input_list[EVENT_IDX]); break;
#endif
      default: AscendC::Trap();
    }
  }

 private:
#if !defined(__DAV_CUBE__)
  __aicore__ inline void ReturnMxProjection(__gm__ const uint64_t *record, uint64_t words, uint32_t ep,
                                           uint32_t local_experts, bool acknowledge) {
    constexpr uint32_t helpers = 4;
    const uint32_t rank = record[1], slots = record[2], epoch = record[3], worker = worker_id_ / 2;
    const uint64_t elements = record[4], incoming_count = record[10], owned_count = record[11];
    if (record[0] != 9 || rank >= ep || slots == 0 || slots >= local_experts || epoch == 0 || elements == 0 ||
        incoming_count > slots || owned_count > ep * slots || record[14] != helpers ||
        record[15] != 16 + 3 * incoming_count + 4 * owned_count || words != record[15] + ep * helpers) {
      AscendC::Trap();
    }
    const auto output = reinterpret_cast<GM_ADDR>(record[5]);
    const auto guest = reinterpret_cast<GM_ADDR>(record[6]);
    const auto ready = reinterpret_cast<GM_ADDR>(record[7]);
    const auto ack = reinterpret_cast<GM_ADDR>(record[8]);
    const auto done = reinterpret_cast<GM_ADDR>(record[9]);
    const auto scratch = reinterpret_cast<GM_ADDR>(record[12]);
    const auto owned = record + 16 + 3 * incoming_count;
    // Publish all local producers before waiting for any remote producer or ACK.
    if (worker == 0) {
      for (uint32_t i = 0; i < incoming_count; ++i) {
        const auto incoming = record + 16 + 3 * i;
        if (incoming[0] >= ep || incoming[1] >= slots) AscendC::Trap();
        WaitForDependency(incoming[2], 5);
        aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(ready +
            (rank * slots + incoming[1]) * DATA_CACHE_LINE_SIZE), epoch, ACLSHMEM_SIGNAL_SET, incoming[0]);
      }
    }
    for (uint32_t i = 0; i < owned_count; ++i) {
      const auto edge = owned + 4 * i;
      const uint32_t peer = edge[0], slot = edge[1], owner = edge[2];
      if (peer >= ep || slot >= slots || owner >= local_experts - slots) AscendC::Trap();
      WaitForDependency(edge[3], 5);
      WaitReplicaCompletion(ready + (peer * slots + slot) * DATA_CACHE_LINE_SIZE, epoch);
      const uint32_t qp = record[record[15] + peer * helpers + worker];
      AddReplicaGradientUdma(output + owner * elements * sizeof(float), guest + slot * elements * sizeof(float),
                            elements, peer, worker, helpers, scratch, record[13], qp);
    }
    if (!acknowledge) return;
    StoreReplicaCompletion(done + worker * DATA_CACHE_LINE_SIZE, epoch);
    if (worker == 0) {
      for (uint32_t i = 0; i < helpers; ++i) WaitReplicaCompletion(done + i * DATA_CACHE_LINE_SIZE, epoch);
      for (uint32_t i = 0; i < owned_count; ++i) {
        const auto edge = owned + 4 * i;
        aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(ack +
            (rank * slots + edge[1]) * DATA_CACHE_LINE_SIZE), epoch, ACLSHMEM_SIGNAL_SET, edge[0]);
      }
      for (uint32_t i = 0; i < incoming_count; ++i) {
        const auto incoming = record + 16 + 3 * i;
        WaitReplicaCompletion(ack + (incoming[0] * slots + incoming[1]) * DATA_CACHE_LINE_SIZE, epoch);
      }
    }
  }
#endif
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
  worker.ReturnMxGradients();
}
