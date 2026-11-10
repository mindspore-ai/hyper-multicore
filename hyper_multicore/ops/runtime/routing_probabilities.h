// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_ROUTING_PROBABILITIES_H_
#define HYPER_MULTICORE_ROUTING_PROBABILITIES_H_
#include "kernel_operator.h"

namespace MulticoreRuntime {
constexpr uint32_t APPLY_ROUTING_PROBABILITIES = 4;
constexpr uint32_t RETURN_ROUTING_PROBABILITIES = 8;

__aicore__ inline GM_ADDR ProbabilityPlane(GM_ADDR workspace, int64_t stride, uint32_t plane) {
  return workspace + static_cast<int64_t>(plane) * stride * sizeof(float);
}

// Operate after the existing SwiGLU kernel's BF16 store. This preserves both
// rounding boundaries of (activation_bf16 * probability_fp32).to(bfloat16).
// Tile boundaries drain pending GM stores before UB reuse.
template <typename T>
class RoutingProbabilityTile {
 public:
  __aicore__ inline RoutingProbabilityTile() {}

  __aicore__ inline void Forward(GM_ADDR output, GM_ADDR probs, uint32_t rows, uint32_t width) {
    InitBuffers(false);
    BeginChunk();
    Load(probabilities.Get<float>(), probs, rows, true);
    const uint32_t elements = rows * width;
    const uint32_t chunk = width <= COLS && width % 8 == 0 ? COLS / width * width : COLS;
    for (uint32_t start = 0; start < elements;) {
      // Whole rows avoid splitting probability reductions across chunks.
      BeginChunk();
      uint32_t count = elements - start < chunk ? elements - start : chunk;
      // Vector tensor slices require 32-byte alignment. Keep unaligned rows
      // in separate chunks so each local float tensor starts at offset zero.
      if (width % 8 != 0 && count > width - start % width) {
        count = width - start % width;
      }
      auto data = first.Get<T>();
      auto fp32 = values.Get<float>();
      GM_ADDR address = output + static_cast<int64_t>(start) * sizeof(T);
      Load(data, address, count);
      Cast(fp32, data, AscendC::RoundMode::CAST_NONE, count);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t cursor = 0; cursor < count;) {
        const uint32_t absolute = start + cursor;
        const uint32_t remaining = width - absolute % width;
        const uint32_t segment = count - cursor < remaining ? count - cursor : remaining;
        Muls(fp32[cursor], fp32[cursor], probabilities.Get<float>().GetValue(absolute / width), segment);
        cursor += segment;
      }
      AscendC::PipeBarrier<PIPE_V>();
      Cast(data, fp32, AscendC::RoundMode::CAST_RINT, count);
      Store(address, data, count);
      start += count;
    }
    Finish();
  }

  __aicore__ inline void Backward(GM_ADDR grad, GM_ADDR activation, GM_ADDR probs,
                                  GM_ADDR grad_probs, uint32_t rows, uint32_t width) {
    InitBuffers(true);
    BeginChunk();
    Load(probabilities.Get<float>(), probs, rows, true);
    for (uint32_t row = 0; row < rows; ++row) {
      gradients.Get<float>().SetValue(row, 0.0f);
    }
    const uint32_t elements = rows * width;
    const uint32_t chunk = width <= COLS && width % 8 == 0 ? COLS / width * width : COLS;
    for (uint32_t start = 0; start < elements;) {
      // Whole rows avoid splitting probability reductions across chunks.
      BeginChunk();
      uint32_t count = elements - start < chunk ? elements - start : chunk;
      // Vector tensor slices require 32-byte alignment. Keep unaligned rows
      // in separate chunks so each local float tensor starts at offset zero.
      if (width % 8 != 0 && count > width - start % width) {
        count = width - start % width;
      }
      const int64_t offset = static_cast<int64_t>(start) * sizeof(T);
      auto dy = first.Get<T>();
      auto act = second.Get<T>();
      auto dy32 = values.Get<float>();
      auto act32 = other.Get<float>();
      auto product = products.Get<float>();
      Load(dy, grad + offset, count);
      Load(act, activation + offset, count);
      Cast(dy32, dy, AscendC::RoundMode::CAST_NONE, count);
      Cast(act32, act, AscendC::RoundMode::CAST_NONE, count);
      AscendC::PipeBarrier<PIPE_V>();
      Mul(product, dy32, act32, count);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t cursor = 0; cursor < count;) {
        const uint32_t absolute = start + cursor;
        const uint32_t row = absolute / width;
        const uint32_t remaining = width - absolute % width;
        const uint32_t segment = count - cursor < remaining ? count - cursor : remaining;
        ReduceSum(sums.Get<float>(), product[cursor], reduction.Get<float>(), segment);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        gradients.Get<float>().SetValue(row, gradients.Get<float>().GetValue(row) + sums.Get<float>().GetValue(0));
        Muls(dy32[cursor], dy32[cursor], probabilities.Get<float>().GetValue(row), segment);
        cursor += segment;
      }
      AscendC::PipeBarrier<PIPE_V>();
      Cast(dy, dy32, AscendC::RoundMode::CAST_RINT, count);
      Store(grad + offset, dy, count);
      start += count;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID3);
    Store(grad_probs, gradients.Get<float>(), rows);
    Finish();
  }

 private:
  static constexpr uint32_t COLS = 4096;
  static constexpr uint32_t MAX_ROWS = 128;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> first, second, values, other, products, reduction;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sums, probabilities, gradients;

  __aicore__ inline void InitBuffers(bool backward) {
    pipe.InitBuffer(first, COLS * sizeof(T));
    pipe.InitBuffer(values, COLS * sizeof(float));
    pipe.InitBuffer(probabilities, MAX_ROWS * sizeof(float));
    if (backward) {
      pipe.InitBuffer(second, COLS * sizeof(T));
      pipe.InitBuffer(other, COLS * sizeof(float));
      pipe.InitBuffer(products, COLS * sizeof(float));
      pipe.InitBuffer(reduction, COLS * sizeof(float));
      pipe.InitBuffer(sums, 32);
      pipe.InitBuffer(gradients, MAX_ROWS * sizeof(float));
    }
  }

  __aicore__ inline void Finish() {
    // Drain GM stores before the caller reuses the tile's UB.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
  }

  __aicore__ inline void BeginChunk() {
    // Protect UB reuse after the preceding chunk's MTE3 store.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
  }

  template <typename U>
  __aicore__ inline void Load(AscendC::LocalTensor<U> local, GM_ADDR address, uint32_t count,
                              bool scalar = false) {
    AscendC::GlobalTensor<U> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(address));
    AscendC::DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(U)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<U> pad{false, 0, 0, 0};
    DataCopyPad(local, global, copy, pad);
    if (scalar) {
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    } else {
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
    }
  }

  template <typename U>
  __aicore__ inline void Store(GM_ADDR address, AscendC::LocalTensor<U> local, uint32_t count) {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
    AscendC::GlobalTensor<U> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(address));
    AscendC::DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(U)), 0, 0, 0};
    DataCopyPad(global, local, copy);
  }
};
}  // namespace MulticoreRuntime
#endif
