/**
 * Copyright 2026 Huawei Technologies Co., Ltd.
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
#ifndef HMC_MOE_GRADIENT_BAND_PRODUCER_H
#define HMC_MOE_GRADIENT_BAND_PRODUCER_H

#include "kernel_operator.h"

namespace MulticoreRuntime {

class GradientBandProducer {
 public:
  static constexpr uint32_t kMaxBands = 64;

  __aicore__ inline void Set(GM_ADDR flags, int32_t epoch, uint32_t band_rows) {
    flags_ = flags;
    epoch_ = epoch;
    band_rows_ = band_rows;
  }

  __aicore__ inline bool Enabled() const { return flags_ != nullptr; }

  __aicore__ inline void Begin(uint32_t rows, uint32_t core, uint32_t cores) {
    if (band_rows_ == 0) {
      AscendC::Trap();
    }
    bands_ = (rows + band_rows_ - 1) / band_rows_;
    core_ = core;
    cores_ = cores;
    if (band_rows_ == 0 || bands_ > kMaxBands || epoch_ <= 0 || core >= cores) {
      AscendC::Trap();
    }
    for (uint32_t band = 0; band < bands_; ++band) {
      pending_[band] = 0;
    }
  }

  __aicore__ inline void Expect(uint32_t row, uint32_t rows) {
    for (uint32_t band = row / band_rows_; band <= (row + rows - 1) / band_rows_; ++band) {
      ++pending_[band];
    }
  }

  __aicore__ inline void PublishEmpty() {
    for (uint32_t band = 0; band < bands_; ++band) {
      if (pending_[band] == 0) {
        Publish(band);
      }
    }
  }

  __aicore__ inline void Retire(uint32_t row, uint32_t rows) {
    const uint32_t first = row / band_rows_;
    const uint32_t last = (row + rows - 1) / band_rows_;
    bool publish = false;
    for (uint32_t band = first; band <= last; ++band) {
      if (pending_[band] == 0) {
        AscendC::Trap();
      }
      publish = --pending_[band] == 0 || publish;
    }
    if (!publish) {
      return;
    }
    // IterateAll<false> can return before the final GM output. This is the
    // installed MatmulClient::WaitIterateAll FIX_MTE2 completion boundary.
    auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(event);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(event);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t band = first; band <= last; ++band) {
      if (pending_[band] == 0) {
        Publish(band);
      }
    }
  }

 private:
  __aicore__ inline void Publish(uint32_t band) {
    AscendC::GlobalTensor<int32_t> flag;
    flag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flags_ + (band * cores_ + core_) * 64), 1);
    AscendC::PipeBarrier<PIPE_ALL>();
    flag.SetValue(0, epoch_);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                      AscendC::DcciDst::CACHELINE_OUT>(flag);
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  GM_ADDR flags_ = nullptr;
  int32_t epoch_ = 0;
  uint32_t band_rows_ = 0;
  uint32_t bands_ = 0;
  uint32_t core_ = 0;
  uint32_t cores_ = 0;
  uint32_t pending_[kMaxBands];
};

}  // namespace MulticoreRuntime

#endif  // HMC_MOE_GRADIENT_BAND_PRODUCER_H
