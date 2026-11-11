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

#include <cstdint>
#include <stdexcept>

#define __aicore__
#define __gm__
using GM_ADDR = uint8_t *;

// Cache operations are no-ops on the host; addressing and record copying below
// compile the production implementation without replacing its arithmetic.
namespace AscendC {
enum class CacheLine { SINGLE_CACHE_LINE, ENTIRE_DATA_CACHE };
enum class DcciDst { CACHELINE_OUT };
constexpr int PIPE_ALL = 0;
template <typename T> class GlobalTensor {
 public:
  void SetGlobalBuffer(T *value, uint64_t) { data_ = value; }
  T GetValue(uint64_t index) const { return data_[index]; }
  void SetValue(uint64_t index, T value) { data_[index] = value; }
  GlobalTensor operator[](uint64_t index) const {
    GlobalTensor result;
    result.data_ = data_ + index;
    return result;
  }
 private:
  T *data_ = nullptr;
};
template <typename T, CacheLine, DcciDst = DcciDst::CACHELINE_OUT>
void DataCacheCleanAndInvalid(GlobalTensor<T>) {}
template <int> void PipeBarrier() {}
[[noreturn]] inline void Trap() { throw std::runtime_error("invalid diagnostic pointer"); }
}  // namespace AscendC

#include "hyper_multicore/ops/runtime/l1_tensor_dump.h"

extern "C" uint64_t resolve_l1_address(GM_ADDR *parameters, uint32_t position, GM_ADDR table, uint32_t mode) {
  try {
    if (mode == 1) {
      return reinterpret_cast<uint64_t>(
        MulticoreRuntime::ResolveL1ParameterAddress<29, 6>(parameters, position, table));
    }
    if (mode == 2) {
      return reinterpret_cast<uint64_t>(
        MulticoreRuntime::ResolveL1ParameterAddress<35, 12>(parameters, position, table));
    }
    return reinterpret_cast<uint64_t>(MulticoreRuntime::ResolveL1ParameterAddress<>(parameters, position, table));
  } catch (const std::runtime_error &) { return 0; }
}

extern "C" bool capture_l1(GM_ADDR buffer, GM_ADDR *parameters, GM_ADDR table,
                           uint32_t task, uint32_t phase, bool forward) {
  try {
    if (forward) {
      MulticoreRuntime::CaptureL1TaskTensors<29, 6>(buffer, parameters, task, phase, 0, table);
    } else {
      MulticoreRuntime::CaptureL1TaskTensors<35, 12>(buffer, parameters, task, phase, 0, table);
    }
    return true;
  } catch (const std::runtime_error &) { return false; }
}
