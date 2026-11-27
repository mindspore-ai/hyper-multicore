/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
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
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <tuple>
#include "csrc/cached_op_api.h"

namespace {
// The extension bridge otherwise resolves these helpers on every call.
// As in native op-plugin, cache code addresses while the provider stays loaded.
// SHMEM addresses and TLS lifecycle remain per-call, including after heap growth.
void InitExecCommonCtx() {
  static const auto initialize =
    reinterpret_cast<int (*)(void *, bool)>(GetOpApiFuncAddr("InitHugeMemThreadLocal"));
  if (initialize != nullptr) {
    initialize(nullptr, false);
  }
}

void ReleaseExecCommonCtx() {
  static const auto release = reinterpret_cast<void (*)(void *, bool)>(GetOpApiFuncAddr("ReleaseHugeMem"));
  if (release != nullptr) {
    release(nullptr, false);
  }
}

void UnInitExecCommonCtx() {
  static const auto uninitialize =
    reinterpret_cast<void (*)(void *, bool)>(GetOpApiFuncAddr("UnInitHugeMemThreadLocal"));
  static const auto clear_cache = reinterpret_cast<void (*)()>(GetOpApiFuncAddr("UnInitPTACacheThreadLocal"));
  if (uninitialize != nullptr) {
    uninitialize(nullptr, false);
  }
  if (clear_cache != nullptr) {
    clear_cache();
  }
}

using PermuteOutputs = std::tuple<at::Tensor &, at::Tensor &>;

void check_permute_outputs(const at::Tensor &tokens, const at::Tensor &indices, const at::Tensor &output,
                           const at::Tensor &mapping) {
  TORCH_CHECK(tokens.dim() == 2 && indices.dim() == 2 && output.dim() == 2 && mapping.dim() == 1,
              "permute requires matrix tokens, indices and output, and a flat mapping");
  TORCH_CHECK(indices.size(0) == tokens.size(0) && output.size(0) == indices.numel() &&
                output.size(1) == tokens.size(1) && mapping.numel() == indices.numel(),
              "permute requires all top-k rows without padding or dropped tokens");
  TORCH_CHECK((tokens.scalar_type() == at::kHalf || tokens.scalar_type() == at::kBFloat16 ||
               tokens.scalar_type() == at::kFloat) &&
                output.scalar_type() == tokens.scalar_type(),
              "permute requires matching float16, bfloat16 or float32 tokens and output");
  TORCH_CHECK(
    (indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong) && mapping.scalar_type() == at::kInt,
    "permute requires int32/int64 indices and an int32 mapping");
  for (const auto *tensor : {&tokens, &indices, &output, &mapping}) {
    TORCH_CHECK(tensor->device() == tokens.device() && tensor->is_contiguous(),
                "permute tensors must be contiguous and share one device");
  }
  at::assert_no_overlap(output, mapping);
  for (const auto *input : {&tokens, &indices}) {
    at::assert_no_overlap(output, *input);
    at::assert_no_overlap(mapping, *input);
  }
}

PermuteOutputs permute_npu(const at::Tensor &tokens, const at::Tensor &indices, at::Tensor &output,
                           at::Tensor &mapping) {
  check_permute_outputs(tokens, indices, output, mapping);
  const c10::DeviceGuard guard(tokens.device());
  if (indices.numel() != 0) {
    int64_t num_out_tokens = indices.numel();
    bool padded_mode = false;
    HYPER_EXEC_NPU_CMD_EXT(aclnnMoeTokenPermute, tokens, indices, num_out_tokens, padded_mode, output, mapping);
  }
  return {output, mapping};
}

PermuteOutputs permute_meta(const at::Tensor &tokens, const at::Tensor &indices, at::Tensor &output,
                            at::Tensor &mapping) {
  check_permute_outputs(tokens, indices, output, mapping);
  return {output, mapping};
}

}  // namespace

TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) { m.impl("moe_token_permute_out", &permute_npu); }

TORCH_LIBRARY_IMPL(hyper_multicore, Meta, m) { m.impl("moe_token_permute_out", &permute_meta); }
