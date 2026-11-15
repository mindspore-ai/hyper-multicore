// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Framework-owned expert-gradient and token-permutation utilities.
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <acl/acl.h>
#include <cstdint>
#include <tuple>

#include "ATen/ATen.h"
#include "ATen/MemoryOverlap.h"
extern "C" void HmcLaunchZeroCold(unsigned blocks, void* stream, uint8_t* first, uint8_t* second, uint8_t* cumulative,
                                  int64_t first_bytes, int64_t second_bytes, int64_t experts);
namespace {
using Outputs = std::tuple<at::Tensor&, at::Tensor&>;

void Check(const at::Tensor& first, const at::Tensor& second, const at::Tensor& groups) {
  TORCH_CHECK(first.dim() == 3 && second.dim() == 3 && groups.dim() == 1,
              "expert gradients must be [E,*,*] and cumulative groups must be [E]");
  TORCH_CHECK(first.size(0) == groups.numel() && second.size(0) == groups.numel(),
              "gradient expert count must match cumulative groups");
  TORCH_CHECK(first.scalar_type() == second.scalar_type() &&
                  (first.scalar_type() == at::kBFloat16 || first.scalar_type() == at::kFloat),
              "expert gradients must share BF16 or FP32 dtype");
  TORCH_CHECK(groups.scalar_type() == at::kLong, "cumulative groups must be INT64");
  TORCH_CHECK(first.device() == second.device() && first.device() == groups.device(),
              "expert gradients and groups must share a device");
  TORCH_CHECK(first.is_contiguous() && second.is_contiguous() && groups.is_contiguous(),
              "expert gradient outputs and cumulative groups must be contiguous");
}

Outputs ZeroEmptyNpu(at::Tensor& first, at::Tensor& second, const at::Tensor& groups) {
  Check(first, second, groups);
  TORCH_CHECK(first.device().type() == c10::DeviceType::PrivateUse1, "expert gradients must be on NPU");
  at::assert_no_overlap(first, second);
  at::assert_no_overlap(first, groups);
  at::assert_no_overlap(second, groups);
  if (groups.numel() == 0 || (first.numel() == 0 && second.numel() == 0)) {
    return {first, second};
  }
  const c10::DeviceGuard guard(first.device());
  const int32_t device = first.get_device();
  int64_t cores = 0;
  const auto status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
  TORCH_CHECK(status == ACL_SUCCESS && cores > 0, "cannot query current NPU vector-core count: ", status);
  const int64_t experts = groups.numel();
  const int64_t first_bytes = first.numel() * first.element_size() / experts;
  const int64_t second_bytes = second.numel() * second.element_size() / experts;
  // RunOpApi submits this launch after the preceding Torch operations on the
  // same task queue. No host synchronization or group-list readback is needed.
  auto stream = c10_npu::getCurrentNPUStream(device).stream(false);
  auto run = [=]() -> int {
    HmcLaunchZeroCold(cores, stream, static_cast<uint8_t*>(first.data_ptr()), static_cast<uint8_t*>(second.data_ptr()),
                      static_cast<uint8_t*>(groups.data_ptr()), first_bytes, second_bytes, experts);
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
  };
  at_npu::native::OpCommand::RunOpApi("HmcZeroEmptyExpertGrad", run);
  return {first, second};
}

Outputs ZeroEmptyMeta(at::Tensor& first, at::Tensor& second, const at::Tensor& groups) {
  Check(first, second, groups);
  return {first, second};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def("zero_empty_expert_grad_(Tensor(a!) first, Tensor(b!) second, Tensor groups) -> (Tensor(a!), Tensor(b!))");
}
TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) { m.impl("zero_empty_expert_grad_", &ZeroEmptyNpu); }
TORCH_LIBRARY_IMPL(hyper_multicore, Meta, m) { m.impl("zero_empty_expert_grad_", &ZeroEmptyMeta); }

extern "C" void HmcLaunchRouteMetadata(unsigned blocks, void* stream, uint8_t* counts, uint8_t* offsets, uint8_t* sizes,
                                       int32_t ep, int32_t experts, int32_t rank, int32_t stride, int32_t group,
                                       int32_t programs, int32_t bytes, int32_t row_stride, int32_t hidden_size);
namespace {
std::tuple<at::Tensor, at::Tensor> RouteOffsets(const at::Tensor& counts, int64_t rank, int64_t ub,
                                                int64_t hidden_size) {
  TORCH_CHECK(counts.dim() == 2 && counts.scalar_type() == at::kInt && counts.stride(1) == 1,
              "expected row-contiguous int32 [source,expert]");
  const int64_t p = counts.size(0), e = counts.size(1);
  TORCH_CHECK(p > 0 && e > 0 && e % p == 0 && rank >= 0 && rank < p && p <= INT32_MAX && e <= INT32_MAX / p,
              "invalid geometry");
  const auto row_stride = counts.stride(0);
  TORCH_CHECK(row_stride >= e && row_stride <= INT32_MAX, "invalid count row stride");
  const auto span = (p - 1) * row_stride + e;
  TORCH_CHECK(span <= INT32_MAX, "count matrix span exceeds the native bound");
  const auto local = e / p, stride = (e + 31) / 32 * 32;
  auto aligned = [](int64_t x) { return (x + 7) / 8 * 8; };
  const auto bytes = (aligned(span) + 3 * stride + 2 * aligned(local)) * 4;
  TORCH_CHECK(ub > 0 && bytes <= ub / 4 * 3 && bytes <= INT32_MAX, "route offset scratch exceeds device UB");
  TORCH_CHECK(hidden_size > 0 && hidden_size <= INT32_MAX, "invalid hidden size");
  const int64_t copies = hidden_size == 1 ? 1 : 2;
  c10::DeviceGuard guard(counts.device());
  auto offsets = at::empty({copies * (4 * stride + local)}, counts.options().dtype(at::kLong));
  auto sizes = at::empty({copies * 2 * stride}, counts.options());
  int64_t group = 1;
  while ((group * local) % 16 != 0) ++group;
  const auto programs = (p + group - 1) / group;
  const auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto run = [=]() -> int {
    HmcLaunchRouteMetadata(2 * programs, stream, (uint8_t*)counts.data_ptr(), (uint8_t*)offsets.data_ptr(),
                           (uint8_t*)sizes.data_ptr(), p, e, rank, stride, group, programs, bytes, row_stride,
                           hidden_size);
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
  };
  at_npu::native::OpCommand::RunOpApi("HmcRouteOffsets", run);
  return {offsets, sizes};
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def("route_offsets(Tensor counts, int rank, int ub, int hidden_size=1) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) { m.impl("route_offsets", RouteOffsets); }

extern "C" void HmcLaunchPermuteFromMapping(unsigned blocks, void* stream, uint8_t* tokens, uint8_t* mapping,
                                            uint8_t* probabilities, uint8_t* output, uint8_t* output_probabilities,
                                            int64_t token_count, int64_t row_bytes, int64_t top_k);
namespace {
inline void CheckPermutationShapes(const at::Tensor& tokens, const at::Tensor& mapping, const at::Tensor& probabilities,
                                   const at::Tensor& output, const at::Tensor& output_probabilities) {
  TORCH_CHECK(
      tokens.dim() == 2 && mapping.dim() == 1 && probabilities.dim() == 2 && output.dim() == 2 &&
          output_probabilities.dim() == 1,
      "expected tokens/output [rows, hidden], mapping/output probabilities [slots], probabilities [tokens, top_k]");
  TORCH_CHECK(tokens.size(1) > 0 && probabilities.size(1) > 0 && probabilities.size(0) == tokens.size(0) &&
                  output.size(0) == probabilities.numel() && output.size(1) == tokens.size(1) &&
                  mapping.numel() == probabilities.numel() && output_probabilities.numel() == mapping.numel(),
              "permutation shapes must describe the same complete Top-K route");
}

void CheckPermutation(const at::Tensor& tokens, const at::Tensor& mapping, const at::Tensor& probabilities,
                      const at::Tensor& output, const at::Tensor& output_probabilities) {
  CheckPermutationShapes(tokens, mapping, probabilities, output, output_probabilities);
  TORCH_CHECK(mapping.scalar_type() == at::kInt && probabilities.scalar_type() == at::kFloat &&
                  output_probabilities.scalar_type() == at::kFloat,
              "mapping must be INT32 and probabilities must be FP32");
  TORCH_CHECK(output.scalar_type() == tokens.scalar_type() &&
                  (tokens.scalar_type() == at::kHalf || tokens.scalar_type() == at::kBFloat16 ||
                   tokens.scalar_type() == at::kFloat),
              "token rows must share FP16, BF16 or FP32 dtype");
  TORCH_CHECK(tokens.device() == mapping.device() && tokens.device() == probabilities.device() &&
                  tokens.device() == output.device() && tokens.device() == output_probabilities.device(),
              "permutation inputs and outputs must share a device");
  TORCH_CHECK(tokens.is_contiguous() && mapping.is_contiguous() && probabilities.is_contiguous() &&
                  output.is_contiguous() && output_probabilities.is_contiguous(),
              "permutation inputs and outputs must be contiguous");
}

Outputs PermuteFromMappingNpu(const at::Tensor& tokens, const at::Tensor& mapping, const at::Tensor& probabilities,
                              at::Tensor& output, at::Tensor& output_probabilities) {
  CheckPermutation(tokens, mapping, probabilities, output, output_probabilities);
  TORCH_CHECK(tokens.device().type() == c10::DeviceType::PrivateUse1, "token rows must be on NPU");
  for (const auto& input : {tokens, mapping, probabilities}) {
    at::assert_no_overlap(output, input);
    at::assert_no_overlap(output_probabilities, input);
  }
  at::assert_no_overlap(output, output_probabilities);
  if (tokens.size(0) == 0) {
    return {output, output_probabilities};
  }
  const c10::DeviceGuard guard(tokens.device());
  int64_t cores = 0;
  const auto status = aclrtGetDeviceInfo(tokens.get_device(), ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
  TORCH_CHECK(status == ACL_SUCCESS && cores > 0, "cannot query current NPU vector-core count: ", status);
  auto stream = c10_npu::getCurrentNPUStream(tokens.get_device()).stream(false);
  auto run = [=]() -> int {
    HmcLaunchPermuteFromMapping(
        cores, stream, static_cast<uint8_t*>(tokens.data_ptr()), static_cast<uint8_t*>(mapping.data_ptr()),
        static_cast<uint8_t*>(probabilities.data_ptr()), static_cast<uint8_t*>(output.data_ptr()),
        static_cast<uint8_t*>(output_probabilities.data_ptr()), tokens.size(0), tokens.size(1) * tokens.element_size(),
        probabilities.size(1));
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
  };
  at_npu::native::OpCommand::RunOpApi("HmcPermuteFromMapping", run);
  return {output, output_probabilities};
}

Outputs PermuteFromMappingMeta(const at::Tensor& tokens, const at::Tensor& mapping, const at::Tensor& probabilities,
                               at::Tensor& output, at::Tensor& output_probabilities) {
  CheckPermutation(tokens, mapping, probabilities, output, output_probabilities);
  return {output, output_probabilities};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def(
      "moe_token_permute_from_mapping_out(Tensor tokens, Tensor mapping, Tensor probabilities, "
      "Tensor(a!) output, Tensor(b!) output_probabilities) -> (Tensor(a!), Tensor(b!))");
}
TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) {
  m.impl("moe_token_permute_from_mapping_out", &PermuteFromMappingNpu);
}
TORCH_LIBRARY_IMPL(hyper_multicore, Meta, m) { m.impl("moe_token_permute_from_mapping_out", &PermuteFromMappingMeta); }
