// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Framework bindings for component-owned MoE utility providers.
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <acl/acl.h>
#include <cstdint>
#include <tuple>
#include <vector>

#include "ATen/ATen.h"
#include "ATen/MemoryOverlap.h"
extern "C" void HmcLaunchZeroCold_arch2201(unsigned blocks, void* stream, uint8_t* first, uint8_t* second,
                                  uint8_t* cumulative, int64_t first_bytes, int64_t second_bytes, int64_t experts);
extern "C" void HmcLaunchZeroCold_arch3510(unsigned blocks, void* stream, uint8_t* first, uint8_t* second,
                                  uint8_t* cumulative, int64_t first_bytes, int64_t second_bytes, int64_t experts);
#if HMC_UTILITY_ARCH2201
#define HMC_UTILITY_ARCH22(name) (&name##_arch2201)
#else
#define HMC_UTILITY_ARCH22(name) (static_cast<decltype(&name##_arch2201)>(nullptr))
#endif
#if HMC_UTILITY_ARCH3510
#define HMC_UTILITY_ARCH35(name) (&name##_arch3510)
#else
#define HMC_UTILITY_ARCH35(name) (static_cast<decltype(&name##_arch3510)>(nullptr))
#endif
namespace {
int64_t UtilityArchitecture(int32_t device) {
  // Hardware identity is stable for a process-local device; route contents are not cached.
  thread_local int32_t previous_device = INT32_MIN;
  thread_local int64_t previous_arch = 0;
  if (device == previous_device) return previous_arch;
  int64_t npu_arch = 0;
  // Older PTA-bundled ACL headers omit the enum name; CANN's public attribute
  // number matches _loader._ACL_DEV_ATTR_NPU_ARCH and is supported by the runtime.
  constexpr auto kNpuArchAttribute = static_cast<aclrtDevAttr>(601);
  const auto status = aclrtGetDeviceInfo(device, kNpuArchAttribute, &npu_arch);
  TORCH_CHECK(status == ACL_SUCCESS, "cannot query NPU architecture for device ", device, ": ", status);
  previous_device = device;
  previous_arch = npu_arch;
  return npu_arch;
}
template <typename Fn> Fn SelectUtility(int32_t device, Fn arch22, Fn arch35) {
  const auto npu_arch = UtilityArchitecture(device);
  switch (npu_arch) {
    case 2201:
      TORCH_CHECK(arch22 != nullptr, "MoE utility provider is missing arch22 (Atlas A2/A3 series)");
      return arch22;
    case 3510:
      TORCH_CHECK(arch35 != nullptr, "MoE utility provider is missing arch35 (Ascend950)");
      return arch35;
    default:
      TORCH_CHECK(false, "unsupported NPU architecture for MoE utility: ", npu_arch);
  }
}
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
    auto launch = SelectUtility(device, HMC_UTILITY_ARCH22(HmcLaunchZeroCold), HMC_UTILITY_ARCH35(HmcLaunchZeroCold));
    launch(cores, stream, static_cast<uint8_t*>(first.data_ptr()), static_cast<uint8_t*>(second.data_ptr()),
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

extern "C" void HmcLaunchRouteMetadata_arch2201(unsigned blocks, void* stream, uint8_t* counts, uint8_t* offsets,
                                       uint8_t* sizes, int32_t ep, int32_t experts,
                                       int32_t rank, int32_t stride, int32_t group,
                                       int32_t programs, int32_t bytes, int32_t row_stride, int32_t hidden_size);
extern "C" void HmcLaunchRouteMetadata_arch3510(unsigned blocks, void* stream, uint8_t* counts, uint8_t* offsets,
                                       uint8_t* sizes, int32_t ep, int32_t experts,
                                       int32_t rank, int32_t stride, int32_t group,
                                       int32_t programs, int32_t bytes, int32_t row_stride, int32_t hidden_size);
extern "C" void HmcLaunchRoutePreparation_arch2201(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
extern "C" void HmcLaunchRoutePreparation_arch3510(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
namespace {
using PreparedRoute = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
PreparedRoute BuildRouteMetadata(const at::Tensor& input, int64_t rank, int64_t ub,
                                int64_t hidden_size, bool with_summary) {
  TORCH_CHECK(input.dim() == 2 && input.scalar_type() == at::kInt && input.stride(1) == 1,
              "expected row-contiguous int32 [source,expert] or [source,header+expert]");
  const int64_t header = with_summary ? 3 : 0;
  const int64_t p = input.size(0), e = input.size(1) - header;
  TORCH_CHECK(p > 0 && e > 0 && e % p == 0 && rank >= 0 && rank < p && p <= INT32_MAX && e <= INT32_MAX / p,
              "invalid geometry");
  const auto row_stride = input.stride(0);
  TORCH_CHECK(row_stride >= e + header && row_stride <= INT32_MAX, "invalid count row stride");
  const auto span = (p - 1) * row_stride + e + header;
  TORCH_CHECK(span <= INT32_MAX, "count matrix span exceeds the native bound");
  const auto local = e / p, stride = (e + 31) / 32 * 32;
  auto aligned = [](int64_t x) { return (x + 7) / 8 * 8; };
  const auto bytes = (aligned(span) + 3 * stride + 2 * aligned(local)) * 4;
  TORCH_CHECK(ub > 0 && bytes <= ub / 4 * 3 && bytes <= INT32_MAX, "route offset scratch exceeds device UB");
  TORCH_CHECK(hidden_size > 0 && hidden_size <= INT32_MAX, "invalid hidden size");
  const int64_t copies = hidden_size == 1 ? 1 : 2;
  c10::DeviceGuard guard(input.device());
  auto offsets = at::empty({copies * (4 * stride + local)}, input.options().dtype(at::kLong));
  auto sizes = at::empty({copies * 2 * stride}, input.options());
  at::Tensor summary;
  if (with_summary) summary = at::empty({p, 8}, input.options().dtype(at::kLong));
  int64_t group = 1;
  while ((group * local) % 16 != 0) ++group;
  const auto programs = (p + group - 1) / group;
  const auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto run = [=]() -> int {
    if (with_summary) {
      auto launch = SelectUtility(input.get_device(), HMC_UTILITY_ARCH22(HmcLaunchRoutePreparation),
                                  HMC_UTILITY_ARCH35(HmcLaunchRoutePreparation));
      launch(2 * programs, stream, static_cast<uint8_t*>(input.data_ptr()), static_cast<uint8_t*>(offsets.data_ptr()),
             static_cast<uint8_t*>(sizes.data_ptr()), static_cast<uint8_t*>(summary.data_ptr()),
             p, e, rank, stride, group, programs, bytes, row_stride, hidden_size);
    } else {
      auto launch = SelectUtility(input.get_device(), HMC_UTILITY_ARCH22(HmcLaunchRouteMetadata),
                                  HMC_UTILITY_ARCH35(HmcLaunchRouteMetadata));
      launch(2 * programs, stream, static_cast<uint8_t*>(input.data_ptr()), static_cast<uint8_t*>(offsets.data_ptr()),
             static_cast<uint8_t*>(sizes.data_ptr()), p, e, rank, stride, group, programs, bytes,
             row_stride, hidden_size);
    }
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
  };
  at_npu::native::OpCommand::RunOpApi(with_summary ? "HmcPrepareRoute" : "HmcRouteOffsets", run);
  return {offsets, sizes, summary};
}
std::tuple<at::Tensor, at::Tensor> RouteOffsets(const at::Tensor& counts, int64_t rank, int64_t ub,
                                              int64_t hidden_size) {
  auto result = BuildRouteMetadata(counts, rank, ub, hidden_size, false);
  return {std::get<0>(result), std::get<1>(result)};
}
PreparedRoute PrepareRoute(const at::Tensor& payload, int64_t rank, int64_t ub, int64_t hidden_size) {
  return BuildRouteMetadata(payload, rank, ub, hidden_size, true);
}
std::vector<at::Tensor> Bf16ForwardBuffers(const at::Tensor& like, int64_t rows,
                                         int64_t intermediate, bool include_dispatch) {
  TORCH_CHECK(like.dim() == 2 && rows >= 0 && intermediate > 0 && intermediate <= INT64_MAX / 2,
              "invalid BF16 forward buffer dimensions");
  TORCH_CHECK(like.scalar_type() == at::kBFloat16, "BF16 forward buffers require BF16 input");
  const auto options = like.options();
  std::vector<at::Tensor> result;
  if (include_dispatch) result.push_back(at::empty({rows, like.size(1)}, options));
  result.push_back(at::empty({rows, 2 * intermediate}, options));
  result.push_back(at::empty({rows, intermediate}, options));
  result.push_back(at::empty({rows, like.size(1)}, options));
  return result;
}
std::vector<at::Tensor> Bf16BackwardBuffers(const at::Tensor& like, const at::Tensor& weight1,
    const at::Tensor& weight2, int64_t rows, int64_t intermediate, bool include_dx,
    bool zero_weights, const std::optional<at::ScalarType> weight_grad_dtype) {
  TORCH_CHECK(like.dim() == 2 && rows >= 0 && intermediate > 0 && intermediate <= INT64_MAX / 2,
              "invalid BF16 backward buffer dimensions");
  TORCH_CHECK(like.scalar_type() == at::kBFloat16 && weight1.device() == like.device() &&
              weight2.device() == like.device(), "BF16 backward operands must share a device");
  const auto options = like.options();
  const auto dtype = weight_grad_dtype.value_or(weight1.scalar_type());
  auto first = zero_weights ? at::zeros_like(weight1, weight1.options().dtype(dtype))
                           : at::empty_like(weight1, weight1.options().dtype(dtype), at::MemoryFormat::Contiguous);
  auto second = zero_weights ? at::zeros_like(weight2, weight2.options().dtype(dtype))
                            : at::empty_like(weight2, weight2.options().dtype(dtype), at::MemoryFormat::Contiguous);
  std::vector<at::Tensor> result{first, second, at::empty({rows, intermediate}, options),
                                at::empty({rows, 2 * intermediate}, options)};
  if (include_dx) result.push_back(at::empty({rows, like.size(1)}, options));
  return result;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def("route_offsets(Tensor counts, int rank, int ub, int hidden_size=1) -> (Tensor, Tensor)");
  m.def("prepare_route(Tensor payload, int rank, int ub, int hidden_size) -> (Tensor, Tensor, Tensor)");
  m.def("bf16_forward_buffers(Tensor like, int rows, int intermediate, bool include_dispatch) -> Tensor[]");
  m.def("bf16_backward_buffers(Tensor like, Tensor weight1, Tensor weight2, int rows, int intermediate, "
        "bool include_dx, bool zero_weights, ScalarType? weight_grad_dtype=None) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) {
  m.impl("route_offsets", RouteOffsets);
  m.impl("prepare_route", PrepareRoute);
  m.impl("bf16_forward_buffers", Bf16ForwardBuffers);
  m.impl("bf16_backward_buffers", Bf16BackwardBuffers);
}
TORCH_LIBRARY_IMPL(hyper_multicore, CPU, m) {
  m.impl("bf16_forward_buffers", Bf16ForwardBuffers);
  m.impl("bf16_backward_buffers", Bf16BackwardBuffers);
}
TORCH_LIBRARY_IMPL(hyper_multicore, Meta, m) {
  m.impl("bf16_forward_buffers", Bf16ForwardBuffers);
  m.impl("bf16_backward_buffers", Bf16BackwardBuffers);
}

extern "C" void HmcLaunchPermuteFromMapping_arch2201(unsigned blocks, void* stream, uint8_t* tokens, uint8_t* mapping,
                                            uint8_t* probabilities, uint8_t* output, uint8_t* output_probabilities,
                                            int64_t token_count, int64_t row_bytes, int64_t top_k);
extern "C" void HmcLaunchPermuteFromMapping_arch3510(unsigned blocks, void* stream, uint8_t* tokens, uint8_t* mapping,
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
    auto launch = SelectUtility(tokens.get_device(), HMC_UTILITY_ARCH22(HmcLaunchPermuteFromMapping),
                                HMC_UTILITY_ARCH35(HmcLaunchPermuteFromMapping));
    launch(cores, stream, static_cast<uint8_t*>(tokens.data_ptr()), static_cast<uint8_t*>(mapping.data_ptr()),
           static_cast<uint8_t*>(probabilities.data_ptr()), static_cast<uint8_t*>(output.data_ptr()),
           static_cast<uint8_t*>(output_probabilities.data_ptr()), tokens.size(0),
           tokens.size(1) * tokens.element_size(),
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
