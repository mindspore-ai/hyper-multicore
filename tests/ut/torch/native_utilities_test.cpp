// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Execute the production RouteOffsets callback with host-only launch failures.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ATen/ATen.h"
#include "ATen/core/dispatch/Dispatcher.h"

namespace launch_test {
int injected_status = 0;
int last_status = 0;
int npu_arch = 2201;
int selected_arch = 0;
int callback_status = -1;
int launch_calls = 0;
int status_reads = 0;
}  // namespace launch_test

extern "C" void HmcLaunchRouteMetadata_arch2201(unsigned, void*, uint8_t*, uint8_t*, uint8_t*,
                                       int32_t, int32_t, int32_t,
                                       int32_t, int32_t, int32_t, int32_t, int32_t, int32_t) {
  launch_test::selected_arch = 2201;
  ++launch_test::launch_calls;
  launch_test::last_status = launch_test::injected_status;
}

extern "C" void HmcLaunchRouteMetadata_arch3510(unsigned, void*, uint8_t*, uint8_t*, uint8_t*,
                                       int32_t, int32_t, int32_t,
                                       int32_t, int32_t, int32_t, int32_t, int32_t, int32_t) {
  launch_test::selected_arch = 3510;
  ++launch_test::launch_calls;
  launch_test::last_status = launch_test::injected_status;
}

extern "C" void HmcLaunchRoutePreparation_arch2201(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t) {
  launch_test::selected_arch = 2201;
  ++launch_test::launch_calls;
  launch_test::last_status = launch_test::injected_status;
}
extern "C" void HmcLaunchRoutePreparation_arch3510(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t) {
  launch_test::selected_arch = 3510;
  ++launch_test::launch_calls;
  launch_test::last_status = launch_test::injected_status;
}

void CheckAllocations() {
  const auto options = at::TensorOptions().dtype(at::kBFloat16);
  auto like = at::empty({7, 4}, options);
  using Forward = std::vector<at::Tensor>(const at::Tensor&, int64_t, int64_t, bool);
  auto forward = c10::Dispatcher::singleton().findSchemaOrThrow("hyper_multicore::bf16_forward_buffers", "")
                     .typed<Forward>();
  for (const int64_t rows : {0, 3, 9}) {
    for (const bool dispatch : {false, true}) {
      const auto values = forward.call(like, rows, 8, dispatch);
      TORCH_CHECK(values.size() == (dispatch ? 4 : 3), "wrong forward allocation count");
      const auto index = dispatch ? 1 : 0;
      TORCH_CHECK(values[index].size(0) == rows && values[index].size(1) == 16 &&
                  values[index + 1].size(0) == rows && values[index + 1].size(1) == 8 &&
                  values[index + 2].size(0) == rows && values[index + 2].size(1) == 4,
                  "forward allocations do not match actual received rows");
      for (const auto& value : values) {
        TORCH_CHECK(value.storage().nbytes() == value.numel() * value.element_size(), "unexpected capacity slack");
      }
    }
  }
  const auto previous = forward.call(like, 3, 8, true);
  previous[0].fill_(2);
  const auto next = forward.call(like, 3, 8, true);
  next[0].fill_(5);
  TORCH_CHECK(previous[0].eq(2).all().item<bool>(), "a later invocation overwrote saved expert input");
  auto first = at::empty({2, 16, 4}, options), second = at::empty({2, 4, 8}, options);
  using Backward = std::vector<at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&,
      int64_t, int64_t, bool, bool, std::optional<at::ScalarType>);
  auto backward = c10::Dispatcher::singleton().findSchemaOrThrow("hyper_multicore::bf16_backward_buffers", "")
                      .typed<Backward>();
  const auto gradients = backward.call(like, first, second, 3, 8, true, true, at::kFloat);
  TORCH_CHECK(gradients.size() == 5 && gradients[0].scalar_type() == at::kFloat &&
              gradients[0].eq(0).all().item<bool>() && gradients[1].eq(0).all().item<bool>() &&
              gradients[2].size(0) == 3 && gradients[3].size(0) == 3 && gradients[4].size(0) == 3,
              "backward allocation or zero-initialization mismatch");
}

extern "C" void HmcLaunchZeroCold_arch2201(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, int64_t, int64_t, int64_t) {
  launch_test::selected_arch = 2201;
  throw std::logic_error("unexpected cold-gradient launch");
}

extern "C" void HmcLaunchZeroCold_arch3510(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, int64_t, int64_t, int64_t) {
  launch_test::selected_arch = 3510;
  throw std::logic_error("unexpected cold-gradient launch");
}

extern "C" void HmcLaunchPermuteFromMapping_arch2201(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
                                            int64_t, int64_t, int64_t) {
  launch_test::selected_arch = 2201;
  throw std::logic_error("unexpected permutation launch");
}

extern "C" void HmcLaunchPermuteFromMapping_arch3510(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
                                            int64_t, int64_t, int64_t) {
  launch_test::selected_arch = 3510;
  throw std::logic_error("unexpected permutation launch");
}

int main(int argc, char** argv) {
  if (argc != 4) {
    return 2;
  }
  launch_test::injected_status = std::stoi(argv[1]);
  launch_test::npu_arch = std::stoi(argv[2]);
  CheckAllocations();
  const bool supported = launch_test::npu_arch == 2201 || launch_test::npu_arch == 3510;
  bool raised = false;
  try {
    auto counts = at::zeros({2, 4}, at::TensorOptions().dtype(at::kInt));
    if (std::string(argv[3]) == "prepare") {
      using Route = std::tuple<at::Tensor, at::Tensor, at::Tensor>(const at::Tensor&, int64_t, int64_t, int64_t);
      auto route = c10::Dispatcher::singleton().findSchemaOrThrow("hyper_multicore::prepare_route", "").typed<Route>();
      auto payload = at::zeros({2, 7}, counts.options());
      route.redispatch(c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), payload, 0, 262144, 128);
    } else {
      using Route = std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, int64_t, int64_t, int64_t);
      auto route = c10::Dispatcher::singleton().findSchemaOrThrow("hyper_multicore::route_offsets", "").typed<Route>();
      route.redispatch(c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), counts, 0, 262144, 128);
    }
  } catch (const c10::Error& error) {
    if (!supported && std::string(error.what()).find("unsupported NPU architecture for MoE utility") !=
                          std::string::npos && launch_test::launch_calls == 0 && launch_test::status_reads == 0) {
      return 0;
    }
    std::cerr << error.what() << '\n';
    return 1;
  } catch (const std::runtime_error& error) {
    raised = true;
    if (std::string(error.what()) != "launch status " + std::to_string(launch_test::injected_status)) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
  if (launch_test::launch_calls != 1 || launch_test::status_reads != 1 ||
      launch_test::callback_status != launch_test::injected_status || raised != (launch_test::injected_status != 0)) {
    std::cerr << "launch status was not propagated to RunOpApi\n";
    return 1;
  }
  if (!supported || launch_test::selected_arch != launch_test::npu_arch) return 9;
  return 0;
}
