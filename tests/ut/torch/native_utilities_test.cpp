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

extern "C" void HmcLaunchRouteMetadata_arch2201(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
                                       int32_t, int32_t, int32_t,
                                       int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool attach_summary) {
  if (attach_summary) throw std::logic_error("unexpected attached summary for route_offsets");
  launch_test::selected_arch = 2201;
  ++launch_test::launch_calls;
  launch_test::last_status = launch_test::injected_status;
}

extern "C" void HmcLaunchRouteMetadata_arch3510(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
                                       int32_t, int32_t, int32_t,
                                       int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool attach_summary) {
  if (attach_summary) throw std::logic_error("unexpected attached summary for route_offsets");
  launch_test::selected_arch = 3510;
  ++launch_test::launch_calls;
  launch_test::last_status = launch_test::injected_status;
}

extern "C" void HmcLaunchMxRouteSummary_arch3510(unsigned, void*, const int32_t*, int64_t*, uint32_t, uint32_t, bool) {
  throw std::logic_error("unexpected standalone summary for route_offsets");
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
  if (argc != 3) {
    return 2;
  }
  launch_test::injected_status = std::stoi(argv[1]);
  launch_test::npu_arch = std::stoi(argv[2]);
  const bool supported = launch_test::npu_arch == 2201 || launch_test::npu_arch == 3510;
  bool raised = false;
  try {
    auto counts = at::zeros({2, 4}, at::TensorOptions().dtype(at::kInt));
    using Route = std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, int64_t, int64_t, int64_t);
    auto route = c10::Dispatcher::singleton().findSchemaOrThrow("hyper_multicore::route_offsets", "").typed<Route>();
    route.redispatch(c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), counts, 0, 262144, 128);
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
