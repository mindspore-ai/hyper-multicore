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
int callback_status = -1;
int launch_calls = 0;
int status_reads = 0;
}  // namespace launch_test

extern "C" void HmcLaunchRouteMetadata(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, int32_t, int32_t, int32_t,
                                       int32_t, int32_t, int32_t, int32_t, int32_t, int32_t) {
  ++launch_test::launch_calls;
  launch_test::last_status = launch_test::injected_status;
}

extern "C" void HmcLaunchZeroCold(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, int64_t, int64_t, int64_t) {
  throw std::logic_error("unexpected cold-gradient launch");
}

extern "C" void HmcLaunchPermuteFromMapping(unsigned, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, int64_t,
                                            int64_t, int64_t) {
  throw std::logic_error("unexpected permutation launch");
}

int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  launch_test::injected_status = std::stoi(argv[1]);
  bool raised = false;
  try {
    auto counts = at::zeros({2, 4}, at::TensorOptions().dtype(at::kInt));
    using Route = std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, int64_t, int64_t, int64_t);
    auto route = c10::Dispatcher::singleton().findSchemaOrThrow("hyper_multicore::route_offsets", "").typed<Route>();
    route.redispatch(c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), counts, 0, 262144, 128);
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
  return 0;
}
