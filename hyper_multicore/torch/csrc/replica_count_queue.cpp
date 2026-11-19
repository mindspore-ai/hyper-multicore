/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <dlfcn.h>
#include <climits>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace {
constexpr int kBlocks = 32;
constexpr int kAlignment = 64;
constexpr int kMaximumExperts = 512;
using PayloadLaunch = int (*)(void *, void *, void *, void *, int64_t, int, int, int, int, int);
using SummaryLaunch = int (*)(void *, void *, void *, void *, int, int);

struct CounterLibrary {
  void *handle;
  PayloadLaunch payload;
  SummaryLaunch summary;

  explicit CounterLibrary(const std::string &path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(handle != nullptr, "could not load replica count library: ", dlerror());
    auto version = reinterpret_cast<int (*)()>(dlsym(handle, "replica_counts_abi_version"));
    payload = reinterpret_cast<PayloadLaunch>(dlsym(handle, "launch_replica_route_payload"));
    summary = reinterpret_cast<SummaryLaunch>(dlsym(handle, "launch_replica_route_summary"));
    TORCH_CHECK(version != nullptr && version() == 2 && payload != nullptr && summary != nullptr,
                "queued replica counts require count ABI 2 with payload and summary launches");
    // Device code and queued launches retain these symbols until process exit.
  }
};

CounterLibrary &GetLibrary(const std::string &path) {
  static std::mutex mutex;
  static std::map<std::string, std::unique_ptr<CounterLibrary>> libraries;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = libraries.find(path);
  if (found == libraries.end()) {
    found = libraries.emplace(path, std::make_unique<CounterLibrary>(path)).first;
  }
  return *found->second;
}

void RecordUse(const at::Tensor &tensor, c10_npu::NPUStream stream) {
  if (tensor.defined()) {
    c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
}

at::Tensor RoutePayload(const at::Tensor &ids, int64_t experts, int64_t autograd_mask, const std::string &library_path) {
  TORCH_CHECK(ids.device().type() == c10::DeviceType::PrivateUse1 && ids.dim() == 2 &&
                  (ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong) &&
                  ids.size(1) >= 1 && ids.size(1) <= 8 && ids.numel() <= INT32_MAX &&
                  experts >= 1 && experts <= kMaximumExperts && autograd_mask >= INT32_MIN &&
                  autograd_mask <= INT32_MAX,
              "invalid queued replica payload geometry, dtype or controls");
  auto launch = GetLibrary(library_path).payload;
  c10::DeviceGuard guard(ids.device());
  auto source = ids.contiguous();
  const int stride = (experts + kAlignment) / kAlignment * kAlignment;
  auto partials = at::empty({kBlocks, stride}, ids.options().dtype(at::kLong));
  auto result = at::empty({experts + 3}, ids.options().dtype(at::kInt));
  const auto stream = c10_npu::getCurrentNPUStream();
  const auto native_stream = stream.stream(false);
  RecordUse(source, stream);
  RecordUse(partials, stream);
  RecordUse(result, stream);
  auto run = [=]() -> int {
    return launch(native_stream, source.data_ptr(), partials.data_ptr(), result.data_ptr(), source.size(0),
                  source.size(1), experts, stride, autograd_mask, source.element_size());
  };
  at_npu::native::OpCommand::RunOpApi("HmcReplicaRoutePayload", run);
  return result;
}

at::Tensor RouteSummary(const at::Tensor &payload, const c10::optional<at::Tensor> &previous,
                        const std::string &library_path) {
  TORCH_CHECK(payload.device().type() == c10::DeviceType::PrivateUse1 && payload.dim() == 2 &&
                  payload.scalar_type() == at::kInt && payload.is_contiguous() && payload.size(0) > 0 &&
                  payload.size(1) > 3 && payload.size(1) - 3 <= kMaximumExperts &&
                  (payload.size(1) - 3) % payload.size(0) == 0,
              "invalid queued replica summary geometry or dtype");
  auto old = previous.value_or(at::Tensor());
  TORCH_CHECK(!old.defined() || (old.device() == payload.device() && old.scalar_type() == payload.scalar_type() &&
                                   old.sizes() == payload.sizes() && old.is_contiguous()),
              "queued replica summary requires an identical previous table layout and device");
  auto launch = GetLibrary(library_path).summary;
  c10::DeviceGuard guard(payload.device());
  auto result = at::empty({payload.size(0), 8}, payload.options().dtype(at::kLong));
  const auto stream = c10_npu::getCurrentNPUStream();
  const auto native_stream = stream.stream(false);
  RecordUse(payload, stream);
  RecordUse(old, stream);
  RecordUse(result, stream);
  auto run = [=]() -> int {
    return launch(native_stream, payload.data_ptr(), old.defined() ? old.data_ptr() : nullptr, result.data_ptr(),
                  payload.size(0), payload.size(1) - 3);
  };
  at_npu::native::OpCommand::RunOpApi("HmcReplicaRouteSummary", run);
  return result;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def("replica_count_queue_version() -> int", []() -> int64_t { return 1; });
  m.def("replica_route_payload(Tensor ids, int experts, int autograd_mask, str library_path) -> Tensor");
  m.def("replica_route_summary(Tensor payload, Tensor? previous, str library_path) -> Tensor");
}
TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) {
  m.impl("replica_route_payload", RoutePayload);
  m.impl("replica_route_summary", RouteSummary);
}
