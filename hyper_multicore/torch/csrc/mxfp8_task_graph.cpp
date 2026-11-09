// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "csrc/cached_op_api.h"

namespace {
void ValidateTaskGraph(at::TensorList buffers, const at::Tensor& runtime, const at::Tensor& events,
                       const at::Tensor& diagnostic, const at::Tensor& launch) {
  TORCH_CHECK(!buffers.empty() && buffers.size() <= 64, "MX graph requires 1..64 buffers");
  for (const auto& buffer : buffers) {
    TORCH_CHECK(buffer.scalar_type() == at::kByte && buffer.is_contiguous() && buffer.dim() == 1,
                "MX buffers must be contiguous UINT8 vectors");
    TORCH_CHECK(buffer.device() == runtime.device(), "MX buffers must use the runtime NPU device");
  }
  for (const auto& resource : {runtime, events, diagnostic, launch}) {
    TORCH_CHECK(resource.scalar_type() == at::kByte && resource.is_contiguous() && resource.dim() == 1,
                "MX runtime resources must be contiguous UINT8 vectors");
    TORCH_CHECK(resource.device() == runtime.device(), "MX resources must use one NPU device");
  }
  TORCH_CHECK(launch.numel() == 592, "MX launch descriptor must contain 592 bytes");
}
void Mxfp8TaskGraph(at::TensorList buffers, const at::Tensor& runtime, const at::Tensor& events,
                   const at::Tensor& diagnostic, const at::Tensor& launch) {
  ValidateTaskGraph(buffers, runtime, events, diagnostic, launch);
  int64_t count = buffers.size();
  HYPER_EXEC_NPU_CMD_EXT(aclnnHyperMegaMoeMxfp8, buffers, runtime, events, diagnostic, launch, buffers, count);
}
void Mxfp8TaskGraphGrad(at::TensorList buffers, const at::Tensor& runtime, const at::Tensor& events,
                       const at::Tensor& diagnostic, const at::Tensor& launch) {
  ValidateTaskGraph(buffers, runtime, events, diagnostic, launch);
  int64_t count = buffers.size();
  HYPER_EXEC_NPU_CMD_EXT(aclnnHyperMegaMoeMxfp8Grad, buffers, runtime, events, diagnostic, launch, buffers, count);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def("mxfp8_task_graph(Tensor(a!)[] buffers, Tensor(b!) runtime_config, Tensor(c!) events, "
        "Tensor(d!) diagnostic, Tensor launch_metadata) -> ()");
  m.def("mxfp8_task_graph_grad(Tensor(a!)[] buffers, Tensor(b!) runtime_config, Tensor(c!) events, "
        "Tensor(d!) diagnostic, Tensor launch_metadata) -> ()");
  m.def("mxfp8_task_graph_version() -> int", []() -> int64_t { return 2; });
}
TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) {
  m.impl("mxfp8_task_graph", &Mxfp8TaskGraph);
  m.impl("mxfp8_task_graph_grad", &Mxfp8TaskGraphGrad);
}
