// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "csrc/mxfp8_byte_tensor_list.h"

namespace {
using ByteBuffers = hyper_multicore::mxfp8_binding::ByteTensorList;
static_assert(std::is_same_v<std::tuple_element_t<0, decltype(CopyTypesV2(std::declval<ByteBuffers&>()))>,
                             std::vector<TensorStructPtr>>);
static_assert(std::is_same_v<std::tuple_element_t<0, decltype(ConvertTypes(std::declval<ByteBuffers&>()))>,
                             aclTensorList*>);
void ValidateMoeBuffers(at::TensorList buffers, const at::Tensor& runtime, const at::Tensor& events,
                       const at::Tensor& diagnostic, const at::Tensor& launch) {
  TORCH_CHECK(!buffers.empty() && buffers.size() <= 64, "MXFP8 MegaMoE requires 1..64 buffers");
  for (const auto& buffer : buffers) {
    TORCH_CHECK(buffer.is_contiguous(),
                "MXFP8 task buffers must be contiguous; implicit copies are forbidden");
    TORCH_CHECK(buffer.device() == runtime.device(), "MXFP8 MoE buffers must use the runtime NPU device");
  }
  for (const auto& resource : {runtime, events, diagnostic, launch}) {
    TORCH_CHECK(resource.scalar_type() == at::kByte && resource.is_contiguous() && resource.dim() == 1,
                "MXFP8 MoE runtime resources must be contiguous UINT8 vectors");
    TORCH_CHECK(resource.device() == runtime.device(), "MXFP8 MoE resources must use one NPU device");
  }
  TORCH_CHECK(launch.numel() == 592, "MXFP8 MoE launch descriptor must contain 592 bytes");
}
void MegaMoeMxfp8(at::TensorList buffers, const at::Tensor& runtime, const at::Tensor& events,
                   const at::Tensor& diagnostic, const at::Tensor& launch) {
  ValidateMoeBuffers(buffers, runtime, events, diagnostic, launch);
  const hyper_multicore::mxfp8_binding::ByteTensorList byte_buffers{buffers, {}};
  RECORD_FUNCTION("hmc_prepare/mxfp8/native_submit", c10::ArrayRef<c10::IValue>{});
  int64_t count = buffers.size();
  HYPER_EXEC_NPU_CMD_EXT(aclnnHyperMegaMoeMxfp8, byte_buffers, runtime, events, diagnostic, launch, byte_buffers, count);
}
void MegaMoeMxfp8Grad(at::TensorList buffers, const at::Tensor& runtime, const at::Tensor& events,
                       const at::Tensor& diagnostic, const at::Tensor& launch) {
  ValidateMoeBuffers(buffers, runtime, events, diagnostic, launch);
  const hyper_multicore::mxfp8_binding::ByteTensorList byte_buffers{buffers, {}};
  RECORD_FUNCTION("hmc_prepare/mxfp8/native_submit", c10::ArrayRef<c10::IValue>{});
  int64_t count = buffers.size();
  HYPER_EXEC_NPU_CMD_EXT(aclnnHyperMegaMoeMxfp8Grad, byte_buffers, runtime, events, diagnostic, launch, byte_buffers, count);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def("mega_moe_mxfp8(Tensor(a!)[] buffers, Tensor(b!) runtime_config, Tensor(c!) events, "
        "Tensor(d!) diagnostic, Tensor launch_metadata) -> ()");
  m.def("mega_moe_mxfp8_grad(Tensor(a!)[] buffers, Tensor(b!) runtime_config, Tensor(c!) events, "
        "Tensor(d!) diagnostic, Tensor launch_metadata) -> ()");
  m.def("mega_moe_mxfp8_abi_version() -> int", []() -> int64_t { return 4; });
}
TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) {
  m.impl("mega_moe_mxfp8", &MegaMoeMxfp8);
  m.impl("mega_moe_mxfp8_grad", &MegaMoeMxfp8Grad);
}
