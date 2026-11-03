// Copyright 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
// Compile/link the installed framework boundary without executing device operations.
#include <c10/core/StorageImpl.h>
#include <torch/torch.h>
#include "cached_op_api.h"
#include "aclnnop/aclnn_silu.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#ifndef EXEC_NPU_CMD_V1_EXT
#error "Installed op-plugin lacks EXEC_NPU_CMD_V1_EXT"
#endif
#ifndef EXEC_NPU_CMD_V2_EXT
#error "Installed op-plugin lacks EXEC_NPU_CMD_V2_EXT"
#endif

void hyper_multicore_framework_probe(at::Tensor tensor) {
  void *execute = nullptr;
  void *workspace = nullptr;
  GetApiFunc("hyper_multicore_probe", "hyper_multicore_probeGetWorkspaceSize", execute, workspace);
  (void)OpApiGetTaskQueueEnable();
  (void)c10_npu::getCurrentNPUStream(0).stream();
  auto *storage = tensor.storage().unsafeGetStorageImpl();
  storage->set_data_ptr(at::DataPtr(nullptr, nullptr, nullptr, tensor.device()));
  storage->set_nbytes(0);
  (void)at_npu::native::from_blob(nullptr, {0}, [](void *) {}, tensor.options(), c10::optional<c10::Device>(tensor.device()));
  HYPER_EXEC_NPU_CMD_EXT(aclnnSilu, tensor, tensor);
}
