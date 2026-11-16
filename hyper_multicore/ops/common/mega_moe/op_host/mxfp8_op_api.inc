// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
// Expand selected public names before the CANN macros stringify or paste them.
#define HYPER_MX_EXPAND(F, ...) F(__VA_ARGS__)
namespace l0op {
HYPER_MX_EXPAND(OP_TYPE_REGISTER, HYPER_MX_OP);
const aclTensorList* HYPER_MX_OP(
    const aclTensorList* buffers, const aclTensor* runtime, const aclTensor* events,
    const aclTensor* diagnostic, const aclTensor* launch, const aclTensorList* outputs,
    int64_t count, aclOpExecutor* executor) {
  auto status = HYPER_MX_EXPAND(ADD_TO_LAUNCHER_LIST_AICORE, HYPER_MX_OP,
      OP_INPUT(buffers, runtime, events, diagnostic, launch), OP_OUTPUT(outputs), OP_ATTR(count));
  return status == ACLNN_SUCCESS ? outputs : nullptr;
}
}
extern "C" {
aclnnStatus HYPER_MX_WORKSPACE_API(
    const aclTensorList* buffers, const aclTensor* runtime, const aclTensor* events,
    const aclTensor* diagnostic, const aclTensor* launch, const aclTensorList* outputs,
    int64_t count, uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  HYPER_MX_EXPAND(L2_DFX_PHASE_1, HYPER_MX_API, DFX_IN(buffers, runtime, events, diagnostic, launch, count),
                 DFX_OUT(outputs));
  CHECK_RET(buffers != nullptr && outputs != nullptr && count > 0 && count <= 64, ACLNN_ERR_PARAM_INVALID);
  auto unique_executor = CREATE_EXECUTOR();
  CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  auto result = l0op::HYPER_MX_OP(buffers, runtime, events, diagnostic, launch,
                                         outputs, count, unique_executor.get());
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = unique_executor->GetWorkspaceSize();
  unique_executor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}
aclnnStatus HYPER_MX_API(void* workspace, uint64_t bytes,
                                  aclOpExecutor* executor, aclrtStream stream) {
  HYPER_MX_EXPAND(L2_DFX_PHASE_2, HYPER_MX_API);
  return CommonOpExecutorRun(workspace, bytes, executor, stream);
}
}
