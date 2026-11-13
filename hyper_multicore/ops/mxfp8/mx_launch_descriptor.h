// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_LAUNCH_DESCRIPTOR_H_
#define HYPER_MULTICORE_MX_LAUNCH_DESCRIPTOR_H_
#include <cstdint>
#include <cstddef>
namespace MulticoreMx {
constexpr uint32_t MX_LAUNCH_MAGIC = 0x4D584C47;
constexpr uint32_t MX_LAUNCH_VERSION = 1;
constexpr uint32_t MX_BUFFER_LIMIT = 64;
struct MxLaunchDescriptor {
  int64_t rank, ep, experts, hidden, local_tokens, cube_count;
  uint64_t runtime_bytes, event_bytes;
  uint32_t magic, version, buffer_count, reserved;
  uint64_t buffer_bytes[MX_BUFFER_LIMIT];
};
static_assert(sizeof(MxLaunchDescriptor) == 592, "MX launch descriptor size mismatch");
static_assert(offsetof(MxLaunchDescriptor, runtime_bytes) == 48, "Runtime bounds mismatch");
#ifdef __NPU_ARCH__
__aicore__ inline
#else
inline
#endif
bool IsMxLaunchTopologyValid(int64_t ep, int64_t experts) {
  return ep > 0 && ep <= 0xffffffffLL && experts > 0 && experts <= 0xffffffffLL && experts % ep == 0;
}
}  // namespace MulticoreMx
#endif
