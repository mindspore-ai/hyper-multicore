// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_GMM_DESCRIPTOR_H_
#define HYPER_MULTICORE_MX_GMM_DESCRIPTOR_H_

#include <cstddef>
#include <cstdint>

namespace MulticoreMx {
constexpr uint32_t MX_GMM_MAGIC = 0x4D58474D;
constexpr uint32_t MX_GMM_VERSION = 1;
constexpr uint32_t MX_GMM_TASK_TYPE = 142;
constexpr uint32_t MX_GMM_MAX_BUFFERS = 64;
constexpr uint32_t MX_GMM_DYNAMIC_GROUP = 1;

enum class MxLayout : uint32_t { NN = 0, NT = 1, TN = 2 };
enum class MxOutput : uint32_t { BF16 = 0, FP32 = 1 };

struct MxGmmDescriptor {
  uint32_t magic, version, size, layout, flags;
  uint32_t x_scale_position, w_scale_position, cube_index, cube_count;
  uint32_t m, n, k, base_m, base_n, base_k;
  uint32_t k_al1, k_bl1, scale_k, db_l0c, output_dtype;
  uint64_t x_offset, w_offset, y_offset, x_scale_offset, w_scale_offset, group_list_offset;
};
static_assert(sizeof(MxGmmDescriptor) == 128, "MX descriptor size mismatch");
static_assert(offsetof(MxGmmDescriptor, x_offset) == 80, "MX byte offset layout mismatch");
static_assert(offsetof(MxGmmDescriptor, group_list_offset) == 120, "MX group offset layout mismatch");

#ifdef __NPU_ARCH__
#define MX_DESCRIPTOR_INLINE __aicore__ inline
#else
#define MX_DESCRIPTOR_INLINE inline
#endif

MX_DESCRIPTOR_INLINE bool IsMxGmmDescriptorValid(const MxGmmDescriptor& r) {
  // These large tiles require the 256 KiB L0C on Ascend950 (3510).
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
  const bool tileM256 = false;
  const bool tileN256M128 = false;
  const bool tileSingleC256 = false;
#else
  const bool tileM256 = r.base_m == 256 && r.base_n == 128;
  const bool tileN256M128 = r.base_m == 128 && r.base_n == 256;
  const bool tileSingleC256 = r.base_m == 256 && r.base_n == 256
      && r.k_al1 == 256 && r.k_bl1 == 256 && r.scale_k == 256 && r.db_l0c == 1;
#endif
  const bool tile256 = ((r.base_m == 64 && r.base_n == 128) || (r.base_m == 128 && r.base_n == 128)
      || (r.base_m == 64 && r.base_n == 256) || tileM256 || tileN256M128)
      && r.k_al1 == 256 && r.k_bl1 == 256 && r.scale_k == 256;
  const bool tile512 = r.base_m == 128 && r.base_n == 128
      && r.k_al1 == 512 && r.k_bl1 == 512 && r.scale_k == 512;
  return r.magic == MX_GMM_MAGIC && r.version == MX_GMM_VERSION && r.size == sizeof(r)
    && r.layout <= static_cast<uint32_t>(MxLayout::TN) && r.cube_count > 0 && r.cube_index < r.cube_count
    && r.n > 0
    && (((tile256 || tile512) && r.db_l0c == 2) || tileSingleC256) && r.base_k == 128
    && r.x_scale_position < MX_GMM_MAX_BUFFERS && r.w_scale_position < MX_GMM_MAX_BUFFERS
    && r.output_dtype <= static_cast<uint32_t>(MxOutput::FP32) && r.group_list_offset % 8 == 0;
}
MX_DESCRIPTOR_INLINE void ResolveMxGmmGroup(MxGmmDescriptor& r, uint64_t start, uint64_t end) {
  const uint64_t expert = r.group_list_offset / 8;
  const uint64_t output_bytes = r.output_dtype == static_cast<uint32_t>(MxOutput::BF16) ? 2 : 4;
  if (r.layout == static_cast<uint32_t>(MxLayout::TN)) {
    r.x_offset += start * r.m;
    r.w_offset += start * r.n;
    const uint64_t scale_pair = start / 64 + expert;
    r.x_scale_offset += scale_pair * r.m * 2;
    r.w_scale_offset += scale_pair * r.n * 2;
    r.k = end - start;
  } else {
    r.x_offset += start * r.k;
    r.y_offset += start * r.n * output_bytes;
    r.x_scale_offset += start * ((r.k + 63) / 64) * 2;
    r.m = end - start;
  }
}
#undef MX_DESCRIPTOR_INLINE
}  // namespace MulticoreMx
#endif
