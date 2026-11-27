/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include "hyper_mega_kda_grad_tiling.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr uint64_t kAlignBytes = 512ULL;

inline uint64_t AlignUp(uint64_t v, uint64_t a = kAlignBytes) {
  return (v + a - 1ULL) / a * a;
}

inline uint32_t FloatToBits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

inline uint32_t CeilLog2(uint32_t n) {
  uint32_t r = 0;
  uint32_t p = 1;
  while (p < n) {
    p <<= 1;
    ++r;
  }
  return r;
}

ge::graphStatus TilingFunc(gert::TilingContext *context) {
  if (context == nullptr) return ge::GRAPH_FAILED;
  const auto *doShape = context->GetInputShape(0);
  const auto *eventShape = context->GetInputShape(16);
  const auto *attrs = context->GetAttrs();
  if (doShape == nullptr || eventShape == nullptr || attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const auto &ds = doShape->GetStorageShape();
  uint32_t seqLocal = 0;
  if (ds.GetDimNum() == 3) {
    if (ds.GetDim(0) != 1) return ge::GRAPH_FAILED;
    seqLocal = static_cast<uint32_t>(ds.GetDim(1));
  } else if (ds.GetDimNum() == 2) {
    seqLocal = static_cast<uint32_t>(ds.GetDim(0));
  } else {
    return ge::GRAPH_FAILED;
  }

  auto getIntAttr = [&](size_t idx, int64_t defVal) -> int64_t {
    const auto *p = attrs->GetAttrPointer<int64_t>(idx);
    return p != nullptr ? *p : defVal;
  };
  auto getBoolAttr = [&](size_t idx, bool defVal) -> bool {
    const auto *p = attrs->GetAttrPointer<bool>(idx);
    return p != nullptr ? *p : defVal;
  };
  auto getFloatAttr = [&](size_t idx, float defVal) -> float {
    const auto *p = attrs->GetAttrPointer<float>(idx);
    return p != nullptr ? *p : defVal;
  };

  const uint32_t numHeads = static_cast<uint32_t>(getIntAttr(0, 16));
  const uint32_t d = static_cast<uint32_t>(getIntAttr(1, 128));
  const uint32_t C = static_cast<uint32_t>(getIntAttr(2, 64));
  const uint32_t cpSize = static_cast<uint32_t>(getIntAttr(3, 1));
  const uint32_t cpRank = static_cast<uint32_t>(getIntAttr(4, 0));
  const uint32_t cpMode = static_cast<uint32_t>(getIntAttr(5, 0));
  const bool hasLowerBound = getBoolAttr(6, true);
  const float lowerBound = getFloatAttr(7, -5.0f);
  const float normEps = getFloatAttr(8, 1e-6f);
  const float scale = getFloatAttr(9, 1.0f / std::sqrt(static_cast<float>(d)));

  if (cpSize == 0 || cpRank >= cpSize || C != 64 || d != 128 || (seqLocal % C) != 0 || (numHeads & 1U) != 0) {
    return ge::GRAPH_FAILED;
  }

  const uint32_t numChunks = seqLocal / C;
  const uint32_t scanRounds = (cpMode == 2U && cpSize > 1U) ? CeilLog2(cpSize) : 0U;

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const uint32_t numCubeCores = platform.GetCoreNumAic();
  const uint32_t numAiv = 2U * numCubeCores;
  context->SetBlockDim(numCubeCores);

  const uint64_t S = static_cast<uint64_t>(seqLocal);
  const uint64_t H = static_cast<uint64_t>(numHeads);
  const uint64_t D = H * d;
  const uint64_t hc = H * numChunks;

  // SHMEM comm layout (cleanly placed at offset 0 of comm_send_buf / comm_recv_buf!)
  const uint64_t haloBytes = AlignUp(3ULL * 3ULL * D * 2ULL);
  const uint64_t scanHalfBytes = AlignUp(H * d * d * 4ULL);
  const uint64_t scanSlotBytes = 2ULL * scanHalfBytes;

  HyperMegaKdaGradTilingData tiling;
  tiling.set_seqLocal(seqLocal);
  tiling.set_numHeads(numHeads);
  tiling.set_headDim(d);
  tiling.set_chunkSize(C);
  tiling.set_numChunks(numChunks);
  tiling.set_cpSize(cpSize);
  tiling.set_cpRank(cpRank);
  tiling.set_cpMode(cpMode);
  tiling.set_scanRounds(scanRounds);
  tiling.set_hasLowerBound(hasLowerBound ? 1U : 0U);
  tiling.set_lowerBoundBits(FloatToBits(lowerBound));
  tiling.set_normEpsBits(FloatToBits(normEps));
  tiling.set_scaleBits(FloatToBits(scale));
  tiling.set_numCubeCores(numCubeCores);

  tiling.set_sendHaloOff(0ULL);
  tiling.set_recvHaloOff(0ULL);
  tiling.set_sendScanOff(haloBytes);
  tiling.set_recvScanOff(haloBytes);
  tiling.set_scanHalfBytes(scanHalfBytes);
  tiling.set_scanSlotBytes(scanSlotBytes);
  tiling.set_eventCounterBytes(static_cast<uint64_t>(eventShape->GetStorageShape().GetShapeSize()) * 4ULL);

  // User workspace layout (all 512B aligned)
  uint64_t ws = 0ULL;
  auto allocWs = [&](uint64_t bytes) -> uint64_t {
    uint64_t cur = ws;
    ws = AlignUp(ws + bytes);
    return cur;
  };

  tiling.set_wsDOOff(allocWs(hc * C * d * 2ULL));
  tiling.set_wsDvNewOff(allocWs(hc * C * d * 2ULL));
  tiling.set_wsDAqkOff(allocWs(hc * C * C * 2ULL));
  tiling.set_wsDqgOff(allocWs(hc * C * d * 2ULL));
  tiling.set_wsDkgOff(allocWs(hc * C * d * 2ULL));
  tiling.set_wsDwOff(allocWs(hc * C * d * 2ULL));
  tiling.set_wsDgammaOff(allocWs(hc * d * 4ULL));
  tiling.set_wsBwdHScratchOff(allocWs(static_cast<uint64_t>(numCubeCores) * 256ULL * 1024ULL));
  tiling.set_wsBwdRingOff(allocWs(static_cast<uint64_t>(numCubeCores) * 1024ULL * 1024ULL));
  tiling.set_wsDSiluOff(allocWs(3ULL * S * D * 2ULL));
  tiling.set_wsDuOff(allocWs(3ULL * S * D * 2ULL));
  tiling.set_wsPartialNormOff(allocWs(static_cast<uint64_t>(numAiv) * d * 4ULL));
  const uint64_t aLogStride = AlignUp(H * 4ULL, 32ULL);
  tiling.set_wsPartialALogOff(allocWs(static_cast<uint64_t>(numAiv) * aLogStride));
  tiling.set_wsPartialDtBiasOff(allocWs(static_cast<uint64_t>(numAiv) * D * 4ULL));

  const uint32_t hTile = ((numHeads & 3U) == 0U && (numHeads / 4U) <= numAiv) ? 4U : 1U;
  const uint32_t numGroups = numHeads / hTile;
  const uint32_t slicesPerGroup = numAiv / numGroups;
  tiling.set_wsPartialWConvOff(allocWs(static_cast<uint64_t>(slicesPerGroup) * 3ULL * D * 4ULL * 4ULL));

  if (cpSize > 1U) {
    tiling.set_wsRevSelfOff(allocWs(scanSlotBytes));
    tiling.set_wsRevExclOff(allocWs(scanSlotBytes));
    tiling.set_wsRevTmpOff(allocWs(H * 3ULL * d * d * 4ULL));
  } else {
    tiling.set_wsRevSelfOff(allocWs(512ULL));
    tiling.set_wsRevExclOff(allocWs(512ULL));
    tiling.set_wsRevTmpOff(allocWs(512ULL));
  }

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t *workspaces = context->GetWorkspaceSizes(1);
  if (workspaces == nullptr) return ge::GRAPH_FAILED;
  const size_t sysWs = platform.GetLibApiWorkSpaceSize();
  workspaces[0] = sysWs + static_cast<size_t>(ws);
  return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(HyperMegaKdaGrad).Tiling(TilingFunc);

}  // namespace optiling
