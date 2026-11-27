/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include "hyper_mega_kda_tiling.h"

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
  const auto *hiddenShape = context->GetInputShape(0);
  const auto *runtimeShape = context->GetInputShape(18);
  const auto *eventShape = context->GetInputShape(19);
  const auto *attrs = context->GetAttrs();
  if (hiddenShape == nullptr || runtimeShape == nullptr || eventShape == nullptr || attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const auto &hs = hiddenShape->GetStorageShape();
  uint32_t seqLocal = 0;
  uint32_t hiddenSize = 0;
  if (hs.GetDimNum() == 3) {
    if (hs.GetDim(0) != 1) return ge::GRAPH_FAILED;
    seqLocal = static_cast<uint32_t>(hs.GetDim(1));
    hiddenSize = static_cast<uint32_t>(hs.GetDim(2));
  } else if (hs.GetDimNum() == 2) {
    seqLocal = static_cast<uint32_t>(hs.GetDim(0));
    hiddenSize = static_cast<uint32_t>(hs.GetDim(1));
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
  const uint32_t headGroups = static_cast<uint32_t>(getIntAttr(5, 2));
  const uint32_t numSegs = static_cast<uint32_t>(getIntAttr(6, 1));
  const uint32_t projMTile = static_cast<uint32_t>(getIntAttr(7, 256));
  const uint32_t oProjMTile = static_cast<uint32_t>(getIntAttr(8, 128));
  const bool useFullRankGate = getBoolAttr(9, false);
  const bool hasLowerBound = getBoolAttr(10, true);
  const float lowerBound = getFloatAttr(11, -5.0f);
  const float normEps = getFloatAttr(12, 1e-6f);
  const float scale = getFloatAttr(13, 1.0f / std::sqrt(static_cast<float>(d)));
  const uint32_t cpMode = static_cast<uint32_t>(getIntAttr(14, 0));
  const bool saveForBwd = getBoolAttr(15, false);

  if (cpSize == 0 || cpRank >= cpSize || headGroups == 0 || numSegs == 0 || C != 64 || d != 128) {
    return ge::GRAPH_FAILED;
  }

  const bool isUlysses = (cpMode == 0U);
  const bool isAllGather = (cpMode == 1U);
  const bool isPrefixSum = (cpMode == 2U);
  if (!isUlysses && !isAllGather && !isPrefixSum) return ge::GRAPH_FAILED;

  const uint32_t seqFull = seqLocal * cpSize;
  const uint32_t seqEff = isUlysses ? seqFull : seqLocal;
  const uint32_t numLocalHeads = isUlysses ? (numHeads / cpSize) : numHeads;
  if (numLocalHeads == 0 || (numLocalHeads % headGroups) != 0) return ge::GRAPH_FAILED;
  const uint32_t headsPerGroup = numLocalHeads / headGroups;
  const uint32_t groupDim = headsPerGroup * d;
  const uint32_t numChunks = seqEff / C;
  if (seqEff % C != 0 || numChunks % numSegs != 0) return ge::GRAPH_FAILED;
  const uint32_t chunksPerSeg = numChunks / numSegs;
  const uint32_t scanRounds = (isPrefixSum && cpSize > 1) ? CeilLog2(cpSize) : 0U;

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const uint32_t numCubeCores = platform.GetCoreNumAic();
  context->SetBlockDim(numCubeCores);

  HyperMegaKdaLaunchTilingData tiling;
  tiling.set_seqLocal(seqLocal);
  tiling.set_seqFull(seqFull);
  tiling.set_seqEff(seqEff);
  tiling.set_hiddenSize(hiddenSize);
  tiling.set_numHeads(numHeads);
  tiling.set_numLocalHeads(numLocalHeads);
  tiling.set_headDim(d);
  tiling.set_chunkSize(C);
  tiling.set_numChunks(numChunks);
  tiling.set_cpSize(cpSize);
  tiling.set_cpRank(cpRank);
  tiling.set_cpMode(cpMode);
  tiling.set_scanRounds(scanRounds);
  tiling.set_headGroups(headGroups);
  tiling.set_headsPerGroup(headsPerGroup);
  tiling.set_numSegs(numSegs);
  tiling.set_chunksPerSeg(chunksPerSeg);
  tiling.set_projMTile(projMTile);
  tiling.set_oProjMTile(oProjMTile);
  tiling.set_useFullRankGate(useFullRankGate ? 1U : 0U);
  tiling.set_hasLowerBound(hasLowerBound ? 1U : 0U);
  tiling.set_lowerBoundBits(FloatToBits(lowerBound));
  tiling.set_normEpsBits(FloatToBits(normEps));
  tiling.set_scaleBits(FloatToBits(scale));
  tiling.set_numCubeCores(numCubeCores);
  tiling.set_saveForBwd(saveForBwd ? 1U : 0U);

  // 1. comm_send_buf & comm_recv_buf layouts
  uint64_t sendQOff = 0, sendKOff = 0, sendVOff = 0, sendGOff = 0, sendBetaOff = 0, sendSlotBytes = 0;
  uint64_t recvQOff = 0, recvKOff = 0, recvVOff = 0, recvGOff = 0, recvBetaOff = 0, recvGroupBytes = 0;
  uint64_t recvOOff = 0, recvOSlotBytes = 0;
  uint64_t sendHaloOff = 0, sendHaloGroupBytes = 0, recvHaloOff = 0;
  uint64_t sendScanOff = 0, sendScanGroupHalfBytes = 0, sendScanSlotBytes = 0, recvScanOff = 0;

  const uint64_t qkvSeqEffBytes = static_cast<uint64_t>(seqEff) * groupDim * 2ULL;
  const uint64_t betaSeqEffBytes = static_cast<uint64_t>(seqEff) * headsPerGroup * 2ULL;
  recvQOff = 0;
  recvKOff = AlignUp(recvQOff + qkvSeqEffBytes);
  recvVOff = AlignUp(recvKOff + qkvSeqEffBytes);
  recvGOff = AlignUp(recvVOff + qkvSeqEffBytes);
  recvBetaOff = AlignUp(recvGOff + qkvSeqEffBytes);
  recvGroupBytes = AlignUp(recvBetaOff + betaSeqEffBytes);

  const uint64_t qkvLocalBytes = static_cast<uint64_t>(seqLocal) * groupDim * 2ULL;
  recvOSlotBytes = AlignUp(qkvLocalBytes);
  if (isUlysses) {
    const uint64_t betaLocalBytes = static_cast<uint64_t>(seqLocal) * headsPerGroup * 2ULL;
    sendQOff = 0;
    sendKOff = AlignUp(sendQOff + qkvLocalBytes);
    sendVOff = AlignUp(sendKOff + qkvLocalBytes);
    sendGOff = AlignUp(sendVOff + qkvLocalBytes);
    sendBetaOff = AlignUp(sendGOff + qkvLocalBytes);
    sendSlotBytes = AlignUp(sendBetaOff + betaLocalBytes);
    recvOOff = AlignUp(static_cast<uint64_t>(headGroups) * recvGroupBytes);
  } else {
    sendHaloGroupBytes = AlignUp(3ULL * 3ULL * groupDim * 2ULL);
    sendHaloOff = 0;
    recvHaloOff = 0;
    sendScanGroupHalfBytes = AlignUp(static_cast<uint64_t>(headsPerGroup) * d * d * 4ULL);
    sendScanSlotBytes = AlignUp(static_cast<uint64_t>(headGroups) * 2ULL * sendScanGroupHalfBytes);
    sendScanOff = AlignUp(static_cast<uint64_t>(headGroups) * sendHaloGroupBytes);
    recvScanOff = AlignUp(static_cast<uint64_t>(headGroups) * sendHaloGroupBytes);
  }

  tiling.set_sendQOff(sendQOff);
  tiling.set_sendKOff(sendKOff);
  tiling.set_sendVOff(sendVOff);
  tiling.set_sendGOff(sendGOff);
  tiling.set_sendBetaOff(sendBetaOff);
  tiling.set_sendSlotBytes(sendSlotBytes);
  tiling.set_sendHaloOff(sendHaloOff);
  tiling.set_sendHaloGroupBytes(sendHaloGroupBytes);
  tiling.set_sendScanOff(sendScanOff);
  tiling.set_sendScanGroupHalfBytes(sendScanGroupHalfBytes);
  tiling.set_sendScanSlotBytes(sendScanSlotBytes);

  tiling.set_recvQOff(recvQOff);
  tiling.set_recvKOff(recvKOff);
  tiling.set_recvVOff(recvVOff);
  tiling.set_recvGOff(recvGOff);
  tiling.set_recvBetaOff(recvBetaOff);
  tiling.set_recvGroupBytes(recvGroupBytes);
  tiling.set_recvOOff(recvOOff);
  tiling.set_recvOSlotBytes(recvOSlotBytes);
  tiling.set_recvHaloOff(recvHaloOff);
  tiling.set_recvScanOff(recvScanOff);

  // 2. Workspace offsets
  uint64_t ws = 0;
  auto allocWs = [&](uint64_t bytes) -> uint64_t {
    uint64_t off = ws;
    ws = AlignUp(ws + bytes);
    return off;
  };

  const uint64_t totalHeadDim = static_cast<uint64_t>(numHeads) * d;
  const uint64_t hcTotal = static_cast<uint64_t>(numLocalHeads) * numChunks;
  const uint64_t seqHeadElems = static_cast<uint64_t>(numLocalHeads) * seqEff * d;

  const bool cp1Fast = (projMTile == 0U && ((numHeads & 1U) == 0U) && (cpSize == 1U || isAllGather || isPrefixSum));
  tiling.set_wsFaOutOff(cp1Fast ? allocWs(512ULL) : allocWs(static_cast<uint64_t>(seqLocal) * d * 2ULL));
  tiling.set_wsGaOutOff(cp1Fast ? allocWs(512ULL) : allocWs(static_cast<uint64_t>(seqLocal) * d * 2ULL));
  tiling.set_wsBetaAllOff(cp1Fast ? allocWs(512ULL) : allocWs(static_cast<uint64_t>(seqLocal) * numHeads * 2ULL));
  tiling.set_wsEye16Off(allocWs(saveForBwd ? (512ULL + 64ULL * 64ULL * 2ULL) : 512ULL));
  tiling.set_wsGOutOff((projMTile == 0U) ? allocWs(512ULL) : allocWs(static_cast<uint64_t>(seqLocal) * totalHeadDim * 2ULL));
  tiling.set_wsLocalProjOff((isUlysses || projMTile == 0U) ? allocWs(512ULL)
                                      : allocWs(static_cast<uint64_t>(headGroups) * recvGroupBytes));

  tiling.set_wsQHatOff(allocWs(seqHeadElems * 2ULL));
  tiling.set_wsKHatOff(allocWs(seqHeadElems * 2ULL));
  tiling.set_wsVConvOff(allocWs(seqHeadElems * 2ULL));
  tiling.set_wsVbetaOff(cp1Fast ? allocWs(512ULL) : allocWs(hcTotal * C * d * 2ULL));
  tiling.set_wsKbgOff(cp1Fast ? allocWs(512ULL) : allocWs(hcTotal * C * d * 2ULL));
  tiling.set_wsUOff(allocWs(hcTotal * C * d * 4ULL));
  tiling.set_wsWOff(allocWs(hcTotal * C * d * 2ULL));
  tiling.set_wsQgOff(allocWs(hcTotal * C * d * 2ULL));
  tiling.set_wsKgOff(allocWs(hcTotal * C * d * 2ULL));
  tiling.set_wsAkkOff(cp1Fast ? allocWs(512ULL) : allocWs(hcTotal * C * C * 2ULL));
  tiling.set_wsAqkOff(allocWs(hcTotal * C * C * 2ULL));
  tiling.set_wsExpGLastOff(allocWs(hcTotal * d * 4ULL));
  tiling.set_wsSubInOff(cp1Fast ? allocWs(static_cast<uint64_t>(numCubeCores) * 576ULL * 1024ULL)
                                : allocWs(hcTotal * 384ULL * d * 2ULL));
  tiling.set_wsSubOutOff(cp1Fast ? allocWs(512ULL) : allocWs(hcTotal * 5120ULL * 4ULL));
  tiling.set_wsHOff(cp1Fast ? allocWs(512ULL) : allocWs(hcTotal * d * d * 2ULL));
  tiling.set_wsVnewOff(cp1Fast ? allocWs(512ULL) : allocWs(hcTotal * C * d * 2ULL));
  tiling.set_wsStateFp32Off(cp1Fast ? allocWs(512ULL) : allocWs(static_cast<uint64_t>(numLocalHeads) * d * d * 4ULL));

  const uint64_t fwdScratchStride = cp1Fast
      ? (320ULL * 1024ULL)
      : AlignUp(2ULL * (static_cast<uint64_t>(C) * d + static_cast<uint64_t>(d) * d) * 4ULL);
  tiling.set_wsFwdHScratchStride(fwdScratchStride);
  tiling.set_wsFwdHScratchOff(allocWs(static_cast<uint64_t>(numCubeCores) * fwdScratchStride));

  tiling.set_wsOLocalSegOff(cp1Fast ? allocWs(512ULL) : allocWs(static_cast<uint64_t>(numLocalHeads) * seqEff * d * 2ULL));
  tiling.set_wsONormedOff((projMTile == 0U) ? allocWs(512ULL) : allocWs(static_cast<uint64_t>(seqLocal) * totalHeadDim * 2ULL));

  if (!isUlysses && cpSize > 1) {
    tiling.set_wsScanSelfOff(allocWs(sendScanSlotBytes));
    tiling.set_wsScanExclOff(allocWs(sendScanSlotBytes));
    tiling.set_wsScanTmpOff(allocWs(static_cast<uint64_t>(numHeads) * 3ULL * d * d * 4ULL));
  } else {
    tiling.set_wsScanSelfOff(allocWs(512ULL));
    tiling.set_wsScanExclOff(allocWs(512ULL));
    tiling.set_wsScanTmpOff(allocWs(512ULL));
  }

  // 3. Runtime storage bounds (ZERO MatmulApiTiling CPU overhead!)
  const uint64_t runtimeBytes = static_cast<uint64_t>(runtimeShape->GetStorageShape().GetShapeSize());
  const uint64_t eventBytes = static_cast<uint64_t>(eventShape->GetStorageShape().GetShapeSize()) * sizeof(int32_t);
  tiling.set_runtimeConfigBytes(runtimeBytes);
  tiling.set_eventCounterBytes(eventBytes);

  auto *wsSizes = context->GetWorkspaceSizes(1);
  if (wsSizes == nullptr) return ge::GRAPH_FAILED;
  wsSizes[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) + static_cast<size_t>(AlignUp(ws + 4ULL * 1024ULL * 1024ULL));

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const auto *hs = context->GetInputShape(0);
  auto *out = context->GetOutputShape(0);
  if (hs == nullptr || out == nullptr) return ge::GRAPH_FAILED;
  *out = *hs;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(HyperMegaKda)
    .Tiling(TilingFunc);

}  // namespace optiling
