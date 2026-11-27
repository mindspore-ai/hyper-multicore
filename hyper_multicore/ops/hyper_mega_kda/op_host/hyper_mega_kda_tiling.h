/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef HYPER_MEGA_KDA_TILING_H
#define HYPER_MEGA_KDA_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HyperMegaKdaLaunchTilingData)
  // Core dimensions
  TILING_DATA_FIELD_DEF(uint32_t, seqLocal);
  TILING_DATA_FIELD_DEF(uint32_t, seqFull);
  TILING_DATA_FIELD_DEF(uint32_t, seqEff);
  TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, numHeads);
  TILING_DATA_FIELD_DEF(uint32_t, numLocalHeads);
  TILING_DATA_FIELD_DEF(uint32_t, headDim);
  TILING_DATA_FIELD_DEF(uint32_t, chunkSize);
  TILING_DATA_FIELD_DEF(uint32_t, numChunks);
  TILING_DATA_FIELD_DEF(uint32_t, cpSize);
  TILING_DATA_FIELD_DEF(uint32_t, cpRank);
  TILING_DATA_FIELD_DEF(uint32_t, cpMode);       // 0=ULYSSES, 1=ALLGATHER, 2=PREFIX_SUM
  TILING_DATA_FIELD_DEF(uint32_t, scanRounds);   // ceil(log2(cpSize)) for PREFIX_SUM
  TILING_DATA_FIELD_DEF(uint32_t, headGroups);
  TILING_DATA_FIELD_DEF(uint32_t, headsPerGroup);
  TILING_DATA_FIELD_DEF(uint32_t, numSegs);
  TILING_DATA_FIELD_DEF(uint32_t, chunksPerSeg);
  TILING_DATA_FIELD_DEF(uint32_t, projMTile);
  TILING_DATA_FIELD_DEF(uint32_t, oProjMTile);
  TILING_DATA_FIELD_DEF(uint32_t, useFullRankGate);
  TILING_DATA_FIELD_DEF(uint32_t, hasLowerBound);
  TILING_DATA_FIELD_DEF(uint32_t, lowerBoundBits);
  TILING_DATA_FIELD_DEF(uint32_t, normEpsBits);
  TILING_DATA_FIELD_DEF(uint32_t, scaleBits);
  TILING_DATA_FIELD_DEF(uint32_t, numCubeCores);
  TILING_DATA_FIELD_DEF(uint32_t, saveForBwd);

  // comm_send_buf layout (bytes)
  TILING_DATA_FIELD_DEF(uint64_t, sendQOff);
  TILING_DATA_FIELD_DEF(uint64_t, sendKOff);
  TILING_DATA_FIELD_DEF(uint64_t, sendVOff);
  TILING_DATA_FIELD_DEF(uint64_t, sendGOff);
  TILING_DATA_FIELD_DEF(uint64_t, sendBetaOff);
  TILING_DATA_FIELD_DEF(uint64_t, sendSlotBytes);
  TILING_DATA_FIELD_DEF(uint64_t, sendHaloOff);
  TILING_DATA_FIELD_DEF(uint64_t, sendHaloGroupBytes);
  TILING_DATA_FIELD_DEF(uint64_t, sendScanOff);
  TILING_DATA_FIELD_DEF(uint64_t, sendScanGroupHalfBytes);
  TILING_DATA_FIELD_DEF(uint64_t, sendScanSlotBytes);

  // comm_recv_buf layout (bytes)
  TILING_DATA_FIELD_DEF(uint64_t, recvQOff);
  TILING_DATA_FIELD_DEF(uint64_t, recvKOff);
  TILING_DATA_FIELD_DEF(uint64_t, recvVOff);
  TILING_DATA_FIELD_DEF(uint64_t, recvGOff);
  TILING_DATA_FIELD_DEF(uint64_t, recvBetaOff);
  TILING_DATA_FIELD_DEF(uint64_t, recvGroupBytes);
  TILING_DATA_FIELD_DEF(uint64_t, recvOOff);
  TILING_DATA_FIELD_DEF(uint64_t, recvOSlotBytes);
  TILING_DATA_FIELD_DEF(uint64_t, recvHaloOff);
  TILING_DATA_FIELD_DEF(uint64_t, recvScanOff);

  // user_workspace layout (bytes)
  TILING_DATA_FIELD_DEF(uint64_t, wsFaOutOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsGaOutOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsBetaAllOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsEye16Off);
  TILING_DATA_FIELD_DEF(uint64_t, wsGOutOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsLocalProjOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsQHatOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsKHatOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsVConvOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsVbetaOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsKbgOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsUOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsWOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsQgOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsKgOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsAkkOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsAqkOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsExpGLastOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsSubInOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsSubOutOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsHOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsVnewOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsStateFp32Off);
  TILING_DATA_FIELD_DEF(uint64_t, wsFwdHScratchOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsFwdHScratchStride);
  TILING_DATA_FIELD_DEF(uint64_t, wsOLocalSegOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsONormedOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsScanSelfOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsScanExclOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsScanTmpOff);

  // Runtime storage bounds (must remain adjacent uint64_t pair for RuntimeStorageBounds)
  TILING_DATA_FIELD_DEF(uint64_t, runtimeConfigBytes);
  TILING_DATA_FIELD_DEF(uint64_t, eventCounterBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HyperMegaKda, HyperMegaKdaLaunchTilingData)

}  // namespace optiling

#endif  // HYPER_MEGA_KDA_TILING_H
