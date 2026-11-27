/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef HYPER_MEGA_KDA_GRAD_TILING_H
#define HYPER_MEGA_KDA_GRAD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HyperMegaKdaGradTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, seqLocal);
  TILING_DATA_FIELD_DEF(uint32_t, numHeads);
  TILING_DATA_FIELD_DEF(uint32_t, headDim);
  TILING_DATA_FIELD_DEF(uint32_t, chunkSize);
  TILING_DATA_FIELD_DEF(uint32_t, numChunks);
  TILING_DATA_FIELD_DEF(uint32_t, cpSize);
  TILING_DATA_FIELD_DEF(uint32_t, cpRank);
  TILING_DATA_FIELD_DEF(uint32_t, cpMode);
  TILING_DATA_FIELD_DEF(uint32_t, scanRounds);
  TILING_DATA_FIELD_DEF(uint32_t, hasLowerBound);
  TILING_DATA_FIELD_DEF(uint32_t, lowerBoundBits);
  TILING_DATA_FIELD_DEF(uint32_t, normEpsBits);
  TILING_DATA_FIELD_DEF(uint32_t, scaleBits);
  TILING_DATA_FIELD_DEF(uint32_t, numCubeCores);

  // SHMEM comm layout (bytes, reusing pool comm_send_buf / comm_recv_buf)
  TILING_DATA_FIELD_DEF(uint64_t, sendHaloOff);
  TILING_DATA_FIELD_DEF(uint64_t, recvHaloOff);
  TILING_DATA_FIELD_DEF(uint64_t, sendScanOff);
  TILING_DATA_FIELD_DEF(uint64_t, recvScanOff);
  TILING_DATA_FIELD_DEF(uint64_t, scanHalfBytes);   // H * d * d * 4 (one state matrix across all H)
  TILING_DATA_FIELD_DEF(uint64_t, scanSlotBytes);   // 2 * scanHalfBytes (D_r + A_r^T)
  TILING_DATA_FIELD_DEF(uint64_t, eventCounterBytes);

  // Workspace layout (bytes)
  TILING_DATA_FIELD_DEF(uint64_t, wsDOOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsDvNewOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsDAqkOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsDqgOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsDkgOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsDwOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsDgammaOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsBwdHScratchOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsBwdRingOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsDSiluOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsDuOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsPartialNormOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsPartialALogOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsPartialDtBiasOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsPartialWConvOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsRevSelfOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsRevExclOff);
  TILING_DATA_FIELD_DEF(uint64_t, wsRevTmpOff);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HyperMegaKdaGrad, HyperMegaKdaGradTilingData)

}  // namespace optiling

#endif  // HYPER_MEGA_KDA_GRAD_TILING_H
