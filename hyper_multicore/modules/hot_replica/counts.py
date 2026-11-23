# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the A3 HotReplica count kernels through the PTA task queue."""

from functools import lru_cache

import torch

from hyper_multicore.torch.ops import _load_native

from .native import native_library_path


@lru_cache(maxsize=1)
def _operators():
    _load_native()
    namespace = torch.ops.hyper_multicore
    if not hasattr(namespace, "replica_route_payload"):
        return None
    return namespace, str(native_library_path("libhmc_hot_replica_kernels.so"))


def route_payload(ids, experts, mask):
    """Pack the histogram and validation flags without expanding token rows."""
    if (ids.device.type != "npu" or ids.dtype not in (torch.int32, torch.int64)
            or not 1 <= ids.shape[1] <= 8 or not 1 <= experts <= 512):
        return None
    operators = _operators()
    if operators is None:
        return None
    namespace, library = operators
    return namespace.replica_route_payload(ids, experts, mask, library)
