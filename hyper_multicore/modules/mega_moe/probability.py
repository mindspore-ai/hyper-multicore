"""Probability-plane layout shared by plans, kernels and diagnostics."""
from __future__ import annotations

from typing import Any

# The extended GMM ABI reserves four FP32 planes behind an auxiliary descriptor.
PROBABILITY_PLANES = 4
PROBABILITY_DTYPE_BYTES = 4  # FP32; the TensorDescC data_type field stores bytes.
PROBABILITY_POSITION_FORWARD = 16
PROBABILITY_POSITION_BACKWARD = 25


def probability_stride(spec: Any) -> int:
    """Return the conservative routed-slot bound of one probability plane."""
    return spec.ep_size * spec.plan_tokens * spec.top_k
