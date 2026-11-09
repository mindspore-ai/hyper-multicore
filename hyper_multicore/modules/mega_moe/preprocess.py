"""Opt-in route preprocessing with the existing native metadata ABI."""
from __future__ import annotations

from collections.abc import Mapping

import torch


def resolve_count_policy(environment: Mapping[str, str], *, bf16: bool) -> str:
    """Resolve the shared policy while preserving the existing BF16-only alias."""
    shared = environment.get("HYPER_MULTICORE_MOE_COUNT_POLICY")
    legacy = environment.get("HYPER_MULTICORE_BF16_COUNT_POLICY")
    if not bf16 and legacy not in (None, "baseline"):
        raise ValueError("HYPER_MULTICORE_BF16_COUNT_POLICY is only supported by the BF16 expert executor")
    if shared is not None and legacy is not None and shared != legacy:
        raise ValueError("Conflicting HYPER_MULTICORE_MOE_COUNT_POLICY and HYPER_MULTICORE_BF16_COUNT_POLICY")
    return shared if shared is not None else (legacy if legacy is not None else "baseline")


def validate_count_policy(counts: str, experts: int, dynamic_tokens: int | None) -> None:
    if counts not in ("baseline", "histc"):
        raise ValueError(f"Unsupported route count policy: {counts}")
    if counts == "histc" and (not 2 <= experts <= 2**24 or dynamic_tokens is not None):
        raise ValueError("Histogram preprocessing requires fixed tokens and 2..2**24 experts")


def fixed_route_counts(ids: torch.Tensor, experts: int) -> torch.Tensor:
    """Use a fixed range; the existing global load readback checks dropped invalid IDs."""
    return torch.histc(ids.to(torch.float32), bins=experts, min=0, max=experts - 1).to(torch.int32)


def validate_global_loads(loads: list[int], expected_slots: int) -> None:
    if sum(loads) != expected_slots:
        raise ValueError(f"Route count conservation failed: observed={sum(loads)}, expected={expected_slots}")
