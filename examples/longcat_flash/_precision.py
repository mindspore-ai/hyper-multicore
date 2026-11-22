# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bounded, CPU-verifiable numerical diagnostics for LongCat acceptance workers."""

import math

import torch


STRICT_GATE = {"atol": 2e-5, "rtol": 0.05, "relative_l2": 0.05}
_CHUNK_ELEMENTS = 8 * 1024 * 1024


def _example(offset: int, index: int, observed: torch.Tensor, target: torch.Tensor) -> dict:
    """Retain one failing flat index and JSON-safe operand values."""
    actual, reference = float(observed[index]), float(target[index])
    return {"flat_index": offset + index,
            "actual": actual if math.isfinite(actual) else str(actual),
            "reference": reference if math.isfinite(reference) else str(reference)}


def compare_tensors(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Check the frozen gates, metadata and finiteness without NPU FP64 emulation.

    Args:
        actual: Candidate tensor, on CPU or the tested accelerator.
        reference: Independently retained reference tensor.
    """
    result = {"actual_shape": list(actual.shape), "reference_shape": list(reference.shape),
              "actual_dtype": str(actual.dtype), "reference_dtype": str(reference.dtype),
              "elements": actual.numel(), "strict_pass": False, "component_pass": False,
              "max_abs": None, "relative_l2": None, "strict_bad": None, "component_bad": None}
    if actual.shape != reference.shape:
        return {**result, "reason": "shape mismatch"}
    flat, expected = actual.detach().reshape(-1), reference.detach().reshape(-1)
    maximum = delta_square = reference_square = 0.0
    strict_bad = component_bad = actual_nonfinite = reference_nonfinite = 0
    examples = []
    for offset in range(0, flat.numel(), _CHUNK_ELEMENTS):
        # NPU FP64 casts fall back to FP32; norm reductions must actually run on CPU.
        observed = flat[offset:offset + _CHUNK_ELEMENTS].to(device="cpu", dtype=torch.float32)
        target = expected[offset:offset + observed.numel()].to(device="cpu", dtype=torch.float32)
        valid_actual, valid_reference = torch.isfinite(observed), torch.isfinite(target)
        finite = valid_actual & valid_reference
        actual_nonfinite += int((~valid_actual).sum())
        reference_nonfinite += int((~valid_reference).sum())
        delta = observed - target
        absolute = delta.abs()
        strict = (~finite) | (absolute > STRICT_GATE["atol"] + STRICT_GATE["rtol"] * target.abs())
        component = (~finite) | (absolute > 2e-3 + 0.02 * target.abs())
        strict_bad += int(strict.sum())
        component_bad += int(component.sum())
        if len(examples) < 8 and bool(strict.any()):
            indices = strict.nonzero().flatten()[:8 - len(examples)].tolist()
            examples.extend(_example(offset, index, observed, target) for index in indices)
        if bool(finite.all()):
            maximum = max(maximum, float(absolute.max()))
            delta_square += float(delta.double().square().sum())
            reference_square += float(target.double().square().sum())
    nonfinite = actual_nonfinite + reference_nonfinite
    relative = None if nonfinite else (delta_square / max(reference_square, 1e-24)) ** 0.5
    metadata_pass = actual.dtype == reference.dtype
    result.update(max_abs=None if nonfinite else maximum, relative_l2=relative,
                  strict_bad=strict_bad, component_bad=component_bad,
                  actual_nonfinite=actual_nonfinite, reference_nonfinite=reference_nonfinite,
                  first_strict_failures=examples, metadata_pass=metadata_pass,
                  strict_pass=metadata_pass and not nonfinite and strict_bad == 0
                  and relative <= STRICT_GATE["relative_l2"],
                  component_pass=metadata_pass and not nonfinite and component_bad == 0 and relative <= 0.02)
    return result


def require_strict_pass(results: list[dict]) -> None:
    """Reject local or remote strict failures after rank reports and cleanup exist.

    Args:
        results: Completed records containing local and all-rank strict verdicts.
    """
    if not results or not all(row["strict_pass"] and row["all_ranks_strict_pass"] for row in results):
        raise RuntimeError("native/MegaMoE strict elementwise precision failed; inspect every rank report")
