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

"""CPU contracts for EP token and optional score reordering."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from examples import _native
from tests.common.mark_utils import arg_mark


class TestNativeExpertPermutation(unittest.TestCase):
    """Check observable routing semantics independently of the NPU kernels."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sparse_blocks_and_empty_input(self):
        """Sparse rank blocks and empty receives preserve values and gradients."""
        counts = torch.tensor([2, 0, 1, 1, 3, 0], dtype=torch.int32)
        inputs = torch.arange(14, dtype=torch.float32).reshape(7, 2).requires_grad_()
        # pylint: disable=protected-access
        shape, permuted, indices, local_counts, fused = _native._permute(inputs, counts, 2, 3)
        expected_indices = torch.tensor([0, 1, 3, 4, 5, 6, 2])
        torch.testing.assert_close(permuted, inputs[expected_indices])
        torch.testing.assert_close(local_counts, torch.tensor([3, 3, 1]))
        self.assertFalse(fused)
        restored = _native._unpermute(permuted, shape, indices, fused=fused)
        probe = torch.arange(14, dtype=torch.float32).reshape(7, 2)
        torch.testing.assert_close(restored, inputs)
        (restored * probe).sum().backward()
        torch.testing.assert_close(inputs.grad, probe)

        empty = torch.empty((0, 2), requires_grad=True)
        shape, permuted, indices, local_counts, fused = _native._permute(empty, counts * 0, 2, 3)
        restored = _native._unpermute(permuted, shape, indices, fused=fused)
        self.assertEqual(tuple(restored.shape), (0, 2))
        torch.testing.assert_close(local_counts, torch.zeros(3, dtype=torch.int64))
        restored.sum().backward()
        self.assertIsNotNone(empty.grad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_optional_scores_follow_both_index_conventions(self):
        """Scores and their gradients follow source-to-destination or gather indices."""
        expected_order = torch.tensor([0, 1, 3, 4, 5, 6, 2])
        scatter_mapping = torch.tensor([0, 1, 6, 2, 3, 4, 5], dtype=torch.int32)
        probe = torch.arange(1, 8, dtype=torch.float32)
        for fused, indices in ((False, expected_order), (True, scatter_mapping)):
            with self.subTest(fused=fused):
                scores = torch.arange(7, dtype=torch.float32, requires_grad=True)
                with patch.object(
                    _native, "differentiable_all_to_all_single", side_effect=lambda value, *_a, **_k: value,
                ):
                    # pylint: disable=protected-access
                    _, reordered = _native.AllToAllTokenDispatcher._exchange_and_reorder_probs(
                        scores, [7], [7], indices, fused, None,
                    )
                torch.testing.assert_close(reordered, scores[expected_order])
                reordered.mul(probe).sum().backward()
                expected_grad = torch.zeros_like(probe)
                expected_grad[expected_order] = probe
                torch.testing.assert_close(scores.grad, expected_grad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_permutation_index_limit_uses_fallback(self):
        """Oversized NPU receives select the ordinary path without allocating huge buffers."""
        class LargeLogicalInput:
            """Use real small tensor values with a synthetic oversized logical shape."""

            shape = ((1 << 24) - 1, 2)
            device = SimpleNamespace(type="npu")
            dtype = torch.bfloat16
            ndim = 2

            def __init__(self) -> None:
                """Keep only a few values for the fallback permutation."""
                self.values = torch.arange(14, dtype=torch.bfloat16).reshape(7, 2)

            def __getitem__(self, indices: torch.Tensor) -> torch.Tensor:
                """Apply the fallback indices to the small value sample."""
                return self.values[indices]

        inputs = LargeLogicalInput()
        counts = torch.tensor([2, 0, 1, 1, 3, 0], dtype=torch.int32)
        # pylint: disable=protected-access
        _, permuted, _, local_counts, fused = _native._permute(inputs, counts, 2, 3)
        self.assertFalse(fused)
        torch.testing.assert_close(permuted, inputs.values[torch.tensor([0, 1, 3, 4, 5, 6, 2])])
        torch.testing.assert_close(local_counts, torch.tensor([3, 3, 1]))


if __name__ == "__main__":
    unittest.main()
