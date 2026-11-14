# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Cross architecture transport, workspace descriptors and FP32 reference checks."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch
import torch.nn.functional as F

import hyper_multicore
from hyper_multicore.modules.mega_moe.probability import (
    ProbabilityLayout, PROBABILITY_POSITION_FORWARD, PROBABILITY_POSITION_BACKWARD,
)
from tests.common.multicore.fp32_expert_reference import fp32_expert_reference


class TestProbabilityAbi(unittest.TestCase):
    """Validate real native consumers and independent rounded-operand reference math."""

    def test_probability_descriptors_reject_incompatible_units(self) -> None:
        """Only the declared FP32 planes and reserved parameter ports are accepted."""
        for position in (PROBABILITY_POSITION_FORWARD, PROBABILITY_POSITION_BACKWARD):
            layout = ProbabilityLayout(512, position)
            self.assertEqual(ProbabilityLayout.from_descriptor(layout.descriptor()), layout)
            self.assertEqual(layout.tensor_spec().shape, [4, 512])
            for field, value in (("data_type", 2), ("input_position", 0), ("base_ptr_offset", 1)):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    descriptor = layout.descriptor()
                    setattr(descriptor, field, value)
                    ProbabilityLayout.from_descriptor(descriptor)

    def test_both_architectures_copy_probability_rows_before_token_elements(self) -> None:
        """Compile the production helper and exercise empty, tail and unweighted transfers."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required for the native transport regression")
        root = Path(hyper_multicore.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix="hp-probability-transport-") as directory:
            work = Path(directory)
            (work / "kernel_operator.h").write_text("#pragma once\nusing GM_ADDR = uint8_t *;\n")
            for architecture in (2201, 3510):
                binary = work / f"transport-{architecture}"
                subprocess.run([
                    compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    f"-D__NPU_ARCH__={architecture}", "-I", str(root), "-I", str(work),
                    str(Path(__file__).with_name("probability_transport_test.cpp")), "-o", str(binary),
                ], check=True)
                subprocess.run([str(binary)], check=True)

    def test_fp32_reference_matches_autograd_at_bf16_boundaries(self) -> None:
        """FP32 leaves preserve accumulation while casts reproduce native stored operands."""
        tokens = torch.full((2, 4), .25, dtype=torch.bfloat16)
        for layout in ("EIO", "EOI"):
            first = torch.full((4, 6), .5, dtype=torch.bfloat16)
            second = torch.full((3, 4), .5, dtype=torch.bfloat16)
            if layout == "EOI":
                first, second = first.T, second.T
            output, dw1, dw2 = fp32_expert_reference(tokens, first, second, layout)
            w1, w2 = first.float().requires_grad_(), second.float().requires_grad_()
            projection = (tokens.float() @ (w1.T if layout == "EOI" else w1)).bfloat16().float()
            gate, up = projection.chunk(2, dim=-1)
            activation = (F.silu(gate) * up).bfloat16().float()
            reference = (activation @ (w2.T if layout == "EOI" else w2)).bfloat16()
            reference.float().square().mean().backward()
            self.assertEqual(dw1.shape, first.shape)
            self.assertEqual(dw2.shape, second.shape)
            self.assertEqual(dw1.dtype, torch.float32)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            torch.testing.assert_close(dw1, w1.grad, rtol=0, atol=0)
            torch.testing.assert_close(dw2, w2.grad, rtol=0, atol=0)
