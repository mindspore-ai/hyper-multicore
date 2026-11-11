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
"""Host/native L1 address contracts for indirect shared tensors and vector tails."""

import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch

import hyper_multicore
from hyper_multicore import prepare_mega_kernel_diagnostics
from hyper_multicore.diagnostics import dump as tensor_dump
from hyper_multicore.diagnostics.dump.l1 import L1TensorView, L1TensorViewContext
from hyper_multicore.modules.mega_moe.diagnostics import bind_shared_diagnostic_ports
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.tensor_dump import resolve_mega_moe_l1_tensor_view


class TestSharedTensorDump(unittest.TestCase):
    """Exercise actual native address lookup and public L1 record preparation on CPU."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compile production capture code with only device cache primitives stubbed."""
        compiler = shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("g++ is required for the native L1 address regression")
        with tempfile.TemporaryDirectory(prefix="multicore-l1-reader-") as directory:
            temporary = Path(directory)
            (temporary / "kernel_operator.h").write_text("", encoding="utf-8")
            library = temporary / "l1_reader.so"
            include_root = Path(hyper_multicore.__file__).resolve().parent.parent
            subprocess.run([
                compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                "-I", str(temporary), "-I", str(include_root), str(Path(__file__).with_name("l1_reader.cpp")),
                "-o", str(library),
            ], check=True)
            cls.reader = ctypes.CDLL(str(library))
        cls.reader.resolve_l1_address.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        cls.reader.resolve_l1_address.restype = ctypes.c_uint64
        cls.reader.capture_l1.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool]
        cls.reader.capture_l1.restype = ctypes.c_bool

    def test_native_shared_ports_and_invalid_bounds(self) -> None:
        """Forward/backward sidecars use pointer-table slots while ordinary ports stay direct."""
        ordinary = (ctypes.c_uint64 * 48)(*range(1000, 1048))
        for mode, base, count in ((1, 29, 6), (2, 35, 12)):
            table = (ctypes.c_uint64 * (count + 1))(0, *range(2000, 2000 + count))
            with self.subTest(mode=mode):
                for slot in range(count):
                    self.assertEqual(self.reader.resolve_l1_address(ordinary, base + slot, table, mode), 2000 + slot)
                self.assertEqual(self.reader.resolve_l1_address(ordinary, base - 1, table, mode), 1000 + base - 1)
                self.assertEqual(self.reader.resolve_l1_address(ordinary, base + count, table, mode), 0)
                self.assertEqual(self.reader.resolve_l1_address(ordinary, base, None, mode), 0)
        self.assertEqual(self.reader.resolve_l1_address(ordinary, 40, None, 0), 1040)

    @staticmethod
    def _runtime_artifacts(tokens: int) -> tuple:
        """Build full-fusion graphs for a real invocation inside a larger group bucket."""
        spec = MegaMoeSpec(
            tokens, 5120, 128, 4, 2, None, 1024, 2, None, 0, 20, schedule_tokens=512, auto_tokens=True,
            forward_fusion_mode="both", backward_fusion_mode="both", shared_intermediate_size=256,
            forward_shared_expert=True, backward_shared_expert=True,
        )
        return _build_runtime_artifacts(spec)

    @classmethod
    def _prepare_capture(cls, tokens: int, forward: bool) -> tuple:
        """Bind real logical sidecars independently of the native pointer-list layout."""
        fg, fc, bg, bc = cls._runtime_artifacts(tokens)
        graph, config = (fg, fc) if forward else (bg, bc)
        graph = bind_shared_diagnostic_ports(graph, config, forward=forward)
        stages = ("shared_swiglu",) if forward else ("shared_swiglu_grad", "shared_dx_add")
        parameters = [torch.empty(0, dtype=torch.bfloat16)] * (35 if forward else 47)
        for stage in stages:
            operator = graph.get_op(stage)
            for tensor in operator.inputs + operator.outputs:
                values = torch.arange(tensor.shape[0], dtype=torch.float32).reshape(-1, 1) + tensor.param_position
                parameters[tensor.param_position] = values.expand(tensor.shape).contiguous().bfloat16()
        adapter = prepare_mega_kernel_diagnostics(
            graph, config, kernel_name="mega_moe", parameter_count=len(parameters), device="cpu",
        )
        return adapter, parameters, stages

    def test_route_endpoint_tail_views(self) -> None:
        """Route reduction and router-gradient snapshots expose only their written tail."""
        for tokens in (0, 129, 257):
            fg, fc, bg, bc = self._runtime_artifacts(tokens)
            for graph, config in ((fg, fc), (bg, bc)):
                cursor = 0
                for operator in graph.topological_sort():
                    last_task = config.all_tasks[cursor + operator.task_num - 1]
                    cursor += operator.task_num
                    if operator.name not in {"weighted_unpermute", "unpermute_grad", "topk_weight_grad"}:
                        continue
                    tensor = operator.outputs[0]
                    context = L1TensorViewContext(
                        operator=operator, task_id=cursor - 1, owner_id=None, task_desc=last_task,
                        tensor_desc=last_task.outputs[0], tensor_spec=tensor, io="output", index=0,
                        parameters=(), runtime_config=config, rank=0, stage=operator.name,
                        default_view=L1TensorView(0, tuple(tensor.shape)), parameter_cache={},
                    )
                    view = resolve_mega_moe_l1_tensor_view(context)
                    with self.subTest(tokens=tokens, stage=operator.name):
                        element_size = 4 if operator.name == "topk_weight_grad" else 2
                        self.assertEqual(last_task.outputs[0].data_type, element_size)
                        self.assertEqual(view.shape, (min(tokens, 1), tensor.shape[1]))
                        expected_offset = max(0, tokens - 1) * tensor.shape[1] * element_size
                        self.assertEqual(view.source_offset_bytes, expected_offset)

    def _capture_records(self, call, parameters, forward: bool) -> None:
        """Copy from real sidecars; poison the same indices in the direct native list."""
        # The private prepared record table is the host/native wire contract under test.
        records = call._prepared_l1.records  # pylint: disable=protected-access
        base, count = (29, 6) if forward else (35, 12)
        poison = torch.full((512, 5120), -1, dtype=torch.bfloat16)
        ordinary = (ctypes.c_uint64 * 48)(*[poison.data_ptr()] * 48)
        table = (ctypes.c_uint64 * (count + 1))(0, *(value.data_ptr() for value in parameters[base:]))
        for task, phase in {(record.task_id, 1 if record.io == "input" else 2) for record in records}:
            self.assertTrue(self.reader.capture_l1(call.diagnostic_buffer.data_ptr(), ordinary, table,
                                                   task, phase, forward))

    def test_shared_tail_and_empty_snapshots_use_exact_rows(self) -> None:
        """Public L1 snapshots match row128/256 tails and empty vector/residual operands."""
        for tokens in (0, 129, 257):
            for forward in (True, False):
                with self.subTest(tokens=tokens, forward=forward), tempfile.TemporaryDirectory() as directory:
                    adapter, parameters, stages = self._prepare_capture(tokens, forward)
                    selection = tensor_dump.TensorDumpConfig(
                        directory, level="L1", stages=stages, stage_task_indices=max(0, (tokens - 1) // 128),
                    )
                    with tensor_dump.tensor_dump(selection) as dumper:
                        with adapter.launch(parameters, direction="forward" if forward else "backward") as call:
                            self._capture_records(call, parameters, forward)
                    self._check_manifest(dumper.last_manifest_paths[0], parameters, stages, tokens)

    def _check_manifest(self, path: Path, parameters: list, stages: tuple, tokens: int) -> None:
        """Verify exported addresses, row extents and data rather than only record existence."""
        manifest = json.loads(path.read_text(encoding="utf-8"))
        tasks = manifest["invocations"][0]["tasks"]
        self.assertEqual({task["stage"] for task in tasks}, set(stages))
        start = max(0, tokens - 1)
        for task in tasks:
            for item in task["tensors"]:
                parameter = parameters[item["paramPosition"]]
                self.assertEqual(item["sourceOffsetBytes"], start * parameter.shape[1] * parameter.element_size())
                self.assertEqual(item["actualShape"], [min(tokens, 1), parameter.shape[1]])
                actual = torch.load(path.parent / item["file"], weights_only=True)
                torch.testing.assert_close(actual, parameter[start:tokens], rtol=0, atol=0)
