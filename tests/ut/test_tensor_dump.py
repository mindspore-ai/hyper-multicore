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
"""Unit tests for schedule-driven Torch MegaKernel Tensor dump."""

import errno
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import csv
from dataclasses import replace
from io import StringIO
from os import chdir
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest import mock
from weakref import ref

from torch import (
    OutOfMemoryError,
    Tensor,
    bfloat16 as torch_bfloat16,
    load as torch_load,
    ones as torch_ones,
    qint8 as torch_qint8,
    quantize_per_tensor,
    tensor as torch_tensor,
    zeros as torch_zeros,
)
from torch.testing import assert_close

from hyper_multicore.diagnostics import prepare_mega_kernel_diagnostics, schedule
from hyper_multicore.diagnostics import profiler as torch_profiler
from hyper_multicore.diagnostics.dumper import DumpConfig, DumpLevel, compare_tensors, dump, load_tensor
from hyper_multicore.diagnostics.common.call import KernelDumpRowGroups
from hyper_multicore.scheduler.runtime import allocate_runtime_config
from hyper_multicore.diagnostics.dumper import api as dumper_api
from hyper_multicore.diagnostics.dumper import _layout, _summary
from hyper_multicore.diagnostics.dumper._layout import _stage_names
from hyper_multicore.diagnostics.common.session import _reset_active_session_for_test
from hyper_multicore.scheduler.graph import (
    ComputeGraph,
    OperatorNode,
    OpType,
    SplitSpec,
    TensorSpec,
)
from tests.common.mark_utils import arg_mark
from tests.common.multicore._dump_faults import fail_dump_output_payload_write


def _make_test_graph() -> ComputeGraph:
    """Build a two-stage graph with all three Tensor roles for Kernel dump."""
    graph_input = TensorSpec("input", [2], param_position=0)
    intermediate = TensorSpec("intermediate", [2], param_position=1)
    graph_output = TensorSpec("output", [2], param_position=2)
    first = OperatorNode(
        "up_proj", OpType.GMM, [graph_input], [intermediate], [0, 1], 1,
        SplitSpec(None, lambda _: 1, [0]), -1, None, diagnostic_stage_name="GMM1",
    )
    second = OperatorNode(
        "down_proj", OpType.GMM, [intermediate], [graph_output], [1, 2], 1,
        SplitSpec([(0, 0)], lambda _: 1, [0]), -1, None, diagnostic_stage_name="GMM2",
    )
    return ComputeGraph().add_op(first).add_op(second).add_edge(first, second)


class TestTensorDumpConfig(unittest.TestCase):
    """Reject contradictory or invalid public dump configuration."""


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_unsupported_level_and_invalid_rank(self):
        """Feature: MegaKernel diagnostics.
        Description: Rejects unsupported level and invalid rank.
        Expectation: Rejects unsupported level and invalid rank.
        """
        for level in ("L0", "L1", "L2", "", None, 0, True):
            with self.subTest(level=level), self.assertRaisesRegex(ValueError, "unsupported Tensor dump level"):
                dumper_api.DumpConfig("dump", level=level)
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            dumper_api.DumpConfig("dump", level="task")
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            dumper_api.DumpConfig("dump", ranks=True)

    def test_options_must_apply_to_selected_level(self):
        """Reject wrong-level options while keeping the shared budget default."""
        options = {
            "task_ids": 0, "stage_task_indices": 0, "owner_ids": 0, "core_types": "AIV",
            "core_ids": 0, "max_bytes_per_tensor": 4, "device_memory_budget_mb": 1024,
        }
        for name, value in options.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"{name} .* Task dump"):
                DumpConfig("dump", **{name: value})
        with self.assertRaisesRegex(ValueError, "roles .* Kernel dump"):
            DumpConfig("dump", level=DumpLevel.TASK, task_ids=0, roles="final_output")
        kernel_config = DumpConfig("dump", device_memory_budget_mb=512)
        task_config = DumpConfig("dump", level=DumpLevel.TASK, task_ids=0)
        self.assertEqual(replace(kernel_config, ranks=1).device_memory_budget_mb, 512)
        self.assertEqual(task_config.device_memory_budget_mb, 512)
        self.assertEqual(replace(task_config, device_memory_budget_mb=1024).device_memory_budget_mb, 1024)


class TestGraphDiagnosticMetadata(unittest.TestCase):
    """Validate generic and explicitly overridden operator stage names."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_stage_names_do_not_depend_on_operator_type(self):
        """Feature: MegaKernel diagnostics.
        Description: Default stage names do not depend on operator type.
        Expectation: Default stage names do not depend on operator type.
        """
        operators = _make_test_graph().topological_sort()
        for operator in operators:
            operator.diagnostic_stage_name = None

        self.assertEqual(
            _stage_names(operators),
            {"up_proj": "UpProj", "down_proj": "DownProj"},
        )


class TestTensorDumpSession(unittest.TestCase):
    """Validate capture boundaries, metadata, filters and diagnostic isolation."""

    def setUp(self) -> None:
        """Prepare the shared two-stage test graph."""
        self.adapter = prepare_mega_kernel_diagnostics(
            _make_test_graph(), allocate_runtime_config(0), kernel_name="test_kernel", parameter_count=3, device="cpu",
        )

    def tearDown(self) -> None:
        """Reset the process-local diagnostic session after each test."""
        _reset_active_session_for_test()

    def _launch_graph(self, *, repeats=1, row_groups=None, adapter=None, direction="forward"):
        """Run a CPU stand-in with an output bound after launch preparation."""
        adapter = self.adapter if adapter is None else adapter
        graph_input = torch_tensor([1.0, 2.0]).repeat(repeats)
        parameters = [graph_input, torch_zeros(graph_input.shape), None]
        with adapter.launch(
            parameters, direction=direction, event_counters=torch_zeros(16), dump_row_groups=row_groups,
        ):
            parameters[1].copy_(parameters[0] + 1)
            parameters[2] = parameters[1] * 2
        return parameters[2]

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_window_preview_limits_reductions_and_preserves_file_outputs(self):
        """Ten calls capture thirty tensors. Only CSV needs statistics beyond the eight preview rows."""
        for outputs in (("log",), ("tensor", "log"), ("csv", "log")):
            with self.subTest(outputs=outputs), tempfile.TemporaryDirectory() as directory:
                config = DumpConfig(directory, outputs=outputs, tags={"step": 50})
                groups = KernelDumpRowGroups(parameter_positions=(1,), resolve_rows=lambda: [(4, 0, 1)])
                callback = mock.Mock()
                with (
                    redirect_stdout(StringIO()) as logs,
                    mock.patch.object(_summary, "_tensor_summary", wraps=_summary._tensor_summary) as reduce_tensor,
                    dump(config, on_dump_ready=callback) as session,
                ):
                    for _ in range(10):
                        self._launch_graph(row_groups=groups)

                header, *preview = logs.getvalue().splitlines()
                self.assertEqual(header, "Dump summary rank=0 window=0 showing=8/30")
                self.assertEqual(len(preview), 8)
                self.assertIn("step=0 invocation=0 GMM1.output[0], GMM2.input[0] phase=after", preview[1])
                self.assertIn("mean=2 nonfinite=0 values=[2]", preview[1])
                self.assertEqual(reduce_tensor.call_count, 30 if "csv" in outputs else 8)
                self.assertEqual(callback.call_count, int("tensor" in outputs))

                if "csv" in outputs:
                    csv_path = Path(directory) / "rank_0" / "window_0000" / "summary.csv"
                    self.assertEqual(list(csv_path.parent.iterdir()), [csv_path])
                    with csv_path.open(encoding="utf-8", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 30)
                    row = rows[1]
                    self.assertEqual(row["range"], "expert_rows")
                    self.assertEqual(json.loads(row["expert_rows"]), [[4, 0, 1]])
                    self.assertEqual(json.loads(row["stored_shape"]), [2])
                    self.assertEqual(float(row["mean"]), 2.0)
                    self.assertEqual(json.loads(row["ports"]), [["GMM1", "output", 0], ["GMM2", "input", 0]])
                    self.assertEqual(json.loads(row["tags"]), {"step": 50})
                elif "tensor" in outputs:
                    manifest, = session.last_manifest_paths
                    calls = json.loads(manifest.read_text(encoding="utf-8"))["invocations"]
                    self.assertEqual([len(call["tensors"]) for call in calls], [3] * 10)
                    last_output = load_tensor(manifest, invocation_id=9, port=("GMM2", "output", 0))
                    assert_close(last_output, torch_tensor([4., 6.]))
                else:
                    self.assertEqual(list(Path(directory).iterdir()), [])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_failed_concurrent_preview_leaves_rows_for_the_next_call(self):
        """A row resolver can wait for another call to publish the remaining preview rows."""
        resolving, completed = Event(), Event()

        def fail_rows() -> list[tuple[int, int, int]]:
            """Wait for the other invocation to finish before reporting invalid rows."""
            resolving.set()
            if not completed.wait(5):
                raise AssertionError("timed out waiting for the second call")
            raise ValueError("invalid expert rows")

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as logs:
            with dump(DumpConfig(directory, outputs="log")), ThreadPoolExecutor(2) as workers:
                self._launch_graph()
                first = workers.submit(self._launch_graph, row_groups=KernelDumpRowGroups((1,), fail_rows))
                self.assertTrue(resolving.wait(5))
                try:
                    workers.submit(self._launch_graph).result(timeout=5)
                finally:
                    completed.set()
                first.result(timeout=5)
                self._launch_graph()
            header, *preview = logs.getvalue().splitlines()
            self.assertEqual(header, "Dump summary rank=0 window=0 showing=8/12")
            self.assertEqual([line.split()[1] for line in preview],
                             ["invocation=0"] * 3 + ["invocation=2"] * 3 + ["invocation=3"] * 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_kernel_summary_selects_the_saved_stage_port(self):
        """Follow a CSV row to the downstream input and locate its first tolerance violation."""
        with tempfile.TemporaryDirectory() as directory:
            tags = {"source": "data-\udcff"}
            config = DumpConfig(directory, stages="GMM2", io="input", outputs=("tensor", "csv", "log"), tags=tags)
            closed = StringIO()
            closed.close()
            callback = mock.Mock()
            with (redirect_stdout(closed), self.assertLogs("hyper_multicore.diagnostics.dumper._summary", "WARNING"),
                  dump(config, on_dump_ready=callback) as dumper):
                self._launch_graph()
                self._launch_graph(repeats=2)
            manifest, = dumper.last_manifest_paths
            _, invocation = json.loads(manifest.read_text(encoding="utf-8"))["invocations"]
            tensor, = invocation["tensors"]
            callback.assert_called_once_with(dumper)
            self.assertEqual(invocation["tags"], tags)
            self.assertEqual(tensor["capturePhase"], "after")
            self.assertEqual([(port["stage"], port["io"]) for port in tensor["ports"]], [("GMM2", "input")])
            self.assertEqual(tensor["producers"], ["up_proj"])
            with manifest.with_name("summary.csv").open(encoding="utf-8", newline="") as stream:
                _, row = csv.DictReader(stream)
            self.assertEqual(int(row["invocation"]), invocation["invocationId"])
            self.assertEqual(json.loads(row["tags"]), tags)
            port, = map(tuple, json.loads(row["ports"]))
            self.assertEqual(port, ("GMM2", "input", 0))
            payload = load_tensor(manifest, invocation_id=int(row["invocation"]), port=port)
            assert_close(payload, torch_tensor([2., 3., 2., 3.]))
            result = compare_tensors(torch_tensor([100., 0., 2., 3.]), payload, rtol=0.99, atol=0)
            self.assertEqual(result["over_tolerance"], 1)
            self.assertEqual(result["max_error_index"], (0,))
            self.assertEqual(result["first_mismatch_index"], (1,))
            self.assertEqual(result["first_mismatch_reference_value"], 0.)
            self.assertEqual(result["first_mismatch_actual_value"], 3.)
            self.assertIsNone(compare_tensors(payload, payload, rtol=0, atol=0)["first_mismatch_index"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_summary_failure_preserves_tensor_for_offline_comparison(self):
        """Keep the raw snapshot readable when CSV publication fails."""
        with tempfile.TemporaryDirectory() as directory:
            config = DumpConfig(directory, outputs=("tensor", "csv"), tensor_names="intermediate")
            with (
                self.assertLogs(dumper_api.__name__, "WARNING") as logs,
                mock.patch.object(dumper_api, "_write_summary_csv", side_effect=OSError("disk full")),
                dump(config) as dumper,
            ):
                assert_close(self._launch_graph(), torch_tensor([4., 6.]))
            self.assertEqual(len(dumper.last_manifest_paths), 1)
            path = dumper.last_manifest_paths[0]
            self.assertIn("summary write failed", logs.output[0])
            self.assertFalse((path.parent / "summary.csv").exists())
            assert_close(load_tensor(path, invocation_id=0, port=("GMM2", "input", 0)), torch_tensor([2., 3.]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_summary_row_read_failure_preserves_device_error(self):
        """Skip optional summaries on allocation failure and propagate other device errors."""
        for error in (OutOfMemoryError("row copy OOM"), RuntimeError("device synchronization failed")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                config = DumpConfig(directory, outputs=("tensor", "csv"), stages="GMM1", io="output")
                resolve_rows = mock.Mock(side_effect=error)
                groups = KernelDumpRowGroups(parameter_positions=(1,), resolve_rows=resolve_rows)
                if isinstance(error, OutOfMemoryError):
                    with self.assertLogs(dumper_api.__name__, "WARNING") as logs, dump(config) as dumper:
                        self._launch_graph(row_groups=groups)
                    path, = dumper.last_manifest_paths
                    self.assertIn("row copy OOM", logs.output[0])
                    self.assertFalse((path.parent / "summary.csv").exists())
                    assert_close(load_tensor(path, invocation_id=0, port=("GMM1", "output", 0)), torch_tensor([2., 3.]))
                else:
                    with self.assertRaisesRegex(RuntimeError, "device synchronization failed"), dump(config):
                        self._launch_graph(row_groups=groups)
                    self.assertFalse(list(Path(directory).rglob("manifest.json")))
                resolve_rows.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unsupported_cpu_summary_keeps_raw_snapshot(self):
        """Preserve serializable tensors whose dtype cannot be reduced for CSV."""
        output = quantize_per_tensor(torch_tensor([4., 6.]), 0.5, 0, torch_qint8)
        with tempfile.TemporaryDirectory() as directory:
            config = DumpConfig(directory, outputs=("tensor", "csv"), stages="GMM2", io="output")
            operands = [torch_ones(2), torch_zeros(2), None]
            with self.assertLogs(dumper_api.__name__, "WARNING") as logs, dump(config) as dumper:
                with self.adapter.launch(operands, direction="forward", event_counters=torch_zeros(16)):
                    operands[2] = output
            path, = dumper.last_manifest_paths
            self.assertIn("cannot summarize Host Tensor", logs.output[0])
            self.assertFalse((path.parent / "summary.csv").exists())
            actual = load_tensor(path, invocation_id=0, port=("GMM2", "output", 0))
            assert_close(actual, output)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_kernel_tensor_filter_miss_does_not_consume_invocation_id(self):
        """Number the first capture zero and clear its paths after a skip-only window."""
        unselected_graph = _make_test_graph()
        unselected_graph.get_op("down_proj").outputs[0].name = "unselected"
        missed_adapter = prepare_mega_kernel_diagnostics(
            unselected_graph, allocate_runtime_config(0), kernel_name="test_kernel", parameter_count=3, device="cpu",
        )
        capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=2)
        with tempfile.TemporaryDirectory() as output_dir:
            config = DumpConfig(output_dir, tensor_names="output", directions="forward")
            with dump(config, schedule=capture_schedule) as dumper:
                with self.assertNoLogs(dumper_api.__name__, level="WARNING"):
                    self._launch_graph(direction="backward")
                with self.assertLogs(dumper_api.__name__, level="WARNING") as warnings:
                    assert_close(self._launch_graph(adapter=missed_adapter), torch_tensor([4., 6.]))
                    self._launch_graph()
                    dumper.step()
                    manifest_path, = dumper.last_manifest_paths
                    self._launch_graph(adapter=missed_adapter)
                    dumper.step()
            self.assertEqual(len(warnings.output), 1)
            self.assertIn(
                "kernel=test_kernel rank=0 direction=forward step=0 reason=empty_selection", warnings.output[0],
            )
            self.assertNotIn("invocation=", warnings.output[0])
            self.assertEqual(dumper.last_manifest_paths, ())
            self.assertEqual(list(Path(output_dir).rglob("manifest.json")), [manifest_path])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([item["invocationId"] for item in manifest["invocations"]], [0])


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_relative_output_dir_is_anchored_each_time_the_session_enters(self):
        """Publish under the entry directory despite later working-directory changes."""
        original_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as output_dir:
            root = Path(output_dir).resolve()
            dumper = dump(DumpConfig("captures", tensor_names="output"))
            try:
                for window_index in range(2):
                    entry_dir = root / f"entry_{window_index}"
                    entry_dir.mkdir()
                    chdir(entry_dir)
                    with dumper:
                        self._launch_graph()
                        chdir(root)
                    rank_dir = entry_dir / "captures" / "rank_0"
                    window_dir = rank_dir / f"window_{window_index:04d}"
                    self.assertEqual(dumper.last_manifest_paths, (window_dir / "manifest.json",))
                    self.assertEqual(list(rank_dir.iterdir()), [window_dir])
                    manifest = json.loads(dumper.last_manifest_paths[0].read_text(encoding="utf-8"))
                    tensor_path = window_dir / manifest["invocations"][0]["tensors"][0]["file"]
                    value = torch_load(tensor_path, map_location="cpu", weights_only=True)
                    assert_close(value, torch_tensor([4.0, 6.0]))
                self.assertFalse((root / "captures").exists())
            finally:
                chdir(original_dir)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_exports_graph_tensors(self):
        """Feature: MegaKernel diagnostics.
        Description: Exports graph tensors.
        Expectation: Exports graph tensors.
        """
        parameters = [torch_tensor([1.0, 2.0]), torch_zeros(2), torch_zeros(2)]
        snapshot_refs = []
        snapshot_tensor = dumper_api._snapshot_tensor

        def _track_snapshot(*args):
            snapshot = snapshot_tensor(*args)
            snapshot_refs.append(ref(snapshot.data))
            return snapshot

        with tempfile.TemporaryDirectory() as output_dir:
            config = dumper_api.DumpConfig(
                output_dir, kernel_names="test_kernel", directions="forward", ranks=0,
            )
            capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
            with (
                dumper_api.dump(config, schedule=capture_schedule) as dumper,
                mock.patch.object(dumper_api, "_snapshot_tensor", side_effect=_track_snapshot),
            ):
                call = self.adapter.launch(parameters, direction="forward", event_counters=torch_zeros(16))
                parameters[1].copy_(parameters[0] + 1)
                parameters[2].copy_(parameters[1] * 2)
                parameters[0].zero_()
                call.complete()
                rank_dir = Path(output_dir) / "rank_0"
                self.assertFalse((rank_dir / "window_0000").exists())
                self.assertTrue(list(rank_dir.glob(".window_0000.*/invocation_*/stages/*/*.pt")))
                self.assertEqual(len(snapshot_refs), 3)
                self.assertTrue(all(snapshot_ref() is None for snapshot_ref in snapshot_refs))
                self.assertFalse(dumper.last_manifest_paths)
                dumper.step()

            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            invocation = manifest["invocations"][0]
            tensors = {item["name"]: item for item in invocation["tensors"]}
            self.assertEqual(manifest["schemaVersion"], 5)
            self.assertEqual(manifest["level"], "kernel")
            self.assertEqual(manifest["tensorSerialization"], "torch.save")
            self.assertEqual(manifest["tensorFileLayout"], "stage_only")
            self.assertEqual(manifest["capturePhaseSemantics"], "mega_kernel_boundary")
            self.assertNotIn("dumpPhaseSemantics", manifest)
            self.assertEqual(manifest["stageViewStorage"], "primary_file_with_hard_links")
            self.assertEqual(invocation["direction"], "forward")
            self.assertEqual(invocation["kernelName"], "test_kernel")
            self.assertEqual(tensors["input"]["capturePhase"], "before")
            self.assertNotIn("dumpPhase", tensors["input"])
            self.assertEqual(tensors["intermediate"]["role"], "stage_output")
            self.assertEqual(tensors["output"]["role"], "final_output")
            self.assertEqual(
                [(port["stage"], port["io"], port["index"]) for port in tensors["intermediate"]["ports"]],
                [("GMM1", "output", 0), ("GMM2", "input", 0)],
            )
            invocation_dir = manifest_path.parent / "invocation_000000_forward"
            self.assertFalse((invocation_dir / "before").exists())
            self.assertFalse((invocation_dir / "after").exists())
            for metadata in tensors.values():
                primary_path = manifest_path.parent / metadata["file"]
                self.assertEqual(metadata["file"], metadata["ports"][0]["file"])
                for port in metadata["ports"]:
                    stage_path = manifest_path.parent / port["file"]
                    self.assertTrue(stage_path.is_file())
                    self.assertTrue(primary_path.samefile(stage_path))
            assert_close(load_tensor(manifest_path, invocation_id=0, port=("GMM1", "input", 0)),
                         torch_tensor([1.0, 2.0]))
            assert_close(
                torch_load(
                    manifest_path.parent / tensors["output"]["file"],
                    map_location="cpu",
                    weights_only=True,
                ),
                torch_tensor([4.0, 6.0]),
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_partial_write_failure_keeps_other_captures_and_summary(self):
        """Keep good captures and every summary when a partially written call fails."""
        with tempfile.TemporaryDirectory() as output_dir:
            callback = mock.Mock()
            config = DumpConfig(output_dir, outputs=("tensor", "csv"))
            with dump(config, on_dump_ready=callback) as dumper:
                self._launch_graph()
                with (
                    fail_dump_output_payload_write(),
                    mock.patch.object(
                        dumper_api, "rmtree", side_effect=PermissionError(errno.EACCES, "cleanup failed"),
                    ),
                    self.assertLogs(dumper_api.__name__, level="WARNING") as warnings,
                ):
                    assert_close(self._launch_graph(repeats=1024), torch_tensor([4.0, 6.0]).repeat(1024))
                self.assertTrue(any("reason=write_failed invocation=1: [Errno 28]" in line for line in warnings.output))
                self.assertTrue(any("cleanup failed" in line for line in warnings.output))
                self._launch_graph()
                callback.assert_not_called()
            callback.assert_called_once_with(dumper)
            self.assertEqual(len(dumper.last_manifest_paths), 1)
            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([item["invocationId"] for item in manifest["invocations"]], [0, 2])
            self.assertEqual(
                sorted(path.name for path in manifest_path.parent.glob("invocation_*")),
                ["invocation_000000_forward", "invocation_000002_forward"],
            )
            self.assertTrue(list(manifest_path.parent.parent.glob(".invocation_000001_forward.*/*/stages/*/*.pt")))
            with manifest_path.with_name("summary.csv").open(encoding="utf-8", newline="") as stream:
                output_means = {int(row["invocation"]): float(row["mean"]) for row in csv.DictReader(stream)
                                if row["tensor"] == "output"}
            self.assertEqual(output_means, {0: 5.0, 1: 5.0, 2: 5.0})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_window_publication_failure_cleans_staging(self):
        """Leave old windows intact and remove staged files on publication failure."""
        capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
        for reason in ("collision", "rename"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as output_dir:
                callback = mock.Mock()
                target = Path(output_dir) / "rank_0" / "window_0000"
                config = DumpConfig(output_dir)
                with self.assertLogs(dumper_api.__name__, level="WARNING") as warnings:
                    with dump(config, schedule=capture_schedule, on_dump_ready=callback) as dumper:
                        self._launch_graph()
                        if reason == "collision":
                            target.mkdir()
                            (target / "manifest.json").write_text("original", encoding="utf-8")
                        else:
                            with mock.patch.object(Path, "rename", side_effect=OSError(errno.EIO, "rename failed")):
                                dumper.step()
                self.assertIn("reason=publish_failed", warnings.output[0])
                self.assertFalse(dumper.last_manifest_paths)
                self.assertFalse(list(target.parent.glob(".window_*")))
                callback.assert_not_called()
                if reason == "collision":
                    self.assertEqual((target / "manifest.json").read_text(encoding="utf-8"), "original")
                else:
                    self.assertFalse(target.exists())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_manifest_write_failure_preserves_summary_and_other_rank(self):
        """Publish only the CSV when one rank cannot index its raw snapshots."""
        adapters = [
            prepare_mega_kernel_diagnostics(
                _make_test_graph(), allocate_runtime_config(0), kernel_name="test_kernel",
                parameter_count=3, device="cpu", rank=rank,
            )
            for rank in (0, 1)
        ]
        def _write_manifest(path: Path, data: str, *, encoding: str) -> int:
            with path.open("w", encoding=encoding) as stream:
                if path.name == "manifest.json" and path.parent.parent.name == "rank_0":
                    stream.write("{")
                    raise OSError(errno.ENOSPC, "manifest disk full")
                return stream.write(data)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as log:
            config = DumpConfig(directory, outputs=("tensor", "csv", "log"), stages="GMM2", io="output")
            callback = mock.Mock()
            with (
                self.assertLogs(dumper_api.__name__, "WARNING") as warnings,
                mock.patch.object(Path, "write_text", new=_write_manifest),
                dump(config, on_dump_ready=callback) as dumper,
            ):
                for adapter in adapters:
                    assert_close(self._launch_graph(adapter=adapter), torch_tensor([4., 6.]))
            rank0 = Path(directory) / "rank_0"
            self.assertEqual(
                [path.relative_to(rank0).as_posix() for path in rank0.rglob("*") if path.is_file()],
                ["window_0000/summary.csv"],
            )
            with (rank0 / "window_0000" / "summary.csv").open(encoding="utf-8", newline="") as stream:
                row, = list(csv.DictReader(stream))
            self.assertEqual(float(row["mean"]), 5.0)
            self.assertIn("rank=0 window=0 showing=1/1", log.getvalue())
            self.assertIn("manifest write failed for rank=0", warnings.output[0])
            rank1 = Path(directory) / "rank_1" / "window_0000"
            self.assertEqual(dumper.last_manifest_paths, (rank1 / "manifest.json",))
            assert_close(load_tensor(dumper.last_manifest_paths[0], invocation_id=1, port=("GMM2", "output", 0)),
                         torch_tensor([4., 6.]))
            self.assertTrue((rank1 / "summary.csv").is_file())
            self.assertFalse(list(Path(directory).glob("rank_*/.window_*")))
            callback.assert_called_once_with(dumper)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_serializer_error_does_not_recover_an_unrelated_file_error(self):
        """Only errors from the actual output stream are classified as file failures."""
        with tempfile.TemporaryDirectory() as output_dir:
            path = Path(output_dir) / "tensor.pt"
            cause = OSError("serialization I/O error")
            runtime_error = RuntimeError("serialization failure")
            runtime_error.__cause__ = cause
            for error, message in ((runtime_error, "serialization failure"), (cause, "failed to serialize Tensor")):
                with (
                    mock.patch.object(_layout.torch, "save", side_effect=error),
                    self.assertRaisesRegex(RuntimeError, message) as failure,
                ):
                    _layout._save_tensor(torch_zeros(2), path)
                self.assertIs(failure.exception.__cause__, cause)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_exception_discards_staged_window_and_keeps_published_window(self):
        """Keep already published results when a later business call raises."""
        capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=2)
        with tempfile.TemporaryDirectory() as output_dir:
            config = DumpConfig(output_dir)
            with self.assertRaisesRegex(RuntimeError, "business failure"):
                with dump(config, schedule=capture_schedule) as dumper:
                    self._launch_graph()
                    dumper.step()
                    self._launch_graph()
                    operands = [torch_ones(2), torch_zeros(2), torch_zeros(2)]
                    with self.adapter.launch(operands, direction="forward", event_counters=torch_zeros(16)) as call:
                        raise RuntimeError("business failure")
            self.assertTrue(call.finished)
            rank_dir = Path(output_dir) / "rank_0"
            self.assertEqual(sorted(path.name for path in rank_dir.iterdir()), ["window_0000"])
            self.assertEqual(dumper.last_manifest_paths, (rank_dir / "window_0000" / "manifest.json",))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_callback_failure_still_releases_the_session(self):
        """A failed callback leaves the published window and releases session ownership."""
        with tempfile.TemporaryDirectory() as output_dir:
            config = DumpConfig(output_dir)
            callback = mock.Mock(side_effect=RuntimeError("callback failure"))
            with (
                self.assertRaisesRegex(RuntimeError, "callback failure"),
                dump(config, on_dump_ready=callback) as dumper,
            ):
                self._launch_graph()
            callback.assert_called_once_with(dumper)
            manifest_path = Path(output_dir) / "rank_0" / "window_0000" / "manifest.json"
            self.assertEqual(dumper.last_manifest_paths, (manifest_path,))
            self.assertTrue(manifest_path.is_file())
            with dump(config) as next_dumper:
                self.assertFalse(next_dumper.last_manifest_paths)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_kernel_preparation_oom_preserves_launch_and_published_results(self):
        """Skip lazy metadata and Tensor selection OOMs before allocating a capture ID."""
        capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=3)
        for preparation in ("metadata", "tensor_selection"):
            with self.subTest(preparation=preparation), tempfile.TemporaryDirectory() as output_dir:
                if preparation == "metadata":
                    failure = mock.patch.object(
                        type(self.adapter._runtime), "metadata", new_callable=mock.PropertyMock,
                        side_effect=MemoryError("metadata allocation"),
                    )
                else:
                    failure = mock.patch.object(
                        dumper_api, "_describe_graph_tensors", side_effect=OutOfMemoryError("Tensor selection"),
                    )
                with dump(DumpConfig(output_dir), schedule=capture_schedule) as dumper:
                    self._launch_graph()
                    dumper.step()
                    first_manifest = dumper.last_manifest_paths[0]
                    with failure, self.assertLogs(dumper_api.__name__, level="WARNING") as logs:
                        assert_close(self._launch_graph(), torch_tensor([4.0, 6.0]))
                    self.assertIn("step=1 reason=allocation_oom", logs.output[0])
                    self.assertNotIn("invocation=", logs.output[0])
                    dumper.step()
                    self.assertEqual(dumper.last_manifest_paths, ())
                    self._launch_graph()
                last_manifest = dumper.last_manifest_paths[0]
                self.assertEqual(sorted(Path(output_dir).rglob("manifest.json")), [first_manifest, last_manifest])
                for invocation_id, path in enumerate((first_manifest, last_manifest)):
                    invocation = json.loads(path.read_text(encoding="utf-8"))["invocations"][0]
                    self.assertEqual(invocation["invocationId"], invocation_id)
                    output = next(tensor for tensor in invocation["tensors"] if tensor["name"] == "output")
                    tensor = torch_load(path.parent / output["file"], weights_only=True)
                    assert_close(tensor, torch_tensor([4.0, 6.0]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_kernel_capture_oom_uses_business_result(self):
        """Skip capture on either side of the business launch when allocation fails."""
        clone = Tensor.clone
        for phase, clone_results in (
            ("before", [MemoryError("input snapshot allocation failed")]),
            ("after", [None, OutOfMemoryError("output snapshot allocation failed")]),
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as output_dir:
                allocations = iter(clone_results)

                def _clone(tensor):
                    error = next(allocations)
                    if error is not None:
                        raise error
                    return clone(tensor)

                with (
                    mock.patch.object(Tensor, "clone", new=_clone),
                    self.assertLogs(dumper_api.__name__, level="WARNING") as warnings,
                    dump(DumpConfig(output_dir)) as dumper,
                ):
                    assert_close(self._launch_graph(), torch_tensor([4.0, 6.0]))
                self.assertIn("reason=allocation_oom", warnings.output[0])
                self.assertFalse(dumper.last_manifest_paths)
                self.assertFalse(list(Path(output_dir).iterdir()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_late_completion_stays_out_of_the_next_window(self):
        """A call prepared before step belongs to the window that step finalized."""
        capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=2)
        with tempfile.TemporaryDirectory() as output_dir:
            with dump(DumpConfig(output_dir), schedule=capture_schedule) as dumper:
                operands = [torch_ones(2), torch_zeros(2), torch_zeros(2)]
                call = self.adapter.launch(operands, direction="forward", event_counters=torch_zeros(16))
                dumper.step()
                call.complete()
                self.assertTrue(call.finished)
                self._launch_graph()
                dumper.step()
            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["windowIndex"], 1)
            self.assertEqual([item["invocationId"] for item in manifest["invocations"]], [1])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_late_completion_after_exit_does_not_stage_files(self):
        """Reject a finished call from a closed window, including after reentry."""
        for reenter in (False, True):
            with self.subTest(reenter=reenter), tempfile.TemporaryDirectory() as output_dir:
                with dump(DumpConfig(output_dir)) as dumper:
                    operands = [torch_ones(2), torch_zeros(2), torch_zeros(2)]
                    call = self.adapter.launch(operands, direction="forward", event_counters=torch_zeros(16))
                if reenter:
                    with dumper:
                        call.complete()
                else:
                    call.complete()
                self.assertTrue(call.finished)
                self.assertFalse(dumper.last_manifest_paths)
                self.assertFalse(list(Path(output_dir).iterdir()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_untyped_allocator_error_propagates(self):
        """Keep an untyped CPU allocation error visible at the capture boundary."""
        allocator_error = RuntimeError(
            "[enforce fail at alloc_cpu.cpp:113] data. DefaultCPUAllocator: "
            "not enough memory: you tried to allocate 8 bytes."
        )
        with (
            tempfile.TemporaryDirectory() as output_dir,
            mock.patch.object(Tensor, "clone", side_effect=allocator_error),
            dump(DumpConfig(output_dir)) as dumper,
        ):
            with self.assertRaises(RuntimeError) as raised:
                self._launch_graph()
            self.assertIs(raised.exception, allocator_error)
        self.assertFalse(dumper.last_manifest_paths)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_filters_and_preserves_bfloat16(self):
        """Feature: MegaKernel diagnostics.
        Description: Filters and preserves bfloat16.
        Expectation: Filters and preserves bfloat16.
        """
        parameters = [torch_ones(2, dtype=torch_bfloat16) for _ in range(3)]
        with tempfile.TemporaryDirectory() as output_dir:
            config = dumper_api.DumpConfig(
                output_dir, tensor_names="input", roles="external_input", directions="forward",
            )
            with dumper_api.dump(config) as dumper:
                call = self.adapter.launch(parameters, direction="forward", event_counters=torch_zeros(16))
                call.complete()

            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = manifest["invocations"][0]["tensors"][0]
            payload = torch_load(
                manifest_path.parent / metadata["file"],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(metadata["originalDtype"], "bfloat16")
            self.assertEqual(metadata["storedDtype"], "bfloat16")
            self.assertEqual(payload.dtype, torch_bfloat16)
            assert_close(payload, torch_ones(2, dtype=torch_bfloat16))
            self.assertEqual(manifest["invocations"][0]["tensorCount"], 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_profiler_and_tensor_dump_are_mutually_exclusive(self):
        """Feature: MegaKernel diagnostics.
        Description: Profiler and tensor dump are mutually exclusive.
        Expectation: Profiler and tensor dump are mutually exclusive.
        """
        config = DumpConfig("dump")
        with torch_profiler.profile() as session:
            self.assertIsInstance(session, torch_profiler.ProfilerSession)
            with self.assertRaisesRegex(RuntimeError, "active=profiler, requested=dumper"):
                with dump(config):
                    pass
        with dump(config):
            with self.assertRaisesRegex(RuntimeError, "active=dumper, requested=profiler"):
                with torch_profiler.profile():
                    pass
