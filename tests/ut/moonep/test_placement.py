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
"""CPU contracts for planner count formats and optional execution dependencies."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_multicore.moonep import placement


class TestMoonEPPlacement(unittest.TestCase):
    """Reject invalid metadata before any optional device kernel is called."""

    def test_native_gmm_uses_owned_weights_without_mutating_shared_bank(self) -> None:
        """Compute staging is read-only and preserves transposed projection values."""
        from hyper_multicore.moonep import grouped_experts
        for dtype in (torch.float32, torch.bfloat16):
            for transposed in (False, True):
                with self.subTest(dtype=dtype, transposed=transposed):
                    bank = torch.arange(48, dtype=dtype).reshape(2, 4, 6)
                    weight = bank.transpose(1, 2) if transposed else bank
                    before = weight.clone()
                    x = torch.ones(3, weight.shape[1], dtype=dtype)
                    counts = torch.tensor([1, 2], dtype=torch.int64)
                    def consume(inputs, weights, **kwargs):
                        staged = weights[0]
                        self.assertNotEqual(staged.data_ptr(), weight.data_ptr())
                        self.assertEqual(staged.stride(), weight.stride())
                        torch.testing.assert_close(staged, weight, rtol=0, atol=0)
                        self.assertIs(inputs[0], x)
                        self.assertIs(kwargs['group_list'], counts)
                        output = torch.cat([rows @ matrix for rows, matrix in
                            zip(x.split((1, 2)), staged)])
                        staged.zero_()
                        return [output]
                    api = SimpleNamespace(npu_grouped_matmul=consume)
                    with patch.dict(sys.modules, {"torch_npu": api}):
                        actual = grouped_experts._gmm(x, weight, counts)
                    expected = torch.cat([rows @ matrix for rows, matrix in
                        zip(x.split((1, 2)), before)])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(weight, before, rtol=0, atol=0)

    def test_vector_transfer_width_uses_device_limit_without_changing_udma(self) -> None:
        """Pure-vector launches use queried capacity; invalid limits fail explicitly."""
        from hyper_multicore.moonep import official_transfer
        device = SimpleNamespace(type="npu", index=0)
        npu = SimpleNamespace(get_device_limit=Mock(return_value={"vector_core_num": 48}))
        with patch.object(torch, "npu", npu, create=True):
            self.assertEqual(official_transfer._transfer_vector_cores(device, 24), 48)
            npu.get_device_limit.assert_called_once_with(0)
            npu.get_device_limit.return_value = {"vector_core_num": 0}
            with self.assertRaisesRegex(RuntimeError, "vector_core_num"):
                official_transfer._transfer_vector_cores(device, 24)
        self.assertEqual(official_transfer._transfer_vector_cores(torch.device("cpu"), 24), 24)

    def test_native_transfers_publish_all_projections_before_returning_gradients(self) -> None:
        """The native executor shares the grouped ABI and keeps FP32 partials until reduction."""
        from hyper_multicore.moonep import official_transfer
        module = ModuleType("npu_patch")
        launchers = ModuleType("npu_patch.launchers")
        launchers._env_use_udma = lambda _kind: False
        native = SimpleNamespace(prefetch_banks=Mock(), reduce_banks=Mock())
        module._C = native
        npu = SimpleNamespace(get_device_properties=lambda _device: SimpleNamespace(cube_core_num=24),
                              Event=Mock())
        allocator = SimpleNamespace(empty=lambda shape, dtype: torch.empty(shape, dtype=dtype), free=Mock())
        with (patch.object(torch, "npu", npu, create=True),
              patch.object(official_transfer.dist, "get_rank", return_value=0),
              patch.object(official_transfer.dist, "get_world_size", return_value=2),
              patch.dict(sys.modules, {"npu_patch": module, "npu_patch.launchers": launchers})):
            resource = official_transfer.ExpertTransfer(None, 4, ((3, 4), (2, 3)),
                torch.device("cpu"), allocator=allocator)
            resource.bind();resource.in_use = True
            resource.vector_cores = 48
            weights = (torch.randn(6, 4).bfloat16(), torch.randn(4, 3).bfloat16())
            copies = torch.tensor([[2], [-1]], dtype=torch.int32)
            native.prefetch_banks.return_value = tuple(p.bank[:3] for p in resource.projections)
            resource.prefetch(weights, copies)
            args = native.prefetch_banks.call_args.args
            self.assertEqual(args[-2:], (48, False))
            self.assertEqual([tuple(w.shape) for w in args[0]], [(2, 3, 4), (2, 2, 3)])
            for actual, original in zip(args[0], weights):
                self.assertEqual(actual.data_ptr(), original.data_ptr())
            gradients = (torch.randn(3, 3, 4), torch.randn(3, 2, 3))
            native.reduce_banks.return_value = tuple(g[:2].bfloat16() for g in gradients)
            output_dtypes = (torch.bfloat16, torch.bfloat16)
            resource.return_generated_gradients(gradients, copies, output_dtypes=output_dtypes)
            call = native.reduce_banks.call_args
            self.assertEqual(call.args[-2:], (48, False))
            self.assertEqual(call.kwargs, {"output_dtypes": output_dtypes})
            for actual, original in zip(call.args[0], gradients):
                self.assertIs(actual, original)
                self.assertEqual(actual.dtype, torch.float32)
            launchers._env_use_udma = lambda _kind: True
            resource.prefetch(weights, copies)
            self.assertEqual(native.prefetch_banks.call_args.args[-2:], (24, True))

    def test_cost_charges_gradient_wire_bytes_and_peer_fanout(self) -> None:
        """Identical arithmetic must not ignore larger gradients or more peers."""
        geometry = (11000, 4000, 5120, 1792, 24, 3 * 3 * 5120 * 1792 * 2)
        common = dict(projection_flops=12, local_experts=3)
        bf16 = geometry[-1]
        self.assertTrue(placement.placement_is_profitable(*geometry, ep_size=2,
                                                        gradient_bytes=bf16, **common))
        self.assertFalse(placement.placement_is_profitable(*geometry, ep_size=8,
                                                         gradient_bytes=2 * bf16, **common))

    def test_cost_scales_with_expert_parameter_size_not_imbalance_ratio(self) -> None:
        """The same rank imbalance can justify small experts but not large experts."""
        base = dict(maximum_received=196000, target_load=131000, hidden_size=2560,
                    intermediate_size=768, cube_cores=24, projection_flops=12, ep_size=8)
        one = 3 * 2560 * 768 * 2
        self.assertTrue(placement.placement_is_profitable(**base, local_experts=64,
                        weight_bytes=64 * one, gradient_bytes=128 * one))
        self.assertFalse(placement.placement_is_profitable(**base, local_experts=64,
                        weight_bytes=256 * one, gradient_bytes=512 * one))

    def test_cost_rejects_invalid_transfer_extents(self) -> None:
        with self.assertRaisesRegex(ValueError, "gradient traffic"):
            placement.placement_is_profitable(100, 20, 64, 32, 24, 1024, gradient_bytes=-1)
        with self.assertRaisesRegex(ValueError, "cost geometry"):
            placement.placement_is_profitable(100, 20, 64, 32, 24, 1024, ep_size=0)

    def test_ungated_cost_accounts_for_two_projections(self) -> None:
        """A marginal gated saving must not enable borrowing for a cheaper ungated FFN."""
        geometry = (200000, 0, 512, 128, 20, 0)
        self.assertTrue(placement.placement_is_profitable(*geometry, projection_flops=6))
        self.assertFalse(placement.placement_is_profitable(*geometry, projection_flops=4))

    def test_native_and_legacy_counts_share_the_int32_planner(self) -> None:
        """Normalize equivalent wire types without changing the ownership decision."""
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                counts = torch.tensor([[512, 0], [0, 0]], dtype=dtype)
                copies = torch.full((2, 1), -1, dtype=torch.int32)
                with (patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}),
                      patch("hyper_multicore.moonep.allocator.allocate_experts", return_value=(torch.zeros(2, 2, dtype=torch.int32), None, None, copies)) as allocate):
                    result = placement.build_placement(
                        torch.empty(512, 4), torch.zeros(512, 1, dtype=torch.int32), counts, 256, 0,
                    )
                self.assertIsNone(result)
                actual_counts, actual_capacity = allocate.call_args.args
                self.assertEqual(actual_capacity, 256)
                self.assertEqual(actual_counts.dtype, torch.int32)
                torch.testing.assert_close(actual_counts, counts.to(torch.int32))

    def test_wide_counts_reject_overflow_before_allocator(self) -> None:
        """A large INT64 wire count must never wrap to a negative planner value."""
        counts = torch.tensor([[2**31, 0], [0, 0]], dtype=torch.int64)
        with (patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}),
              patch("hyper_multicore.moonep.allocator.allocate_experts") as allocate,
              self.assertRaisesRegex(ValueError, "INT32 planner limit")):
            placement.build_placement(torch.empty(0, 4), torch.empty(0, 1, dtype=torch.int32), counts, 1, 0)
        allocate.assert_not_called()

    def test_invalid_topologies_fail_before_allocator(self) -> None:
        """An invalid rank or expert extent cannot launch an asymmetric plan."""
        for counts, rank in ((torch.zeros(2, 3, dtype=torch.int32), 0),
                             (torch.zeros(2, 4, dtype=torch.int32), 2),
                             (torch.zeros(4, dtype=torch.int32), 0)):
            with self.subTest(shape=counts.shape, rank=rank):
                with (patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}),
                      patch("hyper_multicore.moonep.allocator.allocate_experts") as allocate,
                      self.assertRaises(ValueError)):
                    placement.build_placement(
                        torch.empty(0, 4), torch.empty(0, 1, dtype=torch.int32), counts, 1, rank,
                    )
                allocate.assert_not_called()

def test_public_moonep_statistics_are_independent_snapshots():
    from types import SimpleNamespace
    from hyper_multicore.moonep.mega_moe import MoonEPExecution
    from hyper_multicore.modules.mega_moe.module import MegaMoeExperts

    state = MoonEPExecution.__new__(MoonEPExecution)
    state.placement_statistics = {"moonep_calls": 3}
    state._transfer_pool = None
    expert = MegaMoeExperts.__new__(MegaMoeExperts)
    torch.nn.Module.__init__(expert)
    expert._moonep = state
    empty = expert.moonep_statistics()
    assert empty == {
        "enabled": True,
        "placement": {"moonep_calls": 3},
        "transfer": None,
        "bank_in_use": False,
    }
    state._transfer_pool = SimpleNamespace(
        statistics={"gradient_returns": 2}, in_use=True
    )
    snapshot = expert.moonep_statistics()
    assert snapshot["bank_in_use"] is True
    snapshot["placement"]["moonep_calls"] = 99
    snapshot["transfer"]["gradient_returns"] = 99
    assert state.placement_statistics["moonep_calls"] == 3
    assert state._transfer_pool.statistics["gradient_returns"] == 2
    expert._moonep = None
    assert expert.moonep_statistics() == {
        "enabled": False,
        "placement": {},
        "transfer": None,
        "bank_in_use": False,
    }
