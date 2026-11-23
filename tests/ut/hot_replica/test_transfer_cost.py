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
"""Validate placement transfer cost accounting."""

from hyper_multicore.modules.hot_replica.placement import placement_is_profitable


def test_charge_replay_weight_restore():
    """Include backward weight restoration in the placement cost estimate."""
    common = {
        "maximum_received": 300000,
        "target_load": 131072,
        "hidden_size": 2560,
        "intermediate_size": 768,
        "cube_cores": 24,
        "ep_size": 8,
        "local_experts": 64,
        "weight_bytes": 64 * 3 * 2560 * 768 * 2,
        "gradient_bytes": 64 * 3 * 2560 * 768 * 4,
    }
    assert placement_is_profitable(**common, projection_flops=18, backward_prefetch_weight_bytes=common["weight_bytes"])
    assert not placement_is_profitable(
        **common, projection_flops=18, backward_prefetch_weight_bytes=8 * common["weight_bytes"]
    )


def test_actual_copy_count_and_backward_fetch_change_profitability(monkeypatch):
    """Charge every borrowed expert and both forward/backward peer fetches."""
    from hyper_multicore.modules.hot_replica import placement  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(placement, "_A3_FLOPS_PER_CUBE", 1.0)
    monkeypatch.setattr(placement, "_A3_STAGING_BYTES_PER_SECOND", 1e12)
    monkeypatch.setattr(placement, "_A3_PEER_BYTES_PER_SECOND", 250.0)
    monkeypatch.setattr(placement, "_A3_PLACEMENT_SECONDS", 0.0)
    geometry = {
        "maximum_received": 1, "target_load": 0, "hidden_size": 1, "intermediate_size": 1,
        "cube_cores": 1, "weight_bytes": 400, "gradient_bytes": 800, "local_experts": 4,
        "ep_size": 2, "projection_flops": 3,
    }
    assert placement_is_profitable(**geometry, transfer_experts=1, backward_prefetch_weight_bytes=400)
    assert not placement_is_profitable(**geometry, transfer_experts=2, backward_prefetch_weight_bytes=400)
    assert placement_is_profitable(**geometry, transfer_experts=2, backward_prefetch_weight_bytes=0)
