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
"""Immutable EP-rank node identities independent of rank numbering and device count."""

from collections.abc import Sequence






def _validate_rank_nodes(nodes: Sequence[int], size: int) -> tuple[int, ...]:
    """Freeze optional node labels and reject malformed process-group topology."""
    frozen = tuple(nodes)
    invalid = any(not isinstance(node, int) or isinstance(node, bool) or node < 0 for node in frozen)
    if frozen and (len(frozen) != size or invalid):
        raise ValueError('Replica rank_nodes must contain one nonnegative integer node label per EP rank')
    return frozen


def charge_replica_endpoints(ranks: int, edges, nodes: tuple[int, ...], cross_node_cost: float) -> list[float]:
    """Charge each whole-matrix edge to both the owner and guest rank."""
    charges = [0.0] * ranks
    for owner, guest in edges:
        charge = cross_node_cost if nodes and nodes[owner] != nodes[guest] else 1.0
        charges[owner] += charge
        charges[guest] += charge
    return charges
