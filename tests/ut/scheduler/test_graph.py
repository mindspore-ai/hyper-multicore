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
"""Graph node registration by operator name."""

import unittest

from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec


class TestComputeGraph(unittest.TestCase):
    """Check lookup names without duplicating nodes in the DAG."""

    @staticmethod
    def _node(name: str) -> OperatorNode:
        return OperatorNode(
            name=name, op_type=OpType.GMM, inputs=[], outputs=[], param_positions=[],
            split_value=1, split_spec=SplitSpec(None, lambda _: 1, []), tiling_position=-1, fill_config=None,
        )

    def test_additional_name_preserves_lookup_and_topology(self):
        """Resolve both names to one node and allow edges through its alias."""
        graph = ComputeGraph()
        first, second = self._node("up_proj"), self._node("swiglu")
        graph.add_op(first).add_op(second, alias="activation").add_edge("up_proj", "activation")
        self.assertIs(graph.get_op(name="swiglu"), second)
        self.assertIs(graph.get_op("activation"), second)
        self.assertEqual([node.name for node in graph.topological_sort()], ["up_proj", "swiglu"])
        self.assertIs(second.predecessors[0], first)

    def test_conflicting_names_leave_graph_unchanged(self):
        """Reject collisions before registering any part of the new node."""
        graph = ComputeGraph()
        activation = self._node("swiglu")
        graph.add_op(activation, alias="activation")
        for name, alias in (("other", "activation"), ("other", "swiglu"), ("activation", None)):
            with self.subTest(name=name, alias=alias), self.assertRaises(ValueError):
                graph.add_op(self._node(name), alias=alias)
        with self.assertRaises(KeyError):
            graph.get_op("other")
        self.assertIs(graph.topological_sort()[0], activation)
        self.assertEqual(len(graph.topological_sort()), 1)

    def test_same_additional_name_registers_one_node(self):
        """Register one node when its alias equals its concrete name."""
        graph = ComputeGraph()
        node = self._node("activation")
        graph.add_op(node, alias=node.name)
        self.assertIs(graph.get_op(node.name), node)
        self.assertEqual(len(graph.topological_sort()), 1)
