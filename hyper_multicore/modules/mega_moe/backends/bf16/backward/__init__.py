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
"""BF16 backward scheduling contracts."""

from typing import NamedTuple


class BackwardOutputs(NamedTuple):
    """Expert kernel outputs required by an EP invocation."""

    combine: bool = True  # Returns dX and, for pre-FC2 weighting, dProbs.
    weight1: bool = True
    weight2: bool = True
