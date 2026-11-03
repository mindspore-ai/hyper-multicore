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
"""Process-wide one-sided SHMEM communication for the MultiCore component."""

from ._api import (
    all_gather,
    barrier,
    empty,
    free,
    get,
    host_barrier,
    put,
    signal,
    wait_signal,
)
from ._debug import debug_state
from ._lifecycle import _restart
from ._lifecycle import acquire, release, root_members

__all__ = [
    "acquire",
    "all_gather",
    "barrier",
    "debug_state",
    "empty",
    "free",
    "get",
    "host_barrier",
    "put",
    "release",
    "root_members",
    "signal",
    "wait_signal",
]
