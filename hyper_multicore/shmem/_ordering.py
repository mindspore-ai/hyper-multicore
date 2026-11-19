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
"""Order consumer submissions sharing one process-wide set of UDMA QP0 queues."""
from threading import Lock
from typing import Any


class SubmissionOrder:
    """Reuse consumer completion events; do not add synchronizations or device records."""
    def __init__(self) -> None:
        """Initialize the submission lease and reusable completion event."""
        self._lock = Lock()
        self._active = False
        self._stream_key = None
        self._completion = None

    def claim(self, stream: Any) -> None:
        """Acquire QP0 ownership and wait for a different logical stream."""
        with self._lock:
            if self._active:
                raise RuntimeError("UDMA QP0 queues already have an active submission")
            if self._completion is not None and self._stream_key != (stream.device, stream.stream_id):
                stream.wait_event(self._completion)
            self._active = True

    def release(self, stream: Any, completion: Any) -> None:
        """Release QP0 ownership with the submitted completion event."""
        with self._lock:
            if not self._active:
                raise RuntimeError("UDMA QP0 submission released without a lease")
            self._stream_key = (stream.device, stream.stream_id)
            self._completion = completion
            self._active = False


udma_submissions = SubmissionOrder()
