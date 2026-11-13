"""Order consumer submissions sharing one process-wide set of UDMA QP0 queues."""
from threading import Lock


class SubmissionOrder:
    """Reuse consumer completion events; do not add synchronizations or device records."""
    def __init__(self):
        self._lock = Lock()
        self._active = False
        self._stream = None
        self._completion = None

    def claim(self, stream):
        with self._lock:
            if self._active:
                raise RuntimeError("UDMA QP0 queues already have an active submission")
            if self._completion is not None and self._stream != stream.npu_stream:
                stream.wait_event(self._completion)
            self._active = True

    def release(self, stream, completion):
        with self._lock:
            if not self._active:
                raise RuntimeError("UDMA QP0 submission released without a lease")
            self._stream = stream.npu_stream
            self._completion = completion
            self._active = False


udma_submissions = SubmissionOrder()
