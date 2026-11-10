"""Order independent workspaces sharing the process-wide UDMA QP bank."""
import os
import threading
from typing import Any

from hyper_multicore.shmem._runtime import _udma_qps_per_peer


class SubmissionOrder:
    """Keep one host submitter and order device submissions across streams."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = False
        self.stream = None
        self.completion = None

    def claim(self, stream: Any) -> None:
        with self.lock:
            if self.active:
                raise RuntimeError('UDMA QPs do not support concurrent workspace submissions')
            if self.completion is not None and self.stream != stream.npu_stream:
                stream.wait_event(self.completion)
            self.active = True

    def release(self, stream: Any, completion: Any) -> None:
        with self.lock:
            if not self.active:
                raise RuntimeError('UDMA QPs released without an active submission')
            self.stream = stream.npu_stream
            self.completion = completion
            self.active = False


def validate_transport() -> None:
    """Only qualified policies keep each communication QP on its original odd AIV."""
    if os.environ.get('HYPER_MULTICORE_SHMEM_DATA_ENGINE', 'mte') != 'mte_udma':
        return
    if _udma_qps_per_peer() < 32 and os.environ.get('HYPER_MULTICORE_MX_AIV_POLICY') != 'window2':
        raise ValueError('A reduced QP bank requires the qualified MXFP8 window2 queues')
    if (os.environ.get('HYPER_MULTICORE_MX_AIV_POLICY', 'legacy_odd') not in (
            'legacy_odd', 'roles', 'roles_ready', 'roles_rule', 'planned', 'window2')
            or os.environ.get('HYPER_MULTICORE_MX_VECTOR_ROLES', '0') != '0'):
        raise ValueError('UDMA QP ownership requires qualified odd communication owners; ALL remapping is forbidden')


SUBMISSION_ORDER = SubmissionOrder()


def validate_device_transport(tensor: Any) -> None:
    """A5 production tasks use UDMA; reject an uninitialized QP bank before launch."""
    if tensor.device.type != 'npu':
        return
    import torch
    model = str(torch.npu.get_device_name(tensor.device))
    if model.startswith(('Ascend950', 'Ascend910_95')):
        if os.environ.get('HYPER_MULTICORE_SHMEM_DATA_ENGINE') != 'mte_udma':
            raise ValueError('A5 MegaMoE requires HYPER_MULTICORE_SHMEM_DATA_ENGINE=mte_udma')
        validate_transport()
