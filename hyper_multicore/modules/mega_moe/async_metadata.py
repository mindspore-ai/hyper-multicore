# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Bounded per-launch metadata ownership; no reusable CPU source or weight cache."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


class _NpuMetadataBackend:
    def __init__(self) -> None:
        import torch
        self.torch = torch

    def current_stream(self, device: Any) -> Any:
        return self.torch.npu.current_stream(device=device)

    def stream_scope(self, stream: Any) -> Any:
        return self.torch.npu.stream(stream)

    def pin(self, content: bytes, device: Any) -> Any:
        host = self.torch.frombuffer(bytearray(content), dtype=self.torch.uint8).pin_memory(device=device.type)
        if not host.is_pinned(device=device.type):
            raise RuntimeError('Async metadata requires an NPU-compatible pinned CPU source.')
        return host

    def event(self) -> Any:
        return self.torch.npu.Event()

    def copy(self, host: Any, device: Any) -> Any:
        return host.to(device=device, non_blocking=True)

    def blocking_copy(self, content: bytes, device: Any) -> Any:
        return self.torch.frombuffer(bytearray(content), dtype=self.torch.uint8).to(device=device)

    def record_stream(self, tensor: Any, stream: Any) -> None:
        tensor.record_stream(stream)

    def record(self, event: Any, stream: Any) -> None:
        event.record(stream)

    def query(self, event: Any) -> bool:
        return event.query()

    def synchronize(self, stream: Any) -> None:
        stream.synchronize()


@dataclass
class _Packet:
    source: Any
    event: Any
    destination: Any = None


class AsyncMetadataTransfer:
    """Retain immutable packets until a same-stream post-consumer event completes.

    The workspace lease serializes submissions. The owner must synchronize its
    workspace successfully before calling close_after_synchronize; failed close
    must retain this object and all packets.
    """

    def __init__(self, limit: int = 16, backend: Any = None) -> None:
        if limit <= 0:
            raise ValueError('Async metadata retention limit must be positive.')
        self.limit = limit
        self.backend = _NpuMetadataBackend() if backend is None else backend
        self._packets = deque()
        self._closed = self._poisoned = False
        self.stats = {'async_launches': 0, 'fallbacks': 0, 'retired': 0, 'high_water': 0}

    @property
    def pending(self) -> int:
        return len(self._packets)

    def retire(self) -> None:
        # Query a bounded prefix; out-of-order stream completion never permits
        # overwriting another packet because every source is a unique allocation.
        while self._packets and self.backend.query(self._packets[0].event):
            self._packets.popleft()
            self.stats['retired'] += 1

    def launch(self, content: bytes, device: Any, consumer: Callable[[Any], None]) -> None:
        if self._closed:
            raise RuntimeError('Async metadata owner is closed.')
        if self._poisoned:
            raise RuntimeError('Async metadata owner is poisoned after failed stream cleanup.')
        self.retire()
        if self.pending == self.limit:
            self.stats['fallbacks'] += 1
            consumer(self.backend.blocking_copy(content, device))
            return
        stream = self.backend.current_stream(device)
        packet = _Packet(self.backend.pin(content, device), self.backend.event())
        self._packets.append(packet)
        self.stats['high_water'] = max(self.stats['high_water'], self.pending)
        with self.backend.stream_scope(stream):
            try:
                packet.destination = self.backend.copy(packet.source, device)
                self.backend.record_stream(packet.destination, stream)
                consumer(packet.destination)
            except BaseException:
                self._finish(packet, stream)
                raise
            self._finish(packet, stream)
        self.stats['async_launches'] += 1

    def _finish(self, packet: _Packet, stream: Any) -> None:
        try:
            self.backend.record(packet.event, stream)
        except BaseException:
            # Only an exceptional submission/record path may wait on the CPU.
            try:
                self.backend.synchronize(stream)
            except BaseException as error:
                self._poisoned = True
                raise RuntimeError('Async metadata cleanup failed; owner poisoned and packet retained.') from error
            self._packets.pop()
            raise

    def close_after_synchronize(self) -> None:
        self._packets.clear()
        self._closed = True
