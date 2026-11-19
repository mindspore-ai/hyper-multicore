# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A3 projection readiness and full-K gradient return on the HotReplica bank lease."""

from __future__ import annotations

import struct

import torch

from hyper_multicore import shmem
from hyper_multicore.modules.mega_moe.route import _upload_route_tensor

from .bounded_placement import supports_bounded_placement

_LINE_WORDS = 16
_LAUNCH_MAGIC = 0x4D4F4F4E45505752
_BAND_BYTES = 4 * 1024**2
_MAX_BANDS = 64
_MAX_EPOCH = (1 << 31) - 1
_NO_OWNER = (1 << 64) - 1


def projection_shapes(hidden, intermediate, layout):
    """Return the physical matrix shapes of W13 and W2."""
    if layout == "EOI":
        return (2 * intermediate, hidden), (hidden, intermediate)
    return (hidden, 2 * intermediate), (intermediate, hidden)


def _band_layout(shape):
    rows, columns = shape
    rows_per_band = max(1, min(rows, _BAND_BYTES // (columns * 4)))
    return rows_per_band, (rows + rows_per_band - 1) // rows_per_band


def supports_pipeline(spec, borrowed):
    """Require disjoint BF16 weight and FP32 gradient addresses in the existing banks."""
    home = spec.local_experts
    h, i = spec.hidden_size, spec.intermediate_size
    return (
        supports_bounded_placement(spec)
        and 0 < 3 * borrowed <= 2 * home
        and h * i % 16 == 0
        and all(_band_layout(shape)[1] <= _MAX_BANDS
                for shape in projection_shapes(h, i, spec.expert_weight_layout))
    )


def _words_tensor(words, device, padding=0):
    buffer = bytearray(b"\0" * padding + struct.pack(f"<{len(words)}Q", *words))
    return _upload_route_tensor(torch.frombuffer(buffer, dtype=torch.uint8), device)


class ReplicaPipeline:
    """Own persistent peer epochs and stream-ordered projections for one pool."""

    def __init__(self, pool):
        self.pool = pool
        self.storage = None
        self.epoch = 0
        self.resident = None
        self.streams = {}

    @property
    def signal_words(self):
        """Return the same reserved extent on every member of a bank."""
        pool = self.pool
        band_count = sum(_band_layout(shape)[1]
                         for shape in projection_shapes(pool.hidden, pool.intermediate, pool.expert_weight_layout))
        return (3 * pool.ranks + 2 + band_count * pool.blocks) * pool.local_experts * _LINE_WORDS

    def allocate(self):
        """Clear peer words once after the symmetric allocation or a heap restart."""
        self.bind(shmem.empty((self.signal_words,), dtype=torch.int32, alignment=512))

    def bind(self, storage):
        """Start a fresh epoch sequence on the current symmetric allocation."""
        self.storage = storage
        self.epoch = 0
        self.resident = None
        self.storage.zero_()
        shmem.barrier(blocking=False)

    def free(self):
        """Retire epochs before another heap can reuse their addresses."""
        if self.storage is not None:
            shmem.free(self.storage)
            self.storage = None
        self.epoch = 0
        self.resident = None
        self.streams.clear()

    def plan(self, placement):
        """Use the allocator's agreed copy rows without a backward device readback."""
        pool = self.pool
        slots = placement.borrowed_capacity
        table = placement.copies_by_rank
        home = pool.local_experts
        received = tuple(_NO_OWNER if source < 0 else source // home for source in table[pool.rank])
        owned = tuple((peer, slot, source % home) for peer, row in enumerate(table)
                      for slot, source in enumerate(row) if source >= 0 and source // home == pool.rank)
        return slots, received, owned

    def begin(self, weights, binding, backward):
        """Publish projection epochs directly to their current guest consumers."""
        pool = self.pool
        self.epoch += 1
        if self.epoch > _MAX_EPOCH:
            raise RuntimeError("HotReplica replica epochs exhausted")
        slots, received, owned = binding.edges
        rank_lines = pool.ranks * pool.local_experts * _LINE_WORDS
        credit = self.storage[:rank_lines].view(pool.ranks, pool.local_experts, _LINE_WORDS)
        ready = self.storage[3 * rank_lines:3 * rank_lines + 2 * pool.local_experts * _LINE_WORDS]
        ready = ready.view(2, pool.local_experts, _LINE_WORDS)
        bank = pool.leased_banks
        if not backward or self.resident is not binding.token:
            self.resident = None
            for slot, owner in enumerate(received):
                if owner != _NO_OWNER:
                    shmem.signal(credit[pool.rank, slot, :1], self.epoch, owner)
            if owned:
                for peer, slot, _ in owned:
                    shmem.wait_signal(credit[peer, slot, :1], self.epoch, comparison="ge")
                stream = torch.npu.current_stream(pool.device)
                value = _words_tensor((self.epoch,), pool.device)[:4].view(torch.int32)
                credit_ready = torch.npu.Event()
                credit_ready.record(stream)
                copies = {}
                order = (1, 0) if backward else (0, 1)
                try:
                    for peer, slot, owner in owned:
                        if peer not in copies:
                            if peer not in self.streams:
                                self.streams[peer] = torch.npu.Stream(device=pool.device)
                            copies[peer] = self.streams[peer]
                            copies[peer].wait_event(credit_ready)
                        with torch.npu.stream(copies[peer]):
                            value.record_stream(copies[peer])
                            for matrix in order:
                                source = weights[matrix][owner]
                                source.record_stream(copies[peer])
                                shmem._api._put_sdma(bank[matrix][slot], source, peer)
                                shmem._api._put_sdma(ready[matrix, slot, :1], value, peer)
                finally:
                    binding.transfers = tuple(copies.values())
            binding.weight_epoch = self.epoch
            self.resident = binding.token
        binding.launch = (_LAUNCH_MAGIC, pool.local_experts, binding.weight_epoch,
                          *(entry.data_ptr() for entry in ready), *(w.data_ptr() for w in weights),
                          *(w.data_ptr() for w in bank))
        binding.gradient_epoch = self.epoch
        return tuple(w[:pool.local_experts + slots] for w in bank)

    def complete(self, binding):
        """Keep local parameter lifetimes and bank reuse ordered with the peer copies."""
        stream = torch.npu.current_stream(self.pool.device)
        for peer_stream in binding.transfers:
            done = torch.npu.Event()
            done.record(peer_stream)
            stream.wait_event(done)
        binding.transfers = ()
        binding.gradient_storage = ()

    def runtime(self, base, binding, gradients=None):
        """Encode the current projection addresses after the ordinary queue table."""
        gradients_address = 0
        if gradients is not None:
            gradients_address = self._gradient_table(binding, gradients).data_ptr()
        extension = _words_tensor((*binding.launch, gradients_address), self.pool.device, -base.numel() % 8)
        return torch.cat((base.view(torch.uint8), extension))

    def _gradient_table(self, binding, gradients):
        pool = self.pool
        home, cubes = pool.local_experts, pool.blocks
        slots, received, owned = binding.edges
        touched = {expert: slot + 1 for slot, expert in enumerate(sorted({edge[2] for edge in owned}))}
        last = {edge[2]: index for index, edge in enumerate(owned)}
        edges = [value for i, edge in enumerate(owned) for value in (*edge, int(i == last[edge[2]]))]
        mapping = tuple(touched.get(expert, 0) for expert in range(home))
        bank = pool.leased_banks
        rank_lines = pool.ranks * home * _LINE_WORDS
        bands_start = 3 * rank_lines + 2 * home * _LINE_WORDS
        band_layouts = tuple(_band_layout(grad.shape[1:]) for grad in gradients)
        flags = tuple(home * bands * cubes * _LINE_WORDS for _, bands in band_layouts)
        offsets = (bands_start, bands_start + flags[0])
        done_words = cubes * _LINE_WORDS
        records, retained = [], []
        local_flags = None
        if owned:
            local_flags = torch.zeros(sum(flags) + 2 * done_words, dtype=torch.int32, device=pool.device)
            retained.append(local_flags)
        owner_rows = binding.placement.owner_rows
        for matrix in (1, 0):
            grad = gradients[matrix]
            rows, columns = grad.shape[1:]
            per_band, bands = band_layouts[matrix]
            guests = bank[matrix].view(-1).view(torch.float32).view(home, rows, columns)[-slots:]
            local_address = done_address = scratch_address = 0
            if local_flags is not None:
                local_address = local_flags[matrix * flags[0]:].data_ptr()
                done_address = local_flags[sum(flags) + matrix * done_words:].data_ptr()
                scratch = grad.new_empty((len(touched), rows, columns), dtype=torch.float32)
                if owner_rows is None:
                    scratch.zero_()
                else:
                    for expert, slot in touched.items():
                        if not owner_rows[expert]:
                            scratch[slot - 1].zero_()
                scratch_address = scratch.data_ptr()
                retained.append(scratch)
            ack = self.storage[(1 + matrix) * rank_lines:]
            records.append((home, slots, binding.gradient_epoch, rows, columns, per_band, bands,
                            grad.data_ptr(), guests.data_ptr(), local_address,
                            self.storage[offsets[matrix]:].data_ptr(), len(owned), scratch_address,
                            pool.rank, ack.data_ptr(), done_address, *edges, *mapping, *received))
        metadata = _words_tensor((2, 3, 3 + len(records[0]), *records[0], *records[1]), pool.device)
        stream = torch.npu.current_stream(pool.device)
        for tensor in (*retained, metadata):
            tensor.record_stream(stream)
        binding.gradient_storage = (*retained, metadata)
        return metadata
