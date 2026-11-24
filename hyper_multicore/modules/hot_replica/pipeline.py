# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Projection readiness and full-K gradient return on the HotReplica bank lease."""

from __future__ import annotations

from math import gcd
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


def _target_band_rows(columns):
    return max(1, _BAND_BYTES // (columns * 4))


def _band_layout(shape):
    rows, columns = shape
    rows_per_band = max((rows + _MAX_BANDS - 1) // _MAX_BANDS, _target_band_rows(columns))
    if columns % 16:
        # The BF16 producer and peers share packed matrices. Keep band
        # boundaries on 32-byte blocks even with a non-aligned row stride.
        rows_per_block = 16 // gcd(columns, 16)
        rows_per_band = (rows_per_band + rows_per_block - 1) // rows_per_block * rows_per_block
    rows_per_band = min(rows, rows_per_band)
    return rows_per_band, (rows + rows_per_band - 1) // rows_per_band


def _pipeline_capacity(spec, bank_experts=None, *, bf16_partials=False):
    """Limit guest weights and dW to disjoint bank spans."""
    h, i = spec.hidden_size, spec.intermediate_size
    if supports_bounded_placement(spec):
        # A3 EIO copies exact GM byte spans, including matrix tails.
        needs_tail = h * i % 16 or bf16_partials and (h % 16 or i % 16)
        if needs_tail and spec.expert_weight_layout != "EIO":
            return 0
        extent = 2 * spec.local_experts if bank_experts is None else bank_experts
        return min(spec.local_experts, extent // (2 if bf16_partials else 3))
    return 0


def supports_pipeline(spec, borrowed, *, bank_experts=None, bf16_partials=False):
    """Check whether the selected gradient protocol fits the requested width."""
    return 0 < borrowed <= _pipeline_capacity(spec, bank_experts, bf16_partials=bf16_partials)


def fused_replica_slots(spec, placement, *, bank_experts=None, bf16_partials=False):
    """Keep the physical stride and exclude padding from the fused bank extent."""
    width = placement.borrowed_capacity
    slots = min(width, _pipeline_capacity(spec, bank_experts, bf16_partials=bf16_partials))
    if slots and slots < width and any(owner >= 0 for row in placement.copies_by_rank for owner in row[slots:]):
        return 0
    return slots


def _words_tensor(words, device, padding=0):
    buffer = bytearray(b"\0" * padding + struct.pack(f"<{len(words)}Q", *words))
    return _upload_route_tensor(torch.frombuffer(buffer, dtype=torch.uint8), device)


class ReplicaPipeline:
    """Own persistent peer epochs and stream-ordered projections for one pool."""

    def __init__(self, pool, *, shared_layouts=False):
        self.pool = pool
        self.shared_layouts = shared_layouts
        self.storage = None
        self.epoch = 0
        self.resident = None
        self.streams = {}

    @property
    def signal_words(self):
        """Return the same reserved extent on every member of a bank."""
        pool = self.pool
        layouts = ("EIO", "EOI") if self.shared_layouts else (pool.expert_weight_layout,)
        band_count = max(
            sum(_band_layout(shape)[1] for shape in projection_shapes(pool.hidden, pool.intermediate, layout))
            for layout in layouts)
        words = 3 * pool.ranks + 2 + band_count * pool.blocks
        return words * pool.local_experts * _LINE_WORDS

    def invalidate(self):
        """Forget prepared weights after a generic bank write."""
        self.resident = None

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

    def plan(self, placement, slots=None):
        """Use the allocator's agreed copy rows without a backward device readback."""
        pool = self.pool
        if slots is None:
            slots = placement.borrowed_capacity
        table = placement.copies_by_rank
        home = pool.local_experts
        received = tuple(_NO_OWNER if source < 0 else source // home for source in table[pool.rank][:slots])
        owned = tuple((peer, slot, source % home) for peer, row in enumerate(table)
                      for slot, source in enumerate(row[:slots]) if source >= 0 and source // home == pool.rank)
        return slots, received, owned

    def begin(self, weights, binding, backward):
        """Publish projection epochs directly to their current guest consumers."""
        pool = self.pool
        self.epoch += 1
        if self.epoch > _MAX_EPOCH:
            raise RuntimeError("HotReplica replica epochs exhausted")
        # Projection weights and dW alias the generic bank, including a
        # checkpoint suffix staged between this binding's forward and backward.
        pool.generation += 1
        rank_lines = pool.ranks * pool.local_experts * _LINE_WORDS
        ready = self.storage[3 * rank_lines:3 * rank_lines + 2 * pool.local_experts * _LINE_WORDS]
        ready = ready.view(2, pool.local_experts, _LINE_WORDS)
        bank = pool.leased_banks
        if not backward or self.resident is not binding.token:
            self.resident = None
            snapshot = binding.snapshot
            if snapshot is None:
                self._publish_weights(weights, binding, ready, rank_lines, backward)
            else:
                pool.restore_snapshot(snapshot, 0)
                ready.fill_(self.epoch)
                stats = pool.statistics
                stats["fused_snapshot_restores"] = stats.get("fused_snapshot_restores", 0) + 1
            binding.weight_epoch = self.epoch
            self.resident = binding.token
        else:
            stats = pool.statistics
            stats["fused_backward_resident_hits"] = stats.get("fused_backward_resident_hits", 0) + 1
        binding.launch = (_LAUNCH_MAGIC, pool.local_experts, binding.weight_epoch,
                          *(entry.data_ptr() for entry in ready), *(w.data_ptr() for w in weights),
                          *(w.data_ptr() for w in bank))
        binding.gradient_epoch = self.epoch
        # GMM descriptors still use the padded expert count. Zero-row guests
        # have no weight reads or gradient producers.
        return tuple(w[:pool.local_experts + binding.placement.borrowed_capacity] for w in bank)

    def _publish_weights(self, weights, binding, ready, rank_lines, backward):
        """Transfer live owner projections after the guest credits arrive."""
        pool = self.pool
        _, received, owned = binding.edges
        credit = self.storage[:rank_lines].view(pool.ranks, pool.local_experts, _LINE_WORDS)
        for slot, owner in enumerate(received):
            if owner != _NO_OWNER and owner != pool.rank:
                shmem.signal(credit[pool.rank, slot, :1], self.epoch, owner)
        if not owned:
            return
        for peer, slot, _ in owned:
            if peer != pool.rank:
                shmem.wait_signal(credit[peer, slot, :1], self.epoch, comparison="ge")
        stream = torch.npu.current_stream(pool.device)
        value = _words_tensor((self.epoch,), pool.device)[:4].view(torch.int32)
        credit_ready = torch.npu.Event()
        credit_ready.record(stream)
        copies = {}
        order = (1, 0) if backward else (0, 1)
        bank = pool.leased_banks
        try:
            for peer, slot, owner in owned:
                if peer == pool.rank:
                    for matrix in order:
                        bank[matrix][slot].copy_(weights[matrix][owner])
                        ready[matrix, slot, :1].fill_(self.epoch)
                    continue
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
        if backward:
            remote = sum(peer != pool.rank for peer, _, _ in owned)
            weight_bytes = sum(weight.numel() // weight.shape[0] * weight.element_size() for weight in weights)
            stats = pool.statistics
            stats["fused_backward_weight_bytes"] = stats.get("fused_backward_weight_bytes", 0) + remote * weight_bytes

    def complete(self, binding):
        """Keep local parameter lifetimes and bank reuse ordered with the peer copies."""
        stream = torch.npu.current_stream(self.pool.device)
        for peer_stream in binding.transfers:
            done = torch.npu.Event()
            done.record(peer_stream)
            stream.wait_event(done)
        binding.transfers = ()
        binding.gradient_storage = ()

    def runtime(self, base, binding, gradients=None, *, bf16_partials=False):
        """Encode the current projection addresses after the ordinary queue table."""
        gradients_address = 0
        if gradients is not None:
            gradients_address = self._gradient_table(binding, gradients, bf16_partials=bf16_partials).data_ptr()
        words = (*binding.launch, gradients_address)
        extension = _words_tensor(words, self.pool.device, -base.numel() % 8)
        return torch.cat((base.view(torch.uint8), extension))

    def _gradient_table(self, binding, gradients, *, bf16_partials=False):
        pool = self.pool
        home, cubes = pool.local_experts, pool.blocks
        slots, received, owned = binding.edges
        if bf16_partials:
            # Group owners while keeping their contributions in peer/slot order.
            owned = sorted(owned, key=lambda edge: edge[2])
        owners = sorted({edge[2] for edge in owned})
        touched = {expert: slot + 1 for slot, expert in enumerate(owners)}
        last = {edge[2]: index for index, edge in enumerate(owned)}
        edges = [value for i, edge in enumerate(owned) for value in (*edge, int(i == last[edge[2]]))]
        mapping = tuple(touched.get(expert, 0) for expert in range(home))
        bank = pool.leased_banks
        rank_lines = pool.ranks * home * _LINE_WORDS
        bands_start = 3 * rank_lines + 2 * home * _LINE_WORDS
        band_layouts = tuple(_band_layout(grad.shape[1:])
                             for grad in gradients)
        flags = tuple(home * bands * cubes * _LINE_WORDS for _, bands in band_layouts)
        offsets = (bands_start, bands_start + flags[0])
        done_words = cubes * _LINE_WORDS
        records, retained = [], []
        local_flags = None
        if owned:
            local_flags = torch.zeros(sum(flags) + 2 * done_words, dtype=torch.int32, device=pool.device)
            retained.append(local_flags)
        owner_rows = binding.placement.owner_rows
        element_size = torch.bfloat16.itemsize if bf16_partials else torch.float32.itemsize
        for matrix in (1, 0):
            grad = gradients[matrix]
            rows, columns = grad.shape[1:]
            per_band, bands = band_layouts[matrix]
            # The bank lease keeps the contiguous dW suffix alive for remote readers.
            guest_bytes = slots * rows * columns * element_size
            guest_address = bank[matrix].data_ptr() + bank[matrix].nbytes - guest_bytes
            local_address = done_address = scratch_address = 0
            if local_flags is not None:
                local_address = local_flags[matrix * flags[0]:].data_ptr()
                done_address = local_flags[sum(flags) + matrix * done_words:].data_ptr()
                if not bf16_partials:
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
            header = (home, slots, binding.gradient_epoch, rows, columns, per_band, bands,
                      grad.data_ptr(), guest_address, local_address,
                      self.storage[offsets[matrix]:].data_ptr(), len(owned), scratch_address,
                      pool.rank, ack.data_ptr(), done_address)
            records.append((*header, *edges, *mapping, *received))
        protocol = 3 if bf16_partials else 2
        metadata = _words_tensor((protocol, 3, 3 + len(records[0]), *records[0], *records[1]), pool.device)
        stream = torch.npu.current_stream(pool.device)
        for tensor in (*retained, metadata):
            tensor.record_stream(stream)
        binding.gradient_storage = (*retained, metadata)
        return metadata
