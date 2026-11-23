# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe ordered FP32 replica return from complete MX TN producers."""

from dataclasses import dataclass
import math
from contextlib import contextmanager
import struct

import torch

from hyper_multicore import shmem
from hyper_multicore.modules.mega_moe.route import _upload_route_tensor

from ._planner.plan import _replica_transfers
from .shmem_transfer import SharedExpertPool


MAX_MX_GUEST_SLOTS = 2


GRADIENT_HELPERS = 4


@dataclass(frozen=True)
class MXGradientStorage:
    """Addresses for one projection, including symmetric inbox and signal regions."""

    owner: int
    guest: int
    ready: int
    ack: int
    done: int
    staging: int
    chunk_elements: int


def pack_mx_gradient_return(placement, rank, home, events, elements, storage, epoch, qp_count):
    """Build the v9 return record using the actual runtime's complete TN fences."""
    slots = placement.borrowed_capacity
    ranks = len(placement.copies_by_rank)
    if not 0 <= rank < ranks or home <= 0 or slots <= 0 or len(events) != home + slots:
        raise ValueError("MX gradient return geometry does not match its TN producers")
    if any(len(row) != slots for row in placement.copies_by_rank):
        raise ValueError("MX gradient return requires a converged copy-table width")
    if not 0 < epoch < (1 << 31) or any(event <= 0 for event in events):
        raise ValueError("MX gradient return requires a fresh epoch and positive producer events")
    if qp_count not in (8, 16, 32):
        raise ValueError("MX gradient return requires four unowned QPs per peer")
    if elements <= 0 or elements % 8 or not 0 < storage.chunk_elements <= (1 << 19) or storage.chunk_elements % 8:
        raise ValueError("MX gradient return requires aligned FP32 chunks")
    addresses = (storage.owner, storage.guest, storage.ready, storage.ack, storage.done, storage.staging)
    if any(address <= 0 or address % 32 for address in addresses):
        raise ValueError("MX gradient return requires aligned, non-null buffer addresses")
    if any(address % 64 for address in addresses[2:]):
        raise ValueError("MX gradient flags and staging require cache-line alignment")
    edges = _replica_transfers(placement.copies_by_rank, home)
    if any(edge.owner_rank >= ranks or edge.owner_rank == edge.target_rank for edge in edges):
        raise ValueError("MX gradient return requires a remote home expert for each active guest")
    incoming = tuple(edge for edge in edges if edge.target_rank == rank)
    owned = tuple(edge for edge in edges if edge.owner_rank == rank)
    qp_offset = 16 + 3 * len(incoming) + 4 * len(owned)
    values = [9, rank, slots, epoch, elements, *addresses[:5], len(incoming), len(owned),
              storage.staging, storage.chunk_elements, GRADIENT_HELPERS, qp_offset]
    for edge in incoming:
        values.extend((edge.owner_rank, edge.guest_slot(home), events[edge.target_slot]))
    for edge in owned:
        values.extend((edge.target_rank, edge.guest_slot(home), edge.owner_slot, events[edge.owner_slot]))
    # The shared MX transport has one fixed token queue per peer, on QP0.
    # Never move its producers or share the gradient helpers' QP state.
    values.extend(tuple(range(1, GRADIENT_HELPERS + 1)) * ranks)
    return struct.pack(f"<{len(values)}Q", *values)


def pack_mx_backward_returns(down: bytes, up: bytes) -> bytes:
    """Describe independent W2/W13 ready banks with one ACK for both projections."""
    return struct.pack("<3Q", 3, 3, 3 + len(down) // 8) + down + up


class MXGradientPool(SharedExpertPool):
    """Reuse the registered HotReplica heap owner for FP32 inboxes and helper staging."""

    pool_kind = "hot_replica_mx_gradients"
    chunk_elements = 1 << 19
    cache_generation = None

    @property
    def allocation_bytes(self) -> int:
        """Reserve the guest FP32 inbox, flags and UDMA scratch together."""
        elements = sum(math.prod(shape) for shape in self._bank_shapes())
        return elements * self.dtype.itemsize + 1024

    def bind_restarted_allocations(self, spec, dtype, device, receive_capacity, tensors):
        """A restored heap must initialize its completion flags before use."""
        super().bind_restarted_allocations(spec, dtype, device, receive_capacity, tensors)
        self.cache_generation = None

    @property
    def guest_slots(self):
        return min(MAX_MX_GUEST_SLOTS, self.local_experts)

    @property
    def flag_words(self):
        return self.ranks * self.guest_slots * 16

    def _bank_shapes(self, layout=None):
        # BF16 allocations are viewed as FP32 only after the symmetric commit.
        flags = (3 * self.flag_words + GRADIENT_HELPERS * 16) * 2
        scratch = GRADIENT_HELPERS * 2 * self.chunk_elements * 2
        guests = tuple((2 * self.guest_slots, *shape[1:]) for shape in super()._bank_shapes(layout))
        return (*guests, (flags + scratch,))

    @contextmanager
    def native_return(self, placement, runtime, owners):
        """Lease FP32 destinations through the final per-projection peer ACKs."""
        host = runtime.host
        slots = placement.borrowed_capacity
        home = self.local_experts
        if slots > self.guest_slots or len(host.gradient_ready_events) != 2:
            raise ValueError("MX gradient graph does not fit the reserved symmetric inbox")
        self.claim(self.expert_weight_layout)
        try:
            first, second, auxiliary = self.leased_banks
            if self.cache_generation is None:
                state = shmem.debug_state()['config']
                if state['data_engine'] != 'mte_udma' or state['udma_qp_num'] not in (8, 16, 32):
                    raise ValueError("Active MX gradients require at least eight UDMA QPs")
                self._qp_count = state['udma_qp_num']
                flags_words = 3 * self.flag_words + GRADIENT_HELPERS * 16
                auxiliary.view(torch.int32)[:flags_words].zero_()
                shmem.host_barrier()
                self.cache_generation = 0
            epoch = self.cache_generation + 1
            if epoch + 1 >= 1 << 31:
                raise RuntimeError("MX gradient return epochs exhausted")
            self.cache_generation = epoch + 1
            guests = tuple(bank.view(torch.float32).view(self.guest_slots, *owner.shape[1:])[:slots]
                           for bank, owner in zip((first, second), owners))
            for slot, expert in enumerate(placement.copies_by_rank[self.rank]):
                if expert >= 0 and placement.owner_rows[home + slot] == 0:
                    for guest in guests:
                        guest[slot].zero_()
            auxiliary = auxiliary.view(torch.int32)
            ready = auxiliary.data_ptr()
            flag_bytes = self.flag_words * auxiliary.element_size()
            ack = ready + 2 * flag_bytes
            done = ack + flag_bytes
            scratch = done + GRADIENT_HELPERS * 64
            records = []
            for index, projection in enumerate((1, 0)):
                owner, guest = owners[projection], guests[projection]
                storage = MXGradientStorage(owner.data_ptr(), guest.data_ptr(), ready + index * flag_bytes, ack, done,
                                            scratch, self.chunk_elements)
                records.append(pack_mx_gradient_return(placement, self.rank, home, host.gradient_ready_events[index],
                                                       owner[0].numel(), storage, epoch + index, self._qp_count))
            table = torch.frombuffer(bytearray(pack_mx_backward_returns(*records)), dtype=torch.uint8)
            yield guests[1], guests[0], _upload_route_tensor(table, self.device)
            self.statistics['gradient_returns'] += 1
        finally:
            self.release()
