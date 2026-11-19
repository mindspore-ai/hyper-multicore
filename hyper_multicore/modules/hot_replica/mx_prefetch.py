# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lease phase-current MX guest weight buffers during projection prefetch."""
from contextlib import contextmanager
import struct

import torch
import torch.distributed as dist
from hyper_multicore.modules.mega_moe.route import _upload_route_tensor
from hyper_multicore.torch.ops import _load_native

from ._planner.plan import ExpertReplicaTransfer


def _transfers(placement, home):
    """Use the common target-rank/slot order for matching P2P submissions."""
    return tuple(
        ExpertReplicaTransfer(expert, expert // home, expert % home, rank, slot)
        for rank, row in enumerate(placement.copies_by_rank)
        for slot, expert in enumerate(row)
        if expert >= 0
    )


class MXWeightPrefetch:
    """Keep only guest storage and order its reuse after the fused consumers."""

    def __init__(self, group, device, home_experts, guest_slots):
        self.group = group
        self.rank = dist.get_rank(group)
        self.members = dist.get_process_group_ranks(group)
        self.home_experts = home_experts
        self.guest_slots = guest_slots
        self.stream = torch.npu.Stream(device=device)
        self.done = torch.npu.Event()
        self.consumed = torch.npu.Event()
        self.banks = {}
        self.ready = torch.zeros((2, guest_slots, 16), dtype=torch.int32, device=device)
        self.epoch = 0
        self.used = False
        self.leased = False

    def _guest_banks(self, tensors, backward):
        """Reuse data/scale storage within each physical axis."""
        shape = tuple((tensor.dtype, tensor.shape[1:]) for tensor in tensors)
        if backward not in self.banks:
            self.banks[backward] = shape, tuple(
                torch.empty((self.guest_slots, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
                for tensor in tensors
            )
        expected, banks = self.banks[backward]
        if shape != expected:
            raise ValueError("MX guest buffers disagree with the bound expert weight axes")
        return banks

    def _projection(self, originals, banks, transfers, projection, pending):
        """Submit current weight bytes and scales in a deterministic peer order."""
        requests = []
        for edge in transfers:
            for index in (2 * projection, 2 * projection + 1):
                if edge.owner_rank == self.rank:
                    value = originals[index][edge.owner_slot].view(torch.uint8)
                    requests.append(dist.P2POp(dist.isend, value, self.members[edge.target_rank], self.group))
                elif edge.target_rank == self.rank:
                    value = banks[index][edge.target_slot].view(torch.uint8)
                    requests.append(dist.P2POp(dist.irecv, value, self.members[edge.owner_rank], self.group))
        if requests:
            submitted = dist.batch_isend_irecv(requests)
            pending.extend(submitted)
            for work in submitted:
                work.wait()

    @contextmanager
    def lease(self, weights, placement, *, backward):
        """Publish one projection at a time and keep its weight bytes alive until use."""
        if self.leased:
            raise RuntimeError("MX guest buffers already have an active lease")
        fields = ("row_data", "row_scale") if backward else ("col_data", "col_scale")
        originals = tuple(getattr(weight, field) for weight in weights for field in fields)
        if any(tensor is None or not tensor.is_contiguous() or tensor.shape[0] != self.home_experts
               for tensor in originals):
            raise ValueError("MX guest transfer requires every current contiguous data/scale axis")
        if placement.borrowed_capacity != self.guest_slots:
            raise ValueError("MX guest placement must match the bound buffer extent")
        if self.epoch == (1 << 31) - 1:
            raise RuntimeError("MX projection readiness exhausted its epoch range")
        _load_native()
        publish_ready = torch.ops.hyper_multicore.hot_replica_publish_mx_projection_ready_
        banks = self._guest_banks(originals, backward)
        transfers = _transfers(placement, self.home_experts)
        active = tuple(edge.target_slot for edge in transfers if edge.target_rank == self.rank)
        caller = torch.npu.current_stream(originals[0].device)
        self.epoch += 1
        ready = struct.pack("<5Q", self.epoch, self.ready[0].data_ptr(), self.ready[1].data_ptr(),
                            self.home_experts, self.guest_slots)
        metadata = _upload_route_tensor(torch.frombuffer(bytearray(ready), dtype=torch.uint8), originals[0].device)
        # The published int32 epoch is the low word of the same launch descriptor.
        stamp = metadata[:4].view(torch.int32)
        self.leased = True
        pending = []
        try:
            if self.used:
                self.stream.wait_event(self.consumed)
            self.stream.wait_stream(caller)
            with torch.npu.stream(self.stream):
                for tensor in (*originals, *banks, stamp):
                    tensor.record_stream(self.stream)
                for projection in ((1, 0) if backward else (0, 1)):
                    self._projection(originals, banks, transfers, projection, pending)
                    for slot in active:
                        publish_ready(self.ready[projection, slot, :1], stamp)
            yield banks, metadata
        except BaseException:
            # Preserve the lifetime of any HCCL work submitted before a failure.
            for work in pending:
                try:
                    work.wait()
                except RuntimeError:
                    continue
            raise
        finally:
            self.done.record(self.stream)
            caller.wait_event(self.done)
            self.consumed.record(caller)
            self.used, self.leased = True, False
