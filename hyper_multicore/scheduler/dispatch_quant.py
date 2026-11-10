# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Source-fragment waves preserve progress with blocking AIV worker queues."""
from dataclasses import dataclass
from typing import Any, Callable

from .mxfp8_quant import MX_DISPATCH_QUANT_TASK, MxDispatchQuantDescriptor


@dataclass
class DispatchQuantStage:
    columns: int
    vectors: int
    fragments: int
    fragment_rows: int
    event_base: int
    ep: int
    local_experts: int
    offset_position: int
    size_position: int
    records: list
    fill_stage: Callable

    @property
    def wave_workers(self) -> int:
        return 4

    def fill(self, cfg: Any, op: Any, tsv: Any) -> None:
        first_task = tsv.pre_task_num
        indices = []
        for expert in range(self.local_experts):
            for fragment in (*range(self.fragments), -1):
                workers = self.vectors if fragment == -1 else self.wave_workers
                for worker in range(workers):
                    indices.append(len(self.records))
                    self.records.append(MxDispatchQuantDescriptor(
                        self.columns, expert, worker, workers, fragment, self.fragment_rows,
                        self.offset_position, self.size_position, self.ep, self.local_experts))
        per_expert = self.fragments * self.wave_workers + self.vectors
        self.fill_stage(cfg, op, tsv, MX_DISPATCH_QUANT_TASK, per_expert, True, indices)
        interior_base = self.event_base + self.local_experts * self.fragments
        for expert in range(self.local_experts):
            interior = interior_base + expert
            cfg.all_event_num_triggers[interior] = self.fragments * self.wave_workers
            for index in range(per_expert):
                task = cfg.all_tasks[first_task + expert * per_expert + index]
                if index < self.fragments * self.wave_workers:
                    fragment = index // self.wave_workers
                    task.dependent_event = self.event_base + expert * self.fragments + fragment
                    task.trigger_event = interior
                else:
                    task.dependent_event = interior
                    cfg.all_event_num_triggers[task.trigger_event] = self.vectors


def configure_dispatch_quant_queue(cfg: Any, graph: Any, stage: DispatchQuantStage, rank: int) -> None:
    """Wire fragment credits without changing the communication projection."""
    ranges, start = {}, 0
    for op in graph.topological_sort():
        ranges[op.name] = (start, start + op.task_num)
        start += op.task_num
    dispatch_start, dispatch_end = ranges['dispatch']
    quant_name = 'quant_x' if 'quant_x' in ranges else 'quant_dy'
    quant_start, quant_end = ranges[quant_name]
    replacement = []
    per_expert = stage.fragments * stage.wave_workers + stage.vectors
    peers = [(peer + rank) % stage.ep for peer in range(stage.ep)]
    for expert in range(stage.local_experts):
        for fragment in range(stage.fragments):
            event = stage.event_base + expert * stage.fragments + fragment
            cfg.all_event_num_triggers[event] = stage.ep
            for peer in peers:
                task_id = dispatch_start + (peer * stage.local_experts + expert) * stage.fragments + fragment
                cfg.all_tasks[task_id].trigger_event = event
                replacement.append(task_id)
            first = quant_start + expert * per_expert + fragment * stage.wave_workers
            replacement.extend(range(first, first + stage.wave_workers))
        first = quant_start + expert * per_expert + stage.fragments * stage.wave_workers
        replacement.extend(range(first, first + stage.vectors))
    # Role-separated workers do not need communication/quantization wave weaving.
    # Preserve communication projection and its physical QP owners exactly.
