# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""MX experts reuse the managed MoE lifecycle and precision-independent transport."""
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import cached_property

import torch

from hyper_multicore.scheduler.runtime import serialize_runtime_config
from ...resources import _MegaMoeExecutionResources
from .diagnostics import prepare_mx_diagnostics
from .execution import MXFP8Execution
from .autograd import execute_mxfp8_experts
from .graph import build_mxfp8_runtime
from .weights import MXFP8ExpertWeightBinding


def _device_bytes(data, device):
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device=device)


@dataclass
class _DeviceRuntime:
    runtime: torch.Tensor
    gmm: torch.Tensor
    quant: torch.Tensor
    event_capacity: int
    host: object
    spec: object

    @cached_property
    def diagnostics(self):
        return prepare_mx_diagnostics(self.host, self.spec, self.runtime.device)


def _device_runtime(host, device, spec):
    return _DeviceRuntime(_device_bytes(serialize_runtime_config(host.config), device),
                          _device_bytes(host.gmm_descriptors, device),
                          _device_bytes(host.quant_descriptors, device), host.config.event_capacity, host, spec)


@dataclass(frozen=True)
class _MXFP8Plan:
    spec: object
    forward: _DeviceRuntime
    backward: _DeviceRuntime


class Resources(_MegaMoeExecutionResources):
    def __init__(self, specification, tensor, *, shared, active_specifications):
        if query_npu_arch() != 3510:
            raise ValueError("MX expert mathematics currently requires A5")
        super().__init__(specification, tensor, shared=shared, active_specifications=active_specifications)
        self._plans = OrderedDict()

    def _initial_plan(self, tensor):
        # MX plans are selected by physical capacity; never construct a BF16 math plan.
        return None

    def plan_for_capacity(self, capacity, device):
        """Cache pointer-free Pull plans independently of the live per-expert loads."""
        if self.spec.dispatch_mode == 'pull':
            limit = min(self.spec.ep_size * self.spec.routed_slots, (1 << 32) - 1)
            if not 1 <= capacity <= limit:
                raise ValueError(f"Pull receive capacity must be in [1, {limit}]")
            capacity = limit
        key = capacity
        if key not in self._plans:
            spec = replace(self.spec, receive_capacity=capacity)
            self._plans[key] = _MXFP8Plan(
                spec, _device_runtime(build_mxfp8_runtime(spec, forward=True), device, spec),
                _device_runtime(build_mxfp8_runtime(spec, forward=False), device, spec))
        self._plans.move_to_end(key)
        while len(self._plans) > 4:
            self._plans.popitem(last=False)
        return self._plans[key]


def validate_weights(module, weights, hidden):
    shapes = ((module.local_experts, module.hidden_size, 2 * module.intermediate_size),
              (module.local_experts, module.intermediate_size, module.hidden_size))
    for root, shape in zip(weights, shapes):
        MXFP8ExpertWeightBinding(root, shape)


def forward(module, hidden_states, flat, topk_ids, topk_weights,
            tokens_per_expert, weights, resources):
    shapes = ((module.local_experts, module.hidden_size, 2 * module.intermediate_size),
              (module.local_experts, module.intermediate_size, module.hidden_size))
    return execute_mxfp8_experts(hidden_states, *weights, topk_ids, topk_weights, shapes,
                                MXFP8Execution(resources, tokens_per_expert=tokens_per_expert,
                                               nograd_no_clone=module.nograd_no_clone))
