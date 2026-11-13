"""MX experts reuse the managed MoE lifecycle and precision-independent transport."""
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import cached_property
import os

import torch

from hyper_multicore._loader import query_npu_arch
from hyper_multicore.scheduler.runtime import serialize_runtime_config
from .module import MegaMoeExperts, _MegaMoeExecutionResources
from .mxfp8_diagnostics import prepare_mx_diagnostics
from .mxfp8_execution import MXFP8Execution
from .mxfp8_function import execute_mxfp8_experts
from .mxfp8_graph import build_mxfp8_runtime
from .mxfp8_weights import MXFP8ExpertWeightBinding


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


class _MXFP8Resources(_MegaMoeExecutionResources):
    def __init__(self, specification, tensor, *, shared, active_specifications):
        if query_npu_arch() != 3510:
            raise ValueError("MX expert mathematics currently requires A5")
        super().__init__(specification, tensor, shared=shared, active_specifications=active_specifications)
        self._plans = OrderedDict()
        self.ready_depth = self.spec.ready_depth

    def _initial_plan(self, tensor):
        # MX plans are selected by physical capacity; never construct a BF16 math plan.
        return None

    def plan_for_capacity(self, capacity, device, *, swiglu_row_only=False):
        """Cache pointer-free Pull plans independently of the live per-expert loads."""
        if self.spec.dispatch_mode == 'pull':
            limit = min(self.spec.ep_size * self.spec.routed_slots, (1 << 32) - 1)
            if not 1 <= capacity <= limit:
                raise ValueError(f"Pull receive capacity must be in [1, {limit}]")
            capacity = limit
        key = (capacity, swiglu_row_only)
        if key not in self._plans:
            spec = replace(self.spec, receive_capacity=capacity)
            self._plans[key] = _MXFP8Plan(
                spec, _device_runtime(build_mxfp8_runtime(spec, forward=True,
                                                         swiglu_row_only=swiglu_row_only), device, spec),
                _device_runtime(build_mxfp8_runtime(spec, forward=False), device, spec))
        self._plans.move_to_end(key)
        while len(self._plans) > 4:
            self._plans.popitem(last=False)
        return self._plans[key]


class MXFP8MegaMoeExperts(MegaMoeExperts):
    """Accept BF16 activations and externally materialized E4M3FN/E8M0 expert roots."""
    precision = 'mxfp8'

    def __init__(self, *, local_num_tokens, hidden_size, intermediate_size, num_experts, top_k,
                 ep_size=1, ep_group=None, **kwargs):
        if hidden_size % 64 or intermediate_size % 64:
            raise ValueError("MX hidden/intermediate dimensions must be divisible by 64")
        if kwargs.pop('create_parameters', False):
            raise ValueError("MX expert roots must be supplied by their physical-storage owner")
        if kwargs.get('expert_weight_layout', 'EIO') != 'EIO':
            raise ValueError("MX physical expert bindings currently require logical EIO layout")
        super().__init__(local_num_tokens=local_num_tokens, hidden_size=hidden_size,
                         intermediate_size=intermediate_size, num_experts=num_experts, top_k=top_k,
                         ep_size=ep_size, ep_group=ep_group, create_parameters=False, **kwargs)
        policy = os.environ.get('HYPER_MULTICORE_MX_NOGRAD_NO_CLONE', '1')
        if policy not in ('0', '1'):
            raise ValueError("MX no-grad no-clone must be 0 or 1")
        self.nograd_no_clone = policy == '1'

    @classmethod
    def _resolve_specification(cls, specification):
        result = super()._resolve_specification(specification)
        policy = os.environ.get('HYPER_MULTICORE_MX_SWIGLU_ROW_ONLY', '1')
        if policy not in ('0', '1'):
            raise ValueError("MX no-grad row-only policy must be 0 or 1")
        result['swiglu_row_only'] = policy == '1'
        return result

    def _create_execution_resources(self, tensor, *, shared, active_specifications):
        return _MXFP8Resources(self._resource_group.specification, tensor, shared=shared,
                               active_specifications=active_specifications)

    def _validate_tensors(self, hidden_states, expert_weights):
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_npu:
            raise TypeError("MX experts require BF16 NPU hidden states")
        shapes = ((self.local_experts, self.hidden_size, 2 * self.intermediate_size),
                  (self.local_experts, self.intermediate_size, self.hidden_size))
        for root, shape in zip(expert_weights, shapes):
            MXFP8ExpertWeightBinding(root, shape)

    def forward(self, hidden_states, topk_ids, topk_weights, *, expert_weights):
        flat = self._validate_forward_inputs(hidden_states, topk_ids, topk_weights, None, expert_weights)
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MX experts do not currently support graph capture")
        resources = self._get_execution_resources(flat)
        shapes = ((self.local_experts, self.hidden_size, 2 * self.intermediate_size),
                  (self.local_experts, self.intermediate_size, self.hidden_size))
        return execute_mxfp8_experts(hidden_states, *expert_weights, topk_ids, topk_weights, shapes,
                                     MXFP8Execution(resources, nograd_no_clone=self.nograd_no_clone))
