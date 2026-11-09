# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""MXFP8 expert executor with current-root weights and private native task graphs."""

import os
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore import _automatic
from hyper_multicore.scheduler.runtime import serialize_runtime_config
from hyper_multicore.scheduler.expert_window import resolve_mx_expert_window

from .async_metadata import AsyncMetadataTransfer
from .heap_manager import _get_heap_manager, _workspace_layout
from .module import MegaMoeExperts, _MegaMoeExecutionResources
from .mxfp8_execution import MXFP8Execution
from .mxfp8_diagnostics import prepare_mx_diagnostics
from .mxfp8_function import execute_mxfp8_experts
from .mxfp8_graph import MXFP8Runtime, build_mxfp8_runtime
from .mxfp8_weights import MXFP8ExpertWeightBinding
from .spec import bind_mega_moe_spec
from .workspace import MegaMoeWorkspace


def _device_bytes(data: bytes, device: Any) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device=device)


@dataclass(frozen=True)
class _DeviceRuntime:
    runtime: torch.Tensor
    gmm: torch.Tensor
    quant: torch.Tensor
    event_capacity: int
    host: MXFP8Runtime
    spec: Any

    @cached_property
    def diagnostics(self) -> Any:
        return prepare_mx_diagnostics(self.host, self.spec, self.runtime.device)


def _device_runtime(host: MXFP8Runtime, device: Any, spec: Any) -> _DeviceRuntime:
    return _DeviceRuntime(
        _device_bytes(serialize_runtime_config(host.config), device),
        _device_bytes(host.gmm_descriptors, device), _device_bytes(host.quant_descriptors, device),
        host.config.event_capacity, host, spec,
    )


@dataclass(frozen=True)
class _MXFP8Plan:
    spec: Any
    forward: _DeviceRuntime
    backward: _DeviceRuntime


class _MXFP8Resources(_MegaMoeExecutionResources):
    """Use established SHMEM ownership without constructing any BF16 compute plan."""

    _extra_manifest_fields = ("mx_vector_policy", "mx_swiglu_row_only", "mx_expert_window", "mx_poll_policy",
                              "mx_dispatch_quant", "mx_schedule_dir", "mx_empty_credits", "mx_ready_depth")

    def __init__(self, specification: dict[str, Any], tensor: torch.Tensor, *, shared: bool,
                 active_specifications: tuple[Mapping[str, Any], ...]) -> None:
        async_mode = os.environ.get('HYPER_MULTICORE_MEGAMOE_ASYNC_METADATA', '0')
        if async_mode not in ('0', '1'):
            raise ValueError('HYPER_MULTICORE_MEGAMOE_ASYNC_METADATA must be 0 or 1.')
        self.metadata_transfer = AsyncMetadataTransfer() if async_mode == '1' else None
        self.spec = bind_mega_moe_spec(specification, tensor)
        if self.spec.dispatch_mode == 'pull':
            from .mxfp8_ops import require_mxfp8_window_feature
            _automatic.collective_call(lambda: require_mxfp8_window_feature('pull_dispatch'), stage='MX Pull feature validation')
        self._expert_window = specification['mx_expert_window']
        self._dispatch_quant = specification['mx_dispatch_quant']
        self._empty_credits = specification['mx_empty_credits']
        self.ready_depth = specification['mx_ready_depth']
        if self.ready_depth:
            from .mxfp8_ops import require_mxfp8_window_feature
            _automatic.collective_call(lambda: require_mxfp8_window_feature('ready_math'), stage='MX ready-queue feature validation')
        self._plans = OrderedDict()
        if dist.is_initialized():
            self._validate_distributed_configuration(active_specifications)
        if self.spec.mx_vector_policy not in ('legacy_odd', 'window2'):
            raise ValueError('MXFP8 supports legacy_odd or window2 scheduling')
        if self.spec.mx_swiglu_row_only:
            from .mxfp8_ops import require_mxfp8_window_feature
            _automatic.collective_call(lambda: require_mxfp8_window_feature('nograd_row_only'), stage='MX row-only feature validation')
        if self.spec.mx_poll_policy == 'fast':
            def require_a5() -> None:
                name = torch.npu.get_device_name(tensor.device)
                if not name.startswith(('Ascend950', 'Ascend910_95')):
                    raise ValueError('Fast MX polling is limited to Ascend 950 experiments.')
            _automatic.collective_call(require_a5, stage='MX hardware validation')
        reservations = tuple((item, _workspace_layout(item, tensor.dtype)) for item in active_specifications)
        self.heap_manager = _get_heap_manager(self.spec.ep_group, tensor)
        added = self.heap_manager.reserve(
            reservations, allow_resize=any(item['dispatch_mode'] == 'push' for item in active_specifications))
        attached = False
        self.plan = self.workspace = None
        try:
            attached = self.heap_manager.attach_runtime()

            def construct() -> None:
                from .udma_order import validate_device_transport
                validate_device_transport(tensor)
                self.workspace = MegaMoeWorkspace(shared=shared)
                self.workspace.capacity_floor = self.heap_manager.bind(self.workspace, self.spec, specification)
                self.workspace.heap_manager = self.heap_manager

            _automatic.collective_call(construct, stage='MX resource construction')
        except Exception:
            self.heap_manager.rollback_reservations(added, self.workspace)
            if attached and not self.heap_manager.workspaces:
                self.heap_manager.detach_runtime()
            raise
        self._closed = self._workspace_closed = False

    def lifecycle_signature(self) -> tuple[Any, ...]:
        signature = super().lifecycle_signature()
        if self.spec.mx_vector_policy == 'all':
            signature = (*signature, 'mx_aiv_all_v2')
        elif self.spec.mx_vector_policy == 'single_vector':
            signature = (*signature, 'mx_single_vector_v3')
        if self.spec.mx_swiglu_row_only:
            signature = (*signature, 'mx_swiglu_row_only_v1')
        if self._expert_window:
            signature = (*signature, 'mx_expert_window', self._expert_window)
        if self.spec.mx_vector_policy in ('roles', 'roles_ready'):
            signature = (*signature, self.spec.mx_vector_policy, 'mx_roles_v4')
        elif self.spec.mx_vector_policy == 'planned':
            signature = (*signature, 'mx_planned_v6', self.spec.mx_schedule_dir)
        elif self.spec.mx_vector_policy == 'window2':
            signature = (*signature, 'mx_window2_v6')
        elif self.spec.mx_vector_policy == 'roles_rule':
            signature = (*signature, 'mx_roles_rule_v7')
        if self._empty_credits:
            signature = (*signature, 'mx_empty_credits_v1')
        if self.ready_depth:
            signature = (*signature, 'ready_math', self.ready_depth)
        if self._dispatch_quant:
            signature = (*signature, 'mx_dispatch_quant_v2')
        return (*signature, 'mx_poll_fast_10_30') if self.spec.mx_poll_policy == 'fast' else signature

    def close(self) -> None:
        super().close()
        if self.metadata_transfer is not None:
            self.metadata_transfer.close_after_synchronize()

    def plan_for_capacity(self, capacity: int, device: Any, *, swiglu_row_only: bool = False) -> _MXFP8Plan:
        """Cache pointer-free plans independently of Pull's live receive size."""
        if self.spec.dispatch_mode == 'pull':
            # Dynamic GMM resolves each expert's real rows from group_list before
            # choosing tiles or checking buffer ranges. This is a descriptor bound,
            # not an allocation size; Pull buffers retain their live route size.
            # Only M/K are uint32 fields; byte offsets and products stay uint64.
            limit = min(self.spec.ep_size * self.spec.routed_slots, (1 << 32) - 1)
            if not 1 <= capacity <= limit:
                raise ValueError(f'Pull receive capacity must be in [1, {limit}], got {capacity}')
            capacity = limit
        key = (capacity, True) if swiglu_row_only else capacity
        if key not in self._plans:
            spec = replace(self.spec, receive_capacity=capacity)
            self._plans[key] = _MXFP8Plan(
                spec, _device_runtime(
                    build_mxfp8_runtime(spec, forward=True, swiglu_row_only=swiglu_row_only,
                                        expert_window=self._expert_window, dispatch_quant=self._dispatch_quant,
                                        aggregate_empty_credits=self._empty_credits, ready_depth=self.ready_depth),
                    device, spec),
                _device_runtime(build_mxfp8_runtime(spec, forward=False, expert_window=self._expert_window,
                                                  dispatch_quant=self._dispatch_quant,
                                                  aggregate_empty_credits=self._empty_credits,
                                                  ready_depth=self.ready_depth), device, spec),
            )
        self._plans.move_to_end(key)
        while len(self._plans) > 4:
            self._plans.popitem(last=False)
        return self._plans[key]


class MXFP8MegaMoeExperts(MegaMoeExperts):
    """Execute fused MX expert tasks using externally owned, freshly gathered roots."""

    def __init__(self, *, local_num_tokens: int, hidden_size: int, intermediate_size: int,
                 num_experts: int, top_k: int, ep_size: int = 1, ep_group: Any = None,
                 create_parameters: bool = False, swiglu_limit: float | None = None,
                 dispatch_mode: str | None = None, **capacity_options: Any) -> None:
        if hidden_size % 64 or intermediate_size % 64:
            raise ValueError('MXFP8 MegaMoE hidden/intermediate dimensions must be divisible by 64.')
        selected = os.environ.get('HYPER_MULTICORE_MEGAMOE_DISPATCH_MODE')
        if dispatch_mode not in (None, 'push', 'pull') or selected not in (None, 'push', 'pull'):
            raise ValueError('MX dispatch mode and HYPER_MULTICORE_MEGAMOE_DISPATCH_MODE must be push or pull')
        if dispatch_mode is not None and selected is not None and dispatch_mode != selected:
            raise ValueError('Explicit MX dispatch mode conflicts with HYPER_MULTICORE_MEGAMOE_DISPATCH_MODE')
        dispatch_mode = selected or dispatch_mode or 'push'
        if create_parameters or swiglu_limit is not None:
            raise ValueError('MXFP8 MegaMoE requires external roots and unclamped SwiGLU.')
        if dispatch_mode == 'pull':
            capacity_options['initial_capacity_factor'] = None
            capacity_options['capacity_growth_factor'] = None
        communication_split = os.environ.get('HYPER_MULTICORE_MX_COMM_SPLIT')
        if communication_split is not None:
            if communication_split not in ('128', '256', '512', '4096'):
                raise ValueError('HYPER_MULTICORE_MX_COMM_SPLIT must be exactly 128, 256, 512 or 4096.')
            split = int(communication_split)
            for key in ('dispatch_split', 'combine_split'):
                if key in capacity_options:
                    explicit = capacity_options[key]
                    if not isinstance(explicit, int) or isinstance(explicit, bool) or explicit != split:
                        raise ValueError(f'{key}={explicit!r} conflicts with HYPER_MULTICORE_MX_COMM_SPLIT={split}.')
                capacity_options[key] = split
        no_clone = os.environ.get('HYPER_MULTICORE_MX_NOGRAD_NO_CLONE', '0')
        if no_clone not in ('0', '1'):
            raise ValueError('HYPER_MULTICORE_MX_NOGRAD_NO_CLONE must be 0 or 1.')
        super().__init__(
            local_num_tokens=local_num_tokens, hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_experts=num_experts, top_k=top_k, ep_size=ep_size, ep_group=ep_group,
            create_parameters=False, swiglu_limit=None, dispatch_mode=dispatch_mode, **capacity_options,
        )
        self.nograd_no_clone = no_clone == '1'

    @classmethod
    def _resolve_specification(cls, specification: dict[str, Any]) -> dict[str, Any]:
        specification = super()._resolve_specification(specification)
        policy = os.environ.get('HYPER_MULTICORE_MX_AIV_POLICY', 'legacy_odd')
        if policy not in ('legacy_odd', 'window2'):
            raise ValueError('HYPER_MULTICORE_MX_AIV_POLICY must be legacy_odd or window2.')
        if specification['dispatch_mode'] == 'pull' and policy != 'window2':
            raise ValueError('MX Pull requires the qualified window2 source-completion protocol')
        specification['mx_vector_policy'] = policy
        specification['mx_schedule_dir'] = os.environ.get('HYPER_MULTICORE_MX_SCHEDULE_DIR', '')
        if policy == 'planned' and not specification['mx_schedule_dir']:
            raise ValueError('Planned queues require HYPER_MULTICORE_MX_SCHEDULE_DIR')
        row_only = os.environ.get('HYPER_MULTICORE_MX_SWIGLU_ROW_ONLY', '0')
        if row_only not in ('0', '1'):
            raise ValueError('HYPER_MULTICORE_MX_SWIGLU_ROW_ONLY must be 0 or 1.')
        specification['mx_swiglu_row_only'] = row_only == '1'
        specification['mx_expert_window'] = resolve_mx_expert_window(
            os.environ.get('HYPER_MULTICORE_MX_EXPERT_WINDOW', '0'))
        policy = os.environ.get('HYPER_MULTICORE_MX_POLL_POLICY', '0')
        if policy not in ('0', 'fast'):
            raise ValueError('HYPER_MULTICORE_MX_POLL_POLICY must be exactly 0 or fast.')
        specification['mx_poll_policy'] = policy
        dispatch_quant = os.environ.get('HYPER_MULTICORE_MX_DISPATCH_QUANT', '0')
        if dispatch_quant not in ('0', '1'):
            raise ValueError('HYPER_MULTICORE_MX_DISPATCH_QUANT must be 0 or 1.')
        specification['mx_dispatch_quant'] = dispatch_quant == '1'
        empty_credits = os.environ.get('HYPER_MULTICORE_MX_EMPTY_CREDITS', '0')
        if empty_credits not in ('0', '1'):
            raise ValueError('HYPER_MULTICORE_MX_EMPTY_CREDITS must be 0 or 1.')
        specification['mx_empty_credits'] = empty_credits == '1'
        depth = os.environ.get('HYPER_MULTICORE_MX_READY_DEPTH', '0')
        if depth not in ('0', '1', '8'):
            raise ValueError('HYPER_MULTICORE_MX_READY_DEPTH must be 0, 1 or 8')
        if depth != '0' and specification['mx_vector_policy'] != 'window2':
            raise ValueError('HYPER_MULTICORE_MX_READY_DEPTH requires window2')
        specification['mx_ready_depth'] = int(depth)
        return specification

    def _create_execution_resources(self, tensor: torch.Tensor, *, shared: bool,
                                    active_specifications: tuple[Mapping[str, Any], ...]) -> _MXFP8Resources:
        return _MXFP8Resources(self._resource_group.specification, tensor, shared=shared,
                               active_specifications=active_specifications)

    def _validate_tensors(self, hidden_states: torch.Tensor, expert_weights: tuple) -> None:
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_npu:
            raise TypeError('MXFP8 MegaMoE requires BF16 NPU hidden states.')
        shapes = ((self.local_experts, self.hidden_size, 2 * self.intermediate_size),
                  (self.local_experts, self.intermediate_size, self.hidden_size))
        root1, root2 = expert_weights
        for root, shape in zip((root1, root2), shapes):
            MXFP8ExpertWeightBinding(root, shape)

    def forward(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                *, expert_weights: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        flat = self._validate_forward_inputs(hidden_states, topk_ids, topk_weights, None, expert_weights)
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError('MXFP8 MegaMoE push transport does not support graph capture.')
        resources = self._get_execution_resources(flat)
        shapes = ((self.local_experts, self.hidden_size, 2 * self.intermediate_size),
                  (self.local_experts, self.intermediate_size, self.hidden_size))
        return execute_mxfp8_experts(hidden_states, *expert_weights, topk_ids, topk_weights,
                                     shapes, MXFP8Execution(resources, nograd_no_clone=self.nograd_no_clone))
