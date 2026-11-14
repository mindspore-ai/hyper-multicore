"""MoE scheduling policies; task mathematics, transport and runtime remain separate."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from types import MappingProxyType

from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.scheduler.worker_queues import install_worker_queues
from .transport import TRANSPORT_TASKS, transport_queues, validate_transport_owners

TWO_EXPERT_PIPELINE = 'two_expert_pipeline'
BUILDER_ORDER = 'builder_order'


@dataclass(frozen=True)
class MegaMoeScheduleOptions:
    """Select a policy and local math-queue lookahead before resource binding.

    None resolves from the environment, then the existing defaults. Zero keeps
    FIFO; one polls only the head; eight permits bounded head-first selection.
    Communication/termination remain FIFO. No cross-worker task stealing occurs.
    """
    policy: str | None = None
    queue_ready_lookahead: int | None = None


def _option(explicit, names, default):
    values = [os.environ[name] for name in names if name in os.environ]
    if explicit is not None:
        values.append(str(explicit))
    if len(set(values)) > 1:
        raise ValueError(f'Conflicting MegaMoE schedule options: {", ".join(names)}')
    return values[0] if values else str(default)


def resolve_schedule_options(options=None, *, precision='bf16'):
    """Resolve once; reject conflicting aliases rather than silently overriding."""
    if options is not None and not isinstance(options, MegaMoeScheduleOptions):
        raise TypeError('schedule must be MegaMoeScheduleOptions or None')
    options = options or MegaMoeScheduleOptions()
    policy = _option(options.policy, ('HYPER_MULTICORE_MEGAMOE_SCHEDULE_POLICY',), TWO_EXPERT_PIPELINE)
    describe_schedule_policy(policy)
    value = _option(options.queue_ready_lookahead,
                    ('HYPER_MULTICORE_MEGAMOE_QUEUE_READY_LOOKAHEAD',
                     'HYPER_MULTICORE_MEGAMOE_READY_DEPTH',
                     f'HYPER_MULTICORE_{"MX" if precision == "mxfp8" else "BF16"}_READY_DEPTH'), 0)
    if value not in ('0', '1', '8') or isinstance(options.queue_ready_lookahead, bool):
        raise ValueError('MegaMoE queue ready lookahead must be 0, 1 or 8')
    return MegaMoeScheduleOptions(policy, int(value))


@dataclass(frozen=True)
class ExpertStages:
    """Bind logical math roles to existing task ranges, without changing the DAG."""
    ranges: dict
    cubes: tuple
    vectors: tuple
    cores: int
    experts: int
    forward: bool
    original_cubes: tuple
    original_vectors: tuple

    def tasks(self, role, expert, lane):
        begin, count = self.ranges[role]
        if count % self.experts:
            raise ValueError(f'Stage {role} does not partition over local experts')
        width = count // self.experts
        return range(begin + expert * width + lane, begin + (expert + 1) * width, self.cores)


def _expert_stages(graph, spec, forward, cfg):
    # These are mathematical roles, not descriptor ports or physical worker IDs.
    aliases = ({'up': 'up_proj', 'down': 'down_proj', 'activation': 'swiglu',
                'dw_down': 'w2_grad', 'da': 'act_grad', 'dx': 'gate_grad',
                'dw_up': 'w1_grad', 'activation_grad': 'swiglu_grad'}
               if spec.precision == 'bf16' else
               {'activation': 'swiglu_quant', 'dw_down': 'dw2', 'dw_up': 'dw1',
                'quant_input': 'quant_x' if forward else 'quant_dy',
                'activation_grad': 'swiglu_grad_quant'})
    cubes = ('up', 'down') if forward else ('dw_down', 'da', 'dx', 'dw_up')
    vectors = ('activation',) if forward else ('activation_grad',)
    if spec.precision == 'mxfp8':
        if not forward:
            cubes = ('dw_down', 'da', 'dw_up', 'dx')
        vectors = ('quant_input', *vectors)
    offsets, offset = {}, 0
    for op in graph.topological_sort():
        offsets[op.name] = (offset, op.task_num)
        offset += op.task_num
    ranges = {role: offsets[aliases.get(role, role)] for role in (*cubes, *vectors)}
    def project(indices, roles):
        queues = [[] for _ in range(spec.num_cube_cores)]
        for tid in indices:
            for role in roles:
                begin, count = ranges[role]
                if begin <= tid < begin + count:
                    width = count // spec.local_experts
                    queues[(tid - begin) % width % spec.num_cube_cores].append(int(tid))
                    break
        return tuple(tuple(queue) for queue in queues)
    return ExpertStages(MappingProxyType(ranges), cubes, vectors, spec.num_cube_cores,
                        spec.local_experts, forward,
                        project(cfg.cube_task_indices[:cfg.task_index_num[0]], cubes),
                        project(cfg.vector_task_indices[:cfg.task_index_num[1]], vectors))


def _two_expert_pipeline(stages):
    """Interleave adjacent experts; leave task coverage and dependencies intact."""
    cubes = [[] for _ in range(stages.cores)]
    vectors = [[] for _ in range(2 * stages.cores)]
    front = 1 if stages.forward else 2
    for lane in range(stages.cores):
        for expert in range(stages.experts + 1):
            if expert < stages.experts:
                for role in stages.cubes[:front]:
                    cubes[lane].extend(stages.tasks(role, expert, lane))
            if expert:
                for role in stages.cubes[front:]:
                    cubes[lane].extend(stages.tasks(role, expert - 1, lane))
        if len(stages.vectors) == 2:
            vectors[2 * lane].extend(stages.tasks(stages.vectors[0], 0, lane))
        for expert in range(stages.experts):
            if len(stages.vectors) == 2 and expert + 1 < stages.experts:
                vectors[2 * lane].extend(stages.tasks(stages.vectors[0], expert + 1, lane))
            vectors[2 * lane].extend(stages.tasks(stages.vectors[-1], expert, lane))
    return cubes, vectors


def _builder_order(stages):
    """Keep builder math order on the shared fixed owners; no legacy QP rollback."""
    vectors = [[] for _ in range(2 * stages.cores)]
    for lane, queue in enumerate(stages.original_vectors):
        vectors[2 * lane] = list(queue)
    return [list(queue) for queue in stages.original_cubes], vectors


# Add implemented policies here; no placeholder strategies or dynamic plugins.
_POLICIES = {TWO_EXPERT_PIPELINE: _two_expert_pipeline, BUILDER_ORDER: _builder_order}


def list_schedule_policies():
    return tuple(_POLICIES)


def describe_schedule_policy(name):
    if name not in _POLICIES:
        raise ValueError(f'Unknown MegaMoE schedule policy {name!r}; available: {list_schedule_policies()}')
    return {'name': name, 'precision': ['bf16', 'mxfp8'], 'phases': ['forward', 'backward'],
            'queue_ready_lookahead': [0, 1, 8], 'transport_fifo': True,
            'legacy_bf16_mte': 'preserved; explicit math-queue selection requires UDMA'}


def configure_expert_queues(cfg, graph, spec, *, forward):
    """Compile a policy over fixed QP0 owners and serialize the existing queue ABI."""
    if spec.transport_engine != 'udma' and spec.precision == 'bf16':
        if spec.ready_depth:
            raise ValueError('BF16 MTE legacy queues do not support explicit math queue ready selection')
        return cfg
    stages = _expert_stages(graph, spec, forward, cfg)
    describe_schedule_policy(spec.schedule_policy)
    cubes, math = _POLICIES[spec.schedule_policy](stages)
    communication = [tid for tid in cfg.vector_task_indices[:cfg.task_index_num[1]]
                     if cfg.all_tasks[tid].task_type in (*TRANSPORT_TASKS, TaskType.TASK_TERMINATE)]
    vectors = transport_queues(cfg, spec, communication)
    for queue, compute in zip(vectors, math):
        queue.extend(compute)
    validate_transport_owners(cfg, spec, vectors)
    result = install_worker_queues(cfg, cubes, vectors, cube_ready_depth=spec.ready_depth,
                                   vector_ready_depths=[spec.ready_depth if i % 2 == 0 else 0
                                                        for i in range(2 * stages.cores)])
    result.schedule_info = {'policy': spec.schedule_policy, 'precision': spec.precision,
                            'phase': 'forward' if forward else 'backward',
                            'ep_size': spec.ep_size, 'ep_rank': spec.rank_id,
                            'stages': dict(stages.ranges)}
    return result


def explain_schedule(cfg):
    """Export static queues and actual dependencies without device reads or logging."""
    from hyper_multicore.scheduler.runtime import serialize_runtime_config
    if cfg.queue_layout != 1:
        return {'layout': 'legacy'}
    cubes = cfg.num_workers // 2
    queues = []
    for worker, queue in enumerate(cfg.worker_queues):
        indices = cfg.cube_task_indices if worker < cubes else cfg.vector_task_indices
        tasks = [int(tid) for tid in indices[queue.begin:queue.end]]
        queues.append({'kind': 'aic' if worker < cubes else 'aiv',
                       'worker': worker if worker < cubes else worker - cubes,
                       'queue_ready_lookahead': queue.ready_depth, 'tasks': tasks})
    ids = [tid for queue in queues for tid in queue['tasks']]
    result = {'layout': 'explicit', 'schedule': getattr(cfg, 'schedule_info', {}), 'queues': queues,
            'tasks': {tid: {'type': int(cfg.all_tasks[tid].task_type),
                            'dependent_event': int(cfg.all_tasks[tid].dependent_event),
                            'trigger_event': int(cfg.all_tasks[tid].trigger_event)} for tid in ids},
            'event_triggers': list(cfg.all_event_num_triggers[:cfg.event_capacity])}
    result['plan_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    result['runtime_sha256'] = hashlib.sha256(serialize_runtime_config(cfg)).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('list', 'describe', 'compare'))
    parser.add_argument('items', nargs='*')
    args = parser.parse_args()
    if args.action == 'list':
        result = [describe_schedule_policy(name) for name in list_schedule_policies()]
    elif args.action == 'describe' and len(args.items) == 1:
        result = describe_schedule_policy(args.items[0])
    elif args.action == 'compare' and len(args.items) == 2:
        from pathlib import Path
        a, b = (json.loads(Path(path).read_text()) for path in args.items)
        result = {key: {'left': a.get(key), 'right': b.get(key)}
                  for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
    else:
        parser.error('describe requires one policy; compare requires two exported JSON paths')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
