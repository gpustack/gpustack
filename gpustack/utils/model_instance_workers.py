from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Set, Union

from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    Model,
    ModelBase,
    ModelConfigSnapshot,
    ModelInstance,
    ModelInstanceLegacyStop,
    ModelInstanceBase,
    ModelInstanceWorkload,
    ModelInstanceWorkloadStateEnum,
    ModelInstanceMemberReport,
    ModelInstanceWorkloadObservation,
    ModelInstancePreparation,
    ModelInstanceStateEnum,
    member_worker_ids,
    role_effective_model,
)
from gpustack.utils.command import (
    find_bool_parameter,
    find_parameter,
)

from gpustack.schemas.workers import Worker


@dataclass(frozen=True)
class ModelInstanceWorkerMatch:
    is_main_worker: bool = False
    subordinate_worker_indexes: tuple[int, ...] = ()
    workload_indexes: tuple[int, ...] = ()

    @property
    def matched(self) -> bool:
        return (
            self.is_main_worker
            or bool(self.subordinate_worker_indexes)
            or bool(self.workload_indexes)
        )


def get_model_instance_worker_match(
    instance: ModelInstance,
    *,
    worker_name: Optional[str] = None,
    worker_id: Optional[int] = None,
) -> ModelInstanceWorkerMatch:
    workloads = get_model_instance_workloads(instance)
    # Managed ownership is fenced by persistent Worker IDs, not reusable names.
    if workloads is not None or model_instance_legacy_stop(instance) is not None:
        worker_name = None
    is_main_worker = False
    if worker_name is not None and instance.worker_name == worker_name:
        is_main_worker = True
    if worker_id is not None and instance.worker_id == worker_id:
        is_main_worker = True

    subordinate_worker_indexes = []
    subordinate_workers = (
        instance.distributed_servers.subordinate_workers
        if instance.distributed_servers
        and instance.distributed_servers.subordinate_workers
        else []
    )
    for index, subordinate_worker in enumerate(subordinate_workers):
        if worker_name is not None and subordinate_worker.worker_name == worker_name:
            subordinate_worker_indexes.append(index)
            continue
        if worker_id is not None and subordinate_worker.worker_id == worker_id:
            subordinate_worker_indexes.append(index)

    return ModelInstanceWorkerMatch(
        is_main_worker=is_main_worker,
        subordinate_worker_indexes=tuple(subordinate_worker_indexes),
        workload_indexes=tuple(
            index
            for index, item in enumerate(workloads or [])
            if item.worker_id == worker_id
        ),
    )


def get_model_instance_workloads(
    instance: ModelInstanceBase, worker_id: Optional[int] = None
) -> Optional[List[ModelInstanceWorkload]]:
    """Read the manifest without turning an empty managed list into legacy mode."""
    servers = instance.distributed_servers
    if (
        servers is None
        or getattr(
            servers,
            "orchestration_mode",
            "managed" if servers.workloads is not None else "legacy",
        )
        == "legacy"
    ):
        return None
    if worker_id is None:
        return servers.workloads
    return [item for item in servers.workloads if item.worker_id == worker_id]


def workload_topology(workloads: List[ModelInstanceWorkload]) -> Dict:
    return {
        item.name: (
            item.worker_id,
            (
                (
                    item.component.kind,
                    item.component.api_port,
                    item.component.metrics_port,
                    item.component.dp_size,
                    item.component.image,
                    item.component.router_version,
                    item.component.attempt,
                )
                if item.component
                else None
            ),
            {
                (placement.dp_rank, placement.node_rank): (
                    tuple(placement.gpu_indexes),
                    placement.api_port,
                )
                for placement in item.dp_rank_placements
            },
        )
        for item in workloads
    }


def requires_group_recovery(
    instance: ModelInstanceBase, model: Optional[Model] = None
) -> bool:
    """Identify fixed-topology groups whose members must initialize together."""
    if calculate_group_recovery_policy(instance, None) is None:
        return False
    if model_instance_legacy_stop(instance) is not None:
        return True
    if instance.distributed_servers.group_recovery is not None:
        return instance.distributed_servers.group_recovery
    config = getattr(instance, 'model_config_snapshot', None)
    identity = getattr(config, 'launch_identity', None)
    if identity is not None and (
        identity.instance_id,
        identity.worker_id,
        identity.restart_count,
    ) != (instance.id, instance.worker_id, instance.restart_count or 0):
        config = None
    if config is None and model is not None:
        config = role_effective_model(model, instance.role)
    return bool(calculate_group_recovery_policy(instance, config))


def calculate_group_recovery_policy(
    instance: ModelInstanceBase,
    config: Optional[Union[ModelBase, ModelConfigSnapshot]],
) -> Optional[bool]:
    """Calculate a launch policy without consulting a round's stored decision."""
    servers = instance.distributed_servers
    if (
        get_model_instance_workloads(instance) is not None
        or servers is None
        or servers.mode != DistributedServerCoordinateModeEnum.INITIALIZE_LATER
        or not servers.subordinate_workers
    ):
        return None
    backend = instance.backend
    if backend not in (BackendEnum.VLLM, BackendEnum.SGLANG, BackendEnum.ASCEND_MINDIE):
        return None
    parameters = config.backend_parameters if config is not None else []
    if backend == BackendEnum.VLLM:
        if any(
            find_bool_parameter(parameters, [flag], last_wins=True)
            for flag in ('enable-fault-tolerance', 'enable-elastic-ep')
        ):
            return False
        return find_parameter(parameters, ['distributed-executor-backend']) in (
            None,
            'mp',
            'ray',
        )
    return not (
        backend == BackendEnum.SGLANG
        and find_parameter(parameters, ['elastic-ep-backend'])
    )


def build_member_state_report(
    instance: ModelInstanceBase, worker_id: int, **fields
) -> ModelInstanceMemberReport:
    """Report explicit local runtime fields against the observed instance round."""
    for key in list(fields):
        if key.startswith('distributed_servers.subordinate_workers.'):
            subordinate = fields.pop(key)
            if subordinate.worker_id != worker_id:
                raise ValueError('A Worker may only report its own state')
            fields.update(
                {
                    name: getattr(subordinate, name)
                    for name in ('state', 'state_message', 'pid', 'ports', 'arguments')
                }
            )
    preparation = {
        key: fields.pop(key)
        for key in list(fields)
        if key in ModelInstancePreparation.model_fields
    }
    if preparation:
        fields['preparation'] = ModelInstancePreparation(**preparation)
    return ModelInstanceMemberReport(
        restart_count=instance.restart_count or 0, **fields
    )


def worker_workload_report(instance, worker_id, workloads, **fields):
    """Send topology only for registration; retain Server intent in its snapshot."""
    local = get_model_instance_workloads(instance, worker_id) or []
    if local and workload_topology(local) == workload_topology(workloads):
        return ModelInstanceMemberReport(
            restart_count=instance.restart_count or 0,
            observations=[
                ModelInstanceWorkloadObservation(
                    name=item.name,
                    state=item.state,
                    rank_readiness={
                        p.dp_rank: p.ready
                        for p in item.dp_rank_placements
                        if p.node_rank == 0
                    },
                    component_ready=item.component.ready if item.component else None,
                    component_attempt=(
                        item.component.attempt if item.component else None
                    ),
                )
                for item in workloads
            ],
            **fields,
        )
    payload = []
    for item in workloads:
        value = item.model_dump()
        if value['component'] is not None:
            for key in ('stop_requested', 'last_restart_time'):
                value['component'].pop(key)
        payload.append(value)
    return ModelInstanceMemberReport(
        restart_count=instance.restart_count or 0, workloads=payload, **fields
    )


def model_instance_legacy_stop(
    instance: ModelInstanceBase,
) -> Optional[ModelInstanceLegacyStop]:
    servers = getattr(instance, "distributed_servers", None)
    return getattr(servers, "legacy_stop", None)


def has_model_instance_stop_barrier(instance: ModelInstanceBase) -> bool:
    return (
        get_model_instance_workloads(instance) is not None
        or model_instance_legacy_stop(instance) is not None
    )


def requires_model_instance_stop_barrier(instance: ModelInstanceBase) -> bool:
    """Retain physical stop confirmation for PD groups and guarded legacy peers."""
    return bool(
        instance.group_id or instance.role or model_instance_legacy_stop(instance)
    ) and (has_model_instance_stop_barrier(instance))


def worker_supports_model_instance_workloads(worker: Worker) -> bool:
    """Require a bound Worker identity for managed workloads."""
    return bool(worker.system_principal_id)


def model_requires_managed_worker_binding(
    instances: Iterable[ModelInstanceBase], model_id: int
) -> bool:
    """Include legacy members when their model participates in managed stopping."""
    return any(
        instance.model_id == model_id and has_model_instance_stop_barrier(instance)
        for instance in instances
    )


def assigned_managed_workload_worker_ids(
    instances: Iterable[ModelInstanceBase],
) -> Set[int]:
    """Find assigned cohorts without treating unplaced pending rows as admission."""
    instances = list(instances)
    models = {
        instance.model_id
        for instance in instances
        if has_model_instance_stop_barrier(instance) and member_worker_ids(instance)
    }
    return {
        worker_id
        for instance in instances
        if instance.model_id in models
        for worker_id in member_worker_ids(instance)
    }


def model_instance_workloads_stopped(
    instance: ModelInstanceBase,
    *,
    retired_worker_ids: Optional[Set[int]] = None,
) -> bool:
    """Check stop reports, allowing explicitly retired Worker ownership."""
    retired_worker_ids = retired_worker_ids or set()
    guard = model_instance_legacy_stop(instance)
    if guard is not None:
        confirmed = set(guard.confirmed_worker_ids) | (
            retired_worker_ids & set(guard.worker_ids)
        )
        return (
            (instance.restart_count or 0) == guard.restart_count
            and set(member_worker_ids(instance)) == set(guard.worker_ids)
            and confirmed == set(guard.worker_ids)
        )
    workloads = get_model_instance_workloads(instance)
    return workloads is not None and all(
        item.state == ModelInstanceWorkloadStateEnum.STOPPED
        or item.worker_id in retired_worker_ids
        for item in workloads
    )


def managed_instance_update(
    instance: ModelInstance, **fields
) -> ModelInstanceMemberReport:
    """Build a narrow Worker report rather than rewriting an instance snapshot."""
    preparation = None
    if fields.get("state") == ModelInstanceStateEnum.INITIALIZING:
        fields.pop("state")
        preparation = ModelInstancePreparation(
            **{
                key: fields.pop(key)
                for key in list(fields)
                if key in ModelInstancePreparation.model_fields
            }
        )
    return ModelInstanceMemberReport(
        restart_count=instance.restart_count or 0,
        preparation=preparation,
        **fields,
    )


@dataclass(frozen=True)
class ModelInstanceTarget:
    model_id: int
    instance_id: int
    dp_rank: Optional[int] = None
    worker_id: Optional[int] = None


@dataclass(frozen=True)
class ModelInstanceEndpoint:
    model_id: int
    instance_id: int
    worker_id: int
    port: int
    restart_count: Optional[int] = None
    dp_rank: Optional[int] = None
    component: Optional[str] = None
    attempt: Optional[int] = None

    @property
    def registry_name(self) -> str:
        name = f"model-{self.model_id}-{self.instance_id}"
        if self.component == "vllm_hybrid":
            return name + f"-w{self.worker_id}"
        if self.dp_rank is not None:
            name += f"-dp{self.dp_rank}"
        return name


@dataclass(frozen=True)
class ModelInstanceMetricsSource:
    """One real exposition listener and its current task ownership."""

    instance_id: int
    model_id: int
    worker_id: int
    restart_count: int
    workload_name: str
    kind: str
    api_port: int
    metrics_port: int
    dp_rank: Optional[int] = None
    attempt: Optional[int] = None
    gpu_indexes: tuple[int, ...] = ()
    workload_state: Optional[str] = None
    ready: Optional[bool] = None
    stop_requested: bool = False
    dp_size: Optional[int] = None

    @property
    def name(self) -> str:
        if self.dp_rank is not None:
            return f"{self.workload_name}/main-{self.dp_rank}"
        return self.workload_name or self.kind


def model_instance_metrics_source_inventory(
    instance: ModelInstanceBase,
) -> List[ModelInstanceMetricsSource]:
    """Describe registered listeners, including tasks not currently scrape-ready."""
    if model_instance_legacy_stop(instance) is not None:
        return []
    workloads = get_model_instance_workloads(instance)
    if workloads is None and (not instance.worker_id or not instance.ports):
        return []
    common = dict(
        instance_id=instance.id,
        model_id=instance.model_id,
        restart_count=instance.restart_count or 0,
    )
    if workloads is None:
        band = (instance.named_ports or {}).get("prometheus")
        metrics_port = (
            band.get("base") if isinstance(band, dict) else getattr(band, "base", None)
        ) or instance.ports[0]
        return [
            ModelInstanceMetricsSource(
                **common,
                worker_id=instance.worker_id,
                workload_name="",
                kind="legacy",
                api_port=instance.port or instance.ports[0],
                metrics_port=metrics_port,
            )
        ]
    sources = []
    for task in workloads:
        component = task.component
        task_fields = dict(
            worker_id=task.worker_id,
            workload_name=task.name,
            workload_state=getattr(task.state, "value", task.state),
        )
        if component is not None:
            if component.api_port is None:
                continue
            sources.append(
                ModelInstanceMetricsSource(
                    **common,
                    **task_fields,
                    kind=component.kind,
                    api_port=component.api_port,
                    metrics_port=component.metrics_port or component.api_port,
                    attempt=(
                        component.attempt if component.kind == "sglang_smg" else None
                    ),
                    ready=component.ready,
                    stop_requested=component.stop_requested,
                    dp_size=component.dp_size,
                )
            )
            continue
        sources.extend(
            ModelInstanceMetricsSource(
                **common,
                **task_fields,
                kind="vllm_engine",
                api_port=placement.api_port,
                metrics_port=placement.api_port,
                dp_rank=placement.dp_rank,
                gpu_indexes=tuple(sorted(placement.gpu_indexes)),
                ready=placement.ready,
            )
            for placement in task.dp_rank_placements
            if placement.node_rank == 0 and placement.api_port
        )
    return sources


def model_instance_metrics_sources(
    instance: ModelInstanceBase,
) -> List[ModelInstanceMetricsSource]:
    """Read only current scrape-ready main APIs, independently of group health."""
    if (
        instance.state
        not in (
            ModelInstanceStateEnum.RUNNING,
            ModelInstanceStateEnum.STARTING,
            ModelInstanceStateEnum.ERROR,
        )
        or instance.draining_since is not None
    ):
        return []
    return [
        source
        for source in model_instance_metrics_source_inventory(instance)
        if (
            instance.state == ModelInstanceStateEnum.RUNNING
            if source.kind == "legacy"
            else source.workload_state == ModelInstanceWorkloadStateEnum.RUNNING
            and source.ready
            and not source.stop_requested
        )
    ]


@dataclass(frozen=True)
class ModelInstanceMetricsPlacement:
    source: ModelInstanceMetricsSource
    worker_id: int
    workload_name: str
    gpu_index: int
    node_rank: Optional[int] = None


def model_instance_metrics_placements(
    instance: ModelInstanceBase,
    sources: List[ModelInstanceMetricsSource],
    model: Optional[Model] = None,
) -> List[ModelInstanceMetricsPlacement]:
    """Associate allocated devices with real sources, never with token samples."""
    workloads = get_model_instance_workloads(instance)
    hybrid = {s.worker_id: s for s in sources if s.kind == "vllm_hybrid"}
    if hybrid:
        return _hybrid_metrics_placements(instance, workloads, hybrid)
    ranks = {s.dp_rank: s for s in sources if s.kind == "vllm_engine"}
    if ranks:
        return [
            ModelInstanceMetricsPlacement(
                ranks[p.dp_rank], task.worker_id, task.name, index, p.node_rank
            )
            for task in workloads
            for p in task.dp_rank_placements
            if p.dp_rank in ranks
            for index in sorted(set(p.gpu_indexes))
        ]
    shared = [s for s in sources if s.kind in ("legacy", "sglang_engine")]
    if len(shared) != 1:
        return []
    source = shared[0]
    allocations = {instance.worker_id: set(instance.gpu_indexes or [])}
    servers = instance.distributed_servers
    for worker in (servers.subordinate_workers or []) if servers else []:
        if worker.worker_id is not None:
            allocations.setdefault(worker.worker_id, set()).update(
                worker.gpu_indexes or []
            )
    task_names = {
        w.worker_id: w.name
        for w in workloads or []
        if w.component is not None and w.component.kind == "sglang_engine"
    }
    return [
        ModelInstanceMetricsPlacement(
            source, worker_id, task_names.get(worker_id, ""), index
        )
        for worker_id, indexes in allocations.items()
        if worker_id is not None
        for index in sorted(indexes)
    ]


def _hybrid_metrics_placements(instance, workloads, sources):
    """Reuse the launch topology to attach every device to its shared API."""
    from gpustack.utils.vllm_topology import (
        managed_lb_parallelism,
        external_lb_fragments,
        vllm_lb_mode,
    )

    from gpustack.schemas.models import ModelInstanceLaunchIdentity

    snapshot = instance.model_config_snapshot
    launch = ModelInstanceLaunchIdentity.from_instance(instance)
    if snapshot is None or launch is None or snapshot.launch_identity != launch:
        return []
    members = [instance] + list(instance.distributed_servers.subordinate_workers or [])
    counts = [len(member.gpu_indexes or []) for member in members]
    try:
        parameters = snapshot.backend_parameters
        if vllm_lb_mode(parameters) != "hybrid":
            return []
        topology = managed_lb_parallelism(parameters, counts)
    except ValueError:
        return []
    tasks = {task.worker_id: task.name for task in workloads or []}
    return [
        ModelInstanceMetricsPlacement(
            sources[members[fragment.main_worker_index].worker_id],
            member.worker_id,
            tasks.get(member.worker_id, ""),
            member.gpu_indexes[position],
            fragment.node_rank,
        )
        for worker_index, member in enumerate(members)
        for fragment in external_lb_fragments(counts, topology, worker_index)
        if members[fragment.main_worker_index].worker_id in sources
        for position in fragment.positions
    ]


def parse_model_instance_target(destination: str) -> ModelInstanceTarget:
    """Parse a registry identity, never a user-supplied IP or port."""
    match = re.fullmatch(
        r"model-(\d+)-(\d+)(?:-dp(\d+)|-w([1-9]\d*))?\.(?:static|dns)",
        destination,
    )
    if match is None:
        raise ValueError("Invalid model destination")
    return ModelInstanceTarget(
        *[int(v) if v is not None else None for v in match.groups()]
    )


def model_instance_endpoints(
    instance: ModelInstanceBase,
) -> List[ModelInstanceEndpoint]:
    """Publish a complete managed generation as one serving/recovery unit.

    External mode exposes multiple APIs, but its surviving tasks are stopped
    together when the generation fails. Withdraw all APIs during that recovery
    so requests cannot enter siblings that are about to stop.
    """
    if (
        instance.state != ModelInstanceStateEnum.RUNNING
        or instance.draining_since is not None
    ):
        return []
    workloads = get_model_instance_workloads(instance)
    if workloads is None:
        port = (instance.ports or [instance.port])[0]
        return (
            [
                ModelInstanceEndpoint(
                    instance.model_id, instance.id, instance.worker_id, port
                )
            ]
            if port and instance.worker_id
            else []
        )
    if workloads and workloads[0].component is not None:
        if workloads[0].component.kind == "vllm_hybrid":
            members = set(member_worker_ids(instance))
            if (
                len(workloads) != len(members)
                or {w.worker_id for w in workloads} != members
                or any(
                    w.component is None
                    or w.component.kind != "vllm_hybrid"
                    or w.state != ModelInstanceWorkloadStateEnum.RUNNING
                    for w in workloads
                )
            ):
                return []
            apis = [w for w in workloads if w.component.api_port is not None]
            if not apis or any(not w.component.ready for w in apis):
                return []
            return [
                ModelInstanceEndpoint(
                    instance.model_id,
                    instance.id,
                    w.worker_id,
                    w.component.api_port,
                    instance.restart_count or 0,
                    component="vllm_hybrid",
                )
                for w in apis
            ]
        # Engine failure retires the full generation; SMG-only recovery retains
        # the Engines but withholds the router until its complete set is ready.
        engines = _sglang_engine_workloads(instance)
        smgs = [
            w
            for w in workloads
            if w.component is not None and w.component.kind == "sglang_smg"
        ]
        if (
            not engines
            or len(smgs) != 1
            or smgs[0].worker_id != instance.worker_id
            or smgs[0].state != ModelInstanceWorkloadStateEnum.RUNNING
            or not smgs[0].component.ready
            or smgs[0].component.stop_requested
        ):
            return []
        smg = smgs[0].component
        return [
            ModelInstanceEndpoint(
                instance.model_id,
                instance.id,
                instance.worker_id,
                smg.api_port,
                instance.restart_count or 0,
                component="sglang_smg",
                attempt=smg.attempt,
            )
        ]
    if (
        not workloads
        or {w.worker_id for w in workloads} != set(member_worker_ids(instance))
        or any(w.state != ModelInstanceWorkloadStateEnum.RUNNING for w in workloads)
    ):
        return []
    mains = [
        (w.worker_id, p)
        for w in workloads
        for p in w.dp_rank_placements
        if p.node_rank == 0
    ]
    ranks = {p.dp_rank for w in workloads for p in w.dp_rank_placements}
    if (
        {p.dp_rank for _, p in mains} != ranks
        or len(mains) != len(ranks)
        or not all(p.ready for _, p in mains)
    ):
        return []
    return [
        ModelInstanceEndpoint(
            instance.model_id,
            instance.id,
            worker_id,
            p.api_port,
            instance.restart_count or 0,
            p.dp_rank,
        )
        for worker_id, p in sorted(mains, key=lambda item: item[1].dp_rank)
    ]


def resolve_model_instance_endpoint(
    instance: ModelInstanceBase,
    target: ModelInstanceTarget,
    *,
    worker_id: Optional[int] = None,
) -> ModelInstanceEndpoint:
    """Resolve a stable identity against the currently available endpoints."""
    if instance.id != target.instance_id or instance.model_id != target.model_id:
        raise ValueError("Model destination identity mismatch")
    endpoints = model_instance_endpoints(instance)
    if target.dp_rank is None and target.worker_id is None:
        endpoints = endpoints if len(endpoints) == 1 else []
    elif target.worker_id is not None:
        endpoints = [
            endpoint
            for endpoint in endpoints
            if endpoint.component == "vllm_hybrid"
            and endpoint.worker_id == target.worker_id
        ]
    else:
        endpoints = [
            endpoint for endpoint in endpoints if endpoint.dp_rank == target.dp_rank
        ]
    for endpoint in endpoints:
        if worker_id is None or endpoint.worker_id == worker_id:
            return endpoint
    raise ValueError(
        "Model destination is unavailable, ambiguous or belongs to another Worker"
    )


SGLANG_SMG_POLICIES = frozenset(
    ('random', 'round_robin', 'cache_aware', 'power_of_two', 'manual')
)


def sglang_parallelism(parameters: Optional[List[str]]) -> tuple[int, int, bool]:
    """Read routing-rank count, EP size and DPA intent from native options."""
    from gpustack.utils.command import flatten_to_argv

    aliases = {
        '--data-parallel-size': 'dp',
        '--dp-size': 'dp',
        '--dp': 'dp',
        '--attn-dp-size': 'attention_dp',
        '--attention-data-parallel-size': 'attention_dp',
        '--expert-parallel-size': 'ep',
        '--ep-size': 'ep',
        '--ep': 'ep',
    }
    values = {'dp': 1, 'attention_dp': 1, 'ep': 1}
    attention = False
    tokens = flatten_to_argv(parameters)
    index = 0
    while index < len(tokens):
        key, separator, value = tokens[index].partition('=')
        if key in aliases:
            if not separator:
                index += 1
                value = tokens[index] if index < len(tokens) else ''
            try:
                number = int(value)
            except ValueError as exc:
                raise ValueError(f'{key} requires a positive integer') from exc
            if number < 1:
                raise ValueError(f'{key} requires a positive integer')
            values[aliases[key]] = number
        elif key == '--enable-dp-attention':
            if separator:
                raise ValueError('--enable-dp-attention is a flag without a value')
            attention = True
        index += 1
    return (
        values['dp'] * values['attention_dp'],
        values['ep'],
        attention or values['attention_dp'] > 1,
    )


def validate_sglang_dpa_version(model: Model) -> None:
    """Check new DPA syntax against a known Runner-selected Engine version."""
    from packaging.version import InvalidVersion, Version
    from gpustack.schemas.models import BackendEnum, get_backend
    from gpustack.utils.command import find_last_parameter

    if (
        get_backend(model) != BackendEnum.SGLANG
        or model.image_name
        or not model.backend_version
        or find_last_parameter(
            model.backend_parameters, ['attn-dp-size', 'attention-data-parallel-size']
        )
        is None
    ):
        return
    try:
        version = Version(model.backend_version)
    except InvalidVersion:
        return
    if version < Version('0.5.22'):
        raise ValueError(
            '--attn-dp-size and --attention-data-parallel-size require SGLang '
            '0.5.22 or later; use --dp-size N --enable-dp-attention on older Runners.'
        )


def sglang_smg_mode(model) -> bool:
    """Use a DP-aware SMG only when the Engine has multiple routing ranks."""
    from gpustack.schemas.models import BackendEnum, CategoryEnum, get_backend

    if (
        get_backend(model) != BackendEnum.SGLANG
        or model.disaggregation is not None
        or CategoryEnum.IMAGE in (model.categories or [])
    ):
        return False
    dp, _, _ = sglang_parallelism(model.backend_parameters)
    return dp > 1


def sglang_policy_parameters(
    parameters: Optional[List[str]],
    *,
    tokenized: bool = False,
) -> tuple[List[str], str, bool]:
    """Separate native policy options without mutating the input.

    Set tokenized for launch argv; saved parameters still need shell parsing.
    """
    from gpustack.utils.command import flatten_to_argv

    tokens = list(parameters or []) if tokenized else flatten_to_argv(parameters)
    engine, policy, found = [], 'cache_aware', False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        key, separator, value = token.partition('=')
        if key != '--policy':
            engine.append(token)
            index += 1
            continue
        found = True
        if not separator:
            index += 1
            value = tokens[index] if index < len(tokens) else ''
        if not value or value.startswith('-'):
            raise ValueError('--policy requires a nonempty value')
        if value not in SGLANG_SMG_POLICIES:
            raise ValueError(f'Unsupported SGLang SMG policy: {value}')
        policy = value
        index += 1
    return engine, policy, found


def validate_sglang_smg_policy(model) -> None:
    """Reject a policy without an ordinary paired SMG; PD retains its own argv."""
    from gpustack.schemas.models import BackendEnum, get_backend

    if get_backend(model) != BackendEnum.SGLANG or model.disaggregation is not None:
        return
    _, _, found = sglang_policy_parameters(model.backend_parameters)
    if found and not sglang_smg_mode(model):
        raise ValueError(
            '--policy requires an ordinary multi-rank SGLang DP/DPA SMG component'
        )


def _sglang_engine_workloads(
    instance: ModelInstanceBase,
) -> List[ModelInstanceWorkload]:
    """Require all allocated Engine tasks, including their shared main API."""
    workloads = get_model_instance_workloads(instance) or []
    if not workloads or any(w.component is None for w in workloads):
        return []
    engines = [w for w in workloads if w.component.kind == "sglang_engine"]
    mains = [w for w in engines if w.component.api_port is not None]
    if (
        len(engines) != len(member_worker_ids(instance))
        or {w.worker_id for w in engines} != set(member_worker_ids(instance))
        or len(mains) != 1
        or mains[0].worker_id != instance.worker_id
        or not mains[0].component.ready
        or any(w.state != ModelInstanceWorkloadStateEnum.RUNNING for w in engines)
    ):
        return []
    return engines
