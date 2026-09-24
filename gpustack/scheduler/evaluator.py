import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Dict

from gpustack_runtime.detector import ManufacturerEnum
from sqlmodel.ext.asyncio.session import AsyncSession
from cachetools import TTLCache
from aiolimiter import AsyncLimiter

from gpustack.api.exceptions import HTTPException
from gpustack.client.worker_filesystem_client import WorkerFilesystemClient
from gpustack.config.config import Config
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack import envs
from gpustack.routes.models import validate_model_in
from gpustack.scheduler import scheduler
from gpustack.scheduler.calculator import get_pretrained_config_with_workers
from gpustack.server.catalog import get_catalog_spec_by_source_key
from gpustack.server.cache_provider_catalog import get_cache_provider
from gpustack.schemas.cache_services import CacheService
from gpustack.schemas.clusters import Cluster
from gpustack.policies.base import MemberResourceClaim
from gpustack.schemas.model_evaluations import (
    ModelEvaluationResult,
    ModelSpec,
    ResourceClaim,
    RoleResourceClaim,
    RoleResourceDemand,
)
from gpustack.schemas.models import (
    ModelInstance,
    BackendEnum,
    SourceEnum,
    get_backend,
    is_gguf_model,
    is_audio_model,
    role_container_resources,
    role_takes_no_accelerator,
)
from gpustack.schemas.principals import _platform_principal_id
from gpustack.scheduler.group_capacity import (
    GroupCapacity,
    attendant_demands,
    role_demands,
)
from gpustack.scheduler.group_schedule import (
    cache_instances_in,
    gather_request,
    stand_in,
)
from gpustack.scheduler.group_solver import (
    GroupPlacement,
    RoleDemand,
    solve_group_placement,
)
from gpustack.topology.tree import TopologyError
from gpustack.topology.view import build_view
from gpustack.schemas.workers import GPUDeviceStatus, Worker, WorkerStateEnum
from gpustack.server.worker_selector import WorkerSelector

from gpustack.utils.gpu import (
    all_gpu_match,
    any_gpu_match,
    find_one_gpu,
    make_gpu_id,
    compare_compute_capability,
)
from gpustack.utils.command import flatten_to_argv
from gpustack.utils.hybrid_attention import (
    HYBRID_MODEL_DOC_URL,
    is_hybrid_attention,
)
from gpustack.utils.vllm_kv_cache import ascend_local_kv_cache_unsupported_reason
from gpustack.utils.hub import (
    auth_check,
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
    is_repo_cached,
)
from gpustack.utils.task import run_in_thread
from gpustack.utils.profiling import time_decorator

logger = logging.getLogger(__name__)

evaluate_cache = TTLCache(
    maxsize=envs.MODEL_EVALUATION_CACHE_MAX_SIZE, ttl=envs.MODEL_EVALUATION_CACHE_TTL
)

# To reduce the likelihood of hitting the Hugging Face API rate limit (600 RPM)
# Limit the number of concurrent evaluations to 50 per 10 seconds
evaluate_model_limiter = AsyncLimiter(50, 10)

LMCACHE_PROVIDER_NAME = "LMCache"
"""The cache provider whose connector constrains hybrid models: vLLM
raises the attention block size until an attention page holds a whole
recurrent-state page, and the cache server registers a chunk only when
its own chunk size is a multiple of that block size."""

LMCACHE_HYBRID_SERVICE_FIELD = "chunk_size"
LMCACHE_HYBRID_SERVICE_PARAMETER = "--separate-object-groups"
"""What an LMCache service needs before a hybrid model can attach to it.
Neither is a default, so a service created and left alone fails the
engine's chunk/block assertion at startup."""


def _doc_link(text: str = "documentation") -> str:
    """The hybrid-model procedure as a link a message can carry.

    An anchor rather than a bare URL, the way worker state messages carry
    theirs: the address is long enough to be cut off where a message is
    rendered, and it is the words around it the reader acts on.
    """
    return f"<a href='{HYBRID_MODEL_DOC_URL}'>{text}</a>"


LMCACHE_HYBRID_SERVICE_NOTE = (
    "Set the cache service's 'Chunk Size' to a multiple of the model's vLLM "
    f"attention block size, and add '{LMCACHE_HYBRID_SERVICE_PARAMETER}' to "
    "the cache server's parameters."
)
"""What to do about it, quoted the way the message quotes the service's
name: what the reader types or picks out of a form is a literal."""


@time_decorator
async def evaluate_models(
    config: Config,
    session: AsyncSession,
    model_specs: List[ModelSpec],
    cluster_id: Optional[int] = None,
) -> List[ModelEvaluationResult]:
    """
    Evaluate the compatibility of a list of model specs with the available workers.
    """
    fields = {
        "deleted_at": None,
    }
    if cluster_id is not None:
        fields["cluster_id"] = cluster_id
    extra_conditions = [
        ~(
            Worker.state.in_(
                [
                    WorkerStateEnum.PROVISIONING,
                    WorkerStateEnum.DELETING,
                    WorkerStateEnum.ERROR,
                ]
            )
        )
    ]
    workers = await Worker.all_by_fields(
        session, fields=fields, extra_conditions=extra_conditions
    )

    model_instances = await ModelInstance.all_by_fields(session, fields=fields)

    if len(model_specs) == 1:
        # Sort worker for single-model evaluation only. No need for batch evaluation.
        workers = await scheduler.prioritize_workers_with_model_files(
            session, model_specs[0], workers
        )

    async def evaluate(model: ModelSpec):
        return await evaluate_model_with_cache(
            config,
            session,
            model,
            workers,
            model_instances,
            cluster_id=cluster_id,
        )

    tasks = [evaluate(model) for model in model_specs]
    results = await asyncio.gather(*tasks)
    return results


def make_hashable_key(
    model: ModelSpec, workers: List[Worker], extra: Optional[str] = None
) -> str:
    key_data = json.dumps(
        {
            "model": model.model_dump(mode="json"),
            # State outside the spec and the workers that a verdict depends
            # on, rendered by the caller.
            "extra": extra,
            # Excluded from model_dump (response-hidden field), but it
            # changes which Org-scoped backend versions the evaluation
            # sees — without it cached results would leak across Orgs.
            "owner_principal_id": getattr(model, "owner_principal_id", None),
            "workers": [
                w.model_dump(
                    mode="json",
                    exclude={
                        "status": {
                            "cpu": True,
                            "swap": True,
                            "filesystem": True,
                            "os": True,
                            "kernel": True,
                            "uptime": True,
                            "memory": {"utilization_rate", "used"},
                            "gpu_devices": {
                                "__all__": {
                                    "temperature": True,
                                    "core": {"utilization_rate"},
                                    "memory": {"utilization_rate", "used"},
                                },
                            },
                        },
                        "heartbeat_time": True,
                        "created_at": True,
                        "updated_at": True,
                    },
                )
                for w in workers
            ],
        },
        sort_keys=True,
    )
    return hashlib.md5(key_data.encode()).hexdigest()


async def visible_cache_service(
    session: AsyncSession, model: ModelSpec
) -> Optional[CacheService]:
    """The cache service a spec attaches to, as the spec's Org may see it.

    The Org is a condition on the query rather than a test on the row, so
    another Org's service is never loaded at all: evaluation runs no
    shared-cache validation of its own, and what is built from this row
    carries the service's name back to the caller. A spec with no owner
    resolved is the platform admin's, and matches by id alone.

    Returns:
        The service, or None when the spec attaches to none, when it is
        deleted, or when it belongs to another Org.
    """
    ext = getattr(model, "extended_kv_cache", None)
    if not (ext and ext.is_shared() and ext.cache_service_id):
        return None

    fields: Dict[str, Any] = {"id": ext.cache_service_id, "deleted_at": None}
    owner_principal_id = getattr(model, "owner_principal_id", None)
    if owner_principal_id is not None:
        fields["owner_principal_id"] = owner_principal_id
    return await CacheService.one_by_fields(session, fields)


def cache_service_verdict_key(service: Optional[CacheService]) -> Optional[str]:
    """Everything a verdict about the attached cache service reads, for the
    evaluation cache key.

    The model spec carries none of it, so without this a user who acts on a
    verdict — configures the service, renames it, moves it to another
    provider — gets the stale one back until the entry expires. The one
    case where the cache would answer a question the user has just changed
    the answer to.
    """
    if service is None:
        return None
    return json.dumps(
        {
            # Carried in the message the verdict produces.
            "name": service.name,
            # Decides whether the service is one whose connector constrains
            # hybrid models at all.
            "provider_name": service.provider_name,
            # The settings the verdict is about.
            "config": (
                service.config.model_dump(mode="json") if service.config else None
            ),
        },
        sort_keys=True,
    )


async def evaluate_model_with_cache(
    config: Config,
    session: AsyncSession,
    model: ModelSpec,
    workers: List[Worker],
    model_instances: List[ModelInstance],
    cluster_id: Optional[int] = None,
) -> ModelEvaluationResult:
    # Everything that can fail belongs inside: specs are evaluated
    # concurrently over one session, so a query raising here would take the
    # whole request down instead of reporting the one spec it belongs to.
    try:
        # Fetched once: it keys the cache, and the evaluation reads it again.
        cache_service = await visible_cache_service(session, model)
        cache_key = make_hashable_key(
            model, workers, cache_service_verdict_key(cache_service)
        )
        if cache_key in evaluate_cache:
            logger.trace(
                f"Evaluation cache hit for model: {model.name or model.readable_source}"
            )
            return evaluate_cache[cache_key]

        async with evaluate_model_limiter:
            result = await evaluate_model(
                config,
                session,
                model,
                workers,
                model_instances,
                cluster_id=cluster_id,
                cache_service=cache_service,
            )
            evaluate_cache[cache_key] = result
    except Exception as e:
        logger.exception(
            f"Error evaluating model {model.name or model.readable_source}: {e}"
        )
        result = ModelEvaluationResult(
            compatible=False, error=True, error_message=str(e)
        )

    return result


@time_decorator
async def evaluate_model(
    config: Config,
    session: AsyncSession,
    model: ModelSpec,
    workers: List[Worker],
    model_instances: List[ModelInstance],
    cluster_id: Optional[int] = None,
    cache_service: Optional[CacheService] = None,
) -> ModelEvaluationResult:
    result = ModelEvaluationResult()

    if await set_default_spec(session, model):
        result.default_spec = model.model_copy()

    await set_gguf_model_file_path(config, model)

    evaluations = [
        (evaluate_model_input, (session, model, cluster_id)),
        (evaluate_model_metadata, (config, model, workers)),
        (evaluate_environment, (model, workers)),
        # Last: it reads the pretrained config, which the metadata step
        # ahead of it has already proven readable.
        (evaluate_hybrid_model_kv_cache, (session, model, workers, cache_service)),
    ]
    for evaluation, args in evaluations:
        compatible, messages = await evaluation(*args)
        if not compatible:
            result.compatible = False
            result.compatibility_messages = messages
            return result

    workers_by_cluster: Dict[int, List[Worker]] = defaultdict(list)
    for worker in workers:
        workers_by_cluster[worker.cluster_id].append(worker)

    overcommit_clusters = []
    result.resource_claim_by_cluster_id = {}
    role_claims_by_cluster_id: Dict[int, List[RoleResourceClaim]] = {}
    role_demands_by_cluster_id: Dict[int, List[RoleResourceDemand]] = {}

    for cluster_id, cluster_workers in workers_by_cluster.items():
        cluster_model_instances = [
            inst for inst in model_instances if inst.cluster_id == cluster_id
        ]

        # A role-bearing deployment is a group, and a group is placed all at
        # once. Asking `find_candidate` instead answers a question the
        # deployment never asks -- "where would ONE instance of the model-level
        # spec go" -- and its answer is wrong in three directions at the same
        # time: it ignores the role overrides (a decode's tensor parallelism is
        # not the model's), it counts one member where the group has x+y+1, and
        # it never checks that prefill and decode fit *together*.
        if model.roles:
            group = await evaluate_group(
                config,
                session,
                model,
                cluster_workers,
                cluster_model_instances,
                cluster_id,
            )
            if group.total is None:
                result.scheduling_messages.extend(group.messages)
                if group.demands:
                    role_demands_by_cluster_id[cluster_id] = group.demands
                continue
            result.resource_claim_by_cluster_id[cluster_id] = group.total
            role_claims_by_cluster_id[cluster_id] = group.claims
            continue

        candidate, schedule_messages = await scheduler.find_candidate(
            session, config, model, cluster_workers, cluster_model_instances
        )
        if not candidate:
            result.scheduling_messages.extend(schedule_messages)
            continue
        if candidate.overcommit:
            overcommit_clusters.append(cluster_id)
            result.scheduling_messages.extend(schedule_messages)
            continue
        result.resource_claim_by_cluster_id[cluster_id] = (
            summarize_candidate_resource_claim(candidate)
        )

    if result.resource_claim_by_cluster_id:
        first_cluster_id = next(iter(result.resource_claim_by_cluster_id))
        result.resource_claim = result.resource_claim_by_cluster_id[first_cluster_id]
        if role_claims_by_cluster_id:
            # Kept in step with `resource_claim` above rather than derived
            # separately: the two describe the same cluster's placement, and a
            # reader that took the total from one cluster and the breakdown
            # from another would show a breakdown that does not add up.
            result.role_resource_claims_by_cluster_id = role_claims_by_cluster_id
            result.role_resource_claims = role_claims_by_cluster_id.get(
                first_cluster_id
            )
    else:
        result.resource_claim = None
        result.compatible = False
        result.compatibility_messages.append(
            "Unable to find a schedulable worker for the model."
        )
        if role_demands_by_cluster_id:
            # Only when nothing fit anywhere. A cluster that refused the group
            # while another accepted it is not the deployment's problem, and a
            # breakdown of the refusing one beside a claim for the accepting
            # one would be two answers to one question.
            result.role_resource_demands_by_cluster_id = role_demands_by_cluster_id
    return result


@dataclass
class GroupEvaluation:
    """What one cluster's evaluation of a role-bearing deployment found.

    ``total`` and ``claims`` are filled when the group fits, and are read off
    the placement that was proved. ``demands`` is filled when it does not: the
    same breakdown, priced by each role's own selector before anything is
    placed, so a refusal describes the group in the units an approval does
    rather than only counting the members it is short of.
    """

    total: Optional[ResourceClaim] = None
    claims: List[RoleResourceClaim] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    demands: List[RoleResourceDemand] = field(default_factory=list)


async def evaluate_group(
    config: Config,
    session: AsyncSession,
    model: ModelSpec,
    workers: List[Worker],
    model_instances: List[ModelInstance],
    cluster_id: int,
) -> GroupEvaluation:
    """What the whole group would claim in this cluster, or why it cannot land.

    A ``None`` total means the group does not fit, and ``messages`` then
    carries the solver's own refusal -- which names the shortfall, the role the
    walk stopped on and the roomiest domain -- rather than a generic "no
    suitable worker", with ``demands`` carrying what each role asked for.

    **The scheduler's own solver, asked the feasibility half of its question.**
    An answer computed a different way can be right and still disagree with
    what happens at deploy time, and the one thing an evaluation must not do is
    promise a placement the scheduler then refuses. So this goes through
    `solve_group_placement` over the cluster's own topology, with the group's
    gather requirement applied, and reads the claim off the candidates
    `GroupCapacity.commit` produces -- the very objects the group scheduler
    writes onto instance rows.

    What it does not pass is the ranking the scheduler adds: no warm-worker
    preference, no locality score, no candidate limit. Those choose *among* the
    domains that fit, and this answers whether any does. So the domain proved
    here can differ from the one a deployment then lands in -- which costs
    nothing, because the result carries resource claims and refusals, never a
    location.

    **The router is priced here and checked, but still not placed.** It
    stays out of `role_demands` -- counting it among the gang would make a 4P4D
    need nine placements in one domain -- so this function adds its container
    memory to the total afterwards. What changed is that it now travels as an
    `attendant`: the solver verifies a worker can host it before calling the
    group placeable. Until then a cluster with room for the GPU members and
    none for the router evaluated as compatible, in both places for the same
    reason, and the deployment it promised then sat with a router that could
    not be scheduled -- which is a group that serves nothing, since a router
    answers every request.
    """
    # A `ModelSpec` is not quite a `Model`, and the role projection
    # revalidates it through `ModelBase` — where two of these fields are not
    # optional. Left as they arrive, `role_effective_model` raises and the
    # whole evaluation reports "invalid role specification" for a spec that is
    # perfectly valid:
    #
    # - `name` is optional on a spec, because the deployment form evaluates
    #   while it is still being filled in;
    # - `owner_principal_id` is stamped by the route from the caller's context
    #   and is legitimately None there — that is the Platform-only view (an
    #   admin in "All" mode) — while `ModelBase` declares it a plain int.
    #
    # `cluster_id` is stamped for a different reason: `ClusterFilter` reads it,
    # and this call is already scoped to the cluster whose workers we hold.
    group_model = model.model_copy(
        update={
            "cluster_id": cluster_id,
            "name": model.name or "model-evaluation",
            "owner_principal_id": (
                model.owner_principal_id or _platform_principal_id()
            ),
        }
    )

    # Ordered like the deployment declares them, so the breakdown reads
    # prefill, decode, router rather than in whatever order the solver found
    # convenient (it sorts by weight) or a dict happened to keep.
    order = {spec.name: index for index, spec in enumerate(group_model.roles or [])}

    try:
        demands = [RoleDemand(**d) for d in role_demands(group_model)]
    except Exception as e:
        return GroupEvaluation(messages=[f"Invalid role specification: {e}"])

    free_claims = _accelerator_free_claims(group_model)
    if not demands:
        # Nothing in this group occupies an accelerator. There is no placement
        # to solve, but the group still has an honest footprint -- a router's
        # container memory -- and reporting it beats reporting nothing.
        if not free_claims:
            return GroupEvaluation(
                messages=["The group has no member that occupies an accelerator."]
            )
        return GroupEvaluation(total=_total_claim(free_claims), claims=free_claims)

    cluster = await Cluster.one_by_id(session, cluster_id) if cluster_id else None
    try:
        view = build_view(cluster.topology if cluster else None, workers)
    except TopologyError as e:
        return GroupEvaluation(messages=[f"Cluster topology is invalid: {e}"])

    cache_instances = await cache_instances_in(session, cluster_id)
    capacity = GroupCapacity(
        config, group_model, workers, model_instances, cache_instances
    )
    placement = await solve_group_placement(
        view.root,
        demands,
        capacity,
        view.scopes(),
        gather_request(group_model),
        attendants=[RoleDemand(**d) for d in attendant_demands(group_model)],
    )
    if not isinstance(placement, GroupPlacement):
        reason = getattr(placement, "reason", "The group does not fit.")
        blocked = getattr(placement, "role", None)
        notes = capacity.notes_for(blocked)
        demands = await _role_asks(group_model, capacity, view, order)
        # Two answers to one question, and they do not agree. `reason` counts
        # the whole group -- "needs 2 placements and the cluster has room for
        # 0" -- while `demands` breaks the same refusal down role by role, and
        # the two are not measured the same way: `needed` leaves out the roles
        # that take no accelerator, `available` is the running total at the
        # point the solve gave up rather than the cluster's free room, and each
        # role's own figure is that role measured alone against everything
        # free. Put side by side they read as arithmetic that does not add up.
        # The breakdown is the better answer and says more, so it is the one
        # sent -- but only when there is something left to say afterwards.
        # `reason` stays whenever nothing else would name what went wrong,
        # because an empty message list is read downstream as "no explanation"
        # and takes the breakdown down with it.
        messages = [] if (demands and blocked) else [reason]
        if blocked:
            # The notes below are one role's and they name no role themselves:
            # they come from the selectors, which are shared with the
            # single-instance path and speak of "the model". In a group of
            # three roles that reads as the whole deployment's footprint --
            # the one number it is not -- and the figure they quote is what
            # the model's *weights* want, not what a member reserves. So this
            # line has to scope them on both counts.
            messages.append(
                f"Blocked on the '{blocked}' role; the lines below describe "
                f"one of its members, and the VRAM they quote is what the "
                f"model's weights need rather than what the member reserves."
            )
        return GroupEvaluation(
            messages=messages + notes,
            demands=demands,
        )

    # `already` accumulates across roles for the same reason the capacity count
    # does: the second role has to see what the first one took, or both are
    # priced against the same free cards.
    already: List[object] = []
    claims: List[RoleResourceClaim] = []
    for role, worker_ids in placement.assignments.items():
        candidates = await capacity.commit(role, worker_ids, already)
        if len(candidates) != len(worker_ids):
            return GroupEvaluation(
                messages=[
                    "The group's placement could not be turned into GPU "
                    f"assignments for role '{role}'."
                ]
            )
        claims.append(
            _role_claim(
                role, [summarize_candidate_resource_claim(c) for c in candidates]
            )
        )
        already.extend(stand_in(c) for c in candidates)

    claims.extend(free_claims)
    claims.sort(key=lambda claim: order.get(claim.role, len(order)))
    return GroupEvaluation(total=_total_claim(claims), claims=claims)


async def _role_asks(
    model,
    capacity: GroupCapacity,
    view,
    order: Dict[str, int],
) -> List[RoleResourceDemand]:
    """What every role of a refused group asked for, and how much of it fits.

    The refusal's counterpart to the breakdown an approval shows. Each role is
    priced by its own selector and measured on its own in this cluster, which
    is the one question still answerable once the solve has failed -- the
    placement that would have carried the real claims is precisely what does
    not exist.

    Best-effort by design: a breakdown is an explanation, and losing it must
    never cost the refusal itself, which is what the deployment form acts on.

    Runs only after the solve has been abandoned. `demand_for` re-measures
    against an empty placement set, so a `commit` afterwards would be handing
    out cards priced for a different solve.
    """
    worker_ids = view.root.descendant_worker_ids()
    asks: List[RoleResourceDemand] = []
    try:
        for spec in model.roles or []:
            replicas = max(int(spec.replicas or 0), 0)
            claim, placeable = await capacity.demand_for(spec.name, worker_ids)
            if claim is None and role_takes_no_accelerator(model, spec.name):
                # No worker was eligible, so no selector ever ran and priced it
                # -- but an accelerator-free role is a declared floor rather
                # than a computed size, so the declaration still answers. Same
                # source the selector would have read.
                memory = role_container_resources(model, spec.name).memory or 0
                claim = MemberResourceClaim(vram=0, ram=memory)
            asks.append(
                RoleResourceDemand(
                    role=spec.name,
                    replicas=replicas,
                    placeable=placeable,
                    ram=(claim.ram if claim else 0) * replicas,
                    vram=(claim.vram if claim else 0) * replicas,
                    per_replica=(
                        ResourceClaim(ram=claim.ram, vram=claim.vram) if claim else None
                    ),
                )
            )
    except Exception as e:
        logger.warning(
            "Could not break down what the group's roles asked for; the "
            "refusal stands without it: %s",
            e,
        )
        return []
    asks.sort(key=lambda ask: order.get(ask.role, len(order)))
    return asks


def _accelerator_free_claims(model) -> List[RoleResourceClaim]:
    """The roles the solver does not place, priced from what they declare.

    Today that is the router alone, and its claim is a declared floor rather
    than an estimate -- it holds no weights, so there is nothing to size.
    """
    claims: List[RoleResourceClaim] = []
    for spec in model.roles or []:
        if not role_takes_no_accelerator(model, spec.name):
            continue
        replicas = max(int(spec.replicas or 0), 0)
        ram = role_container_resources(model, spec.name).memory or 0
        claims.append(
            RoleResourceClaim(
                role=spec.name,
                replicas=replicas,
                ram=ram * replicas,
                vram=0,
                per_replica=ResourceClaim(ram=ram, vram=0),
            )
        )
    return claims


def _role_claim(role: str, per_member: List[ResourceClaim]) -> RoleResourceClaim:
    first = per_member[0] if per_member else None
    uniform = first is not None and all(
        claim.ram == first.ram and claim.vram == first.vram for claim in per_member
    )
    return RoleResourceClaim(
        role=role,
        replicas=len(per_member),
        ram=sum(claim.ram for claim in per_member),
        vram=sum(claim.vram for claim in per_member),
        # None rather than the first member's numbers when the members
        # disagree: a heterogeneous role has no "per replica" figure, and
        # showing one member's as if it were every member's is how a 2P4D on
        # mixed cards would read as half its real size.
        per_replica=first if uniform else None,
    )


def _total_claim(claims: List[RoleResourceClaim]) -> ResourceClaim:
    return ResourceClaim(
        ram=sum(claim.ram for claim in claims),
        vram=sum(claim.vram for claim in claims),
    )


def summarize_candidate_resource_claim(
    candidate: ModelInstanceScheduleCandidate,
) -> ResourceClaim:
    """
    Summarize the computed resource claim for a schedule candidate.
    """
    computed_resource_claims = [candidate.computed_resource_claim]

    if candidate.subordinate_workers:
        computed_resource_claims.extend(
            sw.computed_resource_claim
            for sw in candidate.subordinate_workers
            if sw.computed_resource_claim is not None
        )

    ram, vram = 0, 0
    for computed_resource_claim in computed_resource_claims:
        ram += computed_resource_claim.ram or 0
        if computed_resource_claim.vram:
            vram += sum(
                v for v in computed_resource_claim.vram.values() if v is not None
            )

    return ResourceClaim(ram=ram, vram=vram)


async def set_gguf_model_file_path(config: Config, model: ModelSpec):
    if (
        model.source == SourceEnum.HUGGING_FACE
        and "gguf" in model.huggingface_repo_id.lower()
        and not model.huggingface_filename
    ):
        model.huggingface_filename = await run_in_thread(
            get_hugging_face_model_min_gguf_path,
            timeout=15,
            model_id=model.huggingface_repo_id,
            token=config.huggingface_token,
        )
    elif (
        model.source == SourceEnum.MODEL_SCOPE
        and "gguf" in model.model_scope_model_id.lower()
        and not model.model_scope_file_path
    ):
        model.model_scope_file_path = await run_in_thread(
            get_model_scope_model_min_gguf_path,
            timeout=15,
            model_id=model.model_scope_model_id,
        )


async def evaluate_environment(
    model: ModelSpec,
    workers: List[Worker],
) -> Tuple[bool, List[str]]:
    backend = get_backend(model)

    if backend == BackendEnum.ASCEND_MINDIE and not any_gpu_match(
        workers, lambda gpu: gpu.vendor == ManufacturerEnum.ASCEND.value
    ):
        return False, [
            "The Ascend MindIE backend requires Ascend NPUs but none are available."
        ]

    if (
        backend == BackendEnum.SGLANG
        and all_gpu_match(
            workers, lambda gpu: gpu.vendor == ManufacturerEnum.NVIDIA.value
        )
        and not any_gpu_match(
            workers,
            lambda gpu: compare_compute_capability(gpu.compute_capability, "8.0") >= 0,
        )
    ):
        # Ref: https://github.com/sgl-project/sglang/issues/6006
        gpu = find_one_gpu(workers)
        return False, [
            "The SGLang backend requires NVIDIA GPUs with compute capability 8.0 or higher "
            "(e.g., A100/SM80, H100/SM90, RTX 3090/SM86). "
            + (
                f"Available GPU: {gpu.name} (compute capability: {gpu.compute_capability})"
                if gpu
                else ""
            )
        ]

    if backend == BackendEnum.VLLM:
        message = evaluate_local_extended_kv_cache(model, workers)
        if message:
            return False, [message]

    return True, []


def candidate_gpus(model: ModelSpec, workers: List[Worker]) -> List[GPUDeviceStatus]:
    """The GPUs a deployment could land on.

    A manual GPU selection narrows this to the picked devices, so a
    deployment pinned to one accelerator is not judged by another one
    elsewhere in the cluster. The label / GPU-type / backend-framework
    filters are not replayed here — they run in ``find_candidate``, further
    down the evaluation.
    """
    selector = model.gpu_selector
    selected = set(selector.gpu_ids or []) if selector else set()

    gpus: List[GPUDeviceStatus] = []
    for worker in workers:
        if not worker.status or not worker.status.gpu_devices:
            continue
        for gpu in worker.status.gpu_devices:
            if (
                selected
                and make_gpu_id(worker.name, gpu.type, gpu.index) not in selected
            ):
                continue
            gpus.append(gpu)
    return gpus


def evaluate_local_extended_kv_cache(
    model: ModelSpec,
    workers: List[Worker],
) -> Optional[str]:
    """Why no GPU the deployment could land on runs vLLM's local extended
    KV cache.

    ``None`` when at least one does, which includes the deployment not asking
    for it. Shared mode is out of scope: the provider catalog decides which
    accelerators a cache service serves, and an unsupported one degrades to
    running without the cache rather than failing the deployment.
    """
    extended_kv_cache = model.extended_kv_cache
    if not (extended_kv_cache and extended_kv_cache.is_local()):
        return None

    def supported(gpu) -> bool:
        if gpu.vendor in (
            ManufacturerEnum.NVIDIA.value,
            ManufacturerEnum.AMD.value,
        ):
            return True
        if gpu.vendor == ManufacturerEnum.ASCEND.value:
            return (
                ascend_local_kv_cache_unsupported_reason(
                    gpu.arch_family, model.backend_version
                )
                is None
            )
        return False

    gpus = candidate_gpus(model, workers)
    if not gpus:
        # Nothing to judge: a pinned GPU whose worker was filtered out, or a
        # fleet with no GPUs at all. Scheduling reports either accurately,
        # and this check runs before it — a verdict here would take its place.
        return None

    if any(supported(gpu) for gpu in gpus):
        return None

    # Every reason among the candidates, not the first one: a 310P beside a
    # 910B on a version below the floor blocks for two different reasons, and
    # acting on one of them alone leaves the deployment where it was.
    reasons = {
        ascend_local_kv_cache_unsupported_reason(gpu.arch_family, model.backend_version)
        for gpu in gpus
        if gpu.vendor == ManufacturerEnum.ASCEND.value
    }
    reasons.discard(None)
    if reasons:
        return " ".join(sorted(reasons))

    return (
        "Extended KV cache with the vLLM backend requires NVIDIA, AMD or "
        "Ascend devices but none are available."
    )


async def evaluate_hybrid_model_kv_cache(
    session: AsyncSession,
    model: ModelSpec,
    workers: List[Worker],
    cache_service: Optional[CacheService] = None,
) -> Tuple[bool, List[str]]:
    """Report a hybrid-attention model whose extended KV cache cannot work
    as configured.

    Hybrid models (recurrent Mamba / linear-attention layers beside full
    attention) need a connector built for them, and a shared one needs its
    cache service configured for the model's block size. Neither is
    something the platform can arrange on the user's behalf: the block size
    comes out of the engine's own startup, so this reports what to do while
    the deployment is still being configured rather than guessing at
    runtime.

    The verdict errs toward silence — an unreadable config, a backend or
    accelerator this was never observed on, a provider declaring no caveat,
    or a service already carrying what its provider asks for all report
    nothing.
    """
    if not hybrid_verdict_possible(model, cache_service):
        return True, []

    try:
        # A second read of a config the metadata step already fetched, and
        # so a local cache hit: worth it over threading the config through
        # every evaluation for the one deployment shape that needs it.
        pretrained_config = await get_pretrained_config_with_workers(
            model, workers=workers
        )
    except Exception as e:
        logger.debug(
            f"Skipping the hybrid-model KV cache check for "
            f"{model.name or model.readable_source}: {e}"
        )
        return True, []

    if not is_hybrid_attention(pretrained_config):
        return True, []

    message = (
        evaluate_local_hybrid_kv_cache(model, workers)
        if model.extended_kv_cache.is_local()
        else await evaluate_shared_hybrid_kv_cache(session, cache_service)
    )
    return (False, [message]) if message else (True, [])


def hybrid_verdict_possible(
    model: ModelSpec, cache_service: Optional[CacheService]
) -> bool:
    """Whether a verdict could come out of this deployment at all.

    Answered from the spec and the already-resolved service, ahead of
    reading the model's config: that read is a local cache hit on the happy
    path but a hub round-trip on a cold or evicted one, and a deployment
    the checks below can only stay silent about — an SGLang backend, a
    service from another provider — should not pay for it on every
    evaluation that misses the cache.
    """
    ext = model.extended_kv_cache
    if not (ext and ext.enabled) or is_gguf_model(model):
        return False

    # Both verdicts are about vLLM's connectors. SGLang's in-process mode
    # runs its own hierarchical cache, and its LMCache adapter has not been
    # run against a hybrid model.
    if get_backend(model) != BackendEnum.VLLM:
        return False

    if ext.is_local():
        return True

    # None means the spec names no service, or names one its Org cannot see
    # (``visible_cache_service`` resolves both). Another provider's
    # connector has not been run against a hybrid model.
    return (
        cache_service is not None
        and (cache_service.provider_name or "").lower() == LMCACHE_PROVIDER_NAME.lower()
    )


def evaluate_local_hybrid_kv_cache(
    model: ModelSpec,
    workers: List[Worker],
) -> Optional[str]:
    """Why a hybrid-attention model cannot run vLLM's local extended KV
    cache. ``None`` when the verdict does not apply to this deployment.

    The in-process connector does not declare support for vLLM's hybrid
    memory allocator, so the engine turns the allocator off and then has to
    unify every layer onto one cache spec — which the recurrent and
    full-attention layers have none in common. No setting changes that, so
    the message points at the shared mode instead of at a knob.
    """
    gpus = candidate_gpus(model, workers)
    if not gpus:
        # Nothing to judge: a pinned GPU whose worker was filtered out, or a
        # fleet with no GPUs at all. Scheduling reports either accurately,
        # and a verdict here would take its place — this check returns
        # before scheduling ever runs.
        return None

    if any(gpu.vendor == ManufacturerEnum.ASCEND.value for gpu in gpus):
        # An Ascend placement runs the connector vllm-ascend ships rather
        # than this one. The label, GPU-type and backend-framework filters
        # are not replayed here, so one Ascend candidate is a placement this
        # cannot rule out — and reporting a deployment that would have run
        # costs more than staying quiet about one that will not.
        return None

    return (
        "Hybrid-attention models (recurrent layers beside full attention) "
        "cannot run with the local extended KV cache: its connector does "
        "not support vLLM's hybrid memory allocator, so the engine turns "
        "the allocator off and fails to start. Attach the deployment to a "
        "cache service configured for the model instead, or turn extended "
        f"KV cache off. See the {_doc_link()}."
    )


async def evaluate_shared_hybrid_kv_cache(
    session: AsyncSession,
    service: CacheService,
) -> Optional[str]:
    """Why the attached LMCache service cannot serve this hybrid-attention
    model. ``None`` when it can.

    LMCache is named here rather than declared in the provider catalog:
    it is the only provider whose connector has been run against a hybrid
    model, and the catalog is a document admins write — a field added
    there is a schema every later version has to keep reading. The cost
    is that a provider an extension ships cannot state a caveat of its
    own; it stays silent, which is what an unverified provider should do
    anyway.
    """
    if await lmcache_service_serves_hybrid_models(session, service):
        return None

    return (
        f"Cache service '{service.name}' is not configured for "
        f"hybrid-attention models, and the deployment would fail to start. "
        f"{LMCACHE_HYBRID_SERVICE_NOTE} See the {_doc_link()}."
    )


async def lmcache_service_serves_hybrid_models(
    session: AsyncSession,
    service: CacheService,
) -> bool:
    """Whether an LMCache service carries what a hybrid model needs.

    Read from what the user set, never from a declared default: the
    question is whether someone configured this service for a hybrid
    model, and a value the catalog supplies on its own is no evidence of
    that. Whether the chunk size is the right multiple is not answerable
    here — the block size comes out of the engine's own startup — so a
    service carrying both settings is taken at its word.
    """
    config = service.config
    fields = (config.fields if config else None) or {}
    if not fields.get(LMCACHE_HYBRID_SERVICE_FIELD):
        return False

    # The flag belongs to the component engines attach to: the cache
    # server, whose parser is the one that takes it. The catalog names
    # that component, so a renamed one is followed rather than guessed.
    provider = await get_cache_provider(session, service.provider_name)
    if provider is None:
        # An admin's own catalog no longer declaring the provider, or an
        # extension's that is not installed here. Which component the
        # parameters belong to is then unknown, and reading the wrong one
        # reports a configured service as unconfigured — so the service is
        # taken at its word, as everywhere else this cannot tell.
        return True

    component = provider.attach_component()
    parameters = ((config.parameters if config else None) or {}).get(component) or []
    argv = flatten_to_argv(list(parameters))
    return any(
        token == LMCACHE_HYBRID_SERVICE_PARAMETER
        or token.startswith(f"{LMCACHE_HYBRID_SERVICE_PARAMETER}=")
        for token in argv
    )


async def evaluate_model_metadata(
    config: Config,
    model: ModelSpec,
    workers: List[Worker],
) -> Tuple[bool, List[str]]:
    try:
        if model.source == SourceEnum.LOCAL_PATH:
            # Check if local path exists on server
            path_exists_on_server = os.path.exists(model.local_path)

            if not path_exists_on_server:
                # Try to check if path exists on any worker
                try:
                    async with WorkerFilesystemClient() as filesystem_client:
                        selector = WorkerSelector(filesystem_client)

                        found_worker = await selector.find_worker_with_path(
                            workers, path=model.local_path
                        )

                        if found_worker:
                            logger.info(
                                f"Found path {model.local_path} on worker {found_worker.id}"
                            )
                        else:
                            # Path not found on any worker
                            return False, [
                                "The model file path you specified does not exist."
                                "Please ensure the model file is accessible from at least one node."
                            ]
                except Exception as e:
                    logger.warning(
                        f"Failed to check path on workers: {e}, falling back to local check"
                    )
                    # Fallback to original warning
                    return False, [
                        "Failed to get model metadata. The model file path you specified does not exist."
                    ]

        if model.source in [
            SourceEnum.HUGGING_FACE,
            SourceEnum.MODEL_SCOPE,
        ]:
            repo_id = model.huggingface_repo_id
            if model.source == SourceEnum.MODEL_SCOPE:
                repo_id = model.model_scope_model_id
            if not is_repo_cached(repo_id, model.source):
                await run_in_thread(
                    auth_check,
                    timeout=15,
                    model=model,
                    huggingface_token=config.huggingface_token,
                )

        if is_gguf_model(model):
            await scheduler.evaluate_gguf_model(model, workers=workers)
        elif not is_audio_model(model):
            await scheduler.evaluate_pretrained_config(model, workers=workers)
    except Exception as e:
        if model.env and model.env.get("GPUSTACK_SKIP_MODEL_EVALUATION"):
            logger.warning(f"Ignore model evaluation error for model {model.name}: {e}")
            return True, []

        return False, [str(e)]

    return True, []


async def evaluate_model_input(
    session: AsyncSession,
    model: ModelSpec,
    cluster_id: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    try:
        await validate_model_in(session, model, cluster_id=cluster_id)
    except HTTPException as e:
        return False, [e.message]
    except Exception as e:
        return False, [str(e)]

    return True, []


async def set_default_spec(session: AsyncSession, model: ModelSpec) -> bool:
    """
    Set the default spec for the model if it matches the catalog spec.
    """
    model_spec_in_catalog = await get_catalog_spec_by_source_key(
        session, model.model_source_key
    )

    modified = False
    if model_spec_in_catalog:
        if (
            model_spec_in_catalog.backend_parameters
            and model.backend_parameters is None
        ):
            model.backend_parameters = model_spec_in_catalog.backend_parameters
            modified = True

        if model_spec_in_catalog.env and model.env is None:
            model.env = model_spec_in_catalog.env
            modified = True

        if model_spec_in_catalog.categories and not model.categories:
            model.categories = model_spec_in_catalog.categories
            modified = True

    gpus_per_replica_modified = scheduler.set_model_gpus_per_replica(model)
    return modified or gpus_per_replica_modified
