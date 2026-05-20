from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional
from fastapi import APIRouter, Query
from sqlmodel import desc, distinct, select, func, col, and_, or_
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.common import ItemList
from gpustack.schemas.dashboard import (
    CurrentSystemLoad,
    HistorySystemLoad,
    ModelSummary,
    ModelUsageStats,
    ModelUsageSummary,
    ModelUsageUserSummary,
    ResourceClaim,
    ResourceCounts,
    SystemLoadSummary,
    SystemSummary,
    TimeSeriesData,
)
from gpustack.api.exceptions import ForbiddenException
from gpustack.api.tenant import assert_cluster_visible
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.principals import OrgRole
from gpustack.schemas.system_load import SystemLoad
from gpustack.schemas.users import User
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.schemas import Worker, Cluster
from gpustack.schemas.model_provider import ModelProvider
from gpustack.server.system_load import compute_system_load

router = APIRouter()


def _resolve_dashboard_scope(ctx) -> Optional[int]:
    """Owner-principal id to scope dashboard queries by, or None for
    unscoped (platform-wide).

    - Platform admin in "All" mode (no current_principal_id) → None,
      every helper aggregates across the platform like before.
    - Platform admin acting inside an Org context → that org id, so
      admins see the same per-org view their Org owners see.
    - Org owner of the current Org → that org id.
    - Anyone else → ForbiddenException.

    The org-owner branch intentionally excludes Personal scope: org_role
    is only set for real ORG memberships, never for a user's own
    USER-principal, so `ctx.org_role == OrgRole.OWNER` already rules
    Personal out.
    """
    if ctx.is_platform_admin:
        return ctx.current_principal_id
    if ctx.current_principal_id is not None and ctx.org_role == OrgRole.OWNER:
        return ctx.current_principal_id
    raise ForbiddenException(message="Insufficient permission to view the dashboard")


@router.get("")
async def dashboard(
    session: SessionDep,
    ctx: TenantContextDep,
    cluster_id: Optional[int] = None,
):
    # Permission split: the cluster-detail header (Cluster Basic +
    # System Load tiles) calls this with ``cluster_id`` and is open to
    # any caller who can see the cluster — that's the same audience
    # cluster-detail itself uses. The aggregate dashboard (no
    # ``cluster_id``) is open to platform admin and to Org owners; the
    # latter sees the same panels scoped to their org via
    # owner_principal_id filters in each helper.
    if cluster_id is not None:
        cluster = await Cluster.one_by_id(session, cluster_id)
        assert_cluster_visible(ctx, cluster, not_found_message="Cluster not found")
        owner_principal_id: Optional[int] = None
    else:
        owner_principal_id = _resolve_dashboard_scope(ctx)

    resource_counts = await get_resource_counts(session, cluster_id, owner_principal_id)
    system_load = await get_system_load(session, cluster_id, owner_principal_id)
    model_usage = await get_model_usage_summary(session, cluster_id, owner_principal_id)
    active_models = await get_active_models(session, cluster_id, owner_principal_id)
    summary = SystemSummary(
        cluster_id=cluster_id,
        resource_counts=resource_counts,
        system_load=system_load,
        model_usage=model_usage,
        active_models=active_models,
    )

    return summary


async def get_resource_counts(
    session: AsyncSession,
    cluster_id: Optional[int] = None,
    owner_principal_id: Optional[int] = None,
) -> ResourceCounts:
    fields = {}
    cluster_count = None
    if cluster_id is not None:
        fields['cluster_id'] = cluster_id
    else:
        cluster_fields = {"deleted_at": None}
        if owner_principal_id is not None:
            cluster_fields["owner_principal_id"] = owner_principal_id
        clusters = await Cluster.all_by_fields(session, fields=cluster_fields)
        cluster_count = len(clusters)
    if owner_principal_id is not None:
        # `owner_principal_id` is denormalized onto Worker, Model, and
        # ModelInstance, so the per-org filter is a column equality on
        # each — no joins.
        fields["owner_principal_id"] = owner_principal_id
    workers = await Worker.all_by_fields(
        session,
        fields=fields,
    )
    worker_count = len(workers)
    gpu_count = 0
    for worker in workers:
        gpu_count += len(worker.status.gpu_devices or [])
    models = await Model.all_by_fields(session, fields=fields)
    model_count = len(models)
    model_instances = await ModelInstance.all_by_fields(session, fields=fields)
    model_instance_count = len(model_instances)
    return ResourceCounts(
        cluster_count=cluster_count,
        worker_count=worker_count,
        gpu_count=gpu_count,
        model_count=model_count,
        model_instance_count=model_instance_count,
    )


async def get_system_load(
    session: AsyncSession,
    cluster_id: Optional[int] = None,
    owner_principal_id: Optional[int] = None,
) -> SystemLoadSummary:
    fields = {}
    if cluster_id is not None:
        fields['cluster_id'] = cluster_id
    if owner_principal_id is not None:
        fields["owner_principal_id"] = owner_principal_id
    workers = await Worker.all_by_fields(session, fields=fields)
    current_system_loads = compute_system_load(workers)
    current_system_load = next(
        (load for load in current_system_loads if load.cluster_id == cluster_id),
        SystemLoad(
            cluster_id=cluster_id,
            cpu=0,
            ram=0,
            gpu=0,
            vram=0,
        ),
    )

    now = datetime.now(timezone.utc)

    one_hour_ago = int((now - timedelta(hours=1)).timestamp())

    if cluster_id is not None:
        statement = select(SystemLoad).where(
            SystemLoad.timestamp >= one_hour_ago,
            SystemLoad.cluster_id == cluster_id,
        )
    elif owner_principal_id is not None:
        # SystemLoad rows aren't denormalized with owner_principal_id, so
        # narrow via the owning Cluster ids. There's one row per cluster
        # per timestamp, so for org dashboards we average the per-cluster
        # rates by timestamp — otherwise the UI would plot N overlapping
        # series for an org with N clusters.
        #
        # AVG (not SUM) because cpu/ram/gpu/vram are utilization rates
        # (avg per worker / per GPU); summing percentages crosses 100%
        # nonsense. AVG of per-cluster averages loses the worker-count
        # weighting the platform-wide pre-aggregate (cluster_id IS NULL)
        # has, but the historical worker-count isn't on SystemLoad, so
        # a true weighted average isn't available without a separate
        # per-org pre-aggregate. Good-enough first pass.
        cluster_subq = select(Cluster.id).where(
            Cluster.owner_principal_id == owner_principal_id,
            Cluster.deleted_at.is_(None),
        )
        statement = (
            select(
                SystemLoad.timestamp.label("timestamp"),
                func.avg(SystemLoad.cpu).label("cpu"),
                func.avg(SystemLoad.ram).label("ram"),
                func.avg(SystemLoad.gpu).label("gpu"),
                func.avg(SystemLoad.vram).label("vram"),
            )
            .where(
                SystemLoad.timestamp >= one_hour_ago,
                SystemLoad.cluster_id.in_(cluster_subq),
            )
            .group_by(SystemLoad.timestamp)
        )
    else:
        # Platform-wide aggregate dashboard preserves the historical
        # behavior: aggregate-cluster_id rows (cluster_id IS NULL) are
        # the platform-level summary, computed by the collector.
        statement = select(SystemLoad).where(
            SystemLoad.timestamp >= one_hour_ago,
            SystemLoad.cluster_id.is_(None),
        )

    system_loads = (await session.exec(statement)).all()

    cpu = []
    ram = []
    gpu = []
    vram = []
    for system_load in system_loads:
        cpu.append(
            TimeSeriesData(
                timestamp=system_load.timestamp,
                value=system_load.cpu,
            )
        )
        ram.append(
            TimeSeriesData(
                timestamp=system_load.timestamp,
                value=system_load.ram,
            )
        )
        gpu.append(
            TimeSeriesData(
                timestamp=system_load.timestamp,
                value=system_load.gpu,
            )
        )
        vram.append(
            TimeSeriesData(
                timestamp=system_load.timestamp,
                value=system_load.vram,
            )
        )
    cpu.sort(key=lambda x: x.timestamp, reverse=False)
    ram.sort(key=lambda x: x.timestamp, reverse=False)
    gpu.sort(key=lambda x: x.timestamp, reverse=False)
    vram.sort(key=lambda x: x.timestamp, reverse=False)
    return SystemLoadSummary(
        current=CurrentSystemLoad(
            cpu=current_system_load.cpu,
            ram=current_system_load.ram,
            gpu=current_system_load.gpu,
            vram=current_system_load.vram,
        ),
        history=HistorySystemLoad(
            cpu=cpu,
            ram=ram,
            gpu=gpu,
            vram=vram,
        ),
    )


async def get_model_usage_stats(
    session: AsyncSession,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    model_ids: Optional[List[int]] = None,
    user_ids: Optional[List[int]] = None,
    cluster_id: Optional[int] = None,
    provider_model_names: Optional[Dict[int, Optional[List[str]]]] = None,
    owner_principal_id: Optional[int] = None,
) -> ModelUsageStats:
    if start_date is None or end_date is None:
        end_date = date.today()
        start_date = end_date - timedelta(days=31)
    if model_ids is None and cluster_id is not None:
        models = await Model.all_by_fields(session, fields={"cluster_id": cluster_id})
        model_ids = [model.id for model in models]
    statement = (
        select(
            ModelUsage.date,
            func.sum(ModelUsage.prompt_token_count).label('total_prompt_tokens'),
            func.sum(ModelUsage.completion_token_count).label(
                'total_completion_tokens'
            ),
            func.sum(ModelUsage.request_count).label('total_requests'),
        )
        .where(ModelUsage.date >= start_date)
        .where(ModelUsage.date <= end_date)
        .group_by(ModelUsage.date)
        .order_by(ModelUsage.date)
    )

    or_conditions = []
    if model_ids is not None:
        or_conditions.append(col(ModelUsage.model_id).in_(model_ids))
    for provider_id, model_names in (provider_model_names or {}).items():
        if provider_id is not None:
            and_conds = [col(ModelUsage.provider_id) == provider_id]
            if model_names:
                and_conds.append(col(ModelUsage.model_name).in_(model_names))
            or_conditions.append(and_(*and_conds))
    if or_conditions:
        statement = statement.where(or_(*or_conditions))

    if user_ids is not None:
        statement = statement.where(col(ModelUsage.user_id).in_(user_ids))

    if owner_principal_id is not None:
        statement = statement.where(ModelUsage.owner_principal_id == owner_principal_id)

    results = (await session.exec(statement)).all()

    prompt_token_history = []
    completion_token_history = []
    api_request_history = []
    for result in results:
        prompt_token_history.append(
            TimeSeriesData(
                timestamp=int(
                    datetime.combine(result.date, datetime.min.time()).timestamp()
                ),
                value=result.total_prompt_tokens,
            )
        )
        completion_token_history.append(
            TimeSeriesData(
                timestamp=int(
                    datetime.combine(result.date, datetime.min.time()).timestamp()
                ),
                value=result.total_completion_tokens,
            )
        )
        api_request_history.append(
            TimeSeriesData(
                timestamp=int(
                    datetime.combine(result.date, datetime.min.time()).timestamp()
                ),
                value=result.total_requests,
            )
        )

    return ModelUsageStats(
        api_request_history=api_request_history,
        prompt_token_history=prompt_token_history,
        completion_token_history=completion_token_history,
    )


async def get_model_usage_summary(
    session: AsyncSession,
    cluster_id: Optional[int] = None,
    owner_principal_id: Optional[int] = None,
) -> ModelUsageSummary:
    model_usage_stats = await get_model_usage_stats(
        session,
        cluster_id=cluster_id,
        owner_principal_id=owner_principal_id,
    )
    # get top users
    today = date.today()
    one_month_ago = today - timedelta(days=31)

    statement = (
        select(
            ModelUsage.user_id,
            User.name.label('username'),
            func.sum(ModelUsage.prompt_token_count).label('total_prompt_tokens'),
            func.sum(ModelUsage.completion_token_count).label(
                'total_completion_tokens'
            ),
        )
        .join(User, ModelUsage.user_id == User.id)
        .where(ModelUsage.date >= one_month_ago)
        .group_by(ModelUsage.user_id, User.name)
        .order_by(
            func.sum(
                ModelUsage.prompt_token_count + ModelUsage.completion_token_count
            ).desc()
        )
        .limit(10)
    )

    if owner_principal_id is not None:
        statement = statement.where(ModelUsage.owner_principal_id == owner_principal_id)

    results = (await session.exec(statement)).all()
    top_users = []
    for result in results:
        top_users.append(
            ModelUsageUserSummary(
                user_id=result.user_id,
                username=result.username,
                prompt_token_count=result.total_prompt_tokens,
                completion_token_count=result.total_completion_tokens,
            )
        )

    return ModelUsageSummary(
        api_request_history=model_usage_stats.api_request_history,
        prompt_token_history=model_usage_stats.prompt_token_history,
        completion_token_history=model_usage_stats.completion_token_history,
        top_users=top_users,
    )


async def _get_maas_active_models(
    session: AsyncSession,
    owner_principal_id: Optional[int] = None,
) -> List[ModelSummary]:
    if owner_principal_id is None:
        all_providers = await ModelProvider.all_by_fields(
            session, fields={"deleted_at": None}
        )
    else:
        # ``ModelProvider.owner_principal_id IS NULL`` means a global,
        # admin-managed provider that every Org can call. An Org owner
        # viewing their dashboard should see usage of those alongside
        # their own org-scoped providers — otherwise active_models
        # would silently drop traffic the org actually generated.
        stmt = select(ModelProvider).where(
            ModelProvider.deleted_at.is_(None),
            or_(
                ModelProvider.owner_principal_id == owner_principal_id,
                ModelProvider.owner_principal_id.is_(None),
            ),
        )
        all_providers = (await session.exec(stmt)).all()
    if not all_providers:
        return []

    provider_ids = [p.id for p in all_providers]
    total_tokens = func.sum(
        ModelUsage.prompt_token_count + ModelUsage.completion_token_count
    )
    # Aggregate model usage in the database for efficiency
    statement = (
        select(
            ModelUsage.provider_id,
            ModelUsage.model_name,
            total_tokens.label("total_token_count"),
        )
        .where(col(ModelUsage.provider_id).in_(provider_ids))
        .group_by(ModelUsage.provider_id, ModelUsage.model_name)
        .order_by(func.coalesce(total_tokens, 0).desc())
        .limit(10)
    )
    top_model_usages = (await session.exec(statement)).all()

    models_by_provider_and_name = {
        (p.id, m.name): m for p in all_providers for m in (p.models or [])
    }

    provider_id_to_name = {p.id: p.name for p in all_providers}

    model_summaries = []
    for usage in top_model_usages:
        model = models_by_provider_and_name.get((usage.provider_id, usage.model_name))

        model_summaries.append(
            ModelSummary(
                provider_id=usage.provider_id,
                provider_name=provider_id_to_name.get(
                    usage.provider_id, "Unknown Provider"
                ),
                name=usage.model_name,
                instance_count=0,
                token_count=int(usage.total_token_count or 0),
                categories=([model.category] if model and model.category else None),
            )
        )

    return model_summaries


async def _get_gpustack_active_models(
    session: AsyncSession,
    cluster_id: Optional[int] = None,
    owner_principal_id: Optional[int] = None,
) -> List[ModelSummary]:
    statement = active_model_statement(
        cluster_id=cluster_id, owner_principal_id=owner_principal_id
    )

    results = (await session.exec(statement)).all()

    top_model_ids = [result.id for result in results]
    extra_conditions = [
        col(ModelInstance.model_id).in_(top_model_ids),
    ]
    model_instances: List[ModelInstance] = await ModelInstance.all_by_fields(
        session, fields={}, extra_conditions=extra_conditions
    )
    model_instances_by_id: Dict[int, List[ModelInstance]] = {}
    for model_instance in model_instances:
        if model_instance.model_id not in model_instances_by_id:
            model_instances_by_id[model_instance.model_id] = []
        model_instances_by_id[model_instance.model_id].append(model_instance)

    model_summary = []
    for result in results:
        # We need to summarize the resource claims for all model instances including distributed servers.
        # It's complicated to do this in a SQL statement, so we do it in Python.
        resource_claim = ResourceClaim(
            ram=0,
            vram=0,
        )
        if result.id in model_instances_by_id:
            for model_instance in model_instances_by_id[result.id]:
                aggregate_resource_claim(resource_claim, model_instance)

        model_summary.append(
            ModelSummary(
                id=result.id,
                name=result.name,
                categories=result.categories,
                resource_claim=resource_claim,
                instance_count=result.instance_count,
                token_count=(
                    result.total_token_count
                    if result.total_token_count is not None
                    else 0
                ),
            )
        )

    return model_summary


async def get_active_models(
    session: AsyncSession,
    cluster_id: Optional[int] = None,
    owner_principal_id: Optional[int] = None,
) -> List[ModelSummary]:
    summary = await _get_gpustack_active_models(session, cluster_id, owner_principal_id)
    if cluster_id is None:
        maas_active_models = await _get_maas_active_models(session, owner_principal_id)
        summary.extend(maas_active_models)
    summary.sort(key=lambda x: x.token_count, reverse=True)
    summary = summary[:10]
    return summary


def aggregate_resource_claim(
    resource_claim: ResourceClaim,
    model_instance: ModelInstance,
):
    if model_instance.computed_resource_claim is not None:
        resource_claim.ram += model_instance.computed_resource_claim.ram or 0
        for vram in (model_instance.computed_resource_claim.vram or {}).values():
            resource_claim.vram += vram

    if (
        model_instance.distributed_servers
        and model_instance.distributed_servers.subordinate_workers
    ):
        for subworker in model_instance.distributed_servers.subordinate_workers:
            if subworker.computed_resource_claim is not None:
                resource_claim.ram += subworker.computed_resource_claim.ram or 0
                for vram in (subworker.computed_resource_claim.vram or {}).values():
                    resource_claim.vram += vram


def active_model_statement(
    cluster_id: Optional[int],
    owner_principal_id: Optional[int] = None,
) -> select:
    usage_sum_query = (
        select(
            Model.id.label('model_id'),
            func.sum(
                ModelUsage.prompt_token_count + ModelUsage.completion_token_count
            ).label('total_token_count'),
        )
        .outerjoin(ModelUsage, Model.id == ModelUsage.model_id)
        .group_by(Model.id)
    ).alias('usage_sum')

    statement = select(
        Model.id,
        Model.name,
        Model.categories,
        func.count(distinct(ModelInstance.id)).label('instance_count'),
        usage_sum_query.c.total_token_count,
    )
    if cluster_id is not None:
        statement = statement.where(Model.cluster_id == cluster_id)
    if owner_principal_id is not None:
        statement = statement.where(Model.owner_principal_id == owner_principal_id)

    statement = (
        statement.join(ModelInstance, Model.id == ModelInstance.model_id)
        .outerjoin(usage_sum_query, Model.id == usage_sum_query.c.model_id)
        .group_by(
            Model.id,
            usage_sum_query.c.total_token_count,
        )
        .order_by(func.coalesce(usage_sum_query.c.total_token_count, 0).desc())
        .limit(10)
    )

    return statement


async def get_model_usages(
    session: AsyncSession,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    model_ids: Optional[List[int]] = None,
    user_ids: Optional[List[int]] = None,
    provider_model_names: Optional[Dict[int, Optional[List[str]]]] = None,
    owner_principal_id: Optional[int] = None,
) -> List[ModelUsage]:
    if start_date is None or end_date is None:
        end_date = date.today()
        start_date = end_date - timedelta(days=31)

    statement = (
        select(ModelUsage)
        .where(ModelUsage.date >= start_date)
        .where(ModelUsage.date <= end_date)
    )

    or_conditions = []
    if model_ids is not None:
        or_conditions.append(col(ModelUsage.model_id).in_(model_ids))
    for provider_id, model_names in (provider_model_names or {}).items():
        if provider_id is not None:
            and_conds = [col(ModelUsage.provider_id) == provider_id]
            if model_names:
                and_conds.append(col(ModelUsage.model_name).in_(model_names))
            or_conditions.append(and_(*and_conds))
    if or_conditions:
        statement = statement.where(or_(*or_conditions))

    if user_ids is not None:
        statement = statement.where(col(ModelUsage.user_id).in_(user_ids))

    if owner_principal_id is not None:
        statement = statement.where(ModelUsage.owner_principal_id == owner_principal_id)

    statement = statement.order_by(
        desc(ModelUsage.date),
        ModelUsage.user_id,
        ModelUsage.completion_token_count,
    )

    return (await session.exec(statement)).all()


def get_models_by_provider_id(
    provider_model_names: List[str],
) -> Optional[Dict[int, Optional[List[str]]]]:
    model_names_by_provider_id = {}
    for id_prefix_name in provider_model_names or []:
        if ":" not in id_prefix_name:
            continue
        id_str, name = id_prefix_name.split(":", 1)
        try:
            provider_id = int(id_str)
        except ValueError:
            continue
        names: List[str] = model_names_by_provider_id.setdefault(provider_id, [])
        names.extend([name] if name else [])
    return model_names_by_provider_id if len(model_names_by_provider_id) > 0 else None


@router.get("/usage")
async def usage(
    session: SessionDep,
    ctx: TenantContextDep,
    start_date: Optional[date] = Query(
        None,
        description="Start date for the usage data (YYYY-MM-DD). Defaults to 31 days ago.",
    ),
    end_date: Optional[date] = Query(
        None, description="End date for the usage data (YYYY-MM-DD). Defaults to today."
    ),
    model_ids: Optional[List[int]] = Query(
        None,
        description="Filter by model IDs. Defaults to all models.",
    ),
    user_ids: Optional[List[int]] = Query(
        None, description="Filter by user IDs. Defaults to all users."
    ),
    provider_model_names: Optional[List[str]] = Query(
        None,
        description="Filter by provider and model names. Format is 'provider_id:model_name'. To filter by provider ID only, use 'provider_id:'. Defaults to no filtering.",
    ),
):
    """
    Get model usage records.
    This endpoint returns detailed model usage records within a specified date range.
    """
    owner_principal_id = _resolve_dashboard_scope(ctx)
    items = await get_model_usages(
        session,
        start_date=start_date,
        end_date=end_date,
        model_ids=model_ids,
        user_ids=user_ids,
        provider_model_names=get_models_by_provider_id(provider_model_names or []),
        owner_principal_id=owner_principal_id,
    )
    return ItemList[ModelUsage](items=items)


@router.get("/usage/stats")
async def usage_stats(
    session: SessionDep,
    ctx: TenantContextDep,
    start_date: Optional[date] = Query(
        None,
        description="Start date for the usage data (YYYY-MM-DD). Defaults to 31 days ago.",
    ),
    end_date: Optional[date] = Query(
        None, description="End date for the usage data (YYYY-MM-DD). Defaults to today."
    ),
    model_ids: Optional[List[int]] = Query(
        None,
        description="Filter by model IDs. Defaults to all models.",
    ),
    user_ids: Optional[List[int]] = Query(
        None, description="Filter by user IDs. Defaults to all users."
    ),
    provider_model_names: Optional[List[str]] = Query(
        None,
        description="Filter by provider and model names. Format is 'provider_id:model_name'. To filter by provider ID only, use 'provider_id:'. Defaults to no filtering.",
    ),
):
    """
    Get model usage statistics.
    This endpoint returns aggregated statistics for model usage, including token counts and request counts.
    It can filter by date range, model IDs, user IDs, model names with provider ID prefix.
    """
    owner_principal_id = _resolve_dashboard_scope(ctx)
    return await get_model_usage_stats(
        session,
        start_date=start_date,
        end_date=end_date,
        model_ids=model_ids,
        user_ids=user_ids,
        provider_model_names=get_models_by_provider_id(provider_model_names or []),
        owner_principal_id=owner_principal_id,
    )
