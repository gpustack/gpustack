from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional
from fastapi import APIRouter, Query
from sqlmodel import desc, distinct, select, func, col
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
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.system_load import SystemLoad
from gpustack.schemas.users import User
from gpustack.server.deps import SessionDep
from gpustack.schemas import Worker, Cluster
from gpustack.schemas.model_provider import ModelProvider
from gpustack.server.system_load import compute_system_load

router = APIRouter()


@router.get("")
async def dashboard(
    session: SessionDep,
    cluster_id: Optional[int] = None,
):
    resoruce_counts = await get_resource_counts(session, cluster_id)
    system_load = await get_system_load(session, cluster_id)
    model_usage = await get_model_usage_summary(session, cluster_id)
    active_models = await get_active_models(session, cluster_id)
    summary = SystemSummary(
        cluster_id=cluster_id,
        resource_counts=resoruce_counts,
        system_load=system_load,
        model_usage=model_usage,
        active_models=active_models,
    )

    return summary


async def get_resource_counts(
    session: AsyncSession, cluster_id: Optional[int] = None
) -> ResourceCounts:
    fields = {}
    cluster_count = None
    if cluster_id is not None:
        fields['cluster_id'] = cluster_id
    else:
        clusters = await Cluster.all_by_field(session, field="deleted_at", value=None)
        cluster_count = len(clusters)
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
    session: AsyncSession, cluster_id: Optional[int] = None
) -> SystemLoadSummary:
    fields = {}
    if cluster_id is not None:
        fields['cluster_id'] = cluster_id
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

    statement = select(SystemLoad)
    statement = statement.where(SystemLoad.cluster_id == cluster_id)
    statement = statement.where(SystemLoad.timestamp >= one_hour_ago)

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

    if model_ids is not None:
        statement = statement.where(col(ModelUsage.model_id).in_(model_ids))

    if user_ids is not None:
        statement = statement.where(col(ModelUsage.user_id).in_(user_ids))

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
    session: AsyncSession, cluster_id: Optional[int] = None
) -> ModelUsageSummary:
    model_usage_stats = await get_model_usage_stats(session, cluster_id=cluster_id)
    # get top users
    today = date.today()
    one_month_ago = today - timedelta(days=31)

    statement = (
        select(
            ModelUsage.user_id,
            User.username,
            func.sum(ModelUsage.prompt_token_count).label('total_prompt_tokens'),
            func.sum(ModelUsage.completion_token_count).label(
                'total_completion_tokens'
            ),
        )
        .join(User, ModelUsage.user_id == User.id)
        .where(ModelUsage.date >= one_month_ago)
        .group_by(ModelUsage.user_id, User.username)
        .order_by(
            func.sum(
                ModelUsage.prompt_token_count + ModelUsage.completion_token_count
            ).desc()
        )
        .limit(10)
    )

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


async def _get_maas_active_models(session: AsyncSession) -> List[ModelSummary]:
    all_providers = await ModelProvider.all_by_field(
        session, field="deleted_at", value=None
    )
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
    session: AsyncSession, cluster_id: Optional[int] = None
) -> List[ModelSummary]:
    statement = active_model_statement(cluster_id=cluster_id)

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
    session: AsyncSession, cluster_id: Optional[int] = None
) -> List[ModelSummary]:
    summary = await _get_gpustack_active_models(session, cluster_id)
    if cluster_id is None:
        maas_active_models = await _get_maas_active_models(session)
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


def active_model_statement(cluster_id: Optional[int]) -> select:
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
) -> List[ModelUsage]:
    if start_date is None or end_date is None:
        end_date = date.today()
        start_date = end_date - timedelta(days=31)

    statement = (
        select(ModelUsage)
        .where(ModelUsage.date >= start_date)
        .where(ModelUsage.date <= end_date)
    )

    if model_ids is not None:
        statement = statement.where(col(ModelUsage.model_id).in_(model_ids))

    if user_ids is not None:
        statement = statement.where(col(ModelUsage.user_id).in_(user_ids))

    statement = statement.order_by(
        desc(ModelUsage.date),
        ModelUsage.user_id,
        ModelUsage.completion_token_count,
    )

    return (await session.exec(statement)).all()


@router.get("/usage")
async def usage(
    session: SessionDep,
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
):
    """
    Get model usage records.
    This endpoint returns detailed model usage records within a specified date range.
    """
    items = await get_model_usages(
        session,
        start_date=start_date,
        end_date=end_date,
        model_ids=model_ids,
        user_ids=user_ids,
    )
    return ItemList[ModelUsage](items=items)


@router.get("/usage/stats")
async def usage_stats(
    session: SessionDep,
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
):
    """
    Get model usage statistics.
    This endpoint returns aggregated statistics for model usage, including token counts and request counts.
    It can filter by date range, model IDs, and user IDs.
    """
    return await get_model_usage_stats(
        session,
        start_date=start_date,
        end_date=end_date,
        model_ids=model_ids,
        user_ids=user_ids,
    )
