from datetime import date, datetime, timedelta, timezone
from typing import List, Optional
from fastapi import APIRouter, Query
from sqlalchemy import JSON, BigInteger, case, cast, text
from sqlmodel import distinct, select, func, col
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
from gpustack.server.db import get_engine
from gpustack.server.deps import SessionDep
from gpustack.schemas import Worker
from gpustack.server.system_load import compute_system_load

router = APIRouter()


@router.get("")
async def dashboard(
    session: SessionDep,
):
    resoruce_counts = await get_resource_counts(session)
    system_load = await get_system_load(session)
    model_usage = await get_model_usage_summary(session)
    active_models = await get_active_models(session)
    summary = SystemSummary(
        resource_counts=resoruce_counts,
        system_load=system_load,
        model_usage=model_usage,
        active_models=active_models,
    )

    return summary


async def get_resource_counts(session: AsyncSession) -> ResourceCounts:
    workers = await Worker.all(session)
    worker_count = len(workers)
    gpu_count = 0
    for worker in workers:
        gpu_count += len(worker.status.gpu_devices or [])
    model_count = await Model.count(session)
    model_instance_count = await ModelInstance.count(session)
    return ResourceCounts(
        worker_count=worker_count,
        gpu_count=gpu_count,
        model_count=model_count,
        model_instance_count=model_instance_count,
    )


async def get_system_load(session: AsyncSession) -> SystemLoadSummary:
    workers = await Worker.all(session)
    current_system_load = compute_system_load(workers)

    now = datetime.now(timezone.utc)

    one_hour_ago = int((now - timedelta(hours=1)).timestamp())

    statement = select(SystemLoad).where(SystemLoad.timestamp >= one_hour_ago)

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
) -> ModelUsageStats:
    if start_date is None or end_date is None:
        end_date = date.today()
        start_date = end_date - timedelta(days=31)

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


async def get_model_usage_summary(session: AsyncSession) -> ModelUsageSummary:
    model_usage_stats = await get_model_usage_stats(session)
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


async def get_active_models(session: AsyncSession) -> List[ModelSummary]:
    statement = active_model_statement()

    results = (await session.exec(statement)).all()
    model_summary = []
    for result in results:
        model_summary.append(
            ModelSummary(
                id=result.id,
                name=result.name,
                categories=result.categories,
                resource_claim=ResourceClaim(
                    ram=result.total_ram_claim,
                    vram=result.total_vram_claim,
                ),
                instance_count=result.instance_count,
                token_count=(
                    result.total_token_count
                    if result.total_token_count is not None
                    else 0
                ),
            )
        )

    return model_summary


def active_model_statement() -> select:
    dialect = get_engine().dialect.name
    if dialect == 'sqlite':
        vram_values = func.json_each(
            ModelInstance.computed_resource_claim, '$.vram'
        ).table_valued('value', joins_implicitly=True)

        ram_claim = func.cast(
            func.json_extract(ModelInstance.computed_resource_claim, '$.ram'),
            BigInteger,
        )
    elif dialect == 'mysql':
        vram_case = case(
            (
                func.json_type(
                    func.json_extract(ModelInstance.computed_resource_claim, '$.vram')
                )
                == 'OBJECT',
                func.json_extract(ModelInstance.computed_resource_claim, '$.vram'),
            ),
            else_=func.cast(cast({"0": 0}, JSON), JSON),
        )

        # Use text() to preserve the native SQL expression
        vram_values = func.json_table(
            vram_case, text("""'$.*' COLUMNS (value BIGINT PATH '$')""")
        ).table_valued('value')

        ram_claim = func.cast(
            func.json_unquote(
                func.json_extract(ModelInstance.computed_resource_claim, '$.ram')
            ),
            BigInteger,
        )
    elif dialect == 'postgresql':
        vram_values = func.json_each_text(
            case(
                (
                    func.json_typeof(ModelInstance.computed_resource_claim['vram'])
                    == text("'object'"),
                    ModelInstance.computed_resource_claim['vram'],
                ),
                else_=cast({"0": 0}, JSON),
            )
        ).table_valued('value')

        ram_claim = func.cast(
            func.coalesce(
                func.json_extract_path_text(
                    ModelInstance.computed_resource_claim, 'ram'
                ),
                '0',
            ),
            BigInteger,
        )
    else:
        raise NotImplementedError(f'Unsupported database {dialect}')

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

    resource_claim_query = (
        select(
            ModelInstance.model_id,
            func.sum(func.coalesce(ram_claim, 0)).label('total_ram_claim'),
            func.sum(
                func.coalesce(func.cast(vram_values.c.value, BigInteger), 0)
            ).label('total_vram_claim'),
        ).group_by(ModelInstance.model_id)
    ).alias('resource_claim')

    statement = (
        select(
            Model.id,
            Model.name,
            Model.categories,
            func.count(distinct(ModelInstance.id)).label('instance_count'),
            func.coalesce(resource_claim_query.c.total_ram_claim, 0).label(
                'total_ram_claim'
            ),
            func.coalesce(resource_claim_query.c.total_vram_claim, 0).label(
                'total_vram_claim'
            ),
            usage_sum_query.c.total_token_count,
        )
        .join(ModelInstance, Model.id == ModelInstance.model_id)
        .outerjoin(
            resource_claim_query,
            Model.id == resource_claim_query.c.model_id,
        )
        .outerjoin(usage_sum_query, Model.id == usage_sum_query.c.model_id)
        .group_by(
            Model.id,
            usage_sum_query.c.total_token_count,
            resource_claim_query.c.total_ram_claim,
            resource_claim_query.c.total_vram_claim,
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
