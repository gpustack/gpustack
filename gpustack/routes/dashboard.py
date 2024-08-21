from datetime import date, datetime, timedelta, timezone
from typing import List
from fastapi import APIRouter
from sqlalchemy import Integer
from sqlmodel import distinct, select, func
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.dashboard import (
    CurrentSystemLoad,
    HistorySystemLoad,
    ModelSummary,
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
from gpustack.schemas import Worker
from gpustack.server.system_load import compute_system_load

router = APIRouter()


@router.get("")
async def dashboard(
    session: SessionDep,
):
    resoruce_counts = await get_resource_counts(session)
    system_load = await get_system_load(session)
    model_usage = await get_model_usage(session)
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
    memory = []
    gpu = []
    gpu_memory = []
    for system_load in system_loads:
        cpu.append(
            TimeSeriesData(
                timestamp=system_load.timestamp,
                value=system_load.cpu,
            )
        )
        memory.append(
            TimeSeriesData(
                timestamp=system_load.timestamp,
                value=system_load.memory,
            )
        )
        gpu.append(
            TimeSeriesData(
                timestamp=system_load.timestamp,
                value=system_load.gpu,
            )
        )
        gpu_memory.append(
            TimeSeriesData(
                timestamp=system_load.timestamp,
                value=system_load.gpu_memory,
            )
        )

    return SystemLoadSummary(
        current=CurrentSystemLoad(
            cpu=current_system_load.cpu,
            memory=current_system_load.memory,
            gpu=current_system_load.gpu,
            gpu_memory=current_system_load.gpu_memory,
        ),
        history=HistorySystemLoad(
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            gpu_memory=gpu_memory,
        ),
    )


async def get_model_usage(session: AsyncSession) -> ModelUsageSummary:
    today = date.today()
    one_week_ago = today - timedelta(days=7)

    statement = (
        select(
            ModelUsage.date,
            func.sum(ModelUsage.prompt_token_count).label('total_prompt_tokens'),
            func.sum(ModelUsage.completion_token_count).label(
                'total_completion_tokens'
            ),
            func.sum(ModelUsage.request_count).label('total_requests'),
        )
        .where(ModelUsage.date >= one_week_ago)
        .group_by(ModelUsage.date)
        .order_by(ModelUsage.date)
    )
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
    # get top users
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
        .where(ModelUsage.date >= one_week_ago)
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
        api_request_history=api_request_history,
        prompt_token_history=prompt_token_history,
        completion_token_history=completion_token_history,
        top_users=top_users,
    )


async def get_active_models(session: AsyncSession) -> List[ModelSummary]:
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

    statement = (
        select(
            Model.id,
            Model.name,
            func.count(distinct(ModelInstance.id)).label('instance_count'),
            func.sum(
                func.coalesce(
                    func.cast(
                        func.json_extract(
                            ModelInstance.computed_resource_claim, '$.memory'
                        ),
                        Integer,
                    ),
                    0,
                )
            ).label('total_memory_claim'),
            func.sum(
                func.coalesce(
                    func.cast(
                        func.json_extract(
                            ModelInstance.computed_resource_claim, '$.gpu_memory'
                        ),
                        Integer,
                    ),
                    0,
                )
            ).label('total_gpu_memory_claim'),
            usage_sum_query.c.total_token_count,
        )
        .join(ModelInstance, Model.id == ModelInstance.model_id)
        .outerjoin(usage_sum_query, Model.id == usage_sum_query.c.model_id)
        .group_by(Model.id, usage_sum_query.c.total_token_count)
        .order_by(usage_sum_query.c.total_token_count.desc())
        .limit(10)
    )
    results = (await session.exec(statement)).all()
    model_summary = []
    for result in results:
        model_summary.append(
            ModelSummary(
                id=result.id,
                name=result.name,
                resource_claim=ResourceClaim(
                    memory=result.total_memory_claim,
                    gpu_memory=result.total_gpu_memory_claim,
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
