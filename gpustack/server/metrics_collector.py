import asyncio
import logging
from datetime import date
from typing import Dict, List, Optional, Set

from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.server.db import async_session
from gpustack.utils.usage_snapshots import build_model_usage_snapshot

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_SECONDS = 10

# Buffer to accumulate pushed gateway metrics: {key: ModelUsageMetrics}
# Key format: "{model_id}.{provider_id}.{model}.{user_id}.{access_key}"
gateway_metrics_buffer: Dict[str, "ModelUsageMetrics"] = {}
gateway_metrics_buffer_lock = asyncio.Lock()


class ModelUsageMetrics(BaseModel):
    model: str
    input_token: int = 0
    output_token: int = 0
    total_token: int = 0
    request_count: int = 1
    user_id: Optional[int] = None
    model_id: Optional[int] = None
    provider_id: Optional[int] = None
    provider_name: Optional[str] = None
    provider_type: Optional[str] = None
    access_key: Optional[str] = None


def _make_buffer_key(metric: ModelUsageMetrics) -> str:
    return ".".join(
        str(part or "")
        for part in [
            metric.model_id,
            metric.provider_id,
            metric.model,
            metric.user_id,
            metric.access_key,
        ]
    )


async def accumulate_gateway_metrics(metrics: List[ModelUsageMetrics]):
    async with gateway_metrics_buffer_lock:
        for metric in metrics:
            key = _make_buffer_key(metric)
            existing = gateway_metrics_buffer.get(key)
            if existing is None:
                gateway_metrics_buffer[key] = metric
            else:
                existing.input_token += metric.input_token
                existing.output_token += metric.output_token
                existing.total_token += metric.total_token
                existing.request_count += metric.request_count


async def flush_gateway_metrics():
    async with gateway_metrics_buffer_lock:
        if not gateway_metrics_buffer:
            return
        pending = list(gateway_metrics_buffer.values())
        gateway_metrics_buffer.clear()

    try:
        await store_usage_metrics(pending)
    except Exception as e:
        logger.error(f"Error flushing gateway metrics to DB: {e}")
        await accumulate_gateway_metrics(pending)


async def flush_gateway_metrics_to_db():
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
        await flush_gateway_metrics()


async def create_or_update_model_usage(
    session: AsyncSession, metric: ModelUsage, auto_commit: bool = True
):
    current_usage = await ModelUsage.one_by_fields(
        session=session,
        fields={
            "model_id": metric.model_id,
            "user_id": metric.user_id,
            "provider_id": metric.provider_id,
            "provider_name": metric.provider_name,
            "provider_type": metric.provider_type,
            "model_name": metric.model_name,
            "access_key": metric.access_key,
            "date": metric.date,
        },
    )
    if current_usage is None:
        await metric.save(session=session, auto_commit=auto_commit)
    else:
        current_usage.prompt_token_count += metric.prompt_token_count
        current_usage.completion_token_count += metric.completion_token_count
        current_usage.request_count += metric.request_count
        await current_usage.save(session=session, auto_commit=auto_commit)


def _validate_usage_metric(
    metric: ModelUsageMetrics,
    models: Dict[int, Model],
    providers: Dict[int, ModelProvider],
    user_ids: Set[int],
) -> bool:
    if metric.model_id is None and metric.provider_id is None:
        logger.debug(
            f"Both model_id and provider_id are None for metric: {metric}, skipping."
        )
        return False
    if metric.model_id is not None:
        model = models.get(metric.model_id)
        if not model:
            logger.debug(f"Model ID {metric.model_id} not found in database.")
            return False
        if model.name != metric.model:
            logger.debug(
                f"Model name {metric.model} does not match database record {model.name} for model ID {metric.model_id}."
            )
            return False
    if metric.provider_id is not None:
        provider = providers.get(metric.provider_id)
        if not provider:
            logger.debug(f"Provider ID {metric.provider_id} not found in database.")
            return False
        if metric.model not in {m.name for m in provider.models}:
            logger.debug(
                f"Model name {metric.model} not found for provider ID {metric.provider_id} in database."
            )
            return False
    if metric.user_id is not None and metric.user_id not in user_ids:
        logger.debug(f"User ID {metric.user_id} not found in database.")
        return False
    return True


async def store_usage_metrics(metrics: List[ModelUsageMetrics]):
    dedup_model_names = {m.model for m in metrics}
    dedup_user_ids = {m.user_id for m in metrics if m.user_id is not None}
    dedup_access_keys = {m.access_key for m in metrics if m.access_key is not None}
    dedup_provider_ids = {m.provider_id for m in metrics if m.provider_id is not None}
    async with async_session() as session:
        try:
            models = await Model.all_by_fields(
                session=session,
                fields={},
                extra_conditions=[Model.name.in_(dedup_model_names)],
            )
            providers = await ModelProvider.all_by_fields(
                session=session,
                fields={},
                extra_conditions=(
                    [ModelProvider.id.in_(dedup_provider_ids)]
                    if dedup_provider_ids
                    else []
                ),
            )
            users = await User.all_by_fields(
                session=session,
                fields={},
                extra_conditions=[User.id.in_(dedup_user_ids)],
            )
            api_keys = await ApiKey.all_by_fields(
                session=session,
                fields={},
                extra_conditions=(
                    [ApiKey.access_key.in_(dedup_access_keys)]
                    if dedup_access_keys
                    else []
                ),
            )
            validated_user_ids = {u.id for u in users}
            user_by_id = {u.id: u for u in users}
            api_key_by_access_key = {k.access_key: k for k in api_keys}
            model_by_id = {m.id: m for m in models}
            cluster_ids = {m.cluster_id for m in models if m.cluster_id is not None}
            clusters = await Cluster.all_by_fields(
                session=session,
                fields={},
                extra_conditions=([Cluster.id.in_(cluster_ids)] if cluster_ids else []),
            )
            cluster_names_by_id = {c.id: c.name for c in clusters}
            provider_by_id = {p.id: p for p in providers}
            for metric in metrics:
                if not _validate_usage_metric(
                    metric, model_by_id, provider_by_id, validated_user_ids
                ):
                    continue
                user = user_by_id.get(metric.user_id)
                api_key = api_key_by_access_key.get(metric.access_key)
                model = model_by_id.get(metric.model_id)
                provider = provider_by_id.get(metric.provider_id)
                if model is None:
                    snapshot = {
                        "model_id": metric.model_id,
                        "model_name": metric.model,
                        "cluster_name": None,
                    }
                    if provider is not None:
                        provider_type = getattr(
                            getattr(provider, "config", None), "type", None
                        )
                        if provider_type is not None and hasattr(
                            provider_type, "value"
                        ):
                            provider_type = provider_type.value
                        snapshot.update(
                            {
                                "provider_id": provider.id,
                                "provider_name": provider.name,
                                "provider_type": provider_type,
                            }
                        )
                    if user is not None:
                        snapshot.update(
                            {
                                "user_id": user.id,
                                "user_name": user.username,
                            }
                        )
                    if api_key is not None:
                        snapshot.update(
                            {
                                "api_key_id": api_key.id,
                                "api_key_name": api_key.name,
                                "access_key": api_key.access_key,
                                "api_key_is_custom": api_key.is_custom,
                            }
                        )
                else:
                    snapshot = build_model_usage_snapshot(
                        model,
                        cluster_name=cluster_names_by_id.get(model.cluster_id),
                        user=user,
                        api_key=api_key,
                        provider=provider,
                    )
                snapshot.setdefault("user_id", metric.user_id)
                snapshot.setdefault("provider_id", metric.provider_id)
                snapshot.setdefault("provider_name", metric.provider_name)
                snapshot.setdefault("provider_type", metric.provider_type)
                snapshot.setdefault("access_key", metric.access_key)
                snapshot.setdefault("api_key_is_custom", None)
                model_usage = ModelUsage(
                    date=date.today(),
                    prompt_token_count=metric.input_token,
                    completion_token_count=metric.output_token,
                    request_count=metric.request_count,
                    **snapshot,
                )
                await create_or_update_model_usage(
                    session, model_usage, auto_commit=False
                )
            await session.commit()
        except Exception as e:
            logger.exception(f"Error storing gateway metrics: {e}")
            await session.rollback()
