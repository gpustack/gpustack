import logging
import asyncio
import copy
from datetime import date
from aiohttp import ClientSession as aiohttp_client, ClientTimeout
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from typing import Optional, Dict, List, Set
from dataclasses import dataclass
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample
from gpustack.server.db import async_session
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.schemas.model_provider import ModelProvider
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack import envs


gateway_metrics_port = 15020

logger = logging.getLogger(__name__)

common_prefix = "route_upstream_model_consumer_metric_"
count_sample_suffix = "_total"

metrics_names = {
    "route_upstream_model_consumer_metric_input_token": "input_token",
    "route_upstream_model_consumer_metric_output_token": "output_token",
    "route_upstream_model_consumer_metric_total_token": "total_token",
    "route_upstream_model_consumer_metric_llm_duration_count": "request_count",
}


@dataclass
class ModelUsageMetrics:
    model: str
    input_token: int = 0
    output_token: int = 0
    total_token: int = 0
    request_count: int = 0
    user_id: Optional[int] = None
    model_id: Optional[int] = None
    provider_id: Optional[int] = None
    access_key: Optional[str] = None


# here is the example metrics to parse
# route_upstream_model_consumer_metric_input_token{ai_route="ai-route-model-1",ai_cluster="outbound|80||model-1-1.static",ai_model="qwen3-0.6b",ai_consumer="d720eeb5b57fbe94.gpustack-2"} 156
# route_upstream_model_consumer_metric_llm_duration_count{ai_route="ai-route-model-1",ai_cluster="outbound|80||model-1-1.static",ai_model="qwen3-0.6b",ai_consumer="d720eeb5b57fbe94.gpustack-2"} 13
# route_upstream_model_consumer_metric_llm_service_duration{ai_route="ai-route-model-1",ai_cluster="outbound|80||model-1-1.static",ai_model="qwen3-0.6b",ai_consumer="d720eeb5b57fbe94.gpustack-2"} 9279
# route_upstream_model_consumer_metric_output_token{ai_route="ai-route-model-1",ai_cluster="outbound|80||model-1-1.static",ai_model="qwen3-0.6b",ai_consumer="d720eeb5b57fbe94.gpustack-2"} 1755
# route_upstream_model_consumer_metric_total_token{ai_route="ai-route-model-1",ai_cluster="outbound|80||model-1-1.static",ai_model="qwen3-0.6b",ai_consumer="d720eeb5b57fbe94.gpustack-2"} 1911


def parse_token_metrics(metrics_text) -> Dict[str, ModelUsageMetrics]:
    metrics_by_model_user_access_key: Dict[str, ModelUsageMetrics] = {}
    for family in text_string_to_metric_families(metrics_text):
        if family.name not in metrics_names:
            continue
        for sample in family.samples:
            metrics = parse_sample_label_to_usage(sample)
            if metrics is None:
                continue
            key = ".".join(
                [
                    str(part or "")
                    for part in [
                        metrics.model_id,
                        metrics.provider_id,
                        metrics.model,
                        metrics.user_id,
                        metrics.access_key,
                    ]
                ]
            )
            existing_metrics = metrics_by_model_user_access_key.get(key, None)
            if existing_metrics is None:
                metrics_by_model_user_access_key[key] = metrics
            else:
                if metrics.input_token:
                    existing_metrics.input_token = metrics.input_token
                if metrics.output_token:
                    existing_metrics.output_token = metrics.output_token
                if metrics.total_token:
                    existing_metrics.total_token = metrics.total_token
                if metrics.request_count:
                    existing_metrics.request_count = metrics.request_count
    return metrics_by_model_user_access_key


def parse_sample_label_to_usage(sample: Sample) -> Optional[ModelUsageMetrics]:
    attr = metrics_names.get(sample.name.removesuffix(count_sample_suffix), None)
    if attr is None:
        logger.debug(f"Unknown metric name: {sample.name}, skipping sample: {sample}")
        return None

    labels = sample.labels
    model = labels.get("ai_model", None)
    consumer = labels.get("ai_consumer", None)
    user_id: Optional[int] = None
    access_key: Optional[str] = None
    if consumer is not None and consumer != "none":
        consumer_parts = consumer.split(".")
        user = consumer_parts[-1]
        if user.startswith("gpustack-"):
            user_id = int(user[len("gpustack-") :])
        if len(consumer_parts) == 2:
            access_key = consumer_parts[0]
    ai_cluster = labels.get("ai_cluster", None)
    model_id = None
    provider_id = None
    if ai_cluster is not None:
        cluster_parts = ai_cluster.split("|", 3)
        if len(cluster_parts) != 4:
            logger.debug(
                f"Unexpected ai_cluster format: {ai_cluster}, expected 4 parts separated by '|', skipping sample: {sample}"
            )
            return None
        target = cluster_parts[3]
        if target.startswith("model-"):
            # extra id `1` from model-1-2.static
            model_split = target.split("-", 2)
            if len(model_split) == 3:
                try:
                    model_id = int(model_split[1])
                except ValueError:
                    logger.debug(f"Invalid model_id in ai_cluster target: {target}")
        if target.startswith("provider-"):
            # extra id `1` from provider-1.dns
            provider_split = target.removeprefix("provider-").split(".", 1)
            if len(provider_split) == 2:
                try:
                    provider_id = int(provider_split[0])
                except ValueError:
                    logger.debug(f"Invalid provider_id in ai_cluster target: {target}")

    value = int(sample.value)
    rtn = ModelUsageMetrics(
        model=model,
        model_id=model_id,
        provider_id=provider_id,
        user_id=user_id,
        access_key=access_key,
    )
    setattr(rtn, attr, value)

    return rtn


async def create_or_update_model_usage(
    session: AsyncSession, metric: ModelUsage, auto_commit: bool = True
):
    current_usage = await ModelUsage.one_by_fields(
        session=session,
        fields={
            "model_id": metric.model_id,
            "user_id": metric.user_id,
            "provider_id": metric.provider_id,
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


class GatewayMetricsCollector:
    _interval: int
    _config: Config
    _disabled_collection: bool = False
    _client: aiohttp_client
    _embedded_gateway_metrics_url: str = (
        f"http://127.0.0.1:{gateway_metrics_port}/stats/prometheus"
    )
    cached_dict: Dict[str, ModelUsageMetrics] = {}

    def __init__(self, cfg: Config, interval=60):
        self._interval = interval
        self._config = cfg
        self._disabled_collection = cfg.gateway_mode not in [
            GatewayModeEnum.embedded,
            GatewayModeEnum.external,
        ]
        self._client = aiohttp_client(timeout=ClientTimeout(total=10))
        if (
            cfg.gateway_mode == GatewayModeEnum.external
            and envs.GATEWAY_EXTERNAL_METRICS_URL is None
        ):
            logger.warning(
                "Gateway is in external mode but GPUSTACK_GATEWAY_EXTERNAL_METRICS_URL is not set, skipped metrics collection."
            )
            self._disabled_collection = True

    @property
    def gateway_metrics_url(self):
        if self._config.gateway_mode == GatewayModeEnum.external:
            return envs.GATEWAY_EXTERNAL_METRICS_URL
        return self._embedded_gateway_metrics_url

    def _metrics_delta(
        self, metrics: Dict[str, ModelUsageMetrics]
    ) -> List[ModelUsageMetrics]:
        """
        Calculate the delta (increment) of model usage metrics since the last collection.

        For each key (model_id+provider_id+model+user+access_key), compare the current metrics with the cached previous metrics.
        - If cached exists, subtract previous values to get the delta.
        - If no cache, use the current value as the delta.
        - If all delta values are zero, skip (no change).
        - If any delta is negative, likely due to gateway restart or metrics reset, reset cache and skip reporting.
        - Otherwise, append the delta metrics and update the cache.

        Args:
            metrics (Dict[str, ModelUsageMetrics]): Current metrics snapshot, keyed by model_id+provider_id+model+user+access_key.

        Returns:
            List[ModelUsageMetrics]: List of delta metrics to be stored/reported.
        """
        rtn: List[ModelUsageMetrics] = []
        for key, metric in metrics.items():
            cached_metric = self.cached_dict.get(key, None)
            copied_metric = copy.deepcopy(metric)
            if cached_metric is not None:
                # Subtract previous values to get the delta
                copied_metric.input_token -= cached_metric.input_token
                copied_metric.output_token -= cached_metric.output_token
                copied_metric.total_token -= cached_metric.total_token
                copied_metric.request_count -= cached_metric.request_count
            # Skip if all delta values are zero (no change)
            if (
                copied_metric.input_token == 0
                and copied_metric.output_token == 0
                and copied_metric.request_count == 0
            ):
                continue
            # If any delta is negative, likely due to gateway restart or metrics reset
            if (
                copied_metric.input_token < 0
                or copied_metric.output_token < 0
                or copied_metric.request_count < 0
            ):
                # Reset cache to current metric, skip reporting this round
                self.cached_dict[key] = metric
                logger.warning(
                    f"Negative delta metrics detected for key {key}, resetting cache."
                )
                continue
            # Valid delta, append to result and update cache
            rtn.append(copied_metric)
            self.cached_dict[key] = metric
        return rtn

    def _validate_metrics(
        self,
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

    async def _store_metrics(self, metrics: List[ModelUsageMetrics]):
        dedup_model_names = {m.model for m in metrics}
        dedup_user_ids = {m.user_id for m in metrics if m.user_id is not None}
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
                )
                users = await User.all_by_fields(
                    session=session,
                    fields={},
                    extra_conditions=[User.id.in_(dedup_user_ids)],
                )
                validated_user_ids = {u.id for u in users}
                model_by_id = {m.id: m for m in models}
                provider_by_id = {p.id: p for p in providers}
                for metric in metrics:
                    if not self._validate_metrics(
                        metric, model_by_id, provider_by_id, validated_user_ids
                    ):
                        continue
                    model_usage = ModelUsage(
                        model_id=metric.model_id,
                        provider_id=metric.provider_id,
                        model_name=metric.model,
                        user_id=metric.user_id,
                        access_key=metric.access_key,
                        date=date.today(),
                        prompt_token_count=metric.input_token,
                        completion_token_count=metric.output_token,
                        request_count=metric.request_count,
                    )
                    await create_or_update_model_usage(
                        session, model_usage, auto_commit=False
                    )
                await session.commit()
            except Exception as e:
                logger.exception(f"Error storing gateway metrics: {e}")
                await session.rollback()

    async def start(self):
        if self._disabled_collection:
            return

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        async def retry_connect() -> str:
            async with self._client.get(self.gateway_metrics_url) as resp:
                if resp.status != 200:
                    raise ConnectionError(
                        f"Failed to connect to gateway metrics endpoint, status: {resp.status}"
                    )
                return await resp.text()

        while True:
            try:
                logger.debug(
                    "Collecting gateway metrics from %s",
                    self.gateway_metrics_url,
                )
                text = await retry_connect()
                metrics = parse_token_metrics(text)
                delta_metrics = self._metrics_delta(metrics)
                for m in delta_metrics:
                    logger.debug("Delta metric: %s", m)
                if len(delta_metrics) != 0:
                    logger.debug("Storing delta metrics to database...")
                    await self._store_metrics(delta_metrics)
                    logger.debug("Delta metrics stored successfully.")
            except Exception as e:
                logger.exception(f"Error collecting gateway metrics: {e}")
            await asyncio.sleep(self._interval)
