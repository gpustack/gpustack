import asyncio
import logging
import time
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.model_routes import ModelRoute
from gpustack.schemas.model_usage import ModelUsage, OperationEnum
from gpustack.schemas.model_usage_details import ModelUsageDetails
from gpustack.schemas.models import Model, is_embedding_model, is_reranker_model
from gpustack.schemas.principals import Principal
from gpustack.server.db import async_session
from gpustack.utils.usage_snapshots import build_model_usage_snapshot

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_SECONDS = 10

# Heuristics for partial-stream usage estimation. The proxy never applies
# these ratios itself — they kick in server-side only when an incomplete
# report leaves token fields blank. Tunable via env (see ``gpustack.envs``).

# Buffer to accumulate pushed gateway metrics: {key: ModelUsageMetrics}.
# Key format (see ``_make_buffer_key``):
#   "{model_id}.{provider_id}.{model}.{user_id}.{access_key}.{operation}.{date}"
# ``operation`` and ``date`` are part of the key so per-operation rollups
# stay separate and a stream that crosses midnight lands in the period
# it ends in (anchored on completed_at).
gateway_metrics_buffer: Dict[str, "ModelUsageMetrics"] = {}
# Raw per-report metrics retained for ``model_usage_details`` audit rows.
# Unlike ``gateway_metrics_buffer``, entries are not aggregated.
gateway_details_buffer: List["ModelUsageMetrics"] = []
# Single lock guarding both rollup and details buffers; ingest writes
# them together, so they must be drained together too.
gateway_buffers_lock = asyncio.Lock()

# Throttle state for the "missing model_route_id" warning. A regressed
# gateway can emit many such rows per second; collapse to one warning per
# interval with the suppressed count so logs stay readable without losing
# the signal.
_MISSING_ROUTE_ID_WARN_INTERVAL_SECONDS = 60.0
_missing_route_id_last_warned_at: float = 0.0
_missing_route_id_suppressed: int = 0


class ModelUsageMetrics(BaseModel):
    model: str
    input_token: int = 0
    output_token: int = 0
    total_token: int = 0
    input_cached_token: int = 0
    request_count: int = 1
    # ``completed`` is True iff the canonical usage chunk was observed before
    # the stream ended. When False, token fields may be 0 (OpenAI/vLLM) or
    # partial (Anthropic message_start carries input_token early), so the
    # server falls back to estimation from the byte/chunk fields below.
    completed: bool = False
    output_chunk_count: int = 0
    request_content_bytes: int = 0
    # Wall-clock UnixMilli stamps captured at request entry and report
    # dispatch respectively. ``None`` means the report didn't carry one;
    # legacy payloads sending literal ``0`` are also treated as absent
    # downstream (see ``_unixmilli_to_naive_utc``).
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    user_id: Optional[int] = None
    model_id: Optional[int] = None
    model_route_id: Optional[int] = None
    # Captured at request time by middleware (request.state.model.cluster_id).
    # Carried on the metric so the historical cluster id survives even if the
    # model is deleted between request and flush.
    cluster_id: Optional[int] = None
    provider_id: Optional[int] = None
    provider_name: Optional[str] = None
    provider_type: Optional[str] = None
    access_key: Optional[str] = None
    # Inference operation type (chat_completion / embedding / rerank / ...).
    # None when the gateway report doesn't carry it; middleware-fed metrics
    # always populate it so per-operation rollups survive unification.
    operation: Optional[OperationEnum] = None
    # Tenant identifier sourced from the gateway's X-Organization-Id header
    # (configurable via the token-usage plugin's ``organizationIDHeader``).
    organization_id: Optional[str] = None


def _unixmilli_to_naive_utc(ms: Optional[int]) -> Optional[datetime]:
    """Convert a UnixMilli stamp to naive UTC, or None if absent / non-positive.

    Accepts ``None`` (current absence sentinel) and ``<= 0`` (legacy absence
    sentinel that some older gateway payloads still send) — both collapse to
    ``None``. The naive-UTC convention matches ``TimestampsMixin._datetime_func``
    and the ``UTCDateTime`` storage type, which both strip tzinfo.
    """
    if ms is None or ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def _resolve_metric_datetime(
    metric: ModelUsageMetrics,
) -> Tuple[date, datetime]:
    """Resolve (date, naive-UTC datetime) anchored on the metric's wall-clock.

    Prefers ``completed_at`` so a stream that crosses a calendar boundary
    lands in the period it ends in (per the proxy contract). Falls back to
    ``started_at`` and finally to server ``now`` if both are absent.
    """
    dt = (
        _unixmilli_to_naive_utc(metric.completed_at)
        or _unixmilli_to_naive_utc(metric.started_at)
        or datetime.now(timezone.utc).replace(tzinfo=None)
    )
    return dt.date(), dt


def _make_buffer_key(metric: ModelUsageMetrics) -> str:
    # Include the completion-anchored date so streams that cross midnight
    # accumulate into the correct billing-period rollup instead of being
    # merged with the next day's traffic. ``organization_id`` is included
    # to match the DB upsert key in ``create_or_update_model_usage`` —
    # otherwise the same user calling from different Org contexts within
    # one flush window would merge in memory but split on write, losing
    # tokens.
    metric_date, _ = _resolve_metric_datetime(metric)
    operation = metric.operation.value if metric.operation else ""
    return ".".join(
        str(part or "")
        for part in [
            metric.model_id,
            metric.provider_id,
            metric.model,
            metric.user_id,
            metric.access_key,
            metric.organization_id,
            metric.model_route_id,
            operation,
            metric_date.isoformat(),
        ]
    )


def _estimate_partial_usage(metric: ModelUsageMetrics) -> None:
    """Backfill input_token / output_token for incomplete reports in place.

    Only fills slots that are still empty so that legitimate partial values
    (e.g. Anthropic's early ``input_token``) survive untouched. Estimation
    is intentionally a server-side concern — the proxy never applies these
    ratios itself.
    """
    if metric.completed:
        return
    if metric.input_token <= 0 and metric.request_content_bytes > 0:
        metric.input_token = max(
            1,
            metric.request_content_bytes // envs.USAGE_ESTIMATED_BYTES_PER_INPUT_TOKEN,
        )
    if metric.output_token <= 0 and metric.output_chunk_count > 0:
        metric.output_token = (
            metric.output_chunk_count * envs.USAGE_ESTIMATED_TOKENS_PER_OUTPUT_CHUNK
        )
    estimated_total = metric.input_token + metric.output_token
    if metric.total_token < estimated_total:
        metric.total_token = estimated_total


def _resolve_usage_tokens(
    metric: ModelUsageMetrics, model: Optional[Model]
) -> tuple[int, int]:
    prompt_tokens = metric.input_token
    completion_tokens = metric.output_token
    if (
        model is not None
        and (is_reranker_model(model) or is_embedding_model(model))
        and metric.total_token > (prompt_tokens + completion_tokens)
    ):
        return metric.total_token - completion_tokens, completion_tokens
    return prompt_tokens, completion_tokens


async def accumulate_gateway_metrics(metrics: List[ModelUsageMetrics]):
    async with gateway_buffers_lock:
        # Product invariant: every inference request resolves to a route
        # (the gateway can't dispatch otherwise). A NULL model_route_id
        # means an ingest source regressed — the row will pollute the
        # "Untracked" bucket reserved for pre-upgrade legacy data. Tally
        # per-batch and surface via a throttled summary so a regressed
        # gateway doesn't flood logs.
        missing_route_id_count = 0
        missing_route_id_sample: Optional[ModelUsageMetrics] = None
        for incoming in metrics:
            if incoming.model_route_id is None:
                missing_route_id_count += 1
                if missing_route_id_sample is None:
                    missing_route_id_sample = incoming
            # Take ownership before any in-place work:
            #   * ``_estimate_partial_usage`` mutates token fields directly.
            #   * The rollup buffer's ``+=`` mutates the stored entry, which
            #     would also mutate the caller's instance (and bleed into the
            #     details audit row) if we shared references.
            # One copy at the top + one for details keeps both buffers, the
            # caller, and the audit trail isolated from one another.
            metric = incoming.model_copy()
            # Backfill estimated tokens before either buffer sees the metric:
            # the rollup buffer aggregates by += and would otherwise lose the
            # per-row byte/chunk context needed for estimation later on.
            _estimate_partial_usage(metric)
            gateway_details_buffer.append(metric.model_copy())
            key = _make_buffer_key(metric)
            existing = gateway_metrics_buffer.get(key)
            if existing is None:
                gateway_metrics_buffer[key] = metric
            else:
                existing.input_token += metric.input_token
                existing.output_token += metric.output_token
                existing.total_token += metric.total_token
                existing.input_cached_token += metric.input_cached_token
                existing.request_count += metric.request_count
        _trim_details_buffer_locked()
        _maybe_warn_missing_route_id(missing_route_id_count, missing_route_id_sample)


def _trim_details_buffer_locked() -> None:
    """Cap ``gateway_details_buffer`` to bound memory under persistent flush
    failure.

    The flush failure path re-accumulates pending details so transient errors
    don't lose the audit trail, but persistent failures (DB down, schema
    drift, constraint violation) would let the buffer grow unbounded as new
    ingest piles on. Drop oldest entries past the cap and log once per
    overflow event so operators notice. Caller must hold
    ``gateway_buffers_lock``.
    """
    cap = envs.USAGE_DETAILS_BUFFER_MAX_SIZE
    overflow = len(gateway_details_buffer) - cap
    if overflow <= 0:
        return
    del gateway_details_buffer[:overflow]
    logger.warning(
        "gateway_details_buffer exceeded cap (%d); dropped %d oldest detail "
        "rows. Likely cause: persistent flush failure to model_usage_details.",
        cap,
        overflow,
    )


def _maybe_warn_missing_route_id(
    batch_missing: int, sample: Optional[ModelUsageMetrics]
) -> None:
    """Emit at most one 'missing model_route_id' warning per interval.

    Accumulates the suppressed count across batches and flushes it the next
    time the interval has elapsed, so a regressed gateway shows up as one
    log line per minute with the running total rather than thousands of
    near-identical lines. Caller must hold ``gateway_buffers_lock``.
    """
    global _missing_route_id_last_warned_at, _missing_route_id_suppressed
    if batch_missing <= 0:
        return
    _missing_route_id_suppressed += batch_missing
    now = time.monotonic()
    if now - _missing_route_id_last_warned_at < _MISSING_ROUTE_ID_WARN_INTERVAL_SECONDS:
        return
    total = _missing_route_id_suppressed
    _missing_route_id_last_warned_at = now
    _missing_route_id_suppressed = 0
    if sample is not None:
        logger.warning(
            "Gateway metrics missing model_route_id (%d in the last ~%ds); "
            "rows land in the Untracked bucket. Sample: model=%s user_id=%s "
            "access_key=%s",
            total,
            int(_MISSING_ROUTE_ID_WARN_INTERVAL_SECONDS),
            sample.model,
            sample.user_id,
            sample.access_key,
        )
    else:
        logger.warning(
            "Gateway metrics missing model_route_id (%d in the last ~%ds); "
            "rows land in the Untracked bucket.",
            total,
            int(_MISSING_ROUTE_ID_WARN_INTERVAL_SECONDS),
        )


async def flush_gateway_metrics():
    async with gateway_buffers_lock:
        if not gateway_metrics_buffer and not gateway_details_buffer:
            return
        pending_rollups = list(gateway_metrics_buffer.values())
        pending_details = list(gateway_details_buffer)
        gateway_metrics_buffer.clear()
        gateway_details_buffer.clear()

    try:
        await store_usage_metrics(pending_rollups, pending_details)
    except Exception as e:
        logger.error(f"Error flushing gateway metrics to DB: {e}")
        # Re-buffering raw details restores both buffers via the same
        # aggregation logic as the original ingest path.
        await accumulate_gateway_metrics(pending_details)


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
            "model_route_id": metric.model_route_id,
            "access_key": metric.access_key,
            "operation": metric.operation,
            "consumer_principal_id": metric.consumer_principal_id,
            "date": metric.date,
        },
    )
    if current_usage is None:
        await metric.save(session=session, auto_commit=auto_commit)
    else:
        current_usage.prompt_token_count += metric.prompt_token_count
        current_usage.completion_token_count += metric.completion_token_count
        current_usage.prompt_cached_token_count += metric.prompt_cached_token_count
        current_usage.request_count += metric.request_count
        # Refresh route name snapshot to the latest non-NULL value so that
        # a mid-day rename converges to one consistent label per (route_id,
        # date) cell. A NULL incoming name means the route was deleted
        # between dispatch and flush — keep the existing snapshot rather
        # than wiping a still-meaningful audit label.
        if metric.model_route_name is not None:
            current_usage.model_route_name = metric.model_route_name
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


def _build_metric_snapshot(
    metric: ModelUsageMetrics,
    model_by_id: Dict[int, Model],
    provider_by_id: Dict[int, ModelProvider],
    user_by_id: Dict[int, Principal],
    api_key_by_access_key: Dict[str, ApiKey],
    cluster_names_by_id: Dict[int, str],
    route_name_by_id: Dict[int, str],
) -> dict:
    user = user_by_id.get(metric.user_id)
    api_key = api_key_by_access_key.get(metric.access_key)
    model = model_by_id.get(metric.model_id)
    provider = provider_by_id.get(metric.provider_id)
    # Route name is resolved from the live table; falls back to None when
    # the route was deleted between dispatch and flush. The id is always
    # preserved verbatim so audit/breakdown can still attribute the row.
    model_route_id = metric.model_route_id
    model_route_name = (
        route_name_by_id.get(model_route_id) if model_route_id is not None else None
    )
    if model is None:
        snapshot = {
            "model_id": metric.model_id,
            "model_name": metric.model,
            "cluster_name": None,
        }
        if provider is not None:
            provider_type = getattr(getattr(provider, "config", None), "type", None)
            if provider_type is not None and hasattr(provider_type, "value"):
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
                    "user_name": user.name,
                }
            )
        if api_key is not None:
            snapshot.update(
                {
                    "api_key_id": api_key.id,
                    "api_key_name": api_key.name,
                    "access_key": api_key.access_key,
                    "api_key_is_custom": api_key.is_custom,
                    "consumer_principal_id": api_key.owner_principal_id,
                }
            )
        if model_route_id is not None:
            snapshot["model_route_id"] = model_route_id
            snapshot["model_route_name"] = model_route_name
        # The "model deleted before flush" branch can no longer source
        # tenant scope from ``model.owner_principal_id``. Fall back to the
        # api_key's owner so cross-tenant attribution still works in this
        # edge case — under the route/target same-Org rule, api_key owner
        # is necessarily aligned with the route's (and the deleted model's)
        # Org for any non-platform deployment.
        if api_key is not None:
            snapshot["owner_principal_id"] = getattr(
                api_key, "owner_principal_id", None
            )
    else:
        snapshot = build_model_usage_snapshot(
            model,
            cluster_name=cluster_names_by_id.get(model.cluster_id),
            user=user,
            api_key=api_key,
            provider=provider,
            model_route_id=model_route_id,
            model_route_name=model_route_name,
        )
    snapshot.setdefault("user_id", metric.user_id)
    snapshot.setdefault("provider_id", metric.provider_id)
    snapshot.setdefault("provider_name", metric.provider_name)
    snapshot.setdefault("provider_type", metric.provider_type)
    snapshot.setdefault("access_key", metric.access_key)
    snapshot.setdefault("api_key_is_custom", None)
    # The api_key path above stamps ``consumer_principal_id`` from
    # ``api_key.owner_principal_id`` whenever a key is present. For
    # cookie-authed traffic (no api_key) the gateway plugin still
    # provides the active tenant via the wire-format
    # ``organization_id`` header — parse it back to int so the row
    # carries its Org scope. Direct-to-gpustack cookie calls don't
    # populate this field and land NULL; that's by design (no Org
    # context known at that path).
    if "consumer_principal_id" not in snapshot and metric.organization_id:
        try:
            snapshot["consumer_principal_id"] = int(metric.organization_id)
        except (TypeError, ValueError):
            pass
    return snapshot


async def store_usage_metrics(
    metrics: List[ModelUsageMetrics],
    detail_metrics: Optional[List[ModelUsageMetrics]] = None,
):
    detail_metrics = list(detail_metrics or [])
    if not metrics and not detail_metrics:
        return

    all_metrics = list(metrics) + detail_metrics
    dedup_model_names = {m.model for m in all_metrics}
    dedup_user_ids = {m.user_id for m in all_metrics if m.user_id is not None}
    dedup_access_keys = {m.access_key for m in all_metrics if m.access_key is not None}
    dedup_provider_ids = {
        m.provider_id for m in all_metrics if m.provider_id is not None
    }
    dedup_route_ids = {
        m.model_route_id for m in all_metrics if m.model_route_id is not None
    }
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
            # Generic principal lookup — the caller of a model_usage
            # record can be a USER (human via login + API key) or a
            # SYSTEM principal (worker proxying inference). No kind
            # filter, no User-shape implication.
            users = await Principal.all_by_fields(
                session=session,
                fields={},
                extra_conditions=[Principal.id.in_(dedup_user_ids)],
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
            route_name_by_id: Dict[int, str] = {}
            if dedup_route_ids:
                routes = await ModelRoute.all_by_fields(
                    session=session,
                    fields={},
                    extra_conditions=[ModelRoute.id.in_(dedup_route_ids)],
                )
                route_name_by_id = {r.id: r.name for r in routes}
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
                snapshot = _build_metric_snapshot(
                    metric,
                    model_by_id,
                    provider_by_id,
                    user_by_id,
                    api_key_by_access_key,
                    cluster_names_by_id,
                    route_name_by_id,
                )
                prompt_tokens, completion_tokens = _resolve_usage_tokens(
                    metric, model_by_id.get(metric.model_id)
                )
                metric_date, _ = _resolve_metric_datetime(metric)
                model_usage = ModelUsage(
                    date=metric_date,
                    prompt_token_count=prompt_tokens,
                    completion_token_count=completion_tokens,
                    prompt_cached_token_count=metric.input_cached_token,
                    request_count=metric.request_count,
                    operation=metric.operation,
                    **snapshot,
                )
                await create_or_update_model_usage(
                    session, model_usage, auto_commit=False
                )

            for metric in detail_metrics:
                if not _validate_usage_metric(
                    metric, model_by_id, provider_by_id, validated_user_ids
                ):
                    continue
                snapshot = _build_metric_snapshot(
                    metric,
                    model_by_id,
                    provider_by_id,
                    user_by_id,
                    api_key_by_access_key,
                    cluster_names_by_id,
                    route_name_by_id,
                )
                prompt_tokens, completion_tokens = _resolve_usage_tokens(
                    metric, model_by_id.get(metric.model_id)
                )
                # cluster_id only lives on the audit/details rows, not on
                # the dashboard rollup (ModelUsage). Prefer the metric's
                # own cluster_id (captured at request time, survives model
                # deletes); fall back to the live model only when the
                # ingest source didn't carry one (older gateway clients).
                cluster_id = metric.cluster_id
                if cluster_id is None:
                    cluster_id = getattr(
                        model_by_id.get(metric.model_id), "cluster_id", None
                    )
                metric_date, metric_dt = _resolve_metric_datetime(metric)
                started_dt = _unixmilli_to_naive_utc(metric.started_at)
                completed_dt = _unixmilli_to_naive_utc(metric.completed_at)
                session.add(
                    ModelUsageDetails(
                        date=metric_date,
                        cluster_id=cluster_id,
                        prompt_token_count=prompt_tokens,
                        completion_token_count=completion_tokens,
                        prompt_cached_token_count=metric.input_cached_token,
                        operation=metric.operation,
                        # Proxy-reported wall-clock — preserved as NULL when
                        # the report didn't carry it, so reconciliation jobs
                        # can tell estimated rows apart from authoritative
                        # ones.
                        started_at=started_dt,
                        completed_at=completed_dt,
                        # Audit timestamps still pinned to the request's
                        # wall-clock so the row's lifecycle stamps don't
                        # drift by the flush interval.
                        created_at=metric_dt,
                        updated_at=metric_dt,
                        **snapshot,
                    )
                )

            await session.commit()
        except Exception as e:
            logger.exception(f"Error storing gateway metrics: {e}")
            await session.rollback()
            # Propagate so flush_gateway_metrics can re-buffer the pending
            # records — without this, a transactional rollback silently
            # drops a flush window's worth of audit rows.
            raise
