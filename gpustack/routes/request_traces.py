import asyncio
import base64
import gzip
import io
import re
import time
import zlib
from typing import Annotated, List, Optional

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.auth import get_worker_principal
from gpustack.api.exceptions import (
    BadRequestException,
    ForbiddenException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.api.tenant import (
    TenantContext,
    assert_resource_visible,
    tenant_list_conditions,
)
from gpustack.config.config import get_global_config
from gpustack.monitoring.prometheus import parse_window
from gpustack.monitoring.time_range import measured_interval
from gpustack.observability.request_traces import (
    SPAN_ID,
    TRACE_ID,
    RequestOutcome,
    RequestTraceCapabilities,
    RequestTraceDetail,
    RequestTraceMetrics,
    RequestTracePage,
    RequestTraceRecord,
    TracePagePosition,
    get_request_trace,
    search_request_trace_page,
)
from gpustack.schemas.model_usage_details import ModelUsageDetails
from gpustack.schemas.models import Model, ModelInstance, member_worker_ids
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.workers import Worker
from gpustack.server.deps import SessionDep, TenantContextDep


async def _trace_model(
    ctx: TenantContextDep, session: SessionDep, id: Annotated[int, Path(gt=0)]
) -> int:
    if ctx.user.kind == PrincipalType.SYSTEM:
        raise ForbiddenException(
            message="Model request traces require a user principal"
        )
    model = await Model.one_by_id(session, id)
    assert_resource_visible(ctx, model, not_found_message="Model not found")
    return id


TraceModelDep = Annotated[int, Depends(_trace_model)]
router = APIRouter(prefix="/{id}/request-traces")


def _query_url() -> str:
    cfg = get_global_config()
    url = cfg.get_trace_query_url()
    if not url or not cfg.effective_trace_otlp_endpoint:
        raise ServiceUnavailableException("Trace query is not configured")
    return url


def _query_client(request: Request) -> aiohttp.ClientSession:
    state = request.app.state
    client = getattr(state, "tempo_query_client", state.http_client_no_proxy)
    if client is None:
        raise ServiceUnavailableException("Trace query client is unavailable")
    return client


async def _populate_request_metrics(
    session: AsyncSession,
    ctx: TenantContext,
    model_id: int,
    rows: List[RequestTraceRecord],
) -> None:
    request_ids = {row.request_id for row in rows if row.request_id}
    if not request_ids:
        return

    usage = ModelUsageDetails
    statement = select(
        usage.request_id,
        usage.ttft_ms,
        usage.completed,
        usage.prompt_token_count,
        usage.completion_token_count,
    ).where(
        usage.model_id == model_id,
        usage.request_id.in_(request_ids),
        *tenant_list_conditions(ctx, usage),
    )
    metrics = {}
    for request_id, ttft, completed, input_tokens, output_tokens in await session.exec(
        statement
    ):
        # A gateway fallback can report multiple observations under one ID.
        if request_id in metrics or (ttft is None and not completed):
            metrics[request_id] = None
        else:
            metrics[request_id] = RequestTraceMetrics(
                ttft_ms=ttft,
                input_tokens=input_tokens if completed else None,
                output_tokens=output_tokens if completed else None,
            )

    for row in rows:
        row.request_metrics = metrics.get(row.request_id)


@router.get("/capabilities")
async def get_capabilities(model_id: TraceModelDep) -> RequestTraceCapabilities:
    cfg = get_global_config()
    tracing_enabled = bool(cfg.effective_trace_otlp_endpoint)
    return RequestTraceCapabilities(
        tracing_enabled=tracing_enabled,
        trace_query_configured=tracing_enabled and bool(cfg.get_trace_query_url()),
    )


@router.get("")
async def list_traces(
    request: Request,
    model_id: TraceModelDep,
    ctx: TenantContextDep,
    session: SessionDep,
    window: str = "24h",
    q: Optional[str] = None,
    outcome: Optional[RequestOutcome] = None,
    instance_id: Annotated[Optional[int], Query(gt=0)] = None,
    no_engine: bool = False,
    start: Annotated[Optional[int], Query(ge=0)] = None,
    end: Annotated[Optional[int], Query(ge=0)] = None,
    page_size: Annotated[int, Query(ge=1, le=50)] = 10,
    cursor: Annotated[Optional[str], Query(max_length=12_000)] = None,
) -> RequestTracePage:
    now = int(time.time())
    if cursor is not None:
        position = _decode_cursor(cursor, now)
    else:
        if start is None and end is None:
            end = now + 1
            try:
                start = end - parse_window(window)
            except ValueError as exc:
                raise BadRequestException(message=str(exc)) from None
        else:
            try:
                interval = measured_interval(start, end, now=now)
            except ValueError:
                raise BadRequestException(
                    message="Provide start and end in the past with a span of up to 7 days"
                ) from None
            start, end = int(interval.start), int(interval.end)
        position = TracePagePosition(start=start, end=end, before=end)
    page = await search_request_trace_page(
        _query_client(request),
        _query_url(),
        model_id,
        position=position,
        page_size=page_size,
        identifier=q or None,
        outcome=outcome,
        instance_id=instance_id,
        no_engine=no_engine,
    )
    await _populate_request_metrics(session, ctx, model_id, page.items)
    return page


def _decode_cursor(cursor: str, now: int) -> TracePagePosition:
    try:
        position = TracePagePosition.model_validate_json(
            base64.urlsafe_b64decode(cursor)
        )
    except ValueError:
        raise BadRequestException(message="Invalid trace page cursor") from None
    if (
        not position.start < position.before <= position.end <= now + 1
        or position.end - position.start > 7 * 24 * 3600
    ):
        raise BadRequestException(message="Invalid trace page cursor")
    return position


@router.get("/{trace_id}")
async def get_trace(
    request: Request,
    model_id: TraceModelDep,
    ctx: TenantContextDep,
    session: SessionDep,
    trace_id: Annotated[str, Path(pattern=f"^{TRACE_ID}$")],
    request_id: Optional[str] = None,
    entry_span_id: Annotated[Optional[str], Query(pattern=f"^{SPAN_ID}$")] = None,
) -> RequestTraceDetail:
    detail = await get_request_trace(
        _query_client(request),
        _query_url(),
        model_id,
        trace_id,
        request_id,
        entry_span_id.lower() if entry_span_id is not None else None,
    )
    if detail is None:
        raise NotFoundException(message="Request trace not found")
    await _populate_request_metrics(session, ctx, model_id, [detail.request])
    return detail


_MAX_TRACE_INGEST_BYTES = 4 * 1024 * 1024
_TRACE_INGEST_TIMEOUT = 5


def _trace_target(
    attributes, instance_key: str, *, resource: bool = False
) -> Optional[tuple[int, int]]:
    model_key = "gpustack.model_id"
    ids = {}
    for attr in attributes:
        if attr.key not in (model_key, instance_key):
            continue
        value = attr.value.int_value
        if resource and re.fullmatch(r"[1-9][0-9]{0,9}", attr.value.string_value):
            value = int(attr.value.string_value)
        if attr.key in ids or not 0 < value <= 2**31 - 1:
            raise ForbiddenException(message="Invalid trace workload identity")
        ids[attr.key] = value
    if not ids:
        return None
    if len(ids) != 2:
        raise ForbiddenException(message="Incomplete trace workload identity")
    return ids[model_key], ids[instance_key]


def _worker_trace_targets(batch: ExportTraceServiceRequest) -> set[tuple[int, int]]:
    targets = set()
    for group in batch.resource_spans:
        attributes = group.resource.attributes
        services = [a.value.string_value for a in attributes if a.key == "service.name"]
        if len(services) > 1 or any(
            name in ("gpustack.server", "gpustack.higress") for name in services
        ):
            raise ForbiddenException(message="Invalid Worker trace service")
        worker_service = services == ["gpustack.worker"]
        engine = _trace_target(attributes, "gpustack.engine_instance_id", resource=True)
        if engine:
            targets.add(engine)
        for scope in group.scope_spans:
            for span in scope.spans:
                target = _trace_target(span.attributes, "gpustack.model_instance_id")
                # Require the identity that model-scoped queries use for this source.
                if (worker_service and not target) or (
                    not worker_service and not engine
                ):
                    raise ForbiddenException(message="Missing trace workload identity")
                if target:
                    targets.add(target)
    return targets


async def _authorize_trace_batch(
    body: bytes, encoding: str, session: AsyncSession, worker: Worker
) -> None:
    try:
        if encoding == "gzip":
            with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
                body = compressed.read(_MAX_TRACE_INGEST_BYTES + 1)
            if len(body) > _MAX_TRACE_INGEST_BYTES:
                raise HTTPException(status_code=413, detail="Trace batch is too large")
        batch = ExportTraceServiceRequest.FromString(body)
    except (DecodeError, gzip.BadGzipFile, EOFError, zlib.error):
        raise HTTPException(
            status_code=400, detail="Invalid OTLP trace batch"
        ) from None

    targets = _worker_trace_targets(batch)
    instance_ids = sorted({instance_id for _, instance_id in targets})
    assigned = set()
    for start in range(0, len(instance_ids), 1000):
        instances = await ModelInstance.all_by_ids(
            session, instance_ids[start : start + 1000]
        )
        assigned.update(
            (instance.model_id, instance.id)
            for instance in instances
            if instance.deleted_at is None
            and instance.cluster_id == worker.cluster_id
            and worker.id in member_worker_ids(instance)
        )
    if not targets <= assigned:
        raise ForbiddenException(
            message="Worker is not assigned to this trace workload"
        )


async def ingest_worker_traces(
    request: Request,
    session: SessionDep,
    principal: Annotated[Principal, Depends(get_worker_principal)],
) -> Response:
    """Forward authenticated Worker OTLP/protobuf trace batches to local Tempo."""
    cfg = get_global_config()
    ingest_url = cfg.get_trace_ingest_url() if cfg else None
    if not ingest_url:
        raise ServiceUnavailableException("Trace ingest is not configured")

    content_type = (
        request.headers.get("content-type", "").partition(";")[0].strip().lower()
    )
    if content_type != "application/x-protobuf":
        raise HTTPException(
            status_code=415, detail="Expected application/x-protobuf payload"
        )
    content_encoding = request.headers.get("content-encoding", "").strip().lower()
    if content_encoding not in ("", "gzip"):
        raise HTTPException(
            status_code=415, detail="Unsupported content-encoding for trace ingest"
        )

    if principal.worker is None:
        raise ForbiddenException(
            message="Trace ingest requires a bound Worker principal"
        )

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_TRACE_INGEST_BYTES:
            raise HTTPException(status_code=413, detail="Trace batch is too large")
        body.extend(chunk)
    payload = bytes(body)
    await _authorize_trace_batch(payload, content_encoding, session, principal.worker)

    client: Optional[aiohttp.ClientSession] = getattr(
        request.app.state, "http_client_no_proxy", None
    )
    if client is None:
        raise ServiceUnavailableException("Trace ingest client is unavailable")

    headers = {"Content-Type": "application/x-protobuf"}
    if content_encoding:
        headers["Content-Encoding"] = content_encoding

    try:
        async with client.post(
            ingest_url,
            data=payload,
            headers=headers,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=_TRACE_INGEST_TIMEOUT),
        ) as resp:
            resp_body = await resp.read()
            media_type = (
                resp.headers.get("Content-Type", "application/x-protobuf")
                .partition(";")[0]
                .strip()
            )
            return Response(
                content=resp_body,
                status_code=resp.status,
                media_type=media_type or "application/x-protobuf",
            )
    except (aiohttp.ClientError, asyncio.TimeoutError):
        raise ServiceUnavailableException(
            "Trace ingest service is unavailable"
        ) from None
