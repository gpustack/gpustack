import asyncio
import base64
import time
from typing import Annotated, List, Optional

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

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
from gpustack.schemas.models import Model
from gpustack.schemas.principals import PrincipalType
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


async def ingest_worker_traces(request: Request) -> Response:
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

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _MAX_TRACE_INGEST_BYTES:
            raise HTTPException(status_code=413, detail="Trace batch is too large")

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
            data=bytes(body),
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
