"""Read bounded, deployment-scoped request observations from Tempo."""

import asyncio
import base64
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Annotated, Any, AsyncIterator, Dict, List, Literal, Optional, Tuple

import aiohttp
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
)

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.observability.tracing import PUBLIC_INFERENCE_PATHS

TRACE_ID = r"[0-9a-fA-F]{1,32}"
SPAN_ID = r"[0-9a-fA-F]{16}"
RequestOutcome = Literal["success", "error", "cancelled", "unknown"]
EnginePhase = Literal["prefill", "decode"]
ObservationSource = Literal["platform", "router", "engine"]

_GATEWAY = "gpustack.higress"
_PLATFORM = {_GATEWAY, "gpustack.server", "gpustack.worker"}
_VLLM_ROUTER = "vllm-router"
_VLLM_PHASE_ATTRIBUTE = "vllm.request_phase"
_PD_ENGINE_SERVICE = re.compile(r"(?:vllm|sglang)[.](prefill|decode)")
_GATEWAY_ROUTE = re.compile(
    r"router outbound[|][0-9]+[|][|]model-(\d+)-(\d+)"
    r"(?:-dp\d+|-w\d+)?[.](?:static|dns) egress"
)
_ENGINE_TIMES = {
    "ttft_ms": "gen_ai.latency.time_to_first_token",
    "queue_ms": "gen_ai.latency.time_in_queue",
    "prefill_ms": "gen_ai.latency.time_in_model_prefill",
    "decode_ms": "gen_ai.latency.time_in_model_decode",
}
_ENGINE_TOKENS = {
    "input_tokens": "gen_ai.usage.prompt_tokens",
    "output_tokens": "gen_ai.usage.completion_tokens",
}
_ENGINE_FIELDS = {**_ENGINE_TIMES, **_ENGINE_TOKENS}
_SGLANG_REQUEST_MODULE = "sglang::request"
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_DETAIL_SPANS = 256
_SEARCH_BATCH_SIZE = 100
_IDENTIFIER_SEARCH_BATCH_SIZE = 10
_TRACE_DETAIL_BATCH_SIZE = 10
_TEMPO_QUERY_TIMEOUT = 8
# Bound metadata overfetch independently of the per-page trace detail budget.
_MAX_SEARCH_LIMIT = 10_000
_LEGACY_CURSOR_ID_LIMIT = 240
_TRACE_ID_BYTES = 16
_NANOSECONDS_PER_SECOND = 1_000_000_000


class RequestTraceCapabilities(BaseModel):
    tracing_enabled: bool
    trace_query_configured: bool


class RequestTraceMetrics(BaseModel):
    ttft_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class RequestEngineMetrics(RequestTraceMetrics):
    queue_ms: Optional[float] = None
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    tpot_ms: Optional[float] = None


class RequestEngineMetricsByPhase(BaseModel):
    prefill: Optional[RequestEngineMetrics] = None
    decode: Optional[RequestEngineMetrics] = None


class RequestTraceRecord(BaseModel):
    trace_id: str
    entry_span_id: Optional[str] = None
    observation_source: ObservationSource = "platform"
    engine_phase: Optional[EnginePhase] = None
    request_id: Optional[str]
    started_at_ms: int
    duration_ms: float
    observed_by: str
    path: Optional[str]
    status_code: Optional[int]
    outcome: RequestOutcome
    stream: Optional[bool]
    instance_id: Optional[int]
    instance_name: Optional[str] = None
    has_engine: bool
    engine: Optional[RequestEngineMetrics]
    engine_by_phase: Optional[RequestEngineMetricsByPhase] = None
    request_metrics: Optional[RequestTraceMetrics] = None


class RequestTracePage(BaseModel):
    items: List[RequestTraceRecord]
    cursor: str = ""
    next_cursor: Optional[str] = None


_TraceId = Annotated[str, StringConstraints(pattern=f"^{TRACE_ID}$")]


class TracePagePosition(BaseModel):
    start: int = Field(ge=0)
    end: int
    before: int
    boundary: Optional[Tuple[int, _TraceId]] = None
    skipped: int = Field(default=0, ge=0)
    seen: List[_TraceId] = Field(
        default_factory=list, max_length=_LEGACY_CURSOR_ID_LIMIT
    )
    partial: Optional[_TraceId] = None
    offset: int = Field(default=0, ge=0)

    @field_serializer("seen", when_used="json")
    def _pack_seen(self, seen: List[str]) -> str:
        # Keep legacy exclusions compact until the cursor leaves their second.
        packed = b"".join(
            bytes.fromhex(trace_id.zfill(_TRACE_ID_BYTES * 2)) for trace_id in seen
        )
        return base64.urlsafe_b64encode(packed).decode()

    @field_validator("seen", mode="before")
    @classmethod
    def _unpack_seen(cls, seen):
        # A cursor from an existing page can still contain a list of hex IDs.
        if not isinstance(seen, str):
            return seen
        packed = base64.b64decode(seen, altchars=b"-_", validate=True)
        if (
            len(packed) % _TRACE_ID_BYTES
            or len(packed) > _LEGACY_CURSOR_ID_LIMIT * _TRACE_ID_BYTES
        ):
            raise ValueError("Invalid packed trace IDs")
        return [
            packed[start : start + _TRACE_ID_BYTES].hex().lstrip("0") or "0"
            for start in range(0, len(packed), _TRACE_ID_BYTES)
        ]

    def encode(self) -> str:
        return base64.urlsafe_b64encode(
            self.model_dump_json(exclude_defaults=True).encode()
        ).decode()


class RequestTraceSpan(BaseModel):
    id: str
    parent_id: Optional[str]
    name: str
    service: str
    role: Literal["gateway", "proxy", "engine", "other"]
    start_offset_ms: float
    duration_ms: float
    status: str
    engine: Optional[RequestEngineMetrics] = None
    reported_service: Optional[str] = None
    phase: Optional[EnginePhase] = None


class RequestTraceDetail(BaseModel):
    request: RequestTraceRecord
    spans: List[RequestTraceSpan]
    span_count: int


@dataclass(frozen=True)
class _Span:
    id: str
    parent_id: Optional[str]
    name: str
    service: str
    start: int
    duration: int
    status: str
    attrs: Dict[str, Any]
    reported_service: Optional[str] = None
    resource: Dict[str, Any] = field(default_factory=dict)
    kind: str = ""

    @property
    def target(self) -> Tuple[Optional[int], Optional[int]]:
        if self.service == _GATEWAY and (match := _GATEWAY_ROUTE.fullmatch(self.name)):
            return int(match[1]), int(match[2])
        if self.service in _PLATFORM:
            return (
                _positive_int(self.attrs.get("gpustack.model_id")),
                _positive_int(self.attrs.get("gpustack.model_instance_id")),
            )
        return None, None

    @property
    def request_id(self) -> Optional[str]:
        value = self.attrs.get("gpustack.request_id")
        if self.service in _PLATFORM and isinstance(value, str):
            return value or None
        return None

    def matches_request_id(self, identifier: str) -> bool:
        if self.request_id == identifier:
            return True
        if not self.native_source:
            return False
        return identifier == self.attrs.get("gen_ai.request.id") or (
            self.attrs.get("module") == _SGLANG_REQUEST_MODULE
            and identifier == self.attrs.get("rid")
        )

    @property
    def native_target(self) -> Tuple[Optional[int], Optional[int]]:
        if self.service in _PLATFORM:
            return None, None
        model = _resource_id(self.resource.get("gpustack.model_id"))
        instance = _resource_id(self.resource.get("gpustack.engine_instance_id"))
        return (model, instance) if model and instance else (None, None)

    @property
    def native_source(self) -> Optional[Literal["router", "engine"]]:
        if self.native_target[0] is None:
            return None
        role = self.resource.get("gpustack.engine_role")
        if (
            role == "router"
            and self.kind in ("server", "2")
            and self.attrs.get("http.request.method") == "POST"
            and self.path
        ):
            return "router"
        if role in ("engine", "prefill", "decode") and (
            self.name == "llm_request"
            or (
                isinstance(self.attrs.get("rid"), str)
                and self.attrs.get("module") == _SGLANG_REQUEST_MODULE
            )
        ):
            return "engine"
        return None

    @property
    def path(self) -> Optional[str]:
        for name in ("http.route", "url.path"):
            value = self.attrs.get(name)
            if isinstance(value, str) and value in PUBLIC_INFERENCE_PATHS:
                return value
        return None

    @property
    def is_engine(self) -> bool:
        return self.service not in _PLATFORM and (
            self.name == "llm_request"
            or any(field in self.attrs for field in _ENGINE_FIELDS.values())
            or self.native_source == "engine"
        )


def _positive_int(value: Any) -> Optional[int]:
    return value if type(value) is int and value > 0 else None


def _resource_id(value: Any) -> Optional[int]:
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,18}", value):
        value = int(value)
    return _positive_int(value)


def _number(value: Any) -> Optional[float]:
    if type(value) in (int, float) and 0 <= value < math.inf:
        return value
    return None


def _attributes(values: List[dict]) -> Dict[str, Any]:
    result = {}
    for item in values:
        name = item["key"]
        for kind, value in item["value"].items():
            if kind == "intValue":
                result[name] = int(value)
            elif kind == "stringValue" and value != "nil":
                result[name] = value
            elif kind in ("boolValue", "doubleValue"):
                result[name] = value
    return result


def _span_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if re.fullmatch(SPAN_ID, value):
        value = value.lower()
    else:
        value = base64.b64decode(value, validate=True).hex()
        if len(value) != 16:
            raise ValueError("Invalid span ID")
    return value if value != "0000000000000000" else None


def _detail_spans(data: dict) -> List[_Span]:
    spans = []
    for batch in data.get("batches", data.get("resourceSpans", [])):
        resource = _attributes(batch.get("resource", {}).get("attributes", []))
        reported_service = resource.get("service.name", "")
        service = reported_service
        module = resource.get("vllm.instrumenting_module_name")
        if (
            (not service or re.match(r"^unknown_service(?::|$)", service))
            and isinstance(module, str)
            and module.startswith("vllm.")
        ):
            service = "vllm"
        for scope in batch.get("scopeSpans", []):
            for item in scope.get("spans", []):
                start = int(item["startTimeUnixNano"])
                status = (
                    str(item.get("status", {}).get("code", "unset"))
                    .lower()
                    .removeprefix("status_code_")
                )
                spans.append(
                    _Span(
                        id=_span_id(item["spanId"]) or "",
                        parent_id=_span_id(item.get("parentSpanId")),
                        name=item.get("name", ""),
                        service=service,
                        start=start,
                        duration=max(0, int(item["endTimeUnixNano"]) - start),
                        status={"0": "unset", "1": "ok", "2": "error"}.get(
                            status, status
                        ),
                        attrs=_attributes(item.get("attributes", [])),
                        resource=resource,
                        kind=str(item.get("kind", ""))
                        .lower()
                        .removeprefix("span_kind_"),
                        reported_service=(
                            reported_service if service != reported_service else None
                        ),
                    )
                )
    return spans


def _entry_for(anchor: _Span, by_id: Dict[str, _Span]) -> _Span:
    # Stop at the first known entry. A caller can continue an existing trace
    # across requests, so the Tempo trace root can belong to another exchange.
    current = anchor
    request_id = anchor.request_id
    model_id = anchor.target[0]
    visited = {current.id}
    while current.parent_id and (parent := by_id.get(current.parent_id)):
        if (
            parent.id in visited
            or parent.service not in _PLATFORM
            or parent.target[0] not in (None, model_id)
            or (parent.request_id and request_id and parent.request_id != request_id)
        ):
            break
        current = parent
        visited.add(current.id)
        # Higress attaches the request ID to dispatch spans as well as ingress.
        if (
            current.request_id
            and current.service in (_GATEWAY, "gpustack.server")
            and not current.name.startswith("router ")
        ):
            break
    return current


def _request_branch(
    entry: _Span,
    anchors: List[_Span],
    children: Dict[Optional[str], List[_Span]],
    model_id: int,
    covered: set[str],
) -> List[_Span]:
    request_id = entry.request_id or next(
        (span.request_id for span in anchors if span.request_id), None
    )
    branch = {}
    pending = [entry, *anchors]
    while pending:
        span = pending.pop()
        if span.id in branch or span.id in covered:
            continue
        if span.target[0] not in (None, model_id) or span.native_target[0] not in (
            None,
            model_id,
        ):
            continue
        if span.request_id and request_id and span.request_id != request_id:
            continue
        branch[span.id] = span
        pending.extend(children[span.id])
    return sorted(branch.values(), key=lambda s: (s.start, s.id))


def _independent_native_entry(
    entry: _Span, by_id: Dict[str, _Span], covered: set[str]
) -> bool:
    visited = {entry.id}
    parent_id = entry.parent_id
    while parent_id:
        parent = by_id.get(parent_id)
        if parent is None:
            return True
        if (
            parent.id in visited
            or parent.id in covered
            or parent.native_source
            or parent.target[0] is not None
        ):
            return False
        visited.add(parent.id)
        parent_id = parent.parent_id
    return True


def _requests(
    spans: List[_Span], model_id: int
) -> List[Tuple[_Span, List[_Span], List[_Span]]]:
    by_id = {span.id: span for span in spans}
    children = defaultdict(list)
    entries = {}
    for span in spans:
        children[span.parent_id].append(span)
        if span.target[0] == model_id:
            entry = _entry_for(span, by_id)
            entries.setdefault(entry.id, (entry, []))[1].append(span)
    requests = [
        (entry, anchors, _request_branch(entry, anchors, children, model_id, set()))
        for entry, anchors in entries.values()
    ]
    covered = {span.id for _, _, branch in requests for span in branch}
    for entry in spans:
        if (
            entry.id not in covered
            and entry.native_target[0] == model_id
            and entry.native_source
            and _independent_native_entry(entry, by_id, covered)
        ):
            anchors = [entry]
            branch = _request_branch(entry, anchors, children, model_id, covered)
            requests.append((entry, anchors, branch))
    return requests


def _http_status(span: _Span) -> Optional[int]:
    value = span.attrs.get(
        "http.response.status_code", span.attrs.get("http.status_code")
    )
    if type(value) is int or (
        isinstance(value, str) and value.isascii() and value.isdigit()
    ):
        code = int(value)
        return code if 100 <= code <= 599 else None
    return None


def _request_outcome(
    status: Optional[int], entry: _Span, spans: List[_Span]
) -> RequestOutcome:
    if status == 499 or entry.attrs.get("gpustack.cancelled") is True:
        return "cancelled"
    if (status is not None and status >= 400) or entry.status in ("error", "2"):
        return "error"
    if any(
        span.attrs.get("gpustack.cancelled") is True
        for span in spans
        if span.service in _PLATFORM
    ):
        return "cancelled"
    if any(
        span.status in ("error", "2") and _http_status(span) == status
        for span in spans
        if span.service in _PLATFORM
    ):
        return "error"
    if status is None:
        return "unknown"
    return "success"


def _span_engine_metrics(span: _Span) -> Optional[RequestEngineMetrics]:
    if not span.is_engine:
        return None
    attrs = span.attrs
    values = {}
    for key, attribute in _ENGINE_TIMES.items():
        if (value := _number(attrs.get(attribute))) is not None:
            values[key] = value * 1000
    for key, attribute in _ENGINE_TOKENS.items():
        value = attrs.get(attribute)
        if type(value) is int and value >= 0:
            values[key] = value
    if "decode_ms" in values and values.get("output_tokens", 0) > 1:
        values["tpot_ms"] = values["decode_ms"] / (values["output_tokens"] - 1)
    return RequestEngineMetrics(**values) if values else None


def _router_phase(span: Optional[_Span]) -> Optional[EnginePhase]:
    if span and span.service == _VLLM_ROUTER:
        phase = span.attrs.get(_VLLM_PHASE_ATTRIBUTE)
        if phase in ("prefill", "decode"):
            return phase
    return None


def _engine_phase(span: _Span) -> Optional[EnginePhase]:
    role = span.resource.get("gpustack.engine_role")
    return role if role in ("prefill", "decode") else None


def _engine_metrics(
    spans: List[_Span],
) -> Tuple[Optional[RequestEngineMetrics], Optional[RequestEngineMetricsByPhase]]:
    by_id = {span.id: span for span in spans}
    observations = []
    by_phase = defaultdict(list)
    for span in spans:
        metrics = _span_engine_metrics(span)
        if metrics is None:
            continue
        observations.append(metrics)
        phase = _router_phase(by_id.get(span.parent_id or "")) or _engine_phase(span)
        if phase is None and (match := _PD_ENGINE_SERVICE.fullmatch(span.service)):
            phase = match[1]
        if phase:
            by_phase[phase].append(metrics)

    # A retry or multiple choices in one phase have no unique observation.
    phases = {
        phase: values[0] for phase, values in by_phase.items() if len(values) == 1
    }
    return (
        observations[0] if len(observations) == 1 else None,
        RequestEngineMetricsByPhase(**phases) if phases else None,
    )


def _instance_name(
    instance_id: int, anchors: List[_Span], spans: List[_Span]
) -> Optional[str]:
    names = {
        name[:255]
        for span in anchors
        if span.service in _PLATFORM and span.target[1] == instance_id
        if isinstance(name := span.attrs.get("gpustack.model_instance_name"), str)
        and name
    }
    if not names:
        names = {
            name[:255]
            for span in spans
            if _resource_id(span.resource.get("gpustack.engine_instance_id"))
            == instance_id
            if isinstance(
                name := span.resource.get("gpustack.engine_instance_name"), str
            )
            and name
        }
    return next(iter(names)) if len(names) == 1 else None


def _summarize(
    trace_id: str, entry: _Span, anchors: List[_Span], spans: List[_Span]
) -> RequestTraceRecord:
    targets = [entry, *sorted(anchors, key=lambda span: span.start, reverse=True)]
    request_id = next((span.request_id for span in targets if span.request_id), None)
    path = next((span.path for span in targets if span.path), None)
    status = next((code for span in targets if (code := _http_status(span))), None)
    # A gateway retry can select a different instance. Only show a single ID
    # when the observations agree.
    instances = {(span.target[1] or span.native_target[1]) for span in anchors} - {None}
    instance_id = next(iter(instances)) if len(instances) == 1 else None
    stream = next(
        (
            value
            for span in targets
            if type(value := span.attrs.get("gpustack.stream")) is bool
        ),
        None,
    )
    engine, engine_by_phase = _engine_metrics(spans)
    return RequestTraceRecord(
        trace_id=trace_id,
        entry_span_id=entry.id,
        observation_source=entry.native_source or "platform",
        engine_phase=_engine_phase(entry),
        request_id=request_id,
        started_at_ms=entry.start // 1_000_000,
        duration_ms=entry.duration / 1_000_000,
        observed_by=entry.service,
        path=path,
        status_code=status,
        outcome=_request_outcome(status, entry, spans),
        stream=stream,
        instance_id=instance_id,
        instance_name=(
            _instance_name(instance_id, anchors, spans) if instance_id else None
        ),
        has_engine=any(span.is_engine for span in spans),
        engine=engine,
        engine_by_phase=engine_by_phase,
    )


def _search_query(model_id: int, identifier: Optional[str]) -> str:
    direct = f"router outbound.*[|][|]model-{model_id}-.* egress"
    scope = f'{{span.gpustack.model_id={model_id}||name=~"{direct}"}}'
    native_scope = (
        f'resource.gpustack.model_id="{model_id}"'
        '&&resource.gpustack.engine_instance_id!=nil'
    )
    if identifier:
        value = json.dumps(identifier, ensure_ascii=False)
        match = (
            f"{{span.gpustack.request_id={value}||span.gen_ai.request.id={value}"
            f'||(span.module="{_SGLANG_REQUEST_MODULE}"&&span.rid={value})}}'
        )
        query = f"({scope}||{{{native_scope}}})&&{match}"
    else:
        paths = "|".join(sorted(PUBLIC_INFERENCE_PATHS))
        native = (
            f'{{{native_scope}'
            f'&&(name="llm_request"||(span.module="{_SGLANG_REQUEST_MODULE}"&&span.rid!=nil)'
            f'||(resource.gpustack.engine_role="router"&&(span:kind=server||span:kind=client)'
            f'&&span.http.request.method="POST"'
            f'&&(span.http.route=~"{paths}"||span.url.path=~"{paths}")))}}'
        )
        query = f"({scope}||{native})"
    return f"{query} with (most_recent=true)"


async def _tempo_get(
    client: aiohttp.ClientSession,
    url: str,
    params: Optional[dict] = None,
) -> Optional[dict]:
    try:
        async with client.get(
            url,
            params=params,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=_TEMPO_QUERY_TIMEOUT),
            headers={"Accept": "application/json"},
        ) as response:
            if response.status == 404 and params is None:
                return None
            if response.status != 200:
                raise ServiceUnavailableException("Trace query service is unavailable")
            data = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                data.extend(chunk)
                if len(data) > _MAX_RESPONSE_BYTES:
                    raise ServiceUnavailableException(
                        "Trace query response is too large"
                    )
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("Invalid trace query response")
            return payload
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise ServiceUnavailableException(
            "Trace query service is unavailable"
        ) from None


async def _search_metadata(
    client: aiohttp.ClientSession,
    base_url: str,
    model_id: int,
    position: TracePagePosition,
    identifier: Optional[str],
    limit: int,
) -> Tuple[Dict[str, dict], Optional[int]]:
    seen = set(position.seen)
    # The extra result establishes whether the oldest timestamp group is complete.
    search_limit = min(position.skipped + len(seen) + limit + 1, _MAX_SEARCH_LIMIT)
    query = _search_query(model_id, identifier)
    while True:
        payload = await _tempo_get(
            client,
            f"{base_url}/api/search",
            {
                "q": query,
                "start": position.start,
                "end": position.before,
                "limit": search_limit,
                "spss": 1,
            },
        )
        traces = payload.get("traces") or []
        # Tempo orders by timestamp only. A full result may contain only part
        # of its oldest timestamp group, so leave that group for a larger search.
        cutoff = (
            min(int(trace["startTimeUnixNano"]) for trace in traces)
            if len(traces) >= search_limit
            else None
        )
        candidates = {}
        skipped = 0
        for trace in traces:
            started, trace_id = key = _candidate_key(trace)
            if not trace_id or trace_id in seen:
                continue
            consumed = position.boundary is not None and key >= position.boundary
            skipped += consumed
            if trace_id == position.partial or (
                not consumed and (cutoff is None or started > cutoff)
            ):
                candidates[trace_id] = {**trace, "traceID": trace_id}
        if candidates or cutoff is None:
            position.skipped = skipped
            return candidates, cutoff
        # Late arrivals can fill the prefix ahead of the cursor. Only metadata
        # is fetched again, and the query remains bounded even for forged cursors.
        if search_limit == _MAX_SEARCH_LIMIT:
            raise ServiceUnavailableException("Trace query reached the search limit")
        search_limit = min(search_limit * 2, _MAX_SEARCH_LIMIT)


async def _search_candidates(
    client: aiohttp.ClientSession,
    base_url: str,
    model_id: int,
    position: TracePagePosition,
    identifier: Optional[str],
    *,
    limit: int,
) -> Tuple[List[dict], bool]:
    search = _search_metadata(client, base_url, model_id, position, identifier, limit)
    trace_id = _tempo_trace_id(identifier or "")
    if trace_id and trace_id not in position.seen:
        (candidates, cutoff), exact = await asyncio.gather(
            search,
            _exact_trace_candidate(client, base_url, trace_id, model_id=model_id),
        )
        if exact is not None:
            if trace_id in candidates:
                candidates[trace_id]["detail_spans"] = exact["detail_spans"]
            else:
                started, _ = key = _candidate_key(exact)
                # An exact lookup must respect both the cursor and any
                # candidates still beyond the search limit.
                if (
                    started < position.before * _NANOSECONDS_PER_SECOND
                    and (cutoff is None or started > cutoff)
                    and (position.boundary is None or key < position.boundary)
                ):
                    candidates[trace_id] = exact
    else:
        candidates, cutoff = await search
    partial = position.partial
    if partial and partial not in candidates and partial not in position.seen:
        if resumed := await _exact_trace_candidate(
            client, base_url, partial, model_id=model_id
        ):
            candidates[partial] = resumed
    ordered = sorted(
        candidates.values(),
        key=lambda trace: (trace["traceID"] == partial, _candidate_key(trace)),
        reverse=True,
    )
    return ordered[:limit], cutoff is not None or len(ordered) > limit


async def _detail_batches(
    client: aiohttp.ClientSession, base_url: str, candidates: List[dict]
) -> AsyncIterator[Tuple[List[dict], bool]]:
    if not candidates:
        yield [], False
        return
    for index in range(0, len(candidates), _TRACE_DETAIL_BATCH_SIZE):
        end = index + _TRACE_DETAIL_BATCH_SIZE
        batch = candidates[index:end]
        lookups = [trace for trace in batch if "detail_spans" not in trace]
        details = await asyncio.gather(
            *(_exact_trace_candidate(client, base_url, t["traceID"]) for t in lookups)
        )
        for trace, detail in zip(lookups, details):
            # Keep the search timestamp even if the trace has since expired.
            trace["detail_spans"] = detail["detail_spans"] if detail else []
        yield batch, end < len(candidates)


async def _exact_trace_candidate(
    client: aiohttp.ClientSession,
    base_url: str,
    trace_id: str,
    *,
    model_id: Optional[int] = None,
) -> Optional[dict]:
    detail = await _tempo_get(client, f"{base_url}/api/traces/{trace_id}")
    if detail is None:
        return None
    try:
        spans = _detail_spans(detail)
        # Direct lookups from a query or cursor bypass the model-scoped Tempo
        # search. Require a visible branch before admitting the trace to a page.
        if spans and (model_id is None or _requests(spans, model_id)):
            return {
                "traceID": trace_id,
                "startTimeUnixNano": min(span.start for span in spans),
                "detail_spans": spans,
            }
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException("Trace query returned invalid data") from None
    return None


@dataclass
class _RowFilter:
    model_id: int
    start: int
    end: int
    identifier: Optional[str]
    outcome: Optional[RequestOutcome]
    instance_id: Optional[int]
    no_engine: bool

    def rows(self, trace: dict) -> List[RequestTraceRecord]:
        trace_id = _tempo_trace_id(trace["traceID"])
        if not trace_id:
            return []
        identifier = self.identifier
        exact_trace = identifier and _tempo_trace_id(identifier) == trace_id
        rows = (
            _summarize(trace_id, entry, anchors, spans)
            for entry, anchors, spans in _requests(trace["detail_spans"], self.model_id)
            if not identifier
            or exact_trace
            or any(span.matches_request_id(identifier) for span in spans)
        )
        return sorted(
            (
                row
                for row in rows
                if self.start * 1000 <= row.started_at_ms < self.end * 1000
                and (self.outcome is None or row.outcome == self.outcome)
                and (self.instance_id is None or row.instance_id == self.instance_id)
                and (not self.no_engine or not row.has_engine)
            ),
            key=lambda row: (
                row.started_at_ms,
                row.request_id or "",
                row.entry_span_id,
            ),
            reverse=True,
        )


def _tempo_trace_id(value: str) -> str:
    # Match Tempo's search metadata so the same trace has one pagination key.
    return value.lower().lstrip("0") if re.fullmatch(TRACE_ID, value) else ""


def _candidate_key(trace: dict) -> Tuple[int, str]:
    return int(trace["startTimeUnixNano"]), _tempo_trace_id(trace["traceID"])


def _page_from_candidates(
    candidates: List[dict],
    position: TracePagePosition,
    filters: _RowFilter,
    page_size: int,
    more: bool,
) -> RequestTracePage:
    cursor = position.encode()
    items: List[RequestTraceRecord] = []
    partial, offset = position.partial, position.offset
    for trace in candidates:
        started, trace_id = key = _candidate_key(trace)
        rows = filters.rows(trace)
        if trace_id == partial:
            rows = rows[offset:]
        if trace_id != partial or position.boundary is None:
            before = max(
                position.start + 1,
                min(position.before, started // _NANOSECONDS_PER_SECOND + 1),
            )
            if before != position.before:
                position.before = before
                position.seen.clear()
                position.skipped = 0
            position.boundary = key
            position.skipped += 1
        remaining = page_size - len(items)
        items.extend(rows[:remaining])
        if len(rows) > remaining:
            position.partial = trace_id
            position.offset = (offset if trace_id == partial else 0) + remaining
            return RequestTracePage(
                items=items, cursor=cursor, next_cursor=position.encode()
            )
    position.partial = None
    position.offset = 0
    return RequestTracePage(
        items=items,
        cursor=cursor,
        next_cursor=position.encode() if more else None,
    )


async def search_request_trace_page(
    client: aiohttp.ClientSession,
    base_url: str,
    model_id: int,
    *,
    position: TracePagePosition,
    page_size: int,
    identifier: Optional[str],
    outcome: Optional[RequestOutcome],
    instance_id: Optional[int],
    no_engine: bool,
) -> RequestTracePage:
    position.seen = list(filter(None, map(_tempo_trace_id, position.seen)))
    position.partial = _tempo_trace_id(position.partial or "") or None
    if position.boundary is not None:
        boundary_id = _tempo_trace_id(position.boundary[1])
        position.boundary = (position.boundary[0], boundary_id) if boundary_id else None
    filters = _RowFilter(
        model_id,
        position.start,
        position.end,
        identifier,
        outcome,
        instance_id,
        no_engine,
    )
    cursor = position.encode()
    items: List[RequestTraceRecord] = []
    next_cursor = None
    scanned = 0
    try:
        async with asyncio.timeout(_TEMPO_QUERY_TIMEOUT):
            while scanned < _SEARCH_BATCH_SIZE:
                fetch_size = (
                    _IDENTIFIER_SEARCH_BATCH_SIZE
                    if identifier
                    else max(_TRACE_DETAIL_BATCH_SIZE, page_size - len(items))
                )
                limit = min(fetch_size, _SEARCH_BATCH_SIZE - scanned)
                candidates, more = await _search_candidates(
                    client,
                    base_url,
                    model_id,
                    position,
                    identifier,
                    limit=limit,
                )
                scanned += len(candidates)
                async for batch, remaining in _detail_batches(
                    client, base_url, candidates
                ):
                    page = _page_from_candidates(
                        batch,
                        position,
                        filters,
                        page_size - len(items),
                        more or remaining,
                    )
                    items.extend(page.items)
                    next_cursor = page.next_cursor
                    if len(items) == page_size:
                        break
                if next_cursor is None or len(items) == page_size:
                    break
    except TimeoutError:
        # Resume from the last materialized batch, without consuming pending IDs.
        if next_cursor is None or next_cursor == cursor:
            raise ServiceUnavailableException("Trace query timed out") from None
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException("Trace query returned invalid data") from None
    items.sort(key=lambda row: row.started_at_ms, reverse=True)
    return RequestTracePage(items=items, cursor=cursor, next_cursor=next_cursor)


def _display_span(span: _Span, start: int, parent: Optional[_Span]) -> RequestTraceSpan:
    role: Literal["gateway", "proxy", "engine", "other"] = "other"
    if span.service == _GATEWAY:
        role = "gateway"
    elif span.service in _PLATFORM:
        role = "proxy"
    elif span.is_engine:
        role = "engine"
    return RequestTraceSpan(
        id=span.id,
        parent_id=parent.id if parent else None,
        name=span.name[:160],
        service=span.service[:100],
        role=role,
        start_offset_ms=(span.start - start) / 1_000_000,
        duration_ms=span.duration / 1_000_000,
        status=span.status,
        engine=_span_engine_metrics(span),
        reported_service=span.reported_service,
        phase=(
            _router_phase(parent) or _engine_phase(span)
            if role == "engine"
            else _router_phase(span)
        ),
    )


async def get_request_trace(
    client: aiohttp.ClientSession,
    base_url: str,
    model_id: int,
    trace_id: str,
    request_id: Optional[str],
    entry_span_id: Optional[str] = None,
) -> Optional[RequestTraceDetail]:
    trace_id = _tempo_trace_id(trace_id)
    if not trace_id:
        return None
    payload = await _tempo_get(client, f"{base_url}/api/traces/{trace_id}")
    if payload is None:
        return None
    try:
        requests = _requests(_detail_spans(payload), model_id)
        if entry_span_id is not None:
            requests = _matching_entry(requests, entry_span_id, model_id)
        for entry, anchors, spans in requests:
            row = _summarize(trace_id, entry, anchors, spans)
            if request_id and row.request_id != request_id:
                continue
            start = spans[0].start
            by_id = {span.id: span for span in spans}
            return RequestTraceDetail(
                request=row,
                spans=[
                    _display_span(span, start, by_id.get(span.parent_id or ""))
                    for span in spans[:_MAX_DETAIL_SPANS]
                ],
                span_count=len(spans),
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException("Trace query returned invalid data") from None
    return None


def _matching_entry(
    requests: List[Tuple[_Span, List[_Span], List[_Span]]],
    entry_span_id: str,
    model_id: int,
) -> List[Tuple[_Span, List[_Span], List[_Span]]]:
    exact = [request for request in requests if request[0].id == entry_span_id]
    if exact:
        return exact
    # If an upstream span arrived after the list query, the original entry can
    # now be a child of the same request. Resolve only an unambiguous match.
    containing = [
        request
        for request in requests
        if any(
            span.id == entry_span_id
            and (
                span.target[0] == model_id
                or (span.native_source and span.native_target[0] == model_id)
            )
            for span in request[2]
        )
    ]
    return containing if len(containing) == 1 else []
