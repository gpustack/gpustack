import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.observability import request_traces
from gpustack.observability.request_traces import (
    TracePagePosition,
    get_request_trace,
    search_request_trace_page,
)

_REQUEST = "a0000000-0000-4000-8000-000000000001"
_OTHER_REQUEST = "a0000000-0000-4000-8000-000000000002"
_TRACE = "ab000000000000000000000000000001"
_NS = 1_000_000_000


def _attributes(**values):
    keys = {
        str: "stringValue",
        int: "intValue",
        bool: "boolValue",
        float: "doubleValue",
    }
    return [{"key": k, "value": {keys[type(v)]: v}} for k, v in values.items()]


def _span(service, id, parent, name, start, duration, **attrs):
    span = {
        "spanId": base64.b64encode(id.to_bytes(8, "big")).decode(),
        "name": name,
        "startTimeUnixNano": str(start * _NS),
        "endTimeUnixNano": str((start + duration) * _NS),
        "attributes": _attributes(**attrs),
    }
    if parent:
        span["parentSpanId"] = base64.b64encode(parent.to_bytes(8, "big")).decode()
    return {
        "resource": {"attributes": _attributes(**{"service.name": service})},
        "scopeSpans": [{"spans": [span]}],
    }


def _client(body, status=200):
    async def read(_size):
        yield json.dumps(body).encode()

    client = MagicMock()
    client.get.return_value.__aenter__.return_value = SimpleNamespace(
        status=status, content=SimpleNamespace(iter_chunked=read)
    )
    return client


def _tempo_client(search, traces):
    client = MagicMock()

    def get(url, **kwargs):
        if url.endswith("/api/search"):
            return _client({"traces": search}).get(url, **kwargs)
        batches = traces.get(url.rsplit("/", 1)[-1])
        response = _client({"batches": batches}) if batches else _client({}, status=404)
        return response.get(url, **kwargs)

    client.get.side_effect = get
    return client


async def _trace_rows(
    client, *, identifier=None, outcome=None, instance_id=None, no_engine=False
):
    page = await search_request_trace_page(
        client,
        "http://tempo",
        42,
        position=TracePagePosition(start=1, end=200, before=200),
        page_size=10,
        identifier=identifier,
        outcome=outcome,
        instance_id=instance_id,
        no_engine=no_engine,
    )
    return page.items


@pytest.mark.asyncio
async def test_detail_follows_only_the_selected_request_branch():
    batches = [
        _span("external", 1, None, "client", 90, 100),
        _span(
            "gpustack.higress",
            2,
            1,
            "egress",
            100,
            4,
            **{"gpustack.request_id": _REQUEST, "http.status_code": "200"},
        ),
        _span(
            "gpustack.higress",
            3,
            2,
            "router outbound|80||model-42-8.static egress",
            100,
            4,
            **{"gpustack.request_id": _REQUEST},
        ),
        _span(
            "gpustack.worker",
            4,
            3,
            "POST inference proxy",
            100,
            3,
            **{
                "gpustack.model_id": 42,
                "gpustack.model_instance_id": 8,
                "gpustack.model_instance_name": "removed-instance",
                "gpustack.request_id": _REQUEST,
                "http.route": "/v1/chat/completions",
                "gpustack.stream": True,
                "gpustack.cancelled": True,
                "gen_ai.prompt": "private prompt",
            },
        ),
        _span(
            "vllm",
            5,
            4,
            "llm_request",
            101,
            2,
            **{
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 25,
                "gen_ai.latency.time_in_model_decode": 1.2,
                "gen_ai.latency.time_to_first_token": 0.125,
            },
        ),
        _span(
            "gpustack.server",
            6,
            1,
            "POST /v1/chat/completions",
            110,
            2,
            **{"gpustack.model_id": 78, "gpustack.request_id": _OTHER_REQUEST},
        ),
    ]
    # Tempo can return the Worker and engine resource batches first.
    client = _client({"batches": [*batches[3:5], *batches[:3], batches[5]]})
    detail = await get_request_trace(client, "http://tempo", 42, _TRACE, _REQUEST)
    assert detail.span_count == 4
    assert {int(span.id, 16) for span in detail.spans} == {2, 3, 4, 5}
    assert detail.request.instance_name == "removed-instance"
    assert detail.request.outcome == "cancelled"
    assert detail.request.status_code == 200
    assert detail.request.observed_by == "gpustack.higress"
    assert detail.request.duration_ms == 4000
    assert detail.request.engine.ttft_ms == 125
    assert detail.request.engine_by_phase is None
    assert detail.request.engine.tpot_ms == 50
    assert detail.request.engine.queue_ms is None
    engine = next(span.engine for span in detail.spans if span.name == "llm_request")
    assert engine.decode_ms == 1200
    assert engine.tpot_ms == 50
    assert all(span.engine is None for span in detail.spans if span.role != "engine")
    assert "private prompt" not in detail.model_dump_json()
    assert (
        await get_request_trace(client, "http://tempo", 42, _TRACE, _OTHER_REQUEST)
        is None
    )
    assert await get_request_trace(client, "http://tempo", 43, _TRACE, _REQUEST) is None


@pytest.mark.asyncio
async def test_detail_exposes_pd_phases_and_explicit_span_errors():
    entry = _span(
        "gpustack.worker",
        1,
        None,
        "POST inference proxy",
        100,
        10,
        **{"gpustack.model_id": 42, "gpustack.request_id": _REQUEST},
    )
    router = _span("vllm-router", 2, 1, "POST /v1/chat/completions", 100, 10)
    batches = [entry, router]
    for dispatch_id, phase in [(3, "prefill"), (5, "decode"), (7, "unknown")]:
        dispatch = _span(
            "vllm-router",
            dispatch_id,
            2,
            "POST /v1/chat/completions",
            101,
            4,
            **{"vllm.request_phase": phase},
        )
        service = f"vllm.{phase}" if phase != "unknown" else "custom-engine"
        engine = _span(service, dispatch_id + 1, dispatch_id, "llm_request", 102, 3)
        if phase == "decode":
            engine["scopeSpans"][0]["spans"][0]["status"] = {"code": 2}
        batches.extend([engine, dispatch])
    batches.append(
        _span(
            "custom",
            9,
            2,
            "custom-call",
            101,
            1,
            **{"vllm.request_phase": "decode"},
        )
    )

    detail = await get_request_trace(
        _client({"batches": batches}), "http://tempo", 42, _TRACE, _REQUEST
    )

    spans = {int(span.id, 16): span for span in detail.spans}
    assert [(spans[i].name, spans[i].service) for i in (4, 6, 8)] == [
        ("llm_request", "vllm.prefill"),
        ("llm_request", "vllm.decode"),
        ("llm_request", "custom-engine"),
    ]
    assert {i: span.phase for i, span in spans.items() if span.phase} == {
        3: "prefill",
        4: "prefill",
        5: "decode",
        6: "decode",
    }
    assert spans[6].status == "error"
    assert spans[4].status == "unset"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("prefill_retry", [False, True])
async def test_list_and_detail_keep_metrics_by_pd_phase(backend, prefill_retry):
    def worker(id, model_id, request_id):
        return _span(
            "gpustack.worker",
            id,
            None,
            "POST inference proxy",
            100,
            6,
            **{"gpustack.model_id": model_id, "gpustack.request_id": request_id},
        )

    def engine(service, id, parent, input_tokens, output_tokens, ttft):
        return _span(
            service,
            id,
            parent,
            "llm_request",
            101,
            3,
            **{
                "gen_ai.usage.prompt_tokens": input_tokens,
                "gen_ai.usage.completion_tokens": output_tokens,
                "gen_ai.latency.time_to_first_token": ttft,
                "gen_ai.latency.time_in_queue": 0.012,
                "gen_ai.latency.time_in_model_prefill": 0.050,
                "gen_ai.latency.time_in_model_decode": 0.250,
            },
        )

    batches = [worker(1, 42, _REQUEST), worker(10, 43, _OTHER_REQUEST)]
    batches.append(engine(f"{backend}.prefill", 11, 10, 999, 99, 1.0))
    for i, (phase, output, ttft) in enumerate(
        (("prefill", 0, 0.125), ("decode", 2, 0.375)), 1
    ):
        parent = 1
        service = f"{backend}.{phase}"
        if backend == "vllm":
            parent = 3 + i
            service = "custom-engine"
            batches.append(
                _span(
                    "vllm-router",
                    parent,
                    1,
                    "POST /v1/chat/completions",
                    100,
                    5,
                    **{"vllm.request_phase": phase},
                )
            )
        batches.append(engine(service, 5 + i, parent, 21, output, ttft))
        if phase == "prefill" and prefill_retry:
            batches.append(engine(service, 8, parent, 21, 1, 0.250))

    client = _native_search_client(batches)
    (row,) = await _trace_rows(client)
    assert row.engine is None
    phases = row.engine_by_phase
    if prefill_retry:
        assert phases.prefill is None
    else:
        assert phases.prefill.ttft_ms == 125
        assert (phases.prefill.input_tokens, phases.prefill.output_tokens) == (21, 0)
    assert phases.decode.ttft_ms == 375
    assert phases.decode.queue_ms == 12
    assert phases.decode.prefill_ms == 50
    assert phases.decode.decode_ms == 250
    assert phases.decode.tpot_ms == 250
    assert (phases.decode.input_tokens, phases.decode.output_tokens) == (21, 2)
    detail = await get_request_trace(
        _client({"batches": batches}), "http://tempo", 42, _TRACE, _REQUEST
    )
    assert detail.request.engine_by_phase == phases


@pytest.mark.asyncio
async def test_list_searches_for_candidates_then_reads_request_branches():
    root = _span(
        "gpustack.higress",
        1,
        None,
        "egress",
        100,
        4,
        **{"gpustack.request_id": _REQUEST},
    )
    upstream = _span(
        "gpustack.higress",
        2,
        1,
        "router outbound|80||model-42-8-w2.dns egress",
        100,
        4,
        **{"gpustack.request_id": _REQUEST},
    )
    worker = _span(
        "gpustack.worker",
        3,
        2,
        "POST inference proxy",
        101,
        2,
        **{
            "gpustack.model_id": 42,
            "gpustack.model_instance_id": 8,
            "gpustack.model_instance_name": "removed-instance",
            "gpustack.request_id": _REQUEST,
            "http.response.status_code": 504,
        },
    )
    client = _native_search_client([root, upstream, worker])
    rows = await _trace_rows(client, outcome="error", instance_id=8, no_engine=True)
    (row,) = rows
    assert not row.has_engine
    assert row.instance_id == 8
    assert row.instance_name == "removed-instance"
    assert row.status_code == 504
    assert row.started_at_ms == 100_000
    search, lookup = client.get.call_args_list
    assert search.args == ("http://tempo/api/search",)
    assert lookup.args == (f"http://tempo/api/traces/{_TRACE}",)
    params = search.kwargs["params"]
    assert params["limit"] == 11
    assert params["spss"] == 1
    assert "most_recent=true" in params["q"]
    assert "select(" not in params["q"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reported", "module", "service", "raw"),
    [
        ("unknown_service", "vllm.llm_engine", "vllm", "unknown_service"),
        ("unknown_service:python", "vllm.llm_engine", "vllm", "unknown_service:python"),
        ("custom-engine", "vllm.llm_engine", "custom-engine", None),
        ("unknown_service", "other.engine", "unknown_service", None),
    ],
)
async def test_detail_resolves_unnamed_vllm_from_reported_resource_metadata(
    reported, module, service, raw
):
    entry = _span(
        "gpustack.worker",
        1,
        None,
        "POST inference proxy",
        100,
        3,
        **{"gpustack.model_id": 42, "gpustack.request_id": _REQUEST},
    )
    engine = _span(reported, 2, 1, "llm_request", 101, 2)
    engine["resource"]["attributes"].extend(
        _attributes(**{"vllm.instrumenting_module_name": module})
    )
    detail = await get_request_trace(
        _client({"batches": [entry, engine]}), "http://tempo", 42, _TRACE, _REQUEST
    )
    assert detail.spans[1].service == service
    assert detail.spans[1].reported_service == raw


@pytest.mark.asyncio
async def test_unavailable_tempo_does_not_report_empty_results():
    with pytest.raises(ServiceUnavailableException):
        await _trace_rows(_client({}, status=503))


@pytest.mark.asyncio
async def test_stream_error_preserves_the_committed_http_status():
    worker = _span(
        "gpustack.worker",
        2,
        1,
        "POST inference proxy",
        100,
        4,
        **{"gpustack.model_id": 42, "http.response.status_code": 200},
    )
    worker["scopeSpans"][0]["spans"][0]["status"] = {"code": "STATUS_CODE_ERROR"}
    gateway = _span(
        "gpustack.higress",
        1,
        None,
        "egress",
        99,
        6,
        **{"gpustack.request_id": _REQUEST, "http.status_code": "200"},
    )
    detail = await get_request_trace(
        _client({"batches": [worker, gateway]}), "http://tempo", 42, _TRACE, _REQUEST
    )
    assert detail.request.outcome == "error"
    assert detail.request.status_code == 200


@pytest.mark.asyncio
async def test_direct_trace_keeps_the_name_of_its_selected_engine():
    gateway = _span(
        "gpustack.higress",
        1,
        None,
        "router outbound|80||model-42-8.static egress",
        100,
        4,
        **{"gpustack.request_id": _REQUEST},
    )
    engine = _span("vllm", 2, 1, "llm_request", 101, 3)
    engine["resource"]["attributes"].extend(
        _attributes(
            **{
                "gpustack.engine_instance_id": "8",
                "gpustack.engine_instance_name": "removed-8",
            }
        )
    )
    other = _span("vllm", 3, 1, "llm_request", 101, 3)
    other["resource"]["attributes"].extend(
        _attributes(
            **{
                "gpustack.engine_instance_id": "90",
                "gpustack.engine_instance_name": "unrelated",
            }
        )
    )

    detail = await get_request_trace(
        _client({"batches": [gateway, engine, other]}),
        "http://tempo",
        42,
        _TRACE,
        _REQUEST,
    )

    assert detail.request.instance_id == 8
    assert detail.request.instance_name == "removed-8"


def _native_request(service, id, parent=None, *, role="engine", model_id=42):
    name = "llm_request"
    attrs = {"gen_ai.usage.prompt_tokens": 21}
    if service.startswith("sglang"):
        phase = role if role in ("prefill", "decode") else ""
        name = f"{phase} Req aaaaaaaa"
        attrs = {"rid": "aaaaaaab1234", "module": "sglang::request"}
    if role == "router":
        name = "POST /v1/chat/completions"
        attrs = {
            "http.request.method": "POST",
            "url.path": "/v1/chat/completions",
            "http.response.status_code": 200,
        }
    batch = _span(service, id, parent, name, 102, 2, **attrs)
    batch["resource"]["attributes"].extend(
        _attributes(
            **{
                "gpustack.model_id": str(model_id),
                "gpustack.engine_role": role,
                "gpustack.engine_instance_id": str(id + 10),
                "gpustack.engine_instance_name": f"removed-instance-{id}",
            }
        )
    )
    if role == "router":
        batch["scopeSpans"][0]["spans"][0]["kind"] = "SPAN_KIND_SERVER"
    return batch


def _native_search_client(batches):
    return _tempo_client(
        [{"traceID": _TRACE, "startTimeUnixNano": str(100 * _NS)}],
        {_TRACE: batches},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("service", ["vllm", "sglang"])
async def test_direct_engine_trace_has_the_same_list_and_detail_entry(service):
    engine = _native_request(service, 2, 1)
    upstream = _span("test-client", 1, None, "client call", 100, 5)
    phase = _span(service, 3, 2, "tokenize", 102, 1)
    # Workload resource attributes also appear on internal stages.
    phase["resource"] = engine["resource"]
    internal = _span(
        service,
        4,
        None,
        " Req bootstrap",
        103,
        1,
        rid="room",
        module="sglang::mooncake",
    )
    internal["resource"] = engine["resource"]
    batches = [upstream, engine, phase, internal]
    client = _native_search_client(batches)
    (row,) = await _trace_rows(client, instance_id=12)
    query = client.get.call_args_list[0].kwargs["params"]["q"]
    assert 'resource.gpustack.model_id="42"' in query
    assert "resource.gpustack.engine_role" in query
    assert "by(span:id)" not in query
    assert row.request_id is None
    assert row.entry_span_id == "0000000000000002"
    assert row.observation_source == "engine"
    assert row.instance_name == "removed-instance-2"
    assert row.status_code is None
    assert row.outcome == "unknown"
    assert row.request_metrics is None
    assert row.has_engine
    assert row.duration_ms == 2000
    detail = await get_request_trace(
        _client({"batches": batches}),
        "http://tempo",
        42,
        _TRACE,
        None,
        row.entry_span_id,
    )
    assert detail.request == row
    assert [span.id for span in detail.spans] == [
        "0000000000000002",
        "0000000000000003",
    ]


@pytest.mark.asyncio
async def test_router_and_platform_requests_do_not_duplicate_engine_entries():
    router = _native_request("vllm-router", 2, 1, role="router")
    dispatch = _span("vllm-router", 3, 2, "POST /v1/chat/completions", 102, 2)
    dispatch["resource"] = router["resource"]
    engine = _native_request("vllm.decode", 4, 3, role="decode")
    batches = [engine, dispatch, router]
    client = _native_search_client(batches)
    (row,) = await _trace_rows(client)
    assert (
        "(span:kind=server||span:kind=client)"
        in client.get.call_args_list[0].kwargs["params"]["q"]
    )
    assert row.observation_source == "router"
    assert row.path == "/v1/chat/completions"
    assert row.status_code == 200
    assert row.has_engine
    platform = _span(
        "gpustack.worker",
        1,
        None,
        "POST inference proxy",
        100,
        4,
        **{"gpustack.model_id": 42, "gpustack.request_id": _REQUEST},
    )
    (row,) = await _trace_rows(_native_search_client([platform, *batches]))
    assert row.observation_source == "platform"
    assert row.request_id == _REQUEST
    assert row.duration_ms == 4000
    detail = await get_request_trace(
        _client({"batches": [platform, *batches]}),
        "http://tempo",
        42,
        _TRACE,
        None,
        "0000000000000002",
    )
    assert detail.request == row


@pytest.mark.asyncio
async def test_direct_pd_entries_share_a_trace_but_have_distinct_details_and_pages():
    prefill = _native_request("sglang.prefill", 2, 1, role="prefill")
    decode = _native_request("sglang.decode", 3, 1, role="decode")
    foreign = _native_request("sglang", 4, model_id=43)
    batches = [prefill, decode, foreign]
    client = _client({"batches": batches})
    position = TracePagePosition(start=1, end=200, before=200)
    params = dict(
        page_size=1, identifier=_TRACE, outcome=None, instance_id=None, no_engine=False
    )
    page = await search_request_trace_page(
        client, "http://tempo", 42, position=position, **params
    )
    assert page.next_cursor
    second = await search_request_trace_page(
        client, "http://tempo", 42, position=position, **params
    )
    assert not second.next_cursor
    assert [row.engine_phase for row in page.items + second.items] == [
        "decode",
        "prefill",
    ]
    detail_client = _client({"batches": batches})
    for row in page.items + second.items:
        detail = await get_request_trace(
            detail_client, "http://tempo", 42, _TRACE, None, row.entry_span_id
        )
        assert detail.request == row
        assert len(detail.spans) == 1
        assert detail.spans[0].phase == row.engine_phase
    assert (
        await get_request_trace(
            detail_client, "http://tempo", 42, _TRACE, None, "0000000000000004"
        )
        is None
    )


@pytest.mark.asyncio
async def test_exact_trace_search_reads_complete_trace_without_search_projection():
    engine = _native_request("vllm", 3, 2)
    caller = _span("client", 2, None, "infer", 100, 4)
    client = _client({"batches": [caller, engine]})
    rows = await _trace_rows(client, identifier=_TRACE)

    (row,) = rows
    assert row.entry_span_id == "0000000000000003"
    assert row.observation_source == "engine"
    assert {call.args[0] for call in client.get.call_args_list} == {
        "http://tempo/api/search",
        f"http://tempo/api/traces/{_TRACE}",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["000abc", "abc", "Tenant-a/00Ab", 'req-"Q"'])
async def test_request_id_search_preserves_exact_identity(identifier):
    requests = [
        _span(
            "gpustack.worker",
            span_id,
            None,
            "POST inference proxy",
            100,
            2,
            **{"gpustack.model_id": 42, "gpustack.request_id": value},
        )
        for span_id, value in enumerate([identifier, identifier + "/other"], 1)
    ]
    client = _tempo_client(
        [{"traceID": _TRACE, "startTimeUnixNano": str(100 * _NS)}],
        {_TRACE: requests},
    )
    rows = await _trace_rows(client, identifier=identifier)
    assert [row.request_id for row in rows] == [identifier]
    (request,) = [
        call
        for call in client.get.call_args_list
        if call.args[0].endswith("/api/search")
    ]
    assert (
        f"span.gpustack.request_id={json.dumps(identifier)}"
        in request.kwargs["params"]["q"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identifier", ["548c943a36f8d9ec", "0000000000000000548c943a36f8d9ec"]
)
async def test_search_can_match_request_and_trace_ids_at_the_same_time(identifier):
    short_trace_id = "548c943a36f8d9ec"
    request = _span(
        "gpustack.worker",
        1,
        None,
        "POST inference proxy",
        100,
        2,
        **{"gpustack.model_id": 42, "gpustack.request_id": identifier},
    )
    engine = _native_request("vllm", 2)
    client = _tempo_client(
        [{"traceID": _TRACE, "startTimeUnixNano": str(100 * _NS)}],
        {_TRACE: [request], short_trace_id: [engine]},
    )

    rows = await _trace_rows(client, identifier=identifier)

    assert {(row.trace_id, row.request_id) for row in rows} == {
        (_TRACE, identifier),
        (short_trace_id, None),
    }
    record = await get_request_trace(client, "http://tempo", 42, identifier, None)
    assert record.request.trace_id == short_trace_id


@pytest.mark.asyncio
async def test_search_accepts_a_native_engine_request_id():
    engine = _native_request("sglang", 2)
    identifier = "chatcmpl-Engine/01"
    span = engine["scopeSpans"][0]["spans"][0]
    span["attributes"].extend(_attributes(**{"gen_ai.request.id": identifier}))
    client = _tempo_client(
        [{"traceID": _TRACE, "startTimeUnixNano": str(100 * _NS)}],
        {_TRACE: [engine]},
    )
    rows = await _trace_rows(client, identifier=identifier)

    (row,) = rows
    assert row.observation_source == "engine"
    assert row.trace_id == _TRACE
    assert row.request_id is None
    (search,) = [
        call
        for call in client.get.call_args_list
        if call.args[0].endswith("/api/search")
    ]
    assert f'span.gen_ai.request.id="{identifier}"' in search.kwargs["params"]["q"]


@pytest.mark.asyncio
async def test_exact_trace_does_not_advance_past_unread_request_matches(monkeypatch):
    monkeypatch.setattr(
        "gpustack.observability.request_traces._IDENTIFIER_SEARCH_BATCH_SIZE", 2
    )
    identifier = "abc"

    matches = [
        {"traceID": str(number), "startTimeUnixNano": start * _NS}
        for number, start in enumerate([103, 102, 101], 1)
    ]
    requests = {
        str(number): [
            _span(
                "gpustack.worker",
                number,
                None,
                "POST inference proxy",
                start,
                2,
                **{"gpustack.model_id": 42, "gpustack.request_id": identifier},
            )
        ]
        for number, start in enumerate([103, 102, 101], 1)
    }
    engine = _native_request("vllm", 4)
    span = engine["scopeSpans"][0]["spans"][0]
    span["startTimeUnixNano"] = str(100 * _NS)
    requests["abc"] = [engine]
    position = TracePagePosition(start=1, end=200, before=200)
    client = _paged_tempo_client(matches, requests)
    options = dict(
        page_size=2,
        identifier=identifier,
        outcome=None,
        instance_id=None,
        no_engine=False,
    )
    first = await search_request_trace_page(
        client, "http://tempo", 42, position=position, **options
    )
    assert first.next_cursor
    second = await search_request_trace_page(
        client, "http://tempo", 42, position=position, **options
    )
    assert not second.next_cursor
    assert [row.trace_id for row in first.items + second.items] == [
        "1",
        "2",
        "3",
        "abc",
    ]


@pytest.mark.asyncio
async def test_expired_identifier_candidate_still_advances_the_cursor(monkeypatch):
    monkeypatch.setattr(
        "gpustack.observability.request_traces._IDENTIFIER_SEARCH_BATCH_SIZE", 1
    )
    client = _paged_tempo_client(
        [{"traceID": _TRACE, "startTimeUnixNano": 100 * _NS}], {}
    )
    position = TracePagePosition(start=1, end=200, before=200)
    page = await search_request_trace_page(
        client,
        "http://tempo",
        42,
        position=position,
        page_size=10,
        identifier="Tenant/001",
        outcome=None,
        instance_id=None,
        no_engine=False,
    )

    assert page.items == []
    assert page.next_cursor is None
    assert position.boundary == (100 * _NS, _TRACE)
    assert position.skipped == 1


def _candidate(trace_number, started):
    return {"traceID": f"{trace_number:x}", "startTimeUnixNano": str(started)}


def _candidate_batches(candidate, *, status=200):
    batch = _span(
        "gpustack.worker",
        1,
        None,
        "POST inference proxy",
        0,
        1,
        **{"gpustack.model_id": 42, "http.response.status_code": status},
    )
    span = batch["scopeSpans"][0]["spans"][0]
    started = int(candidate["startTimeUnixNano"])
    span.update(startTimeUnixNano=str(started), endTimeUnixNano=str(started + _NS))
    return [batch]


def _paged_tempo_client(candidates, traces):
    client = _tempo_client([], traces)
    details = client.get.side_effect

    def get(url, **kwargs):
        if not url.endswith("/api/search"):
            return details(url, **kwargs)
        params = kwargs["params"]
        matches = [
            trace
            for trace in candidates
            if params["start"] * _NS
            <= int(trace["startTimeUnixNano"])
            < params["end"] * _NS
        ]
        return _client({"traces": matches[: params["limit"]]}).get(url, **kwargs)

    client.get.side_effect = get
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", ["identifier", "cursor"])
async def test_foreign_trace_cannot_advance_the_model_cursor(monkeypatch, lookup):
    monkeypatch.setattr(request_traces, "_IDENTIFIER_SEARCH_BATCH_SIZE", 1)
    monkeypatch.setattr(request_traces, "_TRACE_DETAIL_BATCH_SIZE", 1)
    own = _candidate(1, 102 * _NS)
    (own_batch,) = _candidate_batches(own)
    own_span = own_batch["scopeSpans"][0]["spans"][0]
    own_span["attributes"].extend(_attributes(**{"gpustack.request_id": _TRACE}))
    foreign_batch = _span(
        "gpustack.worker",
        2,
        None,
        "POST inference proxy",
        102,
        2,
        **{"gpustack.model_id": 78, "gpustack.request_id": _OTHER_REQUEST},
    )
    foreign_span = foreign_batch["scopeSpans"][0]["spans"][0]
    foreign_span["startTimeUnixNano"] = str(102 * _NS + 500_000_000)
    client = _paged_tempo_client([own], {"1": [own_batch], _TRACE: [foreign_batch]})
    position = TracePagePosition(
        start=1,
        end=200,
        before=103,
        partial=_TRACE if lookup == "cursor" else None,
    )

    page = await search_request_trace_page(
        client,
        "http://tempo",
        42,
        position=position,
        page_size=1,
        identifier=_TRACE if lookup == "identifier" else None,
        outcome=None,
        instance_id=None,
        no_engine=False,
    )

    assert [row.trace_id for row in page.items] == ["1"]
    assert page.next_cursor is None
    assert position.before == 103
    assert position.boundary == (102 * _NS, "1")
    assert position.partial is None


@pytest.mark.asyncio
@pytest.mark.parametrize("budget, expected", [(100, ["1"]), (10, [])])
async def test_filtered_page_scans_more_candidates_within_budget(
    monkeypatch, budget, expected
):
    monkeypatch.setattr(
        "gpustack.observability.request_traces._SEARCH_BATCH_SIZE", budget
    )
    candidates = [_candidate(i, 102 * _NS + i) for i in range(11, 0, -1)]
    client = _paged_tempo_client(
        candidates,
        {
            trace["traceID"]: _candidate_batches(
                trace, status=500 if trace["traceID"] == "1" else 200
            )
            for trace in candidates
        },
    )
    position = TracePagePosition(start=1, end=200, before=200)
    first_cursor = position.encode()
    page = await search_request_trace_page(
        client,
        "http://tempo",
        42,
        position=position,
        page_size=10,
        identifier=None,
        outcome="error",
        instance_id=None,
        no_engine=False,
    )
    assert page.cursor == first_cursor
    assert [row.trace_id for row in page.items] == expected
    assert bool(page.next_cursor) is (not expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "count, tied, legacy", [(501, False, 0), (501, False, 240), (251, True, 0)]
)
async def test_pages_preserve_traces_in_the_boundary_second(count, tied, legacy):
    candidates = [
        _candidate(i, 102 * _NS + (0 if tied else 10_000 * i))
        for i in range(count, 0, -1)
    ]
    candidates.append(_candidate(count + 1, 101 * _NS))
    client = _paged_tempo_client(
        candidates,
        {trace["traceID"]: _candidate_batches(trace) for trace in candidates},
    )
    position = TracePagePosition(
        start=1,
        end=200,
        before=103 if legacy else 200,
        seen=[trace["traceID"] for trace in candidates[:legacy]],
    )
    seen = []
    for _ in range((count - legacy + 1) // 10 + 1):
        page = await search_request_trace_page(
            client,
            "http://tempo",
            42,
            position=position,
            page_size=10,
            identifier=None,
            outcome="success",
            instance_id=None,
            no_engine=False,
        )
        seen.extend(row.trace_id for row in page.items)
        if not page.next_cursor:
            break
        if not legacy:
            assert len(page.next_cursor) < 256
        position = TracePagePosition.model_validate_json(
            base64.urlsafe_b64decode(page.next_cursor)
        )
        assert position.end == 200
        assert position.before == 103

    expected = [
        trace["traceID"]
        for trace in sorted(
            candidates[legacy:],
            key=lambda trace: (int(trace["startTimeUnixNano"]), trace["traceID"]),
            reverse=True,
        )
    ]
    assert seen == expected
    assert page.next_cursor is None
    assert not position.seen
    detail_lookups = []
    for call in client.get.call_args_list:
        url = call.args[0]
        if url.endswith("/api/search"):
            assert "trace:id!=" not in call.kwargs["params"]["q"]
        else:
            detail_lookups.append(url.rsplit("/", 1)[-1])
    assert detail_lookups == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("competing_traces", [1, 20])
async def test_pages_resume_partial_trace_after_late_arrivals(
    monkeypatch, competing_traces
):
    monkeypatch.setattr(
        "gpustack.observability.request_traces._TRACE_DETAIL_BATCH_SIZE", 2
    )
    old = _candidate(1, 102 * _NS)
    older_requests = []
    for index in range(3):
        (batch,) = _candidate_batches(old)
        span = batch["scopeSpans"][0]["spans"][0]
        span["spanId"] = base64.b64encode((index + 1).to_bytes(8, "big")).decode()
        span["attributes"].extend(
            _attributes(**{"gpustack.request_id": f"old-{index}"})
        )
        older_requests.append(batch)
    older = _candidate(2, 101 * _NS)
    candidates = [old, older]
    batches = {"1": older_requests, "2": _candidate_batches(older)}
    client = _paged_tempo_client(candidates, batches)
    position = TracePagePosition(start=1, end=200, before=200)
    options = dict(
        page_size=1, identifier=None, outcome=None, instance_id=None, no_engine=False
    )
    page = await search_request_trace_page(
        client, "http://tempo", 42, position=position, **options
    )
    assert [row.request_id for row in page.items] == ["old-2"]

    candidates.remove(old)
    for index in range(competing_traces):
        late_trace = _candidate(index + 3, 102 * _NS + 500_000_000 - index)
        candidates.insert(index, late_trace)
        batches[late_trace["traceID"]] = _candidate_batches(late_trace)

    for expected in ("old-1", "old-0"):
        position = TracePagePosition.model_validate_json(
            base64.urlsafe_b64decode(page.next_cursor)
        )
        page = await search_request_trace_page(
            client, "http://tempo", 42, position=position, **options
        )
        assert [row.request_id for row in page.items] == [expected]

    position = TracePagePosition.model_validate_json(
        base64.urlsafe_b64decode(page.next_cursor)
    )
    page = await search_request_trace_page(
        client, "http://tempo", 42, position=position, **options
    )
    assert [row.trace_id for row in page.items] == ["2"]
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_large_page_searches_once_for_its_candidates():
    candidates = [_candidate(i, 102 * _NS + i) for i in range(50, 0, -1)]
    client = _paged_tempo_client(
        candidates,
        {trace["traceID"]: _candidate_batches(trace) for trace in candidates},
    )
    page = await search_request_trace_page(
        client,
        "http://tempo",
        42,
        position=TracePagePosition(start=1, end=200, before=200),
        page_size=50,
        identifier=None,
        outcome=None,
        instance_id=None,
        no_engine=False,
    )

    assert len(page.items) == 50
    searches = [
        call.kwargs["params"]["limit"]
        for call in client.get.call_args_list
        if call.args[0].endswith("/api/search")
    ]
    assert searches == [51]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [None, "error"])
async def test_timeout_returns_only_finished_batches_and_resumes_after_them(
    monkeypatch, outcome
):
    timeout = 0.2
    monkeypatch.setattr(request_traces, "_TEMPO_QUERY_TIMEOUT", timeout)
    candidates = [_candidate(i, 102 * _NS + i) for i in range(25, 0, -1)]
    slow_ids = {f"{i:x}" for i in range(1, 16)}
    client = _paged_tempo_client(
        candidates,
        {
            trace["traceID"]: _candidate_batches(
                trace, status=500 if trace["traceID"] in slow_ids else 200
            )
            for trace in candidates
        },
    )
    lookup = request_traces._exact_trace_candidate
    cancelled = set()

    async def load(client, url, trace_id):
        if trace_id in slow_ids:
            try:
                await asyncio.Future()
            finally:
                cancelled.add(trace_id)
        return await lookup(client, url, trace_id)

    monkeypatch.setattr(request_traces, "_exact_trace_candidate", load)
    options = dict(
        page_size=25,
        identifier=None,
        outcome=outcome,
        instance_id=None,
        no_engine=False,
    )
    first = await asyncio.wait_for(
        search_request_trace_page(
            client,
            "http://tempo",
            42,
            position=TracePagePosition(start=1, end=200, before=200),
            **options,
        ),
        timeout=2,
    )
    finished = [f"{i:x}" for i in range(25, 15, -1)]
    assert {call.kwargs["timeout"].total for call in client.get.call_args_list} == {
        timeout
    }
    assert [row.trace_id for row in first.items] == ([] if outcome else finished)
    assert cancelled == {f"{i:x}" for i in range(15, 5, -1)}
    position = TracePagePosition.model_validate_json(
        base64.urlsafe_b64decode(first.next_cursor)
    )
    assert position.boundary == (102 * _NS + 16, finished[-1])
    assert position.skipped == len(finished)

    monkeypatch.setattr(request_traces, "_exact_trace_candidate", lookup)
    second = await search_request_trace_page(
        client, "http://tempo", 42, position=position, **options
    )
    assert [row.trace_id for row in second.items] == [
        f"{i:x}" for i in range(15, 0, -1)
    ]
    assert second.next_cursor is None


@pytest.mark.asyncio
async def test_timeout_before_any_batch_fails_as_unavailable(monkeypatch):
    monkeypatch.setattr(request_traces, "_TEMPO_QUERY_TIMEOUT", 0.02)

    async def load(*args):
        await asyncio.Future()

    monkeypatch.setattr(request_traces, "_tempo_get", load)
    async with asyncio.timeout(1):
        with pytest.raises(ServiceUnavailableException, match="timed out"):
            await _trace_rows(MagicMock())


def test_public_inference_paths_cover_all_inference_router_post_routes():
    from gpustack.observability.tracing import PUBLIC_INFERENCE_PATHS
    from gpustack.routes.routes import inference_router

    router_post_paths = {
        route.path for route in inference_router.routes if "POST" in route.methods
    }
    assert router_post_paths <= PUBLIC_INFERENCE_PATHS
    assert "/v1-openai/chat/completions" in PUBLIC_INFERENCE_PATHS
    assert "/v1-openai/chat/completions" in request_traces._search_query(42, None)


@pytest.mark.asyncio
async def test_legacy_v1_openai_paths_are_preserved_for_platform_and_router_spans():
    platform_span = _span(
        "gpustack.server",
        1,
        None,
        "POST /v1-openai/chat/completions",
        100,
        2,
        **{
            "gpustack.model_id": 42,
            "gpustack.model_instance_id": 7,
            "gpustack.request_id": _REQUEST,
            "http.route": "/v1-openai/chat/completions",
            "http.response.status_code": 200,
        },
    )
    router_span = {
        "resource": {
            "attributes": _attributes(
                **{
                    "service.name": "vllm-router",
                    "gpustack.model_id": "42",
                    "gpustack.engine_instance_id": "9",
                    "gpustack.engine_role": "router",
                }
            )
        },
        "scopeSpans": [
            {
                "spans": [
                    {
                        "spanId": base64.b64encode((2).to_bytes(8, "big")).decode(),
                        "name": "POST /v1-openai/chat/completions",
                        "kind": "SPAN_KIND_SERVER",
                        "startTimeUnixNano": str(105 * _NS),
                        "endTimeUnixNano": str(106 * _NS),
                        "attributes": _attributes(
                            **{
                                "http.request.method": "POST",
                                "url.path": "/v1-openai/chat/completions",
                                "http.response.status_code": 200,
                            }
                        ),
                    }
                ]
            }
        ],
    }
    search = [
        {"traceID": _TRACE, "startTimeUnixNano": str(105 * _NS)},
    ]
    rows = await _trace_rows(
        _tempo_client(search, {_TRACE: [platform_span, router_span]})
    )
    assert [(row.observation_source, row.path) for row in rows] == [
        ("router", "/v1-openai/chat/completions"),
        ("platform", "/v1-openai/chat/completions"),
    ]


@pytest.mark.asyncio
async def test_boundary_cursor_normalizes_uppercase_and_leading_zeros():
    newer = _candidate(11, 102 * _NS)
    boundary_trace = _candidate(10, 102 * _NS)
    older = _candidate(9, 102 * _NS)
    candidates = [newer, boundary_trace, older]
    client = _paged_tempo_client(
        candidates,
        {trace["traceID"]: _candidate_batches(trace) for trace in candidates},
    )
    position = TracePagePosition(
        start=1,
        end=200,
        before=103,
        boundary=(102 * _NS, "0000000000000000000000000000000A"),
        skipped=2,
    )

    page = await search_request_trace_page(
        client,
        "http://tempo",
        42,
        position=position,
        page_size=10,
        identifier=None,
        outcome=None,
        instance_id=None,
        no_engine=False,
    )

    assert [row.trace_id for row in page.items] == ["9"]
    assert page.next_cursor is None
