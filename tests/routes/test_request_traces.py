import base64
import json
from types import SimpleNamespace
import ssl
import time
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.config.config import Config
from gpustack.observability import request_traces as tempo_request_traces
from gpustack.routes import request_traces
from gpustack.routes.routes import management_router
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.app import create_app
from gpustack.server.db import get_session

_TRACE_URL = "/v2/models/42/request-traces"
_TRACE_ID = "abcd0000000000000000000000000000"


@pytest.fixture(scope="module")
def trace_app():
    app = FastAPI()
    app.include_router(management_router)
    register_handlers(app)
    return app


@pytest.fixture
def trace_client(monkeypatch, trace_app):
    ctx = TenantContext(
        user=SimpleNamespace(id=7, kind=PrincipalType.USER),
        is_platform_admin=True,
        current_principal_id=11,
        org_role=None,
    )
    cfg = SimpleNamespace(
        effective_trace_otlp_endpoint=None,
        get_trace_query_url=Mock(return_value=None),
        log_otlp_endpoint=None,
    )
    trace_app.dependency_overrides[get_current_user] = lambda: ctx.user
    trace_app.dependency_overrides[get_tenant_context] = lambda: ctx
    trace_app.dependency_overrides[get_session] = lambda: None
    monkeypatch.setattr(request_traces, "get_global_config", lambda: cfg)
    monkeypatch.setattr(
        request_traces.Model,
        "one_by_id",
        AsyncMock(return_value=SimpleNamespace(owner_principal_id=11)),
    )
    trace_app.state.http_client_no_proxy = None
    with TestClient(trace_app) as client:
        yield client, ctx, cfg
    trace_app.dependency_overrides.clear()


@pytest.mark.parametrize("tracing", [False, True])
@pytest.mark.parametrize("logs", [False, True])
def test_capabilities_report_trace_availability(trace_client, tracing, logs):
    client, _, cfg = trace_client
    if tracing:
        cfg.effective_trace_otlp_endpoint = "tempo:4317"
        cfg.get_trace_query_url.return_value = "http://tempo:3200"
    if logs:
        cfg.log_otlp_endpoint = "https://loki.test/otlp/v1/logs"
    response = client.get(f"{_TRACE_URL}/capabilities")
    assert response.status_code == 200
    assert response.json() == {
        "tracing_enabled": tracing,
        "trace_query_configured": tracing,
    }


@pytest.mark.parametrize("suffix", ["/capabilities", "", f"/{_TRACE_ID}"])
@pytest.mark.parametrize(
    "role,principal_id,personal,is_admin,system,expected",
    [
        (OrgRole.OWNER, 11, False, False, None, 200),
        (OrgRole.OWNER, 12, False, False, None, 404),
        (OrgRole.MEMBER, 11, False, False, None, 403),
        (None, 11, True, False, None, 403),
        (None, 11, False, True, None, 200),
        (None, 12, False, True, None, 404),
        (None, None, False, True, None, 200),
        (None, None, False, False, "worker", 403),
        (None, None, False, False, "server", 403),
    ],
)
def test_query_routes_check_role_and_model_visibility(
    trace_client, suffix, role, principal_id, personal, is_admin, system, expected
):
    client, ctx, cfg = trace_client
    ctx.org_role = role
    ctx.current_principal_id = principal_id
    ctx.current_is_personal_scope = personal
    ctx.is_platform_admin = is_admin
    if personal:
        ctx.user.id = principal_id
    if system:
        ctx.user.kind = PrincipalType.SYSTEM
        ctx.user.name = "system/worker/test"
        ctx.user.id = None if system == "server" else 7
    # Capabilities is available without Tempo. The list and detail resolve
    # caller/model visibility before checking the missing Tempo configuration.
    client.app.state.http_client_no_proxy = object()
    if expected == 200 and suffix != "/capabilities":
        expected = 503
    assert client.get(_TRACE_URL + suffix).status_code == expected
    if expected in (403, 404):
        cfg.get_trace_query_url.assert_not_called()


@pytest.mark.parametrize("view", ["list", "detail"])
def test_owner_only_gets_own_branch_of_exact_trace(trace_client, monkeypatch, view):
    client, ctx, cfg = trace_client
    ctx.is_platform_admin = False
    ctx.org_role = OrgRole.OWNER
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    now = time.time_ns()
    own_span_id = "0000000000000002"
    foreign_span_id = "0000000000000001"

    def span(span_id, parent_id, model_id, request_id):
        return {
            "spanId": span_id,
            "parentSpanId": parent_id,
            "name": request_id,
            "startTimeUnixNano": str(now - 500_000_000),
            "endTimeUnixNano": str(now),
            "attributes": [
                {"key": "gpustack.model_id", "value": {"intValue": model_id}},
                {"key": "gpustack.request_id", "value": {"stringValue": request_id}},
            ],
        }

    payload = {
        "batches": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "gpustack.server"},
                        },
                    ],
                },
                "scopeSpans": [
                    {
                        "spans": [
                            span(foreign_span_id, None, 99, "foreign-request"),
                            span(own_span_id, foreign_span_id, 42, "own-request"),
                        ],
                    },
                ],
            }
        ]
    }

    async def get_tempo(_client, url, params=None):
        if url.endswith("/api/search"):
            assert "span.gpustack.model_id=42" in params["q"]
            assert 'resource.gpustack.model_id="42"' in params["q"]
            return {"traces": []}
        assert url == f"http://tempo:3200/api/traces/{_TRACE_ID}"
        return payload

    monkeypatch.setattr(tempo_request_traces, "_tempo_get", get_tempo)
    session = SimpleNamespace(
        exec=AsyncMock(return_value=[("own-request", 8, True, 9, 10)])
    )
    client.app.dependency_overrides[get_session] = lambda: session

    detail_url = f"{_TRACE_URL}/{_TRACE_ID}"
    response = (
        client.get(_TRACE_URL, params={"q": _TRACE_ID})
        if view == "list"
        else client.get(detail_url)
    )
    assert response.status_code == 200, response.text
    result = response.json()
    if view == "list":
        (row,) = result["items"]
    else:
        row = result["request"]
        (own_span,) = result["spans"]
        assert own_span["id"] == own_span_id
        assert own_span["parent_id"] is None
        assert result["span_count"] == 1
        assert (
            client.get(
                detail_url, params={"entry_span_id": foreign_span_id}
            ).status_code
            == 404
        )
        assert (
            client.get(detail_url, params={"request_id": "foreign-request"}).status_code
            == 404
        )
    assert row["request_id"] == "own-request"
    assert row["entry_span_id"] == own_span_id
    assert row["request_metrics"] == {
        "ttft_ms": 8,
        "input_tokens": 9,
        "output_tokens": 10,
    }
    session.exec.assert_awaited_once()
    statement = session.exec.await_args.args[0].compile()
    assert statement.params["model_id_1"] == 42
    assert statement.params["owner_principal_id_1"] == 11
    assert statement.params["request_id_1"] == ["own-request"]


def test_custom_trace_query_preserves_exact_time_bounds(trace_client, monkeypatch):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    search = AsyncMock(return_value=request_traces.RequestTracePage(items=[]))
    monkeypatch.setattr(request_traces, "search_request_trace_page", search)
    end = int(time.time()) - 300
    start = end - 23

    response = client.get(_TRACE_URL, params={"start": start, "end": end})
    assert response.status_code == 200
    position = search.await_args.kwargs["position"]
    assert (position.start, position.end) == (start, end)


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("548C943A36F8D9EC", "548C943A36F8D9EC"),
        ("0000000000000000548c943a36f8d9ec", "0000000000000000548c943a36f8d9ec"),
        ("8559ada689f94d5bafbe6de1d37447e9", "8559ada689f94d5bafbe6de1d37447e9"),
        (
            "a0000000-0000-4000-8000-000000000001",
            "a0000000-0000-4000-8000-000000000001",
        ),
        ("  Tenant-a/0012  ", "  Tenant-a/0012  "),
        ("", None),
    ],
)
def test_trace_list_accepts_copied_identifiers(
    trace_client, monkeypatch, identifier, expected
):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    search = AsyncMock(return_value=request_traces.RequestTracePage(items=[]))
    monkeypatch.setattr(request_traces, "search_request_trace_page", search)

    response = client.get(_TRACE_URL, params={"q": identifier, "page_size": 10})
    assert response.status_code == 200
    assert search.await_args.kwargs["identifier"] == expected


def test_unknown_identifier_returns_empty_page(trace_client, monkeypatch):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    search = AsyncMock(return_value={"traces": []})
    monkeypatch.setattr("gpustack.observability.request_traces._tempo_get", search)

    response = client.get(_TRACE_URL, params={"q": "chatcmpl-example", "page_size": 10})
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["next_cursor"] is None
    search.assert_awaited_once()
    assert (
        'span.gpustack.request_id="chatcmpl-example"' in search.await_args.args[2]["q"]
    )


def test_trace_list_defaults_to_a_single_page(trace_client, monkeypatch):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    search = AsyncMock(return_value={"traces": []})
    monkeypatch.setattr("gpustack.observability.request_traces._tempo_get", search)

    response = client.get(_TRACE_URL)

    assert response.status_code == 200
    page = response.json()
    assert page["items"] == []
    assert page["cursor"]
    assert page["next_cursor"] is None
    search.assert_awaited_once()
    assert search.await_args.args[2]["limit"] == 10


@pytest.mark.parametrize("credentials_available", [True, False])
def test_trace_queries_use_the_dedicated_tls_client(
    trace_client, monkeypatch, credentials_available
):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "https://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    tls_client = object() if credentials_available else None
    monkeypatch.setattr(
        client.app.state, "tempo_query_client", tls_client, raising=False
    )
    search = AsyncMock(return_value=request_traces.RequestTracePage(items=[]))
    monkeypatch.setattr(request_traces, "search_request_trace_page", search)

    response = client.get(_TRACE_URL, params={"page_size": 10})

    if credentials_available:
        assert response.status_code == 200
        assert search.await_args.args[0] is tls_client
    else:
        assert response.status_code == 503
        search.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutual_tls", [False, True])
async def test_tempo_connection_pool_is_independent_and_closed_with_app(
    tmp_path, monkeypatch, mutual_tls
):
    cfg = Config(
        data_dir=str(tmp_path),
        gateway_mode="disabled",
        jwt_secret_key="test-key",
        trace_otlp_endpoint="tempo:4317",
        trace_query_url="https://tempo:3200" if mutual_tls else "http://tempo:3200",
        trace_query_tls_dir=str(tmp_path) if mutual_tls else None,
    )
    context = ssl.create_default_context()
    load_cert_chain = Mock()
    monkeypatch.setattr(context, "load_cert_chain", load_cert_chain)
    create_context = Mock(return_value=context)
    monkeypatch.setattr(
        "gpustack.server.app.ssl.create_default_context", create_context
    )

    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        tempo = app.state.tempo_query_client
        shared = app.state.http_client_no_proxy
        assert tempo.connector is not shared.connector
        assert not tempo.connector.force_close
        assert shared.connector.force_close
        assert not tempo.trust_env

    assert tempo.closed
    if mutual_tls:
        create_context.assert_called_once_with(cafile=str(tmp_path / "ca.crt"))
        load_cert_chain.assert_called_once_with(
            tmp_path / "client.crt", tmp_path / "client.key"
        )
    else:
        create_context.assert_not_called()
        load_cert_chain.assert_not_called()


@pytest.mark.parametrize(
    ("window", "seconds"), [("5m", 300), ("3h", 10800), ("7d", 604800)]
)
def test_trace_search_supports_the_monitoring_presets(
    trace_client, monkeypatch, window, seconds
):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    search = AsyncMock(return_value=request_traces.RequestTracePage(items=[]))
    monkeypatch.setattr(request_traces, "search_request_trace_page", search)

    response = client.get(_TRACE_URL, params={"window": window})
    assert response.status_code == 200
    position = search.await_args.kwargs["position"]
    assert position.end - position.start == seconds

    response = client.get(_TRACE_URL, params={"window": "8d"})
    assert response.status_code == 400
    assert search.await_count == 1


@pytest.mark.parametrize(
    "offsets", [(None, -600), (-300, -600), (-604900, -10), (-10, 600)]
)
def test_invalid_trace_range_never_queries_tempo(trace_client, monkeypatch, offsets):
    client, _, _ = trace_client
    search = AsyncMock()
    monkeypatch.setattr(request_traces, "search_request_trace_page", search)
    now = int(time.time())
    params = {
        key: now + offset
        for key, offset in zip(("start", "end"), offsets)
        if offset is not None
    }
    response = client.get(_TRACE_URL, params=params)
    assert response.status_code == 400
    search.assert_not_awaited()


def test_cursor_pins_24_hour_window(trace_client, monkeypatch):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    query = AsyncMock(return_value=request_traces.RequestTracePage(items=[]))
    monkeypatch.setattr(request_traces, "search_request_trace_page", query)
    expected_now = int(time.time()) - 60
    url = _TRACE_URL
    with monkeypatch.context() as patch:
        patch.setattr(
            request_traces, "time", SimpleNamespace(time=lambda: expected_now)
        )
        assert client.get(url, params={"page_size": 10}).status_code == 200
        first = query.await_args.kwargs["position"]
        assert first.end - first.start == 86400
        patch.setattr(
            request_traces, "time", SimpleNamespace(time=lambda: expected_now + 5)
        )
        assert (
            client.get(
                url, params={"page_size": 10, "cursor": first.encode()}
            ).status_code
            == 200
        )
        assert query.await_args.kwargs["position"] == first
    assert client.get(url, params={"cursor": "invalid"}).status_code == 400
    assert query.await_count == 2


def test_dense_trace_cursor_fits_an_8_kib_request_line(trace_client, monkeypatch):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    query = AsyncMock(return_value=request_traces.RequestTracePage(items=[]))
    monkeypatch.setattr(request_traces, "search_request_trace_page", query)

    before = int(time.time())
    position = request_traces.TracePagePosition(
        start=before - 300,
        end=before,
        before=before,
        seen=["a" * 30 + f"{index:02x}" for index in range(239)] + ["f"],
        partial="b" * 32,
        offset=7,
    )
    url = _TRACE_URL
    response = client.get(url, params={"cursor": position.encode()})
    assert response.status_code == 200
    assert len(b"GET " + response.request.url.raw_path + b" HTTP/1.1\r\n") < 8192
    assert query.await_args.kwargs["position"] == position

    legacy = base64.urlsafe_b64encode(json.dumps(position.model_dump()).encode())
    assert client.get(url, params={"cursor": legacy.decode()}).status_code == 200
    assert query.await_args.kwargs["position"] == position


@pytest.mark.parametrize("seen", ["!!!!", "YQ==", "A" * 5152])
def test_trace_cursor_rejects_invalid_packed_ids(trace_client, seen):
    client, _, _ = trace_client
    before = int(time.time())
    payload = {"start": before - 300, "end": before, "before": before, "seen": seen}
    cursor = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()

    assert client.get(_TRACE_URL, params={"cursor": cursor}).status_code == 400


@pytest.mark.parametrize("view", ["page", "detail"])
def test_trace_views_use_gateway_request_metrics(trace_client, monkeypatch, view):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    trace_id = "548c943a36f8d9ec"
    request_ids = [f"f0000000-0000-4000-8000-{index:012d}" for index in range(5)]
    rows = [
        request_traces.RequestTraceRecord(
            trace_id=trace_id,
            request_id=request_id,
            started_at_ms=0,
            duration_ms=200,
            observed_by="gpustack.higress",
            path="/v1/chat/completions",
            status_code=200,
            outcome="success",
            stream=True,
            instance_id=7,
            has_engine=True,
            engine={"ttft_ms": 95, "input_tokens": 13, "output_tokens": 99},
        )
        for request_id in request_ids
    ]
    metrics = [
        (request_ids[0], 123, True, 21, 2),
        (request_ids[1], 27, False, 9, 4),
        (request_ids[2], 70, False, 99, 0),
        (request_ids[2], 200, True, 21, 3),
        (request_ids[3], None, True, 21, 0),
    ]
    session = SimpleNamespace(
        exec=AsyncMock(return_value=metrics[:1] if view == "detail" else metrics)
    )
    client.app.dependency_overrides[get_session] = lambda: session

    url = _TRACE_URL
    params = {}
    if view == "detail":
        monkeypatch.setattr(
            request_traces,
            "get_request_trace",
            AsyncMock(
                return_value=request_traces.RequestTraceDetail(
                    request=rows[0], spans=[], span_count=0
                )
            ),
        )
        url += f"/{trace_id}"
        params["request_id"] = request_ids[0]
    else:
        monkeypatch.setattr(
            request_traces,
            "search_request_trace_page",
            AsyncMock(return_value=request_traces.RequestTracePage(items=rows)),
        )

    response = client.get(url, params=params)
    assert response.status_code == 200
    data = response.json()
    expected = {"ttft_ms": 123, "input_tokens": 21, "output_tokens": 2}
    if view == "detail":
        assert data["request"]["request_metrics"] == expected
        assert data["request"]["engine"]["ttft_ms"] == 95
    else:
        assert [row["request_metrics"] for row in data["items"]] == [
            expected,
            {"ttft_ms": 27, "input_tokens": None, "output_tokens": None},
            None,
            {"ttft_ms": None, "input_tokens": 21, "output_tokens": 0},
            None,
        ]
    session.exec.assert_awaited_once()
    statement = session.exec.await_args.args[0].compile()
    assert "model_usage_details.model_id =" in str(statement)
    assert "model_usage_details.owner_principal_id =" in str(statement)
    assert set(statement.params["request_id_1"]) == set(
        request_ids[:1] if view == "detail" else request_ids
    )


def test_native_trace_detail_selects_the_entry_without_querying_usage(
    trace_client, monkeypatch
):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    http = client.app.state.http_client_no_proxy = object()
    trace_id = "abcd0000000000000000000000000000"
    entry_span_id = "abcdef0000000001"
    detail = request_traces.RequestTraceDetail(
        request=request_traces.RequestTraceRecord(
            trace_id=trace_id,
            entry_span_id=entry_span_id,
            observation_source="engine",
            request_id=None,
            started_at_ms=0,
            duration_ms=123,
            observed_by="vllm",
            path=None,
            status_code=None,
            outcome="unknown",
            stream=None,
            instance_id=7,
            has_engine=True,
            engine=None,
        ),
        spans=[],
        span_count=0,
    )
    lookup = AsyncMock(return_value=detail)
    monkeypatch.setattr(request_traces, "get_request_trace", lookup)
    session = SimpleNamespace(exec=AsyncMock())
    client.app.dependency_overrides[get_session] = lambda: session
    url = f"{_TRACE_URL}/{trace_id}"

    response = client.get(
        url, params={"entry_span_id": entry_span_id.upper(), "request_id": "Tenant/001"}
    )
    assert response.status_code == 200
    assert response.json()["request"]["entry_span_id"] == entry_span_id
    lookup.assert_awaited_once_with(
        http, "http://tempo:3200", 42, trace_id, "Tenant/001", entry_span_id
    )
    session.exec.assert_not_awaited()
    assert client.get(url, params={"entry_span_id": "invalid"}).status_code == 422
