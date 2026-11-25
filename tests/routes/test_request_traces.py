from types import SimpleNamespace
import ssl
import time
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.config.config import Config
from gpustack.routes import request_traces
from gpustack.schemas.principals import PrincipalType
from gpustack.server.app import create_app
from gpustack.server.db import get_session


@pytest.fixture
def trace_client(monkeypatch):
    app = FastAPI()
    app.include_router(request_traces.router, prefix="/models")
    register_handlers(app)
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=True,
        current_principal_id=11,
        org_role=None,
    )
    cfg = SimpleNamespace(
        effective_trace_otlp_endpoint=None,
        get_trace_query_url=Mock(return_value=None),
        log_otlp_endpoint=None,
    )
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    app.dependency_overrides[get_session] = lambda: None
    monkeypatch.setattr(request_traces, "get_global_config", lambda: cfg)
    monkeypatch.setattr(
        request_traces.Model,
        "one_by_id",
        AsyncMock(return_value=SimpleNamespace(owner_principal_id=11)),
    )
    with TestClient(app) as client:
        yield client, ctx, cfg


@pytest.mark.parametrize("tracing", [False, True])
@pytest.mark.parametrize("logs", [False, True])
def test_capabilities_report_trace_availability(trace_client, tracing, logs):
    client, _, cfg = trace_client
    if tracing:
        cfg.effective_trace_otlp_endpoint = "tempo:4317"
        cfg.get_trace_query_url.return_value = "http://tempo:3200"
    if logs:
        cfg.log_otlp_endpoint = "https://loki.test/otlp/v1/logs"
    response = client.get("/models/42/request-traces/capabilities")
    assert response.status_code == 200
    assert response.json() == {
        "tracing_enabled": tracing,
        "trace_query_configured": tracing,
    }


@pytest.mark.parametrize(
    "suffix", ["/capabilities", "", "/abcd0000000000000000000000000000"]
)
def test_query_routes_check_platform_role_and_model_visibility(trace_client, suffix):
    client, ctx, _ = trace_client
    url = "/models/42/request-traces" + suffix
    ctx.is_platform_admin = False
    assert client.get(url).status_code == 403
    ctx.is_platform_admin = True
    ctx.current_principal_id = 12
    assert client.get(url).status_code == 404


def test_custom_trace_query_preserves_exact_time_bounds(trace_client, monkeypatch):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    search = AsyncMock(return_value=request_traces.RequestTracePage(items=[]))
    monkeypatch.setattr(request_traces, "search_request_trace_page", search)
    end = int(time.time()) - 300
    start = end - 23

    response = client.get(
        "/models/42/request-traces", params={"start": start, "end": end}
    )
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

    response = client.get(
        "/models/42/request-traces", params={"q": identifier, "page_size": 10}
    )
    assert response.status_code == 200
    assert search.await_args.kwargs["identifier"] == expected


def test_unknown_identifier_returns_empty_page(trace_client, monkeypatch):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    search = AsyncMock(return_value={"traces": []})
    monkeypatch.setattr("gpustack.observability.request_traces._tempo_get", search)

    response = client.get(
        "/models/42/request-traces", params={"q": "chatcmpl-example", "page_size": 10}
    )
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

    response = client.get("/models/42/request-traces")

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
    client.app.state.tempo_query_client = tls_client
    search = AsyncMock(return_value=request_traces.RequestTracePage(items=[]))
    monkeypatch.setattr(request_traces, "search_request_trace_page", search)

    response = client.get("/models/42/request-traces", params={"page_size": 10})

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

    response = client.get("/models/42/request-traces", params={"window": window})
    assert response.status_code == 200
    position = search.await_args.kwargs["position"]
    assert position.end - position.start == seconds

    response = client.get("/models/42/request-traces", params={"window": "8d"})
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
    response = client.get("/models/42/request-traces", params=params)
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
    url = "/models/42/request-traces"
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

    url = "/models/42/request-traces"
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
    url = f"/models/42/request-traces/{trace_id}"

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
