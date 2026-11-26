import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from gpustack.api.exceptions import NotFoundException, ServiceUnavailableException
from gpustack.routes import openai as openai_route
from tests.utils.mock import mock_async_session


def _stub_resolution(monkeypatch, *, route):
    """Wire proxy_request_by_model up to the point of route resolution.

    ``resolve_route_targets`` always comes back empty here (the branch under
    test), and ``get_by_name`` returns ``route``, which is what decides whether
    the caller sees 404 or 503.
    """
    monkeypatch.setattr(openai_route, "async_session", lambda: mock_async_session())
    monkeypatch.setattr(
        openai_route,
        "parse_request_body",
        AsyncMock(return_value=("qwen3", False, {"model": "qwen3"}, None)),
    )
    monkeypatch.setattr(
        openai_route.UserService,
        "model_allowed_for_user",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        "resolve_route_targets",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        "get_by_name",
        AsyncMock(return_value=route),
    )


def _request():
    request = MagicMock()
    request.url.path = "/v1-openai/chat/completions"
    request.receive = AsyncMock(side_effect=asyncio.Event().wait)
    return request


@pytest.mark.asyncio
async def test_unknown_model_name_is_not_found(monkeypatch):
    """No route by that name really is a 404."""
    _stub_resolution(monkeypatch, route=None)

    with pytest.raises(NotFoundException) as exc:
        await openai_route.proxy_request_by_model(
            request=_request(), user=SimpleNamespace(id=1)
        )

    assert exc.value.status_code == 404
    assert exc.value.message == "Model not found"


@pytest.mark.asyncio
async def test_deployed_model_with_no_active_target_is_unavailable(monkeypatch):
    """A route whose targets have all gone UNAVAILABLE must not read as a
    missing model.

    ``ready_replicas`` drops to 0 for every model on a worker as soon as that
    worker misses /healthz, which flips each target to UNAVAILABLE and empties
    ``resolve_route_targets``. Returning 404 there tells the client a deployed
    model no longer exists and that retrying is pointless. It has to be 503.
    """
    _stub_resolution(monkeypatch, route=SimpleNamespace(id=7, name="qwen3"))

    with pytest.raises(ServiceUnavailableException) as exc:
        await openai_route.proxy_request_by_model(
            request=_request(), user=SimpleNamespace(id=1)
        )

    assert exc.value.status_code == 503
    assert exc.value.message == "No running instances available"


@pytest.mark.asyncio
@pytest.mark.parametrize('managed', [False, 'rank', 'hybrid'])
@pytest.mark.parametrize('deleted', [False, True])
async def test_proxy_preserves_api_path_and_selects_rank_owner(
    monkeypatch, caplog, managed, deleted
):
    from gpustack.schemas.models import ModelInstance
    from gpustack.utils.model_instance_workers import ModelInstanceEndpoint
    from tests.utils.model import new_model_instance

    _stub_resolution(monkeypatch, route=SimpleNamespace(id=7, name='qwen3'))
    target = SimpleNamespace(weight=100, model_id=1, overridden_model_name=None)
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        'resolve_route_targets',
        AsyncMock(return_value=[target]),
    )
    monkeypatch.setattr(
        openai_route.ModelService,
        'get_by_id',
        AsyncMock(return_value=SimpleNamespace(id=1)),
    )
    instance = new_model_instance(7, 'moe', 1, worker_id=1)
    instance.worker_ip, instance.port = '10.0.0.1', 41000
    monkeypatch.setattr(
        openai_route, 'get_running_instance', AsyncMock(return_value=instance)
    )
    monkeypatch.setattr(
        openai_route, 'get_model_instance_workloads', lambda _: [] if managed else None
    )
    refresh = AsyncMock(return_value=instance)
    monkeypatch.setattr(ModelInstance, 'one_by_id', refresh)
    monkeypatch.setattr(
        openai_route,
        'model_instance_endpoints',
        lambda _: [
            ModelInstanceEndpoint(
                1,
                7,
                2,
                41001,
                3,
                None if managed == 'hybrid' else 1,
                component='vllm_hybrid' if managed == 'hybrid' else None,
            )
        ],
    )
    get_worker = AsyncMock(
        return_value=SimpleNamespace(
            id=2 if managed else 1,
            ip='10.0.0.2',
            deleted_at='deleted' if deleted else None,
        )
    )
    monkeypatch.setattr(openai_route.WorkerService, 'get_by_id', get_worker)
    monkeypatch.setattr(openai_route, 'mutate_request', MagicMock())
    monkeypatch.setattr(
        openai_route,
        '_prepare_proxy_request',
        lambda request, body, form, headers, **kwargs: (headers, b'{}'),
    )
    forward = AsyncMock(return_value=(SimpleNamespace(status=200, headers={}), b'{}'))
    monkeypatch.setattr(openai_route, 'request_to_worker', forward)
    if deleted:
        from gpustack.api.exceptions import InternalServerErrorException

        with pytest.raises(InternalServerErrorException) as error:
            await openai_route.proxy_request_by_model(
                request=_request(), user=SimpleNamespace(id=1)
            )
        assert error.value.message == f"Worker with ID {2 if managed else 1} not found"
        forward.assert_not_awaited()
        return
    caplog.set_level('DEBUG', logger=openai_route.__name__)
    await openai_route.proxy_request_by_model(
        request=_request(), user=SimpleNamespace(id=1)
    )
    expected_port = 41001 if managed else 41000
    assert f'Worker {2 if managed else 1} at 10.0.0.2:{expected_port}' in caplog.text
    assert '10.0.0.1:41000' not in caplog.text
    assert forward.call_args.kwargs['path'] == 'v1/chat/completions'
    assert forward.call_args.kwargs['headers'][openai_route.router_header_key] == (
        'model-1-7-w2.static'
        if managed == 'hybrid'
        else ('model-1-7-dp1.static' if managed else 'model-1-7.static')
    )
    get_worker.assert_awaited_once_with(2 if managed else 1)
    if managed:
        refresh.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'component,status',
    [('sglang_smg', 200), ('sglang_smg', 422), ('vllm_hybrid', 200), (None, 200)],
)
async def test_server_proxy_sends_component_specific_payload(
    monkeypatch, component, status
):
    import json
    from aiohttp import ClientSession, web
    from gpustack.schemas.models import ModelInstance
    from gpustack.schemas.workers import Worker
    from gpustack.utils.model_instance_workers import ModelInstanceEndpoint
    from tests.routes.test_worker_proxy import _fake_inference_server
    from tests.utils.model import new_model_instance

    _stub_resolution(monkeypatch, route=SimpleNamespace(id=7, name='qwen3'))
    payload = {
        'model': 'qwen3',
        'prompt': '你好',
        'routed_dp_rank': 9,
        'data_parallel_rank': 8,
    }
    monkeypatch.setattr(
        openai_route,
        'parse_request_body',
        AsyncMock(return_value=('qwen3', False, payload, None)),
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        'resolve_route_targets',
        AsyncMock(
            return_value=[
                SimpleNamespace(weight=100, model_id=1, overridden_model_name=None)
            ]
        ),
    )
    monkeypatch.setattr(
        openai_route.ModelService,
        'get_by_id',
        AsyncMock(return_value=SimpleNamespace(id=1)),
    )
    instance = new_model_instance(7, 'moe', 1, worker_id=1)
    monkeypatch.setattr(
        openai_route, 'get_running_instance', AsyncMock(return_value=instance)
    )
    monkeypatch.setattr(openai_route, 'get_model_instance_workloads', lambda _: [])
    monkeypatch.setattr(ModelInstance, 'one_by_id', AsyncMock(return_value=instance))
    monkeypatch.setattr(
        openai_route,
        'model_instance_endpoints',
        lambda _: [
            ModelInstanceEndpoint(
                1,
                7,
                1,
                41000,
                3,
                component=component,
                attempt=2 if component == 'sglang_smg' else None,
            )
        ],
    )
    monkeypatch.setattr(openai_route, 'mutate_request', MagicMock())
    received = []

    async def handler(request):
        raw = await request.read()
        received.append((json.loads(raw), request.headers, raw, request.path))
        return web.json_response({'message': 'native response'}, status=status)

    async with _fake_inference_server(handler) as port:
        worker = Worker(id=1, ip='127.0.0.1', port=port, token='worker-token')
        monkeypatch.setattr(
            openai_route.WorkerService, 'get_by_id', AsyncMock(return_value=worker)
        )
        request = _request()
        request.method = 'POST'
        request.headers = {
            'Content-Type': 'application/json',
            'X-Data-Parallel-Rank': '9',
            'X-SMG-Routing-Key': 'conversation-123',
        }
        async with ClientSession() as client:
            request.app.state.http_client = client
            request.app.state.http_client_no_proxy = client
            response = await openai_route.proxy_request_by_model(
                request, SimpleNamespace(id=1)
            )
    assert response.status_code == status
    assert json.loads(response.body) == {'message': 'native response'}
    data, headers, raw, path = received[0]
    assert path == '/v1/chat/completions'
    assert data == payload
    assert headers.get('X-Data-Parallel-Rank') == '9'
    assert headers.get('X-SMG-Routing-Key') == 'conversation-123'
    assert int(headers['Content-Length']) == len(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, False),
        ({"stream": "false"}, False),
        ({"stream": "False"}, False),
        ({"stream": "0"}, False),
        ({"stream": "true"}, True),
        ({"stream": "True"}, True),
    ],
)
async def test_multipart_stream_field_is_parsed_as_a_boolean(fields, expected):
    """``stream`` arrives as text in a multipart body and is parsed to a bool.

    Only true-like text selects the streaming relay in proxy_request_by_model.
    """
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        _, stream, _, _ = await openai_route.parse_request_body(request)
        return {"stream": stream}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/probe",
            data={"model": "whisper", **fields},
            files={"file": ("a.wav", b"audio", "audio/wav")},
        )

    assert resp.status_code == 200
    assert resp.json() == {"stream": expected}


@pytest.mark.asyncio
async def test_resolve_route_targets_includes_provider_targets(monkeypatch):
    """Active provider-backed targets (model_id=None, provider_id set) are kept
    by resolve_route_targets while targets pointing at neither are dropped."""
    from gpustack.server import services as services_module

    route_id = 31
    targets = [
        SimpleNamespace(
            route_id=route_id,
            model_id=None,
            provider_id=11,
            overridden_model_name="gemini-2.5-flash",
            weight=100,
        ),
        SimpleNamespace(
            route_id=route_id,
            model_id=5,
            provider_id=None,
            overridden_model_name=None,
            weight=50,
        ),
        SimpleNamespace(
            route_id=route_id,
            model_id=None,
            provider_id=None,
            overridden_model_name="dangling",
            weight=10,
        ),
    ]
    monkeypatch.setattr(services_module, "platform_principal_id", lambda: 1)
    monkeypatch.setattr(
        services_module.ModelRouteTarget,
        "all_by_fields",
        AsyncMock(return_value=targets),
    )
    monkeypatch.setattr(
        services_module.ModelRoute,
        "all_by_fields",
        AsyncMock(return_value=[SimpleNamespace(id=route_id)]),
    )

    svc = services_module.ModelRouteService(MagicMock())
    resolved = await svc.resolve_route_targets("provider-route-mixed-targets")

    assert resolved == [
        services_module.RouteTargetResolution(
            model_id=None,
            overridden_model_name="gemini-2.5-flash",
            weight=100,
            provider_id=11,
        ),
        services_module.RouteTargetResolution(
            model_id=5,
            overridden_model_name=None,
            weight=50,
            provider_id=None,
        ),
    ]


class _FakeUpstreamResponse:
    def __init__(self, status: int, headers: dict, body: bytes):
        self.status = status
        self.headers = headers
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_provider_route_target_proxies_directly_in_server_mode(monkeypatch):
    """Active provider-backed route targets dispatch to the provider's OpenAI
    compatible endpoint when server-side OpenAI routing handles /v1 requests."""
    import json
    from gpustack.schemas.model_provider import GeminiConfig, ModelProviderTypeEnum
    from gpustack.server.services import RouteTargetResolution

    captured = {}

    class _FakeClient:
        def request(self, *, method, url, headers, data, proxy, timeout):
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            return _FakeUpstreamResponse(
                status=200,
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                    "Transfer-Encoding": "chunked",
                    "Set-Cookie": "upstream_session=abc",
                    "Connection": "keep-alive, X-Upstream-Internal",
                    "X-Upstream-Internal": "hop-only",
                },
                body=b'{"id":"chatcmpl-1","choices":[]}',
            )

    provider = SimpleNamespace(
        id=11,
        name="gemini-gateway",
        deleted_at=None,
        api_tokens=["test-gemini-token"],
        timeout=60,
        proxy_url=None,
        config=GeminiConfig(type=ModelProviderTypeEnum.GEMINI),
    )

    monkeypatch.setattr(openai_route, "async_session", lambda: mock_async_session())
    monkeypatch.setattr(
        openai_route,
        "parse_request_body",
        AsyncMock(
            return_value=(
                "gemini-flash-route",
                False,
                {
                    "model": "gemini-flash-route",
                    "messages": [{"role": "user", "content": "hi"}],
                    "seed": 42,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "top_k": 40,
                    "chat_template_kwargs": {"enable_thinking": True},
                    "temperature": 0.7,
                    "google": {"thinking_config": {"thinking_budget": 256}},
                    "stop": None,
                },
                None,
            )
        ),
    )
    monkeypatch.setattr(
        openai_route.UserService,
        "model_allowed_for_user",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        "resolve_route_targets",
        AsyncMock(
            return_value=[
                RouteTargetResolution(
                    model_id=None,
                    overridden_model_name="gemini-2.5-flash",
                    weight=100,
                    provider_id=11,
                )
            ]
        ),
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        "get_by_name",
        AsyncMock(return_value=SimpleNamespace(id=42, name="gemini-flash-route")),
    )
    monkeypatch.setattr(
        openai_route.ModelProvider,
        "one_by_id",
        AsyncMock(return_value=provider),
    )

    req = _request()
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.headers = {
        "content-type": "application/json",
        "authorization": "Bearer gpustack_caller_secret",
        "cookie": "session_gpustack=caller_cookie",
        "x-api-key": "gpustack_caller_api_key",
    }
    req.state = SimpleNamespace()
    req.app.state.http_client = _FakeClient()
    req.app.state.http_client_no_proxy = _FakeClient()

    resp = await openai_route.proxy_request_by_model(
        request=req, user=SimpleNamespace(id=1)
    )

    assert resp.status_code == 200
    assert (
        captured["url"]
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    lowered_req_headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert lowered_req_headers["authorization"] == "Bearer test-gemini-token"
    assert "cookie" not in lowered_req_headers
    assert "x-api-key" not in lowered_req_headers

    sent_body = json.loads(captured["data"])
    assert sent_body["model"] == "gemini-2.5-flash"
    assert sent_body["temperature"] == 0.7
    assert sent_body["google"] == {"thinking_config": {"thinking_budget": 256}}
    for stripped_field in (
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "top_k",
        "chat_template_kwargs",
        "stop",
    ):
        assert stripped_field not in sent_body

    lowered_resp_headers = {k.lower() for k in resp.headers.keys()}
    assert "content-encoding" not in lowered_resp_headers
    assert "set-cookie" not in lowered_resp_headers
    assert "connection" not in lowered_resp_headers
    assert "x-upstream-internal" not in lowered_resp_headers
    assert req.state.model is None
    assert req.state.model_name == "gemini-2.5-flash"
    assert req.state.provider_id == 11
    assert req.state.model_route_id == 42


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_factory,endpoint,expected_url",
    [
        (
            lambda m: m.GroqConfig(type=m.ModelProviderTypeEnum.GROQ),
            "chat/completions",
            "https://api.groq.com/openai/v1/chat/completions",
        ),
        (
            lambda m: m.GithubConfig(type=m.ModelProviderTypeEnum.GITHUB),
            "chat/completions",
            "https://models.inference.ai.azure.com/chat/completions",
        ),
        (
            lambda m: m.OpenrouterConfig(type=m.ModelProviderTypeEnum.OPENROUTER),
            "embeddings",
            "https://openrouter.ai/api/v1/embeddings",
        ),
        (
            lambda m: m.OpenAIConfig(
                type=m.ModelProviderTypeEnum.OPENAI,
                openaiCustomUrl="https://openai-proxy.example.com/custom/v1",
            ),
            "chat/completions",
            "https://openai-proxy.example.com/custom/v1/chat/completions",
        ),
    ],
)
async def test_resolve_provider_upstream_url_and_auth_headers(
    config_factory, endpoint, expected_url
):
    from gpustack.schemas import model_provider as mp

    provider = SimpleNamespace(
        id=9,
        name="test-provider",
        api_tokens=["provider-secret"],
        config=config_factory(mp),
    )

    assert (
        openai_route._resolve_provider_upstream_url(provider, endpoint) == expected_url
    )
    assert openai_route._build_provider_auth_headers(provider) == {
        "Authorization": "Bearer provider-secret"
    }


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda m: m.ClaudeConfig(type=m.ModelProviderTypeEnum.CLAUDE),
        lambda m: m.BedrockConfig(
            type=m.ModelProviderTypeEnum.BEDROCK, awsRegion="us-east-1"
        ),
        lambda m: m.AzureOpenAIConfig(
            type=m.ModelProviderTypeEnum.AZURE,
            azureServiceUrl="https://example.openai.azure.com",
        ),
        lambda m: m.TritonConfig(
            type=m.ModelProviderTypeEnum.TRITON, tritonDomain="triton.local:8000"
        ),
        lambda m: m.TypesafeConfig(type=m.ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE),
    ],
)
def test_non_openai_wire_providers_require_gateway(config_factory):
    """Providers whose upstream wire protocol or auth requires Higress ai-proxy
    translation fail closed with a clear 503 when gateway mode is disabled."""
    from gpustack.schemas import model_provider as mp

    provider = SimpleNamespace(
        id=9,
        name="non-openai-provider",
        api_tokens=["provider-secret"],
        config=config_factory(mp),
    )

    with pytest.raises(ServiceUnavailableException) as exc:
        openai_route._resolve_provider_upstream_url(provider, "chat/completions")
    assert exc.value.status_code == 503
    assert "requires the AI gateway" in exc.value.message


@pytest.mark.asyncio
async def test_rewrite_provider_model_replaces_multipart_model_field():
    """Overriding the model on a multipart/form-data request replaces the
    existing model part in place rather than appending a second model field."""
    import aiohttp

    form_data = aiohttp.FormData()
    form_data.add_field("model", "whisper-route")
    form_data.add_field(
        "file", b"audio-bytes", filename="clip.wav", content_type="audio/wav"
    )

    target_model, updated_form = openai_route._rewrite_provider_model(
        "whisper-route", "whisper-large-v3-turbo", None, form_data
    )

    assert target_model == "whisper-large-v3-turbo"
    assert updated_form is not None

    payload = updated_form()
    serialized = bytearray()

    class _Writer:
        async def write(self, chunk: bytes) -> None:
            serialized.extend(chunk)

    await payload.write(_Writer())
    raw_multipart = bytes(serialized)
    assert raw_multipart.count(b'name="model"') == 1
    assert b"whisper-large-v3-turbo" in raw_multipart
    assert b"whisper-route" not in raw_multipart
    assert b'filename="clip.wav"' in raw_multipart
    assert b"audio-bytes" in raw_multipart


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_row",
    [
        None,
        SimpleNamespace(id=11, deleted_at="2026-01-01", api_tokens=["tok"]),
        SimpleNamespace(id=11, deleted_at=None, api_tokens=[]),
    ],
)
async def test_inactive_provider_returns_service_unavailable(monkeypatch, provider_row):
    monkeypatch.setattr(
        openai_route.ModelProvider, "one_by_id", AsyncMock(return_value=provider_row)
    )
    with pytest.raises(ServiceUnavailableException) as exc:
        await openai_route._load_active_provider(MagicMock(), 11)
    assert exc.value.status_code == 503
    assert (
        exc.value.message
        == "Model provider is unavailable or has no API tokens configured"
    )


@pytest.mark.asyncio
async def test_iter_normalized_sse_frames_reassembles_crlf_and_utf8_splits():
    """Upstream CRLF-delimited SSE streams and multi-byte UTF-8 sequences split
    across TCP reads emit complete LF-delimited ChatCompletionChunk frames
    compatible with ModelUsageMiddleware."""
    emoji_bytes = "こんにちは🌸".encode("utf-8")

    class _SplitContent:
        async def iter_any(self):
            yield b'data: {"choices":[{"index":0,"delta":{"content":"' + emoji_bytes[:4]
            yield emoji_bytes[4:] + b'"}}]}\r'
            yield b'\n\r\n: keep-alive\r\n\r\ndata: {"choices":null,"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\r\n\r\ndata: [DONE]\r\n\r\n'

    frames = [
        frame
        async for frame in openai_route._iter_normalized_sse_frames(
            SimpleNamespace(content=_SplitContent()), "gemini-2.5-flash"
        )
    ]

    assert len(frames) == 4
    assert "こんにちは🌸".encode("utf-8") in frames[0]
    assert frames[0].startswith(b"data: {") and frames[0].endswith(b"\n\n")
    assert b'"object":"chat.completion.chunk"' in frames[0]
    assert b'"model":"gemini-2.5-flash"' in frames[0]
    assert frames[1] == b": keep-alive\n\n"
    assert b'"choices":[]' in frames[2]
    assert b'"total_tokens":5' in frames[2]
    assert frames[3] == b"data: [DONE]\n\n"
