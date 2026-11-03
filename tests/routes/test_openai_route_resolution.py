from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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
    monkeypatch, managed, deleted
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
    monkeypatch.setattr(
        openai_route, 'get_running_instance', AsyncMock(return_value=instance)
    )
    monkeypatch.setattr(
        openai_route, 'get_model_instance_workloads', lambda _: [] if managed else None
    )
    refresh = AsyncMock(return_value=instance)
    monkeypatch.setattr(ModelInstance, 'one_by_id_with_refresh', refresh)
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
            id=2 if managed else 1, deleted_at='deleted' if deleted else None
        )
    )
    monkeypatch.setattr(openai_route.WorkerService, 'get_by_id', get_worker)
    monkeypatch.setattr(openai_route, 'mutate_request', MagicMock())
    monkeypatch.setattr(
        openai_route,
        '_prepare_proxy_request',
        lambda request, body, form, headers, **kwargs: (headers, b'{}'),
    )
    import threading

    event_loop_thread = threading.get_ident()
    prepare_component = openai_route._prepare_component_request
    transformed_on = []

    def transform(*args):
        transformed_on.append(threading.get_ident())
        return prepare_component(*args)

    monkeypatch.setattr(openai_route, '_prepare_component_request', transform)
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
    await openai_route.proxy_request_by_model(
        request=_request(), user=SimpleNamespace(id=1)
    )
    assert transformed_on and transformed_on[0] != event_loop_thread
    assert forward.call_args.kwargs['path'] == 'v1/chat/completions'
    assert forward.call_args.kwargs['headers'][openai_route.router_header_key] == (
        'model-1-7-r3-hybrid-w2.static'
        if managed == 'hybrid'
        else ('model-1-7-r3-dp1.static' if managed else 'model-1-7.static')
    )
    get_worker.assert_awaited_once_with(2 if managed else 1)
    if managed:
        refresh.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'headers,code',
    [
        ({'content-length': '-1'}, 400),
        ({'content-length': str(17 * 1024**2)}, 413),
        ({'content-encoding': 'gzip'}, 415),
    ],
)
async def test_component_input_errors_use_openai_response_format(
    monkeypatch, headers, code
):
    import httpx
    from fastapi import FastAPI, Request
    from gpustack.api.exceptions import register_handlers
    from gpustack.utils.model_instance_workers import ModelInstanceEndpoint

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
    instance = SimpleNamespace(id=7, worker_id=1, worker_ip='10.0.0.1', port=41000)
    monkeypatch.setattr(
        openai_route, 'get_running_instance', AsyncMock(return_value=instance)
    )
    endpoint = ModelInstanceEndpoint(
        1, 7, 1, 41000, 3, component='sglang_smg', attempt=0
    )
    monkeypatch.setattr(
        openai_route,
        '_select_managed_endpoint',
        AsyncMock(return_value=(instance, endpoint)),
    )
    monkeypatch.setattr(
        openai_route.WorkerService,
        'get_by_id',
        AsyncMock(return_value=SimpleNamespace(id=1, deleted_at=None)),
    )
    monkeypatch.setattr(openai_route, 'mutate_request', MagicMock())
    forward = AsyncMock()
    monkeypatch.setattr(openai_route, 'request_to_worker', forward)
    app = FastAPI()
    register_handlers(app)

    @app.post('/v1/chat/completions')
    async def proxy(request: Request):
        return await openai_route.proxy_request_by_model(request, SimpleNamespace(id=1))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.post(
            '/v1/chat/completions',
            content=b'{}',
            headers={'content-type': 'application/json', **headers},
        )
    assert response.status_code == code
    assert 'detail' not in response.json()
    assert response.json()['error']['code'] == code
    assert response.json()['error']['message']
    forward.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('component', ['sglang_smg', 'vllm_hybrid', None])
async def test_server_proxy_sends_component_specific_payload(monkeypatch, component):
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
    monkeypatch.setattr(
        ModelInstance, 'one_by_id_with_refresh', AsyncMock(return_value=instance)
    )
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
        return web.json_response({'ok': True})

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
        }
        async with ClientSession() as client:
            request.app.state.http_client = client
            request.app.state.http_client_no_proxy = client
            response = await openai_route.proxy_request_by_model(
                request, SimpleNamespace(id=1)
            )
    assert response.status_code == 200
    data, headers, raw, path = received[0]
    assert path == '/v1/chat/completions'
    assert data == (
        {'model': 'qwen3', 'prompt': '你好'} if component == 'sglang_smg' else payload
    )
    assert headers.get('X-Data-Parallel-Rank') == (
        None if component == 'sglang_smg' else '9'
    )
    assert int(headers['Content-Length']) == len(raw)
