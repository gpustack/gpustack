from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.routes import openai as openai_route
from tests.utils.mock import mock_async_session


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 404, 422])
async def test_non_streaming_worker_errors_are_passed_through(monkeypatch, status_code):
    request = MagicMock()
    request.url.path = "/v1-openai/chat/completions"
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.state = SimpleNamespace(internal_inference=False)
    request.app.state.http_client = MagicMock()
    request.app.state.http_client_no_proxy = MagicMock()
    user = SimpleNamespace(id=1)
    route = SimpleNamespace(id=3)
    model = SimpleNamespace(id=2, name="qwen3")
    instance = SimpleNamespace(id=4, worker_id=5, worker_ip="127.0.0.1", port=8080)
    worker = SimpleNamespace(id=5)
    error_body = b'{"error":"invalid request"}'

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
        AsyncMock(
            return_value=[
                SimpleNamespace(model_id=2, weight=1, overridden_model_name=None)
            ]
        ),
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        "get_by_name",
        AsyncMock(return_value=route),
    )
    monkeypatch.setattr(
        openai_route.ModelService, "get_by_id", AsyncMock(return_value=model)
    )
    monkeypatch.setattr(
        openai_route, "get_running_instance", AsyncMock(return_value=instance)
    )
    monkeypatch.setattr(
        openai_route.WorkerService, "get_by_id", AsyncMock(return_value=worker)
    )
    monkeypatch.setattr(openai_route, "model_instance_prefix", lambda _: "instance")
    request_to_worker = AsyncMock(
        return_value=(SimpleNamespace(status=status_code, headers={}), error_body)
    )
    monkeypatch.setattr(openai_route, "request_to_worker", request_to_worker)

    response = await openai_route.proxy_request_by_model(request, user)

    assert response.status_code == status_code
    assert response.body == error_body
    assert request_to_worker.call_args.kwargs["raise_on_error"] is False
