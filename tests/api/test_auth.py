from unittest.mock import AsyncMock

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from gpustack.api.auth import get_current_user, worker_auth
from gpustack.api.exceptions import UnauthorizedException


class DummyWorkerConfig:
    token = "registration-token"

    def get_server_url(self):
        return "http://example.com"


@pytest.mark.asyncio
async def test_get_current_user_accepts_x_api_key(monkeypatch):
    session = object()
    request = type("Request", (), {})()
    request.state = type("State", (), {})()
    request.headers = {}
    request.client = type("Client", (), {"host": "10.0.0.1"})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type(
        "Config", (), {"gateway_mode": None, "force_auth_localhost": True}
    )()

    expected_user = type("User", (), {"is_active": True})()
    expected_key = object()

    auth_mock = AsyncMock(return_value=(expected_user, expected_key))
    monkeypatch.setattr("gpustack.api.auth.get_user_from_api_token", auth_mock)

    user = await get_current_user(
        request=request,
        session=session,
        x_api_key="sk_test_value",
    )

    auth_mock.assert_awaited_once_with(session, "sk_test_value")
    assert user is expected_user
    assert request.state.user is expected_user
    assert request.state.api_key is expected_key


@pytest.mark.asyncio
async def test_worker_auth_accepts_x_api_key():
    request = type("Request", (), {})()
    request.headers = {"X-Higress-Llm-Model": "claude-sonnet"}
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.token = "worker-token"
    request.app.state.config = DummyWorkerConfig()
    request.app.state.http_client_no_proxy = object()

    assert await worker_auth(request=request, x_api_key="worker-token") is None


@pytest.mark.asyncio
async def test_worker_auth_rejects_missing_credentials():
    request = type("Request", (), {})()
    request.headers = {"X-Higress-Llm-Model": "claude-sonnet"}
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.token = "worker-token"
    request.app.state.config = DummyWorkerConfig()
    request.app.state.http_client_no_proxy = object()

    with pytest.raises(UnauthorizedException):
        await worker_auth(request=request)


@pytest.mark.asyncio
async def test_get_current_user_falls_back_to_x_api_key_when_bearer_empty(
    monkeypatch,
):
    session = object()
    request = type("Request", (), {})()
    request.state = type("State", (), {})()
    request.headers = {}
    request.client = type("Client", (), {"host": "10.0.0.1"})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type(
        "Config", (), {"gateway_mode": None, "force_auth_localhost": True}
    )()

    expected_user = type("User", (), {"is_active": True})()
    expected_key = object()

    auth_mock = AsyncMock(return_value=(expected_user, expected_key))
    monkeypatch.setattr("gpustack.api.auth.get_user_from_api_token", auth_mock)

    user = await get_current_user(
        request=request,
        session=session,
        bearer_token=HTTPAuthorizationCredentials(scheme="Bearer", credentials=""),
        x_api_key="sk_test_value",
    )

    auth_mock.assert_awaited_once_with(session, "sk_test_value")
    assert user is expected_user


@pytest.mark.asyncio
async def test_worker_auth_falls_back_to_x_api_key_when_bearer_empty():
    request = type("Request", (), {})()
    request.headers = {"X-Higress-Llm-Model": "claude-sonnet"}
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.token = "worker-token"
    request.app.state.config = DummyWorkerConfig()
    request.app.state.http_client_no_proxy = object()

    assert (
        await worker_auth(
            request=request,
            bearer_token=HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=""
            ),
            x_api_key="worker-token",
        )
        is None
    )
