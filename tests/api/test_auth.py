import ssl
from unittest.mock import AsyncMock

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from gpustack.api.auth import get_current_user, worker_auth
from gpustack.api.exceptions import UnauthorizedException
from gpustack.routes.auth import oidc_callback


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
    request.app.state.server_config = type("Config", (), {"gateway_mode": None})()

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
    request.app.state.server_config = type("Config", (), {"gateway_mode": None})()

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
            bearer_token=HTTPAuthorizationCredentials(scheme="Bearer", credentials=""),
            x_api_key="worker-token",
        )
        is None
    )


def _make_request(headers=None, client_host="127.0.0.1"):
    request = type("Request", (), {})()
    request.state = type("State", (), {})()
    request.headers = headers or {}
    request.client = type("Client", (), {"host": client_host})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type("Config", (), {"gateway_mode": None})()
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_host,headers",
    [
        # Genuine local request — no longer auto-trusted.
        ("127.0.0.1", {"host": "127.0.0.1:30080"}),
        # Reverse-proxy-fronted remote attacker arriving with TCP peer 127.0.0.1.
        (
            "127.0.0.1",
            {"host": "gpustack.example.com", "x-forwarded-for": "8.8.8.8"},
        ),
        # IPv6 loopback.
        ("::1", {"host": "[::1]:30080"}),
        # External IP.
        ("10.0.0.1", {"host": "gpustack.example.com"}),
    ],
)
async def test_get_current_user_requires_credentials(monkeypatch, client_host, headers):
    # The auto-admin localhost shortcut has been removed entirely.
    # Every unauthenticated request — local, proxied, or remote — must be
    # rejected.
    session = object()
    request = _make_request(headers=headers, client_host=client_host)

    first_by_field = AsyncMock()
    get_by_username = AsyncMock()
    monkeypatch.setattr("gpustack.api.auth.User.first_by_field", first_by_field)
    monkeypatch.setattr(
        "gpustack.api.auth.UserService.get_by_username", get_by_username
    )

    with pytest.raises(UnauthorizedException):
        await get_current_user(request=request, session=session)
    # No DB lookup path may fire when there are no credentials.
    first_by_field.assert_not_awaited()
    get_by_username.assert_not_awaited()


@pytest.mark.asyncio
async def test_oidc_callback_uses_system_trust_store(monkeypatch):
    captured = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, data=None):
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "text": '{"access_token":"token","id_token":"id"}',
                },
            )()

    request = type("Request", (), {})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type(
        "Config",
        (),
        {
            "oidc_client_id": "client-id",
            "oidc_client_secret": "client-secret",
            "oidc_redirect_uri": "https://gpustack.example.com/auth/oidc/callback",
            "openid_configuration": {
                "token_endpoint": "https://issuer.example.com/token"
            },
            "external_auth_name": None,
            "external_auth_full_name": None,
            "external_auth_avatar_url": None,
            "external_auth_default_inactive": False,
            # Group sync defaults to False; this test exercises the
            # trust-store path, not group sync.
            "external_auth_group_sync": False,
            "external_auth_groups": None,
        },
    )()
    request.app.state.jwt_manager = type(
        "JWTManager", (), {"create_jwt_token": lambda self, username: "jwt-token"}
    )()
    request.query_params = {"code": "auth-code", "state": "test-state"}
    request.cookies = {"gpustack_oidc_state": "test-state"}

    monkeypatch.setattr("gpustack.routes.auth.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("gpustack.routes.auth.use_proxy_env_for_url", lambda url: False)
    monkeypatch.setattr(
        "gpustack.routes.auth.get_oidc_user_data",
        AsyncMock(return_value={"email": "user@example.com", "name": "Test User"}),
    )
    monkeypatch.setattr(
        "gpustack.routes.auth.User.first_by_field", AsyncMock(return_value=object())
    )

    response = await oidc_callback(request=request, session=object())

    assert response.status_code in (302, 307)
    assert captured["trust_env"] is False
    assert captured["timeout"] is not None
    assert isinstance(captured["verify"], ssl.SSLContext)
