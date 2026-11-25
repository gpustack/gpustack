import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import Response
from fastapi.testclient import TestClient

from gpustack.api.auth import SESSION_COOKIE_NAME
from gpustack.config.config import Config
from gpustack.routes import auth
from gpustack.server import app as server_app
from gpustack.server.db import get_session


def test_session_timeout_applies_to_login_and_refresh_without_undoing_logout(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(server_app.ui, "register", lambda _: None)
    monkeypatch.setattr(server_app, "patch_docs", lambda *_: None)
    monkeypatch.setattr(
        auth, "authenticate_user", AsyncMock(return_value=SimpleNamespace(name="admin"))
    )
    cfg = Config(
        data_dir=str(tmp_path),
        advertise_address="127.0.0.1",
        gateway_mode="disabled",
        jwt_token_expire_minutes=5,
    )
    app = server_app.create_app(cfg)
    app.dependency_overrides[get_session] = lambda: None

    @app.get("/test-session")
    def session_probe():
        return Response()

    def assert_session_lifetime(response, started_at):
        assert response.status_code == 200
        session_cookies = [
            cookie
            for cookie in response.headers.get_list("set-cookie")
            if cookie.startswith(f"{SESSION_COOKIE_NAME}=")
        ]
        assert len(session_cookies) == 1
        assert "Max-Age=300" in session_cookies[0]
        assert "Secure" in session_cookies[0]
        token = response.cookies[SESSION_COOKIE_NAME]
        payload = app.state.jwt_manager.decode_jwt_token(token)
        assert payload["sub"] == "admin"
        assert started_at + 300 <= payload["exp"] <= int(time.time()) + 300

    with TestClient(app, base_url="https://localhost") as client:
        started_at = int(time.time())
        login = client.post(
            "/auth/login", data={"username": "admin", "password": "test"}
        )
        assert_session_lifetime(login, started_at)

        started_at = int(time.time())
        assert_session_lifetime(client.get("/test-session"), started_at)

        logout = client.post("/auth/logout")
        assert logout.status_code == 200
        session_cookies = [
            cookie
            for cookie in logout.headers.get_list("set-cookie")
            if cookie.startswith(f"{SESSION_COOKIE_NAME}=")
        ]
        assert len(session_cookies) == 1
        assert "Max-Age=0" in session_cookies[0]
        assert SESSION_COOKIE_NAME not in client.cookies
