"""Tests for login CAPTCHA generation, tokens, and HTTP integration."""

import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.config.config import Config
from gpustack.routes import auth as auth_routes
from gpustack.schemas.config import GatewayModeEnum
from gpustack.schemas.login_captcha import LoginCaptchaNonce
from gpustack.security import JWTManager
from gpustack.server.db import get_session
from gpustack.utils import captcha as captcha_util

_SECRET_KEY = "test-secret-key-with-enough-entropy"
_BINDING = "test-browser-binding-value-1234567890"
_REAL_CONSUME_CAPTCHA_NONCE = auth_routes._consume_captcha_nonce


def _request(config, secret_key: str = _SECRET_KEY):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                server_config=config,
                jwt_manager=JWTManager(secret_key),
            )
        ),
        cookies={auth_routes.CAPTCHA_SESSION_COOKIE_NAME: _BINDING},
        headers={"host": "testserver"},
        client=SimpleNamespace(host="testclient"),
        url=SimpleNamespace(scheme="http"),
    )


def _config(enabled: bool = True, length: int = 4):
    return SimpleNamespace(
        enable_login_captcha=enabled,
        login_captcha_length=length,
        server_external_url=None,
        external_auth_type=None,
        gateway_mode=GatewayModeEnum.disabled,
    )


@pytest.fixture(autouse=True)
def isolate_captcha_guards(monkeypatch):
    spent_nonces = set()

    async def consume_once(nonce):
        if nonce in spent_nonces:
            raise exceptions.BadRequestException(
                message="CAPTCHA has already been used"
            )
        spent_nonces.add(nonce)

    monkeypatch.setattr(auth_routes, "_consume_captcha_nonce", consume_once)
    limiters = (
        auth_routes._captcha_issue_limiter,
        auth_routes._captcha_audio_limiter,
        auth_routes._login_ip_limiter,
        auth_routes._login_account_limiter,
    )
    for limiter in limiters:
        limiter.clear()
    yield
    for limiter in limiters:
        limiter.clear()


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/auth")
    exceptions.register_handlers(app)

    config = _config()
    app.state.server_config = config
    app.state.jwt_manager = JWTManager(_SECRET_KEY)

    async def session_override():
        yield object()

    app.dependency_overrides[get_session] = session_override
    authenticate_user = AsyncMock(return_value=SimpleNamespace(name="admin"))
    monkeypatch.setattr(auth_routes, "authenticate_user", authenticate_user)

    with TestClient(app) as test_client:
        yield test_client, config, authenticate_user


def _solved_challenge(test_client):
    issued = test_client.get("/auth/captcha").json()
    challenge = captcha_util.decrypt_challenge(
        _SECRET_KEY,
        issued["captcha_id"],
        ttl_seconds=auth_routes.CAPTCHA_TOKEN_TTL_SECONDS,
    )
    return issued, challenge.code


def test_generate_code_uses_unambiguous_characters():
    code = captcha_util.generate_code(6)

    assert len(code) == 6
    assert set(code) <= set(captcha_util.CAPTCHA_ALPHABET)
    assert not set("01BILOQSZ") & set(code)


@pytest.mark.parametrize("length", [3, 7])
def test_generate_code_rejects_out_of_range_length(length):
    with pytest.raises(ValueError, match="between 4 and 6"):
        captcha_util.generate_code(length)


def test_generate_captcha_returns_valid_png():
    code, image_bytes = captcha_util.generate_captcha(4)

    with Image.open(BytesIO(image_bytes)) as image:
        assert image.format == "PNG"
        assert image.size == (160, 60)
    assert len(code) == 4


def test_generate_audio_returns_valid_wav():
    audio = captcha_util.generate_audio("ACDE")

    assert audio.startswith(b"RIFF")
    assert audio[8:12] == b"WAVE"


def test_generators_are_reused_per_thread(monkeypatch):
    monkeypatch.setattr(captcha_util, "_generators", threading.local())
    barrier = threading.Barrier(2)

    def generators():
        image = captcha_util._get_image_generator()
        audio = captcha_util._get_audio_generator()
        assert image is captcha_util._get_image_generator()
        assert audio is captcha_util._get_audio_generator()
        barrier.wait()
        return image, audio

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(generators)
        second_future = executor.submit(generators)
        first_image, first_audio = first_future.result()
        second_image, second_audio = second_future.result()

    assert first_image is not second_image
    assert first_audio is not second_audio


def test_challenge_token_is_opaque_and_authenticated(monkeypatch):
    image_bytes = captcha_util.generate_captcha(4)[1]
    monkeypatch.setattr(
        captcha_util,
        "generate_captcha",
        lambda length: ("ACDE", image_bytes),
    )

    issued = auth_routes._issue_captcha(_request(_config()), 4, _BINDING)
    token = issued["captcha_id"]
    decoded_token = base64.urlsafe_b64decode(token)
    challenge = captcha_util.decrypt_challenge(
        _SECRET_KEY,
        token,
        ttl_seconds=auth_routes.CAPTCHA_TOKEN_TTL_SECONDS,
    )

    assert b"acde" not in decoded_token.lower()
    assert challenge.code == "acde"
    assert challenge.nonce
    assert challenge.binding == _BINDING
    assert issued["image"].startswith("data:image/png;base64,")


def test_challenge_token_rejects_wrong_secret():
    token = captcha_util.encrypt_challenge(_SECRET_KEY, "ACDE", "nonce-value", _BINDING)

    with pytest.raises(captcha_util.InvalidCaptchaToken):
        captcha_util.decrypt_challenge(
            "different-secret-key",
            token,
            ttl_seconds=auth_routes.CAPTCHA_TOKEN_TTL_SECONDS,
        )


def test_challenge_token_rejects_tampering():
    token = captcha_util.encrypt_challenge(_SECRET_KEY, "ACDE", "nonce-value", _BINDING)
    token_bytes = bytearray(base64.urlsafe_b64decode(token))
    token_bytes[-33] ^= 1
    tampered_token = base64.urlsafe_b64encode(token_bytes).decode("ascii")

    with pytest.raises(captcha_util.InvalidCaptchaToken):
        captcha_util.decrypt_challenge(
            _SECRET_KEY,
            tampered_token,
            ttl_seconds=auth_routes.CAPTCHA_TOKEN_TTL_SECONDS,
        )


def test_challenge_token_expires(monkeypatch):
    now = [1_700_000_000]
    monkeypatch.setattr("cryptography.fernet.time.time", lambda: now[0])
    token = captcha_util.encrypt_challenge(_SECRET_KEY, "ACDE", "nonce-value", _BINDING)
    now[0] += auth_routes.CAPTCHA_TOKEN_TTL_SECONDS + 1

    with pytest.raises(captcha_util.InvalidCaptchaToken):
        captcha_util.decrypt_challenge(
            _SECRET_KEY,
            token,
            ttl_seconds=auth_routes.CAPTCHA_TOKEN_TTL_SECONDS,
        )


@pytest.mark.parametrize("length", [3, 7])
def test_config_rejects_invalid_captcha_length(length):
    with pytest.raises(ValidationError):
        Config(login_captcha_length=length)


@pytest.mark.asyncio
async def test_verify_captcha_accepts_once_and_burns_the_challenge(monkeypatch):
    image_bytes = captcha_util.generate_captcha(4)[1]
    monkeypatch.setattr(
        captcha_util,
        "generate_captcha",
        lambda length: ("ACDE", image_bytes),
    )
    request = _request(_config())
    issued = auth_routes._issue_captcha(request, 4, _BINDING)

    await auth_routes._verify_captcha(request, issued["captcha_id"], " acde ")

    with pytest.raises(exceptions.BadRequestException):
        await auth_routes._verify_captcha(request, issued["captcha_id"], "ACDE")


@pytest.mark.asyncio
async def test_verify_captcha_burns_challenge_after_wrong_answer(monkeypatch):
    image_bytes = captcha_util.generate_captcha(4)[1]
    monkeypatch.setattr(
        captcha_util,
        "generate_captcha",
        lambda length: ("ACDE", image_bytes),
    )
    request = _request(_config())
    issued = auth_routes._issue_captcha(request, 4, _BINDING)

    with pytest.raises(exceptions.BadRequestException):
        await auth_routes._verify_captcha(request, issued["captcha_id"], "WRONG")

    with pytest.raises(exceptions.BadRequestException):
        await auth_routes._verify_captcha(request, issued["captcha_id"], "ACDE")


@pytest.mark.asyncio
async def test_verify_captcha_rejects_non_ascii_answer(monkeypatch):
    image_bytes = captcha_util.generate_captcha(4)[1]
    monkeypatch.setattr(
        captcha_util,
        "generate_captcha",
        lambda length: ("ACDE", image_bytes),
    )
    request = _request(_config())
    issued = auth_routes._issue_captcha(request, 4, _BINDING)

    with pytest.raises(exceptions.BadRequestException) as exc_info:
        await auth_routes._verify_captcha(request, issued["captcha_id"], "验证码")
    assert exc_info.value.message == "Incorrect CAPTCHA"


def test_http_captcha_login_and_replay(client):
    test_client, _, authenticate_user = client
    captcha_response = test_client.get("/auth/captcha")

    assert captcha_response.status_code == 200
    assert captcha_response.headers["cache-control"] == "no-store"
    set_cookie = captcha_response.headers["set-cookie"].lower()
    assert "httponly" in set_cookie
    assert "samesite=strict" in set_cookie
    assert "path=/auth" in set_cookie
    issued = captcha_response.json()
    challenge = captcha_util.decrypt_challenge(
        _SECRET_KEY,
        issued["captcha_id"],
        ttl_seconds=auth_routes.CAPTCHA_TOKEN_TTL_SECONDS,
    )

    login_response = test_client.post(
        "/auth/login",
        data={
            "username": "admin",
            "password": "password",
            "captcha_id": issued["captcha_id"],
            "captcha": challenge.code,
        },
    )

    assert login_response.status_code == 200
    assert "session=" in login_response.headers["set-cookie"]
    authenticate_user.assert_awaited_once()

    replay_response = test_client.post(
        "/auth/login",
        data={
            "username": "admin",
            "password": "password",
            "captcha_id": issued["captcha_id"],
            "captcha": challenge.code,
        },
    )
    assert replay_response.status_code == 400
    authenticate_user.assert_awaited_once()


def test_http_captcha_audio_uses_bound_challenge(client):
    test_client, _, _ = client
    issued = test_client.get("/auth/captcha").json()

    response = test_client.post(
        "/auth/captcha/audio", data={"captcha_id": issued["captcha_id"]}
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    prefix, encoded = response.json()["audio"].split(",", 1)
    assert prefix == "data:audio/wav;base64"
    assert base64.b64decode(encoded).startswith(b"RIFF")


def test_http_login_rejects_challenge_from_another_browser(client):
    test_client, _, authenticate_user = client
    issued = test_client.get("/auth/captcha").json()
    challenge = captcha_util.decrypt_challenge(
        _SECRET_KEY,
        issued["captcha_id"],
        ttl_seconds=auth_routes.CAPTCHA_TOKEN_TTL_SECONDS,
    )
    test_client.cookies.clear()

    response = test_client.post(
        "/auth/login",
        data={
            "username": "admin",
            "password": "password",
            "captcha_id": issued["captcha_id"],
            "captcha": challenge.code,
        },
    )

    assert response.status_code == 400
    authenticate_user.assert_not_awaited()


def test_http_login_rejects_cross_site_origin(client):
    test_client, _, authenticate_user = client

    response = test_client.post(
        "/auth/login",
        data={"username": "admin", "password": "password"},
        headers={"Origin": "https://attacker.example"},
    )

    assert response.status_code == 400
    authenticate_user.assert_not_awaited()


def test_http_login_rejects_cross_site_fetch_metadata(client):
    test_client, _, authenticate_user = client

    response = test_client.post(
        "/auth/login",
        data={"username": "admin", "password": "password"},
        headers={"Sec-Fetch-Site": "cross-site"},
    )

    assert response.status_code == 400
    authenticate_user.assert_not_awaited()


def test_http_login_accepts_external_origin_behind_tls_proxy(client):
    test_client, config, authenticate_user = client
    config.server_external_url = "https://public.example/gpustack"
    issued, code = _solved_challenge(test_client)

    response = test_client.post(
        "/auth/login",
        data={
            "username": "admin",
            "password": "password",
            "captcha_id": issued["captcha_id"],
            "captcha": code,
        },
        headers={"Origin": "https://public.example:443"},
    )

    assert response.status_code == 200
    authenticate_user.assert_awaited_once()


def test_http_login_does_not_trust_raw_forwarded_proto(client):
    test_client, _, authenticate_user = client
    issued, code = _solved_challenge(test_client)

    response = test_client.post(
        "/auth/login",
        data={
            "username": "admin",
            "password": "password",
            "captcha_id": issued["captcha_id"],
            "captcha": code,
        },
        headers={
            "Origin": "https://testserver",
            "X-Forwarded-Proto": "https",
        },
    )

    assert response.status_code == 400
    authenticate_user.assert_not_awaited()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://EXAMPLE.com", ("https", "example.com", 443)),
        ("https://example.com:443", ("https", "example.com", 443)),
        ("http://[::1]:80", ("http", "::1", 80)),
        ("null", None),
        ("https://user@example.com", None),
        ("https://example.com/path", None),
    ],
)
def test_canonical_http_origin(value, expected):
    assert auth_routes._canonical_http_origin(value) == expected


def test_canonical_http_origin_allows_deployment_path_only_when_requested():
    assert auth_routes._canonical_http_origin(
        "https://example.com/gpustack", allow_path=True
    ) == ("https", "example.com", 443)
    assert (
        auth_routes._canonical_http_origin(
            "https://example.com/gpustack?source=test", allow_path=True
        )
        is None
    )


def test_disabled_captcha_preserves_cross_site_login_behavior(client):
    test_client, config, authenticate_user = client
    config.enable_login_captcha = False

    response = test_client.post(
        "/auth/login",
        data={"username": "admin", "password": "password"},
        headers={"Origin": "https://attacker.example"},
    )

    assert response.status_code == 200
    authenticate_user.assert_awaited_once()


def test_http_captcha_issue_is_rate_limited(client, monkeypatch):
    from gpustack.utils.rate_limit import KeyedRateLimiter

    test_client, _, _ = client
    monkeypatch.setattr(
        auth_routes,
        "_captcha_issue_limiter",
        KeyedRateLimiter(max_requests=1, window_seconds=60),
    )

    assert test_client.get("/auth/captcha").status_code == 200
    assert test_client.get("/auth/captcha").status_code == 429


def test_http_login_is_rate_limited_when_captcha_is_enabled(client, monkeypatch):
    from gpustack.utils.rate_limit import KeyedRateLimiter

    test_client, _, authenticate_user = client
    monkeypatch.setattr(
        auth_routes,
        "_login_account_limiter",
        KeyedRateLimiter(max_requests=1, window_seconds=60),
    )

    assert (
        test_client.post(
            "/auth/login", data={"username": "admin", "password": "password"}
        ).status_code
        == 400
    )
    assert (
        test_client.post(
            "/auth/login", data={"username": "admin", "password": "password"}
        ).status_code
        == 429
    )
    authenticate_user.assert_not_awaited()


def test_http_login_requires_captcha_fields(client):
    test_client, _, authenticate_user = client

    response = test_client.post(
        "/auth/login",
        data={"username": "admin", "password": "password"},
    )

    assert response.status_code == 400
    authenticate_user.assert_not_awaited()


def test_http_captcha_is_hidden_and_login_unchanged_when_disabled(client):
    test_client, config, authenticate_user = client
    config.enable_login_captcha = False

    captcha_response = test_client.get("/auth/captcha")
    login_response = test_client.post(
        "/auth/login",
        data={"username": "admin", "password": "password"},
    )

    assert captcha_response.status_code == 404
    assert login_response.status_code == 200
    authenticate_user.assert_awaited_once()


@pytest.mark.asyncio
async def test_database_nonce_ledger_is_single_use_across_sessions(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(LoginCaptchaNonce.__table__.create)

    @asynccontextmanager
    async def session_factory():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    monkeypatch.setattr(auth_routes, "async_session", session_factory)
    try:
        await _REAL_CONSUME_CAPTCHA_NONCE("shared-nonce")
        with pytest.raises(exceptions.BadRequestException) as exc_info:
            await _REAL_CONSUME_CAPTCHA_NONCE("shared-nonce")
        assert exc_info.value.message == "CAPTCHA has already been used"
    finally:
        await engine.dispose()
