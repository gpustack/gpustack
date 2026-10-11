from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api import auth as auth_module
from gpustack.api.auth import get_user_from_api_token, worker_auth, worker_proxy_auth
from gpustack.api.exceptions import UnauthorizedException
from gpustack.logging import TRACE_LEVEL


class _Session:
    async def rollback(self):
        pass


def _request():
    return SimpleNamespace(
        headers={"X-Higress-Llm-Model": "allowed-model"},
        app=SimpleNamespace(
            state=SimpleNamespace(
                token="worker-token",
                config=SimpleNamespace(
                    token="registration-token",
                    get_server_url=lambda: "http://server",
                ),
                http_client_no_proxy=object(),
            )
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "access_key,masked",
    [
        ("abcd1234", "****1234"),
        ("abcd", "****"),
        ("", "<empty>"),
        ("x" * 10000, "*" * 64 + "xxxx"),
    ],
    ids=["suffix", "short", "legacy-empty", "bounded"],
)
async def test_api_key_trace_log_masks_the_access_key(
    monkeypatch, caplog, access_key, masked
):
    key = SimpleNamespace(
        access_key=access_key,
        expires_at=None,
        secret_key_digest=None,
        hashed_secret_key="stored-hash",
        is_custom=False,
        user_id=7,
    )
    user = SimpleNamespace(is_active=True, id=7)
    lookup = AsyncMock(return_value=key)
    monkeypatch.setattr(auth_module.APIKeyService, "get_by_access_key", lookup)
    monkeypatch.setattr(
        auth_module.UserService, "get_by_id", AsyncMock(return_value=user)
    )
    monkeypatch.setattr(auth_module, "verify_hashed_secret", lambda *_: True)
    secret = "abcdefghijklmnop"
    token = f"gpustack_{access_key}_{secret}"
    caplog.set_level(TRACE_LEVEL, logger=auth_module.__name__)

    assert await get_user_from_api_token(_Session(), token) == (user, key)

    lookup.assert_awaited_once_with(access_key)
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == auth_module.__name__
    ]
    assert messages == [f"Found API key for access key: {masked}"]
    if access_key:
        assert access_key not in caplog.text
    assert token not in caplog.text and secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", ["abcd", "long-inference-only-key"])
@pytest.mark.parametrize("allowed", [False, True])
async def test_proxy_authorization_logs_no_credential_and_management_stays_denied(
    monkeypatch, caplog, credential, allowed
):
    request = _request()
    authorize = AsyncMock(return_value=allowed)
    factory = MagicMock(return_value=authorize)
    monkeypatch.setattr(auth_module, "make_auth_token_via_server", factory)
    caplog.set_level(TRACE_LEVEL, logger=auth_module.__name__)

    with pytest.raises(UnauthorizedException):
        await worker_auth(request, x_api_key=credential)
    factory.assert_not_called()
    assert caplog.records == []

    if allowed:
        await worker_proxy_auth(request, x_api_key=credential)
    else:
        with pytest.raises(UnauthorizedException):
            await worker_proxy_auth(request, x_api_key=credential)

    authorize.assert_awaited_once_with("http://server", credential, "allowed-model")
    assert [record.getMessage() for record in caplog.records] == [
        "Verifying inference proxy credentials via server authentication"
    ]
    assert credential not in caplog.text
