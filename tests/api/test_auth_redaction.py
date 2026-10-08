from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import gpustack.api.auth as auth_module
from gpustack.api.auth import get_user_from_api_token, worker_auth


class _Session:
    async def rollback(self):
        pass


class _Config:
    token = "registration-token"

    def get_server_url(self):
        return "http://example.com"


def _api_key(access_key):
    return SimpleNamespace(
        access_key=access_key,
        expires_at=None,
        secret_key_digest=None,
        hashed_secret_key="stored-hash",
        is_custom=False,
        user_id=7,
    )


@pytest.mark.asyncio
async def test_api_key_trace_log_masks_the_access_key(monkeypatch):
    key = _api_key("abcd1234")
    monkeypatch.setattr(
        auth_module.APIKeyService,
        "get_by_access_key",
        AsyncMock(return_value=key),
    )
    monkeypatch.setattr(
        auth_module.UserService,
        "get_by_id",
        AsyncMock(return_value=SimpleNamespace(is_active=True, id=7)),
    )
    monkeypatch.setattr(auth_module, "verify_hashed_secret", lambda *_: True)

    with patch.object(auth_module.logger, "trace") as trace:
        await get_user_from_api_token(_Session(), "gpustack_abcd1234_abcdefghijklmnop")

    rendered = " ".join(str(arg) for call in trace.call_args_list for arg in call.args)
    assert "abcd1234" not in rendered
    assert "****1234" in rendered


@pytest.mark.asyncio
async def test_worker_token_debug_log_masks_short_tokens(monkeypatch):
    request = SimpleNamespace(
        headers={"X-Higress-Llm-Model": "model"},
        app=SimpleNamespace(
            state=SimpleNamespace(
                token="registration-token",
                config=_Config(),
                http_client_no_proxy=object(),
            )
        ),
    )
    monkeypatch.setattr(
        auth_module,
        "make_auth_token_via_server",
        lambda _: AsyncMock(return_value=True),
    )

    with patch.object(auth_module.logger, "debug") as debug:
        await worker_auth(request=request, x_api_key="abcd")

    rendered = " ".join(str(arg) for call in debug.call_args_list for arg in call.args)
    assert "abcd" not in rendered
    assert "****" in rendered
