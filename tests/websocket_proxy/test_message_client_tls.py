import asyncio
import ssl
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import certifi
import pytest

from gpustack import ssl_context
from gpustack.config import config as config_module
from gpustack.websocket_proxy import message_client


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize(
    "config_insecure,env_insecure,expected_insecure",
    [
        (None, False, False),
        (False, False, False),
        (True, False, True),
        (None, True, True),
    ],
)
async def test_tunnel_tls_policy_on_connect_and_reconnect(
    monkeypatch, scheme, config_insecure, env_insecure, expected_insecure
):
    monkeypatch.setattr(ssl_context.envs, "INSECURE_TLS", env_insecure)
    monkeypatch.setattr(
        config_module,
        "get_global_config",
        lambda: SimpleNamespace(insecure_tls=config_insecure),
    )
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    factory = Mock(wraps=ssl_context.make_ssl_context)
    monkeypatch.setattr(message_client, "make_ssl_context", factory)
    connect = AsyncMock(
        side_effect=[OSError("connection lost"), asyncio.CancelledError]
    )
    monkeypatch.setattr(message_client, "connect", connect)
    monkeypatch.setattr(message_client.asyncio, "sleep", AsyncMock())
    client = message_client.MessageClient(
        server_endpoint=f"{scheme}://server.example",
        client_id=uuid.uuid4(),
    )

    ssl_context.make_ssl_context.cache_clear()
    try:
        await client.run()

        assert connect.await_count == 2
        for call in connect.await_args_list:
            context = call.kwargs.get("ssl")
            if scheme == "http":
                assert context is None
                factory.assert_not_called()
            else:
                assert isinstance(context, ssl.SSLContext)
                assert context.check_hostname is not expected_insecure
                assert context.verify_mode == (
                    ssl.CERT_NONE if expected_insecure else ssl.CERT_REQUIRED
                )
                if not expected_insecure:
                    assert context.cert_store_stats()["x509_ca"] > 0
    finally:
        ssl_context.make_ssl_context.cache_clear()
