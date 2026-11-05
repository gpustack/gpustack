from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.client import generated_clientset

CLIENT_CLASSES = [
    value
    for name, value in vars(generated_clientset).items()
    if name.endswith('Client') and name != 'HTTPClient'
]


@pytest.mark.parametrize('client_class', CLIENT_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('watching', [False, True])
@pytest.mark.parametrize('present', [False, True])
def test_strict_cache_lookup_never_requests_http(
    client_class, enabled, watching, present
):
    transport = MagicMock()
    client = client_class(transport, enable_cache=enabled)
    item = SimpleNamespace(id=7)
    client._watch_started = watching
    client._cache = {7: item} if present else {}

    if enabled and watching:
        assert client.get_cached(7) is (item if present else None)
    else:
        with pytest.raises(ServiceUnavailableException):
            client.get_cached(7)
    transport.get_httpx_client.assert_not_called()
    transport.get_async_httpx_client.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('client_class', CLIENT_CLASSES, ids=lambda cls: cls.__name__)
async def test_disconnected_watch_invalidates_strict_cache(client_class):
    transport = MagicMock()
    client = client_class(transport)
    stale = SimpleNamespace(id=7)
    client._watch_started = True
    client._cache = {7: stale}

    async def lines():
        assert client.get_cached(7) is None
        yield ''

    response = transport.get_async_httpx_client.return_value.stream.return_value
    response.__aenter__.return_value.status_code = 200
    response.__aenter__.return_value.aiter_lines = lines
    with pytest.raises(ConnectionError, match='closed'):
        await client.awatch()

    with pytest.raises(ServiceUnavailableException):
        client.get_cached(7)
    assert client._cache == {}
    transport.get_httpx_client.assert_not_called()
