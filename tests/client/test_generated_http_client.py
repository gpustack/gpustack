from unittest.mock import Mock

import pytest

from gpustack.client.generated_http_client import (
    AuthenticatedHTTPClient,
    HTTPClient,
)


@pytest.fixture(
    params=[
        lambda: HTTPClient(base_url="http://server.example"),
        lambda: AuthenticatedHTTPClient(base_url="http://server.example", token="t"),
    ],
    ids=["http-client", "authenticated-http-client"],
)
def client_factory(request):
    return request.param


def test_close_releases_constructed_sync_client(client_factory):
    client = client_factory()
    constructed = client.get_httpx_client()

    client.close()

    assert constructed.is_closed
    assert client._client is None


def test_close_is_safe_before_any_client_is_constructed(client_factory):
    client = client_factory()

    client.close()

    assert client._client is None


def test_close_allows_lazily_constructing_a_fresh_client(client_factory):
    client = client_factory()
    first = client.get_httpx_client()

    client.close()
    second = client.get_httpx_client()

    assert second is not first
    assert not second.is_closed


def test_close_leaves_async_client_untouched(client_factory):
    client = client_factory()
    async_client = client.get_async_httpx_client()

    client.close()

    assert client._async_client is async_client
    assert not async_client.is_closed


@pytest.mark.asyncio
async def test_aclose_releases_constructed_clients(client_factory):
    client = client_factory()
    sync_client = client.get_httpx_client()
    async_client = client.get_async_httpx_client()

    await client.aclose()

    assert sync_client.is_closed
    assert async_client.is_closed
    assert client._client is None
    assert client._async_client is None


@pytest.mark.asyncio
async def test_aclose_is_idempotent(client_factory):
    client = client_factory()

    await client.aclose()
    await client.aclose()

    assert client._client is None
    assert client._async_client is None


@pytest.mark.asyncio
async def test_aclose_is_safe_before_any_client_is_constructed(client_factory):
    client = client_factory()

    await client.aclose()

    assert client._client is None
    assert client._async_client is None


@pytest.mark.asyncio
async def test_aclose_closes_async_client_when_sync_close_fails(client_factory):
    client = client_factory()
    sync_client = client.get_httpx_client()
    async_client = client.get_async_httpx_client()
    sync_client.close = Mock(side_effect=RuntimeError("close failed"))

    with pytest.raises(RuntimeError):
        await client.aclose()

    assert async_client.is_closed
    assert client._async_client is None
