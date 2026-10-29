import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from gpustack.client.generated_worker_client import WorkerClient
from gpustack.schemas.workers import WorkerPublic
from gpustack.server.bus import EventType


@pytest.fixture
def client():
    http_client = MagicMock()
    client = WorkerClient(http_client)
    client._watch_started = True
    for worker_id, labels, cluster_id in [
        (1, {"region": "beijing", "env": "production"}, 11),
        (2, {"region": "shanghai"}, 22),
        (3, {"zone": "beijing"}, 22),
        (4, {}, 11),
    ]:
        client._cache[worker_id] = WorkerPublic(
            id=worker_id,
            name=f"worker-{worker_id}",
            hostname=f"host-{worker_id}",
            ip=f"10.0.0.{worker_id}",
            ifname="eth0",
            port=10150,
            cluster_id=cluster_id,
            labels=labels,
            status=None,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    return client


@pytest.mark.parametrize(
    "query,expected",
    [
        ("region", [1, 2]),
        ("beijing", [1, 3]),
        ("BEI", [1, 3]),
        ("region beijing", [1]),
        ("  REGION\tbeijing  ", [1]),
        ("region shanghai", [2]),
        ("region production", [1]),
        ("region missing", []),
        (" \t\n", [1, 2, 3, 4]),
        ("region=beijing", [1]),
        ("region:beijing", [1]),
        ("region:bei", []),
        ("region=bei", []),
        ("Region=beijing", []),
        ("%", []),
        ("", [1, 2, 3, 4]),
        (None, [1, 2, 3, 4]),
    ],
)
def test_worker_cache_filters_label_keys_and_values(client, query, expected):
    result = client.list(params={"label": query})
    assert [worker.id for worker in result.items] == expected
    assert result.pagination.total == len(expected)
    client._client.get_httpx_client.assert_not_called()


def test_worker_cache_combines_label_and_cluster_filters(client):
    result = client.list(params={"label": "region beijing", "cluster_id": 11})
    assert [worker.id for worker in result.items] == [1]
    assert result.pagination.total == 1


def _api_list(client, items):
    client._client.get_httpx_client.return_value.get.return_value = httpx.Response(
        200,
        json={
            "items": [item.model_dump(mode="json") for item in items],
            "pagination": {
                "page": 1,
                "perPage": 100,
                "total": len(items),
                "totalPage": 1 if items else 0,
            },
        },
    )
    return client._client.get_httpx_client.return_value.get


@asynccontextmanager
async def _watching(client, params, replay):
    frames = asyncio.Queue()
    callbacks = asyncio.Queue()

    def enqueue(event_type, data):
        frames.put_nowait(
            json.dumps(
                {
                    "type": event_type.value,
                    "data": (
                        data.model_dump(mode="json")
                        if isinstance(data, WorkerPublic)
                        else data
                    ),
                }
            )
        )

    async def lines():
        while True:
            yield await frames.get()

    @asynccontextmanager
    async def stream(*args, **kwargs):
        yield SimpleNamespace(status_code=200, aiter_lines=lines)

    client._client.get_async_httpx_client.return_value.stream.side_effect = stream
    for item in replay:
        enqueue(EventType.CREATED, item)
    task = asyncio.create_task(client.awatch(callback=callbacks.put, params=params))

    async def send(event_type, data):
        enqueue(event_type, data)
        return await asyncio.wait_for(callbacks.get(), timeout=2)

    try:
        for _ in replay:
            await asyncio.wait_for(callbacks.get(), timeout=2)
        yield send
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "watch_params",
    [
        {"label": "region=beijing"},
        {"label": "region:beijing"},
        {"label": "region beijing"},
        {"label": "beijing"},
        {"label": "region=beijing", "cluster_id": 11},
        {"cluster_id": 11},
        {"name": "worker-1"},
        {"search": "worker-1"},
        {"uuid": "worker-uuid"},
    ],
)
async def test_filtered_worker_watch_lists_fall_back_to_complete_api(
    client, watch_params
):
    rows = list(client._cache.values())
    async with _watching(client, dict(watch_params), rows[:1]):
        api = _api_list(client, rows)
        assert [item.id for item in client.list().items] == [1, 2, 3, 4]
        api.assert_called_with("/workers", params={"page": -1})

        _api_list(client, rows[1:2])
        params = {"label": "region=shanghai"}
        result = client.list(params=params)
        assert [item.id for item in result.items] == [2]
        assert result.pagination.total == 1
        api.assert_called_with(
            "/workers", params={"label": "region=shanghai", "page": -1}
        )
        assert params == {"label": "region=shanghai"}

        _api_list(client, rows[:1])
        assert [item.id for item in client.list(params=watch_params).items] == [1]
        api.assert_called_with("/workers", params={**watch_params, "page": -1})

        pagination = {"label": "beijing", "page": 2, "perPage": 1}
        client.list(params=pagination)
        api.assert_called_with("/workers", params=pagination)
    assert not client._watch_started
    assert not client._cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "watch_params",
    [
        None,
        {},
        {"watch": True},
        {"label": ""},
        {"label": None},
        {"page": 1, "perPage": 1, "sort_by": "name"},
    ],
)
async def test_unfiltered_worker_watch_keeps_cached_label_searches(
    client, watch_params
):
    rows = list(client._cache.values())
    async with _watching(client, watch_params, rows):
        assert [item.id for item in client.list().items] == [1, 2, 3, 4]
        result = client.list(params={"label": "region beijing"})
        assert [item.id for item in result.items] == [1]
        client._client.get_httpx_client.assert_not_called()


@pytest.mark.asyncio
async def test_filtered_worker_watch_enriches_removal_callbacks(client):
    row = client._cache[1]
    async with _watching(client, {"label": "region=beijing"}, [row]) as send:
        removed = await send(EventType.DELETED, {"id": row.id})
        assert removed.type == EventType.DELETED
        assert isinstance(removed.data, WorkerPublic)
        assert removed.data.id == row.id
        assert row.id not in client._cache

        entered = await send(EventType.CREATED, row)
        assert entered.data["id"] == row.id
        api = _api_list(client, [row])
        assert [item.id for item in client.list().items] == [row.id]
        api.assert_called_once_with("/workers", params={"page": -1})


@pytest.mark.asyncio
async def test_worker_watch_reconnect_switches_cache_authority(client):
    rows = list(client._cache.values())
    async with _watching(client, {"label": "region=beijing"}, rows[:1]):
        _api_list(client, rows)
        assert [item.id for item in client.list().items] == [1, 2, 3, 4]
    assert not client._cache

    client._client.get_httpx_client.reset_mock()
    async with _watching(client, None, rows):
        assert [item.id for item in client.list().items] == [1, 2, 3, 4]
        client._client.get_httpx_client.assert_not_called()

    async with _watching(client, {"label": "region=shanghai"}, rows[1:2]):
        api = _api_list(client, rows)
        assert [item.id for item in client.list().items] == [1, 2, 3, 4]
        api.assert_called_once_with("/workers", params={"page": -1})
    assert not client._watch_started
    assert not client._cache
