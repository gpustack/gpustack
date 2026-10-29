from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from gpustack.client.generated_worker_client import WorkerClient
from gpustack.schemas.workers import WorkerPublic


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
