"""Cluster watches preserve grants while restricting cluster-bound accounts."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpustack.api.tenant import TenantContext
from gpustack.routes.clusters import get_clusters
from gpustack.schemas.clusters import Cluster, ClusterListParams
from gpustack.schemas.principals import PrincipalType
from gpustack.server.bus import Event, EventType


@pytest.mark.asyncio
@pytest.mark.parametrize("mine", [False, True])
@pytest.mark.parametrize("scope", ["org", "scoped-system", "system", "admin"])
async def test_cluster_watch_scope(monkeypatch, mine, scope):
    ctx = TenantContext(
        user=SimpleNamespace(
            kind=PrincipalType.SYSTEM if "system" in scope else PrincipalType.USER
        ),
        is_platform_admin=scope == "admin",
        current_principal_id=101 if scope == "org" else None,
        org_role=None,
        scoped_cluster_id=1 if scope == "scoped-system" else None,
        accessible_cluster_ids={2},
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        Cluster(
            id=i,
            name=f"cluster-{i}",
            owner_principal_id=owner,
            created_at=now,
            updated_at=now,
        )
        for i, owner in [(1, 101), (2, 202), (3, 303)]
    ]

    async def subscribe(**kwargs):
        for event_type in (EventType.CREATED, EventType.UPDATED, EventType.DELETED):
            for row in rows:
                yield Event(type=event_type, data=row)

    monkeypatch.setattr(Cluster, "subscribe", subscribe)
    response = await get_clusters(
        ctx=ctx,
        params=ClusterListParams(watch=True),
        mine=mine,
        gpu_instance_enabled=None,
    )
    frames = [json.loads(frame) async for frame in response.body_iterator]
    expected = [1, 2, 3]
    if scope == "org":
        expected = [1] if mine else [1, 2]
    elif scope == "scoped-system":
        expected = [1]
    assert [frame["data"]["id"] for frame in frames] == expected * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("mine", [False, True])
async def test_scoped_system_watch_accepts_own_id_only_deletion(monkeypatch, mine):
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.SYSTEM),
        is_platform_admin=False,
        current_principal_id=None,
        org_role=None,
        scoped_cluster_id=1,
        accessible_cluster_ids={2},
    )

    async def subscribe(**kwargs):
        # Cluster binding authorizes only the matching ID, even with a grant
        # for another cluster. Missing IDs cannot establish visibility.
        for data in ({"id": 2}, {}, {"id": None}, None, {"id": 1}):
            yield Event(type=EventType.DELETED, data=data)

    monkeypatch.setattr(Cluster, "subscribe", subscribe)
    response = await get_clusters(
        ctx=ctx,
        params=ClusterListParams(watch=True),
        mine=mine,
        gpu_instance_enabled=None,
    )
    frames = [json.loads(frame) async for frame in response.body_iterator]
    assert [frame["data"] for frame in frames] == [{"id": 1}]
    assert [frame["type"] for frame in frames] == [EventType.DELETED.value]
