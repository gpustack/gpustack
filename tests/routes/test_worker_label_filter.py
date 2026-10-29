import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import workers as workers_route
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.workers import Worker, WorkerListParams
from gpustack.server.bus import Event, EventType

CTX = TenantContext(
    user=SimpleNamespace(kind=PrincipalType.USER),
    is_platform_admin=False,
    current_principal_id=101,
    org_role=OrgRole.OWNER,
    current_is_personal_scope=False,
)
USER = SimpleNamespace(worker=None)


def _worker(worker_id, labels, owner=101, cluster=11):
    return Worker(
        id=worker_id,
        name=f"gpu-{worker_id}",
        labels=labels,
        owner_principal_id=owner,
        cluster_id=cluster,
        hostname=f"host-{worker_id}",
        ip=f"10.0.0.{worker_id}",
        ifname="eth0",
        port=10150,
        worker_uuid=f"uuid-{worker_id}",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest_asyncio.fixture
async def worker_db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Worker.__table__.create)
    async with AsyncSession(engine) as session:
        session.add_all(
            [
                _worker(1, {"region": "beijing", "env": "production"}),
                _worker(2, {"region": "beijing-east"}),
                _worker(3, {"Region": "Beijing"}, cluster=22),
                _worker(4, {"region": "shanghai", "city": "beijing"}),
                _worker(5, {}),
                _worker(6, {"region": "beijing"}, owner=202),
                _worker(7, {"city": "bei_jing%/\\"}),
                _worker(8, {"a.b/c": "北京=x=y"}),
                _worker(9, {"region": ""}),
                _worker(10, {'a"b': 'c\\d'}),
                _worker(11, {"city": "New York"}),
            ]
        )
        await session.commit()

    monkeypatch.setattr(
        workers_route,
        "async_session",
        lambda: AsyncSession(engine, expire_on_commit=False),
    )

    async def allocations(worker_ids):
        return {worker_id: SimpleNamespace(ram=0, vram={}) for worker_id in worker_ids}

    monkeypatch.setattr(workers_route, "get_workers_allocated", allocations)
    yield engine
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected_ids",
    [
        ("region=beijing", [1]),
        ("region=BEIJING", []),
        ("Region=Beijing", [3]),
        ("region=bei", []),
        ("region=beijing-east", [2]),
        ("region=", [9]),
        ("city=beijing", [4]),
        ("city=New York", [11]),
        ("city=bei_jing%/\\", [7]),
        ("a.b/c=北京=x=y", [8]),
        ('a"b=c\\d', [10]),
        (" region=beijing", []),
        ("region=beijing ", []),
        ("region=beijing env=production", []),
        ("missing=value", []),
        (None, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]),
        ("", [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]),
    ],
)
async def test_worker_label_filter_precedes_pagination(worker_db, query, expected_ids):
    actual_ids = []
    for page in range(1, max(1, (len(expected_ids) + 1) // 2) + 1):
        result = await workers_route.get_workers(
            USER,
            CTX,
            WorkerListParams(page=page, perPage=2, sort_by="name"),
            label=query,
        )
        actual_ids.extend(worker.id for worker in result.items)
        assert result.pagination.total == len(expected_ids)
        assert result.pagination.totalPage == (len(expected_ids) + 1) // 2
    assert actual_ids == sorted(expected_ids, key=lambda worker_id: f"gpu-{worker_id}")


@pytest.mark.asyncio
async def test_worker_label_filter_combines_with_existing_filters(worker_db):
    result = await workers_route.get_workers(
        USER,
        CTX,
        WorkerListParams(),
        label="region=beijing",
        cluster_id=11,
        search="gpu-1",
        uuid="uuid-1",
        name="gpu-1",
    )
    assert [worker.id for worker in result.items] == [1]
    assert result.pagination.total == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("sql_null", [False, True])
async def test_worker_label_filter_skips_null_labels(worker_db, sql_null):
    async with AsyncSession(worker_db) as session:
        session.add(_worker(12, None))
        await session.commit()
        if sql_null:
            await session.exec(text("UPDATE workers SET labels = NULL WHERE id = 12"))
            await session.commit()
    result = await workers_route.get_workers(
        USER, CTX, WorkerListParams(), label="region=beijing"
    )
    assert [worker.id for worker in result.items] == [1]
    assert result.pagination.total == 1


@pytest.mark.asyncio
async def test_me_lookup_overrides_label_filter(worker_db):
    result = await workers_route.get_workers(
        SimpleNamespace(worker=SimpleNamespace(id=5)),
        CTX,
        WorkerListParams(),
        label="region=beijing",
        me=True,
        search="missing",
        cluster_id=22,
    )
    assert [worker.id for worker in result.items] == [5]


def test_label_parameter_documents_exact_filter_and_invalid_input():
    app = FastAPI()
    app.include_router(workers_route.router, prefix="/workers")
    parameters = app.openapi()["paths"]["/workers"]["get"]["parameters"]
    label = next(p for p in parameters if p["name"] == "label")
    assert label["in"] == "query"
    assert "case-sensitive key=value" in label["description"]
    assert "422" in label["description"]


@pytest.mark.parametrize("watch", [False, True])
@pytest.mark.parametrize(
    "label", ["region", "beijing", "region beijing", "region:beijing", "=beijing", " "]
)
def test_invalid_label_returns_422_before_query_or_watch(monkeypatch, label, watch):
    session = MagicMock(side_effect=AssertionError("Unexpected database access"))
    subscribe = MagicMock(side_effect=AssertionError("Unexpected watch subscription"))
    monkeypatch.setattr(workers_route, "async_session", session)
    monkeypatch.setattr(Worker, "subscribe", subscribe)
    app = FastAPI()
    register_handlers(app)
    app.include_router(workers_route.router, prefix="/workers")
    app.dependency_overrides[get_current_user] = lambda: USER
    app.dependency_overrides[get_tenant_context] = lambda: CTX
    with TestClient(app) as client:
        response = client.get("/workers", params={"label": label, "watch": watch})
    assert response.status_code == 422, response.text
    assert "label" in response.text
    session.assert_not_called()
    subscribe.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "labels", [{"region": "beijing"}, {"Region": "Beijing"}, {"region": ""}]
)
async def test_worker_label_watch_tracks_membership_and_tenant_visibility(
    monkeypatch, labels
):
    key, value = next(iter(labels.items()))
    first = _worker(1, labels)
    outside = _worker(2, {"city": "shanghai"})
    other_tenant = _worker(3, labels, owner=202)
    entering = outside.model_copy(update={"labels": labels})
    leaving = first.model_copy(update={"labels": {"city": "shanghai"}})
    renamed = entering.model_copy(update={"name": "other"})
    events = [
        Event(type=EventType.CREATED, data=row)
        for row in [first, outside, other_tenant]
    ] + [
        Event(type=EventType.UPDATED, data=other_tenant),
        Event(type=EventType.UPDATED, data=first),
        Event(type=EventType.UPDATED, data=leaving),
        Event(type=EventType.UPDATED, data=leaving),
        Event(type=EventType.UPDATED, data=entering),
        Event(type=EventType.UPDATED, data=renamed),
        Event(type=EventType.UPDATED, data=first),
        Event(type=EventType.DELETED, data=first),
        Event(type=EventType.HEARTBEAT, data=None),
    ]

    async def subscribe(**kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(Worker, "subscribe", subscribe)
    enrich = AsyncMock()
    monkeypatch.setattr(workers_route, "_inject_allocated_into_event", enrich)
    response = await workers_route.get_workers(
        USER,
        CTX,
        WorkerListParams(watch=True),
        label=f"{key}={value}",
        search="gpu",
        cluster_id=11,
    )
    frames = [frame async for frame in response.body_iterator]
    assert frames[-1] == "\n\n"
    payloads = [json.loads(frame) for frame in frames[:-1]]
    assert [(p["type"], p["data"]["id"]) for p in payloads] == [
        (EventType.CREATED.value, 1),
        (EventType.UPDATED.value, 1),
        (EventType.DELETED.value, 1),
        (EventType.CREATED.value, 2),
        (EventType.DELETED.value, 2),
        (EventType.CREATED.value, 1),
        (EventType.DELETED.value, 1),
    ]
    assert payloads[2]["data"] == {"id": 1}
    assert payloads[4]["data"] == {"id": 2}
    assert enrich.await_count == 4


@pytest.mark.asyncio
async def test_worker_label_watch_requires_exact_pair(monkeypatch):
    labels = [
        {"region": "beijing-east"},
        {"city": "beijing"},
        {"region": "beijing"},
        {"region": "Beijing"},
        {"region": "beijing"},
        {},
    ]

    async def subscribe(**kwargs):
        for value in labels:
            yield Event(type=EventType.UPDATED, data=_worker(1, value))

    monkeypatch.setattr(Worker, "subscribe", subscribe)
    monkeypatch.setattr(workers_route, "_inject_allocated_into_event", AsyncMock())
    response = await workers_route.get_workers(
        USER, CTX, WorkerListParams(watch=True), label="region=beijing"
    )
    payloads = [json.loads(frame) async for frame in response.body_iterator]
    assert [(p["type"], p["data"]["id"]) for p in payloads] == [
        (EventType.CREATED.value, 1),
        (EventType.DELETED.value, 1),
        (EventType.CREATED.value, 1),
        (EventType.DELETED.value, 1),
    ]


@pytest.mark.asyncio
async def test_label_watch_only_forwards_known_id_only_deletions(monkeypatch):
    async def subscribe(**kwargs):
        yield Event(type=EventType.CREATED, data=_worker(1, {"region": "beijing"}))
        yield Event(type=EventType.DELETED, data={"id": 1})
        yield Event(type=EventType.DELETED, data={"id": 2})

    monkeypatch.setattr(Worker, "subscribe", subscribe)
    monkeypatch.setattr(workers_route, "_inject_allocated_into_event", AsyncMock())
    response = await workers_route.get_workers(
        USER,
        replace(CTX, is_platform_admin=True, current_principal_id=None),
        WorkerListParams(watch=True),
        label="region=beijing",
    )
    payloads = [json.loads(frame) async for frame in response.body_iterator]
    assert [(p["type"], p["data"]["id"]) for p in payloads] == [
        (EventType.CREATED.value, 1),
        (EventType.DELETED.value, 1),
    ]
