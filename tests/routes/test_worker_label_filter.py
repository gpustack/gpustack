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
from gpustack.server.db import get_session

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
                _worker(6, {"region": "beijing", "env": "production"}, owner=202),
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
            label=[query] if query is not None else None,
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
        label=["region=beijing"],
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
        USER, CTX, WorkerListParams(), label=["region=beijing"]
    )
    assert [worker.id for worker in result.items] == [1]
    assert result.pagination.total == 1


@pytest.mark.asyncio
async def test_me_lookup_overrides_label_filter(worker_db):
    result = await workers_route.get_workers(
        SimpleNamespace(worker=SimpleNamespace(id=5)),
        CTX,
        WorkerListParams(),
        label=["region=beijing", "env=production"],
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
    assert "same key (OR)" in label["description"]
    assert "different key (AND)" in label["description"]
    assert "422" in label["description"]
    schema = next(s for s in label["schema"]["anyOf"] if s["type"] == "array")
    assert "pattern" in schema["items"]


@pytest.mark.parametrize("watch", [False, True])
@pytest.mark.parametrize("position", ["only", "first", "last"])
@pytest.mark.parametrize(
    "label", ["region", "beijing", "region beijing", "region:beijing", "=beijing", " "]
)
def test_invalid_label_returns_422_before_query_or_watch(
    monkeypatch, label, watch, position
):
    session = MagicMock(side_effect=AssertionError("Unexpected database access"))
    subscribe = MagicMock(side_effect=AssertionError("Unexpected watch subscription"))
    monkeypatch.setattr(workers_route, "async_session", session)
    monkeypatch.setattr(Worker, "subscribe", subscribe)
    app = FastAPI()
    register_handlers(app)
    app.include_router(workers_route.router, prefix="/workers")
    app.dependency_overrides[get_current_user] = lambda: USER
    app.dependency_overrides[get_tenant_context] = lambda: CTX
    labels = {
        "only": [label],
        "first": [label, "env=production"],
        "last": ["region=beijing", "", label],
    }[position]
    with TestClient(app) as client:
        response = client.get("/workers", params={"label": labels, "watch": watch})
    assert response.status_code == 422, response.text
    assert "label" in response.text
    session.assert_not_called()
    subscribe.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "labels",
    [
        {"region": "beijing"},
        {"Region": "Beijing"},
        {"region": ""},
        {"region": "beijing", "env": "production"},
    ],
)
async def test_worker_label_watch_tracks_membership_and_tenant_visibility(
    monkeypatch, labels
):
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
        label=[f"{key}={value}" for key, value in labels.items()],
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
        USER, CTX, WorkerListParams(watch=True), label=["region=beijing"]
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
        label=["region=beijing"],
    )
    payloads = [json.loads(frame) async for frame in response.body_iterator]
    assert [(p["type"], p["data"]["id"]) for p in payloads] == [
        (EventType.CREATED.value, 1),
        (EventType.DELETED.value, 1),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "queries,expected_ids",
    [
        (["region=beijing", "env=production"], [1, 12]),
        (["region=beijing", "env=staging"], [13]),
        (["region=beijing", "region=shanghai"], [1, 12, 13, 14, 4]),
        (["region=beijing", "region=beijing"], [1, 12, 13]),
        (["region=beijing", "region=shanghai", "env=production"], [1, 12, 14]),
        (["region=beijing", "env=production", "env=staging"], [1, 12, 13]),
        (["region=beijing", "Region=Beijing"], []),
        (["region=beijing", "region=BEIJING"], [1, 12, 13]),
        (["region=", "region=shanghai"], [14, 4, 9]),
        (["region=beijing", "missing=value", "missing=other"], []),
        (["city=New York", "city=beijing"], [11, 4]),
        (["", "region=beijing", "env=production", ""], [1, 12]),
        (["city=New York", "env=production"], []),
    ],
)
async def test_multiple_label_filters_group_values_before_pagination(
    worker_db, queries, expected_ids
):
    async with AsyncSession(worker_db) as session:
        session.add_all(
            [
                _worker(12, {"region": "beijing", "env": "production"}),
                _worker(13, {"region": "beijing", "env": "staging"}),
                _worker(14, {"region": "shanghai", "env": "production"}),
            ]
        )
        await session.commit()
    actual_ids = []
    for page in range(1, max(1, len(expected_ids)) + 1):
        result = await workers_route.get_workers(
            USER,
            CTX,
            WorkerListParams(page=page, perPage=1, sort_by="name"),
            label=queries,
        )
        actual_ids.extend(worker.id for worker in result.items)
        assert result.pagination.total == len(expected_ids)
        assert result.pagination.totalPage == len(expected_ids)
    assert actual_ids == expected_ids


@pytest.mark.asyncio
async def test_grouped_labels_preserve_name_cluster_and_tenant_filters(worker_db):
    async with AsyncSession(worker_db) as session:
        session.add_all(
            [
                _worker(12, {"region": "shanghai", "env": "production"}),
                _worker(13, {"region": "shanghai", "env": "production"}, owner=202),
                _worker(14, {"region": "shanghai", "env": "production"}, cluster=22),
            ]
        )
        await session.commit()
    result = await workers_route.get_workers(
        USER,
        CTX,
        WorkerListParams(),
        label=["region=beijing", "region=shanghai", "env=production"],
        search="gpu-1",
        cluster_id=11,
    )
    assert [worker.id for worker in result.items] == [1, 12]
    assert result.pagination.total == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "queries,expected_ids",
    [
        (["region=beijing", "env=production"], [1]),
        (["region=beijing", "env=missing"], []),
        (["region=beijing", "region=shanghai"], [1, 4]),
        (["region=beijing", "region=shanghai", "env=production"], [1]),
        (["region=beijing", "env=production", "env=staging"], [1]),
        (["region=beijing", "Region=Beijing"], []),
        (["region=", "region=shanghai"], [4, 9]),
        (["", "region=beijing", "env=production"], [1]),
        (["region=beijing"], [1]),
        (["", ""], [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]),
    ],
)
async def test_repeated_label_http_parameters(worker_db, queries, expected_ids):
    app = FastAPI()
    register_handlers(app)
    app.include_router(workers_route.router, prefix="/workers")
    app.dependency_overrides[get_current_user] = lambda: USER
    app.dependency_overrides[get_tenant_context] = lambda: CTX
    with TestClient(app) as client:
        response = client.get("/workers", params={"label": queries, "perPage": 100})
    assert response.status_code == 200, response.text
    data = response.json()
    assert sorted(worker["id"] for worker in data["items"]) == expected_ids
    assert data["pagination"]["total"] == len(expected_ids)


@pytest.mark.asyncio
async def test_multiple_label_watch_requires_all_pairs(monkeypatch):
    labels = [
        {"region": "beijing"},
        {"env": "production"},
        {"region": "beijing", "env": "production"},
        {"region": "beijing", "env": "Production"},
        {"region": "beijing", "env": "production"},
        {"region": "shanghai", "env": "production"},
        {"region": "beijing", "env": "production"},
    ]

    async def subscribe(**kwargs):
        for value in labels:
            yield Event(type=EventType.UPDATED, data=_worker(1, value))

    monkeypatch.setattr(Worker, "subscribe", subscribe)
    monkeypatch.setattr(workers_route, "_inject_allocated_into_event", AsyncMock())
    response = await workers_route.get_workers(
        USER,
        CTX,
        WorkerListParams(watch=True),
        label=["region=beijing", "env=production"],
    )
    payloads = [json.loads(frame) async for frame in response.body_iterator]
    assert [(p["type"], p["data"]["id"]) for p in payloads] == [
        (EventType.CREATED.value, 1),
        (EventType.DELETED.value, 1),
        (EventType.CREATED.value, 1),
        (EventType.DELETED.value, 1),
        (EventType.CREATED.value, 1),
    ]


@pytest.mark.asyncio
async def test_label_watch_groups_values_and_tracks_membership(monkeypatch):
    first = _worker(1, {"region": "beijing", "env": "production"})
    second = _worker(2, {"region": "shanghai", "env": "production"})
    wrong_env = _worker(3, {"region": "beijing", "env": "staging"})
    other_tenant = _worker(4, second.labels, owner=202)
    other_cluster = _worker(5, second.labels, cluster=22)
    empty_value = _worker(6, {"region": "", "env": "production"})
    missing_key = _worker(7, {"env": "production"})
    wrong_case = _worker(8, {"Region": "Beijing", "env": "production"})
    events = [
        Event(type=EventType.CREATED, data=row)
        for row in [
            first,
            second,
            wrong_env,
            other_tenant,
            other_cluster,
            empty_value,
            missing_key,
            wrong_case,
        ]
    ] + [
        Event(
            type=EventType.UPDATED,
            data=first.model_copy(update={"labels": second.labels}),
        ),
        Event(
            type=EventType.UPDATED,
            data=second.model_copy(update={"labels": wrong_env.labels}),
        ),
        Event(
            type=EventType.UPDATED,
            data=second.model_copy(update={"labels": first.labels}),
        ),
        Event(
            type=EventType.UPDATED,
            data=first.model_copy(
                update={"labels": {"region": "hangzhou", "env": "production"}}
            ),
        ),
        Event(type=EventType.UPDATED, data=first),
        Event(type=EventType.UPDATED, data=first.model_copy(update={"name": "other"})),
        Event(type=EventType.UPDATED, data=other_tenant),
        Event(type=EventType.DELETED, data=second),
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
        label=["region=beijing", "region=shanghai", "region=", "env=production"],
        search="gpu",
        cluster_id=11,
    )
    frames = [frame async for frame in response.body_iterator]
    assert frames[-1] == "\n\n"
    payloads = [json.loads(frame) for frame in frames[:-1]]
    assert [(p["type"], p["data"]["id"]) for p in payloads] == [
        (EventType.CREATED.value, 1),
        (EventType.CREATED.value, 2),
        (EventType.CREATED.value, 6),
        (EventType.UPDATED.value, 1),
        (EventType.DELETED.value, 2),
        (EventType.CREATED.value, 2),
        (EventType.DELETED.value, 1),
        (EventType.CREATED.value, 1),
        (EventType.DELETED.value, 1),
        (EventType.DELETED.value, 2),
    ]
    for payload in payloads[:-1]:
        if payload["type"] == EventType.DELETED.value:
            assert payload["data"] == {"id": payload["data"]["id"]}
    assert enrich.await_count == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ctx,cluster_id,expected",
    [
        (CTX, None, ["", "beijing", "beijing-east", "shanghai"]),
        (CTX, 11, ["", "beijing", "beijing-east", "shanghai"]),
        (CTX, 22, ["beijing-east"]),
        (CTX, 999, []),
        (CTX, 0, []),
        (
            replace(CTX, is_platform_admin=True),
            None,
            ["", "beijing", "beijing-east", "shanghai"],
        ),
        (
            replace(CTX, org_role=OrgRole.MEMBER),
            None,
            ["", "beijing", "beijing-east", "shanghai"],
        ),
        (
            replace(CTX, current_is_personal_scope=True, org_role=None),
            None,
            ["", "beijing", "beijing-east", "shanghai"],
        ),
        (
            replace(CTX, is_platform_admin=True, current_principal_id=None),
            None,
            ["", "beijing", "beijing-east", "private", "shanghai"],
        ),
        (
            replace(
                CTX,
                user=SimpleNamespace(kind=PrincipalType.SYSTEM),
                scoped_cluster_id=22,
                scoped_cluster_owner_id=101,
            ),
            None,
            ["beijing-east"],
        ),
        (
            replace(
                CTX,
                user=SimpleNamespace(kind=PrincipalType.SYSTEM),
                scoped_cluster_id=22,
                scoped_cluster_owner_id=101,
            ),
            11,
            [],
        ),
    ],
)
async def test_worker_label_options_use_visible_live_workers(
    worker_db, ctx, cluster_id, expected
):
    rows = [
        _worker(12, {"region": "beijing-east"}, cluster=22),
        _worker(13, {"region": "private"}, owner=202),
        _worker(14, {"region": "deleted"}),
    ]
    rows[-1].deleted_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    async with AsyncSession(worker_db) as session:
        session.add_all(rows)
        await session.commit()
        options = await workers_route.get_worker_label_options(ctx, session, cluster_id)
    regions = next((option.values for option in options if option.key == "region"), [])
    assert regions == expected


@pytest.mark.asyncio
async def test_worker_label_options_keep_literal_strings_and_worker_name(worker_db):
    labels = {
        "worker-name": "gpu-12",
        "region": "beijing",
        "Region": "Beijing",
        "city": " New York, US ",
        "url": "http://host:8000?a=b",
        " ": " ",
        "": "invalid",
        "bad=key": "invalid",
        "number": 12,
        "null": None,
        "list": ["invalid"],
        "nested": {"invalid": "value"},
    }
    async with AsyncSession(worker_db) as session:
        session.add_all([_worker(12, labels), _worker(13, None)])
        await session.commit()
        options = await workers_route.get_worker_label_options(CTX, session)
    assert [option.key for option in options] == sorted(
        option.key for option in options
    )
    values = {option.key: option.values for option in options}
    assert values["worker-name"] == ["gpu-12"]
    assert values["region"] == ["", "beijing", "beijing-east", "shanghai"]
    assert values["Region"] == ["Beijing"]
    assert values["city"] == [" New York, US ", "New York", "bei_jing%/\\", "beijing"]
    assert values["url"] == ["http://host:8000?a=b"]
    assert values[" "] == [" "]
    assert values["a.b/c"] == ["北京=x=y"]
    assert not set(values) & {"", "bad=key", "number", "null", "list", "nested"}


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
async def test_worker_label_options_select_only_labels_in_one_query(
    worker_db, monkeypatch, dialect
):
    from sqlalchemy.dialects import mysql, postgresql

    allocations = AsyncMock(
        side_effect=AssertionError("Resource allocations are not label candidates")
    )
    projection = MagicMock(
        side_effect=AssertionError("Full worker projection is unnecessary")
    )
    monkeypatch.setattr(workers_route, "get_workers_allocated", allocations)
    monkeypatch.setattr(workers_route, "to_worker_public", projection)
    async with AsyncSession(worker_db) as session:
        execute = AsyncMock(wraps=session.exec)
        monkeypatch.setattr(session, "exec", execute)
        await workers_route.get_worker_label_options(CTX, session, 11)
    execute.assert_awaited_once()
    statement = execute.await_args.args[0]
    assert [column.name for column in statement.selected_columns] == ["labels"]
    compiled = statement.compile(
        dialect={"mysql": mysql.dialect(), "postgresql": postgresql.dialect()}[dialect]
    )
    sql = str(compiled)
    assert "workers.deleted_at IS NULL" in sql
    assert "workers.owner_principal_id =" in sql
    assert "workers.cluster_id =" in sql
    assert set(compiled.params.values()) == {101, 11}
    allocations.assert_not_awaited()
    projection.assert_not_called()


@pytest.mark.asyncio
async def test_worker_label_options_cover_one_thousand_workers_without_pagination(
    worker_db,
):
    async with AsyncSession(worker_db) as session:
        session.add_all(
            [
                _worker(
                    100 + index,
                    {f"key-{key}": f"value-{index:04d}-{key}" for key in range(10)},
                    cluster=99,
                )
                for index in range(1000)
            ]
        )
        await session.commit()
        options = await workers_route.get_worker_label_options(CTX, session, 99)
    assert len(options) == 10
    assert sum(len(option.values) for option in options) == 10000
    for option in options:
        assert option.values == sorted(set(option.values))


@pytest.mark.asyncio
async def test_worker_label_options_http_route_and_documentation(worker_db):
    app = FastAPI()
    app.include_router(workers_route.router, prefix="/workers")
    app.dependency_overrides[get_tenant_context] = lambda: CTX

    async def session_override():
        async with AsyncSession(worker_db) as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    with TestClient(app) as client:
        response = client.get(
            "/workers/label-options",
            params={"cluster_id": 22, "label": "missing=value", "search": "missing"},
        )
    assert response.status_code == 200, response.text
    assert response.json() == [{"key": "Region", "values": ["Beijing"]}]
    operation = app.openapi()["paths"]["/workers/label-options"]["get"]
    assert "resource allocations" in operation["description"]
    assert {parameter["name"] for parameter in operation["parameters"]} >= {
        "cluster_id"
    }
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["type"]
        == "array"
    )
