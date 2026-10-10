"""Target list and watch visibility derives from the owning route."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from gpustack.api.tenant import TenantContext
from gpustack.mixins import active_record
from gpustack.routes import model_routes
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    ModelRouteTargetListParams,
)
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.bus import Event, EventBus, EventType


def _ctx(scope):
    return TenantContext(
        user=SimpleNamespace(
            kind=PrincipalType.SYSTEM if "system" in scope else PrincipalType.USER
        ),
        is_platform_admin=scope.startswith("admin"),
        current_principal_id=(
            None
            if scope in ("admin-all", "system", "scoped-system")
            else 7 if scope == "personal" else 101
        ),
        org_role=OrgRole.OWNER,
        scoped_cluster_id=11 if scope == "scoped-system" else None,
    )


def _target(i):
    return ModelRouteTarget(
        id=i,
        name=f"target-{i}",
        route_id=i,
        route_name=f"route-{i}",
        model_id=i,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest_asyncio.fixture
async def parents(monkeypatch):
    rows = {
        i: SimpleNamespace(id=i, owner_principal_id=owner, deleted_at=None)
        for i, owner in [(1, 101), (2, 202), (3, 303)]
    }
    monkeypatch.setattr(model_routes, "async_session", MagicMock())
    monkeypatch.setattr(
        ModelRoute, "one_by_id", AsyncMock(side_effect=lambda session, i: rows.get(i))
    )
    monkeypatch.setattr("gpustack.server.cache._coordinator", None)
    monkeypatch.setattr(
        ModelRoute,
        "_do_cached_all_query",
        AsyncMock(side_effect=lambda options: list(rows.values())),
    )
    await ModelRoute._invalidate_cached_all()
    try:
        yield rows
    finally:
        await ModelRoute._invalidate_cached_all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,expected",
    [
        ("org", [1]),
        ("personal", []),
        ("admin-org", [1]),
        ("admin-all", [1, 2, 3]),
        ("system", [1, 2, 3]),
        ("scoped-system", [1, 2, 3]),
    ],
)
async def test_target_watch_filters_replay_and_bus(
    monkeypatch, parents, scope, expected
):
    rows = [_target(i) for i in (1, 2, 3)]
    monkeypatch.setattr(ModelRouteTarget, "cached_all", AsyncMock(return_value=rows))
    events = [
        Event(type=event_type, data=row)
        for event_type in (EventType.CREATED, EventType.UPDATED, EventType.DELETED)
        for row in rows
    ]
    subscriber = SimpleNamespace(
        receive=AsyncMock(side_effect=[*events, asyncio.CancelledError()])
    )
    bus = MagicMock()
    bus.subscribe.return_value = subscriber
    monkeypatch.setattr(active_record, "event_bus", bus)
    response = await model_routes.get_model_route_targets(
        ctx=_ctx(scope), params=ModelRouteTargetListParams(watch=True)
    )
    frames = []
    with pytest.raises(asyncio.CancelledError):
        async for frame in response.body_iterator:
            frames.append(json.loads(frame))
    assert [frame["data"]["id"] for frame in frames] == expected * 4
    bus.unsubscribe.assert_called_once()
    ModelRouteTarget.cached_all.assert_awaited_once()
    ModelRoute.one_by_id.assert_not_called()
    model_routes.async_session.assert_not_called()
    if scope in ("org", "personal", "admin-org"):
        ModelRoute._do_cached_all_query.assert_awaited_once()
    else:
        ModelRoute._do_cached_all_query.assert_not_called()


@pytest.mark.asyncio
async def test_target_watch_tracks_parent_changes(monkeypatch, parents):
    async def subscribe(**kwargs):
        yield Event(type=EventType.CREATED, data=_target(1))
        parents[1] = SimpleNamespace(id=1, owner_principal_id=202, deleted_at=None)
        await ModelRoute._invalidate_cached_all()
        yield Event(type=EventType.UPDATED, data=_target(1))
        parents[4] = SimpleNamespace(id=4, owner_principal_id=101, deleted_at=None)
        await ModelRoute._invalidate_cached_all()
        yield Event(type=EventType.CREATED, data=_target(4))
        parents[4].deleted_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
        await ModelRoute._invalidate_cached_all()
        yield Event(type=EventType.UPDATED, data=_target(4))
        yield Event(type=EventType.DELETED, data=_target(99))
        yield Event(type=EventType.DELETED, data={"id": 1})

    monkeypatch.setattr(ModelRouteTarget, "subscribe", subscribe)
    response = await model_routes.get_model_route_targets(
        ctx=_ctx("org"), params=ModelRouteTargetListParams(watch=True)
    )
    frames = [json.loads(frame) async for frame in response.body_iterator]
    assert [frame["data"]["id"] for frame in frames] == [1, 4]


@pytest.mark.asyncio
@pytest.mark.parametrize("row_count", [0, 1, 1000])
async def test_target_replay_and_live_events_share_parent_query(
    monkeypatch, parents, row_count
):
    rows = [_target(i) for i in range(1, row_count + 1)]
    for row in rows:
        row.route_id = 1
    monkeypatch.setattr(ModelRouteTarget, "cached_all", AsyncMock(return_value=rows))
    subscriber = SimpleNamespace(
        receive=AsyncMock(
            side_effect=[
                *(Event(type=EventType.UPDATED, data=row) for row in rows),
                asyncio.CancelledError(),
            ]
        )
    )
    bus = MagicMock()
    bus.subscribe.return_value = subscriber
    monkeypatch.setattr(active_record, "event_bus", bus)

    response = await model_routes.get_model_route_targets(
        ctx=_ctx("org"),
        params=ModelRouteTargetListParams(watch=True, route_id=1),
        search="target",
    )
    frames = []
    with pytest.raises(asyncio.CancelledError):
        async for frame in response.body_iterator:
            frames.append(json.loads(frame))

    assert [frame["data"]["id"] for frame in frames] == (
        list(range(1, row_count + 1)) * 2
    )
    assert ModelRoute._do_cached_all_query.await_count == (1 if row_count else 0)
    ModelRoute.one_by_id.assert_not_called()
    model_routes.async_session.assert_not_called()
    ModelRouteTarget.cached_all.assert_awaited_once()
    bus.unsubscribe.assert_called_once()


@pytest.mark.asyncio
async def test_shared_snapshot_keeps_tenant_decisions_separate(parents):
    own = model_routes._target_stream_visibility(_ctx("org"))
    other_ctx = _ctx("org")
    other_ctx.current_principal_id = 202
    other = model_routes._target_stream_visibility(other_ctx)
    assert await own(_target(1))
    assert not await other(_target(1))
    assert not await own(_target(2))
    assert await other(_target(2))
    ModelRoute._do_cached_all_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_parent_visibility_changes_when_shared_snapshot_is_invalidated(parents):
    visible = model_routes._target_stream_visibility(_ctx("org"))
    assert await visible(_target(1))
    parents[1] = SimpleNamespace(id=1, owner_principal_id=202, deleted_at=None)
    # Authorization uses the cached snapshot until invalidation or expiry.
    assert await visible(_target(1))
    ModelRoute._do_cached_all_query.assert_awaited_once()
    await ModelRoute._invalidate_cached_all()
    assert not await visible(_target(1))
    assert ModelRoute._do_cached_all_query.await_count == 2


@pytest.mark.asyncio
async def test_parent_cache_refresh_failure_does_not_reuse_old_authorization(
    monkeypatch, parents
):
    async def subscribe(**kwargs):
        yield Event(type=EventType.CREATED, data=_target(1))
        await ModelRoute._invalidate_cached_all()
        ModelRoute._do_cached_all_query.side_effect = RuntimeError("cache unavailable")
        yield Event(type=EventType.UPDATED, data=_target(1))

    monkeypatch.setattr(ModelRouteTarget, "subscribe", subscribe)
    response = await model_routes.get_model_route_targets(
        ctx=_ctx("org"), params=ModelRouteTargetListParams(watch=True)
    )
    frames = [json.loads(frame) async for frame in response.body_iterator]
    assert [frame["data"]["id"] for frame in frames] == [1]
    assert [frame["type"] for frame in frames] == [EventType.CREATED.value]
    assert ModelRoute._do_cached_all_query.await_count == 2


@pytest.mark.asyncio
async def test_events_during_replay_use_refreshed_parent_snapshot(monkeypatch, parents):
    bus = EventBus()
    monkeypatch.setattr(active_record, "event_bus", bus)
    topic = "modelroutetarget"

    async def load_targets(**kwargs):
        assert bus.subscribers.get(topic)
        await bus.publish(topic, Event(type=EventType.CREATED, data=_target(1)))
        await bus.publish(topic, Event(type=EventType.CREATED, data=_target(4)))
        await bus.publish(topic, Event(type=EventType.HEARTBEAT, data=None))
        return [_target(1)]

    monkeypatch.setattr(
        ModelRouteTarget, "cached_all", AsyncMock(side_effect=load_targets)
    )
    response = await model_routes.get_model_route_targets(
        ctx=_ctx("org"), params=ModelRouteTargetListParams(watch=True)
    )

    async def collect():
        frames = []
        try:
            frames.append(json.loads(await anext(response.body_iterator)))
            parents[1] = SimpleNamespace(id=1, owner_principal_id=202, deleted_at=None)
            parents[4] = SimpleNamespace(id=4, owner_principal_id=101, deleted_at=None)
            await ModelRoute._invalidate_cached_all()
            async for frame in response.body_iterator:
                if frame == "\n\n":
                    break
                frames.append(json.loads(frame))
        finally:
            await response.body_iterator.aclose()
        return frames

    frames = await asyncio.wait_for(collect(), timeout=1)
    assert [frame["data"]["id"] for frame in frames] == [1, 4]
    assert all(frame["type"] == EventType.CREATED.value for frame in frames)
    assert ModelRoute._do_cached_all_query.await_count == 2
    ModelRoute.one_by_id.assert_not_called()
    assert bus.subscribers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,owner", [("org", 101), ("personal", 7), ("admin-org", 101)]
)
async def test_target_list_scopes_parent_even_with_explicit_route_filter(
    monkeypatch, parents, scope, owner
):
    query = AsyncMock(return_value=SimpleNamespace(items=[]))
    monkeypatch.setattr(ModelRouteTarget, "paginated_by_query", query)
    monkeypatch.setattr(model_routes, "_apply_target_plugin_sections", AsyncMock())
    await model_routes.get_model_route_targets(
        ctx=_ctx(scope), params=ModelRouteTargetListParams(route_id=2)
    )
    kwargs = query.call_args.kwargs
    assert kwargs["fields"]["route_id"] == 2
    (condition,) = kwargs["extra_conditions"]
    sql = str(condition.compile(compile_kwargs={"literal_binds": True}))
    assert "model_route_targets.route_id IN (SELECT model_routes.id" in sql
    assert f"model_routes.owner_principal_id = {owner}" in sql
    assert "model_routes.deleted_at IS NULL" in sql
