"""Tests for ResourceEventLogger dedup — only metering-relevant transitions are
recorded, not every reconciler status heartbeat."""

from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from unittest.mock import AsyncMock, patch

import gpustack.server.resource_event_logger as rel
from gpustack.server.bus import EventType
from gpustack.server.resource_event_logger import (
    ResourceEventLogger,
    _resolve_principals,
)
from gpustack.schemas.resource_events import (
    EVENT_TYPE_CREATED,
    EVENT_TYPE_DELETED,
    EVENT_TYPE_PHASE_LEFT_METERED,
    EVENT_TYPE_PHASE_TO_METERED,
    RESOURCE_TYPE_GPU_INSTANCE,
    ResourceEvent,
)


def _evt(event_type, phase, rid=1):
    instance = SimpleNamespace(
        id=rid,
        name="g1",
        status={"phase": phase},
        spec={"resources": {"accelerator": "1", "cpu": "2", "ram": "2Gi"}},
        owner_principal_id=5,
        creator_id=None,
        cluster_id=None,
    )
    return SimpleNamespace(type=event_type, data=instance, id=rid, changed_fields={})


async def _drive(logger, events):
    types = []
    with patch.object(logger, "_write_event", new=AsyncMock()) as w:
        for e in events:
            await logger._handle_instance(e)
        for call in w.await_args_list:
            types.append(call.kwargs["event_type"])
    return types


@contextmanager
def _patch_services(principals: dict, clusters: dict):
    """Patch the cached ClusterService / PrincipalService the resolver uses,
    so tests exercise the resolution logic without a DB or the global cache.
    Yields the PrincipalService.get_by_id mock so call counts can be asserted."""
    cluster_get = AsyncMock(side_effect=lambda i: clusters.get(i))
    principal_get = AsyncMock(side_effect=lambda i: principals.get(i))
    with (
        patch("gpustack.server.resource_event_logger.ClusterService") as CS,
        patch("gpustack.server.resource_event_logger.PrincipalService") as PS,
    ):
        CS.return_value.get_by_id = cluster_get
        PS.return_value.get_by_id = principal_get
        yield principal_get


@pytest.mark.asyncio
async def test_resolve_principals_derives_owner_from_cluster():
    # consumer 5 created an instance on cluster 1, owned by provider 3.
    principals = {
        3: SimpleNamespace(name="provider-org"),
        5: SimpleNamespace(name="acme-org"),
        7: SimpleNamespace(name="bob"),
    }
    clusters = {1: SimpleNamespace(name="default", owner_principal_id=3)}
    with _patch_services(principals, clusters):
        out = await _resolve_principals(None, 5, 7, 1)
    # (owner_principal_id, owner_name, consumer_name, creator_name, cluster_name)
    assert out == (3, "provider-org", "acme-org", "bob", "default")


@pytest.mark.asyncio
async def test_resolve_principals_no_cluster_owner_equals_consumer():
    # No cluster (e.g. PV) → provider == consumer; resolved once (per-call memo).
    principals = {9: SimpleNamespace(name="alice")}
    with _patch_services(principals, clusters={}) as principal_get:
        out = await _resolve_principals(None, 9, 9, None)
    assert out == (9, "alice", "alice", "alice", None)
    assert principal_get.await_count == 1  # owner==consumer==creator → one lookup


@pytest.mark.asyncio
async def test_resolve_principals_missing_entity_yields_none():
    with _patch_services(principals={}, clusters={}):  # nothing resolves
        out = await _resolve_principals(None, 404, 404, 404)
    # cluster 404 missing → owner falls back to consumer id (404), name None
    assert out == (404, None, None, None, None)


@pytest.mark.asyncio
async def test_metered_recorded_once_across_status_storm():
    logger = ResourceEventLogger()
    # reconciler storm: None → Pending → Starting → Starting → Ready
    types = await _drive(
        logger,
        [
            _evt(EventType.CREATED, None),
            _evt(EventType.UPDATED, "Pending"),
            _evt(EventType.UPDATED, "Starting"),
            _evt(EventType.UPDATED, "Starting"),
            _evt(EventType.UPDATED, "Ready"),
        ],
    )
    assert types.count(EVENT_TYPE_CREATED) == 1
    # exactly one open-window event despite four metered updates
    assert types.count(EVENT_TYPE_PHASE_TO_METERED) == 1
    assert EVENT_TYPE_PHASE_LEFT_METERED not in types


@pytest.mark.asyncio
async def test_created_already_metered_opens_once():
    logger = ResourceEventLogger()
    types = await _drive(
        logger,
        [
            _evt(EventType.CREATED, "Pending"),  # created already metered
            _evt(EventType.UPDATED, "Ready"),
        ],
    )
    assert types.count(EVENT_TYPE_CREATED) == 1
    assert types.count(EVENT_TYPE_PHASE_TO_METERED) == 1


@pytest.mark.asyncio
async def test_leaving_metered_then_deleted():
    logger = ResourceEventLogger()
    types = await _drive(
        logger,
        [
            _evt(EventType.CREATED, "Ready"),
            _evt(EventType.UPDATED, "CreateFailed"),  # leaves metered
            _evt(EventType.DELETED, "CreateFailed"),  # already closed
        ],
    )
    assert types.count(EVENT_TYPE_PHASE_TO_METERED) == 1
    # one close on the failure transition; delete doesn't double-close
    assert types.count(EVENT_TYPE_PHASE_LEFT_METERED) == 1
    assert types.count(EVENT_TYPE_DELETED) == 1


# ---------------------------------------------------------------------------
# Restart warmup — rehydrate dedup guards from the event log so a restart
# doesn't re-emit a duplicate "Started".
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _yield(session):
    yield session


@pytest_asyncio.fixture
async def events_session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(ResourceEvent.__table__.create)
    async with AsyncSession(engine) as s:
        yield s
    await engine.dispose()


def _row(occurred_at, event_type, rid=1, phase=None):
    return ResourceEvent(
        occurred_at=occurred_at,
        event_type=event_type,
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        resource_id=rid,
        resource_name="g11",
        phase=phase,
    )


@pytest.mark.asyncio
async def test_warmup_no_duplicate_metered_after_restart(events_session):
    # Pre-restart history: instance became metered and never closed.
    events_session.add_all(
        [
            _row(datetime(2026, 5, 26, 10, 0, 0), EVENT_TYPE_CREATED),
            _row(
                datetime(2026, 5, 26, 10, 0, 5),
                EVENT_TYPE_PHASE_TO_METERED,
                phase="Ready",
            ),
        ]
    )
    await events_session.commit()

    logger_ = ResourceEventLogger()
    with patch.object(rel, "async_session", lambda: _yield(events_session)):
        await logger_._warmup_state()
    assert logger_._instance_metered.get(1) is True
    assert 1 in logger_._instance_created

    # First post-restart status update (still Ready) must NOT re-emit anything.
    types = await _drive(logger_, [_evt(EventType.UPDATED, "Ready", rid=1)])
    assert EVENT_TYPE_PHASE_TO_METERED not in types
    assert EVENT_TYPE_CREATED not in types


@pytest.mark.asyncio
async def test_warmup_skips_deleted_resource(events_session):
    # Latest event is a delete → the resource is gone, don't track it.
    events_session.add_all(
        [
            _row(
                datetime(2026, 5, 26, 10, 0, 5),
                EVENT_TYPE_PHASE_TO_METERED,
                phase="Ready",
            ),
            _row(datetime(2026, 5, 26, 11, 0, 0), EVENT_TYPE_DELETED),
        ]
    )
    await events_session.commit()

    logger_ = ResourceEventLogger()
    with patch.object(rel, "async_session", lambda: _yield(events_session)):
        await logger_._warmup_state()
    assert 1 not in logger_._instance_metered
    assert 1 not in logger_._instance_created
