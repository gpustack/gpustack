"""Tests for ``ResourceUsageCollector`` — window extraction, event dispatch
state machine, and the ``settled_until`` idempotency clamp.

The DB-touching upsert path (``_upsert_day``) is patched out; the math it
depends on is covered by ``_clamped_seconds`` (pure) here and by the splitter
tests in ``tests/utils/test_resource_usage.py``.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

import gpustack.server.resource_usage_collector as rc
from gpustack.schemas.resource_events import (
    EVENT_TYPE_CREATED,
    EVENT_TYPE_DELETED,
    EVENT_TYPE_PHASE_LEFT_METERED,
    EVENT_TYPE_PHASE_TO_METERED,
    RESOURCE_TYPE_GPU_INSTANCE,
    ResourceEvent,
)
from gpustack.server.resource_usage_collector import (
    ResourceUsageCollector,
    _clamped_seconds,
    _open_window_from_event,
)


def make_event(
    *,
    occurred_at: datetime,
    event_type: str,
    resource_id: int = 1,
    resource_type: str = RESOURCE_TYPE_GPU_INSTANCE,
    resource_name: str = "gpu-instance-1",
    phase: str = "Ready",
    spec_snapshot: dict | None = None,
) -> ResourceEvent:
    snap = spec_snapshot or {
        "name": resource_name,
        "display_name": "GPU 1",
        # Device descriptor blob (JSON string) carrying per-card VRAM.
        "description": '{"spec": {"product": "NVIDIA-H100", "memory": "80Gi"}}',
        "spec": {
            "type_": "gpustack-nvidia-h100-c9bjn",
            "resources": {"cpu": "8", "ram": "125Gi", "accelerator": "2"},
            "volume": {"ephemeral": {"capacity": "500Gi"}},
        },
    }
    return ResourceEvent(
        occurred_at=occurred_at,
        owner_principal_id=42,
        owner_name="acme-org",
        creator_id=7,
        creator_name="bob",
        cluster_id=1,
        cluster_name="default",
        resource_type=resource_type,
        resource_id=resource_id,
        resource_name=resource_name,
        event_type=event_type,
        phase=phase,
        spec_snapshot=snap,
    )


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------


def test_open_window_extracts_sku_and_dims():
    evt = make_event(
        occurred_at=datetime(2026, 5, 26, 10, 0, 5),
        event_type=EVENT_TYPE_PHASE_TO_METERED,
    )
    w = _open_window_from_event(evt)
    assert w is not None
    assert w.resource_id == 1
    assert w.resource_type == RESOURCE_TYPE_GPU_INSTANCE
    assert w.gpu_count == 2
    # sku = the spec's ``type`` verbatim (the flavor name).
    assert w.sku == "gpustack-nvidia-h100-c9bjn"
    assert w.dimensions["gpu_type"] == "nvidia-h100"
    assert w.dimensions["memory_mib"] == 125 * 1024
    assert w.dimensions["vram_mib"] == 80 * 1024  # per-card VRAM from description
    assert w.window_start == datetime(2026, 5, 26, 10, 0, 5)
    # owner/creator/cluster name snapshots propagate from the event so the
    # rollup row stays attributable after the principals/cluster are deleted.
    assert w.owner_name == "acme-org"
    assert w.creator_name == "bob"
    assert w.cluster_name == "default"


def test_open_window_none_without_resource_id():
    evt = make_event(
        occurred_at=datetime(2026, 5, 26, 10, 0, 5),
        event_type=EVENT_TYPE_PHASE_TO_METERED,
        resource_id=None,
    )
    assert _open_window_from_event(evt) is None


# ---------------------------------------------------------------------------
# settled_until clamp (the v2 idempotency core)
# ---------------------------------------------------------------------------


def test_clamped_seconds_fresh():
    # no prior high-water → full window counts
    assert (
        _clamped_seconds(
            datetime(2026, 5, 26, 10, 0, 0), datetime(2026, 5, 26, 10, 10, 0), None
        )
        == 600
    )


def test_clamped_seconds_partial_overlap():
    # already settled to 10:05 → only 10:05..10:10 counts
    assert (
        _clamped_seconds(
            datetime(2026, 5, 26, 10, 0, 0),
            datetime(2026, 5, 26, 10, 10, 0),
            datetime(2026, 5, 26, 10, 5, 0),
        )
        == 300
    )


def test_clamped_seconds_full_overlap_is_zero():
    # re-processing an already-settled segment adds nothing (replay-safe)
    assert (
        _clamped_seconds(
            datetime(2026, 5, 26, 10, 0, 0),
            datetime(2026, 5, 26, 10, 10, 0),
            datetime(2026, 5, 26, 10, 10, 0),
        )
        == 0
    )


def test_clamped_seconds_gap_window_not_double_counted():
    # window2 [10:20,10:30] with prior settled at 10:10 (window1 close) →
    # the 10:10–10:20 stop gap is excluded, full 600s of window2 counts.
    assert (
        _clamped_seconds(
            datetime(2026, 5, 26, 10, 20, 0),
            datetime(2026, 5, 26, 10, 30, 0),
            datetime(2026, 5, 26, 10, 10, 0),
        )
        == 600
    )


# ---------------------------------------------------------------------------
# Event dispatch state machine (upsert patched out)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_open_then_close_settles_once():
    c = ResourceUsageCollector()
    with patch.object(c, "_settle_locked", new=AsyncMock()) as settle:
        await c._handle_event(
            make_event(
                occurred_at=datetime(2026, 5, 26, 10, 0, 5),
                event_type=EVENT_TYPE_PHASE_TO_METERED,
            )
        )
        assert 1 in c._open  # window opened
        await c._handle_event(
            make_event(
                occurred_at=datetime(2026, 5, 26, 12, 0, 0),
                event_type=EVENT_TYPE_PHASE_LEFT_METERED,
            )
        )
        assert 1 not in c._open  # window closed
        settle.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_duplicate_close_is_idempotent():
    c = ResourceUsageCollector()
    with patch.object(c, "_settle_locked", new=AsyncMock()) as settle:
        await c._handle_event(
            make_event(
                occurred_at=datetime(2026, 5, 26, 10, 0, 5),
                event_type=EVENT_TYPE_PHASE_TO_METERED,
            )
        )
        await c._handle_event(
            make_event(
                occurred_at=datetime(2026, 5, 26, 12, 0, 0),
                event_type=EVENT_TYPE_DELETED,
            )
        )
        # second close has no tracked window → no extra settle
        await c._handle_event(
            make_event(
                occurred_at=datetime(2026, 5, 26, 12, 0, 1),
                event_type=EVENT_TYPE_PHASE_LEFT_METERED,
            )
        )
        settle.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_created_does_not_open_window():
    c = ResourceUsageCollector()
    with patch.object(c, "_settle_locked", new=AsyncMock()):
        await c._handle_event(
            make_event(
                occurred_at=datetime(2026, 5, 26, 10, 0, 0),
                event_type=EVENT_TYPE_CREATED,
            )
        )
        # only phase_to_metered opens a window; created is recorded by the
        # logger separately and does not by itself start metering here.
        assert c._open == {}


# ---------------------------------------------------------------------------
# Startup reconcile — rebuild open windows from resource_events
# ---------------------------------------------------------------------------

NOW = datetime(2026, 5, 29, 12, 0, 0)


def _ue(**kw):
    base = dict(
        occurred_at=kw.pop("occurred_at"),
        owner_principal_id=5,
        creator_id=3,
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        resource_id=1,
        resource_name="g1",
        event_type=kw.pop("event_type"),
        phase=kw.pop("phase", "Ready"),
        spec_snapshot={
            "name": "g1",
            "spec": {
                "type": "gpustack-nvidia-geforce-rtx-5090-d-xpw9t",
                "resources": {"cpu": "2", "ram": "2Gi", "accelerator": "1"},
            },
        },
        created_at=NOW,
        updated_at=NOW,
    )
    base.update(kw)
    return ResourceEvent(**base)


@pytest_asyncio.fixture
async def events_session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(ResourceEvent.__table__.create)
    async with AsyncSession(engine) as s:
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_seeds_open_window(events_session):
    # latest event is phase_to_metered with no later close → window open
    events_session.add_all(
        [
            _ue(
                occurred_at=datetime(2026, 5, 29, 10, 0, 0),
                event_type=EVENT_TYPE_CREATED,
                phase=None,
            ),
            _ue(
                occurred_at=datetime(2026, 5, 29, 10, 0, 5),
                event_type=EVENT_TYPE_PHASE_TO_METERED,
            ),
        ]
    )
    await events_session.commit()

    c = ResourceUsageCollector()
    with patch.object(rc, "async_session", lambda: _yield(events_session)):
        await c._reconcile_open_windows()
    assert 1 in c._open
    w = c._open[1]
    assert w.gpu_count == 1
    assert w.sku == "gpustack-nvidia-geforce-rtx-5090-d-xpw9t"


@pytest.mark.asyncio
async def test_reconcile_skips_closed_window(events_session):
    # latest event is a close → no open window seeded
    events_session.add_all(
        [
            _ue(
                occurred_at=datetime(2026, 5, 29, 10, 0, 5),
                event_type=EVENT_TYPE_PHASE_TO_METERED,
            ),
            _ue(
                occurred_at=datetime(2026, 5, 29, 11, 0, 0),
                event_type=EVENT_TYPE_DELETED,
            ),
        ]
    )
    await events_session.commit()

    c = ResourceUsageCollector()
    with patch.object(rc, "async_session", lambda: _yield(events_session)):
        await c._reconcile_open_windows()
    assert c._open == {}


@pytest.mark.asyncio
async def test_reconcile_remetered_window_is_open(events_session):
    # stop→start within history: the latest event (by id) is a re-meter, so the
    # window must be open again — guards the MAX(id)-per-resource reduction.
    events_session.add_all(
        [
            _ue(
                occurred_at=datetime(2026, 5, 29, 10, 0, 0),
                event_type=EVENT_TYPE_PHASE_TO_METERED,
            ),
            _ue(
                occurred_at=datetime(2026, 5, 29, 10, 30, 0),
                event_type=EVENT_TYPE_PHASE_LEFT_METERED,
            ),
            _ue(
                occurred_at=datetime(2026, 5, 29, 11, 0, 0),
                event_type=EVENT_TYPE_PHASE_TO_METERED,
            ),
        ]
    )
    await events_session.commit()

    c = ResourceUsageCollector()
    with patch.object(rc, "async_session", lambda: _yield(events_session)):
        await c._reconcile_open_windows()
    assert 1 in c._open
    assert c._open[1].window_start == datetime(2026, 5, 29, 11, 0, 0)


@pytest_asyncio.fixture
async def events_and_metered_session():
    """A session backing both tables — reconcile rebuilds windows from
    ``resource_events`` then seeds ``settled_through`` from ``metered_usage``."""
    from gpustack.schemas.metered_usage import MeteredUsage

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(ResourceEvent.__table__.create)
        await conn.run_sync(MeteredUsage.__table__.create)
    async with AsyncSession(engine) as s:
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_seeds_settled_through_from_high_water_mark(
    events_and_metered_session,
):
    """On restart, a long-running instance resumes from MAX(settled_until)
    instead of re-iterating back to window_start."""
    from gpustack.schemas.metered_usage import (
        METER_INSTANCE_UPTIME,
        MeteredUsage,
    )

    s = events_and_metered_session
    s.add(
        _ue(
            occurred_at=datetime(2026, 5, 29, 10, 0, 5),
            event_type=EVENT_TYPE_PHASE_TO_METERED,
        )
    )
    # Two already-settled hourly buckets; the later high-water mark wins.
    hwm = datetime(2026, 5, 29, 11, 30, 0)
    s.add_all(
        [
            MeteredUsage(
                meter_key=METER_INSTANCE_UPTIME,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=1,
                resource_name="g1",
                bucket_start=datetime(2026, 5, 29, 10, 0, 0),
                quantity=3600,
                unit="seconds",
                settled_until=datetime(2026, 5, 29, 11, 0, 0),
            ),
            MeteredUsage(
                meter_key=METER_INSTANCE_UPTIME,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=1,
                resource_name="g1",
                bucket_start=datetime(2026, 5, 29, 11, 0, 0),
                quantity=1800,
                unit="seconds",
                settled_until=hwm,
            ),
        ]
    )
    await s.commit()

    c = ResourceUsageCollector()
    # Use a *closing* session CM: if the seeding ever escapes the reconcile's
    # `async with` block again, it would hit a closed session and the resume
    # below would silently regress to None — caught here, not swallowed.
    with patch.object(rc, "async_session", lambda: _closing_yield(s)):
        await c._reconcile_open_windows()

    assert 1 in c._open
    # window_start is the original event time; settled_through skips ahead to
    # the persisted high-water mark so the first settle resumes from there.
    assert c._open[1].window_start == datetime(2026, 5, 29, 10, 0, 5)
    assert c._open[1].settled_through == hwm


@asynccontextmanager
async def _yield(session):
    yield session


@asynccontextmanager
async def _closing_yield(session):
    """Like ``_yield`` but closes the session on exit — mirrors the real
    ``async_session`` so a seed/query that escaped the ``async with`` block
    (using a closed session) is caught by the test, not silently swallowed."""
    try:
        yield session
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Bucket sealing
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def metered_session():
    from gpustack.schemas.metered_usage import MeteredUsage

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(MeteredUsage.__table__.create)
    async with AsyncSession(engine) as s:
        yield s
    await engine.dispose()


def _mu_row(bucket_hour: int):
    from gpustack.schemas.metered_usage import (
        METER_INSTANCE_UPTIME,
        MeteredUsage,
    )

    return MeteredUsage(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        resource_id=1,
        resource_name="g1",
        bucket_start=datetime(2026, 6, 1, bucket_hour, 0, 0),
        quantity=3600,
        unit="seconds",
    )


@pytest.mark.asyncio
async def test_seal_due_seals_elapsed_only(metered_session):
    from sqlmodel import select

    from gpustack.schemas.metered_usage import (
        METER_INSTANCE_UPTIME,
        MeteredUsage,
    )

    now = datetime(2026, 6, 1, 12, 30, 0)
    metered_session.add(_mu_row(10))  # ended 11:00 — past +grace
    metered_session.add(_mu_row(12))  # current hour — still open
    await metered_session.commit()

    await MeteredUsage.seal_due(metered_session, METER_INSTANCE_UPTIME, now, 600)

    rows = (
        await metered_session.exec(
            select(MeteredUsage).order_by(MeteredUsage.bucket_start)
        )
    ).all()
    sealed = {r.bucket_start.hour: r.sealed_at is not None for r in rows}
    assert sealed == {10: True, 12: False}


@pytest.mark.asyncio
async def test_seal_due_within_grace_stays_open(metered_session):
    from sqlmodel import select

    from gpustack.schemas.metered_usage import (
        METER_INSTANCE_UPTIME,
        MeteredUsage,
    )

    # Hour 11 ended at 12:00; at 12:05 with a 10-min grace it is NOT yet sealable.
    now = datetime(2026, 6, 1, 12, 5, 0)
    metered_session.add(_mu_row(11))
    await metered_session.commit()

    await MeteredUsage.seal_due(metered_session, METER_INSTANCE_UPTIME, now, 600)

    row = (await metered_session.exec(select(MeteredUsage))).first()
    assert row.sealed_at is None
