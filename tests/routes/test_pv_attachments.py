"""A persistent volume's mount relationship is exposed on the API.

A PV has an independent lifecycle: it is created separately and keeps being
metered while nothing is attached. That is correct, but it was also invisible —
a detached-but-billed volume looked identical to one in use, and a user had no
way to tell whether deleting it was safe. The list / detail responses now resolve
``attached_instances`` from ``GPUInstance.persistent_volume_id``.

The reference is NOT phase-filtered, so a Stopped instance holds the volume
exactly as a running one does (and blocks its reclaim). Every holder is reported
with its phase rather than being filtered down to "active" ones.
"""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.routes import gpu_instance_persistent_volumes as pv_routes
from gpustack.schemas.gpu_instances import GPUInstance
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumePublic,
    GPUInstancePersistentVolumeSpec,
)
from gpustack.schemas.principals import PrincipalType
from gpustack.server.bus import Event, EventType

CTX = SimpleNamespace(
    user=SimpleNamespace(kind=PrincipalType.SYSTEM, id=1),
    is_platform_admin=True,
    current_principal_id=None,
)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(GPUInstancePersistentVolume.__table__.create)
        await conn.run_sync(GPUInstance.__table__.create)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _add_pv(session, id_=1, name="pv-1"):
    pv = GPUInstancePersistentVolume(
        id=id_,
        name=name,
        owner_principal_id=1,
        persistent_volume_type_id=2,
        spec=GPUInstancePersistentVolumeSpec(type_="pvt-1", capacity="100Gi"),
    )
    session.add(pv)
    await session.commit()
    return pv


async def _add_instance(session, *, id_, name, pv_id, phase):
    session.add(
        GPUInstance(
            id=id_,
            name=name,
            owner_principal_id=1,
            cluster_id=10,
            spec={"type_": "gpu", "image": "busybox"},
            persistent_volume_id=pv_id,
            status={"phase": phase},
        )
    )
    await session.commit()


@pytest.mark.asyncio
async def test_detail_reports_every_holder_with_its_phase(session):
    await _add_pv(session)
    await _add_instance(session, id_=1, name="gi-ready", pv_id=1, phase="Ready")
    await _add_instance(session, id_=2, name="gi-stopped", pv_id=1, phase="Stopped")

    out = await pv_routes.get_gpu_instance_persistent_volume(session, CTX, 1)

    assert {(a.name, a.phase) for a in out.attached_instances} == {
        ("gi-ready", "Ready"),
        # A Stopped instance still holds the volume — not filtered out.
        ("gi-stopped", "Stopped"),
    }


@pytest.mark.asyncio
async def test_detached_volume_reports_an_empty_list_not_null(session):
    """The distinction carries meaning: ``[]`` = "confirmed nothing attached, so
    this volume is being billed while idle"; ``None`` = "not resolved"."""
    await _add_pv(session)

    out = await pv_routes.get_gpu_instance_persistent_volume(session, CTX, 1)

    assert out.attached_instances == []


@pytest.mark.asyncio
async def test_attachments_resolved_in_one_query_for_the_list(session, monkeypatch):
    await _add_pv(session, id_=1, name="pv-1")
    await _add_pv(session, id_=2, name="pv-2")
    await _add_instance(session, id_=1, name="gi-1", pv_id=1, phase="Ready")

    monkeypatch.setattr(pv_routes, "async_session", lambda: _NullCM(session))
    params = SimpleNamespace(watch=False, order_by=None, page=1, perPage=10)
    out = await pv_routes.get_gpu_instance_persistent_volumes(CTX, params)

    by_name = {pv.name: pv for pv in out.items}
    assert [a.name for a in by_name["pv-1"].attached_instances] == ["gi-1"]
    assert by_name["pv-2"].attached_instances == []


@pytest.mark.asyncio
async def test_watch_events_carry_the_same_field(session, monkeypatch):
    """Without the stream hook a UI in watch mode would blank the column on every
    update, so the watch payload has to match the REST response."""
    await _add_pv(session)
    await _add_instance(session, id_=1, name="gi-1", pv_id=1, phase="Ready")

    monkeypatch.setattr(pv_routes, "async_session", lambda: _NullCM(session))
    pv = await GPUInstancePersistentVolume.one_by_id(session, 1)
    event = Event(
        type=EventType.UPDATED,
        data=GPUInstancePersistentVolumePublic.model_validate(pv, from_attributes=True),
    )
    await pv_routes._inject_attachments_into_event(event)

    assert [a.name for a in event.data.attached_instances] == ["gi-1"]


@pytest.mark.asyncio
async def test_delete_event_is_left_alone(session, monkeypatch):
    """DELETED events arrive id-only — there is nothing to enrich, and touching
    them would be a lookup per teardown."""
    monkeypatch.setattr(pv_routes, "async_session", lambda: _NullCM(session))
    event = Event(type=EventType.DELETED, data={"id": 1})
    await pv_routes._inject_attachments_into_event(event)
    assert event.data == {"id": 1}


class _NullCM:
    """Hands out the test session without closing it on exit."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *a):
        return False
