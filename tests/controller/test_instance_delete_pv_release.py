"""Instance deletion soft-deletes a template PV that opted into release.

The deleting branch (``_reconcile_deleting``) tears the worker side down and,
once the CR is gone, calls ``_release_template_persistent_volume`` before
hard-deleting the
row. That helper soft-deletes (``phase=Deleting``) only a ``persistent_template``
volume with ``release_with_instance=True`` so the PV finalizer reclaims it; an
existing-PV reference, ``release_with_instance=False``, or no volume leaves the
PV untouched. ``_release_persistent_volume`` is idempotent on an already-Deleting
row.
"""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances.controllers import GPUInstanceController
from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstancePhase,
    GPUInstancePersistentVolumeReference,
    GPUInstancePersistentVolumeTemplate,
    GPUInstanceSpec,
    GPUInstanceStatus,
    GPUInstanceVolume,
)
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeSpec,
    GPUInstancePersistentVolumeStatus,
)


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        await conn.run_sync(GPUInstancePersistentVolume.__table__.create)
    yield e
    await e.dispose()


@pytest.fixture
def controller():
    return GPUInstanceController(SimpleNamespace(get_api_port=lambda: 80))


async def _seed_pv(engine, *, phase=None):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        status = GPUInstancePersistentVolumeStatus(phase=phase) if phase else None
        s.add(
            GPUInstancePersistentVolume(
                id=1,
                name="pv-1",
                owner_principal_id=1,
                persistent_volume_type_id=2,
                spec=GPUInstancePersistentVolumeSpec(type_="t"),
                status=status,
            )
        )
        await s.commit()


async def _get_pv(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await GPUInstancePersistentVolume.one_by_id(s, 1)


def _deleting_instance(volume):
    return GPUInstance(
        id=1,
        name="gi-1",
        owner_principal_id=1,
        cluster_id=10,
        spec=GPUInstanceSpec(type_="gpu", image="busybox", volume=volume),
        status=GPUInstanceStatus(phase=GPUInstancePhase.DELETING),
    )


def _template_volume(release_with_instance=True):
    return GPUInstanceVolume(
        persistent_template=GPUInstancePersistentVolumeTemplate(
            name="pv-1",
            spec=GPUInstancePersistentVolumeSpec(type_="t"),
            release_with_instance=release_with_instance,
        )
    )


@pytest.mark.asyncio
async def test_release_with_instance_soft_deletes_pv(engine, controller):
    await _seed_pv(engine)

    async with AsyncSession(engine, expire_on_commit=False) as s:
        await controller._release_template_persistent_volume(
            s, _deleting_instance(_template_volume(True))
        )

    pv = await _get_pv(engine)
    assert pv is not None  # soft delete, not hard delete
    assert pv.status.phase == GPUInstancePhase.DELETING


@pytest.mark.asyncio
async def test_release_with_instance_false_keeps_pv(engine, controller):
    await _seed_pv(engine)

    async with AsyncSession(engine, expire_on_commit=False) as s:
        await controller._release_template_persistent_volume(
            s, _deleting_instance(_template_volume(False))
        )

    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status is None  # untouched


@pytest.mark.asyncio
async def test_existing_pv_reference_is_not_released(engine, controller):
    await _seed_pv(engine)
    volume = GPUInstanceVolume(
        persistent=GPUInstancePersistentVolumeReference(name="pv-1")
    )

    async with AsyncSession(engine, expire_on_commit=False) as s:
        await controller._release_template_persistent_volume(
            s, _deleting_instance(volume)
        )

    pv = await _get_pv(engine)
    assert pv.status is None  # a referenced existing PV is never auto-released


@pytest.mark.asyncio
async def test_no_volume_is_noop(engine, controller):
    await _seed_pv(engine)

    async with AsyncSession(engine, expire_on_commit=False) as s:
        await controller._release_template_persistent_volume(
            s, _deleting_instance(None)
        )

    assert (await _get_pv(engine)).status is None


@pytest.mark.asyncio
async def test_release_is_idempotent_when_already_deleting(
    engine, controller, monkeypatch
):
    await _seed_pv(engine, phase=GPUInstancePhase.DELETING)

    calls = []
    original = GPUInstancePersistentVolume.update

    async def _counting_update(self, *a, **k):
        calls.append(1)
        return await original(self, *a, **k)

    monkeypatch.setattr(GPUInstancePersistentVolume, "update", _counting_update)

    async with AsyncSession(engine, expire_on_commit=False) as s:
        await controller._release_persistent_volume(
            s, owner_principal_id=1, name="pv-1"
        )

    assert calls == []  # already Deleting → no redundant write
