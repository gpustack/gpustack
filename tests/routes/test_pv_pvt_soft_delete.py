"""E3: PV/PVT DELETE is a soft delete.

The route stamps ``status.phase = Deleting`` and retains the row (returning
202) instead of hard-deleting it — the finalizer controller (E4) cleans up
the downstream CRs across clusters and then removes the row.
"""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.routes import gpu_instance_persistent_volumes as pv_routes
from gpustack.routes import gpu_instance_persistent_volume_types as pvt_routes
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeSpec,
    GPUInstancePersistentVolumeStatus,
)
from gpustack.schemas.gpu_instance_persistent_volume_types import (
    GPUInstancePersistentVolumeType,
    GPUInstancePersistentVolumeTypeSpec,
)
from gpustack.schemas.principals import PrincipalType

# Bypass tenant scoping: a SYSTEM principal passes assert_org_owned_writable.
CTX = SimpleNamespace(
    user=SimpleNamespace(kind=PrincipalType.SYSTEM, id=1),
    is_platform_admin=True,
    current_principal_id=None,
)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(GPUInstancePersistentVolumeType.__table__.create)
        await conn.run_sync(GPUInstancePersistentVolume.__table__.create)
    # Mirror the app session (gpustack.server.db): expire_on_commit=False is
    # required for async SQLAlchemy so a post-commit flush doesn't try to
    # lazily reload expired attributes synchronously.
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _add_pvt(session, id_=2, name="pvt-1", status=None):
    pvt = GPUInstancePersistentVolumeType(
        id=id_,
        name=name,
        owner_principal_id=1,
        spec=GPUInstancePersistentVolumeTypeSpec(),
        status=status,
    )
    session.add(pvt)
    await session.commit()
    return pvt


async def _add_pv(session, id_=1, name="pv-1", pvt_id=2, status=None):
    pv = GPUInstancePersistentVolume(
        id=id_,
        name=name,
        owner_principal_id=1,
        persistent_volume_type_id=pvt_id,
        spec=GPUInstancePersistentVolumeSpec(type_="pvt-1"),
        status=status,
    )
    session.add(pv)
    await session.commit()
    return pv


def _delete_route(router):
    for r in router.routes:
        if "DELETE" in getattr(r, "methods", set()):
            return r
    raise AssertionError("delete route not found")


@pytest.mark.asyncio
async def test_delete_pv_soft_deletes_and_retains_row(session):
    await _add_pvt(session)
    await _add_pv(session)

    ret = await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 1)

    assert ret.status is not None
    assert ret.status.phase == "Deleting"
    # Row is retained, not hard-deleted.
    row = await GPUInstancePersistentVolume.one_by_id(session=session, id=1)
    assert row is not None
    assert row.status.phase == "Deleting"


@pytest.mark.asyncio
async def test_delete_pv_clears_message_and_preserves_finalizing(session):
    await _add_pvt(session)
    await _add_pv(
        session,
        status=GPUInstancePersistentVolumeStatus(
            phase="Deleting", phase_message="waiting on cluster 3", finalizing=[3]
        ),
    )

    ret = await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 1)

    assert ret.status.phase == "Deleting"
    assert ret.status.phase_message is None  # reset, like _build_update_phase_source
    assert ret.status.finalizing == [3]  # preserved for the finalizer


@pytest.mark.asyncio
async def test_delete_pvt_soft_deletes_and_retains_row(session):
    await _add_pvt(session)

    ret = await pvt_routes.delete_gpu_instance_persistent_volume_type(session, CTX, 2)

    assert ret.status.phase == "Deleting"
    row = await GPUInstancePersistentVolumeType.one_by_id(session=session, id=2)
    assert row is not None
    assert row.status.phase == "Deleting"


@pytest.mark.asyncio
async def test_delete_pvt_allowed_while_referenced_by_pv(session):
    # In the finalizer model a PVT can be marked Deleting even while a PV
    # references it; the controller waits for the PV to clear (PV-before-PVT).
    await _add_pvt(session)
    await _add_pv(session)  # references pvt 2

    ret = await pvt_routes.delete_gpu_instance_persistent_volume_type(session, CTX, 2)

    assert ret.status.phase == "Deleting"


def test_delete_routes_return_202():
    assert _delete_route(pv_routes.router).status_code == 202
    assert _delete_route(pvt_routes.router).status_code == 202
