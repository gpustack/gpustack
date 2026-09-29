"""DELETE rejects a persistent volume whose holders are not all going away.

The route's soft delete (202 + ``phase = Deleting``) is only safe when every
holder's instance is itself already Deleting — that teardown is what releases
the reference. Otherwise the row would hang in ``Deleting`` forever, so the
route answers 409 up front, naming the holders and their phases; the finalizer
controller's blocked-reason stays as the backstop for the check-then-act race.
"""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import ConflictException, NotFoundException
from gpustack.routes import gpu_instance_persistent_volumes as pv_routes
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeSpec,
    GPUInstancePersistentVolumeStatus,
)
from gpustack.schemas.gpu_instances import GPUInstance
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
        await conn.run_sync(GPUInstancePersistentVolume.__table__.create)
        await conn.run_sync(GPUInstance.__table__.create)
    # Mirror the app session (gpustack.server.db): expire_on_commit=False is
    # required for async SQLAlchemy so a post-commit flush doesn't try to
    # lazily reload expired attributes synchronously.
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _add_pv(session, id_=1, name="pv-1", status=None):
    pv = GPUInstancePersistentVolume(
        id=id_,
        name=name,
        owner_principal_id=1,
        persistent_volume_type_id=2,
        spec=GPUInstancePersistentVolumeSpec(type_="pvt-1", capacity="100Gi"),
        status=status,
    )
    session.add(pv)
    await session.commit()
    return pv


async def _add_instance(session, *, id_, name, pv_id, phase=None):
    session.add(
        GPUInstance(
            id=id_,
            name=name,
            owner_principal_id=1,
            cluster_id=10,
            spec={"type_": "gpu", "image": "busybox"},
            persistent_volume_id=pv_id,
            status={"phase": phase} if phase is not None else None,
        )
    )
    await session.commit()


@pytest.mark.parametrize("holder_phase", ["Stopped", "Running", "Ready"])
@pytest.mark.asyncio
async def test_delete_attached_volume_rejected_with_409(session, holder_phase):
    await _add_pv(session)
    await _add_instance(session, id_=1, name="gi-1", pv_id=1, phase=holder_phase)

    with pytest.raises(ConflictException) as exc_info:
        await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 1)

    assert exc_info.value.status_code == 409
    message = exc_info.value.message
    assert "gi-1" in message
    assert holder_phase in message
    # "Detach" stays actionable: a Stopped instance can release the volume
    # through the instance update path.
    assert "Detach the volume or delete the instance" in message
    # Rejected before stamping: the row is untouched, not left in Deleting.
    row = await GPUInstancePersistentVolume.one_by_id(session=session, id=1)
    assert row.status is None


@pytest.mark.asyncio
async def test_delete_blocked_when_any_holder_is_not_deleting(session):
    """One Deleting holder releases its own reference; the live one still
    blocks, so the delete must not proceed."""
    await _add_pv(session)
    await _add_instance(session, id_=1, name="gi-del", pv_id=1, phase="Deleting")
    await _add_instance(session, id_=2, name="gi-live", pv_id=1, phase="Stopped")

    with pytest.raises(ConflictException) as exc_info:
        await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 1)

    assert "gi-live" in exc_info.value.message


@pytest.mark.asyncio
async def test_conflict_message_caps_holders_at_three(session):
    await _add_pv(session)
    for i in range(1, 5):
        await _add_instance(session, id_=i, name=f"gi-{i}", pv_id=1, phase="Stopped")

    with pytest.raises(ConflictException) as exc_info:
        await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 1)

    message = exc_info.value.message
    # Same cap as the controller's blocked-reason: three names, then the
    # overflow is stated rather than silently dropped.
    assert "gi-1 (Stopped), gi-2 (Stopped), gi-3 (Stopped), and others" in message
    assert "gi-4" not in message


@pytest.mark.asyncio
async def test_holder_with_unknown_phase_still_blocks(session):
    """An instance row without a resolved status cannot be proven to be
    releasing the volume, so it blocks like any other holder."""
    await _add_pv(session)
    await _add_instance(session, id_=1, name="gi-x", pv_id=1, phase=None)

    with pytest.raises(ConflictException) as exc_info:
        await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 1)

    assert "gi-x" in exc_info.value.message


@pytest.mark.asyncio
async def test_delete_proceeds_when_all_holders_are_deleting(session):
    await _add_pv(session)
    await _add_instance(session, id_=1, name="gi-del", pv_id=1, phase="Deleting")

    ret = await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 1)

    assert ret.status.phase == "Deleting"
    row = await GPUInstancePersistentVolume.one_by_id(session=session, id=1)
    assert row is not None  # soft delete retains the row for the finalizer
    assert row.status.phase == "Deleting"


@pytest.mark.asyncio
async def test_delete_unattached_volume_unchanged(session):
    await _add_pv(session)

    ret = await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 1)

    assert ret.status.phase == "Deleting"


@pytest.mark.asyncio
async def test_redelete_deleting_volume_stays_idempotent(session):
    await _add_pv(session, status=GPUInstancePersistentVolumeStatus(phase="Deleting"))

    ret = await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 1)

    assert ret.status.phase == "Deleting"


@pytest.mark.asyncio
async def test_delete_missing_volume_still_not_found(session):
    """``ensure_writable`` keeps running first: a missing volume is 404, not
    the new 409."""
    with pytest.raises(NotFoundException):
        await pv_routes.delete_gpu_instance_persistent_volume(session, CTX, 99)
