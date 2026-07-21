"""GPU instance update route tests: ``_build_update_source`` gating + volume.

``display_name`` / ``description`` are metadata and editable from any phase;
``spec`` is a full replacement editable only while Stopped, and a volume change
re-resolves the ``persistent_volume_id`` FK (an unchanged volume keeps it).
Exercised directly over a real in-memory sqlite DB with a fake ``ctx``.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import InvalidException
from gpustack.routes import gpu_instances as routes
from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstanceCreate,
    GPUInstanceEphemeralVolume,
    GPUInstancePersistentVolumeReference,
    GPUInstancePhase,
    GPUInstancePublic,
    GPUInstanceSpec,
    GPUInstanceSSHPublicKeyReference,
    GPUInstanceStatus,
    GPUInstanceUpdate,
    GPUInstanceVolume,
)
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeSpec,
    GPUInstancePersistentVolumeStatus,
)
from gpustack.schemas.gpu_instance_types import (
    GPUInstanceType,
    GPUInstanceTypeSpec,
)

NAMESPACE = "gpustack-user-1"
CTX = SimpleNamespace(user=SimpleNamespace(id=1))


def _ephemeral_spec(image="busybox", type_="gpu"):
    return GPUInstanceSpec(
        type_=type_,
        image=image,
        volume=GPUInstanceVolume(ephemeral=GPUInstanceEphemeralVolume()),
    )


def _persistent_spec(name="pv-1", image="busybox"):
    return GPUInstanceSpec(
        type_="gpu",
        image=image,
        volume=GPUInstanceVolume(
            persistent=GPUInstancePersistentVolumeReference(name=name)
        ),
    )


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        await conn.run_sync(GPUInstance.__table__.create)
        await conn.run_sync(GPUInstancePersistentVolume.__table__.create)
        await conn.run_sync(GPUInstanceType.__table__.create)
    yield e
    await e.dispose()


async def _seed_type(
    engine, *, cluster_id=2, name="gpu", snapshot="sha1:new", deleted=False
):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=cluster_id,
                name=name,
                spec=GPUInstanceTypeSpec(),
                snapshot=snapshot,
                deleted_at=datetime(2020, 1, 1) if deleted else None,
            )
        )
        await s.commit()


async def _seed(
    engine, *, phase, spec=None, persistent_volume_id=None, type_snapshot=None
):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstance(
                id=1,
                name="gi-1",
                owner_principal_id=1,
                cluster_id=2,
                spec=spec or _ephemeral_spec(),
                status=GPUInstanceStatus(phase=phase, namespace=NAMESPACE),
                persistent_volume_id=persistent_volume_id,
                type_snapshot=type_snapshot,
            )
        )
        await s.commit()


async def _row(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await GPUInstance.one_by_id(s, 1)


async def _build(engine, update, row):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await routes._build_update_source(s, CTX, update, row)


# --- phase gating ---------------------------------------------------------- #


@pytest.mark.asyncio
async def test_metadata_editable_from_any_phase(engine):
    await _seed(engine, phase=GPUInstancePhase.READY)
    row = await _row(engine)

    source = await _build(
        engine, GPUInstanceUpdate(display_name="dn", description="d"), row
    )

    assert source == {"display_name": "dn", "description": "d"}


@pytest.mark.asyncio
async def test_non_ssh_spec_edit_rejected_when_not_stopped(engine):
    # A field other than sshPublicKeys (here the image) changed while Ready.
    await _seed(engine, phase=GPUInstancePhase.READY)
    row = await _row(engine)

    with pytest.raises(InvalidException):
        await _build(engine, GPUInstanceUpdate(spec=_ephemeral_spec("new")), row)


@pytest.mark.asyncio
async def test_non_ssh_spec_edit_rejected_before_first_phase(engine):
    # Pre-create (phase is None) is not Stopped either — non-ssh edits rejected.
    await _seed(engine, phase=None)
    row = await _row(engine)

    with pytest.raises(InvalidException):
        await _build(engine, GPUInstanceUpdate(spec=_ephemeral_spec("new")), row)


@pytest.mark.asyncio
async def test_ssh_only_edit_allowed_while_running(engine):
    # sshPublicKeys is the one field editable outside Stopped: an edit whose
    # only diff is the key list is accepted while the instance is Ready.
    spec = _ephemeral_spec()
    await _seed(engine, phase=GPUInstancePhase.READY, spec=spec)
    row = await _row(engine)

    new_spec = spec.model_copy(
        update={"ssh_public_keys": [GPUInstanceSSHPublicKeyReference(name="k1")]}
    )
    source = await _build(engine, GPUInstanceUpdate(spec=new_spec), row)

    assert source["spec"].ssh_public_keys[0].name == "k1"
    # An ssh-only edit never re-points the volume FK.
    assert "persistent_volume_id" not in source


# --- spec edit while stopped ----------------------------------------------- #


@pytest.mark.asyncio
async def test_spec_edit_stopped_unchanged_volume_keeps_fk(engine):
    await _seed(
        engine,
        phase=GPUInstancePhase.STOPPED,
        spec=_persistent_spec(),
        persistent_volume_id=7,
    )
    row = await _row(engine)

    new_spec = _persistent_spec(image="busybox:2")  # same volume, new image
    source = await _build(engine, GPUInstanceUpdate(spec=new_spec), row)

    assert source["spec"].image == "busybox:2"
    assert source["persistent_volume_id"] == 7  # FK preserved, not re-resolved


@pytest.mark.asyncio
async def test_spec_edit_stopped_swap_to_ephemeral_clears_fk(engine):
    await _seed(
        engine,
        phase=GPUInstancePhase.STOPPED,
        spec=_persistent_spec(),
        persistent_volume_id=7,
    )
    row = await _row(engine)

    source = await _build(engine, GPUInstanceUpdate(spec=_ephemeral_spec()), row)

    assert source["persistent_volume_id"] is None


@pytest.mark.asyncio
async def test_spec_edit_stopped_swap_to_existing_pv_resolves_fk(engine):
    await _seed(engine, phase=GPUInstancePhase.STOPPED)  # ephemeral, FK None
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstancePersistentVolume(
                id=5,
                name="pv-2",
                owner_principal_id=1,
                persistent_volume_type_id=1,
                spec=GPUInstancePersistentVolumeSpec(type_="pvt"),
                status=GPUInstancePersistentVolumeStatus(phase="Ready"),
            )
        )
        await s.commit()
    row = await _row(engine)

    source = await _build(
        engine, GPUInstanceUpdate(spec=_persistent_spec(name="pv-2")), row
    )

    assert source["persistent_volume_id"] == 5


@pytest.mark.asyncio
async def test_spec_edit_stopped_swap_to_missing_pv_rejected(engine):
    await _seed(engine, phase=GPUInstancePhase.STOPPED)
    row = await _row(engine)

    with pytest.raises(InvalidException):
        await _build(engine, GPUInstanceUpdate(spec=_persistent_spec(name="nope")), row)


# --- type_snapshot column -------------------------------------------------- #


@pytest.mark.asyncio
async def test_type_snapshot_column_round_trips(engine):
    await _seed(engine, phase=GPUInstancePhase.READY, type_snapshot="sha1:abc123")
    row = await _row(engine)

    assert row.type_snapshot == "sha1:abc123"


@pytest.mark.asyncio
async def test_type_snapshot_surfaced_on_public(engine):
    await _seed(engine, phase=GPUInstancePhase.READY, type_snapshot="sha1:abc123")
    row = await _row(engine)

    public = GPUInstancePublic.model_validate(row, from_attributes=True)

    assert public.type_snapshot == "sha1:abc123"


def test_type_snapshot_is_not_a_client_input():
    # Server-stamped only: the create/update DTOs must not bind it, so a client
    # can never set it; it is exposed read-only on the public view + table.
    assert "type_snapshot" not in GPUInstanceCreate.model_fields
    assert "type_snapshot" not in GPUInstanceUpdate.model_fields
    assert "type_snapshot" in GPUInstancePublic.model_fields
    assert "type_snapshot" in GPUInstance.model_fields


# --- resolve & stamp on create/update -------------------------------------- #


async def _resolve(engine, *, cluster_id, type_name):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await routes._resolve_type_snapshot(
            s, cluster_id=cluster_id, type_name=type_name
        )


@pytest.mark.asyncio
async def test_resolve_type_snapshot_hit(engine):
    await _seed_type(engine, cluster_id=2, name="gpu", snapshot="sha1:hit")

    assert await _resolve(engine, cluster_id=2, type_name="gpu") == "sha1:hit"


@pytest.mark.asyncio
async def test_resolve_type_snapshot_miss_rejected(engine):
    with pytest.raises(InvalidException):
        await _resolve(engine, cluster_id=2, type_name="absent")


@pytest.mark.asyncio
async def test_resolve_type_snapshot_soft_deleted_rejected(engine):
    await _seed_type(engine, cluster_id=2, name="gpu", deleted=True)

    with pytest.raises(InvalidException):
        await _resolve(engine, cluster_id=2, type_name="gpu")


def test_build_create_source_stamps_type_snapshot():
    create_obj = GPUInstanceCreate(name="gi-1", spec=_ephemeral_spec(), cluster_id=2)

    source = routes._build_create_source(create_obj, 1, None, "sha1:stamped")

    assert source["type_snapshot"] == "sha1:stamped"


@pytest.mark.asyncio
async def test_create_without_cluster_id_rejected():
    # cluster_id is required to resolve the instance type; omitting it must be a
    # clear client error, not a confusing "not found in cluster None" downstream.
    create_obj = GPUInstanceCreate(name="gi-1", spec=_ephemeral_spec())  # no cluster_id

    with pytest.raises(InvalidException):
        await routes.create_gpu_instance(session=None, ctx=None, create_obj=create_obj)


@pytest.mark.asyncio
async def test_update_stopped_type_change_restamps_type_snapshot(engine):
    # Re-stamp fires only when spec.type_ changes; here gpu -> other.
    await _seed(engine, phase=GPUInstancePhase.STOPPED, type_snapshot="sha1:old")
    await _seed_type(engine, cluster_id=2, name="other", snapshot="sha1:fresh")
    row = await _row(engine)

    source = await _build(
        engine, GPUInstanceUpdate(spec=_ephemeral_spec(type_="other")), row
    )

    assert source["type_snapshot"] == "sha1:fresh"


@pytest.mark.asyncio
async def test_update_stopped_type_change_missing_type_rejected(engine):
    # Changing to a type with no active row is rejected, same as at create.
    await _seed(engine, phase=GPUInstancePhase.STOPPED, type_snapshot="sha1:old")
    row = await _row(engine)  # no "ghost" type seeded

    with pytest.raises(InvalidException):
        await _build(
            engine, GPUInstanceUpdate(spec=_ephemeral_spec(type_="ghost")), row
        )


@pytest.mark.asyncio
async def test_update_stopped_unrelated_edit_keeps_snapshot(engine):
    # Editing a non-type field (image) while stopped keeps the original snapshot
    # and never re-resolves the type — so it does not fail even with no type row.
    await _seed(engine, phase=GPUInstancePhase.STOPPED, type_snapshot="sha1:old")
    row = await _row(engine)  # no type seeded on purpose

    source = await _build(engine, GPUInstanceUpdate(spec=_ephemeral_spec("new")), row)

    assert "type_snapshot" not in source  # original snapshot preserved


@pytest.mark.asyncio
async def test_update_ssh_only_does_not_restamp(engine):
    spec = _ephemeral_spec()
    await _seed(
        engine, phase=GPUInstancePhase.READY, spec=spec, type_snapshot="sha1:old"
    )
    row = await _row(engine)

    new_spec = spec.model_copy(
        update={"ssh_public_keys": [GPUInstanceSSHPublicKeyReference(name="k1")]}
    )
    source = await _build(engine, GPUInstanceUpdate(spec=new_spec), row)

    # SSH-only edit keeps the type unchanged, so it is not re-resolved/re-stamped.
    assert "type_snapshot" not in source
