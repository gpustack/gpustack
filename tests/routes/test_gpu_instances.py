"""GPU instance update route tests: ``_build_update_source`` gating + volume.

``display_name`` / ``description`` are metadata and editable from any phase;
``spec`` is a full replacement editable only while Stopped, and a volume change
re-resolves the ``persistent_volume_id`` FK (an unchanged volume keeps it).
Exercised directly over a real in-memory sqlite DB with a fake ``ctx``.
"""

from datetime import datetime
from types import SimpleNamespace
import re

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import ConflictException, InvalidException
from gpustack.routes import gpu_instances as routes
from gpustack.schemas.clusters import GpuInstanceOptions, K8sOptions
from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstanceCreate,
    GPUInstanceEphemeralVolume,
    GPUInstancePersistentVolumeReference,
    GPUInstancePhase,
    GPUInstancePort,
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
    GPUInstanceTypeUnitResources,
)
from gpustack.server.bus import EventType

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
async def test_spec_edit_stopped_swap_leaves_the_old_pv_intact(engine):
    """V4: a volume swap re-points the FK and nothing else.

    The previously bound PV is deliberately NOT released — releasing it would
    destroy user data — so it stays a live row with an unchanged phase and keeps
    being metered (`storage.capacity` runs from created to deleted, see
    tests/server/test_pv_metering_lifecycle.py). It has to be deleted explicitly
    via the PV endpoint, which is why the storage list needs the "attached
    instances" column to make an idle-but-billed volume visible.
    """
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for pv_id, pv_name in ((4, "pv-1"), (5, "pv-2")):
            s.add(
                GPUInstancePersistentVolume(
                    id=pv_id,
                    name=pv_name,
                    owner_principal_id=1,
                    persistent_volume_type_id=1,
                    spec=GPUInstancePersistentVolumeSpec(type_="pvt"),
                    status=GPUInstancePersistentVolumeStatus(phase="Ready"),
                )
            )
        await s.commit()
    await _seed(
        engine,
        phase=GPUInstancePhase.STOPPED,
        spec=_persistent_spec(name="pv-1"),
        persistent_volume_id=4,
    )
    row = await _row(engine)

    source = await _build(
        engine, GPUInstanceUpdate(spec=_persistent_spec(name="pv-2")), row
    )

    assert source["persistent_volume_id"] == 5
    async with AsyncSession(engine, expire_on_commit=False) as s:
        old = await GPUInstancePersistentVolume.one_by_id(s, 4)
    assert old is not None, "the swapped-out volume must not be deleted"
    assert old.status.phase == "Ready", "nor marked for deletion"


@pytest.mark.asyncio
async def test_spec_edit_stopped_swap_to_missing_pv_rejected(engine):
    await _seed(engine, phase=GPUInstancePhase.STOPPED)
    row = await _row(engine)

    with pytest.raises(InvalidException):
        await _build(engine, GPUInstanceUpdate(spec=_persistent_spec(name="nope")), row)


@pytest.mark.asyncio
async def test_update_stopped_resolves_generated_token(engine):
    # A spec replacement may carry {{generated_token}} (e.g. re-applying a
    # template); it must resolve to a concrete value just like at create.
    await _seed(engine, phase=GPUInstancePhase.STOPPED)
    row = await _row(engine)

    new_spec = _ephemeral_spec()
    new_spec.command = ["jupyter", "lab", "--ServerApp.token={{generated_token}}"]
    new_spec.ports = [
        GPUInstancePort(
            name="JUPYTER", port=8888, access_params={"token": "{{generated_token}}"}
        )
    ]
    source = await _build(engine, GPUInstanceUpdate(spec=new_spec), row)

    token = source["spec"].command[2].split("=", 1)[1]
    assert re.fullmatch(r"[0-9a-f]{32}", token)
    assert source["spec"].ports[0].access_params["token"] == token


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


def test_build_create_source_resolves_generated_token():
    # The persisted spec carries the concrete token, so stop/start replays and
    # the UI's accessParams link building always see the same value.
    spec = _ephemeral_spec()
    spec.command = ["jupyter", "lab", "--ServerApp.token={{generated_token}}"]
    spec.ports = [
        GPUInstancePort(
            name="JUPYTER", port=8888, access_params={"token": "{{generated_token}}"}
        )
    ]
    create_obj = GPUInstanceCreate(name="gi-1", spec=spec, cluster_id=2)

    source = routes._build_create_source(create_obj, 1, None, "sha1:stamped")

    token = source["spec"]["command"][2].split("=", 1)[1]
    assert re.fullmatch(r"[0-9a-f]{32}", token)
    assert source["spec"]["ports"][0]["access_params"]["token"] == token


@pytest.mark.asyncio
async def test_create_without_cluster_id_rejected():
    # cluster_id is required to resolve the instance type; omitting it must be a
    # clear client error, not a confusing "not found in cluster None" downstream.
    create_obj = GPUInstanceCreate(name="gi-1", spec=_ephemeral_spec())  # no cluster_id

    with pytest.raises(InvalidException):
        await routes.create_gpu_instance(session=None, ctx=None, create_obj=create_obj)


def _patch_create_preflight(monkeypatch, *, gpu_service):
    """Stub everything ahead of the purpose guard in ``create_gpu_instance``.

    Leaves exactly one decision under test: whether a cluster's purpose lets a
    GPU Instance be created on it. ``_validate_create_obj`` is the first step
    *after* the guard, so patching it to raise proves the guard runs before any
    of the payload is resolved against the cluster.
    """
    cluster = SimpleNamespace(
        id=2,
        name="cluster-2",
        deleted_at=None,
        k8s_options=K8sOptions(
            gpu_instance_options=GpuInstanceOptions() if gpu_service else None
        ),
    )

    async def fake_one_by_id(session, id=None, *args, **kwargs):
        return cluster

    monkeypatch.setattr(routes.Cluster, "one_by_id", fake_one_by_id)
    monkeypatch.setattr(routes, "assert_cluster_visible", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "validate_owner_principal", lambda *a, **kw: None)

    async def past_the_guard(*a, **kw):
        raise RuntimeError("reached the payload validation")

    monkeypatch.setattr(routes, "_validate_create_obj", past_the_guard)


@pytest.mark.asyncio
async def test_create_on_a_model_service_cluster_rejected(monkeypatch):
    # A GPU Instance is a workload for GPU Service capacity. A Model Service
    # cluster has none, so create must refuse it with a 409 that says why —
    # before resolving any of the payload against the cluster.
    _patch_create_preflight(monkeypatch, gpu_service=False)
    create_obj = GPUInstanceCreate(
        name="gi-1", spec=_ephemeral_spec(), cluster_id=2, owner_principal_id=1
    )

    with pytest.raises(ConflictException) as excinfo:
        await routes.create_gpu_instance(session=None, ctx=CTX, create_obj=create_obj)

    assert excinfo.value.status_code == 409
    assert "cluster-2" in excinfo.value.message
    assert "model service" in excinfo.value.message


@pytest.mark.asyncio
async def test_create_on_a_gpu_service_cluster_passes_the_purpose_guard(monkeypatch):
    # The negative case above must fail for the right reason: on a GPU Service
    # cluster the same call gets past the guard and on to payload validation.
    _patch_create_preflight(monkeypatch, gpu_service=True)
    create_obj = GPUInstanceCreate(
        name="gi-1", spec=_ephemeral_spec(), cluster_id=2, owner_principal_id=1
    )

    with pytest.raises(RuntimeError, match="reached the payload validation"):
        await routes.create_gpu_instance(session=None, ctx=CTX, create_obj=create_obj)


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


# --- resolved type summary on the public payload ---------------------------- #
#
# The GPU Instances page renders the instance type from the summary carried by
# ``GPUInstancePublic.type_snapshot_detail`` (joined from the stamped
# ``type_snapshot``), never from ``description`` — so the payload contract is:
# however ``description`` is set (empty, cleared by a later PUT), the summary
# must resolve as long as the stamped type row exists.


async def _seed_accelerated_type(engine, *, snapshot, deleted=False):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=2,
                name="a10g",
                spec=GPUInstanceTypeSpec(
                    display_name="A10G Pool",
                    acceleratable=True,
                    unit_resources=GPUInstanceTypeUnitResources(
                        cpu="7250m",
                        ram="26214Mi",
                    ),
                ),
                snapshot=snapshot,
                deleted_at=datetime(2020, 1, 1) if deleted else None,
            )
        )
        await s.commit()


async def _to_public(engine, row):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await routes._to_public_with_type(s, row)


@pytest.mark.asyncio
async def test_type_snapshot_detail_resolved_from_the_stamped_row(engine):
    await _seed_accelerated_type(engine, snapshot="sha1:a10g")
    await _seed(engine, phase=GPUInstancePhase.READY, type_snapshot="sha1:a10g")
    row = await _row(engine)

    public = await _to_public(engine, row)

    detail = public.type_snapshot_detail
    assert detail is not None
    assert detail.name == "a10g"
    assert detail.spec.display_name == "A10G Pool"
    assert detail.spec.acceleratable is True
    assert detail.spec.unit_resources.cpu == "7250m"
    assert detail.spec.unit_resources.ram == "26214Mi"


@pytest.mark.asyncio
async def test_type_snapshot_detail_survives_a_cleared_description(engine):
    # The reported bug: an instance whose description is empty (or cleared by a
    # later PUT) rendered as CPU-only because the UI parsed the type out of the
    # description. The payload contract is that the resolved summary — not
    # description — carries everything the type rendering needs.
    await _seed_accelerated_type(engine, snapshot="sha1:a10g")
    await _seed(
        engine,
        phase=GPUInstancePhase.READY,
        type_snapshot="sha1:a10g",
    )
    row = await _row(engine)
    assert row.description is None

    public = await _to_public(engine, row)

    assert public.description is None
    assert public.type_snapshot == "sha1:a10g"
    assert public.type_snapshot_detail is not None
    assert public.type_snapshot_detail.spec.acceleratable is True
    assert public.type_snapshot_detail.spec.display_name == "A10G Pool"


@pytest.mark.asyncio
async def test_type_snapshot_detail_absent_without_a_snapshot(engine):
    await _seed(engine, phase=GPUInstancePhase.READY, type_snapshot=None)
    row = await _row(engine)

    public = await _to_public(engine, row)

    assert public.type_snapshot_detail is None


@pytest.mark.asyncio
async def test_type_snapshot_detail_absent_when_the_type_row_is_gone(engine):
    # A dangling snapshot degrades to None (no crash): the client falls back to
    # its no-type rendering.
    await _seed(engine, phase=GPUInstancePhase.READY, type_snapshot="sha1:ghost")
    row = await _row(engine)

    public = await _to_public(engine, row)

    assert public.type_snapshot_detail is None


@pytest.mark.asyncio
async def test_type_snapshot_detail_resolves_a_soft_deleted_type(engine):
    # The instance legitimately points at the definition it was created
    # against, so a soft-deleted type still resolves for display.
    await _seed_accelerated_type(engine, snapshot="sha1:a10g", deleted=True)
    await _seed(engine, phase=GPUInstancePhase.READY, type_snapshot="sha1:a10g")
    row = await _row(engine)

    public = await _to_public(engine, row)

    assert public.type_snapshot_detail is not None
    assert public.type_snapshot_detail.name == "a10g"


def test_type_snapshot_detail_is_not_a_client_input():
    # Resolved server-side on every read; the create/update DTOs must not bind
    # it, mirroring type_snapshot.
    assert "type_snapshot_detail" not in GPUInstanceCreate.model_fields
    assert "type_snapshot_detail" not in GPUInstanceUpdate.model_fields
    assert "type_snapshot_detail" in GPUInstancePublic.model_fields


def _patch_sessions(monkeypatch, engine):
    def fake_async_session():
        return AsyncSession(engine, expire_on_commit=False)

    monkeypatch.setattr(routes, "async_session", fake_async_session)


@pytest.mark.asyncio
async def test_watch_event_carries_the_resolved_summary(monkeypatch, engine):
    # The list page streams updates in watch mode and replaces the row with the
    # event payload, so the stream must carry the summary or a watch update
    # would blank the type rendering.
    await _seed_accelerated_type(engine, snapshot="sha1:a10g")
    await _seed(engine, phase=GPUInstancePhase.READY, type_snapshot="sha1:a10g")
    row = await _row(engine)
    _patch_sessions(monkeypatch, engine)

    event = SimpleNamespace(
        type=EventType.UPDATED,
        data=GPUInstancePublic.model_validate(row, from_attributes=True),
        id=1,
    )
    await routes._inject_type_snapshot_detail_into_event(event)

    assert event.data.type_snapshot_detail is not None
    assert event.data.type_snapshot_detail.name == "a10g"


@pytest.mark.asyncio
async def test_watch_event_skips_deleted_and_snapshotless_rows(monkeypatch, engine):
    # DELETED events carry id-only payloads and snapshotless rows have nothing
    # to resolve — both must pass through untouched.
    await _seed(engine, phase=GPUInstancePhase.READY, type_snapshot=None)
    row = await _row(engine)
    _patch_sessions(monkeypatch, engine)

    deleted = SimpleNamespace(type=EventType.DELETED, data={"id": 1}, id=2)
    await routes._inject_type_snapshot_detail_into_event(deleted)
    assert deleted.data == {"id": 1}

    updated = SimpleNamespace(
        type=EventType.UPDATED,
        data=GPUInstancePublic.model_validate(row, from_attributes=True),
        id=3,
    )
    await routes._inject_type_snapshot_detail_into_event(updated)
    assert updated.data.type_snapshot_detail is None


@pytest.mark.asyncio
async def test_types_by_snapshot_batches_a_page_into_one_lookup(engine):
    # The list route resolves a whole page's types in one query: rows sharing a
    # type deduplicate onto one key, and a snapshotless row contributes none.
    await _seed_accelerated_type(engine, snapshot="sha1:a10g")
    await _seed(engine, phase=GPUInstancePhase.READY, type_snapshot="sha1:a10g")
    row = await _row(engine)
    snapshotless = row.model_copy(update={"id": 2, "type_snapshot": None})

    async with AsyncSession(engine, expire_on_commit=False) as s:
        rows = await routes._types_by_snapshot(s, [row, row, snapshotless])

    assert list(rows) == ["sha1:a10g"]
    assert rows["sha1:a10g"].name == "a10g"
