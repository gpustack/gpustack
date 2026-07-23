"""GPUInstanceTypeController: watch-event mapping + sqlite catalog projection.

The controller has no DB bus source — the operator ``watch_instance_types``
stream is authoritative. ``_on_event`` maps each ADDED/MODIFIED/DELETED line
onto a ``WorkEvent`` keyed by the stable ``(cluster_id, name)`` identity (the
raw object is carried so the reconcile needs no second fetch); DELETED gets the
queue's default coalesce priority. ``_reconcile`` upserts the row on
ADDED/MODIFIED (reviving a soft-deleted one) and soft-deletes it on DELETED,
over a real in-memory sqlite DB.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances.controllers import GPUInstanceTypeController
from gpustack.schemas.gpu_instance_types import (
    GPUInstanceType,
    GPUInstanceTypeSpec,
)
from gpustack.server.workqueue import WorkEvent, WorkEventType


def _line(etype, *, cluster="1", name="a10g", spec=None):
    obj = {"cluster": cluster, "metadata": {"name": name}, "spec": spec or {}}
    return json.dumps({"type": etype, "object": obj})


def _pending(controller, keys):
    return controller._queue._pending[keys]


# --- event mapping (no DB) ------------------------------------------------- #


def test_added_maps_to_added_keyed_on_cluster_and_name():
    controller = GPUInstanceTypeController()

    controller._on_event(_line("ADDED", cluster="2", name="a100"))

    event = _pending(controller, (2, "a100"))
    assert event.type == WorkEventType.ADDED
    assert event.object["metadata"]["name"] == "a100"  # raw object carried


def test_modified_maps_to_modified():
    controller = GPUInstanceTypeController()

    controller._on_event(_line("MODIFIED"))

    assert _pending(controller, (1, "a10g")).type == WorkEventType.MODIFIED


def test_deleted_maps_to_deleted():
    controller = GPUInstanceTypeController()

    controller._on_event(_line("DELETED"))

    assert _pending(controller, (1, "a10g")).type == WorkEventType.DELETED


def test_deleted_coalesces_over_pending_modified():
    controller = GPUInstanceTypeController()

    controller._on_event(_line("MODIFIED"))
    controller._on_event(_line("DELETED"))

    # Latest-wins: the later DELETED replaces the earlier MODIFIED in the slot.
    assert _pending(controller, (1, "a10g")).type == WorkEventType.DELETED


def test_added_after_pending_deleted_wins():
    # A catalog DELETED is NOT terminal: a later ADDED (recreate) for the same
    # key must win, not be discarded by DELETED stickiness (latest-wins policy).
    controller = GPUInstanceTypeController()

    controller._on_event(_line("DELETED"))
    controller._on_event(_line("ADDED"))

    assert _pending(controller, (1, "a10g")).type == WorkEventType.ADDED


def test_malformed_line_is_skipped():
    controller = GPUInstanceTypeController()

    controller._on_event("{not json")

    assert len(controller._queue._pending) == 0


def test_unexpected_verb_is_skipped():
    controller = GPUInstanceTypeController()

    controller._on_event(_line("BOOKMARK"))

    assert len(controller._queue._pending) == 0


def test_non_integer_cluster_is_skipped():
    controller = GPUInstanceTypeController()

    controller._on_event(_line("ADDED", cluster="not-a-number"))

    assert len(controller._queue._pending) == 0


def test_missing_name_is_skipped():
    controller = GPUInstanceTypeController()

    controller._on_event(json.dumps({"type": "ADDED", "object": {"cluster": "1"}}))

    assert len(controller._queue._pending) == 0


# --- sqlite reconcile ------------------------------------------------------ #


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        await conn.run_sync(GPUInstanceType.__table__.create)
    yield e
    await e.dispose()


@pytest.fixture
def controller(engine, monkeypatch):
    monkeypatch.setattr(
        "gpustack.gpu_instances.controllers.async_session",
        lambda: AsyncSession(engine, expire_on_commit=False),
    )
    return GPUInstanceTypeController()


def _event(etype, *, cluster_id=1, name="a10g", spec=None):
    obj = {
        "cluster": str(cluster_id),
        "metadata": {"name": name},
        "spec": spec or {},
    }
    wtype = {
        "ADDED": WorkEventType.ADDED,
        "MODIFIED": WorkEventType.MODIFIED,
        "DELETED": WorkEventType.DELETED,
    }[etype]
    return WorkEvent(keys=(cluster_id, name), type=wtype, object=obj)


async def _active(engine, *, cluster_id=1, name="a10g"):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await GPUInstanceType.first_by_fields(
            s, fields={"cluster_id": cluster_id, "name": name, "deleted_at": None}
        )


async def _all(engine, *, cluster_id=1, name="a10g"):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await GPUInstanceType.all_by_fields(
            s, fields={"cluster_id": cluster_id, "name": name}
        )


@pytest.mark.asyncio
async def test_added_creates_row(engine, controller):
    await controller._reconcile(
        _event("ADDED", spec={"acceleratorGroup": "nvidia-a10g"})
    )

    row = await _active(engine)
    assert row is not None
    assert row.spec.accelerator_group == "nvidia-a10g"
    assert row.snapshot.startswith("sha1:")
    assert row.deleted_at is None


@pytest.mark.asyncio
async def test_display_name_edit_updates_same_row(engine, controller):
    # A MODIFIED only ever changes display_name, which is excluded from the
    # snapshot — so it refreshes the SAME row and keeps the snapshot stable.
    await controller._reconcile(_event("ADDED", spec={"displayName": "A10G"}))
    before = await _active(engine)

    await controller._reconcile(_event("MODIFIED", spec={"displayName": "Renamed"}))
    after = await _active(engine)

    assert after.id == before.id  # same row, not a duplicate
    assert after.spec.display_name == "Renamed"
    assert after.snapshot == before.snapshot  # display_name is not hashed
    assert len(await _all(engine)) == 1


@pytest.mark.asyncio
async def test_deleted_soft_deletes_active_row(engine, controller):
    await controller._reconcile(_event("ADDED"))

    await controller._reconcile(_event("DELETED"))

    assert await _active(engine) is None  # nothing active
    rows = await _all(engine)
    assert len(rows) == 1 and rows[0].deleted_at is not None  # soft-deleted history


@pytest.mark.asyncio
async def test_readd_same_spec_after_delete_revives_row(engine, controller):
    await controller._reconcile(
        _event("ADDED", spec={"acceleratorGroup": "nvidia-a10g"})
    )
    original = await _active(engine)
    await controller._reconcile(_event("DELETED"))

    await controller._reconcile(
        _event("ADDED", spec={"acceleratorGroup": "nvidia-a10g"})
    )

    revived = await _active(engine)
    assert revived.id == original.id  # same snapshot -> revived, no duplicate
    assert revived.deleted_at is None
    assert len(await _all(engine)) == 1


@pytest.mark.asyncio
async def test_readd_changed_spec_after_delete_creates_new_row(engine, controller):
    # A same-named type recreated with different resources is a DIFFERENT type:
    # the old snapshot is kept as soft-deleted history, the new one is a new row.
    # (unitResources is definitional, so it diverges the snapshot.)
    await controller._reconcile(_event("ADDED", spec={"unitResources": {"ram": "1Mi"}}))
    old = await _active(engine)
    await controller._reconcile(_event("DELETED"))

    await controller._reconcile(_event("ADDED", spec={"unitResources": {"ram": "2Mi"}}))

    rows = await _all(engine)
    assert len(rows) == 2  # old + new coexist
    new = await _active(engine)
    assert new.id != old.id
    assert new.spec.unit_resources.ram == "2Mi" and new.deleted_at is None
    retired = next(r for r in rows if r.id == old.id)
    assert retired.snapshot == old.snapshot  # old snapshot preserved for resolve
    assert retired.deleted_at is not None


@pytest.mark.asyncio
async def test_new_snapshot_supersedes_stale_active_row(engine, controller):
    # If a DELETE was missed (watch has no resourceVersion resume), a new
    # snapshot must still retire the stale active row so exactly one stays active.
    await controller._reconcile(_event("ADDED", spec={"unitResources": {"ram": "1Mi"}}))

    await controller._reconcile(_event("ADDED", spec={"unitResources": {"ram": "2Mi"}}))

    active = [r for r in await _all(engine) if r.deleted_at is None]
    assert len(active) == 1 and active[0].spec.unit_resources.ram == "2Mi"


@pytest.mark.asyncio
async def test_delete_absent_row_is_noop(engine, controller):
    # DELETED for a type never seen must not raise or create anything.
    await controller._reconcile(_event("DELETED", name="ghost"))

    assert await _all(engine, name="ghost") == []


@pytest.mark.asyncio
async def test_integrity_error_race_falls_back_to_revive(
    engine, controller, monkeypatch
):
    # Simulate the concurrent-insert race: the snapshot lookup misses once, so
    # the create path runs and hits the snapshot unique constraint (a
    # soft-deleted row already holds it); the fallback re-queries and revives it.
    async with AsyncSession(engine, expire_on_commit=False) as s:
        seeded = GPUInstanceType(
            cluster_id=1,
            name="a10g",
            spec=GPUInstanceTypeSpec.model_validate({"unitResources": {"ram": "1Mi"}}),
        )
        seeded.snapshot = seeded.compute_snapshot()
        seeded.deleted_at = datetime(2020, 1, 1)  # soft-deleted (retire is a no-op)
        s.add(seeded)
        await s.commit()

    real = GPUInstanceType.first_by_fields.__func__
    missed = {"done": False}

    async def flaky_first_by_fields(cls, session, fields):
        # Force the FIRST snapshot lookup to miss; leave the active-row query
        # (retire) and the fallback lookup untouched.
        if "snapshot" in fields and not missed["done"]:
            missed["done"] = True
            return None
        return await real(cls, session, fields)

    monkeypatch.setattr(
        GPUInstanceType, "first_by_fields", classmethod(flaky_first_by_fields)
    )

    await controller._reconcile(_event("ADDED", spec={"unitResources": {"ram": "1Mi"}}))

    revived = await _active(engine)
    assert revived is not None and revived.spec.unit_resources.ram == "1Mi"
    assert len(await _all(engine)) == 1  # revived, not duplicated


@pytest.mark.asyncio
async def test_integrity_error_race_retires_stale_active_row(
    engine, controller, monkeypatch
):
    # The race with a DIFFERENT active row present: the new snapshot's insert
    # loses to a concurrent writer (IntegrityError), and that rollback also
    # reverts the just-issued retire of the old active row. The fallback must
    # re-retire it so exactly one row stays active — never two.
    spec_a = GPUInstanceTypeSpec.model_validate({"unitResources": {"ram": "1Mi"}})
    spec_b = GPUInstanceTypeSpec.model_validate({"unitResources": {"ram": "2Mi"}})
    async with AsyncSession(engine, expire_on_commit=False) as s:
        old_active = GPUInstanceType(cluster_id=1, name="a10g", spec=spec_a)
        old_active.snapshot = old_active.compute_snapshot()  # stays active
        s.add(old_active)
        # The new snapshot already exists (soft-deleted) so the create path hits
        # the unique constraint, standing in for the concurrent writer.
        seeded_new = GPUInstanceType(cluster_id=1, name="a10g", spec=spec_b)
        seeded_new.snapshot = seeded_new.compute_snapshot()
        seeded_new.deleted_at = datetime(2020, 1, 1)
        s.add(seeded_new)
        await s.commit()

    real = GPUInstanceType.first_by_fields.__func__
    missed = {"done": False}

    async def flaky_first_by_fields(cls, session, fields):
        # Force only the FIRST snapshot lookup to miss, so the create path runs.
        if "snapshot" in fields and not missed["done"]:
            missed["done"] = True
            return None
        return await real(cls, session, fields)

    monkeypatch.setattr(
        GPUInstanceType, "first_by_fields", classmethod(flaky_first_by_fields)
    )

    await controller._reconcile(_event("ADDED", spec={"unitResources": {"ram": "2Mi"}}))

    active = [r for r in await _all(engine) if r.deleted_at is None]
    assert len(active) == 1  # old_active retired, only the revived row is active
    assert active[0].spec.unit_resources.ram == "2Mi"


# --- resync (list-then-watch) ---------------------------------------------- #


@pytest.mark.asyncio
async def test_resync_enqueues_present_and_retires_absent(
    engine, controller, monkeypatch
):
    # An active row the fresh catalog no longer lists is retired; a listed type
    # is (re-)projected. This is the missed-DELETE + fresh-start recovery.
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=1,
                name="gone",
                spec=GPUInstanceTypeSpec(),
                snapshot="sha1:gone",
            )
        )
        await s.commit()

    listed = {
        "items": [
            {
                "cluster": "1",
                "metadata": {"name": "a10g"},
                "spec": {"acceleratorGroup": "nvidia-a10g"},
            }
        ]
    }
    monkeypatch.setattr(
        "gpustack.gpu_instances.controllers.gateway_client.list_instance_types",
        AsyncMock(return_value=listed),
    )

    await controller._resync()

    assert controller._queue._pending[(1, "a10g")].type == WorkEventType.ADDED
    assert controller._queue._pending[(1, "gone")].type == WorkEventType.DELETED


@pytest.mark.asyncio
async def test_resync_empty_list_skips_retire(engine, controller, monkeypatch):
    # An empty result is indistinguishable from an outage, so it must NOT retire
    # every active row — the retire pass is skipped entirely.
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=1,
                name="keep",
                spec=GPUInstanceTypeSpec(),
                snapshot="sha1:keep",
            )
        )
        await s.commit()

    monkeypatch.setattr(
        "gpustack.gpu_instances.controllers.gateway_client.list_instance_types",
        AsyncMock(return_value={"items": []}),
    )

    await controller._resync()

    assert len(controller._queue._pending) == 0  # nothing retired
