"""GPUInstanceTypeController: watch-event mapping + sqlite catalog projection.

The controller has no DB bus source for the catalog — one list-then-watch per
cluster over ``ClusterOps`` is authoritative, and the ``Cluster`` bus only
decides which clusters get a watcher. ``_on_watch_event`` maps each
ADDED/MODIFIED/DELETED native watch event onto a ``WorkEvent`` keyed by the
stable ``(cluster_id, name)`` identity, with the cluster id supplied by the
watcher (a raw CR carries none) and the raw object carried so the reconcile
needs no second fetch. ``_reconcile`` upserts the row on ADDED/MODIFIED
(reviving a soft-deleted one) and soft-deletes it on DELETED, over a real
in-memory sqlite DB.
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances.controllers import GPUInstanceTypeController
from gpustack.schemas.clusters import ClusterProvider
from gpustack.schemas.gpu_instance_types import (
    GPUInstanceType,
    GPUInstanceTypeSpec,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.workqueue import WorkEvent, WorkEventType


def _controller():
    return GPUInstanceTypeController(MagicMock())


def _obj(*, name="a10g", spec=None):
    return {"metadata": {"name": name}, "spec": spec or {}}


def _pending(controller, keys):
    return controller._queue._pending[keys]


# --- event mapping (no DB) ------------------------------------------------- #


def test_added_maps_to_added_keyed_on_cluster_and_name():
    controller = _controller()

    controller._on_watch_event(2, "ADDED", _obj(name="a100"))

    event = _pending(controller, (2, "a100"))
    assert event.type == WorkEventType.ADDED
    assert event.object["metadata"]["name"] == "a100"  # raw object carried


def test_modified_maps_to_modified():
    controller = _controller()

    controller._on_watch_event(1, "MODIFIED", _obj())

    assert _pending(controller, (1, "a10g")).type == WorkEventType.MODIFIED


def test_deleted_maps_to_deleted():
    controller = _controller()

    controller._on_watch_event(1, "DELETED", _obj())

    assert _pending(controller, (1, "a10g")).type == WorkEventType.DELETED


def test_deleted_coalesces_over_pending_modified():
    controller = _controller()

    controller._on_watch_event(1, "MODIFIED", _obj())
    controller._on_watch_event(1, "DELETED", _obj())

    # Latest-wins: the later DELETED replaces the earlier MODIFIED in the slot.
    assert _pending(controller, (1, "a10g")).type == WorkEventType.DELETED


def test_added_after_pending_deleted_wins():
    # A catalog DELETED is NOT terminal: a later ADDED (recreate) for the same
    # key must win, not be discarded by DELETED stickiness (latest-wins policy).
    controller = _controller()

    controller._on_watch_event(1, "DELETED", _obj())
    controller._on_watch_event(1, "ADDED", _obj())

    assert _pending(controller, (1, "a10g")).type == WorkEventType.ADDED


def test_unexpected_verb_is_skipped():
    controller = _controller()

    controller._on_watch_event(1, "BOOKMARK", _obj())

    assert len(controller._queue._pending) == 0


def test_missing_name_is_skipped():
    controller = _controller()

    controller._on_watch_event(1, "ADDED", {"spec": {}})

    assert len(controller._queue._pending) == 0


# --- watcher set (Cluster bus -> per-cluster watchers) --------------------- #


def _cluster_event(
    etype,
    *,
    cluster_id=1,
    provider=ClusterProvider.Kubernetes,
    token="tok",
):
    return Event(
        type=etype,
        data=SimpleNamespace(
            id=cluster_id, provider=provider, registration_token=token
        ),
    )


async def _cancelled(task) -> bool:
    """Whether ``task`` ended cancelled. ``Task.cancel()`` only *requests* the
    cancellation, so the task has to be given a chance to unwind first."""
    await asyncio.gather(task, return_exceptions=True)
    return task.cancelled()


@pytest_asyncio.fixture
async def watcher_controller(monkeypatch):
    """A controller whose per-cluster watcher is a no-op sleeper, so the watcher
    set can be exercised without any cluster I/O."""
    controller = _controller()

    async def idle(cluster_id, registration_token):
        await asyncio.Event().wait()

    monkeypatch.setattr(controller, "_watch_cluster", idle)
    yield controller
    for _, task in controller._watchers.values():
        task.cancel()
    await asyncio.gather(
        *(task for _, task in controller._watchers.values()), return_exceptions=True
    )


@pytest.mark.asyncio
async def test_kubernetes_cluster_starts_a_watcher(watcher_controller):
    watcher_controller._reconcile_cluster(_cluster_event(EventType.CREATED))

    assert set(watcher_controller._watchers) == {1}


@pytest.mark.asyncio
async def test_non_kubernetes_cluster_is_ignored(watcher_controller):
    watcher_controller._reconcile_cluster(
        _cluster_event(EventType.CREATED, provider=ClusterProvider.Docker)
    )

    assert watcher_controller._watchers == {}


@pytest.mark.asyncio
async def test_repeated_event_keeps_the_same_watcher(watcher_controller):
    # The bus republishes a cluster on every heartbeat-driven update; restarting
    # the watcher each time would re-list the catalog for no reason.
    watcher_controller._reconcile_cluster(_cluster_event(EventType.CREATED))
    first = watcher_controller._watchers[1][1]

    watcher_controller._reconcile_cluster(_cluster_event(EventType.UPDATED))

    assert watcher_controller._watchers[1][1] is first


@pytest.mark.asyncio
async def test_rotated_token_restarts_the_watcher(watcher_controller):
    # The token authenticates the cluster proxy, so a running watcher built with
    # the old one can no longer reach the cluster.
    watcher_controller._reconcile_cluster(_cluster_event(EventType.CREATED))
    first = watcher_controller._watchers[1][1]

    watcher_controller._reconcile_cluster(
        _cluster_event(EventType.UPDATED, token="rotated")
    )

    assert watcher_controller._watchers[1][1] is not first
    assert await _cancelled(first)


@pytest.mark.asyncio
async def test_deleted_cluster_stops_its_watcher(watcher_controller):
    watcher_controller._reconcile_cluster(_cluster_event(EventType.CREATED))
    task = watcher_controller._watchers[1][1]

    watcher_controller._reconcile_cluster(_cluster_event(EventType.DELETED))

    assert watcher_controller._watchers == {}
    assert await _cancelled(task)


@pytest.mark.asyncio
async def test_heartbeat_without_a_row_is_ignored(watcher_controller):
    watcher_controller._reconcile_cluster(Event(type=EventType.HEARTBEAT, data=None))

    assert watcher_controller._watchers == {}


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
    return _controller()


def _event(etype, *, cluster_id=1, name="a10g", spec=None, status=None):
    obj = {
        "metadata": {"name": name},
        "spec": spec or {},
    }
    if status is not None:
        obj["status"] = status
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
async def test_modified_backfills_status_detail(engine, controller):
    # The operator backfills status.detail asynchronously: the ADDED carries no
    # status, the MODIFIED carries it with an unchanged spec — same row, same
    # snapshot, detail recorded.
    await controller._reconcile(
        _event("ADDED", spec={"acceleratorGroup": "nvidia-a10g"})
    )
    before = await _active(engine)
    assert before.status is None  # not backfilled yet

    detail = {"manufacturer": "nvidia", "product": "A10G", "memory": "24576Mi"}
    await controller._reconcile(
        _event(
            "MODIFIED",
            spec={"acceleratorGroup": "nvidia-a10g"},
            status={"phase": "Active", "detail": detail},
        )
    )
    after = await _active(engine)

    assert after.id == before.id  # same row, not a duplicate
    assert after.snapshot == before.snapshot  # status is not hashed
    assert after.status.detail.manufacturer == "nvidia"
    assert after.status.detail.product == "A10G"
    # Only ``detail`` is persisted; the rest of the status is read live.
    assert set(after.status.model_dump(exclude_none=True)) == {"detail"}
    assert len(await _all(engine)) == 1


@pytest.mark.asyncio
async def test_status_less_modified_keeps_persisted_detail(engine, controller):
    # An event carrying no status means "no status information", not "cleared":
    # it must not wipe an already-backfilled detail.
    detail = {"manufacturer": "nvidia", "memory": "24576Mi"}
    await controller._reconcile(
        _event(
            "ADDED",
            spec={"acceleratorGroup": "nvidia-a10g"},
            status={"detail": detail},
        )
    )

    # A watch object carries the FULL spec; a display_name edit changes nothing
    # definitional, so the snapshot still matches and the row is refreshed.
    await controller._reconcile(
        _event(
            "MODIFIED",
            spec={"acceleratorGroup": "nvidia-a10g", "displayName": "Renamed"},
        )
    )

    row = await _active(engine)
    assert row.spec.display_name == "Renamed"
    assert row.status.detail.manufacturer == "nvidia"


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


def _ops(items, resource_version="1"):
    return SimpleNamespace(
        list_instance_types=AsyncMock(
            return_value={
                "metadata": {"resourceVersion": resource_version},
                "items": items,
            }
        )
    )


async def _seed(engine, *, cluster_id, name):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=cluster_id,
                name=name,
                spec=GPUInstanceTypeSpec(),
                snapshot=f"sha1:{cluster_id}-{name}",
            )
        )
        await s.commit()


@pytest.mark.asyncio
async def test_resync_enqueues_present_and_retires_absent(engine, controller):
    # An active row the fresh catalog no longer lists is retired; a listed type
    # is (re-)projected. This is the missed-DELETE + fresh-start recovery.
    await _seed(engine, cluster_id=1, name="gone")

    await controller._resync(
        _ops(
            [
                {
                    "metadata": {"name": "a10g"},
                    "spec": {"acceleratorGroup": "nvidia-a10g"},
                }
            ]
        ),
        1,
    )

    assert controller._queue._pending[(1, "a10g")].type == WorkEventType.ADDED
    assert controller._queue._pending[(1, "gone")].type == WorkEventType.DELETED


@pytest.mark.asyncio
async def test_resync_empty_list_retires(engine, controller):
    # A per-cluster list either answers for that cluster or raises, so an empty
    # result really is an empty catalog and its rows must be retired.
    await _seed(engine, cluster_id=1, name="gone")

    await controller._resync(_ops([]), 1)

    assert controller._queue._pending[(1, "gone")].type == WorkEventType.DELETED


@pytest.mark.asyncio
async def test_resync_returns_the_list_resource_version(engine, controller):
    # The watch resumes from it, so the snapshot and the stream join with no gap
    # — and a version-less watch would be rejected outright (WatchList semantics).
    assert await controller._resync(_ops([], resource_version="913"), 1) == "913"


@pytest.mark.asyncio
async def test_resync_leaves_other_clusters_alone(engine, controller):
    # Each cluster has its own watcher and its own resync: one cluster's catalog
    # must never retire another's rows.
    await _seed(engine, cluster_id=2, name="other")

    await controller._resync(_ops([]), 1)

    assert controller._queue._pending == {}
