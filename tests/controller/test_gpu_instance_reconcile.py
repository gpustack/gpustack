"""B2: the upstream phase state machine + change-gated write-back + backoff.

``_reconcile_instance`` re-fetches the row and drives the worker side per phase
against a mocked ``ClusterOps`` over a real in-memory sqlite DB:

- creating -> provision, then observe (synthesize Unknown if not yet readable)
- starting -> clear spec.stop (rebuild if the CR is gone); hold Starting until
  the worker leaves Stopped
- stopping -> issue stop until the worker reports Stopped
- deleting -> tear down; delete the row once the CR is gone
- ready / not-ready / unknown -> observe; a vanished CR settles to Stopped
- stopped / *failed -> settled, no-op

``_db_update_instance_status`` writes only on change and re-observes a still-
transitioning row via ``add_after``; ``_process`` retries a failed reconcile
with ``add_rate_limited`` and resets on success.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances.controllers import GPUInstanceController
from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstancePersistentVolumeReference,
    GPUInstancePhase,
    GPUInstanceSpec,
    GPUInstanceStatus,
    GPUInstanceVolume,
)
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeSpec,
    GPUInstancePersistentVolumeStatus,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.workqueue import WorkEvent, WorkEventType

NAMESPACE = "gpustack-user-1"


class FakeOps:
    """ClusterOps stand-in exposing the worker methods the machine touches."""

    def __init__(self, read_return=None, read_side_effect=None):
        self.org_namespace = NAMESPACE
        self.cluster_owner_principal_identifier = "user-1"
        self.read_instance = AsyncMock(
            return_value=read_return, side_effect=read_side_effect
        )
        self.create_instance = AsyncMock()
        self.delete_instance = AsyncMock()
        self.delete_ssh_public_key = AsyncMock()
        self.upsert_ssh_public_key = AsyncMock()
        self.stop_instance = AsyncMock()
        self.start_instance = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _read(phase, **status):
    return {"metadata": {"namespace": NAMESPACE}, "status": {"phase": phase, **status}}


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        await conn.run_sync(GPUInstance.__table__.create)
        await conn.run_sync(GPUInstancePersistentVolume.__table__.create)
    yield e
    await e.dispose()


@pytest.fixture
def controller(engine, monkeypatch):
    monkeypatch.setattr(
        "gpustack.gpu_instances.controllers.async_session",
        lambda: AsyncSession(engine, expire_on_commit=False),
    )
    return GPUInstanceController(SimpleNamespace(get_api_port=lambda: 80))


def _with_ops(controller, ops):
    controller._build_ops = AsyncMock(return_value=(ops, "user-1"))
    return ops


async def _seed(
    engine, *, phase=None, phase_message=None, spec=None, unreadable_since=None
):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        status = (
            GPUInstanceStatus(
                phase=phase,
                phase_message=phase_message,
                namespace=NAMESPACE,
                unreadable_since=unreadable_since,
            )
            if phase is not None
            else None
        )
        s.add(
            GPUInstance(
                id=1,
                name="gi-1",
                owner_principal_id=1,
                cluster_id=2,
                spec=spec or GPUInstanceSpec(type_="gpu", image="busybox"),
                status=status,
            )
        )
        await s.commit()


async def _get(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await GPUInstance.one_by_id(s, 1)


def _requeued(controller) -> int:
    return len(controller._queue._delayed)


# --- creating -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_creating_provisions_then_synthesizes_unknown(engine, controller):
    await _seed(engine, phase=None)  # is_creating
    ops = _with_ops(controller, FakeOps(read_return=None))  # CR not readable yet

    await controller._reconcile_instance(1, {})

    ops.create_instance.assert_awaited_once()
    row = await _get(engine)
    # A non-None phase is persisted so the next pass leaves is_creating.
    assert row.status.phase == GPUInstancePhase.UNKNOWN


@pytest.mark.asyncio
async def test_creating_failure_records_create_failed(engine, controller):
    await _seed(engine, phase=None)
    ops = _with_ops(controller, FakeOps())
    ops.create_instance = AsyncMock(side_effect=RuntimeError("boom"))

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row.status.phase == GPUInstancePhase.CREATE_FAILED
    ops.read_instance.assert_not_awaited()  # stopped before observe


@pytest.mark.asyncio
async def test_creating_rejects_deleting_pv(engine, controller):
    # A soft-deleted (Deleting) PV must not be provisioned against: the create
    # fails with PV_CREATE_FAILED rather than racing the PV finalizer.
    spec = GPUInstanceSpec(
        type_="gpu",
        image="busybox",
        volume=GPUInstanceVolume(
            persistent=GPUInstancePersistentVolumeReference(name="pv-1")
        ),
    )
    await _seed(engine, phase=None, spec=spec)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstancePersistentVolume(
                id=1,
                name="pv-1",
                owner_principal_id=1,
                persistent_volume_type_id=2,
                spec=GPUInstancePersistentVolumeSpec(type_="pvt-1"),
                status=GPUInstancePersistentVolumeStatus(phase="Deleting"),
            )
        )
        await s.commit()
    ops = _with_ops(controller, FakeOps())

    await controller._reconcile_instance(1, {})

    assert (await _get(engine)).status.phase == GPUInstancePhase.PV_CREATE_FAILED
    ops.create_instance.assert_not_awaited()  # never provisioned the instance CR


@pytest.mark.asyncio
async def test_creating_present_merges_worker_phase(engine, controller):
    await _seed(engine, phase=None)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.NOT_READY)))

    await controller._reconcile_instance(1, {})

    ops.create_instance.assert_awaited_once()
    assert (await _get(engine)).status.phase == GPUInstancePhase.NOT_READY


# --- starting -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_starting_rebuilds_when_cr_absent(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.STARTING)
    ops = _with_ops(controller, FakeOps(read_return=None))

    await controller._reconcile_instance(1, {})

    ops.create_instance.assert_awaited_once()
    ops.start_instance.assert_not_awaited()
    assert _requeued(controller) == 1  # re-observe after rebuild
    assert (await _get(engine)).status.phase == GPUInstancePhase.STARTING


@pytest.mark.asyncio
async def test_starting_holds_while_worker_still_stopped(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.STARTING)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.STOPPED)))

    await controller._reconcile_instance(1, {})

    ops.start_instance.assert_awaited_once()  # un-stop patch issued
    assert _requeued(controller) == 1
    # Held at Starting (worker has not left Stopped yet) — not reverted.
    assert (await _get(engine)).status.phase == GPUInstancePhase.STARTING


@pytest.mark.asyncio
async def test_starting_writes_worker_phase_once_up(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.STARTING)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.NOT_READY)))

    await controller._reconcile_instance(1, {})

    ops.start_instance.assert_awaited_once()
    assert (await _get(engine)).status.phase == GPUInstancePhase.NOT_READY


@pytest.mark.asyncio
async def test_start_failure_message_is_stable_across_errors(engine, controller):
    # Same backoff guard as stop/delete: the un-stop retry message stays stable
    # so repeat failures dedup instead of self-publishing past the backoff.
    await _seed(engine, phase=GPUInstancePhase.STARTING)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    ops.start_instance = AsyncMock(side_effect=RuntimeError("boom-1"))
    with pytest.raises(RuntimeError):
        await controller._reconcile_instance(1, {})
    first = (await _get(engine)).status.phase_message

    ops.start_instance = AsyncMock(side_effect=RuntimeError("boom-2-different"))
    with pytest.raises(RuntimeError):
        await controller._reconcile_instance(1, {})
    second = (await _get(engine)).status.phase_message

    assert first == second
    assert "boom" not in (first or "")


@pytest.mark.asyncio
async def test_starting_reapplies_spec_on_resume(engine, controller):
    # A config edit made while Stopped is re-applied to the worker CR on resume:
    # start_instance carries the full row spec, not just the un-stop patch, so
    # the edit takes effect instead of silently no-op-ing.
    spec = GPUInstanceSpec(type_="gpu", image="busybox:edited")
    await _seed(engine, phase=GPUInstancePhase.STARTING, spec=spec)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.NOT_READY)))

    await controller._reconcile_instance(1, {})

    ops.start_instance.assert_awaited_once()
    sent_spec = ops.start_instance.await_args.kwargs["spec"]
    assert sent_spec["image"] == "busybox:edited"


# --- stopping -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stopping_issues_stop_until_worker_stops(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.STOPPING)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    ops.stop_instance.assert_awaited_once()
    assert _requeued(controller) == 1
    assert (await _get(engine)).status.phase == GPUInstancePhase.STOPPING  # held


@pytest.mark.asyncio
async def test_stopping_settles_when_worker_reports_stopped(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.STOPPING)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.STOPPED)))

    await controller._reconcile_instance(1, {})

    ops.stop_instance.assert_not_awaited()  # already stopped
    assert (await _get(engine)).status.phase == GPUInstancePhase.STOPPED


@pytest.mark.asyncio
async def test_stopping_settles_when_cr_gone(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.STOPPING)
    _with_ops(controller, FakeOps(read_return=None))

    await controller._reconcile_instance(1, {})

    assert (await _get(engine)).status.phase == GPUInstancePhase.STOPPED


@pytest.mark.asyncio
async def test_stop_failure_writes_message_and_raises(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.STOPPING)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))
    ops.stop_instance = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row.status.phase == GPUInstancePhase.STOPPING  # phase kept
    assert row.status.phase_message is not None


@pytest.mark.asyncio
async def test_stop_failure_message_is_stable_across_errors(engine, controller):
    # The retry message must not embed the exception text: a varying message
    # would change the status on every failure, self-publish an event, and slip
    # past the work-queue backoff (hot-loop). A stable message lets
    # _status_equivalent dedup repeat failures. The full error stays in the logs.
    await _seed(engine, phase=GPUInstancePhase.STOPPING)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    ops.stop_instance = AsyncMock(side_effect=RuntimeError("boom-1"))
    with pytest.raises(RuntimeError):
        await controller._reconcile_instance(1, {})
    first = (await _get(engine)).status.phase_message

    ops.stop_instance = AsyncMock(side_effect=RuntimeError("boom-2-different"))
    with pytest.raises(RuntimeError):
        await controller._reconcile_instance(1, {})
    second = (await _get(engine)).status.phase_message

    assert first == second  # stable across errors
    assert "boom" not in (first or "")  # exception text stays in the logs only


# --- deleting -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_deleting_tears_down_and_requeues_while_cr_present(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    ops.delete_instance.assert_awaited_once()
    ops.delete_ssh_public_key.assert_awaited_once()
    assert _requeued(controller) == 1
    assert (await _get(engine)) is not None  # row not yet deleted


@pytest.mark.asyncio
async def test_deleting_deletes_row_when_cr_gone(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    _with_ops(controller, FakeOps(read_return=None))

    await controller._reconcile_instance(1, {})

    assert (await _get(engine)) is None  # hard-deleted


@pytest.mark.asyncio
async def test_delete_failure_message_is_stable_across_errors(engine, controller):
    # Same backoff guard as the stopping path: the retry message stays stable so
    # repeat failures dedup instead of self-publishing past the backoff.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    ops.delete_instance = AsyncMock(side_effect=RuntimeError("boom-1"))
    with pytest.raises(RuntimeError):
        await controller._reconcile_instance(1, {})
    first = (await _get(engine)).status.phase_message

    ops.delete_instance = AsyncMock(side_effect=RuntimeError("boom-2-different"))
    with pytest.raises(RuntimeError):
        await controller._reconcile_instance(1, {})
    second = (await _get(engine)).status.phase_message

    assert first == second
    assert "boom" not in (first or "")


# --- observe (ready / not-ready / unknown) --------------------------------- #


@pytest.mark.asyncio
async def test_steady_ready_writes_nothing_and_does_not_requeue(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.READY)
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    assert _requeued(controller) == 0  # settled: no re-observe
    assert (await _get(engine)).status.phase == GPUInstancePhase.READY


@pytest.mark.asyncio
async def test_transitioning_unchanged_requeues(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.NOT_READY)
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.NOT_READY)))

    await controller._reconcile_instance(1, {})

    assert _requeued(controller) == 1  # still transitioning -> re-observe in 60s


@pytest.mark.asyncio
async def test_observe_settles_to_stopped_when_cr_gone(engine, controller):
    # A fully-Ready workload whose CR vanished -> Stopped (a /start rebuilds it).
    await _seed(engine, phase=GPUInstancePhase.READY)
    _with_ops(controller, FakeOps(read_return=None))

    await controller._reconcile_instance(1, {})

    assert (await _get(engine)).status.phase == GPUInstancePhase.STOPPED


@pytest.mark.asyncio
async def test_unknown_absent_keeps_observing_not_stopped(engine, controller):
    # A not-yet-Ready row whose CR is (still) absent must keep observing as
    # Unknown, never prematurely settle to Stopped — the CR may just be lagging
    # (eventual consistency); settling would strand it once it appears.
    since = datetime.now(timezone.utc) - timedelta(seconds=60)
    await _seed(
        engine,
        phase=GPUInstancePhase.UNKNOWN,
        phase_message="Not found in cluster",
        unreadable_since=since,
    )
    _with_ops(controller, FakeOps(read_return=None))

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row.status.phase == GPUInstancePhase.UNKNOWN  # not Stopped
    assert _requeued(controller) == 1  # keeps re-observing
    # The hold's clock is carried forward, not re-stamped: a synthetic status
    # that changed every pass would write on every tick AND reset the bound.
    assert row.status.unreadable_since == since


@pytest.mark.asyncio
async def test_unknown_absent_settles_to_stopped_past_the_tolerance(engine, controller):
    """The eventual-consistency hold above is BOUNDED. ``Unknown`` is a metered
    phase, so a CR that is gone for good (cluster torn down, or an
    uninstall-and-reinstall upgrade that drops every Instance CR) would otherwise
    keep accruing uptime forever. Past the tolerance the row settles to Stopped,
    which stops metering and leaves it restartable."""
    # The hold's own clock, backdated past the tolerance.
    await _seed(
        engine,
        phase=GPUInstancePhase.UNKNOWN,
        phase_message="Not found in cluster",
        unreadable_since=datetime.now(timezone.utc) - timedelta(seconds=3600),
    )
    controller._unreadable_cr_tolerance = 1800
    _with_ops(controller, FakeOps(read_return=None))

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row.status.phase == GPUInstancePhase.STOPPED
    assert row.status.phase_message == "Worker-side instance not found"


@pytest.mark.asyncio
async def test_an_unrelated_edit_does_not_restart_the_tolerance(engine, controller):
    """The bound must survive a write to the row that has nothing to do with it.

    The clock used to be ``updated_at``, which reads plausible — ``_write_status``
    writes only on a real change, so it does stop advancing once the hold settles.
    But it is ROW-level, and ``display_name`` is editable from any phase. So
    renaming an instance while its CR was unreadable pushed the 30-minute ceiling
    back to zero, and doing it repeatedly kept a metered Unknown row alive
    indefinitely — with no trace that it had happened.
    """
    since = datetime.now(timezone.utc) - timedelta(seconds=3600)
    await _seed(
        engine,
        phase=GPUInstancePhase.UNKNOWN,
        phase_message="Not found in cluster",
        unreadable_since=since,
    )
    # The user renames it right now: ``updated_at`` becomes NOW.
    async with AsyncSession(engine) as s:
        row = await GPUInstance.one_by_id(s, 1)
        await row.update(s, source={"display_name": "renamed just now"})
    fresh = await _get(engine)
    assert (
        datetime.now(timezone.utc) - fresh.updated_at.replace(tzinfo=timezone.utc)
    ).total_seconds() < 60

    controller._unreadable_cr_tolerance = 1800
    _with_ops(controller, FakeOps(read_return=None))
    await controller._reconcile_instance(1, {})

    # Still settles: the hold began an hour ago and the rename did not touch that.
    assert (await _get(engine)).status.phase == GPUInstancePhase.STOPPED


@pytest.mark.asyncio
async def test_the_hold_clock_survives_the_json_round_trip_unchanged(
    engine, controller
):
    """Two observations in a row must produce a byte-identical status.

    The stamp goes out through ``model_dump`` into a JSON column and comes back
    through pydantic validation. If that round trip altered it at all — dropped
    microseconds, changed the offset — ``_status_equivalent`` would see a change on
    every pass, so the row would be written on every tick AND its bound would move
    forward each time. That is the failure the row-level ``updated_at`` had, in a
    new place, so equality here is the invariant the whole fix rests on.
    """
    await _seed(
        engine, phase=GPUInstancePhase.UNKNOWN, phase_message="Not found in cluster"
    )
    _with_ops(controller, FakeOps(read_return=None))

    await controller._reconcile_instance(1, {})  # stamps + writes
    stamped = (await _get(engine)).status.unreadable_since
    requeues_after_write = _requeued(controller)

    await controller._reconcile_instance(1, {})  # must be a no-op write-wise

    row = await _get(engine)
    assert row.status.unreadable_since == stamped
    assert _requeued(controller) == requeues_after_write + 1  # requeued, not written


@pytest.mark.asyncio
async def test_a_readable_cr_clears_the_hold_clock(engine, controller):
    """The stamp exists only while the hold does.

    ``merge_from_kuberes`` builds the status purely from the CR payload and the
    worker never reports this field, so a real read drops it — which is what makes
    a later hold start its own clock instead of inheriting a stale one.
    """
    await _seed(
        engine,
        phase=GPUInstancePhase.UNKNOWN,
        phase_message="Not found in cluster",
        unreadable_since=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    _with_ops(
        controller,
        FakeOps(
            read_return={
                "metadata": {"namespace": NAMESPACE},
                "status": {"phase": GPUInstancePhase.READY},
            }
        ),
    )

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row.status.phase == GPUInstancePhase.READY
    assert row.status.unreadable_since is None


@pytest.mark.asyncio
async def test_a_hold_from_an_older_version_gets_stamped_on_first_observe(
    engine, controller
):
    """A row already at Unknown when the server upgrades has no stamp.

    It gets one on the next observation, which restarts its tolerance once — at
    the version boundary only, since every later pass carries the stamp forward.
    Better than the alternatives: falling back to ``updated_at`` would keep the
    resettable clock alive, and refusing to expire without a stamp would leave
    those rows metered forever.
    """
    await _seed(
        engine, phase=GPUInstancePhase.UNKNOWN, phase_message="Not found in cluster"
    )
    assert (await _get(engine)).status.unreadable_since is None
    controller._unreadable_cr_tolerance = 1800
    _with_ops(controller, FakeOps(read_return=None))

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row.status.phase == GPUInstancePhase.UNKNOWN  # not swept up
    assert row.status.unreadable_since is not None


@pytest.mark.asyncio
async def test_tolerance_does_not_sweep_up_other_transitioning_phases(
    engine, controller
):
    """Only the "Unknown + CR unreadable" hold is bounded. A row that is
    Starting for its own reasons is not being held by this tolerance and must
    keep its phase even though it is long past the window."""
    await _seed(
        engine,
        phase=GPUInstancePhase.STARTING,
        unreadable_since=datetime.now(timezone.utc) - timedelta(seconds=99999),
    )
    controller._unreadable_cr_tolerance = 1800
    row = await _get(engine)
    assert controller._unreadable_cr_expired(row) is False


@pytest.mark.asyncio
async def test_tolerance_of_zero_restores_the_unbounded_hold(engine, controller):
    await _seed(
        engine,
        phase=GPUInstancePhase.UNKNOWN,
        phase_message="Not found in cluster",
        unreadable_since=datetime.now(timezone.utc) - timedelta(days=30),
    )
    controller._unreadable_cr_tolerance = 0
    row = await _get(engine)
    assert controller._unreadable_cr_expired(row) is False


# --- terminal -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stopped_is_noop(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.STOPPED)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    ops.read_instance.assert_not_awaited()
    assert (await _get(engine)).status.phase == GPUInstancePhase.STOPPED


@pytest.mark.asyncio
async def test_failed_is_noop(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.CREATE_FAILED)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    ops.read_instance.assert_not_awaited()
    assert (await _get(engine)).status.phase == GPUInstancePhase.CREATE_FAILED


# --- SSH provisioning and resync ------------------------------------------- #


@pytest.mark.asyncio
async def test_spec_change_resyncs_ssh_public_key(engine, controller):
    await _seed(engine, phase=GPUInstancePhase.READY)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {"spec": (None, None)})

    ops.upsert_ssh_public_key.assert_awaited_once()


@pytest.mark.asyncio
async def test_creating_provisions_the_key_even_with_no_keys_referenced(
    engine, controller
):
    # ``to_kuberes`` writes ``spec.sshPublicKey`` unconditionally, so the
    # operator always mounts a secret named after the instance. Skipping the
    # upsert when nothing is referenced leaves that mount dangling and the pod
    # sits in ContainerCreating forever on a secret that will never appear.
    # An empty body is the right answer, not a missing one: the operator
    # renders it as an empty ``authorized_keys``.
    await _seed(engine, phase=None)  # is_creating; spec has no ssh_public_keys
    ops = _with_ops(controller, FakeOps(read_return=None))

    await controller._reconcile_instance(1, {})

    ops.upsert_ssh_public_key.assert_awaited_once_with(name="gi-1", spec={"data": ""})


# --- _process backoff wiring ----------------------------------------------- #


def _work_event():
    return WorkEvent(
        keys=(1,),
        type=WorkEventType.MODIFIED,
        object=Event(type=EventType.UPDATED, data=GPUInstance(id=1)),
    )


@pytest.mark.asyncio
async def test_process_backs_off_on_failure(controller, monkeypatch):
    controller._reconcile = AsyncMock(side_effect=RuntimeError("boom"))
    rate = MagicMock()
    forget = MagicMock()
    monkeypatch.setattr(controller._queue, "add_rate_limited", rate)
    monkeypatch.setattr(controller._queue, "forget", forget)

    ev = _work_event()
    await controller._process(ev)

    rate.assert_called_once_with(ev)
    forget.assert_not_called()


@pytest.mark.asyncio
async def test_process_resets_backoff_on_success(controller, monkeypatch):
    controller._reconcile = AsyncMock()
    rate = MagicMock()
    forget = MagicMock()
    monkeypatch.setattr(controller._queue, "add_rate_limited", rate)
    monkeypatch.setattr(controller._queue, "forget", forget)

    await controller._process(_work_event())

    forget.assert_called_once_with((1,))
    rate.assert_not_called()


@pytest.mark.asyncio
async def test_ready_sweep_reenqueues_only_ready_rows(engine, controller):
    # The opt-in Ready sweep re-observes settled Ready rows only; transitioning
    # rows already self-requeue and terminal rows no-op, so they are skipped.
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for iid, phase in (
            (1, GPUInstancePhase.READY),
            (2, GPUInstancePhase.STOPPED),
            (3, GPUInstancePhase.STARTING),
        ):
            s.add(
                GPUInstance(
                    id=iid,
                    name=f"gi-{iid}",
                    owner_principal_id=1,
                    cluster_id=2,
                    spec=GPUInstanceSpec(type_="gpu", image="busybox"),
                    status=GPUInstanceStatus(phase=phase, namespace=NAMESPACE),
                )
            )
        await s.commit()

    await controller._sweep_ready_once()

    assert set(controller._queue._pending.keys()) == {(1,)}  # only the Ready row
