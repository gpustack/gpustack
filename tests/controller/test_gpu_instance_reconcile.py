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

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
import pytest_asyncio
from kubernetes_asyncio import client
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances.controllers import GPUInstanceController, _TRANSPORT_ERRORS
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstancePersistentVolumeReference,
    GPUInstancePersistentVolumeTemplate,
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
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.workqueue import WorkEvent, WorkEventType

NAMESPACE = "gpustack-user-1"

# The cluster the seeded instance lives in.
CLUSTER_ID = 2


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
        await conn.run_sync(Worker.__table__.create)
        await conn.run_sync(Principal.__table__.create)
        await conn.run_sync(Cluster.__table__.create)
    # The delete path asks the cluster whether it can be reached, and the proxy
    # serves the downstream API through a READY worker. The default is a
    # reachable cluster; the unreachable cases change the worker row.
    async with AsyncSession(e, expire_on_commit=False) as s:
        s.add(
            Worker(
                id=1,
                name="w-1",
                hostname="h-1",
                ip="10.0.0.1",
                ifname="eth0",
                port=80,
                worker_uuid="uuid-1",
                cluster_id=CLUSTER_ID,
                state=WorkerStateEnum.READY,
            )
        )
        await s.commit()
    yield e
    await e.dispose()


@pytest.fixture
def controller(engine, monkeypatch):
    monkeypatch.setattr(
        "gpustack.gpu_instances.controllers.async_session",
        lambda: AsyncSession(engine, expire_on_commit=False),
    )
    return GPUInstanceController(SimpleNamespace(get_api_port=lambda: 80))


def _with_ops(controller, ops, *, cluster=None):
    """Point the controller at ``ops`` for the instance's cluster.

    ``_build_ops`` resolves the cluster + principal and hands back the client
    with the cluster row, so the delete path can ask whether the cluster can be
    reached at all.
    """
    controller._build_ops = AsyncMock(
        return_value=SimpleNamespace(
            cluster=cluster if cluster is not None else _cluster(),
            ops=ops,
            principal_identifier="user-1",
        )
    )
    return ops


def _cluster(*, cluster_id=CLUSTER_ID, name="cluster-2", deleted_at=None):
    return Cluster(id=cluster_id, name=name, deleted_at=deleted_at)


async def _seed_owner_and_cluster(engine, *, deleted_at=None):
    """Seed the rows the real ``_build_ops`` looks up.

    The instance's owning principal and its cluster, the latter optionally
    soft-deleted.
    """
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(Principal(id=1, name="user-1", kind=PrincipalType.USER))
        s.add(
            Cluster(
                id=CLUSTER_ID,
                name="cluster-2",
                registration_token="token-2",
                owner_principal_id=1,
                deleted_at=deleted_at,
            )
        )
        await s.commit()


async def _set_worker_state(engine, state):
    """Move the cluster's worker out of READY — the cluster stays, its worker
    does not answer, so the proxy cannot serve the cluster."""
    async with AsyncSession(engine, expire_on_commit=False) as s:
        worker = await Worker.one_by_id(s, 1)
        worker.state = state
        await s.commit()


async def _remove_worker(engine):
    """Drop the cluster's worker row entirely: a worker without an
    ``external_id`` is deleted rather than soft-deleted, so the cluster is left
    with no worker row at all."""
    async with AsyncSession(engine, expire_on_commit=False) as s:
        worker = await Worker.one_by_id(s, 1)
        await s.delete(worker)
        await s.commit()


async def _seed(
    engine,
    *,
    phase=None,
    phase_message=None,
    spec=None,
    unreadable_since=None,
    deleting_unreachable_since=None,
):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        status = (
            GPUInstanceStatus(
                phase=phase,
                phase_message=phase_message,
                namespace=NAMESPACE,
                unreadable_since=unreadable_since,
                deleting_unreachable_since=deleting_unreachable_since,
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


async def _seed_pv(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstancePersistentVolume(
                id=1,
                name="pv-1",
                owner_principal_id=1,
                persistent_volume_type_id=2,
                spec=GPUInstancePersistentVolumeSpec(type_="t"),
            )
        )
        await s.commit()


async def _get_pv(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await GPUInstancePersistentVolume.one_by_id(s, 1)


def _template_volume_spec():
    """A spec whose template PV opted into being released with the instance."""
    return GPUInstanceSpec(
        type_="gpu",
        image="busybox",
        volume=GPUInstanceVolume(
            persistent_template=GPUInstancePersistentVolumeTemplate(
                name="pv-1",
                spec=GPUInstancePersistentVolumeSpec(type_="t"),
                release_with_instance=True,
            )
        ),
    )


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


@pytest.mark.asyncio
async def test_deleting_makes_no_call_when_the_cluster_has_no_ready_worker(
    engine, controller
):
    # The proxy serves the downstream API through a READY worker and answers
    # 503 without one, so calling it would fail anyway — and the failure would
    # read as a fault instead of the wait it is.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    await _set_worker_state(engine, WorkerStateEnum.NOT_READY)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    ops.read_instance.assert_not_awaited()
    ops.delete_instance.assert_not_awaited()
    row = await _get(engine)
    assert row is not None
    assert row.status.phase == GPUInstancePhase.DELETING
    assert _requeued(controller) == 1


@pytest.mark.asyncio
async def test_deleting_holds_when_the_read_fails(engine, controller):
    # A read that fails is not the "the custom resource is gone" answer: the
    # proxy could not serve the call, so the cluster is out of reach for now.
    # The row waits instead of failing — a failed reconcile is retried forever.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    _with_ops(
        controller,
        FakeOps(
            read_side_effect=client.exceptions.ApiException(
                status=503, reason="Service Unavailable"
            )
        ),
    )

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.phase == GPUInstancePhase.DELETING
    assert _requeued(controller) == 1
    assert "cannot reach the cluster" in row.status.phase_message


@pytest.mark.asyncio
async def test_deleting_tells_a_refused_cluster_apart_from_a_silent_one(
    engine, controller
):
    # A refusal comes back from the cluster itself: it answered, and said no.
    # Calling that "the server cannot reach the cluster" blames the network for
    # something the network did not do, and hides what the user has to fix.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    _with_ops(
        controller,
        FakeOps(
            read_side_effect=client.exceptions.ApiException(
                status=403, reason="Forbidden"
            )
        ),
    )

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    message = row.status.phase_message
    assert message
    assert "cannot reach the cluster" not in message
    assert "403" in message


@pytest.mark.asyncio
async def test_deleting_keeps_the_row_when_the_read_fails_locally(engine, controller):
    # A failure on our side is no evidence that the cluster is unreachable: the
    # call may never have reached it, and the custom resource may still be
    # sitting there. Releasing the row would leave that resource behind with
    # nobody to clean it up — so the row stays, and the failure is retried.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        deleting_unreachable_since=datetime.now(timezone.utc) - timedelta(seconds=3600),
    )
    _with_ops(
        controller,
        FakeOps(
            read_side_effect=client.exceptions.ApiException(
                status=0, reason="Failed to parse '2024-01-01' as date object"
            )
        ),
    )

    with pytest.raises(client.exceptions.ApiException):
        await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.phase == GPUInstancePhase.DELETING


@pytest.mark.asyncio
async def test_deleting_tells_the_user_about_a_local_read_failure(engine, controller):
    # The row waits with a reason; a reason that blames the network for our own
    # failure sends the user after the wrong thing.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    _with_ops(
        controller,
        FakeOps(
            read_side_effect=client.exceptions.ApiException(
                status=0, reason="Failed to parse '2024-01-01' as date object"
            )
        ),
    )

    with pytest.raises(client.exceptions.ApiException):
        await controller._reconcile_instance(1, {})

    row = await _get(engine)
    message = row.status.phase_message
    assert message
    assert "cannot reach the cluster" not in message
    assert "internal error" in message
    assert "cluster-2" in message


@pytest.mark.asyncio
async def test_deleting_still_holds_on_a_transport_failure(engine, controller):
    # The wait this change guards must keep working: a cluster that cannot be
    # reached at all still waits, and the wait is still clocked.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    _with_ops(
        controller,
        FakeOps(read_side_effect=aiohttp.ClientConnectionError("connection refused")),
    )

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.deleting_unreachable_since is not None
    assert "cannot reach the cluster" in row.status.phase_message


@pytest.mark.asyncio
async def test_deleting_gives_up_on_a_transport_failure_at_the_bound(
    engine, controller
):
    # Only a transport failure may end the row: a broken link to the cluster
    # says the cluster is out of reach, and past the bound that is what the
    # give-up is for.
    #
    # The timeout is spelled the way a client raises it. On Python 3.10
    # ``asyncio.TimeoutError`` is a class of its own, so the builtin
    # ``TimeoutError`` would not stand in for it there.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        deleting_unreachable_since=datetime.now(timezone.utc) - timedelta(seconds=3600),
    )
    _with_ops(
        controller,
        FakeOps(read_side_effect=asyncio.TimeoutError("timed out")),
    )

    await controller._reconcile_instance(1, {})

    assert await _get(engine) is None


@pytest.mark.asyncio
async def test_deleting_holds_on_a_truncated_response(engine, controller):
    # A response that stops halfway is a broken link, not our bug: the cluster
    # was reached and the connection died on the way back.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    _with_ops(
        controller,
        FakeOps(read_side_effect=aiohttp.ClientPayloadError("response ended early")),
    )

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.deleting_unreachable_since is not None
    assert "cannot reach the cluster" in row.status.phase_message


@pytest.mark.asyncio
async def test_deleting_holds_on_a_client_timeout(engine, controller):
    # A client that stops waiting for an answer raises a timeout. The cluster
    # was reached and never replied, so this is a broken link, not our bug:
    # the row waits, and the wait is clocked like any other.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    _with_ops(
        controller,
        FakeOps(read_side_effect=asyncio.TimeoutError("timed out")),
    )

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.deleting_unreachable_since is not None
    assert "cannot reach the cluster" in row.status.phase_message


def test_the_transport_bucket_names_the_client_timeout():
    # ``asyncio.TimeoutError`` is the builtin ``TimeoutError`` — and so an
    # ``OSError`` — from Python 3.11 on, and ``OSError`` alone would cover the
    # timeout there. On 3.10 it is a class of its own, and a hung cluster
    # would then read as a fault of this server: the row would be retried
    # forever as "an internal error" instead of waiting out the unreachable
    # tolerance. CI runs 3.11 and 3.12, so no other test would notice that
    # gap — the bucket has to name the class for 3.10 to be covered.
    assert asyncio.TimeoutError in _TRANSPORT_ERRORS


@pytest.mark.asyncio
async def test_deleting_keeps_the_row_on_a_programming_error(engine, controller):
    # A bug in this server is our own: the cluster may be perfectly reachable,
    # with the custom resource sitting on it. The row must not be released for
    # that, and the failure must not pass silently either.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    _with_ops(
        controller,
        FakeOps(read_side_effect=ValueError("bad payload")),
    )

    with pytest.raises(ValueError):
        await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.phase == GPUInstancePhase.DELETING


@pytest.mark.asyncio
async def test_deleting_holds_when_the_cluster_has_no_worker_at_all(engine, controller):
    # The cluster is the instance's own: it was created through this cluster's
    # proxy, so it may still hold the custom resource even when no worker row
    # is left to prove it. The row waits for the cluster rather than claiming a
    # deletion that never happened.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    await _remove_worker(engine)
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    ops.read_instance.assert_not_awaited()
    row = await _get(engine)
    assert row is not None
    assert row.status.phase == GPUInstancePhase.DELETING
    assert _requeued(controller) == 1


@pytest.mark.asyncio
async def test_deleting_stamps_the_hold_clock_once(engine, controller):
    # The clock belongs to the hold, not to the row: an unrelated write must
    # not restart the window, so the second pass carries the first one's clock.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    await _set_worker_state(engine, WorkerStateEnum.NOT_READY)
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})
    first = (await _get(engine)).status.deleting_unreachable_since
    assert first is not None

    await controller._reconcile_instance(1, {})
    second = (await _get(engine)).status.deleting_unreachable_since

    assert second == first


@pytest.mark.asyncio
async def test_deleting_explains_the_wait_to_the_user(engine, controller):
    # A row stuck in Deleting with no message is the symptom this change is
    # about: the user must learn which cluster is unreachable, why, and that
    # the wait ends.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    await _set_worker_state(engine, WorkerStateEnum.NOT_READY)
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row.status.phase == GPUInstancePhase.DELETING
    message = row.status.phase_message
    assert message
    assert "cluster-2" in message
    assert "no ready worker" in message
    assert "1800" in message
    assert _requeued(controller) == 1


@pytest.mark.asyncio
async def test_deleting_gives_up_at_the_bound(engine, controller, caplog):
    # The cluster stayed unreachable past the window: the row is released, the
    # template PV with it, and the operator is told which cluster to clean up.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        deleting_unreachable_since=datetime.now(timezone.utc) - timedelta(seconds=3600),
        spec=_template_volume_spec(),
    )
    await _seed_pv(engine)
    await _set_worker_state(engine, WorkerStateEnum.NOT_READY)
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    with caplog.at_level(logging.WARNING):
        await controller._reconcile_instance(1, {})

    assert await _get(engine) is None
    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.phase == GPUInstancePhase.DELETING
    assert "cluster-2" in caplog.text
    assert "may still exist" in caplog.text
    assert _requeued(controller) == 0


@pytest.mark.asyncio
async def test_the_give_up_says_why_the_cluster_failed(engine, controller, caplog):
    # The warning must not call a refusal an unreachable cluster: an operator
    # chasing a 403 needs the status, not a story about the network.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        deleting_unreachable_since=datetime.now(timezone.utc) - timedelta(seconds=3600),
    )
    _with_ops(
        controller,
        FakeOps(
            read_side_effect=client.exceptions.ApiException(
                status=403, reason="Forbidden"
            )
        ),
    )

    with caplog.at_level(logging.WARNING):
        await controller._reconcile_instance(1, {})

    assert "403" in caplog.text
    assert "could not be reached" not in caplog.text
    assert await _get(engine) is None


@pytest.mark.asyncio
async def test_the_give_up_says_how_long_the_row_waited(engine, controller, caplog):
    # The wait is what ended the row, not a fresh outage: an operator who only
    # reads "could not be reached" would go after the network instead of after
    # the wait.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        deleting_unreachable_since=datetime.now(timezone.utc) - timedelta(seconds=3600),
    )
    await _set_worker_state(engine, WorkerStateEnum.NOT_READY)
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    with caplog.at_level(logging.WARNING):
        await controller._reconcile_instance(1, {})

    assert "after waiting 3600s" in caplog.text


@pytest.mark.asyncio
async def test_the_successful_release_is_logged(engine, controller, caplog):
    # The other two paths that release a row both warn. The release that can
    # leave a custom resource behind must not be the silent one.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    _with_ops(controller, FakeOps(read_return=None))

    with caplog.at_level(logging.INFO):
        await controller._reconcile_instance(1, {})

    assert "custom resource is gone" in caplog.text


@pytest.mark.asyncio
async def test_deleting_gives_up_at_once_when_the_tolerance_is_zero(engine, controller):
    # A tolerance of 0 means "wait no time at all": the cluster is not
    # reachable, so the row goes at once.
    controller._deleting_unreachable_tolerance = 0
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    await _set_worker_state(engine, WorkerStateEnum.NOT_READY)
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    assert await _get(engine) is None
    assert _requeued(controller) == 0


@pytest.mark.asyncio
async def test_deleting_clears_the_hold_clock_when_the_cluster_answers(
    engine, controller
):
    # The clock lives exactly as long as the hold. A cluster that recovers ends
    # the hold, so the next outage starts its own window instead of inheriting
    # an expired one.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        deleting_unreachable_since=datetime.now(timezone.utc) - timedelta(seconds=3600),
    )
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.deleting_unreachable_since is None


@pytest.mark.asyncio
async def test_deleting_clears_the_hold_message_when_the_cluster_answers(
    engine, controller
):
    # The message explains the wait, so it ends with the wait: a row that says
    # "the server cannot reach the cluster" while the cluster is fine misleads
    # the user about what is happening to their instance.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        phase_message=(
            "Waiting for cluster 'cluster-2' (id 2) to finish the deletion: the "
            "server cannot reach the cluster (no ready worker); the row is "
            "released after 1800s anyway"
        ),
        deleting_unreachable_since=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.deleting_unreachable_since is None
    assert row.status.phase_message is None


@pytest.mark.asyncio
async def test_deleting_keeps_the_hold_clock_when_the_delete_fails(engine, controller):
    # A read that succeeds says the cluster answers, not that the deletion
    # works. Ending the hold on the read would hand a cluster that keeps
    # failing its delete a fresh window every time it comes back, so the
    # bound would never run out.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        deleting_unreachable_since=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))
    ops.delete_instance = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.deleting_unreachable_since is not None


@pytest.mark.asyncio
async def test_deleting_gives_up_after_a_cluster_that_answers_but_cannot_delete(
    engine, controller
):
    # The cluster answers, but its delete never lands, and it drops out again.
    # The hold must not start over on every answer, or the row waits for a
    # cluster that keeps coming back — the bound is what ends the wait.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        deleting_unreachable_since=datetime.now(timezone.utc) - timedelta(seconds=3600),
    )
    ops = _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))
    ops.delete_instance = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await controller._reconcile_instance(1, {})

    # The cluster goes dark again: the hold picks up the clock it already had.
    await _set_worker_state(engine, WorkerStateEnum.NOT_READY)
    await controller._reconcile_instance(1, {})

    assert await _get(engine) is None


@pytest.mark.asyncio
async def test_deleting_holds_when_the_cluster_was_soft_deleted(engine, controller):
    # The proxy answers 404 for a soft-deleted cluster, and the read takes
    # that for "the custom resource is gone" — a deletion that never happened.
    # The cluster row is still there, so the row waits for it.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    cluster = _cluster(deleted_at=datetime.now(timezone.utc))
    ops = _with_ops(controller, FakeOps(read_return=None), cluster=cluster)

    await controller._reconcile_instance(1, {})

    ops.read_instance.assert_not_awaited()
    row = await _get(engine)
    assert row is not None
    assert row.status.phase == GPUInstancePhase.DELETING
    assert "deleted" in row.status.phase_message


@pytest.mark.asyncio
async def test_deleting_holds_when_the_soft_deleted_cluster_comes_from_the_db(
    engine, controller
):
    # The "a proxy 404 is not a missing custom resource" defense rests on the
    # cluster lookup returning soft-deleted rows: the gate reads the cluster's
    # deleted_at and holds instead of completing the deletion. The case above
    # hands a soft-deleted cluster to a mocked _build_ops, so nothing there
    # would notice if the lookup started filtering soft-deleted rows out — the
    # gate would go dead, and the row would be released at once, as a
    # deletion that never happened.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    await _seed_owner_and_cluster(engine, deleted_at=datetime.now(timezone.utc))

    await controller._reconcile_instance(1, {})

    row = await _get(engine)
    assert row is not None
    assert row.status.phase == GPUInstancePhase.DELETING
    assert "deleted" in row.status.phase_message


@pytest.mark.asyncio
async def test_deleting_releases_the_row_when_the_cluster_row_is_gone(
    engine, controller
):
    # The server can never reach the downstream custom resource again, so the
    # row would wait in Deleting forever, keeping the instance's name
    # reserved and telling the user nothing.
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        spec=_template_volume_spec(),
    )
    await _seed_pv(engine)
    controller._build_ops = AsyncMock(return_value=None)

    await controller._reconcile_instance(1, {})

    assert await _get(engine) is None
    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.phase == GPUInstancePhase.DELETING


@pytest.mark.asyncio
async def test_the_hold_does_not_raise_the_log_level_to_error(
    engine, controller, caplog
):
    # An unreachable cluster during the hold is an expected wait, not a fault.
    # An ERROR traceback on every pass would drown the log, and the hold is
    # re-observed every few seconds.
    await _seed(engine, phase=GPUInstancePhase.DELETING)
    await _set_worker_state(engine, WorkerStateEnum.NOT_READY)
    _with_ops(controller, FakeOps(read_return=_read(GPUInstancePhase.READY)))

    with caplog.at_level(logging.DEBUG):
        await controller._reconcile_instance(1, {})
        await controller._reconcile_instance(1, {})

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    holds = [r for r in caplog.records if "holding the deletion" in r.getMessage()]
    assert len(holds) == 1
    assert holds[0].levelno == logging.WARNING


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


# --- the hold clocks (a status field, not a phase) ------------------------ #


@pytest.mark.asyncio
async def test_the_hold_clocks_round_trip_through_the_status_column(engine):
    # Both clocks live in the JSON status column, so a new one needs no
    # migration — which is why the round-trip through that column is worth a
    # test of its own.
    since = datetime.now(timezone.utc)
    await _seed(
        engine,
        phase=GPUInstancePhase.DELETING,
        unreadable_since=since,
        deleting_unreachable_since=since,
    )

    stored = await _get(engine)

    assert stored.status.unreadable_since == since
    assert stored.status.deleting_unreachable_since == since


def test_merge_from_kuberes_drops_the_hold_clocks():
    # A merge carries only what the worker reports, so a clock ends with its
    # hold. The worker reports neither clock — the server stamps them — so a
    # merge of a real CR must not resurrect a stale one.
    inst = GPUInstance(
        id=1,
        name="gi-1",
        owner_principal_id=1,
        cluster_id=CLUSTER_ID,
        spec=GPUInstanceSpec(type_="gpu", image="busybox"),
        status=GPUInstanceStatus(
            phase=GPUInstancePhase.DELETING,
            namespace=NAMESPACE,
            unreadable_since=datetime.now(timezone.utc),
            deleting_unreachable_since=datetime.now(timezone.utc),
        ),
    )

    merged = inst.merge_from_kuberes({"status": {"phase": "Ready"}})

    assert merged.unreadable_since is None
    assert merged.deleting_unreachable_since is None
