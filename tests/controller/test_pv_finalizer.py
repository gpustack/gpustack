"""E4a: GPUInstancePersistentVolumeController finalizes soft-deleted PVs.

Finalize enumerates every cluster and issues an idempotent, namespace-scoped
delete of the downstream PV CR; the delete reports whether it still existed, so
the clusters still holding it are recorded in ``status.finalizing`` and the row
is hard-deleted once none remain. A PV still referenced by an active GPUInstance
waits. The cluster enumeration + downstream ops are mocked; the DB is a real
in-memory sqlite so the ORM queries / delete run for real.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from kubernetes_asyncio import client
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances.controllers import GPUInstancePersistentVolumeController
from gpustack.schemas.clusters import Cluster, ClusterProvider
from gpustack.schemas.gpu_instances import GPUInstance
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeSpec,
    GPUInstancePersistentVolumeStatus,
)
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.workers import Worker, WorkerStateEnum

NAMESPACE = "gpustack-acme"


class FakeOps:
    """Stands in for ClusterOps: an async context manager whose
    ``delete_persistent_volume`` is idempotent and returns whether the object
    still existed on this cluster (``present``) — the finalizer's "still
    holding it" signal."""

    def __init__(self, present: bool = False, org_namespace: str = NAMESPACE):
        self.org_namespace = org_namespace
        self.delete_persistent_volume = AsyncMock(return_value=present)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        for table in (
            Principal.__table__,
            GPUInstancePersistentVolume.__table__,
            GPUInstance.__table__,
            Worker.__table__,
        ):
            await conn.run_sync(table.create)
    yield e
    await e.dispose()


@pytest.fixture
def controller(engine, monkeypatch):
    monkeypatch.setattr(
        "gpustack.gpu_instances.controllers.async_session",
        lambda: AsyncSession(engine, expire_on_commit=False),
    )
    return GPUInstancePersistentVolumeController(
        SimpleNamespace(get_api_port=lambda: 80)
    )


async def _seed(
    engine,
    *,
    phase="Deleting",
    finalizing=None,
    instance_ref=False,
    instance_phase=None,
    instance_count=1,
    workers=((10, WorkerStateEnum.READY), (20, WorkerStateEnum.READY)),
    unreachable_since=None,
):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(Principal(id=1, kind=PrincipalType.ORG, name="acme"))
        for index, worker in enumerate(workers, start=1):
            cluster_id, state = worker[0], worker[1]
            # A removed worker keeps its row: the delete route stamps DELETING
            # and soft-deletes it. That is what tells "never had a worker" apart
            # from "had a worker that is gone".
            removed = len(worker) > 2 and worker[2]
            s.add(
                Worker(
                    id=index,
                    name=f"w-{index}",
                    hostname=f"h-{index}",
                    ip=f"10.0.0.{index}",
                    ifname="eth0",
                    port=80,
                    status={},
                    worker_uuid=f"uuid-{index}",
                    cluster_id=cluster_id,
                    state=state,
                    deleted_at=datetime(2026, 10, 22) if removed else None,
                )
            )
        status = (
            GPUInstancePersistentVolumeStatus(
                phase=phase,
                finalizing=finalizing,
                unreachable_since=unreachable_since,
            )
            if phase
            else None
        )
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
        if instance_ref:
            for n in range(1, instance_count + 1):
                s.add(
                    GPUInstance(
                        id=n,
                        name=f"gi-{n}",
                        owner_principal_id=1,
                        cluster_id=10,
                        spec={"type_": "gpu", "image": "busybox"},
                        persistent_volume_id=1,
                        status=({"phase": instance_phase} if instance_phase else None),
                    )
                )
        await s.commit()


async def _get_pv(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await GPUInstancePersistentVolume.one_by_id(s, 1)


async def _release_instance(engine):
    """Drop the instance reference that blocks the row."""
    async with AsyncSession(engine, expire_on_commit=False) as s:
        instance = await GPUInstance.one_by_id(s, 1)
        if instance is not None:
            instance.persistent_volume_id = None
            await s.commit()


async def _set_worker_state(engine, cluster_id, state):
    """Flip the state of the cluster's worker."""
    async with AsyncSession(engine, expire_on_commit=False) as s:
        worker = await Worker.one_by_fields(s, fields={"cluster_id": cluster_id})
        worker.state = state
        await s.commit()


def _mock_clusters(monkeypatch, controller, ops_by_cluster, deleted=()):
    """Mock the live-cluster enumeration.

    The mock honours the ``fields`` argument the way the real query does, so a
    regression in the provider / deleted_at filter shows up in every test that
    uses it — not only in the one that spells the filter out.

    ``deleted`` names the clusters that are soft-deleted: they are still in the
    database, but they cannot hold a downstream object any more.
    """
    clusters = [
        SimpleNamespace(
            id=cid,
            provider=ClusterProvider.Kubernetes,
            deleted_at=datetime(2026, 10, 21) if cid in deleted else None,
        )
        for cid in ops_by_cluster
    ]

    async def fake_all_by_fields(session, fields=None, **kwargs):
        return [
            c
            for c in clusters
            if fields is None
            or all(getattr(c, k, None) == v for k, v in fields.items())
        ]

    monkeypatch.setattr(Cluster, "all_by_fields", fake_all_by_fields)
    monkeypatch.setattr(
        controller, "_build_ops", lambda cluster, owner: ops_by_cluster[cluster.id]
    )


@pytest.mark.asyncio
async def test_absent_downstream_hard_deletes_row(engine, controller, monkeypatch):
    await _seed(engine)
    ops = {10: FakeOps(present=False), 20: FakeOps(present=False)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    assert await _get_pv(engine) is None  # nothing still downstream → hard-deleted
    for o in ops.values():
        # Probed by an idempotent delete (returns False = already gone).
        o.delete_persistent_volume.assert_awaited_once_with("pv-1")


@pytest.mark.asyncio
async def test_present_downstream_records_finalizing_and_requeues(
    engine, controller, monkeypatch
):
    await _seed(engine)
    ops = {10: FakeOps(present=True), 20: FakeOps(present=False)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume.assert_awaited_once_with("pv-1")
    ops[20].delete_persistent_volume.assert_awaited_once_with("pv-1")  # gone here
    pv = await _get_pv(engine)
    assert pv is not None  # retained until downstream clears
    assert pv.status.phase == "Deleting"
    assert pv.status.finalizing == [10]  # only cluster 10 still held it
    assert len(controller._queue._delayed) == 1  # re-probe scheduled


@pytest.mark.asyncio
async def test_active_instance_reference_blocks_finalize(
    engine, controller, monkeypatch
):
    await _seed(engine, instance_ref=True)
    ops = {10: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume.assert_not_called()  # not probed while in use
    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.phase == "Deleting"
    assert pv.status.phase_message is not None
    assert len(controller._queue._delayed) == 1


@pytest.mark.asyncio
async def test_blocked_reason_names_the_holder_and_its_phase(
    engine, controller, monkeypatch
):
    """The FK is not phase-filtered, so a *Stopped* instance blocks reclaim just
    as much as a running one. Saying "active GPU instance(s)" was the misleading
    part — the message has to name the instance and its real phase, or the user
    cannot act on it."""
    await _seed(engine, instance_ref=True, instance_phase="Stopped")
    _mock_clusters(monkeypatch, controller, {10: FakeOps(present=True)})

    await controller._finalize(1)

    message = (await _get_pv(engine)).status.phase_message
    assert "gi-1" in message
    assert "Stopped" in message
    assert "active" not in message


@pytest.mark.asyncio
async def test_unchanged_finalizing_skips_db_write(engine, controller, monkeypatch):
    await _seed(engine, finalizing=[10])  # already recorded last round
    ops = {10: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    calls = []
    original = GPUInstancePersistentVolume.update

    async def _counting_update(self, *a, **k):
        calls.append(1)
        return await original(self, *a, **k)

    monkeypatch.setattr(GPUInstancePersistentVolume, "update", _counting_update)

    await controller._finalize(1)

    assert calls == []  # finalizing unchanged → no DB write (no churn)
    assert len(controller._queue._delayed) == 1  # but still re-probes


@pytest.mark.asyncio
async def test_finalizing_order_normalized_skips_db_write(
    engine, controller, monkeypatch
):
    # finalizing is compared order-insensitively: clusters probed in a different
    # order than stored (Cluster.all_by_fields has no ORDER BY) must not trip a
    # DB write.
    await _seed(engine, finalizing=[10, 20])
    # _mock_clusters preserves dict order, so these probe as [20, 10] — the
    # reverse of the stored [10, 20].
    ops = {20: FakeOps(present=True), 10: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    calls = []
    original = GPUInstancePersistentVolume.update

    async def _counting_update(self, *a, **k):
        calls.append(1)
        return await original(self, *a, **k)

    monkeypatch.setattr(GPUInstancePersistentVolume, "update", _counting_update)

    await controller._finalize(1)

    assert calls == []  # [20, 10] normalizes to [10, 20] == stored → no write
    assert len(controller._queue._delayed) == 1


@pytest.mark.asyncio
async def test_non_deleting_row_is_noop(engine, controller, monkeypatch):
    await _seed(engine, phase=None)
    ops = {10: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume.assert_not_called()
    assert await _get_pv(engine) is not None  # untouched


@pytest.mark.asyncio
async def test_missing_principal_hard_deletes_row(engine, controller, monkeypatch):
    # A soft-deleted PV whose owning principal is gone can never run cluster
    # ops, so the finalizer must hard-delete it instead of stranding it in
    # Deleting forever.
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstancePersistentVolume(
                id=1,
                name="pv-1",
                owner_principal_id=404,  # no such principal
                persistent_volume_type_id=2,
                spec=GPUInstancePersistentVolumeSpec(type_="t"),
                status=GPUInstancePersistentVolumeStatus(phase="Deleting"),
            )
        )
        await s.commit()
    clusters = AsyncMock(return_value=[])
    monkeypatch.setattr(Cluster, "all_by_fields", clusters)

    await controller._finalize(1)

    assert await _get_pv(engine) is None  # hard-deleted, not stranded
    clusters.assert_not_awaited()  # short-circuited before cluster enumeration


@pytest.mark.asyncio
async def test_blocked_reason_says_when_it_listed_only_some_holders(
    engine, controller, monkeypatch
):
    """A truncated list of holders has to admit it is truncated.

    The message names at most three instances. Cut silently, it told the user
    exactly which instances to release — and releasing all three left the volume
    still blocked, with the reason naming nothing new. So the query fetches one
    over the cap purely to detect the overflow and say so.
    """
    await _seed(engine, instance_ref=True, instance_phase="Ready", instance_count=5)
    _mock_clusters(monkeypatch, controller, {10: FakeOps(present=True)})

    await controller._finalize(1)

    message = (await _get_pv(engine)).status.phase_message
    assert message.count("gi-") == 3
    assert "and others" in message


@pytest.mark.asyncio
async def test_non_kubernetes_clusters_are_not_probed(engine, controller, monkeypatch):
    """Downstream CRDs only exist on active Kubernetes clusters, and
    non-Kubernetes or soft-deleted clusters cannot serve the apiserver proxy.
    Verify that cluster enumeration only probes active Kubernetes clusters,
    preventing unsupported proxy failures from holding the row in Deleting."""
    await _seed(engine)
    k8s_ops = {20: FakeOps(present=False)}

    all_clusters = [
        SimpleNamespace(id=1, provider=ClusterProvider.Docker, deleted_at=None),
        SimpleNamespace(id=20, provider=ClusterProvider.Kubernetes, deleted_at=None),
        SimpleNamespace(
            id=30, provider=ClusterProvider.Kubernetes, deleted_at="2026-10-21"
        ),
    ]

    async def fake_all_by_fields(session, fields=None, **kwargs):
        return [
            c
            for c in all_clusters
            if (
                fields is None
                or all(getattr(c, k, None) == v for k, v in fields.items())
            )
        ]

    monkeypatch.setattr(Cluster, "all_by_fields", fake_all_by_fields)

    def build_ops(cluster, owner):
        if cluster.id in (1, 30):
            raise RuntimeError("Inactive or non-Kubernetes cluster cannot be probed")
        return k8s_ops[cluster.id]

    monkeypatch.setattr(controller, "_build_ops", build_ops)

    await controller._finalize(1)

    assert await _get_pv(engine) is None
    k8s_ops[20].delete_persistent_volume.assert_awaited_once_with("pv-1")


@pytest.mark.asyncio
async def test_cluster_without_any_worker_is_skipped(engine, controller, monkeypatch):
    """A cluster that never had a worker never held the downstream object: the
    server creates it through the same proxy, and the proxy needs a worker. It
    must not keep the row in Deleting."""
    await _seed(engine, workers=())
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[20].delete_persistent_volume.assert_not_called()
    assert await _get_pv(engine) is None


@pytest.mark.asyncio
async def test_a_cluster_the_row_recorded_is_not_skipped_when_its_worker_row_is_gone(
    engine, controller, monkeypatch
):
    """A cluster a probe found holding the object is not skipped just because
    its worker row is gone now.

    A worker without an ``external_id`` is deleted rather than soft-deleted, so
    the cluster looks like it never had one. The row's own record says
    otherwise, and it is the stronger evidence: the row waits for the
    connection to come back instead of being released silently.
    """
    await _seed(engine, finalizing=[20], workers=())
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[20].delete_persistent_volume.assert_not_called()
    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.finalizing == [20]
    assert pv.status.unreachable_since is not None  # the wait is clocked


@pytest.mark.asyncio
async def test_a_recorded_cluster_is_given_up_on_after_the_window(
    engine, controller, monkeypatch, caplog
):
    """The wait on a recorded cluster is bounded like any other: after the
    window the row is released, and the release says the custom resource may
    still be there."""
    await _seed(
        engine,
        finalizing=[20],
        workers=(),
        unreachable_since={20: datetime(2020, 1, 1)},
    )
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    with caplog.at_level("WARNING"):
        await controller._finalize(1)

    ops[20].delete_persistent_volume.assert_not_called()
    assert "gives up on unreachable cluster(s) 20" in caplog.text
    assert await _get_pv(engine) is None  # released


@pytest.mark.asyncio
async def test_cluster_with_unready_workers_waits_for_the_connection(
    engine, controller, monkeypatch
):
    """The cluster had workers, so it may still hold the downstream object. The
    deletion waits for the connection to come back, and the remote API is not
    called while it is down."""
    await _seed(engine, workers=((20, WorkerStateEnum.NOT_READY),))
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[20].delete_persistent_volume.assert_not_called()
    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.phase == "Deleting"
    assert pv.status.finalizing == [20]
    assert "20" in pv.status.phase_message
    assert pv.status.unreachable_since is not None  # the wait is clocked


@pytest.mark.asyncio
async def test_the_wait_clock_is_not_restarted_by_the_next_round(
    engine, controller, monkeypatch
):
    """The stamp is the clock of the hold itself, so the next round has to reuse
    it. A clock that restarted every round would wait forever — the very
    symptom this fix removes."""
    await _seed(engine, workers=((20, WorkerStateEnum.NOT_READY),))
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)
    first = (await _get_pv(engine)).status.unreachable_since
    assert first is not None

    await controller._finalize(1)
    second = (await _get_pv(engine)).status.unreachable_since
    assert second == first


@pytest.mark.asyncio
async def test_cluster_with_unready_workers_is_given_up_on_after_the_window(
    engine, controller, monkeypatch
):
    """The wait is bounded: a cluster that stays unreachable must not hold the
    row forever. Once the window has passed the row is released, and the skip
    is reported."""
    await _seed(
        engine,
        workers=((20, WorkerStateEnum.NOT_READY),),
        unreachable_since={20: datetime(2020, 1, 1)},
    )
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[20].delete_persistent_volume.assert_not_called()
    assert await _get_pv(engine) is None


@pytest.mark.asyncio
async def test_zero_tolerance_gives_up_on_an_unreachable_cluster_at_once(
    engine, controller, monkeypatch
):
    await _seed(engine, workers=((20, WorkerStateEnum.NOT_READY),))
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)
    controller._unreachable_tolerance = 0

    await controller._finalize(1)

    assert await _get_pv(engine) is None


@pytest.mark.asyncio
async def test_removed_cluster_is_dropped_from_finalizing_while_blocked(
    engine, controller, monkeypatch
):
    """A cluster that is no longer in the database cannot hold the downstream
    object, so it must not keep the row in Deleting — not even on the path
    where the status write keeps the existing list."""
    await _seed(
        engine,
        finalizing=[20],
        instance_ref=True,
        workers=(),
    )
    _mock_clusters(monkeypatch, controller, {})

    await controller._finalize(1)

    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.finalizing == []


@pytest.mark.asyncio
async def test_soft_deleted_cluster_is_dropped_from_finalizing_while_blocked(
    engine, controller, monkeypatch
):
    """A cluster that was soft-deleted cannot hold the downstream object any
    more, so it must not keep the row in Deleting — not even on the path where
    the status write keeps the existing list."""
    await _seed(
        engine,
        finalizing=[20],
        instance_ref=True,
        workers=(),
    )
    _mock_clusters(monkeypatch, controller, {20: FakeOps(present=True)}, deleted={20})

    await controller._finalize(1)

    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.finalizing == []


@pytest.mark.asyncio
async def test_the_give_up_is_reported_when_the_row_outlives_the_wait(
    engine, controller, monkeypatch
):
    """The give-up drops the cluster from ``finalizing``, so the row no longer
    names it — but the downstream custom resource may still be there, and the
    user has to be able to see that."""
    await _seed(
        engine,
        workers=((10, WorkerStateEnum.READY), (20, WorkerStateEnum.NOT_READY)),
        unreachable_since={20: datetime(2020, 1, 1)},
    )
    ops = {10: FakeOps(present=True), 20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.finalizing == [10]  # 20 was given up on
    assert "20" in pv.status.phase_message
    # The clock is kept, so the give-up sticks (see the stickiness test below).
    assert pv.status.unreachable_since is not None


@pytest.mark.asyncio
async def test_the_give_up_is_logged_once_and_not_on_every_reprobe(
    engine, controller, monkeypatch, caplog
):
    """The row is re-probed on every requeue, so an unconditional warning would
    repeat it every few seconds for as long as the row lives — the log flooding
    the proxy gate exists to avoid."""
    await _seed(
        engine,
        workers=((10, WorkerStateEnum.READY), (20, WorkerStateEnum.NOT_READY)),
        unreachable_since={20: datetime(2020, 1, 1)},
    )
    ops = {10: FakeOps(present=True), 20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    with caplog.at_level("WARNING"):
        await controller._finalize(1)
        await controller._finalize(1)
        await controller._finalize(1)

    assert caplog.text.count("gives up on unreachable cluster(s) 20") == 1


@pytest.mark.asyncio
async def test_the_give_up_is_sticky_across_rounds(engine, controller, monkeypatch):
    """The give-up has to stick.

    A cleared clock would be re-stamped on the next round, the cluster would go
    back into ``finalizing``, and the row would wait — and be released — over
    and over instead of once. The release is the whole point of the window.
    """
    await _seed(
        engine,
        workers=((10, WorkerStateEnum.READY), (20, WorkerStateEnum.NOT_READY)),
        unreachable_since={20: datetime(2020, 1, 1)},
    )
    ops = {10: FakeOps(present=True), 20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)
    pv = await _get_pv(engine)
    assert pv.status.finalizing == [10]
    stamp = pv.status.unreachable_since

    await controller._finalize(1)
    pv = await _get_pv(engine)
    assert pv.status.finalizing == [10]  # 20 does not come back
    assert pv.status.unreachable_since == stamp


@pytest.mark.asyncio
async def test_a_new_outage_does_not_inherit_another_cluster_s_clock(
    engine, controller, monkeypatch
):
    """The wait clock is per cluster.

    A row-level clock would let a fresh outage inherit an expired clock from
    another cluster's outage: cluster 10 recovers and cluster 20 goes down in
    the same round, and cluster 20 — unreachable for zero seconds — would be
    given up on at once. The row would then be released while cluster 20 still
    holds the downstream object, which is the one thing the window exists to
    prevent.
    """
    await _seed(
        engine,
        workers=((10, WorkerStateEnum.NOT_READY), (20, WorkerStateEnum.READY)),
        unreachable_since={10: datetime(2020, 1, 1)},
    )
    ops = {10: FakeOps(present=True), 20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await _set_worker_state(engine, 10, WorkerStateEnum.READY)
    await _set_worker_state(engine, 20, WorkerStateEnum.NOT_READY)

    await controller._finalize(1)

    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.finalizing == [10, 20]  # the row waits for cluster 20
    assert set(pv.status.unreachable_since) == {20}
    assert pv.status.unreachable_since[20] > datetime(2020, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_a_stamp_carried_through_a_block_does_not_expire_a_fresh_hold(
    engine, controller, monkeypatch
):
    """The blocked path does not probe the clusters, so it cannot tell whether an
    unreachable-cluster hold is still running. Carrying an unverifiable stamp
    across it would let a stamp from an earlier hold expire a fresh one: the row
    would be released with no wait at all, which is the one thing the window
    exists to prevent."""
    await _seed(
        engine,
        workers=((10, WorkerStateEnum.READY), (20, WorkerStateEnum.NOT_READY)),
        unreachable_since={20: datetime(2020, 1, 1)},
        instance_ref=True,
    )
    ops = {10: FakeOps(present=True), 20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    # Blocked: the row waits for the referencing instance, and the stamp of the
    # hold it cannot verify is dropped.
    await controller._finalize(1)
    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.unreachable_since is None

    # The instance is released and the cluster is still down, so the row has to
    # wait for the window instead of being released at once.
    await _release_instance(engine)
    await controller._finalize(1)
    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.finalizing == [10, 20]
    assert pv.status.unreachable_since is not None


@pytest.mark.asyncio
async def test_a_removed_worker_makes_the_cluster_unreachable_not_skipped(
    engine, controller, monkeypatch
):
    """A worker that was removed keeps its (soft-deleted) row, so the cluster
    still counts as one that HAD a worker: it may still hold the downstream
    object, so the row waits for it instead of skipping it. Only a cluster that
    never had a worker is skipped."""
    await _seed(engine, workers=((20, WorkerStateEnum.DELETING, True),))
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[20].delete_persistent_volume.assert_not_called()
    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.finalizing == [20]
    assert "20" in pv.status.phase_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection reset"),
        client.exceptions.ApiException(status=503, reason="Service Unavailable"),
    ],
    ids=["connection-reset", "apiserver-503"],
)
async def test_a_failed_probe_still_keeps_the_cluster_in_finalizing(
    engine, controller, monkeypatch, error
):
    """The worker pre-check must not swallow a real failure. A cluster that
    answered but whose call failed still holds the object, so the row stays and
    the next round retries it — a transient error and an apiserver 503 alike."""
    await _seed(engine)
    ops = {10: FakeOps(present=True)}
    ops[10].delete_persistent_volume.side_effect = error
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.phase == "Deleting"
    assert pv.status.finalizing == [10]
