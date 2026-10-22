"""E4b: GPUInstancePersistentVolumeTypeController finalizes soft-deleted PVTs.

PV before PVT: a PVT with any referencing PV row waits (the PV->PVT FK is ON
DELETE RESTRICT). Otherwise finalize enumerates every cluster and issues an
idempotent delete of the cluster-scoped downstream PVT by its owner-folded name;
the delete reports whether it still existed, so the still-holding clusters are
recorded and the row is hard-deleted once none remain. Cluster enumeration +
downstream ops are mocked; the DB is a real in-memory sqlite.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances.cluster_apis_util import get_persistent_volume_type_name
from gpustack.gpu_instances.controllers import (
    GPUInstancePersistentVolumeTypeController,
)
from gpustack.schemas.clusters import Cluster, ClusterProvider
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeSpec,
)
from gpustack.schemas.gpu_instance_persistent_volume_types import (
    GPUInstancePersistentVolumeType,
    GPUInstancePersistentVolumeTypeSpec,
    GPUInstancePersistentVolumeTypeStatus,
)
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.workers import Worker, WorkerStateEnum

# Owner ORG "acme" -> principal identifier "acme" -> this downstream PVT name.
CLUSTER_NAME = get_persistent_volume_type_name("pvt-1", principal_identifier="acme")


class FakeOps:
    """Stands in for ClusterOps: ``delete_persistent_volume_type`` is idempotent
    and returns whether the object still existed on this cluster (``present``) —
    the finalizer's "still holding it" signal."""

    def __init__(self, present: bool = False):
        self.delete_persistent_volume_type = AsyncMock(return_value=present)

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
            GPUInstancePersistentVolumeType.__table__,
            GPUInstancePersistentVolume.__table__,
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
    return GPUInstancePersistentVolumeTypeController(
        SimpleNamespace(get_api_port=lambda: 80)
    )


async def _seed(
    engine,
    *,
    phase="Deleting",
    finalizing=None,
    pv_ref=False,
    workers=((10, WorkerStateEnum.READY), (20, WorkerStateEnum.READY)),
    unreachable_since=None,
):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(Principal(id=1, kind=PrincipalType.ORG, name="acme"))
        for index, (cluster_id, state) in enumerate(workers, start=1):
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
                )
            )
        status = (
            GPUInstancePersistentVolumeTypeStatus(
                phase=phase,
                finalizing=finalizing,
                unreachable_since=unreachable_since,
            )
            if phase
            else None
        )
        s.add(
            GPUInstancePersistentVolumeType(
                id=1,
                name="pvt-1",
                owner_principal_id=1,
                spec=GPUInstancePersistentVolumeTypeSpec(),
                status=status,
            )
        )
        if pv_ref:
            s.add(
                GPUInstancePersistentVolume(
                    id=1,
                    name="pv-1",
                    owner_principal_id=1,
                    persistent_volume_type_id=1,
                    spec=GPUInstancePersistentVolumeSpec(type_="pvt-1"),
                )
            )
        await s.commit()


async def _get_pvt(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await GPUInstancePersistentVolumeType.one_by_id(s, 1)


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

    assert await _get_pvt(engine) is None  # nothing still downstream → hard-deleted
    for o in ops.values():
        o.delete_persistent_volume_type.assert_awaited_once_with(CLUSTER_NAME)


@pytest.mark.asyncio
async def test_present_downstream_records_finalizing_and_requeues(
    engine, controller, monkeypatch
):
    await _seed(engine)
    ops = {10: FakeOps(present=True), 20: FakeOps(present=False)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume_type.assert_awaited_once_with(CLUSTER_NAME)
    ops[20].delete_persistent_volume_type.assert_awaited_once_with(CLUSTER_NAME)
    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.phase == "Deleting"
    assert pvt.status.finalizing == [10]  # only cluster 10 still held it
    assert len(controller._queue._delayed) == 1


@pytest.mark.asyncio
async def test_referencing_pv_blocks_finalize(engine, controller, monkeypatch):
    await _seed(engine, pv_ref=True)
    ops = {10: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume_type.assert_not_called()  # PV before PVT
    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.phase == "Deleting"
    assert pvt.status.phase_message is not None
    assert len(controller._queue._delayed) == 1


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

    assert await _get_pvt(engine) is None
    k8s_ops[20].delete_persistent_volume_type.assert_awaited_once_with(CLUSTER_NAME)


@pytest.mark.asyncio
async def test_cluster_without_any_worker_is_skipped(engine, controller, monkeypatch):
    """A cluster that never had a worker never held the downstream object: the
    server creates it through the same proxy, and the proxy needs a worker. It
    must not keep the row in Deleting."""
    await _seed(engine, workers=())
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[20].delete_persistent_volume_type.assert_not_called()
    assert await _get_pvt(engine) is None


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

    ops[20].delete_persistent_volume_type.assert_not_called()
    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.finalizing == [20]
    assert pvt.status.unreachable_since is not None  # the wait is clocked


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

    ops[20].delete_persistent_volume_type.assert_not_called()
    assert "gives up on unreachable cluster(s) 20" in caplog.text
    assert await _get_pvt(engine) is None  # released


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

    ops[20].delete_persistent_volume_type.assert_not_called()
    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.phase == "Deleting"
    assert pvt.status.finalizing == [20]
    assert "20" in pvt.status.phase_message
    assert pvt.status.unreachable_since is not None  # the wait is clocked


@pytest.mark.asyncio
async def test_cluster_with_unready_workers_is_given_up_on_after_the_window(
    engine, controller, monkeypatch
):
    """The wait is bounded: a cluster that stays unreachable must not hold the
    row forever. Once the window has passed the row is released."""
    await _seed(
        engine,
        workers=((20, WorkerStateEnum.NOT_READY),),
        unreachable_since={20: datetime(2020, 1, 1)},
    )
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[20].delete_persistent_volume_type.assert_not_called()
    assert await _get_pvt(engine) is None


@pytest.mark.asyncio
async def test_zero_tolerance_gives_up_on_an_unreachable_cluster_at_once(
    engine, controller, monkeypatch
):
    await _seed(engine, workers=((20, WorkerStateEnum.NOT_READY),))
    ops = {20: FakeOps(present=True)}
    _mock_clusters(monkeypatch, controller, ops)
    controller._unreachable_tolerance = 0

    await controller._finalize(1)

    assert await _get_pvt(engine) is None


@pytest.mark.asyncio
async def test_a_new_outage_does_not_inherit_another_cluster_s_clock(
    engine, controller, monkeypatch
):
    """The wait clock is per cluster: a fresh outage must not inherit an expired
    clock from another cluster's outage, or the row would be released with no
    wait at all while the new cluster still holds the downstream object."""
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

    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.finalizing == [10, 20]  # the row waits for cluster 20
    assert set(pvt.status.unreachable_since) == {20}


@pytest.mark.asyncio
async def test_removed_cluster_is_dropped_from_finalizing_while_blocked(
    engine, controller, monkeypatch
):
    """A cluster that is no longer in the database cannot hold the downstream
    object, so it must not keep the row in Deleting — not even on the path
    where the status write keeps the existing list."""
    await _seed(engine, finalizing=[20], pv_ref=True, workers=())
    _mock_clusters(monkeypatch, controller, {})

    await controller._finalize(1)

    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.finalizing == []


@pytest.mark.asyncio
async def test_soft_deleted_cluster_is_dropped_from_finalizing_while_blocked(
    engine, controller, monkeypatch
):
    """A cluster that was soft-deleted cannot hold the downstream object any
    more, so it must not keep the row in Deleting — not even on the path where
    the status write keeps the existing list."""
    await _seed(engine, finalizing=[20], pv_ref=True, workers=())
    _mock_clusters(monkeypatch, controller, {20: FakeOps(present=True)}, deleted={20})

    await controller._finalize(1)

    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.finalizing == []


@pytest.mark.asyncio
async def test_a_transient_probe_error_still_keeps_the_cluster_in_finalizing(
    engine, controller, monkeypatch
):
    """The worker pre-check must not swallow a real failure. A cluster that
    answered but whose call failed still holds the object, so the row stays and
    the next round retries it."""
    await _seed(engine)
    ops = {10: FakeOps(present=True)}
    ops[10].delete_persistent_volume_type.side_effect = ConnectionError(
        "connection reset"
    )
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.phase == "Deleting"
    assert pvt.status.finalizing == [10]
