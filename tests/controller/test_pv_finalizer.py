"""E4a: GPUInstancePersistentVolumeController finalizes soft-deleted PVs.

Finalize enumerates every cluster, probes the downstream PV CR, deletes it where
present, records the still-holding clusters in ``status.finalizing``, and
hard-deletes the row once none remain. A PV still referenced by an active
GPUInstance waits. The cluster enumeration + downstream ops are mocked; the DB is
a real in-memory sqlite so the ORM queries / delete run for real.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances.controllers import GPUInstancePersistentVolumeController
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.gpu_instances import GPUInstance
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeSpec,
    GPUInstancePersistentVolumeStatus,
)
from gpustack.schemas.principals import Principal, PrincipalType

NAMESPACE = "gpustack-acme"


class FakeOps:
    """Stands in for ClusterOps: an async context manager exposing the two PV
    methods the PV controller touches. ``items`` is the all-namespace list."""

    def __init__(self, items=None, org_namespace: str = NAMESPACE):
        self.org_namespace = org_namespace
        self.list_persistent_volumes = AsyncMock(return_value=list(items or []))
        self.delete_persistent_volume = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _pv_item(name="pv-1", namespace=NAMESPACE):
    return {"metadata": {"name": name, "namespace": namespace}}


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        for table in (
            Principal.__table__,
            GPUInstancePersistentVolume.__table__,
            GPUInstance.__table__,
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


async def _seed(engine, *, phase="Deleting", finalizing=None, instance_ref=False):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(Principal(id=1, kind=PrincipalType.ORG, name="acme"))
        status = (
            GPUInstancePersistentVolumeStatus(phase=phase, finalizing=finalizing)
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
            s.add(
                GPUInstance(
                    id=1,
                    name="gi-1",
                    owner_principal_id=1,
                    cluster_id=10,
                    spec={"type_": "gpu", "image": "busybox"},
                    persistent_volume_id=1,
                )
            )
        await s.commit()


async def _get_pv(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await GPUInstancePersistentVolume.one_by_id(s, 1)


def _mock_clusters(monkeypatch, controller, ops_by_cluster):
    clusters = [SimpleNamespace(id=cid) for cid in ops_by_cluster]
    monkeypatch.setattr(Cluster, "all", AsyncMock(return_value=clusters))
    monkeypatch.setattr(
        controller, "_build_ops", lambda cluster, owner: ops_by_cluster[cluster.id]
    )


@pytest.mark.asyncio
async def test_absent_downstream_hard_deletes_row(engine, controller, monkeypatch):
    await _seed(engine)
    ops = {10: FakeOps(items=[]), 20: FakeOps(items=[])}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    assert await _get_pv(engine) is None  # hard-deleted
    for o in ops.values():
        o.delete_persistent_volume.assert_not_called()


@pytest.mark.asyncio
async def test_present_downstream_records_finalizing_and_requeues(
    engine, controller, monkeypatch
):
    await _seed(engine)
    ops = {10: FakeOps(items=[_pv_item()]), 20: FakeOps(items=[])}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume.assert_awaited_once_with("pv-1")
    ops[20].delete_persistent_volume.assert_not_called()
    pv = await _get_pv(engine)
    assert pv is not None  # retained until downstream clears
    assert pv.status.phase == "Deleting"
    assert pv.status.finalizing == [10]
    assert len(controller._queue._delayed) == 1  # re-probe scheduled


@pytest.mark.asyncio
async def test_active_instance_reference_blocks_finalize(
    engine, controller, monkeypatch
):
    await _seed(engine, instance_ref=True)
    ops = {10: FakeOps(items=[_pv_item()])}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume.assert_not_called()  # not probed while in use
    pv = await _get_pv(engine)
    assert pv is not None
    assert pv.status.phase == "Deleting"
    assert pv.status.phase_message is not None
    assert len(controller._queue._delayed) == 1


@pytest.mark.asyncio
async def test_probe_ignores_other_namespace(engine, controller, monkeypatch):
    await _seed(engine)
    # A same-named PV, but in a different Org's namespace → not ours.
    other = FakeOps(items=[_pv_item(namespace="gpustack-other")])
    _mock_clusters(monkeypatch, controller, {10: other})

    await controller._finalize(1)

    other.delete_persistent_volume.assert_not_called()
    assert await _get_pv(engine) is None  # nothing ours downstream → hard-deleted


@pytest.mark.asyncio
async def test_unchanged_finalizing_skips_db_write(engine, controller, monkeypatch):
    await _seed(engine, finalizing=[10])  # already recorded last round
    ops = {10: FakeOps(items=[_pv_item()])}
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
async def test_non_deleting_row_is_noop(engine, controller, monkeypatch):
    await _seed(engine, phase=None)
    ops = {10: FakeOps(items=[_pv_item()])}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume.assert_not_called()
    assert await _get_pv(engine) is not None  # untouched
