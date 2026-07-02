"""E4b: GPUInstancePersistentVolumeTypeController finalizes soft-deleted PVTs.

PV before PVT: a PVT with any referencing PV row waits (the PV->PVT FK is ON
DELETE RESTRICT). Otherwise finalize enumerates every cluster, probes the
cluster-scoped downstream PVT (matched by its owner-folded name), deletes it
where present, and hard-deletes the row once none remain. Cluster enumeration +
downstream ops are mocked; the DB is a real in-memory sqlite.
"""

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
from gpustack.schemas.clusters import Cluster
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

# Owner ORG "acme" -> principal identifier "acme" -> this downstream PVT name.
CLUSTER_NAME = get_persistent_volume_type_name("pvt-1", principal_identifier="acme")


class FakeOps:
    """Stands in for ClusterOps: the two PVT methods the controller touches.
    ``items`` is the cluster-scoped PVT list result."""

    def __init__(self, items=None):
        self.list_persistent_volume_types = AsyncMock(return_value=list(items or []))
        self.delete_persistent_volume_type = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _pvt_item(name=CLUSTER_NAME):
    return {"metadata": {"name": name}}


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        for table in (
            Principal.__table__,
            GPUInstancePersistentVolumeType.__table__,
            GPUInstancePersistentVolume.__table__,
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


async def _seed(engine, *, phase="Deleting", finalizing=None, pv_ref=False):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(Principal(id=1, kind=PrincipalType.ORG, name="acme"))
        status = (
            GPUInstancePersistentVolumeTypeStatus(phase=phase, finalizing=finalizing)
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

    assert await _get_pvt(engine) is None  # hard-deleted
    for o in ops.values():
        o.delete_persistent_volume_type.assert_not_called()


@pytest.mark.asyncio
async def test_present_downstream_records_finalizing_and_requeues(
    engine, controller, monkeypatch
):
    await _seed(engine)
    ops = {10: FakeOps(items=[_pvt_item()]), 20: FakeOps(items=[])}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume_type.assert_awaited_once_with(CLUSTER_NAME)
    ops[20].delete_persistent_volume_type.assert_not_called()
    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.phase == "Deleting"
    assert pvt.status.finalizing == [10]
    assert len(controller._queue._delayed) == 1


@pytest.mark.asyncio
async def test_referencing_pv_blocks_finalize(engine, controller, monkeypatch):
    await _seed(engine, pv_ref=True)
    ops = {10: FakeOps(items=[_pvt_item()])}
    _mock_clusters(monkeypatch, controller, ops)

    await controller._finalize(1)

    ops[10].delete_persistent_volume_type.assert_not_called()  # PV before PVT
    pvt = await _get_pvt(engine)
    assert pvt is not None
    assert pvt.status.phase == "Deleting"
    assert pvt.status.phase_message is not None
    assert len(controller._queue._delayed) == 1


@pytest.mark.asyncio
async def test_probe_ignores_other_type_name(engine, controller, monkeypatch):
    await _seed(engine)
    # A different owner's PVT sharing the cluster → different folded name.
    other = FakeOps(items=[_pvt_item(name="instance-pv-type.other.pvt-1")])
    _mock_clusters(monkeypatch, controller, {10: other})

    await controller._finalize(1)

    other.delete_persistent_volume_type.assert_not_called()
    assert await _get_pvt(engine) is None  # nothing ours downstream → hard-deleted
