"""Turning a pool's ``replicas`` into Worker rows.

Both cases here are about a create payload that leaves the optional fields
out, which is what an API client writes and what the failure looked like from
outside: a cloud cluster that stays PENDING with no node ever appearing.
"""

import warnings

import pytest
from unittest.mock import AsyncMock, MagicMock

from gpustack.schemas.clusters import (
    Cluster,
    ClusterProvider,
    CloudOptions,
    WorkerPool,
    WorkerPoolCreate,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.controllers import new_workers_from_pool


def _pool(**kwargs) -> WorkerPool:
    cluster = Cluster(id=1, name="c", provider=ClusterProvider.DigitalOcean)
    return WorkerPool(id=1, name="p", cluster=cluster, **kwargs)


@pytest.mark.asyncio
async def test_a_pool_without_a_batch_size_still_provisions(monkeypatch):
    """``batch_size`` is optional, so an omitted one has to read as "no cap".

    Comparing it anyway raised TypeError on ``None <= 0``, which the caller
    swallows as a failed reconcile -- one error line, and then a pool that
    never creates a worker, because every retry reaches the same comparison.
    """
    monkeypatch.setattr(Worker, "all_by_fields", AsyncMock(return_value=[]))

    workers = await new_workers_from_pool(MagicMock(), _pool(replicas=2))

    assert len(workers) == 2
    assert all(worker.state == WorkerStateEnum.PENDING for worker in workers)


@pytest.mark.asyncio
async def test_a_batch_size_still_caps_what_is_in_flight(monkeypatch):
    """The other half: the cap has to keep working where one is set."""
    in_flight = Worker(id=1, name="w", state=WorkerStateEnum.PROVISIONING)
    monkeypatch.setattr(Worker, "all_by_fields", AsyncMock(return_value=[in_flight]))

    assert (
        await new_workers_from_pool(MagicMock(), _pool(replicas=5, batch_size=1)) == []
    )


def test_an_omitted_cloud_options_is_a_model_not_a_dict():
    """A ``default`` is not validated, so ``default={}`` left a raw dict on the
    field and every ``model_dump`` of the input -- which is how both create
    routes build the row -- warned PydanticSerializationUnexpectedValue."""
    pool = WorkerPoolCreate(
        name="p", instance_type="t", os_image="img", image_name="image"
    )

    assert isinstance(pool.cloud_options, CloudOptions)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pool.model_dump()
