"""Existing clusters are told when they still relay inference through a worker.

New clusters default to direct proxy mode, but existing clusters keep the mode
they were on. The startup advisory is how a deployment learns that switching to
direct needs the instance port range reachable from the gateway.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.clusters import Cluster
from gpustack.schemas.config import (
    ModelInstanceProxyModeEnum,
    PredefinedConfigNoDefaults,
)
from gpustack.server.server import Server
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.mock import mock_async_session


def _worker(worker, *, cluster_id=None, proxy_mode):
    worker.cluster_id = cluster_id
    worker.proxy_mode = proxy_mode
    return worker


def _run_advisory(workers, clusters):
    return (
        patch(
            "gpustack.server.server.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.server.Worker.all_by_fields",
            AsyncMock(return_value=workers),
        ),
        patch(
            "gpustack.server.server.Cluster.all_by_fields",
            AsyncMock(return_value=clusters),
        ),
    )


@pytest.mark.asyncio
async def test_advisory_names_the_clusters_still_on_worker_proxy_mode(caplog):
    caplog.set_level(logging.WARNING)
    relayed = _worker(
        linux_nvidia_1_4090_24gx1(),
        cluster_id=1,
        proxy_mode=ModelInstanceProxyModeEnum.WORKER,
    )
    direct = _worker(
        linux_nvidia_2_4080_16gx2(),
        cluster_id=2,
        proxy_mode=ModelInstanceProxyModeEnum.DIRECT,
    )
    worker_patch, cluster_patch, session_patch = _run_advisory(
        [relayed, direct],
        [
            Cluster(id=1, name="relayed-cluster"),
            Cluster(id=2, name="direct-cluster"),
        ],
    )

    with worker_patch, cluster_patch, session_patch:
        await Server(SimpleNamespace(), None)._warn_worker_proxy_mode()

    assert "relayed-cluster" in caplog.text
    assert "40000-40063" in caplog.text
    assert "41000-41999" in caplog.text
    assert "direct-cluster" not in caplog.text


@pytest.mark.asyncio
async def test_advisory_is_silent_when_every_worker_is_direct(caplog):
    caplog.set_level(logging.WARNING)
    direct = _worker(
        linux_nvidia_1_4090_24gx1(),
        cluster_id=1,
        proxy_mode=ModelInstanceProxyModeEnum.DIRECT,
    )
    worker_patch, cluster_patch, session_patch = _run_advisory(
        [direct], [Cluster(id=1, name="direct-cluster")]
    )

    with worker_patch, cluster_patch, session_patch:
        await Server(SimpleNamespace(), None)._warn_worker_proxy_mode()

    assert "proxy mode" not in caplog.text


@pytest.mark.asyncio
async def test_advisory_counts_a_legacy_null_proxy_mode_as_worker(caplog):
    """Rows created before the column existed have no value and relay anyway."""
    caplog.set_level(logging.WARNING)
    legacy = _worker(linux_nvidia_1_4090_24gx1(), cluster_id=1, proxy_mode=None)
    worker_patch, cluster_patch, session_patch = _run_advisory(
        [legacy], [Cluster(id=1, name="legacy-cluster")]
    )

    with worker_patch, cluster_patch, session_patch:
        await Server(SimpleNamespace(), None)._warn_worker_proxy_mode()

    assert "legacy-cluster" in caplog.text


@pytest.mark.asyncio
async def test_advisory_reads_an_unreported_mode_from_the_cluster_default(caplog):
    caplog.set_level(logging.WARNING)
    legacy = _worker(linux_nvidia_1_4090_24gx1(), cluster_id=1, proxy_mode=None)
    cluster = Cluster(
        id=1,
        name="worker-cluster",
        worker_config=PredefinedConfigNoDefaults(
            proxy_mode=ModelInstanceProxyModeEnum.WORKER
        ),
    )
    worker_patch, cluster_patch, session_patch = _run_advisory([legacy], [cluster])

    with worker_patch, cluster_patch, session_patch:
        await Server(SimpleNamespace(), None)._warn_worker_proxy_mode()

    assert "worker-cluster" in caplog.text


@pytest.mark.asyncio
async def test_advisory_tells_unclustered_workers_how_to_switch(caplog):
    """There is no cluster worker configuration to point an unclustered worker at."""
    caplog.set_level(logging.WARNING)
    unclustered = _worker(
        linux_nvidia_1_4090_24gx1(),
        cluster_id=None,
        proxy_mode=ModelInstanceProxyModeEnum.WORKER,
    )
    worker_patch, cluster_patch, session_patch = _run_advisory([unclustered], [])

    with worker_patch, cluster_patch, session_patch:
        await Server(SimpleNamespace(), None)._warn_worker_proxy_mode()

    assert "--proxy-mode direct" in caplog.text
    assert "cluster's worker configuration" not in caplog.text


@pytest.mark.asyncio
async def test_advisory_reports_the_cluster_port_overrides(caplog):
    caplog.set_level(logging.WARNING)
    relayed = _worker(
        linux_nvidia_1_4090_24gx1(),
        cluster_id=1,
        proxy_mode=ModelInstanceProxyModeEnum.WORKER,
    )
    cluster = Cluster(
        id=1,
        name="custom-ports",
        worker_config=PredefinedConfigNoDefaults(
            service_port_range="45000-45063",
            ray_port_range="46000-46999",
        ),
    )
    worker_patch, cluster_patch, session_patch = _run_advisory([relayed], [cluster])

    with worker_patch, cluster_patch, session_patch:
        await Server(SimpleNamespace(), None)._warn_worker_proxy_mode()

    assert "45000-45063" in caplog.text
    assert "46000-46999" in caplog.text
    assert "40000-40063" not in caplog.text


@pytest.mark.asyncio
async def test_advisory_only_reads_live_rows():
    workers_query = AsyncMock(return_value=[])
    clusters_query = AsyncMock(return_value=[])
    with (
        patch(
            "gpustack.server.server.async_session",
            return_value=mock_async_session(),
        ),
        patch("gpustack.server.server.Worker.all_by_fields", workers_query),
        patch("gpustack.server.server.Cluster.all_by_fields", clusters_query),
    ):
        await Server(SimpleNamespace(), None)._warn_worker_proxy_mode()

    assert workers_query.await_args.kwargs["fields"] == {"deleted_at": None}
    assert clusters_query.await_args.kwargs["fields"] == {"deleted_at": None}
