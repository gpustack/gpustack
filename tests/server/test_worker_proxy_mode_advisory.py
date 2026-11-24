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
from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.server.server import Server
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.mock import mock_async_session


def _worker(worker, *, cluster_id, proxy_mode):
    worker.cluster_id = cluster_id
    worker.proxy_mode = proxy_mode
    return worker


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

    with (
        patch(
            "gpustack.server.server.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.server.Worker.all",
            AsyncMock(return_value=[relayed, direct]),
        ),
        patch(
            "gpustack.server.server.Cluster.all",
            AsyncMock(
                return_value=[
                    Cluster(id=1, name="relayed-cluster"),
                    Cluster(id=2, name="direct-cluster"),
                ]
            ),
        ),
    ):
        await Server(SimpleNamespace(), None)._warn_worker_proxy_mode()

    assert "relayed-cluster" in caplog.text
    assert "40000-40063" in caplog.text
    assert "direct-cluster" not in caplog.text


@pytest.mark.asyncio
async def test_advisory_is_silent_when_every_worker_is_direct(caplog):
    caplog.set_level(logging.WARNING)
    direct = _worker(
        linux_nvidia_1_4090_24gx1(),
        cluster_id=1,
        proxy_mode=ModelInstanceProxyModeEnum.DIRECT,
    )

    with (
        patch(
            "gpustack.server.server.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.server.Worker.all",
            AsyncMock(return_value=[direct]),
        ),
        patch(
            "gpustack.server.server.Cluster.all",
            AsyncMock(return_value=[Cluster(id=1, name="direct-cluster")]),
        ),
    ):
        await Server(SimpleNamespace(), None)._warn_worker_proxy_mode()

    assert "proxy mode" not in caplog.text
