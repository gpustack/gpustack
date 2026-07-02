"""D1: the operator worker subscription must include the Instance GVK (besides
InstanceType) so the gateway pushes Instance change events for the downstream
watcher to consume.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.gpu_instances import gateway, gateway_client
from gpustack.schemas.clusters import (
    ClusterProvider,
    GpuInstanceOptions,
    K8sOptions,
)
from gpustack.server.bus import Event, EventType


@pytest.mark.asyncio
async def test_reconcile_subscribes_instance_and_instancetype_gvk(monkeypatch):
    sub = AsyncMock()
    monkeypatch.setattr(gateway_client, "subscribe_worker", sub)

    cluster = MagicMock()
    cluster.provider = ClusterProvider.Kubernetes
    cluster.id = 7
    cluster.registration_token = "tok"
    cluster.k8s_options = K8sOptions(gpu_instance_options=GpuInstanceOptions())

    await gateway._reconcile(Event(type=EventType.UPDATED, data=cluster))

    sub.assert_awaited_once()
    gvk = sub.call_args.kwargs["gvk"]
    assert ("worker.gpustack.ai", "v1", "InstanceType") in gvk
    assert ("worker.gpustack.ai", "v1", "Instance") in gvk
