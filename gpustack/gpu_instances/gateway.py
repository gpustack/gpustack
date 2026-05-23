"""Subscribe Kubernetes-provider clusters to the gpustack-operator worker gateway.

Runs on every server instance (not leader-only) so each server's in-process
``gateway_client`` keeps the operator subprocess it spawned in sync with the
current set of clusters.
"""

import logging

from gpustack.gpu_instances import gateway_client
from gpustack.schemas.clusters import Cluster, ClusterProvider
from gpustack.server.bus import Event, EventType

logger = logging.getLogger(__name__)


async def reconcile_gpustack_operator_subscription():
    """Watch Cluster events and (un)subscribe Kubernetes clusters on the operator gateway."""
    async for event in Cluster.subscribe(source="gpustack_operator_subscription"):
        if event.type == EventType.HEARTBEAT:
            continue
        try:
            await _reconcile(event)
        except Exception as e:
            logger.error(f"Failed to reconcile gpustack-operator subscription: {e}")


async def _reconcile(event: Event):
    cluster: Cluster = event.data
    if cluster is None or cluster.provider != ClusterProvider.Kubernetes:
        return
    if event.type == EventType.DELETED:
        await gateway_client.unsubscribe_worker(str(cluster.id))
    else:
        await gateway_client.subscribe_worker(
            str(cluster.id),
            cluster.registration_token,
        )
