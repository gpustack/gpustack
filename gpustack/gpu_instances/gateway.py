"""Subscribe Kubernetes-provider clusters to the gpustack-operator worker gateway.

Runs on every server instance (not leader-only) so each server's in-process
``gateway_client`` keeps the operator subprocess it spawned in sync with the
current set of clusters.

A cluster is subscribed only while it has at least one READY worker. The
operator reaches a subscribed cluster through this server's own
``/v2/clusters/{id}/proxy``, which fails with 503 "No reachable workers" while
the cluster is empty; the operator's readiness probe then retries forever and
every probe is logged as a server error (gpustack#5947).
"""

import asyncio
import logging
from typing import Dict, Optional, Set

from gpustack.gpu_instances import gateway_client
from gpustack.schemas.clusters import Cluster, ClusterProvider, is_gpu_service_cluster
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

_SOURCE = "gpustack_operator_subscription"

_GVK = [
    ("worker.gpustack.ai", "v1", "InstanceType"),
    ("worker.gpustack.ai", "v1", "Instance"),
]


async def reconcile_gpustack_operator_subscription():
    """Watch Cluster and Worker events and keep the operator gateway in sync."""
    await OperatorSubscriptionReconciler().start()


class OperatorSubscriptionReconciler:
    """Keeps the operator gateway subscribed to exactly the reachable clusters.

    Two event streams feed one decision: ``Cluster`` carries eligibility and the
    registration token, ``Worker`` carries reachability. Both are needed — a
    Kubernetes cluster is created already READY and worker registration only
    writes the cluster row when its state is *not* READY, so no Cluster event
    ever fires when a cluster becomes reachable.

    Calls are emitted on state transitions only: the pinned operator starts one
    readiness goroutine per subscribe, so a repeated subscribe multiplies work
    inside the operator.
    """

    def __init__(self):
        # Cluster ids that should be subscribed once reachable, mapped to the
        # registration token to subscribe with — a Worker event carries the
        # cluster id but not the token.
        self._eligible: Dict[int, Optional[str]] = {}
        self._subscribed: Set[int] = set()
        # Both watchers mutate the state above across await points.
        self._lock = asyncio.Lock()

    async def start(self):
        watchers = [
            asyncio.create_task(
                self._watch(
                    "cluster",
                    Cluster.subscribe(source=_SOURCE),
                    self._reconcile_cluster,
                )
            ),
            # ``replay_existing=False``: the Cluster stream's own startup replay
            # already resolves every eligible cluster's reachability from the
            # database, so replaying each worker row would only repeat that. A
            # worker event that lands before its cluster is known is dropped, and
            # recovered by the cluster heartbeat's retry.
            asyncio.create_task(
                self._watch(
                    "worker",
                    Worker.subscribe(source=_SOURCE, replay_existing=False),
                    self._reconcile_worker,
                )
            ),
        ]
        try:
            await asyncio.gather(*watchers)
        finally:
            # gather leaves the other watcher running when one raises. Both are needed
            # to decide a subscription, so a half-dead reconciler would keep the process
            # up while silently subscribing nothing. The cancellation is joined so the
            # survivor is stopped, not merely asked to stop, before this returns.
            for watcher in watchers:
                watcher.cancel()
            await asyncio.gather(*watchers, return_exceptions=True)

    async def _watch(self, name, stream, reconcile):
        async for event in stream:
            try:
                await reconcile(event)
            except Exception as e:
                logger.exception(
                    f"Failed to reconcile gpustack-operator subscription "
                    f"on a {name} event: {e}"
                )

    async def _reconcile_cluster(self, event: Event):
        if event.type == EventType.HEARTBEAT:
            await self._retry_stale_subscriptions()
            return

        cluster: Cluster = event.data
        if cluster is None or cluster.provider != ClusterProvider.Kubernetes:
            return

        async with self._lock:
            if event.type == EventType.DELETED or not _has_gpu_instances(cluster):
                self._eligible.pop(cluster.id, None)
                await self._unsubscribe(cluster.id)
                return

            self._eligible[cluster.id] = cluster.registration_token
            if cluster.id in self._subscribed:
                return
            if await _count_ready_workers(cluster.id) > 0:
                await self._subscribe(cluster.id)

    async def _retry_stale_subscriptions(self):
        """Retry the gateway calls that failed, in both directions.

        A failed call leaves the operator out of sync with no event to recover from:
        a deleted cluster produces no further event, and a cluster that is already
        eligible and reachable produces no further transition either. The heartbeat
        is their only second chance — without it the operator keeps proxying a
        cluster that is gone, or never learns about one it should be serving.

        Each cluster is retried on its own, so one that keeps failing cannot starve
        the rest of the set.
        """
        async with self._lock:
            for cluster_id in sorted(self._subscribed - set(self._eligible)):
                try:
                    await self._unsubscribe(cluster_id)
                except Exception as e:
                    logger.error(f"Failed to retry unsubscribing {cluster_id}: {e}")

            # Reachability is only known from the database, so this costs one count
            # per eligible cluster that is not subscribed — the empty ones — per
            # heartbeat. That is the price of recovering a lost subscribe.
            for cluster_id in sorted(set(self._eligible) - self._subscribed):
                try:
                    if await _count_ready_workers(cluster_id) > 0:
                        await self._subscribe(cluster_id)
                except Exception as e:
                    logger.error(f"Failed to retry subscribing {cluster_id}: {e}")

    async def _reconcile_worker(self, event: Event):
        worker: Worker = event.data
        if worker is None or worker.cluster_id is None:
            return

        if event.type == EventType.UPDATED and "state" not in (
            event.changed_fields or {}
        ):
            # Every worker still posting status is republished every few seconds. Only
            # a state change can move a cluster in or out of reachability, so anything
            # else is dropped before it can cost a query.
            return

        async with self._lock:
            cluster_id = worker.cluster_id
            if cluster_id not in self._eligible:
                return

            if (
                event.type != EventType.DELETED
                and worker.state == WorkerStateEnum.READY
            ):
                if cluster_id not in self._subscribed:
                    await self._subscribe(cluster_id)
                return

            # A worker leaving READY only matters to a subscribed cluster, and
            # only when it was the last reachable one — so the count is queried
            # for that case alone, keeping worker status events cheap.
            if cluster_id not in self._subscribed:
                return
            if await _count_ready_workers(cluster_id) == 0:
                await self._unsubscribe(cluster_id)

    async def _subscribe(self, cluster_id: int):
        await gateway_client.subscribe_worker(
            str(cluster_id), self._eligible[cluster_id], gvk=_GVK
        )
        # Recorded after the call so a failure is retried on the next event.
        self._subscribed.add(cluster_id)
        logger.info(f"Subscribed cluster {cluster_id} to the operator gateway.")

    async def _unsubscribe(self, cluster_id: int):
        if cluster_id not in self._subscribed:
            return
        await gateway_client.unsubscribe_worker(str(cluster_id))
        self._subscribed.discard(cluster_id)
        logger.info(f"Unsubscribed cluster {cluster_id} from the operator gateway.")


def _has_gpu_instances(cluster: Cluster) -> bool:
    # Over the bus the ``k8s_options`` JSON column can arrive as a plain dict
    # (nested pydantic_column_type isn't re-validated on replay). The schema-level
    # predicate reads that raw shape directly — both key spellings — so the dict no
    # longer has to be re-validated back into a model here just to read one field.
    return is_gpu_service_cluster(cluster)


async def _count_ready_workers(cluster_id: int) -> int:
    async with async_session() as session:
        return await Worker.count_by_fields(
            session, {"cluster_id": cluster_id, "state": WorkerStateEnum.READY}
        )
