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
from gpustack.server.bus import Event, EventType, event_field, resolve_event_id
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
        if cluster is None:
            return

        # A DELETED event can arrive as the raw row id rather than a Cluster
        # (see :func:`deleted_cluster_id`). The id is all this path needs — the
        # cluster is gone either way — and reading ``provider`` off that dict
        # would raise, be swallowed by the watcher, and leave the operator
        # proxying a deleted cluster forever: the retry sweep only unsubscribes
        # what left ``_eligible``, which is exactly what the raise skipped.
        if isinstance(cluster, dict):
            cluster_id = deleted_cluster_id(event)
            if cluster_id is not None:
                async with self._lock:
                    self._eligible.pop(cluster_id, None)
                    await self._unsubscribe(cluster_id)
            return

        if cluster.provider != ClusterProvider.Kubernetes:
            return

        async with self._lock:
            if event.type == EventType.DELETED or not _has_gpu_instances(cluster):
                self._eligible.pop(cluster.id, None)
                await self._unsubscribe(cluster.id)
                return

            self._eligible[cluster.id] = cluster.registration_token
            if cluster.id in self._subscribed:
                return
            if await count_ready_workers(cluster.id) > 0:
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
                    if await count_ready_workers(cluster_id) > 0:
                        await self._subscribe(cluster_id)
                except Exception as e:
                    logger.error(f"Failed to retry subscribing {cluster_id}: {e}")

    async def _reconcile_worker(self, event: Event):
        worker: Worker = event.data
        # Same id-only payload as the cluster path above, for the same reason
        # (see :func:`deleted_cluster_id`) -- but here the id is not enough:
        # this reconciler is keyed on the worker's cluster, which a deleted
        # row can no longer name.
        #
        # Known gap, and the heartbeat does not close it: the retry sweep only
        # reconciles _subscribed against _eligible, so it never revisits a
        # cluster that is still eligible but has just lost its last READY
        # worker. That is exactly the transition this event carries, so
        # dropping it leaves this instance proxying an unreachable cluster
        # until the cluster itself leaves _eligible. Every instance runs this
        # reconciler over its own _subscribed set, so the instance that served
        # the delete fixes only itself.
        cluster_id = event_field(worker, "cluster_id")
        if cluster_id is None:
            logger.warning(
                f"Worker {resolve_event_id(event)} {event.type} not reconciled: "
                f"the event carries only an id, so its cluster is unknown; a "
                f"subscription to it may be left in place"
            )
            return

        if event.type == EventType.UPDATED and "state" not in (
            event.changed_fields or {}
        ):
            # Every worker still posting status is republished every few seconds. Only
            # a state change can move a cluster in or out of reachability, so anything
            # else is dropped before it can cost a query.
            return

        async with self._lock:
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
            if await count_ready_workers(cluster_id) == 0:
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


async def count_ready_workers(cluster_id: int) -> int:
    """How many of a cluster's workers are READY — its reachability.

    Public because ``settings.py``'s reconciler decides reachability the same
    way and from the same query: a cluster with no READY worker has no reachable
    proxy, so every call would 503. Shared rather than copied so the two cannot
    drift apart on what "reachable" means.
    """
    async with async_session() as session:
        return await Worker.count_by_fields(
            session, {"cluster_id": cluster_id, "state": WorkerStateEnum.READY}
        )


def deleted_cluster_id(event: Event) -> Optional[int]:
    """The row id a cluster DELETED event carries when its payload never became
    a ``Cluster``, or ``None`` when the event is anything else.

    The bus enriches a DELETED event from the change-detector cache, and that
    cache is empty for clusters — the topic is not preloaded (see
    ``Server._preload_change_detector_cache``) — so the event is routed carrying
    nothing but ``{"id": ...}``. Every reconciler watching clusters therefore
    has to read the id off a raw dict, and a deletion is all the id is needed
    for; ``None`` for any other event type on such a payload, which has no model
    to read and nothing safe to do.
    """
    if event.type != EventType.DELETED:
        return None
    return resolve_event_id(event)
