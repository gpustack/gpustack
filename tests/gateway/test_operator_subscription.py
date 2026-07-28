"""Guards for the gpustack-operator worker-gateway subscription reconciler.

G1-G3 pin gpustack#5947: a cluster is subscribed only while it has a READY
worker. The operator reaches a subscribed cluster through this server's own
``/v2/clusters/{id}/proxy``, which 503s without a reachable worker, and the
operator then retries its readiness probe forever — one ERROR every ~2s.

D1 (kept): the subscription must carry the Instance GVK besides InstanceType so
the gateway pushes Instance change events for the downstream watcher.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.gpu_instances import gateway, gateway_client
from gpustack.schemas.clusters import (
    ClusterProvider,
    GpuInstanceOptions,
    K8sOptions,
)
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.bus import Event, EventType

CLUSTER_ID = 7


def _cluster(
    cluster_id: int = CLUSTER_ID,
    gpu_instances: bool = True,
    provider: ClusterProvider = ClusterProvider.Kubernetes,
):
    cluster = MagicMock()
    cluster.provider = provider
    cluster.id = cluster_id
    cluster.registration_token = "tok"
    cluster.k8s_options = K8sOptions(
        gpu_instance_options=GpuInstanceOptions() if gpu_instances else None
    )
    return cluster


def _worker(
    cluster_id: int = CLUSTER_ID,
    state: WorkerStateEnum = WorkerStateEnum.READY,
    worker_id: int = 1,
):
    worker = MagicMock()
    worker.id = worker_id
    worker.cluster_id = cluster_id
    worker.state = state
    return worker


def _state_change(
    old: WorkerStateEnum = WorkerStateEnum.NOT_READY,
    new: WorkerStateEnum = WorkerStateEnum.READY,
):
    """The ``changed_fields`` the bus carries when a worker's state moves."""
    return {"state": (old, new)}


@pytest.fixture
def harness(monkeypatch):
    """A reconciler with the gateway client and the READY-worker count stubbed."""
    h = SimpleNamespace(
        subscribe=AsyncMock(),
        unsubscribe=AsyncMock(),
        ready_workers=0,
        queries=0,
        reconciler=gateway.OperatorSubscriptionReconciler(),
    )
    monkeypatch.setattr(gateway_client, "subscribe_worker", h.subscribe)
    monkeypatch.setattr(gateway_client, "unsubscribe_worker", h.unsubscribe)

    async def count_ready_workers(cluster_id: int) -> int:
        h.queries += 1
        return h.ready_workers

    monkeypatch.setattr(gateway, "_count_ready_workers", count_ready_workers)
    return h


@pytest.mark.asyncio
async def test_eligible_cluster_without_ready_worker_is_not_subscribed(harness):
    """G1: an empty cluster must not be subscribed — that is the reported flood."""
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster())
    )

    harness.subscribe.assert_not_awaited()
    harness.unsubscribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_cluster_with_ready_worker_subscribes_once(harness):
    """G1: a cluster that already has a READY worker subscribes on replay, once."""
    harness.ready_workers = 1

    for event_type in (EventType.CREATED, EventType.UPDATED, EventType.UPDATED):
        await harness.reconciler._reconcile_cluster(
            Event(type=event_type, data=_cluster())
        )

    harness.subscribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_first_ready_worker_subscribes_once(harness):
    """G2: reachability arrives as a Worker event — no Cluster event ever fires."""
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster())
    )
    harness.subscribe.assert_not_awaited()

    harness.ready_workers = 1
    await harness.reconciler._reconcile_worker(
        Event(type=EventType.UPDATED, data=_worker(), changed_fields=_state_change())
    )
    harness.subscribe.assert_awaited_once()

    # A repeat state report and a second worker joining add no further calls.
    await harness.reconciler._reconcile_worker(
        Event(type=EventType.UPDATED, data=_worker(), changed_fields=_state_change())
    )
    await harness.reconciler._reconcile_worker(
        Event(type=EventType.CREATED, data=_worker(worker_id=2))
    )
    harness.subscribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_subscription_carries_instance_and_instancetype_gvk(harness):
    """D1: both GVKs must be requested, otherwise Instance events never arrive."""
    harness.ready_workers = 1

    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.UPDATED, data=_cluster())
    )

    harness.subscribe.assert_awaited_once()
    gvk = harness.subscribe.call_args.kwargs["gvk"]
    assert ("worker.gpustack.ai", "v1", "InstanceType") in gvk
    assert ("worker.gpustack.ai", "v1", "Instance") in gvk


@pytest.mark.asyncio
async def test_last_ready_worker_lost_unsubscribes_once(harness):
    """G3: losing the last reachable worker unsubscribes exactly once."""
    harness.ready_workers = 1
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster())
    )

    harness.ready_workers = 0
    await harness.reconciler._reconcile_worker(
        Event(
            type=EventType.UPDATED,
            data=_worker(state=WorkerStateEnum.UNREACHABLE),
            changed_fields=_state_change(
                WorkerStateEnum.READY, WorkerStateEnum.UNREACHABLE
            ),
        )
    )
    harness.unsubscribe.assert_awaited_once()

    await harness.reconciler._reconcile_worker(
        Event(type=EventType.DELETED, data=_worker())
    )
    harness.unsubscribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_remaining_ready_worker_keeps_subscription(harness):
    """G3: one worker leaving READY while another stays is not a transition."""
    harness.ready_workers = 2
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster())
    )

    harness.ready_workers = 1
    await harness.reconciler._reconcile_worker(
        Event(type=EventType.DELETED, data=_worker(worker_id=2))
    )

    harness.unsubscribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_cluster_unsubscribes_once(harness):
    """G3: deleting a cluster unsubscribes once; repeats add nothing."""
    harness.ready_workers = 1
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster())
    )

    for _ in range(2):
        await harness.reconciler._reconcile_cluster(
            Event(type=EventType.DELETED, data=_cluster())
        )

    harness.unsubscribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_never_subscribed_cluster_is_not_unsubscribed(harness):
    """G3: deleting an empty cluster must not call the gateway at all."""
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster())
    )
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.DELETED, data=_cluster())
    )

    harness.subscribe.assert_not_awaited()
    harness.unsubscribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_cluster_losing_gpu_instance_options_unsubscribes(harness):
    """An eligible cluster that drops GPU instances is unsubscribed."""
    harness.ready_workers = 1
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster())
    )

    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.UPDATED, data=_cluster(gpu_instances=False))
    )

    harness.unsubscribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_of_unknown_cluster_is_ignored(harness):
    """A worker whose cluster was never eligible never triggers a subscription."""
    harness.ready_workers = 1

    await harness.reconciler._reconcile_worker(
        Event(
            type=EventType.UPDATED,
            data=_worker(cluster_id=99),
            changed_fields=_state_change(),
        )
    )

    harness.subscribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_dying_watcher_takes_the_other_down(harness, monkeypatch):
    """Both streams are needed to decide a subscription, so one dying must not leave the
    other running: a half-dead reconciler keeps the process up and subscribes nothing.
    """
    worker_cancelled = asyncio.Event()

    async def failing_clusters(**kwargs):
        raise RuntimeError("cluster stream died")
        yield  # unreachable, but makes this an async generator

    async def idle_workers(**kwargs):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            worker_cancelled.set()
            raise
        yield  # unreachable, but makes this an async generator

    monkeypatch.setattr(gateway.Cluster, "subscribe", failing_clusters)
    monkeypatch.setattr(gateway.Worker, "subscribe", idle_workers)

    with pytest.raises(RuntimeError):
        await harness.reconciler.start()

    # The cancellation is joined before start() returns, so the survivor cannot still
    # be running — nor be garbage-collected while pending — once the caller is back.
    assert worker_cancelled.is_set()


@pytest.mark.asyncio
async def test_failed_unsubscribe_is_retried_on_heartbeat(harness):
    """A deleted cluster sends no further event, so a failed unsubscribe has no other
    chance to be retried — and the operator would keep proxying the deleted cluster."""
    harness.ready_workers = 1
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster())
    )

    harness.unsubscribe.side_effect = RuntimeError("worker gateway is unreachable")
    with pytest.raises(RuntimeError):
        await harness.reconciler._reconcile_cluster(
            Event(type=EventType.DELETED, data=_cluster())
        )

    harness.unsubscribe.side_effect = None
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.HEARTBEAT, data=None)
    )
    assert harness.unsubscribe.await_count == 2

    # Once it succeeds the retry stops.
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.HEARTBEAT, data=None)
    )
    assert harness.unsubscribe.await_count == 2


@pytest.mark.asyncio
async def test_failed_subscribe_is_retried_on_heartbeat(harness):
    """A cluster that is already eligible and reachable emits no further transition, so
    a failed subscribe has no other chance to be retried — and the operator would never
    learn about a cluster it should be serving."""
    harness.ready_workers = 1
    harness.subscribe.side_effect = RuntimeError("worker gateway is unreachable")
    with pytest.raises(RuntimeError):
        await harness.reconciler._reconcile_cluster(
            Event(type=EventType.CREATED, data=_cluster())
        )

    # The worker keeps flushing its status, but an event that does not move its state is
    # dropped, so nothing in the worker stream retries either.
    await harness.reconciler._reconcile_worker(
        Event(type=EventType.UPDATED, data=_worker())
    )
    assert harness.subscribe.await_count == 1

    harness.subscribe.side_effect = None
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.HEARTBEAT, data=None)
    )
    assert harness.subscribe.await_count == 2

    # Once it succeeds the retry stops.
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.HEARTBEAT, data=None)
    )
    assert harness.subscribe.await_count == 2


@pytest.mark.asyncio
async def test_one_failing_retry_does_not_starve_the_others(harness):
    """The heartbeat retries every stale cluster under one lock, in id order, so a
    cluster that keeps failing must not stop the rest from being retried."""
    harness.ready_workers = 1
    for cluster_id in (1, 2):
        await harness.reconciler._reconcile_cluster(
            Event(type=EventType.CREATED, data=_cluster(cluster_id=cluster_id))
        )

    harness.unsubscribe.side_effect = RuntimeError("worker gateway is unreachable")
    for cluster_id in (1, 2):
        with pytest.raises(RuntimeError):
            await harness.reconciler._reconcile_cluster(
                Event(type=EventType.DELETED, data=_cluster(cluster_id=cluster_id))
            )

    async def only_cluster_1_keeps_failing(cluster_id, *args, **kwargs):
        if cluster_id == "1":
            raise RuntimeError("worker gateway is unreachable")

    harness.unsubscribe.reset_mock()
    harness.unsubscribe.side_effect = only_cluster_1_keeps_failing
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.HEARTBEAT, data=None)
    )

    assert [c.args[0] for c in harness.unsubscribe.await_args_list] == ["1", "2"]


@pytest.mark.asyncio
async def test_worker_status_flush_does_not_query_the_database(harness):
    """Every posting worker is republished every few seconds, so an event that does not
    move a worker's state must not cost a query — and must not be a transition."""
    harness.ready_workers = 2
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster())
    )
    harness.queries = 0

    await harness.reconciler._reconcile_worker(
        Event(type=EventType.UPDATED, data=_worker(state=WorkerStateEnum.UNREACHABLE))
    )

    assert harness.queries == 0
    harness.unsubscribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_kubernetes_cluster_is_ignored(harness):
    """Only Kubernetes-provider clusters are handled by the operator gateway."""
    harness.ready_workers = 1

    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.CREATED, data=_cluster(provider=ClusterProvider.Docker))
    )

    harness.subscribe.assert_not_awaited()
    harness.unsubscribe.assert_not_awaited()
