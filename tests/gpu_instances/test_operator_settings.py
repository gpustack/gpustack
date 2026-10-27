"""Guards for the operator ``Setting`` client and its convergence reconciler.

The operator seeds each ``Setting`` from ``GPUSTACK_<UPPER_SNAKE_NAME>`` on its
first deploy and never overwrites a stored value afterwards, so editing a
registered cluster's environment is inert and the ``Setting`` itself has to be
patched. Three facts measured against a live cluster shape everything below and
are pinned here so a regression cannot pass:

* ``spec`` reads back as ``{}`` — ``spec.value`` is write-only and
  ``status.value`` is the only observable value. The fake below therefore always
  answers ``spec={}``, so an implementation that compared ``spec`` would diff
  against nothing forever and patch on every tick.
* ``Setting`` has no Creater (a create answers 405), so the path is patch-only.
* A wrong namespace answers 404, which ``_patch_spec`` swallows into ``None`` —
  a mis-namespaced reconciler would loop forever writing nothing. ``None`` from
  a read or a patch is a failure here, never "absent, fine".
"""

import asyncio
import http
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from kubernetes_asyncio import client

from gpustack.gpu_instances import settings
from gpustack.gpu_instances.cluster_apis import ClusterOps
from gpustack.schemas.clusters import (
    ClusterProvider,
    GpuInstanceOptions,
    K8sOptions,
)
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.bus import Event, EventType

CLUSTER_ID = 7

# The operator's setting names and the default system namespace, written out
# instead of imported from the module under test: an assertion that reads the
# same constant the code reads would mirror a regression rather than catch it.
DERIVED = "instance-type-derived-from-node"
MIXED = "instance-type-mixed-on-node"
ADDRESS = "instance-access-static-address"
DEFAULT_NAMESPACE = "gpustack-system"

_MISSING = object()


def _cluster(
    gpu_instance_options=_MISSING,
    cluster_id: int = CLUSTER_ID,
    namespace=None,
    provider: ClusterProvider = ClusterProvider.Kubernetes,
    k8s_options=_MISSING,
):
    """A cluster row as the bus delivers it.

    ``k8s_options`` may be handed in as a raw dict: the JSON column is not
    re-validated on replay, so the reconciler has to read both shapes.
    """
    cluster = MagicMock()
    cluster.id = cluster_id
    cluster.provider = provider
    cluster.registration_token = "tok"
    if k8s_options is _MISSING:
        options = (
            GpuInstanceOptions(gpu_instance_type_derived_from_node=False)
            if gpu_instance_options is _MISSING
            else gpu_instance_options
        )
        k8s_options = K8sOptions(namespace=namespace, gpu_instance_options=options)
    cluster.k8s_options = k8s_options
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


class _FakeOps:
    """A ``ClusterOps`` double serving one in-memory ``Setting`` catalog.

    Mirrors the two shapes that matter: a read answers ``spec={}`` plus the
    value under ``status``, and an unknown name answers ``None`` the way a 404
    does.
    """

    def __init__(self, harness, **kwargs):
        self._h = harness
        self.kwargs = kwargs
        self.cluster_id = kwargs["cluster_id"]
        # ClusterOps' own fallback, mirrored so the namespace on the wire can
        # be asserted rather than the argument that produced it.
        self.system_namespace = kwargs.get("system_namespace") or DEFAULT_NAMESPACE
        self.closed = False
        harness.ops.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def read_setting(self, name):
        self._h.reads.append((self.cluster_id, name))
        # Lets a test park one cluster's read mid-flight, the way an
        # unreachable proxy does — deterministically, without a sleep.
        stall = self._h.stall.get(self.cluster_id)
        if stall is not None:
            stall["entered"].set()
            await stall["release"].wait()
        if self._h.read_error is not None:
            raise self._h.read_error
        value = self._h.remote.get(name, _MISSING)
        if value is _MISSING:
            return None
        return {
            "apiVersion": "gpustack.ai/v1",
            "kind": "Setting",
            "metadata": {"name": name, "namespace": self.system_namespace},
            "spec": {},
            "status": {"value": value, "editable": True, "sensitive": False},
        }

    async def patch_setting_value(self, name, value):
        self._h.patches.append((self.cluster_id, name, value))
        if name in self._h.failing_names:
            raise _api_exception(http.HTTPStatus.INTERNAL_SERVER_ERROR)
        if self._h.patch_returns_none:
            return None
        self._h.remote[name] = value
        return {"metadata": {"name": name}, "spec": {"value": value}}


def _api_exception(status: int) -> client.exceptions.ApiException:
    return client.exceptions.ApiException(status=status, reason="boom")


@pytest.fixture
def harness(monkeypatch):
    """A reconciler whose cluster client and reachability query are stubbed."""
    config = MagicMock()
    config.get_api_port.return_value = 80

    h = SimpleNamespace(
        ops=[],
        reads=[],
        patches=[],
        # The operator's own defaults, as a freshly deployed catalog reports them.
        remote={DERIVED: "true", MIXED: "true", ADDRESS: ""},
        read_error=None,
        stall={},
        failing_names=set(),
        patch_returns_none=False,
        ready_workers=1,
        reconciler=settings.OperatorSettingsReconciler(config),
    )
    monkeypatch.setattr(settings, "ClusterOps", lambda **kwargs: _FakeOps(h, **kwargs))

    async def count_ready_workers(cluster_id: int) -> int:
        return h.ready_workers

    monkeypatch.setattr(settings, "count_ready_workers", count_ready_workers)
    return h


async def _cluster_event(harness, cluster, event_type=EventType.CREATED):
    await harness.reconciler._reconcile_cluster(Event(type=event_type, data=cluster))


async def _heartbeats(harness, count: int = 1):
    for _ in range(count):
        await harness.reconciler._reconcile_cluster(
            Event(type=EventType.HEARTBEAT, data=None)
        )


def _names(entries):
    """The setting names touched, whatever the entry arity."""
    return {entry[1] for entry in entries}


#
# The three knobs: what is managed, and what is left alone
#


@pytest.mark.asyncio
async def test_unmanaged_knobs_are_never_read_or_written(harness):
    """AC4.4: an unset knob is not GPUStack's, so it is not even looked at —
    the operator catalog is administered by ``kubectl`` too, and reverting an
    administrator's edit is the failure this rule exists to prevent."""
    await _cluster_event(
        harness,
        _cluster(GpuInstanceOptions(gpu_instance_type_derived_from_node=False)),
    )

    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]
    assert _names(harness.reads) == {DERIVED}
    assert MIXED not in _names(harness.reads) | _names(harness.patches)
    assert ADDRESS not in _names(harness.reads) | _names(harness.patches)


@pytest.mark.asyncio
async def test_false_and_blank_are_managed_values(harness):
    """Only ``None`` is unmanaged: ``False`` and ``""`` are values an
    administrator asked for, and a bool goes out as the operator's string."""
    harness.remote[ADDRESS] = "10.0.0.1"
    await _cluster_event(
        harness,
        _cluster(
            GpuInstanceOptions(
                gpu_instance_type_derived_from_node=False,
                gpu_instance_type_mixed_on_node=True,
                gpu_instances_access_static_address="",
            )
        ),
    )

    # ``mixed_on_node=True`` already matches the catalog, so it is read and left.
    assert sorted(harness.patches) == [
        (CLUSTER_ID, ADDRESS, ""),
        (CLUSTER_ID, DERIVED, "false"),
    ]
    assert _names(harness.reads) == {DERIVED, MIXED, ADDRESS}


@pytest.mark.asyncio
async def test_status_value_is_what_desired_is_compared_against(harness):
    """AC4.2: ``spec`` reads back as ``{}`` and ``status.value`` carries the
    value, so an equal setting must produce no write at all. An implementation
    comparing ``spec`` would diff against ``{}`` and patch here."""
    harness.remote[DERIVED] = "false"

    await _cluster_event(harness, _cluster())

    assert harness.reads == [(CLUSTER_ID, DERIVED)]
    assert harness.patches == []


@pytest.mark.asyncio
async def test_drift_patches_exactly_once(harness):
    """The catalog says ``true``, the cluster row says ``false``: one patch, and
    no repeat while the desired value is unchanged."""
    await _cluster_event(harness, _cluster())
    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]

    # A re-published row and the heartbeats before the next resync add nothing.
    await _cluster_event(harness, _cluster(), event_type=EventType.UPDATED)
    await _heartbeats(harness, settings.RESYNC_HEARTBEAT_INTERVAL - 1)

    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]


@pytest.mark.asyncio
async def test_a_desired_value_edit_converges(harness):
    """The administrator's edit is the whole point: a new desired value on an
    already converged cluster is patched through."""
    await _cluster_event(harness, _cluster())

    await _cluster_event(
        harness,
        _cluster(GpuInstanceOptions(gpu_instance_type_derived_from_node=True)),
        event_type=EventType.UPDATED,
    )

    assert harness.patches == [
        (CLUSTER_ID, DERIVED, "false"),
        (CLUSTER_ID, DERIVED, "true"),
    ]


#
# Which clusters are touched at all
#


@pytest.mark.asyncio
async def test_model_service_cluster_is_never_touched(harness):
    """A cluster with no ``gpu_instance_options`` is registered for Model
    Service; it has no operator settings GPUStack owns, so no client is even
    built for it."""
    await _cluster_event(harness, _cluster(None))
    await _heartbeats(harness, settings.RESYNC_HEARTBEAT_INTERVAL)

    assert harness.ops == []
    assert harness.reads == []
    assert harness.patches == []


@pytest.mark.asyncio
async def test_gpu_service_cluster_without_a_managed_knob_is_not_contacted(harness):
    """A GPU Service cluster that manages nothing costs no call either."""
    await _cluster_event(harness, _cluster(GpuInstanceOptions()))
    await _heartbeats(harness, settings.RESYNC_HEARTBEAT_INTERVAL)

    assert harness.ops == []


@pytest.mark.asyncio
async def test_non_kubernetes_cluster_is_ignored(harness):
    """Docker and cloud clusters have no operator to converge."""
    await _cluster_event(harness, _cluster(provider=ClusterProvider.Docker))

    assert harness.ops == []


@pytest.mark.asyncio
async def test_switching_to_model_service_stops_convergence(harness):
    """After a purpose switch the settings stop being GPUStack's, so later drift
    in the cluster's own catalog is left alone."""
    await _cluster_event(harness, _cluster())
    await _cluster_event(harness, _cluster(None), event_type=EventType.UPDATED)

    harness.remote[DERIVED] = "true"
    await _heartbeats(harness, settings.RESYNC_HEARTBEAT_INTERVAL)

    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]


@pytest.mark.asyncio
async def test_deleted_cluster_is_forgotten(harness):
    """A deleted cluster keeps no work behind it."""
    harness.ready_workers = 0
    await _cluster_event(harness, _cluster())
    await _cluster_event(harness, _cluster(), event_type=EventType.DELETED)

    harness.ready_workers = 1
    await _heartbeats(harness, settings.RESYNC_HEARTBEAT_INTERVAL)

    assert harness.ops == []


@pytest.mark.asyncio
async def test_an_unreachable_cluster_does_not_block_another(harness):
    """A cluster hung mid-converge must not hold up anyone else.

    ``ClusterOps`` sets no request timeout of its own, so a converge against an
    unreachable cluster runs until ``SETTING_REQUEST_TIMEOUT`` gives up on it —
    once per managed knob. If the reconciler held ``_lock`` across that, one
    cluster would stall every other cluster's convergence and every event this
    reconciler handles, which is the isolation ``_on_heartbeat``'s docstring
    promises.
    """
    other = CLUSTER_ID + 1
    harness.ready_workers = 0
    await _cluster_event(harness, _cluster())
    await _cluster_event(harness, _cluster(cluster_id=other))
    harness.ready_workers = 1

    harness.stall[CLUSTER_ID] = {
        "entered": asyncio.Event(),
        "release": asyncio.Event(),
    }
    hung = asyncio.create_task(
        harness.reconciler._reconcile_worker(
            Event(type=EventType.CREATED, data=_worker())
        )
    )
    await asyncio.wait_for(harness.stall[CLUSTER_ID]["entered"].wait(), timeout=2)

    # The second cluster gets its writes while the first is still parked.
    await asyncio.wait_for(
        harness.reconciler._reconcile_worker(
            Event(type=EventType.CREATED, data=_worker(cluster_id=other))
        ),
        timeout=2,
    )
    assert other in {patch[0] for patch in harness.patches}

    harness.stall[CLUSTER_ID]["release"].set()
    await asyncio.wait_for(hung, timeout=2)


@pytest.mark.asyncio
async def test_a_silent_proxy_times_out_instead_of_hanging(harness, monkeypatch):
    """A call that is accepted and never answered is bounded, not waited on.

    ``kubernetes_asyncio`` builds a bare ``aiohttp.ClientTimeout()`` when no
    request timeout is given — every field ``None``, so nothing ever expires.
    A cluster whose workers are still READY while the tunnel behind them is a
    black hole would therefore park the watcher task forever, taking every
    other cluster's events and retries with it. The bound turns that into an
    ordinary failed convergence: nothing written, still pending, retried on the
    next heartbeat.
    """
    monkeypatch.setattr(settings, "SETTING_REQUEST_TIMEOUT", 0.05)
    harness.stall[CLUSTER_ID] = {
        "entered": asyncio.Event(),
        "release": asyncio.Event(),
    }

    await asyncio.wait_for(_cluster_event(harness, _cluster()), timeout=2)

    assert harness.patches == []
    assert CLUSTER_ID in harness.reconciler._pending

    harness.stall[CLUSTER_ID]["release"].set()


@pytest.mark.asyncio
async def test_an_update_during_convergence_is_not_swallowed(harness):
    """A cluster edited mid-converge stays pending.

    Convergence runs outside the lock, so a Cluster update can land while it is
    awaiting the proxy. The pass that finishes is carrying the *previous*
    desired values, so clearing the cluster's pending flag on its way out would
    drop the newer edit on the floor — and nothing would notice until the next
    resync, up to five minutes later.
    """
    harness.ready_workers = 0
    await _cluster_event(harness, _cluster())
    harness.ready_workers = 1

    stall = {"entered": asyncio.Event(), "release": asyncio.Event()}
    harness.stall[CLUSTER_ID] = stall
    converging = asyncio.create_task(
        harness.reconciler._reconcile_worker(
            Event(type=EventType.CREATED, data=_worker())
        )
    )
    await asyncio.wait_for(stall["entered"].wait(), timeout=2)

    # A new desired value lands while that pass is still parked. Unreachable
    # for the moment, so the update only records the work rather than racing
    # the pass in flight for it.
    harness.ready_workers = 0
    await _cluster_event(
        harness,
        _cluster(
            gpu_instance_options=GpuInstanceOptions(
                gpu_instance_type_derived_from_node=True
            )
        ),
    )

    stall["release"].set()
    await asyncio.wait_for(converging, timeout=2)

    assert CLUSTER_ID in harness.reconciler._pending


@pytest.mark.asyncio
async def test_deleted_cluster_is_forgotten_from_an_id_only_event(harness):
    """A DELETED event carrying only an id still forgets the cluster.

    In distributed mode the bus hands subscribers ``Event(data={"id": N})``
    whenever its change-detector cache holds no entry for the row
    (``bus.py``: "No cached object, route ID-only event for DELETED"). That is
    the normal state for a freshly elected leader, because ``cluster`` is not
    in ``_preload_change_detector_cache``'s topic list. Reading an attribute
    off that dict raises, ``_watch`` swallows the exception, and the target —
    registration token and all — would then outlive the cluster and be retried
    on every resync.
    """
    harness.ready_workers = 0
    await _cluster_event(harness, _cluster())
    await harness.reconciler._reconcile_cluster(
        Event(type=EventType.DELETED, data={"id": CLUSTER_ID})
    )

    harness.ready_workers = 1
    await _heartbeats(harness, settings.RESYNC_HEARTBEAT_INTERVAL)

    assert harness.ops == []


@pytest.mark.parametrize(
    "options_key, knob_key",
    [
        pytest.param("gpuInstanceOptions", "gpuInstanceTypeMixedOnNode", id="camel"),
        pytest.param(
            "gpu_instance_options", "gpu_instance_type_mixed_on_node", id="snake"
        ),
    ],
)
@pytest.mark.asyncio
async def test_bus_delivered_dict_options_are_understood(
    harness, options_key, knob_key
):
    """``k8s_options`` arrives from the bus as a plain dict — camel from an API
    submission, snake from ``model_dump`` — and both must read the same."""
    k8s_options = {"namespace": "ops-ns", options_key: {knob_key: False}}

    await _cluster_event(harness, _cluster(k8s_options=k8s_options))

    assert harness.patches == [(CLUSTER_ID, MIXED, "false")]
    assert harness.ops[-1].system_namespace == "ops-ns"


@pytest.mark.asyncio
async def test_client_addresses_the_cluster_system_namespace(harness):
    """AC4.1: settings live beside the operator in the cluster's *system*
    namespace, not in the per-Org namespace every other namespaced resource
    uses — and a wrong namespace fails silently, so this is pinned."""
    await _cluster_event(harness, _cluster(namespace="ops-ns"))
    assert harness.ops[-1].system_namespace == "ops-ns"
    assert harness.ops[-1].kwargs["cluster_registration_token"] == "tok"

    # A cluster registered without an explicit namespace falls back to the
    # namespace the manifest renderer defaults to.
    await _cluster_event(harness, _cluster(cluster_id=8))
    assert harness.ops[-1].system_namespace == DEFAULT_NAMESPACE


#
# Reachability, failure and retry
#


@pytest.mark.asyncio
async def test_unreachable_cluster_is_retried_not_dropped(harness):
    """AC4.3: a cluster that is unreachable when the administrator saves must
    converge on its own once it comes back, with no further user action."""
    harness.read_error = _api_exception(http.HTTPStatus.SERVICE_UNAVAILABLE)

    await _cluster_event(harness, _cluster())
    assert harness.patches == []

    await _heartbeats(harness)
    assert len(harness.reads) == 2  # retried, not dropped
    assert harness.patches == []

    harness.read_error = None
    await _heartbeats(harness)
    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]


@pytest.mark.asyncio
async def test_empty_cluster_is_not_contacted_until_a_worker_is_ready(harness):
    """Reachability arrives as a Worker event: the cluster row does not change
    when a worker registers, so no Cluster event ever fires for it."""
    harness.ready_workers = 0
    await _cluster_event(harness, _cluster())
    assert harness.ops == []

    harness.ready_workers = 1
    await harness.reconciler._reconcile_worker(
        Event(
            type=EventType.UPDATED,
            data=_worker(),
            changed_fields={
                "state": (WorkerStateEnum.NOT_READY, WorkerStateEnum.READY)
            },
        )
    )

    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]


@pytest.mark.asyncio
async def test_worker_status_flush_does_not_reconverge(harness):
    """Every posting worker is republished every few seconds; an event that does
    not move its state cannot change reachability, so it must cost nothing."""
    await _cluster_event(harness, _cluster())
    reads = len(harness.reads)

    await harness.reconciler._reconcile_worker(
        Event(type=EventType.UPDATED, data=_worker())
    )

    assert len(harness.reads) == reads


@pytest.mark.asyncio
async def test_a_worker_leaving_ready_costs_nothing(harness):
    """A worker going away can only make a cluster *less* reachable, and a
    cluster that cannot be reached is already stale and already retried."""
    harness.read_error = _api_exception(http.HTTPStatus.SERVICE_UNAVAILABLE)
    await _cluster_event(harness, _cluster())
    reads = len(harness.reads)

    await harness.reconciler._reconcile_worker(
        Event(
            type=EventType.UPDATED,
            data=_worker(state=WorkerStateEnum.UNREACHABLE),
            changed_fields={
                "state": (WorkerStateEnum.READY, WorkerStateEnum.UNREACHABLE)
            },
        )
    )
    await harness.reconciler._reconcile_worker(
        Event(type=EventType.DELETED, data=_worker())
    )

    assert len(harness.reads) == reads


@pytest.mark.asyncio
async def test_a_worker_without_a_cluster_is_ignored(harness):
    """A worker that names no cluster resolves to nothing to converge."""
    await harness.reconciler._reconcile_worker(
        Event(type=EventType.CREATED, data=_worker(cluster_id=None))
    )

    assert harness.ops == []


@pytest.mark.asyncio
async def test_a_silent_404_from_a_patch_is_a_failure(harness):
    """``_patch_spec`` answers ``None`` on a 404 — the shape a wrong namespace
    takes. Treating it as "absent, fine" would loop forever writing nothing, so
    it keeps the cluster stale and is retried."""
    harness.patch_returns_none = True

    await _cluster_event(harness, _cluster())
    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]

    await _heartbeats(harness)
    assert len(harness.patches) == 2

    harness.patch_returns_none = False
    await _heartbeats(harness)
    assert len(harness.patches) == 3
    await _heartbeats(harness)
    assert len(harness.patches) == 3  # converged, so the retry stops


@pytest.mark.asyncio
async def test_an_absent_setting_is_a_failure(harness):
    """A name the catalog does not serve is a misconfiguration, not a no-op:
    the catalog is fixed, so there is nothing to create and nothing to skip."""
    harness.remote.pop(DERIVED)

    await _cluster_event(harness, _cluster())

    assert harness.patches == []
    await _heartbeats(harness)
    assert len(harness.reads) == 2


@pytest.mark.asyncio
async def test_a_failing_setting_does_not_block_the_others(harness):
    """AC4.5: a patch failure is logged and retried; it never wedges the loop,
    and it never costs the sibling knobs their convergence."""
    harness.failing_names = {DERIVED}

    await _cluster_event(
        harness,
        _cluster(
            GpuInstanceOptions(
                gpu_instance_type_derived_from_node=False,
                gpu_instance_type_mixed_on_node=False,
            )
        ),
    )

    # The raising knob did not take its sibling down with it ...
    assert (CLUSTER_ID, MIXED, "false") in harness.patches
    assert harness.remote[MIXED] == "false"

    # ... and the cluster stays stale, so the failing knob is retried.
    await _heartbeats(harness)
    assert len([p for p in harness.patches if p[1] == DERIVED]) == 2
    # The converged sibling is re-read on the retry but not written again.
    assert len([p for p in harness.patches if p[1] == MIXED]) == 1


@pytest.mark.asyncio
async def test_one_failing_cluster_does_not_starve_the_others(harness, monkeypatch):
    """The heartbeat retries every stale cluster under one lock, so a cluster
    that keeps failing must not stop the rest from being retried."""
    harness.ready_workers = 0
    await _cluster_event(harness, _cluster(cluster_id=1))
    await _cluster_event(harness, _cluster(cluster_id=2))

    async def count_ready_workers(cluster_id: int) -> int:
        if cluster_id == 1:
            raise RuntimeError("database is down")
        return 1

    monkeypatch.setattr(settings, "count_ready_workers", count_ready_workers)
    await _heartbeats(harness)

    assert harness.patches == [(2, DERIVED, "false")]


#
# The event streams themselves
#


@pytest.mark.asyncio
async def test_a_failing_event_does_not_kill_the_watcher(harness, caplog):
    """One event that blows up must not take its stream down: the next event
    still has to be reconciled, or the reconciler dies without saying so."""
    seen = []

    async def stream():
        yield Event(type=EventType.CREATED, data=None)
        yield Event(type=EventType.CREATED, data=_cluster())

    async def reconcile(event):
        seen.append(event)
        if event.data is None:
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger=settings.logger.name):
        await harness.reconciler._watch("cluster", stream(), reconcile)

    assert len(seen) == 2
    assert caplog.records


@pytest.mark.asyncio
async def test_a_dying_watcher_takes_the_other_down(harness, monkeypatch):
    """Both streams are needed to decide a convergence, so one dying must not
    leave the other running: a half-dead reconciler keeps the process up while
    converging nothing."""
    worker_watching = asyncio.Event()
    worker_cancelled = asyncio.Event()

    async def failing_clusters(**kwargs):
        # Die only once the other watcher is actually watching. A watcher
        # cancelled before its first step never enters its own body, so without
        # this the evidence below would be a matter of scheduling rather than of
        # the reconciler's behaviour.
        await worker_watching.wait()
        raise RuntimeError("cluster stream died")
        yield  # unreachable, but makes this an async generator

    async def idle_workers(**kwargs):
        worker_watching.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            worker_cancelled.set()
            raise
        yield  # unreachable, but makes this an async generator

    monkeypatch.setattr(settings.Cluster, "subscribe", failing_clusters)
    monkeypatch.setattr(settings.Worker, "subscribe", idle_workers)

    with pytest.raises(RuntimeError):
        await harness.reconciler.start()

    # The cancellation is joined before start() returns, so the survivor cannot
    # still be running — nor be garbage-collected while pending — once the
    # caller is back.
    assert worker_cancelled.is_set()


#
# Level-triggered resync
#


@pytest.mark.asyncio
async def test_external_drift_is_repaired_by_the_periodic_resync(harness):
    """AC4.3: convergence is level-triggered. A ``kubectl`` edit of one of the
    three settings GPUStack manages would otherwise leave the UI and the cluster
    silently disagreeing — which is exactly what F4 exists to prevent."""
    # ~5 minutes at the bus's 15s idle heartbeat: cheap enough to run forever,
    # short enough that a drifted cluster is not left wrong for long.
    assert settings.RESYNC_HEARTBEAT_INTERVAL == 20

    await _cluster_event(harness, _cluster())
    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]

    harness.remote[DERIVED] = "true"  # an administrator's kubectl edit

    await _heartbeats(harness, settings.RESYNC_HEARTBEAT_INTERVAL - 1)
    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]

    await _heartbeats(harness)
    assert harness.patches == [
        (CLUSTER_ID, DERIVED, "false"),
        (CLUSTER_ID, DERIVED, "false"),
    ]


@pytest.mark.asyncio
async def test_the_resync_reads_but_does_not_write_a_converged_cluster(harness):
    """The resync is a comparison, not a write: a cluster still carrying the
    desired value costs one read per managed knob and no patch."""
    await _cluster_event(harness, _cluster())
    reads = len(harness.reads)

    await _heartbeats(harness, settings.RESYNC_HEARTBEAT_INTERVAL)

    assert len(harness.reads) == reads + 1
    assert harness.patches == [(CLUSTER_ID, DERIVED, "false")]


#
# Log volume: a broken cluster must neither flood nor vanish
#


@pytest.mark.asyncio
async def test_a_persistent_failure_is_reported_once_then_quietly(harness, caplog):
    """One WARNING when it breaks, DEBUG while it stays broken, INFO when it
    recovers — a bad cluster must not put four lines a minute in the log, and a
    long outage must not disappear from it either."""
    harness.read_error = _api_exception(http.HTTPStatus.SERVICE_UNAVAILABLE)

    with caplog.at_level(logging.DEBUG, logger=settings.logger.name):
        await _cluster_event(harness, _cluster())
        await _heartbeats(harness, 3)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert DERIVED in warnings[0].getMessage()
        assert DEFAULT_NAMESPACE in warnings[0].getMessage()
        assert [r for r in caplog.records if r.levelno == logging.DEBUG]

        caplog.clear()
        harness.read_error = None
        await _heartbeats(harness)

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    assert [r for r in caplog.records if r.levelno == logging.INFO]


#
# The ClusterOps client the reconciler drives
#


@pytest_asyncio.fixture
async def ops():
    o = ClusterOps(
        server_api_port=1,
        cluster_id=CLUSTER_ID,
        cluster_registration_token="tok",
        cluster_owner_principal_identifier="default",
        system_namespace="ops-ns",
    )
    yield o
    await o.close()


def _fake_crd(monkeypatch, ops):
    crd = MagicMock()
    for name in (
        "get_namespaced_custom_object",
        "patch_namespaced_custom_object",
        "get_cluster_custom_object",
        "patch_cluster_custom_object",
        "create_namespaced_custom_object",
    ):
        setattr(crd, name, AsyncMock(return_value={"ok": True}))
    monkeypatch.setattr(ops, "_crd", lambda: crd)
    return crd


@pytest.mark.asyncio
async def test_read_setting_addresses_the_operator_group(monkeypatch, ops):
    """``Setting`` is ``gpustack.ai/v1``, not the ``worker.gpustack.ai/v1`` every
    other resource here uses, and it is namespaced in the system namespace."""
    crd = _fake_crd(monkeypatch, ops)

    assert await ops.read_setting(DERIVED) == {"ok": True}

    crd.get_namespaced_custom_object.assert_awaited_once_with(
        group="gpustack.ai",
        version="v1",
        plural="settings",
        namespace="ops-ns",
        name=DERIVED,
    )
    crd.get_cluster_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_patch_setting_value_merge_patches_spec_value(monkeypatch, ops):
    """The write goes to ``spec.value`` — ``status`` is the operator's — as a
    merge patch, and never falls through to a create: the catalog is fixed and
    the aggregated apiserver answers a create with 405."""
    crd = _fake_crd(monkeypatch, ops)

    assert await ops.patch_setting_value(DERIVED, "false") == {"ok": True}

    crd.patch_namespaced_custom_object.assert_awaited_once_with(
        group="gpustack.ai",
        version="v1",
        plural="settings",
        namespace="ops-ns",
        name=DERIVED,
        body={"spec": {"value": "false"}},
        _content_type="application/merge-patch+json",
    )
    crd.create_namespaced_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_setting_calls_answer_none_when_absent(monkeypatch, ops):
    """A wrong namespace 404s, and both helpers report it as ``None`` — the
    signal the reconciler has to treat as a failure rather than as nothing to
    do."""
    crd = _fake_crd(monkeypatch, ops)
    crd.get_namespaced_custom_object.side_effect = _api_exception(
        http.HTTPStatus.NOT_FOUND
    )
    crd.patch_namespaced_custom_object.side_effect = _api_exception(
        http.HTTPStatus.NOT_FOUND
    )

    assert await ops.read_setting(DERIVED) is None
    assert await ops.patch_setting_value(DERIVED, "false") is None
