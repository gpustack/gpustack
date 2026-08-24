"""Converge the operator settings GPUStack manages onto each GPU Service cluster.

The operator seeds every ``Setting`` from ``GPUSTACK_<UPPER_SNAKE_NAME>`` on its
first deploy and never overwrites a stored value afterwards, so editing an
already-registered cluster's environment is inert — the ``Setting`` itself has
to be patched. The two paths do not replace each other: the rendered environment
covers the window before the operator has ever run (so a cluster never derives
instance types the administrator disabled), and this reconciler covers
everything after.

Delivery is a background reconciler rather than an inline patch inside
``PUT /v2/clusters/{id}`` because of registration itself: the administrator
configures a cluster *before* applying its manifest, so at save time there is no
operator to talk to. An inline patch would silently do nothing exactly when the
values matter most, or would make the cluster uneditable while it is
unreachable. Here a failure only leaves work stale, and stale work is retried.

Runs leader-only, unlike the gateway subscription in ``gateway.py``: that one
feeds each server's own operator subprocess, while this one writes into the
cluster, and one writer is enough.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

from gpustack.config.config import Config
from gpustack.gpu_instances.cluster_apis import ClusterOps

# Reachability and the bus's id-only DELETED payload are both read exactly as
# the operator subscription reads them. Shared rather than copied so the two
# reconcilers cannot drift apart on what "reachable" means, or on which events
# carry a model at all.
from gpustack.gpu_instances.gateway import count_ready_workers, deleted_cluster_id
from gpustack.schemas.clusters import Cluster, ClusterProvider, is_gpu_service_cluster
from gpustack.schemas.principals import PLATFORM_PRINCIPAL_NAME
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.bus import Event, EventType, event_field

logger = logging.getLogger(__name__)

_SOURCE = "gpustack_operator_settings"

# One entry per knob GPUStack can manage: the ``GpuInstanceOptions`` field, its
# serialized camelCase spelling, and the operator ``Setting`` it mirrors. The
# names are the operator's; ``gpu_instances_access_static_address`` keeps its
# legacy field name because renaming it would break every existing client.
_MANAGED_SETTINGS = (
    (
        "gpu_instances_access_static_address",
        "gpuInstancesAccessStaticAddress",
        "instance-access-static-address",
    ),
    (
        "gpu_instance_type_derived_from_node",
        "gpuInstanceTypeDerivedFromNode",
        "instance-type-derived-from-node",
    ),
    (
        "gpu_instance_type_mixed_on_node",
        "gpuInstanceTypeMixedOnNode",
        "instance-type-mixed-on-node",
    ),
)

SETTING_REQUEST_TIMEOUT = 15
"""How long one setting's read or patch may take before it counts as failed.

``ClusterOps`` sets no request timeout and ``kubernetes_asyncio`` builds a bare
``aiohttp.ClientTimeout()`` when none is given, so a call that gets no answer
waits forever. Convergence is driven from a watcher task and from the heartbeat,
both of which walk their clusters one at a time — so a single cluster whose proxy
accepts the connection and then goes silent (its workers still READY, the tunnel
behind them a black hole) would stop every other cluster's event and retry for
the lifetime of the process.

Bounding it here rather than on the client keeps the blast radius to this
reconciler: a timeout is just another failed convergence, reported once and
retried on the next heartbeat.
"""

RESYNC_HEARTBEAT_INTERVAL = 20
"""How many heartbeats apart the level-triggered resync runs, as a multiple of
the bus's 15s idle heartbeat — so roughly five minutes.

Stale work is retried on *every* heartbeat; this is the slower sweep that
re-reads clusters already converged, so an administrator's ``kubectl`` edit of
one of the three managed settings is repaired instead of leaving the cluster
silently disagreeing with what GPUStack shows. It is a divisor on the cost of
that guarantee: each sweep is one read per managed knob per GPU Service cluster,
over the cluster proxy.
"""


async def reconcile_gpustack_operator_settings(config: Config):
    """Watch Cluster and Worker events and keep the operator settings converged."""
    await OperatorSettingsReconciler(config).start()


@dataclass
class _Target:
    """What one cluster needs and what it takes to reach it.

    Compared as a whole to decide whether a cluster event changed anything: a
    rotated registration token or a moved system namespace invalidates a
    previous convergence just as a new desired value does.
    """

    registration_token: Optional[str]
    system_namespace: Optional[str]
    settings: Dict[str, str] = field(default_factory=dict)


class OperatorSettingsReconciler:
    """Keeps each GPU Service cluster's managed operator settings at the value
    its cluster row asks for.

    Two event streams feed one decision, as in
    :class:`~gpustack.gpu_instances.gateway.OperatorSubscriptionReconciler`:
    ``Cluster`` carries the desired values and the registration token, ``Worker``
    carries reachability — and no Cluster event fires when a cluster becomes
    reachable, so the desired values alone would strand a cluster that was empty
    when the administrator saved.

    Convergence is level-triggered in both directions. A cluster whose write
    failed stays in ``_pending`` and is retried on every heartbeat; a cluster
    that succeeded is re-read every :data:`RESYNC_HEARTBEAT_INTERVAL`
    heartbeats, which is what repairs an external edit of a managed setting.
    Only the settings GPUStack manages are ever touched: an unset knob is not
    read and not written, so an administrator's ``kubectl`` edit of anything
    else cannot be reverted.
    """

    def __init__(self, config: Config):
        self._config = config
        # Every cluster with at least one managed knob. A cluster absent here is
        # one this reconciler has nothing to say about — Model Service, no
        # managed knob, or not Kubernetes at all.
        self._targets: Dict[int, _Target] = {}
        # Clusters whose desired values are not known to be converged yet.
        self._pending: Set[int] = set()
        # ``(cluster id, setting name)`` currently failing, so a persistent
        # failure is reported when it starts and when it ends instead of on
        # every retry.
        self._failing: Set[Tuple[int, str]] = set()
        self._heartbeats = 0
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
            # already resolves every target's reachability from the database, so
            # replaying each worker row would only repeat that. A worker event
            # that lands before its cluster is known is dropped, and recovered
            # by the cluster heartbeat's retry.
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
            # gather leaves the other watcher running when one raises. Both are
            # needed to decide a convergence, so a half-dead reconciler would
            # keep the process up while silently converging nothing. The
            # cancellation is joined so the survivor is stopped, not merely
            # asked to stop, before this returns.
            for watcher in watchers:
                watcher.cancel()
            await asyncio.gather(*watchers, return_exceptions=True)

    async def _watch(self, name, stream, reconcile):
        async for event in stream:
            try:
                await reconcile(event)
            except Exception as e:
                logger.exception(
                    f"Failed to reconcile the operator settings "
                    f"on a {name} event: {e}"
                )

    async def _reconcile_cluster(self, event: Event):
        if event.type == EventType.HEARTBEAT:
            await self._on_heartbeat()
            return

        cluster: Cluster = event.data
        if cluster is None:
            return

        # A DELETED event can arrive as the raw row id rather than a Cluster
        # (see :func:`~gpustack.gpu_instances.gateway.deleted_cluster_id`). The
        # id is all this path needs — a deletion is a forget either way — and
        # reading any other attribute off that dict would raise, be swallowed by
        # the watcher, and strand the target's registration token for the
        # lifetime of the process.
        if isinstance(cluster, dict):
            cluster_id = deleted_cluster_id(event)
            if cluster_id is not None:
                async with self._lock:
                    self._forget(cluster_id)
            return

        if cluster.provider != ClusterProvider.Kubernetes:
            return

        async with self._lock:
            target = None if event.type == EventType.DELETED else _target_of(cluster)
            if target is None:
                self._forget(cluster.id)
                return

            if self._targets.get(cluster.id) != target:
                self._targets[cluster.id] = target
                self._pending.add(cluster.id)
            stale = cluster.id in self._pending

        if stale and await count_ready_workers(cluster.id) > 0:
            await self._converge(cluster.id)

    async def _reconcile_worker(self, event: Event):
        worker: Worker = event.data
        # Same id-only payload as the cluster path above, for the same reason
        # (see :func:`~gpustack.gpu_instances.gateway.deleted_cluster_id`).
        # Unlike there the id alone is no use: this reconciler is keyed on the
        # worker's cluster, which a deleted row cannot name. Nothing is lost
        # here though -- unlike its counterpart in gateway.py, this reconciler
        # only ever acts on a worker becoming READY, and returns on DELETE a
        # few lines below regardless.
        cluster_id = event_field(worker, "cluster_id")
        if cluster_id is None:
            return

        if event.type == EventType.UPDATED and "state" not in (
            event.changed_fields or {}
        ):
            # Every worker still posting status is republished every few
            # seconds. Only a state change can make a cluster reachable, so
            # anything else is dropped before it can cost a call.
            return

        if event.type == EventType.DELETED or worker.state != WorkerStateEnum.READY:
            # A worker leaving READY can only make a cluster unreachable, and
            # unreachable work is already pending and retried on the heartbeat.
            return

        async with self._lock:
            stale = cluster_id in self._pending

        if stale:
            await self._converge(cluster_id)

    async def _on_heartbeat(self):
        """Retry stale work, and periodically re-read what already converged.

        The sweep is serial, so one cluster cannot be allowed to hold it: each
        is retried on its own so a raise cannot starve the rest, and every
        round-trip inside is bounded by :data:`SETTING_REQUEST_TIMEOUT` so
        silence cannot either.
        """
        async with self._lock:
            self._heartbeats += 1
            resync = self._heartbeats % RESYNC_HEARTBEAT_INTERVAL == 0
            cluster_ids = sorted(self._targets if resync else self._pending)

        for cluster_id in cluster_ids:
            try:
                if await count_ready_workers(cluster_id) > 0:
                    await self._converge(cluster_id)
            except Exception as e:
                logger.error(
                    f"Failed to retry the operator settings "
                    f"of cluster {cluster_id}: {e}"
                )

    def _forget(self, cluster_id: int):
        """Drop everything about a cluster this reconciler no longer owns — it
        was deleted, switched to Model Service, or stopped managing any knob."""
        self._targets.pop(cluster_id, None)
        self._pending.discard(cluster_id)
        self._failing = {key for key in self._failing if key[0] != cluster_id}

    async def _converge(self, cluster_id: int):
        """Bring one cluster's managed settings in line with its row.

        Each setting is converged on its own so a failing one cannot cost its
        siblings their write. The cluster leaves ``_pending`` only when every
        one of them is known good, which is what makes a partial failure retry.

        Runs **outside** ``_lock``: it is one proxy round-trip per managed knob,
        each bounded by :data:`SETTING_REQUEST_TIMEOUT`, so holding the lock
        across it would let a single unreachable cluster block every event this
        reconciler handles — and every other cluster's retry — for as long as
        that budget lasts. Callers therefore snapshot under the lock
        and call this after releasing it, which means the cluster may be
        forgotten in between; that is what the ``None`` check below is for. The
        writes are idempotent patches, so overlapping passes over one cluster
        cost an extra round-trip and nothing else.

        The same window is why the cluster is only cleared from ``_pending``
        when its target is still the one this pass carried: an edit that landed
        mid-flight has already re-marked it, and clearing that would drop the
        newer values until the next resync.
        """
        target = self._targets.get(cluster_id)
        if target is None:
            return
        converged = True
        async with ClusterOps(
            server_api_port=self._config.get_api_port(),
            cluster_id=cluster_id,
            cluster_registration_token=target.registration_token,
            # Setting is system-namespaced, so the Org namespace this identifier
            # derives never reaches the wire; the constructor requires one anyway.
            cluster_owner_principal_identifier=PLATFORM_PRINCIPAL_NAME,
            system_namespace=target.system_namespace,
        ) as ops:
            for name, desired in target.settings.items():
                try:
                    ok = await asyncio.wait_for(
                        self._converge_one(ops, cluster_id, name, desired),
                        SETTING_REQUEST_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    # Reported by hand: a TimeoutError stringifies to nothing.
                    ok = False
                    self._report_failure(
                        cluster_id,
                        name,
                        ops.system_namespace,
                        f"it did not answer within {SETTING_REQUEST_TIMEOUT}s",
                    )
                except Exception as e:
                    ok = False
                    self._report_failure(cluster_id, name, ops.system_namespace, e)
                converged = converged and ok
        async with self._lock:
            if converged and self._targets.get(cluster_id) == target:
                self._pending.discard(cluster_id)

    async def _converge_one(
        self, ops: ClusterOps, cluster_id: int, name: str, desired: str
    ) -> bool:
        """Patch one setting when it differs, and report whether it is now good.

        ``spec.value`` is write-only and reads back as ``{}``, so the comparison
        is against ``status.value`` and the write goes to ``spec.value`` —
        comparing against the spec would diff against nothing and patch on every
        pass.

        That comparison assumes none of :data:`_MANAGED_SETTINGS` is a sensitive
        setting, which holds for all three: a sensitive one reads back as the
        literal ``"(sensitive)"`` rather than its value, so it could never
        compare equal and every resync would rewrite it. The failure mode is
        write amplification, not a loop — the patch still succeeds and clears
        ``_pending`` — so should the operator ever mark one of the three
        sensitive, this is where to short-circuit it.
        """
        observed = await ops.read_setting(name)
        if observed is None:
            self._report_failure(
                cluster_id, name, ops.system_namespace, "it does not exist"
            )
            return False

        if (observed.get("status") or {}).get("value") == desired:
            self._report_recovery(cluster_id, name)
            return True

        if await ops.patch_setting_value(name, desired) is None:
            # A 404 on a name the read just resolved means the write went
            # somewhere else; never read it as "nothing to do".
            self._report_failure(
                cluster_id, name, ops.system_namespace, "the patch found it absent"
            )
            return False

        logger.info(
            "Set operator setting %s to %r in namespace %s of cluster %s.",
            name,
            desired,
            ops.system_namespace,
            cluster_id,
        )
        self._report_recovery(cluster_id, name)
        return True

    def _report_failure(self, cluster_id: int, name: str, namespace: str, reason: Any):
        """Report a setting that could not be converged — once when it breaks,
        then quietly, so one broken cluster neither floods the log every
        heartbeat nor disappears from it during a long outage."""
        message = (
            "Failed to converge operator setting %s in namespace %s of cluster %s: %s"
        )
        args = (name, namespace, cluster_id, reason)
        key = (cluster_id, name)
        if key in self._failing:
            logger.debug(message, *args)
            return
        self._failing.add(key)
        logger.warning(message, *args)

    def _report_recovery(self, cluster_id: int, name: str):
        key = (cluster_id, name)
        if key not in self._failing:
            return
        self._failing.discard(key)
        logger.info(
            "Operator setting %s of cluster %s converged again.", name, cluster_id
        )


def _target_of(cluster: Cluster) -> Optional[_Target]:
    """What this reconciler owes the cluster, or ``None`` when it owes nothing."""
    settings = _desired_settings(cluster)
    if not settings:
        return None
    return _Target(
        registration_token=cluster.registration_token,
        # The operator's own namespace, which ClusterOps resolves to
        # ``gpustack-system`` when the cluster did not name one.
        system_namespace=_field(cluster.k8s_options, "namespace"),
        settings=settings,
    )


def _desired_settings(cluster: Cluster) -> Dict[str, str]:
    """The operator settings GPUStack manages for this cluster, on-the-wire.

    Empty for a Model Service cluster, and for a GPU Service cluster whose knobs
    are all unset — in both cases nothing is read and nothing is written, which
    is what keeps an unmanaged setting safe from this reconciler (AC4.4).
    """
    if not is_gpu_service_cluster(cluster):
        return {}

    options = _field(cluster.k8s_options, "gpu_instance_options", "gpuInstanceOptions")
    desired: Dict[str, str] = {}
    for snake, camel, setting in _MANAGED_SETTINGS:
        value = _field(options, snake, camel)
        # ``None`` — the field absent from the persisted JSON — is the only
        # unmanaged state. ``False`` and ``""`` are values an administrator
        # asked for, and asserting them is the whole point of the feature.
        if value is None:
            continue
        desired[setting] = _wire_value(value)
    return desired


def _field(source: Any, snake: str, camel: Optional[str] = None) -> Any:
    """Read one field off a value that may be a model or the raw dict the bus
    replays — nested ``pydantic_column_type`` is not re-validated there — while
    tolerating both serialized key spellings (snake from ``model_dump``, camel
    from an API/UI submission). Mirrors ``is_gpu_service_k8s_options``: the raw
    shape is read directly rather than parsed back into a model, so schema drift
    cannot turn a watch tick into a ``ValidationError``."""
    if isinstance(source, dict):
        value = source.get(snake)
        return value if value is not None else source.get(camel or snake)
    return getattr(source, snake, None)


def _wire_value(value: Any) -> str:
    """A ``Setting`` value is a string on the wire, so a bool knob has to go out
    in the operator's spelling rather than Python's ``True`` / ``False``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
