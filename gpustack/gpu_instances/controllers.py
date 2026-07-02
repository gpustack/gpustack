import json
import logging
import asyncio
import re
from typing import Any, Dict, List, Tuple, Optional
from cachetools import TTLCache
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.config.config import (
    Config,
)
from gpustack.gpu_instances import gateway_client
from gpustack.gpu_instances.cluster_apis import ClusterOps
from gpustack.gpu_instances.cluster_apis_util import (
    spec_persistent_volume,
    spec_persistent_volume_type,
    get_persistent_volume_type_name,
    parse_namespace_name,
    principal_namespace_identifier,
)
from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstancePhase,
    GPUInstanceStatus,
    KUBERES_INSTANCE_ID_LABEL,
)
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeStatus,
)
from gpustack.schemas.gpu_instance_persistent_volume_types import (
    GPUInstancePersistentVolumeType,
    GPUInstancePersistentVolumeTypeStatus,
)
from gpustack.schemas.gpu_instance_ssh_public_keys import (
    GPUInstanceSSHPublicKey,
)
from gpustack.schemas.principals import (
    Principal,
    PrincipalType,
    PLATFORM_PRINCIPAL_NAME,
    platform_principal_id,
)
from gpustack.schemas.clusters import (
    Cluster,
)

from gpustack.server.bus import Event, EventType
from gpustack.server.workqueue import WorkEvent, WorkEventType, WorkQueue
from gpustack import envs
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

# A USER principal's namespace identifier is ``user-<principal.id>`` (see
# ``principal_namespace_identifier``); the digit IS the owner_principal_id, so
# mechanism X can skip a DB lookup for personal-scope namespaces.
_USER_NAMESPACE_RE = re.compile(r"^user-(\d+)$")

# Backoff before reconnecting the downstream watch stream after it ends/errors.
_WATCH_RECONNECT_INTERVAL = 5.0


class _InstanceAssetsError(Exception):
    """A worker-side asset (SSH / PVT / PV / Instance) create failed.

    Carries the ``*CreateFailed`` phase the caller should record on the row, so
    the granular per-asset failure phase survives the wrapped-method boundary.
    """

    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase
        self.message = message


class GPUInstanceController:
    """Reconciles ``GPUInstance`` rows against the worker cluster CRDs.

    The Python row is the source of truth; this controller projects each
    ``GPUInstance`` change onto ``worker.gpustack.ai/v1`` CRs via
    :class:`ClusterOps` and writes back the observed phase.

    Two event sources feed one per-keys-serial work queue, drained by a single
    consumer that runs the phase state machine::

        UPSTREAM (DB bus)                    DOWNSTREAM (operator watch stream)
        CREATED -> ADDED (+row)              MODIFIED / DELETED
        UPDATED -> MODIFIED                  MODIFIED + deletionTimestamp
        (spec / phase edits)                   -> DELETED (delete intent)
              |                              else -> MODIFIED   (id-only stub,
              |                                                  never the object)
              +---------------------+----------------------------+
                                    v
                        +-------------------------------+
                        |  WorkQueue                    |
                        |  per-keys serial              |
                        |  coalesce: DELETED-sticky     |
                        +-------------------------------+
                                    |
                                    v   _dispatch -> _process (backoff on error)
                        _reconcile_instance(iid): branch on the DB phase
                          creating / starting / stopping / deleting / observe
                          (unchanged & transitioning -> add_after re-observe)

    Re-observation of a still-transitioning row is an in-memory ``add_after``
    requeue (no DB write); a settled Ready row is left alone and its drift is
    picked up by the downstream watch.
    """

    PHASE_CREATE_FAILED = GPUInstancePhase.CREATE_FAILED
    PHASE_SSH_KEY_CREATE_FAILED = GPUInstancePhase.SSH_KEY_CREATE_FAILED
    PHASE_PV_TYPE_CREATE_FAILED = GPUInstancePhase.PV_TYPE_CREATE_FAILED
    PHASE_PV_CREATE_FAILED = GPUInstancePhase.PV_CREATE_FAILED
    PHASE_DELETING = GPUInstancePhase.DELETING
    PHASE_STOPPING = GPUInstancePhase.STOPPING
    PHASE_STOPPED = GPUInstancePhase.STOPPED
    PHASE_STARTING = GPUInstancePhase.STARTING
    PHASE_UNKNOWN = GPUInstancePhase.UNKNOWN
    PHASE_READY = GPUInstancePhase.READY

    def __init__(self, cfg: Config):
        self._config = cfg
        # Generic work queue keyed by ``(iid,)``. The default coalescer is
        # latest-wins + DELETED-sticky (the consumer always re-fetches the row,
        # so a superseded event's stale snapshot is harmless). Per-keys backoff
        # (``add_rate_limited`` / ``forget``) is driven from ``_process``.
        self._queue: WorkQueue = WorkQueue(coalesce=self._coalesce_events)
        # In-flight per-keys worker tasks, tracked so shutdown can cancel them.
        # The queue guarantees at most one in-flight task per keys.
        self._inflight: Dict[Any, asyncio.Task] = {}
        # Consumes ``_queue`` and fans out one worker task per keys.
        self._dispatch_task: Optional[asyncio.Task] = None
        # Downstream watcher: consumes the operator's Instance watch stream and
        # pushes changes back onto ``_queue`` (leader-only, like this controller).
        self._watch_task: Optional[asyncio.Task] = None
        # Mechanism-X cache: ``namespace -> owner_principal_id`` so the label-absent
        # fallback doesn't hit the DB for every downstream event of a known org.
        self._ns_owner_cache: TTLCache = TTLCache(maxsize=2048, ttl=600)
        # Cadence for re-observing a still-transitioning row via an in-memory
        # requeue (no DB write). Clamped to >= 1s so a misconfigured 0 can't turn
        # ``add_after`` into a busy loop.
        self._transitioning_interval: float = max(
            1, envs.GPU_INSTANCE_TRANSITIONING_REQUEUE_INTERVAL
        )

    async def start(self):
        self._dispatch_task = asyncio.create_task(self._dispatch())
        self._watch_task = asyncio.create_task(self._watch_downstream())
        try:
            async for event in GPUInstance.subscribe(source="gpu_instance_controller"):
                if event.type == EventType.HEARTBEAT or event.data is None:
                    continue
                # A DB ``DELETED`` means the row is already gone — the deleting
                # branch tears the worker side down before hard-deleting it, so
                # there is nothing left to reconcile. ``CREATED`` (->ADDED) and
                # ``UPDATED`` (->MODIFIED) both run the phase state machine, which
                # re-fetches the row and branches on its phase.
                if event.type == EventType.DELETED:
                    continue
                self._enqueue(event)
        finally:
            tasks: List[asyncio.Task] = []
            for task in (self._dispatch_task, self._watch_task):
                if task is not None:
                    task.cancel()
                    tasks.append(task)
            for task in list(self._inflight.values()):
                task.cancel()
                tasks.append(task)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    # ======================================================================= #
    # Work queue — producers, coalesce policy & consumer (shared)
    # ======================================================================= #

    @staticmethod
    def _coalesce_events(existing: WorkEvent, incoming: WorkEvent) -> WorkEvent:
        """Per-keys slot policy: ``DELETED`` is sticky, otherwise latest wins.

        Only invoked when a pending event already occupies the slot. The consumer
        always re-fetches the row, so a superseded event's stale snapshot never
        matters — the only thing worth preserving is the terminal delete intent.
        """
        if existing.type == WorkEventType.DELETED:
            return existing
        return incoming

    @staticmethod
    def _to_work_event(iid: Any, event: Event) -> WorkEvent:
        if event.type == EventType.DELETED:
            wtype = WorkEventType.DELETED
        elif event.type == EventType.CREATED:
            wtype = WorkEventType.ADDED
        else:
            wtype = WorkEventType.MODIFIED
        return WorkEvent(keys=(iid,), type=wtype, object=event)

    def _enqueue(self, event: Event):
        """Map a bus event onto the work queue (coalescing happens in the queue)."""
        iid = event.data.id if event.data is not None else event.id
        if iid is None:
            return
        self._queue.add(self._to_work_event(iid, event))

    def _requeue_after(self, instance: GPUInstance, delay: float) -> None:
        """Re-observe ``instance`` after ``delay`` via an in-memory requeue.

        No DB write, no bus event — just a delayed re-add so a still-transitioning
        row keeps being reconciled until it settles. ``changed_fields`` is empty
        (a re-observation, not a spec change), so it skips the spec-driven SSH
        resync.
        """
        event = Event(type=EventType.UPDATED, data=GPUInstance(id=instance.id))
        self._queue.add_after(self._to_work_event(instance.id, event), delay)

    async def _dispatch(self):
        """Consume the queue and fan out one worker task per keys.

        The queue's per-keys serialization guarantees a keys handed out here is
        not handed out again until ``done`` is called, so distinct keys run
        concurrently while a single keys stays strictly serial.
        """
        while True:
            event = await self._queue.get()
            self._inflight[event.keys] = asyncio.create_task(self._process(event))

    async def _process(self, event: WorkEvent):
        keys = event.keys
        try:
            await self._reconcile(event.object)
            # Success — reset the per-keys backoff counter.
            self._queue.forget(keys)
        except Exception:
            logger.exception(f"Failed to reconcile GPU instance {keys[0]}")
            # Failure — retry with a capped exponential backoff.
            self._queue.add_rate_limited(event)
        finally:
            # done() and the pop run without an intervening await, so the
            # dispatch loop cannot re-hand this keys (and overwrite the entry)
            # between them.
            self._queue.done(keys)
            # Drop the finished task; the returned Task is intentionally discarded.
            _ = self._inflight.pop(keys, None)

    # ======================================================================= #
    # DOWNSTREAM QUEUE — operator watch stream -> upstream trigger
    # ======================================================================= #

    async def _watch_downstream(self):
        """Consume the operator's Instance watch stream and push changes back.

        The reconciler is upstream-driven (DB bus events); this closes the loop
        so a worker-side change (phase drift, CR deleted out of band) flows back
        into the same work queue without a DB-triggered re-read. Runs leader-only
        because the whole controller is. The stream is reconnected on any error.
        """
        while True:
            try:
                async for line in gateway_client.watch_instances():
                    await self._on_downstream_event(line)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "GPU instance downstream watch stream failed; reconnecting"
                )
            # Stream ended or errored — back off briefly before reconnecting.
            await asyncio.sleep(_WATCH_RECONNECT_INTERVAL)

    async def _on_downstream_event(self, line: str):
        """Map one downstream ``WorkerEvent`` line onto an upstream reconcile.

        The downstream object is only a trigger and is never carried (an id-only
        stub is enqueued): the phase-based reconcile re-fetches the row and
        re-reads the worker. A CR that is going away — a ``DELETED`` or a
        ``MODIFIED`` already carrying a ``deletionTimestamp`` — is pushed as an
        upstream ``DELETED`` so the delete intent gets coalesce priority (DELETED
        sticky) over a pending ``MODIFIED``; consumption is still phase-keyed, so
        the type only affects queue ordering.
        """
        try:
            # ``watch_instances`` re-frames each event with a trailing ``\n\n``
            # (see gateway_client._stream); ``json.loads`` strips it.
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Discarding malformed downstream event: %r", line)
            return
        etype = event.get("type")
        if etype not in ("ADDED", "MODIFIED", "DELETED"):
            # Ignore BOOKMARK / ERROR and anything unexpected.
            return
        cr = event.get("object") or {}
        iid = await self._resolve_instance_id(cr)
        if iid is None:
            logger.debug(
                "Downstream event for unresolvable instance CR %s/%s",
                (cr.get("metadata") or {}).get("namespace"),
                (cr.get("metadata") or {}).get("name"),
            )
            return
        deleting = etype == "DELETED" or bool(
            (cr.get("metadata") or {}).get("deletionTimestamp")
        )
        upstream_type = EventType.DELETED if deleting else EventType.UPDATED
        # A re-observation, not a spec edit — empty changed_fields (mirrors
        # _requeue_after) so the reconcile skips the spec-driven SSH resync.
        self._enqueue(Event(type=upstream_type, data=GPUInstance(id=iid)))

    async def _resolve_instance_id(self, cr: dict) -> Optional[int]:
        """Mechanism X: resolve a downstream Instance CR to its upstream id.

        Label-first (``gpustack.ai/instance-id`` is stamped on every CR we
        create) needs no DB, so it is checked before opening a session. The
        fallback reverse-resolves the namespace to an ``owner_principal_id`` and
        looks the row up by the unique ``(owner_principal_id, name)``.
        """
        metadata = cr.get("metadata") or {}
        label = (metadata.get("labels") or {}).get(KUBERES_INSTANCE_ID_LABEL)
        if label is not None:
            try:
                return int(label)
            except (TypeError, ValueError):
                logger.warning("Ignoring non-integer instance-id label %r", label)

        namespace = metadata.get("namespace")
        name = metadata.get("name")
        if not namespace or not name:
            return None
        async with async_session() as session:
            owner_principal_id = await self._resolve_owner_principal_id(
                session, namespace
            )
            if owner_principal_id is None:
                return None
            row = await GPUInstance.first_by_fields(
                session,
                fields={"owner_principal_id": owner_principal_id, "name": name},
            )
            return row.id if row is not None else None

    async def _resolve_owner_principal_id(
        self, session: AsyncSession, namespace: str
    ) -> Optional[int]:
        """Reverse-resolve a ``gpustack-<identifier>`` namespace to its owner.

        ``default`` is the platform ORG; ``user-<id>`` carries the id directly;
        any other identifier is an ORG name looked up (and cached) in the DB.
        """
        if namespace in self._ns_owner_cache:
            return self._ns_owner_cache[namespace]

        identifier = parse_namespace_name(namespace)
        if not identifier:
            return None

        owner_principal_id: Optional[int] = None
        user_match = _USER_NAMESPACE_RE.match(identifier)
        if identifier == PLATFORM_PRINCIPAL_NAME:
            owner_principal_id = platform_principal_id()
        elif user_match is not None:
            owner_principal_id = int(user_match.group(1))
        else:
            principal = await Principal.first_by_fields(
                session,
                fields={"kind": PrincipalType.ORG, "name": identifier},
            )
            owner_principal_id = principal.id if principal is not None else None

        if owner_principal_id is not None:
            self._ns_owner_cache[namespace] = owner_principal_id
        return owner_principal_id

    # ======================================================================= #
    # UPSTREAM QUEUE — DB bus events -> phase state machine
    # ======================================================================= #

    async def _reconcile(self, event: Event):
        instance: GPUInstance = event.data
        if instance is None or instance.id is None:
            return
        # The event is only a trigger: the phase state machine re-fetches the row
        # and branches on its phase. ``changed_fields`` (upstream spec edits only;
        # empty for requeue / downstream triggers) drives the SSH resync.
        await self._reconcile_instance(instance.id, event.changed_fields or {})

    async def _ops_create_instance_and_assets(  # noqa: C901
        self,
        session: AsyncSession,
        fresh: GPUInstance,
        ops: ClusterOps,
        principal_identifier: str,
    ) -> None:
        """Provision a brand-new instance's worker-side assets.

        Creates the referenced SSH public key, persistent volume (type), and the
        Instance CR (create is idempotent via ``ignore_existed``). On any failure
        it raises :class:`_InstanceAssetsError` carrying the matching
        ``*CreateFailed`` phase; the caller records that phase and stops.
        """
        # SSH public key.
        if fresh.spec.ssh_public_keys is not None:
            try:
                data = await self._aggregate_ssh_public_key_data(session, fresh)
                await ops.upsert_ssh_public_key(name=fresh.name, spec={"data": data})
            except Exception as e:
                logger.exception(
                    f"Failed to sync worker-side ssh public key for {fresh.name}"
                )
                raise _InstanceAssetsError(
                    self.PHASE_SSH_KEY_CREATE_FAILED,
                    f"Failed to sync worker-side ssh public key: {e}",
                )

        # Persistent volume (type).
        pv_name: Optional[str] = None
        volume = fresh.spec.volume
        if volume is not None:
            if volume.persistent is not None:
                pv_name = volume.persistent.name
            elif volume.persistent_template is not None:
                pv_name = volume.persistent_template.name
        if pv_name is not None:
            pv = await GPUInstancePersistentVolume.first_by_fields(
                session,
                fields={
                    "owner_principal_id": fresh.owner_principal_id,
                    "name": pv_name,
                },
            )
            if pv is None:
                logger.error(
                    f"Failed to find server-side pv for {fresh.name} with pv name {pv_name}"
                )
                raise _InstanceAssetsError(
                    self.PHASE_PV_CREATE_FAILED,
                    f"Not found server-side persistent volume: {pv_name}",
                )

            pvt_name = pv.spec.type_
            pvt = await GPUInstancePersistentVolumeType.one_by_id(
                session, pv.persistent_volume_type_id
            )
            if pvt is None:
                logger.error(
                    f"Failed to find server-side pv type for {fresh.name} with pvt name {pvt_name}"
                )
                raise _InstanceAssetsError(
                    self.PHASE_PV_TYPE_CREATE_FAILED,
                    f"Not found server-side persistent volume type: {pvt_name}",
                )

            pvt_cluster_name = get_persistent_volume_type_name(
                pvt_name,
                principal_identifier=principal_identifier,
            )
            try:
                await ops.create_persistent_volume_type(
                    name=pvt_cluster_name,
                    spec=spec_persistent_volume_type(
                        pvt, ops.cluster_owner_principal_identifier
                    ),
                )
            except Exception as e:
                logger.exception(
                    f"Failed to create worker-side pv type for {fresh.name}"
                )
                raise _InstanceAssetsError(
                    self.PHASE_PV_TYPE_CREATE_FAILED,
                    f"Failed to create worker-side persistent volume type: {e}",
                )

            try:
                pv_spec = spec_persistent_volume(pv)
                # The worker-side PV references the worker-side PVT name
                # (prefixed with the principal name); overwrite the raw type id.
                pv_spec["type"] = pvt_cluster_name
                await ops.create_persistent_volume(name=pv.name, spec=pv_spec)
            except Exception as e:
                logger.exception(f"Failed to create worker-side pv for {fresh.name}")
                raise _InstanceAssetsError(
                    self.PHASE_PV_CREATE_FAILED,
                    f"Failed to create worker-side persistent volume: {e}",
                )

        # Instance CR.
        try:
            await ops.create_instance(fresh.convert_to_kuberes())
        except Exception as e:
            logger.exception(f"Failed to create worker-side instance for {fresh.name}")
            raise _InstanceAssetsError(
                self.PHASE_CREATE_FAILED,
                f"Failed to create worker-side instance: {e}",
            )

    @staticmethod
    async def _ops_delete_instance_and_assets(ops: ClusterOps, name: str) -> None:
        """Delete the instance's worker-side Instance CR and SSH public key
        (both idempotent — a missing object is a no-op)."""
        await ops.delete_instance(name)
        await ops.delete_ssh_public_key(name)

    async def _reconcile_instance(  # noqa: C901
        self, iid: Any, changed_fields: Dict[str, Tuple[Any, Any]]
    ):
        """Reconcile one GPUInstance by its DB phase (the upstream state machine).

        Re-fetches the row (the trigger event is only a hint), resyncs the SSH
        key on a spec edit, then drives the worker side per phase:

        - creating -> provision assets, then observe
        - starting -> clear ``spec.stop`` (rebuild if the CR is gone), then observe
        - stopping -> issue stop until the worker reports Stopped
        - deleting -> tear the worker side down, then delete the row
        - ready / not-ready / unknown -> observe (read + merge back)
        - stopped / *failed -> settled, nothing to do
        """
        async with async_session() as session:
            fresh = await GPUInstance.one_by_id(session, iid)
            if fresh is None or fresh.deleted_at is not None:
                return

            built = await self._build_ops(session, fresh)
            if built is None:
                return
            ops, principal_identifier = built
            async with ops:
                # Update referenced SSH public keys if needed.
                if "spec" in changed_fields:
                    try:
                        data = await self._aggregate_ssh_public_key_data(session, fresh)
                        await ops.upsert_ssh_public_key(
                            name=fresh.name,
                            spec={"data": data},
                        )
                    except Exception:
                        logger.exception(
                            f"Failed to sync worker-side ssh public key for {fresh.name}"
                        )

                # --- upstream phase state machine ---
                if fresh.is_creating():
                    await self._reconcile_creating(
                        session,
                        fresh,
                        ops,
                        principal_identifier,
                    )
                elif fresh.is_starting():
                    await self._reconcile_starting(
                        session,
                        fresh,
                        ops,
                        principal_identifier,
                    )
                elif fresh.is_stopping():
                    await self._reconcile_stopping(
                        session,
                        fresh,
                        ops,
                    )
                elif fresh.is_deleting():
                    await self._reconcile_deleting(
                        session,
                        fresh,
                        ops,
                    )
                elif not (fresh.is_stopped() or fresh.is_failed()):
                    # Ready / NotReady / Unknown: observe the worker side.
                    await self._reconcile_observe(
                        session,
                        fresh,
                        ops,
                    )
                # else settled (Stopped / *Failed): nothing to do.

    async def _reconcile_creating(
        self,
        session: AsyncSession,
        fresh: GPUInstance,
        ops: ClusterOps,
        principal_identifier: str,
    ):
        """Provision a brand-new row's worker-side assets, then observe."""
        logger.debug(
            f"GPUInstance {fresh.name} is new, provisioning worker-side assets"
        )
        try:
            await self._ops_create_instance_and_assets(
                session, fresh, ops, principal_identifier
            )
        except _InstanceAssetsError as e:
            await self._write_status(
                session,
                fresh,
                GPUInstanceStatus(
                    phase=e.phase,
                    phase_message=e.message,
                    namespace=ops.org_namespace,
                ),
            )
            return
        # Created — observe. If the CR is not readable yet, ``_reconcile_observe``
        # writes a synthetic Unknown (a non-None phase, so the next pass leaves
        # ``is_creating``) and keeps observing until the CR appears.
        await self._reconcile_observe(session, fresh, ops)

    async def _reconcile_starting(
        self,
        session: AsyncSession,
        fresh: GPUInstance,
        ops: ClusterOps,
        principal_identifier: str,
    ):
        """Clear ``spec.stop`` (rebuild if the CR is gone), then observe.

        ``stop`` keeps the CR alive (worker reports Stopped), so /start must
        patch ``spec.stop`` off; without it the observe below would merge the
        worker's Stopped back and revert the row.
        """
        read = await ops.read_instance(fresh.name)
        if read is None:
            # No CR (a Stopped row whose CR was removed, or a retry before the
            # initial create) — rebuild from scratch, then re-observe.
            try:
                await self._ops_create_instance_and_assets(
                    session, fresh, ops, principal_identifier
                )
            except _InstanceAssetsError as e:
                await self._write_status(
                    session,
                    fresh,
                    GPUInstanceStatus(
                        phase=e.phase,
                        phase_message=e.message,
                        namespace=ops.org_namespace,
                    ),
                )
                return
            self._requeue_after(fresh, self._transitioning_interval)
            return
        try:
            await ops.start_instance(fresh.name)
        except Exception:
            logger.exception(f"Failed to start worker-side instance for {fresh.name}")
            await self._write_phase_message(
                session,
                fresh,
                "Failed to start worker-side instance, will retry",
            )
            raise
        if fresh.merge_from_kuberes(read).phase == self.PHASE_STOPPED:
            # Worker has not picked up the un-stop yet — hold Starting, re-observe.
            self._requeue_after(fresh, self._transitioning_interval)
            return
        await self._db_update_instance_status(session, fresh, read)

    async def _reconcile_stopping(
        self, session: AsyncSession, fresh: GPUInstance, ops: ClusterOps
    ):
        """Issue stop until the worker reports Stopped, then settle to Stopped."""
        read = await ops.read_instance(fresh.name)
        if read is None:
            # CR gone — treat as Stopped (a later /start rebuilds it).
            await self._write_status(
                session,
                fresh,
                GPUInstanceStatus(
                    phase=self.PHASE_STOPPED,
                    phase_message="Worker-side instance not found",
                    namespace=ops.org_namespace,
                ),
            )
            return
        worker_phase = fresh.merge_from_kuberes(read).phase
        if worker_phase == self.PHASE_STOPPED:
            # Worker confirmed Stopped — write the merged status through.
            await self._db_update_instance_status(session, fresh, read)
            return
        if worker_phase != self.PHASE_STOPPING:
            # Worker has not begun stopping yet — (re)issue the stop patch.
            try:
                await ops.stop_instance(fresh.name)
            except Exception:
                logger.exception(
                    f"Failed to stop worker-side instance for {fresh.name}"
                )
                await self._write_phase_message(
                    session,
                    fresh,
                    "Failed to stop worker-side instance, will retry",
                )
                raise
        # Still stopping — hold the phase and re-observe.
        self._requeue_after(fresh, self._transitioning_interval)

    async def _reconcile_deleting(
        self, session: AsyncSession, fresh: GPUInstance, ops: ClusterOps
    ):
        """Tear the worker side down; delete the row once the CR is gone."""
        read = await ops.read_instance(fresh.name)
        if read is None:
            # CR gone — release a template PV opted into release_with_instance,
            # then hard-delete the row.
            await self._release_template_persistent_volume(session, fresh)
            await fresh.delete(session)
            return
        try:
            await self._ops_delete_instance_and_assets(ops, fresh.name)
        except Exception:
            logger.exception(f"Failed to delete worker-side instance for {fresh.name}")
            await self._write_phase_message(
                session,
                fresh,
                "Failed to delete worker-side instance, will retry",
            )
            raise
        # Still deleting — re-observe until the CR is gone.
        self._requeue_after(fresh, self._transitioning_interval)

    async def _reconcile_observe(
        self, session: AsyncSession, fresh: GPUInstance, ops: ClusterOps
    ):
        """Read the worker CR and merge its status back (change-gated).

        A vanished CR settles a fully-Ready row to Stopped (its workload is gone;
        a later /start rebuilds it). A not-yet-Ready row (creating / not-ready /
        unknown) instead keeps observing as Unknown — the CR may simply not be
        visible yet (eventual consistency), so it must not be prematurely
        stopped and stranded once it does appear.
        """
        read = await ops.read_instance(fresh.name)
        if read is None:
            if fresh.is_ready():
                await self._write_status(
                    session,
                    fresh,
                    GPUInstanceStatus(
                        phase=self.PHASE_STOPPED,
                        phase_message="Worker-side instance not found",
                        namespace=ops.org_namespace,
                    ),
                )
                return
            read = {
                "metadata": {"namespace": ops.org_namespace},
                "status": {
                    "phase": self.PHASE_UNKNOWN,
                    "phaseMessage": "Not found in cluster",
                },
            }
        await self._db_update_instance_status(session, fresh, read)

    async def _release_template_persistent_volume(
        self, session: AsyncSession, fresh: GPUInstance
    ):
        """Soft-delete the instance's ``persistent_template`` PV when it opted
        into ``release_with_instance`` (its finalizer then reclaims it). Existing
        PV references and non-opted-in templates are left untouched."""
        volume = fresh.spec.volume
        if volume is None or volume.persistent_template is None:
            return
        if not volume.persistent_template.release_with_instance:
            return
        await self._release_persistent_volume(
            session,
            owner_principal_id=fresh.owner_principal_id,
            name=volume.persistent_template.name,
        )

    async def _db_update_instance_status(
        self, session: AsyncSession, fresh: GPUInstance, downstream: dict
    ):
        """Merge the worker CR status onto the row and write it back — only on a
        real change; a still-transitioning row that did not change is re-observed
        after the transitioning interval. Sole concern: the merge + write."""
        merged = fresh.merge_from_kuberes(downstream)
        await self._write_status(
            session,
            fresh,
            merged,
            requeue_if_transitioning=True,
        )

    async def _write_status(
        self,
        session: AsyncSession,
        fresh: GPUInstance,
        expected: GPUInstanceStatus,
        *,
        requeue_if_transitioning: bool = False,
    ):
        """Persist ``expected`` onto the row — but only on a real change.

        DELETING is sticky: if /delete landed mid-reconcile the DB now reads
        DELETING, so a stale non-DELETING status from this pass is dropped;
        ``session.refresh`` closes the cross-session race. When unchanged and the
        row is still transitioning, re-observe via an in-memory requeue instead
        of writing (no DB write, no bus event).
        """
        await session.refresh(fresh)
        current = fresh.status or GPUInstanceStatus()
        if (
            current.phase == GPUInstancePhase.DELETING
            and expected.phase != GPUInstancePhase.DELETING
        ):
            return
        if self._status_equivalent(current, expected):
            if requeue_if_transitioning and fresh.is_transitioning():
                self._requeue_after(fresh, self._transitioning_interval)
            return
        await fresh.update(
            session,
            source={"status": expected.model_dump(by_alias=True, exclude_none=True)},
        )

    async def _write_phase_message(
        self, session: AsyncSession, fresh: GPUInstance, message: str
    ):
        """Update only ``phase_message`` (keep the current phase) — for a
        transient action failure that will be retried via backoff."""
        base = fresh.status or GPUInstanceStatus()
        await self._write_status(
            session,
            fresh,
            base.model_copy(update={"phase_message": message}),
        )

    async def _release_persistent_volume(
        self, session: AsyncSession, *, owner_principal_id: int, name: str
    ):
        """Soft-delete a template-created PV so its finalizer tears it down.

        Mirrors the DELETE route's soft delete (``phase=Deleting``): the PV
        finalizer (``GPUInstancePersistentVolumeController``) then deletes the
        downstream CRs across clusters and hard-deletes the row, waiting on any
        other active instance still sharing the volume. No-op if the PV row is
        already gone or already Deleting."""
        pv = await GPUInstancePersistentVolume.first_by_fields(
            session,
            fields={"owner_principal_id": owner_principal_id, "name": name},
        )
        if pv is None or (
            pv.status is not None and pv.status.phase == GPUInstancePhase.DELETING
        ):
            return
        base = pv.status or GPUInstancePersistentVolumeStatus()
        updated = base.model_copy(
            update={"phase": GPUInstancePhase.DELETING, "phase_message": None}
        )
        await pv.update(session, source={"status": updated})

    # ======================================================================= #
    # Shared helpers
    # ======================================================================= #

    @staticmethod
    def _status_equivalent(a: GPUInstanceStatus, b: GPUInstanceStatus) -> bool:
        """Whether two statuses carry the same observable state."""
        return a.model_dump() == b.model_dump()

    async def _build_ops(
        self,
        session: AsyncSession,
        instance: GPUInstance,
    ) -> Optional[Tuple[ClusterOps, str]]:
        """Resolve the cluster + principal and construct a ``ClusterOps``.

        Returns ``(ops, principal_identifier)`` or ``None`` when either
        the cluster or the owning principal has gone away.
        """

        cluster = await Cluster.one_by_id(session, instance.cluster_id)
        if cluster is None:
            logger.warning(
                "GPU instance %s references missing cluster %s",
                instance.name,
                instance.cluster_id,
            )
            return None

        principal = await Principal.one_by_id(session, instance.owner_principal_id)
        if principal is None:
            logger.warning(
                "GPU instance %s references missing principal %s",
                instance.name,
                instance.owner_principal_id,
            )
            return None

        owner_identifier = principal_namespace_identifier(principal)
        ops = ClusterOps(
            server_api_port=self._config.get_api_port(),
            cluster_id=cluster.id,
            cluster_registration_token=cluster.registration_token,
            cluster_owner_principal_identifier=owner_identifier,
        )
        return ops, owner_identifier

    @staticmethod
    async def _aggregate_ssh_public_key_data(
        session: AsyncSession,
        instance: GPUInstance,
    ) -> str:
        parts: List[str] = []
        for ref in instance.spec.ssh_public_keys or []:
            key = await GPUInstanceSSHPublicKey.first_by_fields(
                session,
                fields={
                    "owner_principal_id": instance.owner_principal_id,
                    "name": ref.name,
                },
            )
            if key is None:
                raise RuntimeError(
                    f"GPU instance SSH public key '{ref.name}' not found"
                )
            parts.append(key.spec.data)
        return "\n".join(parts)


class _PersistentVolumeFinalizeController:
    """Shared machinery for the PV / PVT soft-delete finalizers.

    A PV / PVT ``DELETE`` is a soft delete (``status.phase = Deleting``); these
    leader-only controllers finalize such rows: they enumerate every cluster,
    probe the downstream CR, delete it where present, record the clusters still
    holding it in ``status.finalizing``, and hard-delete the row once none
    remain. Steady polling is an in-memory ``add_after`` requeue with no DB
    write, and ``status.finalizing`` is persisted only when it changes, so the
    finalizer adds no write-to-self churn.

    Subclasses bind the model + the downstream probe/delete + the "still
    referenced?" gate; the base owns the queue, the Deleting-only enqueue, the
    all-cluster finalize loop, the change-only ``status`` write, and the
    hard-delete.
    """

    MODEL: Any = None  # SQLModel table class
    STATUS_CLS: Any = None  # its status BaseModel
    SOURCE: str = ""  # subscribe source label

    def __init__(self, cfg: Config):
        self._config = cfg
        self._queue: WorkQueue = WorkQueue()
        self._inflight: Dict[Any, asyncio.Task] = {}
        self._dispatch_task: Optional[asyncio.Task] = None
        # Cadence for re-probing a still-finalizing row via an in-memory requeue
        # (no DB write). Clamped to >= 1s so a misconfigured 0 can't turn
        # ``add_after`` into a busy loop.
        self._finalize_interval: float = max(
            1, envs.GPU_INSTANCE_TRANSITIONING_REQUEUE_INTERVAL
        )

    async def start(self):
        self._dispatch_task = asyncio.create_task(self._dispatch())
        try:
            async for event in self.MODEL.subscribe(source=self.SOURCE):
                if event.type == EventType.HEARTBEAT or event.data is None:
                    continue
                # A DELETED bus event means the row is already gone; nothing to
                # finalize. Only rows that entered ``Deleting`` need work.
                if event.type == EventType.DELETED:
                    continue
                row = event.data
                if row.status and row.status.phase == GPUInstancePhase.DELETING:
                    self._enqueue(event)
        finally:
            tasks: List[asyncio.Task] = []
            if self._dispatch_task is not None:
                self._dispatch_task.cancel()
                tasks.append(self._dispatch_task)
            for task in list(self._inflight.values()):
                task.cancel()
                tasks.append(task)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _to_work_event(row_id: Any, event: Event) -> WorkEvent:
        return WorkEvent(keys=(row_id,), type=WorkEventType.MODIFIED, object=event)

    def _enqueue(self, event: Event):
        """Map a bus event onto the work queue (coalescing happens in the queue)."""
        row_id = event.data.id if event.data is not None else event.id
        if row_id is None:
            return
        self._queue.add(self._to_work_event(row_id, event))

    def _requeue_after(self, row, delay: float) -> None:
        """Re-probe ``row`` after ``delay`` via an in-memory requeue.

        No DB write, no bus event — just a delayed re-add so a still-finalizing
        row keeps being probed until it settles.
        """
        event = Event(type=EventType.UPDATED, data=self.MODEL(id=row.id))
        self._queue.add_after(self._to_work_event(row.id, event), delay)

    async def _dispatch(self):
        while True:
            event = await self._queue.get()
            self._inflight[event.keys] = asyncio.create_task(self._process(event))

    async def _process(self, event: WorkEvent):
        keys = event.keys
        try:
            await self._reconcile(event.object)
            # Success — reset the per-keys backoff counter.
            self._queue.forget(keys)
        except Exception:
            logger.exception("Failed to finalize %s %s", self.SOURCE, keys[0])
            # Failure — retry with a capped exponential backoff.
            self._queue.add_rate_limited(event)
        finally:
            self._queue.done(keys)
            # Drop the finished task; the returned Task is intentionally discarded.
            _ = self._inflight.pop(keys, None)

    async def _reconcile(self, event: Event):
        row = event.data
        if row is None or row.id is None:
            return
        await self._finalize(row.id)

    async def _finalize(self, row_id: int):
        async with async_session() as session:
            row = await self.MODEL.one_by_id(session, row_id)
            if row is None or not (
                row.status and row.status.phase == GPUInstancePhase.DELETING
            ):
                # Already hard-deleted, or no longer marked for deletion.
                return

            principal = await Principal.one_by_id(session, row.owner_principal_id)
            if principal is None:
                logger.warning(
                    "%s %s references missing principal %s",
                    self.SOURCE,
                    row.name,
                    row.owner_principal_id,
                )
                return
            owner_identifier = principal_namespace_identifier(principal)

            blocked = await self._blocked_reason(session, row)
            if blocked is not None:
                await self._write_status(session, row, phase_message=blocked)
                self._requeue_after(row, self._finalize_interval)
                return

            finalizing = await self._probe_clusters(session, row, owner_identifier)
            if not finalizing:
                await self._hard_delete(session, row)
                return

            await self._write_status(session, row, finalizing=finalizing)
            self._requeue_after(row, self._finalize_interval)

    async def _probe_clusters(
        self, session: AsyncSession, row, owner_identifier: str
    ) -> List[int]:
        """Probe every cluster; delete the downstream CR where present. Returns
        the cluster ids that still hold the object (a cluster whose probe errored
        is kept so the next round retries it)."""
        finalizing: List[int] = []
        for cluster in await Cluster.all(session):
            try:
                ops = self._build_ops(cluster, owner_identifier)
                async with ops:
                    if await self._probe_and_delete(ops, row, owner_identifier):
                        finalizing.append(cluster.id)
            except Exception:
                logger.exception(
                    "Failed probing cluster %s for %s %s",
                    cluster.id,
                    self.SOURCE,
                    row.name,
                )
                finalizing.append(cluster.id)
        return finalizing

    def _build_ops(self, cluster: Cluster, owner_identifier: str) -> ClusterOps:
        return ClusterOps(
            server_api_port=self._config.get_api_port(),
            cluster_id=cluster.id,
            cluster_registration_token=cluster.registration_token,
            cluster_owner_principal_identifier=owner_identifier,
        )

    async def _write_status(
        self,
        session: AsyncSession,
        row,
        *,
        finalizing: Optional[List[int]] = None,
        phase_message: Optional[str] = None,
    ):
        """Persist ``finalizing`` / ``phase_message`` onto status — but only on a
        real change, so a steady wait re-queues without a DB write."""
        base = row.status or self.STATUS_CLS()
        target_finalizing = finalizing if finalizing is not None else base.finalizing
        if base.phase_message == phase_message and (base.finalizing or []) == (
            target_finalizing or []
        ):
            return
        updated = base.model_copy(
            update={"finalizing": target_finalizing, "phase_message": phase_message}
        )
        await row.update(session, source={"status": updated})

    async def _hard_delete(self, session: AsyncSession, row):
        try:
            await row.delete(session)
        except IntegrityError:
            # RESTRICT backstop (a PVT still referenced by a PV): the reference
            # check above should have caught it, but a concurrent PV create could
            # slip in — wait and retry rather than crash.
            logger.info(
                "Deferring hard-delete of %s %s: still referenced downstream",
                self.SOURCE,
                row.name,
            )
            self._requeue_after(row, self._finalize_interval)

    # -- subclass hooks ----------------------------------------------------- #

    async def _blocked_reason(self, session: AsyncSession, row) -> Optional[str]:
        """A human-readable reason the row must not be finalized yet, or None."""
        raise NotImplementedError

    async def _probe_and_delete(
        self, ops: ClusterOps, row, owner_identifier: str
    ) -> bool:
        """Whether the downstream CR is present on this cluster; delete it when
        so (it stays 'present' until a later probe confirms it is gone)."""
        raise NotImplementedError


class GPUInstancePersistentVolumeController(_PersistentVolumeFinalizeController):
    """Finalizes soft-deleted ``GPUInstancePersistentVolume`` rows."""

    MODEL = GPUInstancePersistentVolume
    STATUS_CLS = GPUInstancePersistentVolumeStatus
    SOURCE = "gpu_instance_persistent_volume_controller"

    async def _blocked_reason(self, session: AsyncSession, row) -> Optional[str]:
        # GPUInstance -> PV is ON DELETE SET NULL, so a hard-delete would silently
        # detach a volume still mounted by a running instance. Wait until every
        # active reference clears (a deleting instance clears its ref only once
        # its own hard-delete SET NULLs it).
        stmt = (
            select(GPUInstance.id)
            .where(GPUInstance.persistent_volume_id == row.id)
            .limit(1)
        )
        if (await session.exec(stmt)).first() is not None:
            return "waiting for active GPU instance(s) to release this volume"
        return None

    async def _probe_and_delete(
        self, ops: ClusterOps, row, owner_identifier: str
    ) -> bool:
        # PV is namespaced; list across all namespaces (ClusterOps._list) and
        # match our own name + namespace so a same-named PV in another Org's
        # namespace is never touched.
        for item in await ops.list_persistent_volumes():
            metadata = item.get("metadata") or {}
            if (
                metadata.get("name") == row.name
                and metadata.get("namespace") == ops.org_namespace
            ):
                await ops.delete_persistent_volume(row.name)
                return True
        return False


class GPUInstancePersistentVolumeTypeController(_PersistentVolumeFinalizeController):
    """Finalizes soft-deleted ``GPUInstancePersistentVolumeType`` rows."""

    MODEL = GPUInstancePersistentVolumeType
    STATUS_CLS = GPUInstancePersistentVolumeTypeStatus
    SOURCE = "gpu_instance_persistent_volume_type_controller"

    async def _blocked_reason(self, session: AsyncSession, row) -> Optional[str]:
        # PV -> PVT is ON DELETE RESTRICT, so PV before PVT: wait until no PV row
        # references this type before clearing the downstream PVTs and hard-
        # deleting the row (the RESTRICT is a hard-delete backstop).
        stmt = (
            select(GPUInstancePersistentVolume.id)
            .where(GPUInstancePersistentVolume.persistent_volume_type_id == row.id)
            .limit(1)
        )
        if (await session.exec(stmt)).first() is not None:
            return "waiting for referencing persistent volume(s) to be deleted first"
        return None

    async def _probe_and_delete(
        self, ops: ClusterOps, row, owner_identifier: str
    ) -> bool:
        # PVT is cluster-scoped; its downstream name folds in the owner
        # identifier, so match that computed name against the cluster list.
        cluster_name = get_persistent_volume_type_name(
            row.name, principal_identifier=owner_identifier
        )
        for item in await ops.list_persistent_volume_types():
            if (item.get("metadata") or {}).get("name") == cluster_name:
                await ops.delete_persistent_volume_type(cluster_name)
                return True
        return False
