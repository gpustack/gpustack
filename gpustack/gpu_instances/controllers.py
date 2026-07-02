import json
import logging
import asyncio
import re
from typing import Any, Dict, List, Set, Tuple, Optional
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


class GPUInstanceController:
    """Reconciles ``GPUInstance`` rows against the worker cluster CRDs.

    The Python row is the source of truth; this controller projects each
    ``GPUInstance`` change onto ``worker.gpustack.ai/v1`` CRs via
    :class:`ClusterOps` and writes back the observed phase. SSH keys,
    persistent-volume types, and persistent volumes are sub-resources
    that aren't globally synced — their lifecycle is bound to the owning
    instance and handled inline here.
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
        # Generic work queue replacing the old per-iid pending slot + worker
        # tasks. Keyed by ``(iid,)``. Configured to reproduce the previous
        # coalescing exactly: the dedup window / backoff / requeueAfter stay
        # off in this phase, and the custom coalescer preserves the
        # reconfirm-droppable, DELETED-sticky and DELETING-semi-sticky slot
        # policy the old ``_enqueue`` enforced.
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
        # iids with a delayed re-confirm already scheduled on the queue. Each
        # Ready row schedules its own re-confirm via ``add_after`` and
        # reschedules it from ``_reconcile_reconfirm`` (the per-key replacement
        # for the old full-table timer sweep); this guard collapses repeated
        # Ready observations onto a single chain. See ``_schedule_reconfirm``.
        self._reconfirm_scheduled: Set[Any] = set()
        # Cadence for re-observing an in-flight row via an in-memory requeue
        # (replaces the old DB-write self-poll). Clamped to >= 1s so a
        # misconfigured 0 can't turn ``add_after`` into a busy loop.
        self._requeue_interval: float = max(1, envs.GPU_INSTANCE_REQUEUE_INTERVAL)

    async def start(self):
        self._dispatch_task = asyncio.create_task(self._dispatch())
        self._watch_task = asyncio.create_task(self._watch_downstream())
        try:
            async for event in GPUInstance.subscribe(source="gpu_instance_controller"):
                if event.type == EventType.HEARTBEAT:
                    continue
                if event.data is None:
                    continue

                if event.type == EventType.CREATED:
                    instance: GPUInstance = event.data
                    if instance.is_ready():
                        # Re-discover Ready rows after a restart and seed the
                        # per-key re-confirm — the timer sweep used to do this
                        # via a full-table scan.
                        self._schedule_reconfirm(instance)
                        continue
                    # Terminal *Failed / STOPPED: nothing to reconcile.
                    if instance.is_failed() or instance.is_stopped():
                        continue
                    # Brand-new (phase=None) and in-flight rows both go through
                    # the unified phase-based reconcile on the work queue
                    # (CREATED->ADDED), replacing the old inline _reconcile_created.
                    self._enqueue(event)
                    continue

                # UPDATED / DELETED — coalesce per iid and run on a worker task.
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

    def _enqueue(self, event: Event):
        """Map a bus event onto the work queue (coalescing happens in the queue)."""
        iid = event.data.id if event.data is not None else event.id
        if iid is None:
            return
        self._queue.add(self._to_work_event(iid, event))

    @staticmethod
    def _to_work_event(iid: Any, event: Event) -> WorkEvent:
        if event.type == EventType.DELETED:
            wtype = WorkEventType.DELETED
        elif event.type == EventType.CREATED:
            wtype = WorkEventType.ADDED
        else:
            wtype = WorkEventType.MODIFIED
        return WorkEvent(keys=(iid,), type=wtype, object=event)

    def _requeue_after(self, instance: GPUInstance, delay: float) -> None:
        """Re-observe ``instance`` after ``delay`` via an in-memory requeue.

        Replaces the old "write the same-phase status back to the DB to re-fire
        the bus and re-enqueue" self-poll: no DB write, no bus event — just a
        delayed re-add so an in-flight row keeps being reconciled until it
        settles. ``changed_fields`` is empty (a re-observation, not a spec
        change), matching what the old status-write bus event carried.
        """
        event = Event(type=EventType.UPDATED, data=instance)
        self._queue.add_after(self._to_work_event(instance.id, event), delay)

    def _schedule_reconfirm(self, instance: GPUInstance) -> None:
        """Schedule one delayed re-confirm for a Ready row (idempotent per iid).

        The per-key replacement for the old 60s full-table sweep: rather than
        scanning every non-deleted row on a timer, each Ready row schedules its
        own re-confirm via ``add_after`` and reschedules from
        ``_reconcile_reconfirm`` while it stays Ready — so the cadence is
        per-key and no O(N) scan runs on a timer. ``_reconfirm_scheduled``
        collapses repeated Ready observations (e.g. a spec-edit UPDATED landing
        on a Ready row) onto a single chain.
        """
        if envs.GPU_INSTANCE_RECONFIRM_INTERVAL <= 0:
            return
        if instance.id in self._reconfirm_scheduled:
            return
        self._reconfirm_scheduled.add(instance.id)
        event = Event(type=EventType.UPDATED, data=instance, reconfirm=True)
        self._queue.add_after(
            self._to_work_event(instance.id, event),
            envs.GPU_INSTANCE_RECONFIRM_INTERVAL,
        )

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
            await asyncio.sleep(self._requeue_interval)

    async def _on_downstream_event(self, line: str):
        """Map one downstream ``WorkerEvent`` line onto an upstream reconcile.

        Both ``UPDATED`` and ``DELETED`` push an upstream ``UPDATED`` keyed by the
        resolved instance id: the phase-based reconcile re-fetches the row and
        re-reads the worker, so it handles a CR-gone (delete) uniformly without a
        separate delete path. The downstream object is only a trigger.
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
        # A re-observation, not a spec edit — empty changed_fields (mirrors
        # _requeue_after) so the reconcile skips the spec-driven SSH resync.
        self._enqueue(Event(type=EventType.UPDATED, data=GPUInstance(id=iid)))

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

    def _coalesce_events(self, existing: WorkEvent, incoming: WorkEvent) -> WorkEvent:
        """Per-iid slot policy the old ``_enqueue`` enforced.

        Only invoked when a pending event already occupies the slot for a keys.
        """
        incoming_event: Event = incoming.object
        # A periodic re-confirm is best-effort and must never displace a real
        # pending event; conversely a real event displaces a pending re-confirm.
        # Either way the dropped re-confirm won't reach ``_reconcile_reconfirm``
        # to reschedule itself, so free its guard whenever it leaves the slot in
        # *either* direction — otherwise the id stays marked scheduled forever
        # and the row is never re-confirmed again. The winning real event
        # re-seeds the chain (via the is_ready short-circuit) once it settles,
        # if the row is still Ready.
        if incoming_event.reconfirm:
            self._reconfirm_scheduled.discard(incoming.keys[0])
            return existing
        # DELETED is terminal — nothing supersedes it.
        if existing.type == WorkEventType.DELETED:
            return existing
        # An in-flight MODIFIED whose snapshot already reads DELETING is
        # terminal-bound: a later MODIFIED can only carry equal-or-stale data,
        # so don't let it replace the slot. DELETED is still allowed through
        # (terminal upgrade).
        existing_event: Event = existing.object
        if (
            existing.type == WorkEventType.MODIFIED
            and incoming.type == WorkEventType.MODIFIED
            and existing_event.data is not None
            and (existing_event.data.status or GPUInstanceStatus()).phase
            == GPUInstancePhase.DELETING
        ):
            return existing
        # A real event is displacing a pending re-confirm out of the slot: free
        # its guard (see the note above) so the chain can re-seed.
        if existing_event.reconfirm:
            self._reconfirm_scheduled.discard(existing.keys[0])
        return incoming

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
        except Exception:
            logger.exception(f"Failed to reconcile GPU instance {keys[0]}")
        finally:
            # done() and the pop run without an intervening await, so the
            # dispatch loop cannot re-hand this keys (and overwrite the entry)
            # between them.
            self._queue.done(keys)
            self._inflight.pop(keys, None)

    async def _reconcile(self, event: Event):
        instance: GPUInstance = event.data
        if instance is None:
            return
        # Periodic re-confirmation (see ``_schedule_reconfirm``): a process-local
        # marker carried on the event. It rides the same per-iid worker so it
        # serializes with real /stop, /delete events, but reads the worker side
        # back without the Ready short-circuit.
        if event.reconfirm:
            await self._reconcile_reconfirm(instance)
            return
        if event.type in (EventType.CREATED, EventType.UPDATED):
            # CREATED (brand-new) and UPDATED both run the unified phase-based
            # reconcile; the ``is_creating`` branch provisions a new row.
            await self._reconcile_updated(instance, event.changed_fields or {})
        elif event.type == EventType.DELETED:
            await self._reconcile_deleted(instance)

    async def _create_downstream(  # noqa: C901
        self,
        session: AsyncSession,
        fresh: GPUInstance,
        ops: ClusterOps,
        principal_identifier: str,
    ) -> bool:
        """Provision a brand-new instance's worker-side resources.

        Creates the referenced SSH public key, persistent volume (type), and the
        Instance CR. On any failure it records the matching ``*CreateFailed``
        phase and returns ``False`` so the caller stops; on success returns
        ``True`` and the caller falls through to read + merge. Extracted from the
        old inline ``_reconcile_created`` and folded into ``_reconcile_updated``'s
        ``is_creating`` branch (the caller already holds ``session`` and an open
        ``ops``).
        """
        # Create/Update referenced SSH public keys if needed.
        if fresh.spec.ssh_public_keys is not None:
            try:
                data = await self._aggregate_ssh_public_key_data(session, fresh)
                await ops.upsert_ssh_public_key(
                    name=fresh.name,
                    spec={"data": data},
                )
            except Exception as e:
                logger.exception(
                    f"Failed to sync worker-side ssh public key for {fresh.name}"
                )
                if not fresh.is_ready():
                    await self._set_status(
                        session,
                        fresh,
                        GPUInstanceStatus(
                            phase=self.PHASE_SSH_KEY_CREATE_FAILED,
                            phase_message=f"Failed to sync worker-side ssh public key: {e}",
                            namespace=ops.org_namespace,
                        ),
                    )
                return False

        # Create referenced persistent volume type and persistent volume if needed.
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
                await self._set_status(
                    session,
                    fresh,
                    GPUInstanceStatus(
                        phase=self.PHASE_PV_CREATE_FAILED,
                        phase_message=f"Not found server-side persistent volume: {pv_name}",
                        namespace=ops.org_namespace,
                    ),
                )
                return False

            pvt_name = pv.spec.type_
            pvt = await GPUInstancePersistentVolumeType.one_by_id(
                session, pv.persistent_volume_type_id
            )
            if pvt is None:
                logger.error(
                    f"Failed to find server-side pv type for {fresh.name} with pvt name {pvt_name}"
                )
                await self._set_status(
                    session,
                    fresh,
                    GPUInstanceStatus(
                        phase=self.PHASE_PV_TYPE_CREATE_FAILED,
                        phase_message=f"Not found server-side persistent volume type: {pvt_name}",
                        namespace=ops.org_namespace,
                    ),
                )
                return False

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
                await self._set_status(
                    session,
                    fresh,
                    GPUInstanceStatus(
                        phase=self.PHASE_PV_TYPE_CREATE_FAILED,
                        phase_message=f"Failed to create worker-side persistent volume type: {e}",
                        namespace=ops.org_namespace,
                    ),
                )
                return False

            try:
                pv_spec = spec_persistent_volume(pv)
                # The worker-side PV references the worker-side PVT
                # name (prefixed with the principal name); overwrite the
                # raw type id from the row.
                pv_spec["type"] = pvt_cluster_name
                await ops.create_persistent_volume(
                    name=pv.name,
                    spec=pv_spec,
                )
            except Exception as e:
                logger.exception(f"Failed to create worker-side pv for {fresh.name}")
                await self._set_status(
                    session,
                    fresh,
                    GPUInstanceStatus(
                        phase=self.PHASE_PV_CREATE_FAILED,
                        phase_message=f"Failed to create worker-side persistent volume: {e}",
                        namespace=ops.org_namespace,
                    ),
                )
                return False

        # Create Instance.
        try:
            await ops.create_instance(fresh.convert_to_kuberes())
        except Exception as e:
            logger.exception(f"Failed to create worker-side instance for {fresh.name}")
            await self._set_status(
                session,
                fresh,
                GPUInstanceStatus(
                    phase=self.PHASE_CREATE_FAILED,
                    phase_message=f"Failed to create worker-side instance: {e}",
                    namespace=ops.org_namespace,
                ),
            )
            return False

        return True

    async def _reconcile_updated(  # noqa: C901
        self, instance: GPUInstance, changed_fields: Dict[str, Tuple[Any, Any]]
    ):
        async with async_session() as session:
            fresh = await GPUInstance.one_by_id(session, instance.id)
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

                # Phase-specific handling.
                phase = (fresh.status or GPUInstanceStatus()).phase
                if fresh.is_creating():
                    # Brand-new row (phase=None): provision the worker-side
                    # resources, then fall through to read + merge. Create is
                    # idempotent (ignore_existed); a failure records a
                    # ``*CreateFailed`` phase and stops here.
                    logger.debug(
                        f"GPUInstance {fresh.name} is new, provisioning worker-side resources"
                    )
                    if not await self._create_downstream(
                        session, fresh, ops, principal_identifier
                    ):
                        return
                elif phase is not None:
                    if fresh.is_deleting():
                        # Delete the worker-side CR (and SSH key) and fall
                        # through to read_instance, which drives the DB cleanup
                        # when the CR is gone.
                        logger.debug(
                            f"GPUInstance {fresh.name} is in {phase} phase, try deleting worker-side instance"
                        )
                        try:
                            await ops.delete_instance(fresh.name)
                            await ops.delete_ssh_public_key(fresh.name)
                        except Exception as e:
                            logger.exception(
                                f"Failed to delete worker-side instance for {fresh.name}"
                            )
                            await self._set_status(
                                session,
                                fresh,
                                (fresh.status or GPUInstanceStatus()).model_copy(
                                    update={
                                        "phase": phase,
                                        "phase_message": f"Failed to delete worker-side instance, will retry: {e}",
                                        "namespace": ops.org_namespace,
                                    }
                                ),
                            )
                            return
                    elif fresh.is_stopping():
                        # Patch ``spec.stop=true``; the worker operator
                        # tears the workload down and eventually reports
                        # ``status.phase=Stopped``, at which point the
                        # read_instance block flips the row to STOPPED.
                        logger.debug(
                            f"GPUInstance {fresh.name} is in {phase} phase, try stopping worker-side instance"
                        )
                        try:
                            await ops.stop_instance(fresh.name)
                        except Exception as e:
                            logger.exception(
                                f"Failed to stop worker-side instance for {fresh.name}"
                            )
                            await self._set_status(
                                session,
                                fresh,
                                (fresh.status or GPUInstanceStatus()).model_copy(
                                    update={
                                        "phase_message": f"Failed to stop worker-side instance, will retry: {e}",
                                        "namespace": ops.org_namespace,
                                    }
                                ),
                            )
                            return
                    elif fresh.is_starting():
                        # Patch ``spec.stop`` off (merge-patch null). When
                        # the CR is missing (legacy STOPPED row whose CR
                        # was deleted, or retry from *CreateFailed before
                        # the initial create succeeded), bootstrap the CR
                        # from scratch so /start works uniformly.
                        logger.debug(
                            f"GPUInstance {fresh.name} is in {phase} phase, try starting worker-side instance"
                        )
                        try:
                            patched = await ops.start_instance(fresh.name)
                            if patched is None:
                                await ops.create_instance(fresh.convert_to_kuberes())
                        except Exception as e:
                            logger.exception(
                                f"Failed to start worker-side instance for {fresh.name}"
                            )
                            await self._set_status(
                                session,
                                fresh,
                                (fresh.status or GPUInstanceStatus()).model_copy(
                                    update={
                                        "phase_message": f"Failed to start worker-side instance, will retry: {e}",
                                        "namespace": ops.org_namespace,
                                    }
                                ),
                            )
                            return
                    elif fresh.is_failed():
                        logger.warning(
                            f"GPUInstance {fresh.name} is in {phase} phase, skip updating"
                        )
                        return
                    elif fresh.is_stopped():
                        # Terminal Stopped: the worker-side instance was
                        # confirmed gone in a previous reconcile and we don't
                        # want to bounce back to Unknown via the read_instance
                        # block below. The row stays here until /start
                        # (Starting) or /delete (Deleting).
                        logger.debug(
                            f"GPUInstance {fresh.name} is in {phase} phase, skip updating"
                        )
                        return
                    elif fresh.is_ready():
                        logger.debug(
                            f"GPUInstance {fresh.name} is already in {phase} phase, skip updating"
                        )
                        # The event loop stops touching a Ready row here, so seed
                        # (or re-seed) its per-key re-confirm chain. Idempotent
                        # via ``_reconfirm_scheduled``.
                        self._schedule_reconfirm(fresh)
                        return
                    else:
                        logger.debug(
                            f"GPUInstance {fresh.name} is in {phase} phase, try updating and reconciling"
                        )

                # Read the instance to get the latest status.
                try:
                    read = await ops.read_instance(fresh.name)
                    if read is None:
                        if fresh.is_deleting():
                            # CR gone — hard-delete the DB row. The DELETED
                            # event drives _reconcile_deleted for SSH key /
                            # PV cleanup; return so we don't touch ``fresh``
                            # again.
                            await fresh.delete(session)
                            return
                        elif fresh.is_stopping():
                            # CR vanished while stopping — treat as stopped.
                            await self._set_status(
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
                    # else: the normal path — read back the real status to sync the DB
                except Exception:
                    logger.exception(
                        f"Failed to read worker-side instance for {fresh.name}"
                    )

                    # If reading fails, keep the in-flight phase.
                    if phase not in (
                        self.PHASE_DELETING,
                        self.PHASE_STOPPING,
                        self.PHASE_STARTING,
                    ):
                        phase = self.PHASE_UNKNOWN

                    read = {
                        "metadata": {"namespace": ops.org_namespace},
                        "status": (fresh.status or GPUInstanceStatus())
                        .model_copy(
                            update={
                                "phase": phase,
                                "phase_message": "Failed to confirm from cluster, will retry",
                            }
                        )
                        .model_dump(by_alias=True, exclude_none=True),
                    }

                merged = fresh.merge_from_kuberes(read)

                # Hold the in-flight phase until the worker operator has
                # actually moved: spec.stop was just patched, so the CR's
                # status.phase typically still reads the pre-patch value
                # for one reconcile tick. Letting it through would flicker
                # the row (Stopping→Ready, Starting→Stopped).
                worker_phase = merged.phase
                if (
                    (phase == self.PHASE_DELETING)
                    or (
                        phase == self.PHASE_STOPPING
                        and worker_phase != self.PHASE_STOPPED
                    )
                    or (
                        phase == self.PHASE_STARTING
                        and worker_phase == self.PHASE_STOPPED
                    )
                ):
                    merged.phase = phase

                await self._set_status(session, fresh, merged)

    async def _reconcile_deleted(self, instance: GPUInstance):
        # A deleted row has no re-confirm chain; drop any lingering guard (the
        # queue cancels the delayed entry itself when the DELETED event lands).
        self._reconfirm_scheduled.discard(instance.id)
        async with async_session() as session:
            built = await self._build_ops(session, instance)
            if built is None:
                return
            ops, _ = built

            # When the row went through DELETING, _reconcile_updated
            # already confirmed the CR was gone before hard-deleting the
            # row, so skip the redundant cluster delete. STOPPED rows
            # still have a live (stopped) CR — those need the delete.
            already_deleted_in_cluster = instance.is_deleting()

            async with ops:
                # Delete Instance.
                if not already_deleted_in_cluster:
                    try:
                        await ops.delete_instance(instance.name)
                    except Exception:
                        logger.exception(
                            f"Failed to delete worker-side instance for {instance.name}"
                        )

                # Delete referenced SSH public key.
                try:
                    await ops.delete_ssh_public_key(instance.name)
                except Exception:
                    logger.exception(
                        f"Failed to delete worker-side ssh public key for {instance.name}"
                    )

            # Persistent volume release. PV / PVT are independent
            # finalizer-driven resources now (their controllers own downstream
            # cleanup), so instance deletion no longer tears down their worker
            # side directly. The one instance-scoped case is a
            # ``persistent_template`` that opted into ``release_with_instance``:
            # soft-delete its PV row so the PV finalizer reclaims it.
            volume = instance.spec.volume
            if volume is None or volume.persistent_template is None:
                return
            if not volume.persistent_template.release_with_instance:
                return
            await self._release_persistent_volume(
                session,
                owner_principal_id=instance.owner_principal_id,
                name=volume.persistent_template.name,
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

    async def _reconcile_reconfirm(self, instance: GPUInstance):
        """Re-read the worker side for a Ready row, write back only on drift.

        The recurring half of the per-key re-confirm (see ``_schedule_reconfirm``,
        which replaced the full-table timer sweep). The reconciler is
        event-driven and stops touching a row once it is fully Ready, but the
        worker side has no write-back path into the ``GPUInstance`` row, so a
        post-Ready change on the worker (workload crash, allocation/address
        change, CR dropping out of Ready, CR deleted) would otherwise never be
        observed. This re-reads and only writes back on real drift, so a steady
        Ready row produces no DB churn or bus traffic.

        While the row stays Ready it reschedules its own next re-confirm; a
        drift out of Ready ends the chain and it is re-seeded by
        ``_reconcile_updated``'s ``is_ready`` short-circuit once the row settles
        back into Ready.
        """
        # This scheduled re-confirm has fired; free the guard so a fresh one can
        # be scheduled (below on success, or by the is_ready short-circuit).
        self._reconfirm_scheduled.discard(instance.id)
        async with async_session() as session:
            fresh = await GPUInstance.one_by_id(session, instance.id)
            if fresh is None or fresh.deleted_at is not None:
                return
            # Re-check under the fresh row: a real event may have moved the
            # phase between enqueue and now, in which case the normal loop
            # already owns it and will re-seed on the next Ready.
            if not fresh.is_ready():
                return

            built = await self._build_ops(session, fresh)
            if built is None:
                # Cluster/principal transiently gone — keep the chain alive so
                # the row is revisited, matching the old sweep's per-tick retry.
                self._schedule_reconfirm(fresh)
                return
            ops, _ = built
            async with ops:
                try:
                    read = await ops.read_instance(fresh.name)
                except Exception:
                    logger.exception(
                        f"Failed to re-confirm worker-side instance for {fresh.name}"
                    )
                    # Transient — leave the row as-is and retry on the next tick.
                    self._schedule_reconfirm(fresh)
                    return

                if read is None:
                    # The CR vanished under a Ready row: real drift. Hand it
                    # back to the event/count loop via Unknown rather than
                    # writing a synthetic Ready. The chain ends here and
                    # re-seeds once the row settles back into Ready.
                    await self._set_status(
                        session,
                        fresh,
                        GPUInstanceStatus(
                            phase=self.PHASE_UNKNOWN,
                            phase_message="Not found in cluster",
                            namespace=ops.org_namespace,
                        ),
                        require_current_phase=self.PHASE_READY,
                    )
                    return

                candidate = fresh.merge_from_kuberes(read)
                current = fresh.status or GPUInstanceStatus()
                if not self._status_equivalent(current, candidate):
                    # Guard the read→write window: only persist while the row is
                    # still Ready, so a /stop or /start that landed during
                    # read_instance isn't clobbered (the real event drives it).
                    await self._set_status(
                        session,
                        fresh,
                        candidate,
                        require_current_phase=self.PHASE_READY,
                    )

                # Keep the chain alive only while the row remains Ready; a drift
                # to another phase lets the normal loop take over and re-seed.
                if candidate.phase == self.PHASE_READY:
                    self._schedule_reconfirm(fresh)

    @staticmethod
    def _status_equivalent(a: GPUInstanceStatus, b: GPUInstanceStatus) -> bool:
        """Compare two statuses ignoring the server-only retry ``count``."""
        return a.model_dump(exclude={"count"}) == b.model_dump(exclude={"count"})

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

    async def _set_status(
        self,
        session: AsyncSession,
        instance: GPUInstance,
        expected: GPUInstanceStatus,
        require_current_phase: Optional[str] = None,
    ):
        """Persist ``expected`` onto the row's status — but only on a real change.

        DELETING is sticky: if /delete landed on the row mid-reconcile,
        the DB now reads DELETING and we drop the write so a stale
        Stopping/Starting/etc. from this reconcile cannot resurrect a
        non-DELETING phase. ``session.refresh`` reloads the column
        values from the DB to close the cross-session race.

        ``require_current_phase`` is an optional precondition for callers
        that must not clobber a phase a concurrent action moved the row
        into. The periodic re-confirm passes ``Ready``: between its
        ``is_ready`` check and this write a /stop or /start could land
        Stopping/Starting, and blindly writing the (stale) re-read status
        would overwrite it and strand the instance — the re-confirm only
        revisits Ready rows. When the live phase no longer matches, the write
        is dropped and the real pending event drives the transition.

        When ``expected`` matches the current status (ignoring the retry
        ``count``), the write is skipped entirely — no DB write, no bus event.
        An in-flight row (``require_current_phase`` unset) is instead re-observed
        via an in-memory requeue, so it keeps being reconciled until it settles
        without the old status-write self-poll. Reconfirm writes target Ready
        rows driven by the per-key re-confirm, so they never requeue here.
        """
        await session.refresh(instance)
        current = instance.status or GPUInstanceStatus()
        if (
            current.phase == GPUInstancePhase.DELETING
            and expected.phase != GPUInstancePhase.DELETING
        ):
            return
        if require_current_phase is not None and current.phase != require_current_phase:
            return
        if self._status_equivalent(current, expected):
            if require_current_phase is None:
                self._requeue_after(instance, self._requeue_interval)
            return
        if current.phase == expected.phase:
            expected.count = current.count + 1
        expected_dump = expected.model_dump(by_alias=True, exclude_none=True)
        await instance.update(session, source={"status": expected_dump})

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


# Cadence for re-probing a still-finalizing PV / PVT row. Uses ``add_after``
# (in-memory, no DB write), so a row that keeps waiting produces no churn.
_FINALIZE_REQUEUE_INTERVAL = 15.0


class _FinalizerController:
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
                if self._is_deleting(event.data):
                    self._enqueue(event.data.id)
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
    def _is_deleting(row) -> bool:
        return bool(row.status and row.status.phase == GPUInstancePhase.DELETING)

    def _enqueue(self, row_id: Any) -> None:
        if row_id is None:
            return
        self._queue.add(
            WorkEvent(keys=(row_id,), type=WorkEventType.MODIFIED, object=row_id)
        )

    def _requeue_after(self, row_id: Any, delay: float) -> None:
        self._queue.add_after(
            WorkEvent(keys=(row_id,), type=WorkEventType.MODIFIED, object=row_id),
            delay,
        )

    async def _dispatch(self):
        while True:
            event = await self._queue.get()
            self._inflight[event.keys] = asyncio.create_task(self._process(event))

    async def _process(self, event: WorkEvent):
        keys = event.keys
        try:
            await self._finalize(event.object)
        except Exception:
            logger.exception("Failed to finalize %s %s", self.SOURCE, keys[0])
        finally:
            self._queue.done(keys)
            self._inflight.pop(keys, None)

    async def _finalize(self, row_id: int):
        async with async_session() as session:
            row = await self.MODEL.one_by_id(session, row_id)
            if row is None or not self._is_deleting(row):
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
                self._requeue_after(row_id, _FINALIZE_REQUEUE_INTERVAL)
                return

            finalizing = await self._probe_clusters(session, row, owner_identifier)
            if not finalizing:
                await self._hard_delete(session, row)
                return

            await self._write_status(session, row, finalizing=finalizing)
            self._requeue_after(row_id, _FINALIZE_REQUEUE_INTERVAL)

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
            self._requeue_after(row.id, _FINALIZE_REQUEUE_INTERVAL)

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


class GPUInstancePersistentVolumeController(_FinalizerController):
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


class GPUInstancePersistentVolumeTypeController(_FinalizerController):
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
