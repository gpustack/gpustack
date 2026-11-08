import asyncio
import httpx
import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial, wraps
import json
import multiprocessing
from multiprocessing.connection import Connection
import shutil
import threading
import time

import requests
import setproctitle
import psutil
import os
from typing import Dict, Optional, Set, List, Callable, Tuple
from pathlib import Path
import logging
from uuid import UUID, uuid4
from weakref import WeakValueDictionary

from gpustack_runtime.deployer import (
    create_workload,
    get_workload,
    list_workloads,
    WorkloadPlan,
    WorkloadStatus,
    WorkloadStatusStateEnum,
    delete_workload,
)
from gpustack_runtime.deployer.__utils__ import compare_versions

from gpustack import envs
from gpustack.api.exceptions import HTTPException, NotFoundException
from gpustack.config.config import Config
from gpustack.config import registration
from gpustack.logging import (
    RedirectStdoutStderr,
)
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    is_built_in_backend,
    is_built_in_backend_custom_version,
    is_custom_backend,
)
from gpustack.utils import network
from gpustack.utils.convert import safe_int
from gpustack.utils.attrs import set_attr
from gpustack.utils.process import (
    terminate_multiprocessing_process,
    terminate_process,
    add_signal_handlers,
)
from gpustack.worker.backends.base import PD_CONFIGURATION_CHANGED_MESSAGE
from gpustack.worker.backends.ascend_mindie import AscendMindIEServer
from gpustack.worker.backends.sglang import (
    SGLangServer,
    build_sglang_smg_workload,
    split_sglang_smg_parameters,
)
from gpustack.utils.runtime import transform_workload_plan
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.worker.backends.vox_box import VoxBoxServer
from gpustack.worker.backends.custom import CustomServer
from gpustack.worker.log_sources import (
    CappedLogWriter,
    write_log_sources,
    canonical_log_path,
    container_log_path,
    engine_log_targets,
    find_instance_log_dir,
    flat_instance_logs,
    instance_log_dir,
    legacy_main_log_path,
    main_log_path,
    restart_log_dirs,
    sidecar_container_log_path,
    list_log_sources,
)
from gpustack.worker.container_logs import ContainerLogPersister
from gpustack.worker.model_meta import get_meta_from_running_instance
from gpustack.client import ClientSet
from gpustack.client.distributed_port_client import DistributedPortClient
from gpustack.schemas.port_reservations import ModelPortClaim
from gpustack.worker.pd_router import (
    apply_managed_router,
    is_managed_router,
    managed_router_health_path,
)
from gpustack.worker.pd_diagnostics import RestartTracker, diagnose
from gpustack.worker.port_reservations import (
    PortReservationSnapshot,
    WorkerPortReservations,
    model_instance_ports,
)

from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelUpdate,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStopReport,
    ModelInstanceWorkload,
    ModelInstanceWorkloadStateEnum,
    validate_workload_manifest,
    ModelInstanceStateEnum,
    get_backend,
    DistributedServerCoordinateModeEnum,
    ModelInstanceSubordinateWorker,
    CategoryEnum,
    PortBand,
    member_worker_ids,
    RoleNameEnum,
    role_effective_model,
)
from gpustack.utils.model_instance_workers import (
    get_model_instance_workloads,
    model_instance_legacy_stop,
    has_model_instance_stop_barrier,
    managed_instance_update,
    requires_group_recovery,
    calculate_group_recovery_policy,
    build_member_state_report,
    worker_workload_report,
    ModelInstanceTarget,
    ModelInstanceEndpoint,
    resolve_model_instance_endpoint,
)
from gpustack.utils.vllm_topology import vllm_lb_mode
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.utils.model_ports import model_port_requests
from gpustack.worker.pd_injection import band_specs_for
from gpustack.server.bus import Event, EventType
from gpustack.worker.inference_backend_manager import InferenceBackendManager

logger = logging.getLogger(__name__)

_MANAGED_LOG_FLUSH_TIMEOUT = 5.0

# Inference health check error message
_INFERENCE_HEALTH_CHECK_FAILED_MESSAGE = "Inference health check failed."

# Last-resort message for a workload that stopped serving without saying why.
_WORKLOAD_FAILED_MESSAGE = "Inference server exited or unhealthy."
_SUBORDINATE_ERROR_PREFIX = "Distributed serving error in subordinate worker "

_RANK_HEALTH_CONCURRENCY = 4
_MANAGED_TASK_CONCURRENCY = 4


@dataclass
class _RankHealthCheck:
    startup: bool
    failures: int = 0
    last_check: Optional[float] = None
    future: Optional[Future] = None


_PORT_CLEANUP_RETRIES = 15
_PORT_CLEANUP_RETRY_INTERVAL = 2

# Name the mp band is recorded under in `named_ports`. Prefixed so it cannot
# collide with a band a pd-mode declares: the catalog's names come from
# connectors and this one comes from the executor.
_VLLM_MP_CONNECTING_BAND_NAME = "_vllm_mp_connecting"

_SERVER_CLASS_MAPPING = {
    BackendEnum.VLLM: VLLMServer,
    BackendEnum.SGLANG: SGLangServer,
    BackendEnum.VOX_BOX: VoxBoxServer,
    BackendEnum.ASCEND_MINDIE: AscendMindIEServer,
}

# Annotation the operator device plugin writes onto the Pod after allocation,
# e.g. {"<container>": {"devices": {"groups": [{"accelerators": [{"id": "<GPU UUID>",
# "index": 0, "mode": 3, "allocated": 640000}]}]}, "deviceIDs": [...]}}.
_ALLOCATED_ACCELERATORS_ANNOTATION = "device.gpustack.ai/accelerator.allocated"


def _parse_allocated_accelerators(annotations: Optional[Dict[str, str]]) -> List[dict]:
    """
    Parse the allocated-accelerators annotation into a flat accelerator list.
    Tolerates missing/malformed payloads (device-plugin version skew):
    any parse problem means "allocation unknown", never a sync failure.
    """
    raw = (annotations or {}).get(_ALLOCATED_ACCELERATORS_ANNOTATION)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return []

    accelerators = []
    for container_allocation in parsed.values():
        if not isinstance(container_allocation, dict):
            continue
        devices = container_allocation.get("devices")
        if not isinstance(devices, dict):
            continue
        groups = devices.get("groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_accelerators = group.get("accelerators")
            if not isinstance(group_accelerators, list):
                continue
            for accelerator in group_accelerators:
                if isinstance(accelerator, dict) and accelerator.get("id"):
                    accelerators.append(accelerator)
    return accelerators


class _LogPersistence:
    """The threads and stop events of one generation of a model instance's log
    persistence.

    A sidecar thread only mutates its own generation's record, never the shared
    registry -- which is what lets the registry be swapped under a lock while
    threads are joined, with no joined thread waiting on that lock.
    """

    def __init__(
        self,
        stop_event: threading.Event,
        main_thread: threading.Thread,
        restart_count: int,
    ):
        self.stop_event = stop_event
        self.main_thread = main_thread
        # Which restart's files the generation writes.
        self.restart_count = restart_count
        self.workload_name: Optional[str] = None
        self.workload_writers: Dict[str, Tuple[threading.Thread, threading.Event]] = {}
        self.component_writer = None
        self.component_attempt = -1
        self._lock = threading.Lock()
        self._stopping = False
        self._aux_threads: List[threading.Thread] = []
        self._aux_stop_events: List[threading.Event] = []

    def start_aux_thread(
        self, thread: threading.Thread, stop_event: Optional[threading.Event] = None
    ) -> bool:
        """Start a thread of this generation, unless teardown has begun.

        Tracked and started in one step: a thread started outside it could
        slip past a stop() already under way, which would then report the
        generation gone while the thread still writes.

        Args:
            thread: The thread to start.
            stop_event: Its own stop event, if it has one.

        Returns:
            Whether it was started.
        """
        with self._lock:
            if self._stopping:
                return False
            self._aux_threads.append(thread)
            if stop_event is not None:
                self._aux_stop_events.append(stop_event)
            thread.start()
            return True

    def discard_aux_thread(self, thread, stop_event) -> None:
        """Release a confirmed stopped writer instead of retaining attempt history."""
        with self._lock:
            if thread in self._aux_threads:
                self._aux_threads.remove(thread)
            if stop_event in self._aux_stop_events:
                self._aux_stop_events.remove(stop_event)

    def stop(self, model_instance_id: int, timeout: float) -> bool:
        """Signal every thread of this generation and wait for it to finish.

        Args:
            model_instance_id: The model instance ID, for logging.
            timeout: Total time to wait for all threads (seconds).

        Returns:
            Whether every thread of the generation is now gone.
        """
        self.stop_event.set()
        with self._lock:
            stopping = self._stopping
            self._stopping = True
            stop_events = list(self._aux_stop_events)
            threads = [self.main_thread] + list(self._aux_threads)
        deadline = time.monotonic() + (0 if stopping else timeout)

        for stop_event in stop_events:
            stop_event.set()

        gone = True
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=max(0, deadline - time.monotonic()))
                if thread.is_alive():
                    gone = False
                    if not stopping:
                        logger.warning(
                            f"Log persistence thread {thread.name} for model instance "
                            f"{model_instance_id} did not stop within {timeout}s"
                        )
        return gone


def _describe_workload_failure(workload) -> str:
    """
    Explain why a workload stopped serving, for the instance's state message.

    The workload's `state_message` is the most specific text the runtime has --
    a Pod's admission rejection, or an image-pull reason plus the registry error
    behind it (gpustack/gpustack#5869) -- but it never carries an exit code, and
    a container that merely crashed leaves it empty on Kubernetes. The
    per-container `exits` carry both, so take the reason from there when there
    is no message, and append the exit code either way
    (gpustack/gpustack#4217).

    Args:
        workload: The runtime WorkloadStatus, None if the workload is gone.

    Returns:
        The failure message to surface on the model instance.
    """
    if not workload:
        return _WORKLOAD_FAILED_MESSAGE

    message = getattr(workload, "state_message", "") or ""
    # A container blocked from starting reports no exit code, and its reason is
    # already what the state message is built from, so it adds nothing here.
    exits = [
        exit_
        for exit_ in (getattr(workload, "exits", None) or [])
        if exit_.exit_code is not None
    ]
    if not exits:
        return message or _WORKLOAD_FAILED_MESSAGE

    if not message:
        # Deduplicated but order-preserving: sidecars usually die of one cause.
        message = ", ".join(
            dict.fromkeys(exit_.reason for exit_ in exits if exit_.reason)
        )
    codes = ", ".join(
        (
            f"{exit_.name} exit code {exit_.exit_code}"
            if len(exits) > 1
            else f"exit code {exit_.exit_code}"
        )
        for exit_ in exits
    )
    return f"{message or _WORKLOAD_FAILED_MESSAGE} ({codes})"


class _LocalPortReservationChanged(RuntimeError):
    def __init__(self, ports: Set[int]):
        super().__init__("Local model port reservation changed")
        self.ports = ports


def _model_lifecycle(method, *, blocking: bool = True):
    @wraps(method)
    def serialized(self, *args, **kwargs):
        instance = args[0] if args else kwargs["mi"]
        with self._model_instance_start_lock:
            lock = self._model_lifecycle_locks.get(instance.id)
            if lock is None:
                lock = threading.RLock()
                self._model_lifecycle_locks[instance.id] = lock
        if not lock.acquire(blocking=blocking):
            return
        try:
            return method(self, *args, **kwargs)
        finally:
            lock.release()

    return serialized


class ServeManager(ContainerLogPersister):
    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    """
    The ID of current worker.
    """
    _config: Config
    """
    Global configuration.
    """
    _serve_log_dir: str
    """
    The directory to store logs of serving model instances(in subprocess).
    """

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    """
    The clientset to access the API server.
    """
    _inference_backend_manager: InferenceBackendManager
    """
    The inference backend manager.
    """
    _provisioning_processes: Dict[int, multiprocessing.Process]
    """
    The mapping of model instance ID to provisioning (sub)process.
    When the (sub)process is alive, the model instance is provisioning.
    If the (sub)process exited, the model instance is either running or failed.
    """
    _log_persistence: Dict[int, _LogPersistence]
    """
    The mapping of model instance ID to the current generation of its log
    persistence threads. Replacing an entry retires the previous generation.
    """
    _log_persistence_lock: threading.Lock
    """
    Serializes starting and stopping log persistence, which both the watch
    thread and the periodic sync thread do. A half-applied start would leave
    threads running that nothing can ever signal.
    """
    _error_model_instances: Dict[int, ModelInstance]
    """
    The mapping of model instance ID to error model instances.
    Used to restart error model instances.
    """
    _model_cache_by_instance: Dict[int, Model]
    """
    The cache of models by model instance ID.
    Used to avoid redundant API calls to get model information.
    """
    _model_instance_by_instance_id: Dict[int, ModelInstance]

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
        port_reservations: Optional[WorkerPortReservations] = None,
    ):
        self._worker_id_getter = worker_id_getter
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._clientset_getter = clientset_getter

        self._provisioning_processes = {}
        self._log_persistence = {}
        # None records local cleanup completed before an automatic stop ACK.
        self._legacy_stop_names: Dict[Tuple[int, str], Optional[Set[str]]] = {}
        self._log_persistence_lock = threading.Lock()
        self._error_model_instances = {}
        self._pending_model_instance_errors: Dict[int, dict] = {}
        self._model_cache_by_instance = {}
        self._model_instance_by_instance_id = {}
        self._starting_model_instances: Set[int] = set()
        self._model_instance_start_lock = threading.Lock()
        self._model_lifecycle_locks = WeakValueDictionary()
        self._managed_stopping: Dict[Tuple[int, int], bool] = {}
        self._managed_launches: Dict[Tuple[int, int], Connection] = {}
        self._managed_tasks: Dict[int, Future] = {}
        self._managed_task_executor: Optional[ThreadPoolExecutor] = None
        self._managed_stop_tasks: Dict[int, Future] = {}
        self._managed_stop_executor: Optional[ThreadPoolExecutor] = None
        self._managed_stop_retry_cursor: Optional[int] = None
        self._smg_stopping: Dict[tuple, tuple] = {}
        self._distributed_port_claim_lock = threading.Lock()
        self._distributed_port_client: Optional[DistributedPortClient] = None
        self._distributed_port_claims: Dict[int, UUID] = {}
        self._pending_distributed_port_releases: Set[Tuple[int, UUID]] = set()

        self._port_reservations = port_reservations or WorkerPortReservations()
        self._assigned_ports = self._port_reservations.model_ports
        self._restart_backoff_counts: Dict[int, int] = {}
        # Recognises a member that keeps restarting without ever serving. In
        # this process rather than on the row because the signal is a rate, and
        # the row records only a cumulative count and the last restart's time.
        self._restart_tracker = RestartTracker()

        # Inference health check failure tracking
        # {model_instance_id: failure_count}
        self._inference_health_check_failures: Dict[int, int] = {}

        # Track last successful inference per model instance (set by worker proxy)
        self._last_successful_inference: Dict[int, float] = {}
        # Track last health check time per model instance
        self._last_health_check_time: Dict[int, float] = {}
        self._rank_health_checks: Dict[Tuple, _RankHealthCheck] = {}
        self._rank_health_futures: Dict[Tuple, Future] = {}
        self._rank_health_last_dispatch: Dict[int, float] = {}
        self._rank_health_executor: Optional[ThreadPoolExecutor] = None
        self._rank_health_lock = threading.RLock()

        # Timestamp of the last authoritative (uncached) DB reconciliation in
        # the state sync, for the optional periodic backstop. Starts "now" so
        # the first forced reconciliation is one full period in, not immediate.
        self._last_state_reconcile_time: float = time.time()

        os.makedirs(self._serve_log_dir, exist_ok=True)

    def record_successful_inference(self, instance_id: int):
        """Called by worker proxy on successful inference response."""
        self._last_successful_inference[instance_id] = time.time()

    def _is_local_model_instance(self, mi: ModelInstance) -> bool:
        if get_model_instance_workloads(mi) is not None:
            return self._worker_id in member_worker_ids(mi)
        return mi.get_deployment_metadata(self._worker_id) is not None

    def _managed_workload_labels(self, mi: ModelInstance) -> Dict[str, str]:
        return {
            "gpustack.ai/model-instance-id": str(mi.id),
            "gpustack.ai/restart-count": str(mi.restart_count or 0),
            "gpustack.ai/worker-id": str(self._worker_id),
        }

    @staticmethod
    def _component_workload_labels(item: ModelInstanceWorkload) -> Dict[str, str]:
        if item.component is None:
            return {}
        return {
            "gpustack.ai/component-kind": item.component.kind,
            "gpustack.ai/component-attempt": str(item.component.attempt),
        }

    @staticmethod
    def _managed_launcher_available(mi, model) -> bool:
        from gpustack.utils.model_instance_workers import sglang_smg_mode

        return (
            get_backend(model) == BackendEnum.VLLM
            and mi.role in (None, "prefill", "decode")
            and vllm_lb_mode(model.backend_parameters)
            in ("external", "multi_port_external", "hybrid")
        ) or (mi.role is None and sglang_smg_mode(model))

    @staticmethod
    def _has_hybrid_components(mi: ModelInstance) -> bool:
        return any(
            w.component is not None and w.component.kind == "vllm_hybrid"
            for w in get_model_instance_workloads(mi) or []
        )

    @staticmethod
    def _component_plan_current(mi, item) -> bool:
        if item.component is None:
            return True
        current = next(
            (
                w
                for w in get_model_instance_workloads(mi, item.worker_id) or []
                if w.name == item.name
            ),
            None,
        )
        return bool(
            current is not None
            and current.component is not None
            and current.component.kind == item.component.kind
            and current.component.attempt == item.component.attempt
            and not current.component.stop_requested
            and current.state
            not in (
                ModelInstanceWorkloadStateEnum.STOPPED,
                ModelInstanceWorkloadStateEnum.FAILED,
            )
        )

    def _managed_runtime_workloads(
        self, mi: ModelInstance, *, allow_older_generation: bool = False
    ) -> Dict[str, List[WorkloadStatus]]:
        """Read local matches and probe names omitted by the node-scoped listing.

        Teardown may include older objects only after fresh ownership checks.
        """
        found = {
            item.name: [] for item in get_model_instance_workloads(mi, self._worker_id)
        }
        statuses = list_workloads()
        if statuses is None:
            raise RuntimeError("Runtime workload listing is unknown")
        statuses = list(statuses)
        listed = {status.name for status in statuses}
        # Node-scoped listings can omit admission-gated Pods until binding.
        # A named lookup keeps these pending objects from looking absent.
        for name in found.keys() - listed:
            status = get_workload(name)
            if status is not None:
                if status.name != name:
                    raise RuntimeError("Runtime workload lookup identity mismatch")
                statuses.append(status)
        local = {
            item.name: item
            for item in get_model_instance_workloads(mi, self._worker_id)
        }
        labels = self._managed_workload_labels(mi)
        for status in statuses:
            if status.name not in found:
                continue
            if any(
                (status.labels or {}).get(key) != value
                for key, value in {
                    **labels,
                    **self._component_workload_labels(local[status.name]),
                }.items()
            ) and not (
                allow_older_generation
                and self._can_cleanup_older_workload(mi, local[status.name], status)
            ):
                raise RuntimeError(
                    f"Workload {status.name} does not match this instance generation"
                )
            found[status.name].append(status)
        return found

    def _can_cleanup_older_workload(self, mi, item, status) -> bool:
        """Authorize older runtime cleanup against the current registered owner."""
        labels = status.labels or {}
        if any(
            labels.get(key) != value
            for key, value in {
                'gpustack.ai/model-instance-id': str(mi.id),
                'gpustack.ai/worker-id': str(self._worker_id),
                'gpustack.ai/component-kind': (
                    item.component.kind if item.component is not None else None
                ),
            }.items()
        ):
            return False
        try:
            restart = int(labels.get('gpustack.ai/restart-count', ''))
        except (TypeError, ValueError):
            return False
        if not 0 <= restart < (mi.restart_count or 0):
            return False
        current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
        registered = next(
            (
                workload
                for workload in get_model_instance_workloads(current, self._worker_id)
                or []
                if workload.name == item.name
            ),
            None,
        )
        return bool(
            current.id == mi.id
            and current.restart_count == mi.restart_count
            and current.cluster_id == mi.cluster_id
            and self._worker_id in member_worker_ids(current)
            and registered is not None
            and self._component_workload_labels(registered)
            == self._component_workload_labels(item)
        )

    def _hold_managed_instance(self, mi: ModelInstance) -> None:
        with self._model_instance_start_lock:
            self._model_instance_by_instance_id[mi.id] = mi.model_copy(deep=True)
        ports = model_instance_ports(mi, self._worker_id)
        if (mi.id, mi.restart_count or 0) in self._managed_launches:
            ports |= self._assigned_ports.get(mi.id, set())
        self._port_reservations.hold_model(mi.id, ports)

    def _report_managed_workloads(
        self, mi: ModelInstance, workloads: List[ModelInstanceWorkload], **diagnostic
    ) -> ModelInstance:
        update = worker_workload_report(mi, self._worker_id, workloads, **diagnostic)
        result = self._clientset.model_instances.report_workloads(
            id=mi.id, report=update
        )
        return ModelInstance.model_validate(result)

    @staticmethod
    def _managed_stop_requested(mi: ModelInstance) -> bool:
        if mi.state == ModelInstanceStateEnum.ERROR:
            return True
        return bool(
            mi.draining_since is not None
            and mi.distributed_servers is not None
            and mi.distributed_servers.stop_requested
        )

    @staticmethod
    def _preserve_registered_observations(local, previous) -> None:
        """Preparation cannot reset the observations or Server-owned recovery intent."""
        for item in local:
            old = previous.get(item.name)
            if old:
                if item.component is not None and old.component is not None:
                    item.component.ready = old.component.ready
                    item.component.stop_requested = old.component.stop_requested
                    item.component.last_restart_time = old.component.last_restart_time
                item.state = old.state
                ready = {
                    (p.dp_rank, p.node_rank): p.ready for p in old.dp_rank_placements
                }
                for placement in item.dp_rank_placements:
                    placement.ready = ready.get(
                        (placement.dp_rank, placement.node_rank), placement.ready
                    )

    @_model_lifecycle
    def _start_managed_workloads(
        self,
        mi: ModelInstance,
        workloads: List[ModelInstanceWorkload],
        plans: List[WorkloadPlan],
    ) -> bool:
        """Register resolved plans in the Worker's default runtime namespace."""
        key = (mi.id, mi.restart_count or 0)
        registered = starting = False
        try:
            current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
            if (
                get_model_instance_workloads(current) is None
                or current.restart_count != mi.restart_count
                or current.cluster_id != mi.cluster_id
                or self._worker_id not in member_worker_ids(current)
                or self._managed_stop_requested(current)
                or current.draining_since is not None
                or key in self._managed_stopping
                or any(
                    item.state
                    in (
                        ModelInstanceWorkloadStateEnum.FAILED,
                        ModelInstanceWorkloadStateEnum.STOPPED,
                    )
                    and (item.component is None or item.component.kind != 'sglang_smg')
                    for item in get_model_instance_workloads(current, self._worker_id)
                )
            ):
                return False
            tracked = self._model_instance_by_instance_id.get(mi.id)
            if tracked and tracked.restart_count != current.restart_count:
                return False
            local = [
                ModelInstanceWorkload.model_validate(item.model_dump())
                for item in workloads
            ]
            validate_workload_manifest(local)
            if (
                not local
                or any(item.worker_id != self._worker_id for item in local)
                or len(plans) != len(local)
                or {plan.name for plan in plans} != {item.name for item in local}
                or any(plan.namespace is not None for plan in plans)
            ):
                raise ValueError(
                    "Runtime plans must match the local manifest and use the Worker's default namespace"
                )
            previous = {
                item.name: item
                for item in get_model_instance_workloads(current, self._worker_id)
            }
            self._preserve_registered_observations(local, previous)
            staged = ModelInstance.model_validate(current)
            staged.distributed_servers = staged.distributed_servers.model_copy(
                deep=True
            )
            staged.distributed_servers.workloads = [
                item
                for item in staged.distributed_servers.workloads
                if item.worker_id != self._worker_id
            ] + local
            self._hold_managed_instance(staged)
            with self._model_instance_start_lock:
                self._starting_model_instances.add(mi.id)
            starting = True
            mi = self._report_managed_workloads(current, local)
            registered = True
            self._hold_managed_instance(mi)
            observed = self._managed_runtime_workloads(mi)
            registered_states = {
                item.name: item.state
                for item in get_model_instance_workloads(mi, self._worker_id)
            }
            self._ensure_managed_log_persistence(mi)
            if not self._create_registered_plans(
                mi, local, plans, observed, registered_states
            ):
                return False
            return key not in self._managed_stopping
        except Exception as exc:
            logger.warning("Cannot start managed model instance %s: %s", mi.id, exc)
            if registered:
                self._report_managed_launch_failure(mi)
                with self._model_instance_start_lock:
                    self._managed_stopping.setdefault(key, False)
            return False
        finally:
            if starting:
                with self._model_instance_start_lock:
                    self._starting_model_instances.discard(mi.id)
            with self._model_instance_start_lock:
                delete_logs = self._managed_stopping.get(key)
            if registered and delete_logs is not None:
                self._stop_model_instance(mi, delete_logs=delete_logs)

    def _component_create_invalid(self, mi, item, latest):
        if (
            latest.restart_count != mi.restart_count
            or latest.cluster_id != mi.cluster_id
            or self._worker_id not in member_worker_ids(latest)
            or self._managed_stop_requested(latest)
            or latest.draining_since is not None
            or (mi.id, mi.restart_count or 0) in self._managed_stopping
        ):
            return True
        return not self._component_plan_current(latest, item)

    def _handle_cpu_create_failure(self, mi, item, latest):
        if (
            item.component is None
            or item.component.kind != 'sglang_smg'
            or latest.restart_count != mi.restart_count
            or latest.cluster_id != mi.cluster_id
            or self._worker_id not in member_worker_ids(latest)
            or self._managed_stop_requested(latest)
            or latest.draining_since is not None
        ):
            return False
        self._stop_smg(mi, item)
        return True

    def _create_registered_plans(self, mi, local, plans, observed, states):
        """Keep CPU plan failures outside the Engine generation stop intent."""
        key = (mi.id, mi.restart_count or 0)
        by_name = {item.name: item for item in local}
        for plan in plans:
            item = by_name[plan.name]
            latest = mi
            try:
                latest = self._clientset.model_instances.get(id=mi.id, use_cache=False)
                if self._component_create_invalid(mi, item, latest):
                    if self._handle_cpu_create_failure(mi, item, latest):
                        return True
                    with self._model_instance_start_lock:
                        self._managed_stopping.setdefault(key, False)
                    return False
                existing = observed[plan.name]
                if existing:
                    if len(existing) != 1 or existing[0].state not in (
                        WorkloadStatusStateEnum.RUNNING,
                        WorkloadStatusStateEnum.PENDING,
                        WorkloadStatusStateEnum.INITIALIZING,
                    ):
                        raise RuntimeError(
                            f'Registered workload {plan.name} cannot be recreated'
                        )
                    continue
                if states[
                    plan.name
                ] != ModelInstanceWorkloadStateEnum.PENDING or latest.state in (
                    ModelInstanceStateEnum.RUNNING,
                    ModelInstanceStateEnum.UNREACHABLE,
                ):
                    raise RuntimeError(
                        f'Workload {plan.name} needs a new generation before recovery'
                    )
                plan.labels = {
                    **(plan.labels or {}),
                    **self._managed_workload_labels(mi),
                    **self._component_workload_labels(item),
                }
                create_workload(plan)
                if item.component is not None:
                    latest = self._clientset.model_instances.get(
                        id=mi.id, use_cache=False
                    )
                    if self._component_create_invalid(mi, item, latest):
                        if self._handle_cpu_create_failure(mi, item, latest):
                            return True
                        with self._model_instance_start_lock:
                            self._managed_stopping.setdefault(key, False)
                        return False
            except Exception:
                if self._handle_cpu_create_failure(mi, item, latest):
                    return True
                raise
        return True

    def _adopt_managed_model_instance(
        self, mi: ModelInstance, tracked: Optional[ModelInstance]
    ) -> bool:
        """Retain ownership, finishing any older generation before adopting this one."""
        if tracked and tracked.restart_count != mi.restart_count:
            if (tracked.restart_count or 0) > (mi.restart_count or 0):
                return False
            self._schedule_managed_stop(tracked)
            return False
        if not self._is_local_model_instance(mi):
            if tracked:
                self._schedule_managed_stop(tracked)
            return False
        local = get_model_instance_workloads(mi, self._worker_id)
        if local is None:
            return False
        if (
            tracked
            and not local
            and get_model_instance_workloads(tracked, self._worker_id)
        ):
            self._schedule_managed_stop(tracked)
            return False
        self._hold_managed_instance(mi)
        key = (mi.id, mi.restart_count or 0)
        if self._managed_stop_requested(mi) or key in self._managed_stopping:
            self._schedule_managed_stop(mi)
            return False
        return True

    def _observe_managed_workloads(
        self, mi: ModelInstance
    ) -> List[ModelInstanceWorkload]:
        try:
            observed = self._managed_runtime_workloads(mi)
        except Exception as exc:
            logger.warning("Cannot observe managed model instance %s: %s", mi.id, exc)
            observed = None
        states = {
            WorkloadStatusStateEnum.RUNNING: ModelInstanceWorkloadStateEnum.RUNNING,
            WorkloadStatusStateEnum.PENDING: ModelInstanceWorkloadStateEnum.PENDING,
            WorkloadStatusStateEnum.INITIALIZING: ModelInstanceWorkloadStateEnum.PENDING,
            WorkloadStatusStateEnum.UNHEALTHY: ModelInstanceWorkloadStateEnum.FAILED,
            WorkloadStatusStateEnum.FAILED: ModelInstanceWorkloadStateEnum.FAILED,
            WorkloadStatusStateEnum.INACTIVE: ModelInstanceWorkloadStateEnum.FAILED,
        }
        report = [
            item.model_copy(deep=True)
            for item in get_model_instance_workloads(mi, self._worker_id)
        ]
        for item in report:
            if item.component is not None and item.state in (
                ModelInstanceWorkloadStateEnum.STOPPED,
                ModelInstanceWorkloadStateEnum.FAILED,
            ):
                continue
            matches = observed.get(item.name) if observed is not None else None
            if matches is None or len(matches) > 1:
                item.state = ModelInstanceWorkloadStateEnum.UNKNOWN
            elif not matches:
                item.state = (
                    ModelInstanceWorkloadStateEnum.PENDING
                    if item.component is not None
                    and item.component.kind == 'sglang_smg'
                    and item.state == ModelInstanceWorkloadStateEnum.PENDING
                    and item.component.attempt > 0
                    else ModelInstanceWorkloadStateEnum.FAILED
                )
            else:
                item.state = states.get(
                    matches[0].state, ModelInstanceWorkloadStateEnum.UNKNOWN
                )
            if item.state != ModelInstanceWorkloadStateEnum.RUNNING:
                if item.component is not None and item.component.api_port is not None:
                    item.component.ready = False
                for placement in item.dp_rank_placements:
                    if placement.node_rank == 0:
                        placement.ready = False
        return report

    def _report_managed_launch_failure(self, mi: ModelInstance) -> None:
        """Publish observed failure or uncertainty before cleanup can remain pending."""
        try:
            observed = self._observe_managed_workloads(mi)
            current = self._report_managed_workloads(mi, observed)
            self._hold_managed_instance(current)
        except Exception as exc:
            logger.warning(
                "Cannot report managed launch failure for %s: %s", mi.id, exc
            )

    @staticmethod
    def _has_sglang_components(mi: ModelInstance) -> bool:
        workloads = get_model_instance_workloads(mi) or []
        return bool(workloads) and all(
            w.component is not None
            and w.component.kind in ("sglang_engine", "sglang_smg")
            for w in workloads
        )

    def _smg_key(self, mi, item):
        return (mi.id, mi.restart_count or 0, item.name, item.component.attempt)

    def _stop_smg(self, mi, item) -> bool:
        """Confirm one exact CPU task stopped while retaining Engine ownership."""
        key = self._smg_key(mi, item)
        with self._model_instance_start_lock:
            self._smg_stopping[key] = (
                mi.model_copy(deep=True),
                item.model_copy(deep=True),
            )
        try:
            connection = self._managed_launches.pop(
                (mi.id, mi.restart_count or 0), None
            )
            if connection is not None:
                connection.close()
            if self._is_provisioning(mi):
                terminate_multiprocessing_process(self._provisioning_processes[mi.id])
                if self._is_provisioning(mi):
                    return False
            # Observe the old identity even when the Server has replaced it.
            old = mi.model_copy(deep=True)
            old.distributed_servers = old.distributed_servers.model_copy(deep=True)
            old.distributed_servers.workloads = [item.model_copy(deep=True)]
            observed = self._managed_runtime_workloads(old)
            if observed[item.name]:
                delete_workload(item.name)
            if self._managed_runtime_workloads(old)[item.name]:
                return False
            worker = self._clientset.workers.get(self._worker_id, use_cache=False)
            if worker.state not in (
                WorkerStateEnum.READY,
                WorkerStateEnum.MAINTENANCE,
            ) or getattr(worker, 'unreachable', False):
                return False
            try:
                current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
            except NotFoundException:
                current = None
            if current is not None and (current.restart_count or 0) == (
                mi.restart_count or 0
            ):
                local = get_model_instance_workloads(current, self._worker_id) or []
                report = [w.model_copy(deep=True) for w in local]
                target = next((w for w in report if w.name == item.name), None)
                if target is not None and target.component == item.component:
                    target.state = ModelInstanceWorkloadStateEnum.STOPPED
                    target.component.ready = False
                    if report != local:
                        current = self._report_managed_workloads(current, report)
                        self._hold_managed_instance(current)
            with self._model_instance_start_lock:
                self._smg_stopping.pop(key, None)
            return True
        except Exception as exc:
            logger.warning('Cannot confirm SMG %s stopped: %s', item.name, exc)
            return False

    def _smg_create_current(self, mi, item):
        try:
            current = ModelInstance.model_validate(
                self._clientset.model_instances.get(
                    id=mi.id, use_cache=False, check_configuration=True
                )
            )
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            # A stale spec blocks CPU creation while Engine observation continues.
            return None
        if (
            current.deleted_at is not None
            or (current.restart_count or 0) != (mi.restart_count or 0)
            or current.cluster_id != mi.cluster_id
            or self._worker_id != current.worker_id
            or current.draining_since is not None
            or self._managed_stop_requested(current)
            or current.state != ModelInstanceStateEnum.STARTING
            or not self._component_plan_current(current, item)
            or (current.id, current.restart_count or 0) in self._managed_stopping
        ):
            return None
        target = next(
            (
                w
                for w in get_model_instance_workloads(current, self._worker_id)
                if w.name == item.name
            ),
            None,
        )
        if target is None or target.state != ModelInstanceWorkloadStateEnum.PENDING:
            return None
        model = self._clientset.models.get(current.model_id, use_cache=False)
        if (
            getattr(model, 'deleted_at', None) is not None
            or model.replicas == 0
            or not model.restart_on_error
            or model.restarting_since is not None
            # The checked read validates this instance; aggregate staleness can
            # describe another replica. Unstamped and PD deployments stay fenced.
            or (model.stale and (current.group_id or not current.spec_digest))
        ):
            return None
        worker = self._clientset.workers.get(self._worker_id, use_cache=False)
        if (
            worker.state != WorkerStateEnum.READY
            or getattr(worker, 'unreachable', False)
            or getattr(worker, 'me', None) is not True
            or getattr(worker, 'id', self._worker_id) != self._worker_id
            or getattr(worker, 'cluster_id', current.cluster_id) != current.cluster_id
        ):
            return None
        return current, model, worker

    @staticmethod
    def _smg_preparation_identity(authority):
        current, model, worker = authority
        return (
            model.model_dump(
                include={
                    'id',
                    'cluster_id',
                    'owner_principal_id',
                    'backend',
                    'backend_version',
                    'backend_parameters',
                    'image_name',
                    'run_command',
                    'env',
                }
            ),
            (
                getattr(worker, 'id', None),
                getattr(worker, 'cluster_id', None),
                getattr(worker, 'me', None),
                worker.ip,
            ),
            current.worker_ip,
            [
                w.model_dump()
                for w in get_model_instance_workloads(current)
                if w.component.kind == 'sglang_engine'
            ],
        )

    def _smg_authority_matches(self, mi, item, identity):
        authority = self._smg_create_current(mi, item)
        return (
            authority is not None
            and self._smg_preparation_identity(authority) == identity
        )

    def _create_pending_smg(self, mi, item) -> None:
        """Create only a Server-registered CPU attempt using its Engine image."""
        authority = self._smg_create_current(mi, item)
        if authority is None:
            return
        current, model, worker = authority
        identity = self._smg_preparation_identity(authority)
        engines = [
            w
            for w in get_model_instance_workloads(current)
            if w.component.kind == 'sglang_engine'
        ]
        main = next((w for w in engines if w.worker_id == current.worker_id), None)
        if (
            len(engines) != len(member_worker_ids(current))
            or any(w.state != ModelInstanceWorkloadStateEnum.RUNNING for w in engines)
            or main is None
            or not main.component.ready
        ):
            return
        if not main.component.image or main.component.router_version != '0.3.2':
            message = (
                'SMG recovery is blocked: Engine image or compatible SMG version '
                'is missing.'
            )
            if current.state_message != message:
                current = self._report_managed_workloads(
                    current,
                    get_model_instance_workloads(current, self._worker_id),
                    state_message=message,
                )
                self._hold_managed_instance(current)
            return
        observed = self._managed_runtime_workloads(current)
        if (
            len(observed.get(main.name, [])) != 1
            or observed[main.name][0].state != WorkloadStatusStateEnum.RUNNING
        ):
            return
        if observed.get(item.name):
            return
        _, policy = split_sglang_smg_parameters(
            model.backend_parameters, {'sglang-router': main.component.router_version}
        )
        plan = build_sglang_smg_workload(
            item.name,
            main.component.image,
            policy,
            item.component,
            worker.ip,
            f'http://{current.worker_ip}:{main.component.api_port}',
        )
        fallback = registration.determine_default_registry(
            self._config.system_default_container_registry
        )
        plan = transform_workload_plan(self._config, plan, fallback)
        plan.labels = {
            **(plan.labels or {}),
            **self._managed_workload_labels(current),
            **self._component_workload_labels(item),
        }
        # CPU preparation is pure and synchronous under the instance lifecycle
        # lock; a stop cannot acknowledge absence while this create is in flight.
        if not self._smg_authority_matches(current, item, identity):
            return
        try:
            create_workload(plan)
            if not self._smg_authority_matches(current, item, identity):
                self._stop_smg(current, item)
        except Exception as exc:
            logger.warning('SMG %s creation is uncertain: %s', item.name, exc)
            self._stop_smg(current, item)

    def _same_component_epoch(self, previous, current):
        return (
            (previous.restart_count or 0) == (current.restart_count or 0)
            and previous.cluster_id == current.cluster_id
            and self._worker_id in member_worker_ids(current)
            and self._has_sglang_components(current)
        )

    def _retire_smg_attempts(self, current, tracked, local):
        # Retire stale CPU identities without replacing the adopted Engine.
        for old in (
            get_model_instance_workloads(tracked, self._worker_id) or []
            if tracked
            else []
        ):
            if (
                old.component
                and old.component.kind == 'sglang_smg'
                and not any(w.name == old.name for w in local)
            ):
                with self._model_instance_start_lock:
                    self._smg_stopping[self._smg_key(tracked, old)] = (
                        tracked.model_copy(deep=True),
                        old.model_copy(deep=True),
                    )
        with self._model_instance_start_lock:
            pending = [
                (old_mi, old)
                for key, (old_mi, old) in self._smg_stopping.items()
                if key[0] == current.id
            ]
        for old_mi, old in pending:
            self._schedule_managed_task(
                old_mi, partial(self._retry_smg_stop, item=old), teardown=True
            )
        return not pending

    def _sync_sglang_components(self, current, tracked) -> None:
        local = get_model_instance_workloads(current, self._worker_id)
        if any(
            w.component.kind == 'sglang_engine'
            and w.state
            in (
                ModelInstanceWorkloadStateEnum.FAILED,
                ModelInstanceWorkloadStateEnum.STOPPED,
            )
            for w in local
        ):
            self._schedule_managed_stop(current)
            return
        if not self._retire_smg_attempts(current, tracked, local):
            self._observe_sglang_components(current)
            return
        smg = next((w for w in local if w.component.kind == 'sglang_smg'), None)
        if smg and (
            smg.component.stop_requested
            or smg.state
            in (
                ModelInstanceWorkloadStateEnum.FAILED,
                ModelInstanceWorkloadStateEnum.STOPPED,
            )
        ):
            self._schedule_managed_task(
                current, partial(self._retry_smg_stop, item=smg), teardown=True
            )
            self._observe_sglang_components(current)
            return
        if (
            smg
            and smg.state == ModelInstanceWorkloadStateEnum.PENDING
            and smg.component.attempt > 0
        ):
            if self._schedule_managed_task(
                current, partial(self._prepare_pending_smg, item=smg)
            ):
                return
        self._observe_sglang_components(current)

    @_model_lifecycle
    def _prepare_pending_smg(self, mi, item):
        try:
            self._create_pending_smg(mi, item)
        except Exception as exc:
            logger.warning('Cannot prepare SMG %s: %s', item.name, exc)
        try:
            self._observe_sglang_components(mi, background=True)
        except Exception as exc:
            logger.warning('Cannot observe SGLang components %s: %s', mi.id, exc)

    def _observe_sglang_components(self, current, *, background=False):
        fresh = ModelInstance.model_validate(
            self._clientset.model_instances.get(id=current.id, use_cache=False)
        )
        if not self._same_component_epoch(current, fresh):
            return
        current = fresh
        local = get_model_instance_workloads(current, self._worker_id)
        if self._managed_stop_requested(current):
            if background:
                self._stop_model_instance(current)
            else:
                self._schedule_managed_stop(current)
            return
        report = self._observe_managed_workloads(current)
        if report != local:
            current = self._report_managed_workloads(current, report)
            self._hold_managed_instance(current)
        failed = [w for w in report if w.state == ModelInstanceWorkloadStateEnum.FAILED]
        if any(w.component.kind == 'sglang_engine' for w in failed):
            if background:
                self._stop_model_instance(current)
            else:
                self._schedule_managed_stop(current)
        elif failed:
            if background:
                self._stop_smg(current, failed[0])
            else:
                self._schedule_managed_task(
                    current,
                    partial(self._retry_smg_stop, item=failed[0]),
                    teardown=True,
                )
        self._sync_managed_rank_health(current)

    def _resume_managed_stop(self, mi):
        key = (mi.id, mi.restart_count or 0)
        with self._model_instance_start_lock:
            delete_logs = self._managed_stopping.get(key)
        if delete_logs is None:
            return False
        self._schedule_managed_stop(mi, delete_logs=delete_logs)
        return True

    @partial(_model_lifecycle, blocking=False)
    def _sync_managed_model_instance(self, mi: ModelInstance) -> None:
        tracked = self._model_instance_by_instance_id.get(mi.id)
        if tracked and self._resume_managed_stop(tracked):
            return
        try:
            current = ModelInstance.model_validate(
                self._clientset.model_instances.get(id=mi.id, use_cache=False)
            )
        except NotFoundException:
            self._schedule_managed_stop(tracked or mi, delete_logs=True)
            return
        except Exception as exc:
            logger.warning("Cannot reconcile managed model instance %s: %s", mi.id, exc)
            return
        if not self._adopt_managed_model_instance(current, tracked):
            return
        local = get_model_instance_workloads(current, self._worker_id)
        if not local:
            if (
                current.state
                in (
                    ModelInstanceStateEnum.SCHEDULED,
                    ModelInstanceStateEnum.INITIALIZING,
                    ModelInstanceStateEnum.STARTING,
                )
                and current.draining_since is None
            ):
                if (current.id, current.restart_count or 0) in self._managed_launches:
                    self._schedule_managed_task(
                        current, self._prepare_managed_model_instance
                    )
                else:
                    self._prepare_managed_model_instance(current)
            return
        if mi.id in self._starting_model_instances:
            return
        self._ensure_managed_log_persistence(current)
        self._sync_registered_workloads(current, tracked)

    def _ensure_managed_log_persistence(self, mi: ModelInstance) -> None:
        """Reconcile accepted Engine logs without turning logging into launch failure."""
        if mi.draining_since is not None or not any(
            (not item.component or item.component.kind != 'sglang_smg')
            and item.state
            not in (
                ModelInstanceWorkloadStateEnum.FAILED,
                ModelInstanceWorkloadStateEnum.STOPPED,
            )
            for item in get_model_instance_workloads(mi, self._worker_id) or []
        ):
            return
        try:
            self._ensure_container_log_persistence(mi)
        except Exception as exc:
            logger.warning('Cannot attach Engine logs for instance %s: %s', mi.id, exc)

    def _schedule_managed_stop(
        self, mi: ModelInstance, *, delete_logs: bool = False
    ) -> None:
        """Retain whole-instance teardown intent even when task admission is busy."""
        key = (mi.id, mi.restart_count or 0)
        with self._model_instance_start_lock:
            self._managed_stopping[key] = (
                self._managed_stopping.get(key, False) or delete_logs
            )
            if mi.id not in self._model_instance_by_instance_id:
                self._model_instance_by_instance_id[mi.id] = mi.model_copy(deep=True)
        self._schedule_managed_task(
            mi,
            partial(self._stop_model_instance, delete_logs=delete_logs),
            teardown=True,
        )

    def _schedule_managed_task(
        self,
        mi: ModelInstance,
        operation: Callable[[ModelInstance], None],
        *,
        teardown: bool = False,
    ) -> bool:
        """Keep slow creation and teardown outside shared reconciliation loops."""
        with self._model_instance_start_lock:
            for tasks in (self._managed_tasks, self._managed_stop_tasks):
                for instance_id, job in list(tasks.items()):
                    if job.done():
                        try:
                            job.result()
                        except Exception as exc:
                            logger.warning(
                                'Cannot reconcile managed model instance %s: %s',
                                instance_id,
                                exc,
                            )
                        tasks.pop(instance_id)
            tasks = self._managed_stop_tasks if teardown else self._managed_tasks
            if (
                mi.id in self._managed_tasks
                or mi.id in self._managed_stop_tasks
                or len(tasks) >= _MANAGED_TASK_CONCURRENCY
            ):
                return False
            executor = (
                self._managed_stop_executor if teardown else self._managed_task_executor
            )
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=_MANAGED_TASK_CONCURRENCY,
                    thread_name_prefix=(
                        'managed-stop' if teardown else 'managed-lifecycle'
                    ),
                )
                if teardown:
                    self._managed_stop_executor = executor
                else:
                    self._managed_task_executor = executor
            tasks[mi.id] = executor.submit(operation, mi.model_copy(deep=True))
            return True

    def _sync_registered_workloads(self, current, tracked):
        local = get_model_instance_workloads(current, self._worker_id)
        if self._has_sglang_components(current):
            try:
                self._sync_sglang_components(current, tracked)
            except Exception as exc:
                logger.warning(
                    'Cannot reconcile SGLang components %s: %s', current.id, exc
                )
            return
        if any(
            item.state
            in (
                ModelInstanceWorkloadStateEnum.FAILED,
                ModelInstanceWorkloadStateEnum.STOPPED,
            )
            for item in local
        ):
            self._schedule_managed_stop(current)
            return
        report = self._observe_managed_workloads(current)
        try:
            if (
                tracked is None
                or tracked.restart_count != current.restart_count
                or report != local
            ):
                current = self._report_managed_workloads(current, report)
                self._hold_managed_instance(current)
            if any(
                item.state == ModelInstanceWorkloadStateEnum.FAILED for item in report
            ):
                self._schedule_managed_stop(current)
            else:
                self._sync_managed_rank_health(
                    current, startup_only=tracked is not None
                )
        except Exception as exc:
            logger.warning(
                "Cannot report managed model instance %s: %s", current.id, exc
            )

    @_model_lifecycle
    def _prepare_managed_model_instance(self, mi: ModelInstance) -> None:
        """Receive resolved plans from the preparer; create only through T4 fences."""
        current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
        if (
            get_model_instance_workloads(current) is None
            or current.restart_count != mi.restart_count
            or get_model_instance_workloads(current, self._worker_id)
            or self._worker_id not in member_worker_ids(current)
            or self._managed_stop_requested(current)
            or (current.id, current.restart_count or 0) in self._managed_stopping
            or current.draining_since is not None
            or current.state
            not in (
                ModelInstanceStateEnum.SCHEDULED,
                ModelInstanceStateEnum.INITIALIZING,
                ModelInstanceStateEnum.STARTING,
            )
        ):
            return
        mi = ModelInstance.model_validate(current)
        key = (mi.id, mi.restart_count or 0)
        connection = self._managed_launches.get(key)
        if connection is None:
            model = self._get_model(mi)
            if self._managed_launcher_available(mi, model):
                self._launch_model_instance(mi)
            return
        if not connection.poll():
            if self._is_provisioning(mi):
                return
            self._managed_launches.pop(key).close()
            self._handle_model_instance_launch_failure(
                mi, RuntimeError("Model preparation ended without a workload plan")
            )
            return
        try:
            workloads, plans = connection.recv()
            model = self._clientset.models.get(mi.model_id, use_cache=False)
            if mi.group_id and getattr(model, "stale", None) is True:
                raise ValueError(
                    "The deployment configuration has changed; restart the model instead."
                )
            if not self._start_managed_workloads(mi, workloads, plans):
                self._handle_model_instance_launch_failure(
                    mi, RuntimeError("Managed workload registration or creation failed")
                )
        except (EOFError, OSError, ValueError, httpx.TransportError) as exc:
            self._handle_model_instance_launch_failure(mi, exc)
        finally:
            self._managed_launches.pop(key, None)
            connection.close()

    def _delete_managed_workloads(self, mi: ModelInstance) -> bool:
        self._clear_rank_health(mi.id)
        connection = self._managed_launches.pop((mi.id, mi.restart_count or 0), None)
        if connection is not None:
            connection.close()
        if self._is_provisioning(mi):
            terminate_multiprocessing_process(self._provisioning_processes[mi.id])
            if self._is_provisioning(mi):
                return False
        observed = self._managed_runtime_workloads(mi, allow_older_generation=True)
        self._wait_for_managed_engine_logs(mi, observed)
        for name, matches in observed.items():
            if matches:
                try:
                    delete_workload(name)
                except Exception as exc:
                    logger.warning(
                        "Managed workload %s deletion is pending: %s", name, exc
                    )
        # Runtime deletion may only be accepted. Even INACTIVE/FAILED objects
        # still reserve their identity until runtime reads confirm absence.
        return not any(
            self._managed_runtime_workloads(mi, allow_older_generation=True).values()
        )

    def _wait_for_managed_engine_logs(self, mi: ModelInstance, observed) -> None:
        """Give existing collectors a bounded flush window before runtime deletion."""
        names = {
            name
            for name, _ in engine_log_targets(mi, self._worker_id)
            if any(
                (status.labels or {}).get('gpustack.ai/restart-count')
                == str(mi.restart_count or 0)
                and getattr(status, 'loggable', None) != []
                for status in observed.get(name, [])
            )
        }
        if not names:
            return
        self._record_log_sources(mi)
        deadline = time.monotonic() + _MANAGED_LOG_FLUSH_TIMEOUT
        previous = None
        while time.monotonic() < deadline:
            try:
                sources = list_log_sources(
                    Path(self._serve_log_dir),
                    mi.id,
                    self._worker_id,
                    mi.restart_count or 0,
                )
                try:
                    snapshot = {
                        (source.descriptor.workload_name, str(path)): (
                            stat.st_size,
                            stat.st_mtime_ns,
                        )
                        for source in sources
                        if not source.descriptor.previous
                        and source.descriptor.kind == 'workload'
                        and source.descriptor.workload_name in names
                        for path in source.paths
                        for stat in [path.stat()]
                    }
                except OSError:
                    # Rotation can remove a shard after source discovery. Only
                    # complete consecutive snapshots establish archive stability.
                    previous = None
                else:
                    captured = {
                        name for (name, _), (size, _) in snapshot.items() if size
                    }
                    if names <= captured and snapshot == previous:
                        return
                    previous = snapshot
            except Exception as exc:
                logger.warning('Cannot inspect Engine archives for %s: %s', mi.id, exc)
                break
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        logger.warning(
            'Engine log flush window expired for %s; retained logs may be incomplete',
            mi.id,
        )

    def _confirm_managed_stop(self, mi: ModelInstance) -> bool:
        worker = self._clientset.workers.get(self._worker_id, use_cache=False)
        if worker.state not in (
            WorkerStateEnum.READY,
            WorkerStateEnum.MAINTENANCE,
        ) or getattr(worker, "unreachable", False):
            return False
        try:
            current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
        except NotFoundException:
            current = None
        if (
            current is not None
            and current.restart_count == mi.restart_count
            and current.cluster_id == mi.cluster_id
            and self._worker_id in member_worker_ids(current)
        ):
            local = get_model_instance_workloads(current, self._worker_id)
            stopped = [item.model_copy(deep=True) for item in local]
            for item in stopped:
                item.state = ModelInstanceWorkloadStateEnum.STOPPED
                if item.component is not None and item.component.api_port is not None:
                    item.component.ready = False
                for placement in item.dp_rank_placements:
                    if placement.node_rank == 0:
                        placement.ready = False
            if stopped != local:
                self._report_managed_workloads(mi, stopped)
        return True

    @_model_lifecycle
    def _retry_smg_stop(self, mi, item):
        self._stop_smg(mi, item)

    def _retry_managed_stops(self) -> None:
        """Continue accepted local teardown even when Server snapshots are unavailable."""
        with self._model_instance_start_lock:
            smg_stopping = list(self._smg_stopping.values())
            managed_stopping = dict(self._managed_stopping.items())
            cursor = self._managed_stop_retry_cursor
        pending = {}
        for (instance_id, restart_count), delete_logs in managed_stopping.items():
            mi = self._model_instance_by_instance_id.get(instance_id)
            if (
                mi is not None
                and (mi.restart_count or 0) == restart_count
                and instance_id not in self._starting_model_instances
            ):
                pending[instance_id] = (
                    mi,
                    partial(self._stop_model_instance, delete_logs=delete_logs),
                )
        for old_mi, item in smg_stopping:
            if (old_mi.id, old_mi.restart_count or 0) not in managed_stopping:
                pending.setdefault(
                    old_mi.id, (old_mi, partial(self._retry_smg_stop, item=item))
                )
        # Rotate across instances; whole-instance intent wins within each instance.
        for instance_id in sorted(
            pending,
            key=lambda identity: (cursor is not None and identity <= cursor, identity),
        ):
            mi, operation = pending[instance_id]
            if self._schedule_managed_task(mi, operation, teardown=True):
                with self._model_instance_start_lock:
                    self._managed_stop_retry_cursor = instance_id

    def _stop_managed_model_instance(
        self, mi: ModelInstance, delete_logs: bool
    ) -> bool:
        key = (mi.id, mi.restart_count or 0)
        tracked = self._model_instance_by_instance_id.get(mi.id)
        if tracked and tracked.restart_count != mi.restart_count:
            with self._model_instance_start_lock:
                self._managed_stopping.pop(key, None)
            connection = self._managed_launches.pop(key, None)
            if connection is not None:
                connection.close()
            return (tracked.restart_count or 0) > (mi.restart_count or 0)
        if tracked and get_model_instance_workloads(tracked, self._worker_id):
            mi = tracked
        self._hold_managed_instance(mi)
        if mi.id in self._starting_model_instances:
            return False
        try:
            if not self._delete_managed_workloads(mi):
                return False
            with self._model_instance_start_lock:
                pending_smg = [
                    (old_mi, item)
                    for component_key, (old_mi, item) in self._smg_stopping.items()
                    if component_key[:2] == key
                ]
            for old_mi, item in pending_smg:
                if not self._stop_smg(old_mi, item):
                    return False
            if not self._confirm_managed_stop(mi):
                return False
        except Exception as exc:
            logger.warning(
                "Cannot confirm managed model instance %s stopped: %s", mi.id, exc
            )
            return False
        self._stop_container_log_persistence(mi.id)
        with self._model_instance_start_lock:
            delete_logs = self._managed_stopping.get(key, delete_logs)
        if delete_logs:
            self._purge_instance_logs(mi.id)
        self._forget_stopped_model_instance(
            mi, delete_logs=delete_logs, retain_instance=True
        )
        with self._model_instance_start_lock:
            # Log and port cleanup may yield while a permanent stop is appended.
            if self._managed_stopping.get(key, False) and not delete_logs:
                return False
            self._model_instance_by_instance_id.pop(mi.id, None)
            self._managed_stopping.pop(key, None)
            for component_key in list(self._smg_stopping):
                if component_key[:2] == key:
                    self._smg_stopping.pop(component_key, None)
        return True

    def _legacy_preparers_absent(self, mi):
        """Find orphaned legacy preparers without trusting a reused numeric PID."""
        title = f"gpustack_model_instance_{mi.id}"
        recorded = (
            mi.pid
            if mi.worker_id == self._worker_id
            else next(
                (
                    worker.pid
                    for worker in mi.distributed_servers.subordinate_workers or []
                    if worker.worker_id == self._worker_id
                ),
                None,
            )
        )
        processes = list(psutil.process_iter(["pid", "name", "cmdline", "ppid"]))
        other_pids = self._other_legacy_preparer_pids(mi.id, processes)
        owned, seen = [], set()
        for process in processes:
            seen.add(process.pid)
            command = process.info.get("cmdline")
            if command and command[0] == title:
                owned.append(process)
            elif process.pid in other_pids or process.info.get("ppid") in other_pids:
                continue
            elif process.pid == recorded or process.info.get("name") in (
                title,
                title[:15],
            ):
                return False
            elif command and "--multiprocessing-fork" in command:
                return False
        if recorded and recorded not in seen and psutil.pid_exists(recorded):
            return False
        stopped = [
            self._stop_identified_legacy_preparer(process, title) for process in owned
        ]
        return all(stopped)

    def _other_legacy_preparer_pids(self, instance_id, processes):
        """Identify other live preparers without relying on truncated OS names."""
        other_pids = {
            process.pid
            for other_id, process in list(self._provisioning_processes.items())
            if other_id != instance_id and process.is_alive()
        }
        prefix = "gpustack_model_instance_"
        for process in processes:
            command = process.info.get("cmdline")
            if not command:
                continue
            title = command[0]
            suffix = title.removeprefix(prefix)
            if title in (
                "gpustack_model_file_manager",
                "gpustack_model_file_download",
            ) or (
                title.startswith(prefix)
                and suffix.isascii()
                and suffix.isdecimal()
                and suffix != str(instance_id)
            ):
                other_pids.add(process.pid)
        return other_pids

    @staticmethod
    def _stop_identified_legacy_preparer(process, title):
        """Signal Process objects only after rechecking their preparer identity."""
        try:
            if process.cmdline()[:1] != [title]:
                return False
            children = process.children(recursive=True)
            for child in children + [process]:
                terminate_process(child)
            return not any(child.is_running() for child in children + [process])
        except psutil.NoSuchProcess:
            return False

    def _legacy_stop_runtime_absent(self, mi, names):
        process = self._provisioning_processes.get(mi.id)
        if process is not None:
            process.join(timeout=0)
            if process.is_alive():
                return False
        if not self._legacy_preparers_absent(mi):
            return False
        observed = list_workloads()
        if observed is None or not names:
            return False
        names.update(
            item.name
            for item in observed
            if (getattr(item, "labels", None) or {}).get(
                "gpustack.ai/model-instance-id"
            )
            == str(mi.id)
        )
        for name in sorted(names):
            current = get_workload(name)
            owner = (getattr(current, "labels", None) or {}).get(
                "gpustack.ai/model-instance-id"
            )
            if owner is not None and owner != str(mi.id):
                return False
            if current is not None:
                delete_workload(name)
            if get_workload(name) is not None:
                return False
        observed = list_workloads()
        if observed is None or any(item.name in names for item in observed):
            return False
        return self._legacy_stop_other_workloads_known(mi, observed)

    def _legacy_stop_other_workloads_known(self, mi, observed):
        if any(
            (getattr(item, "labels", None) or {}).get("gpustack.ai/model-instance-id")
            == str(mi.id)
            for item in observed
        ):
            return False
        ambiguous = {
            item.name
            for item in observed
            if not (getattr(item, "labels", None) or {}).get(
                "gpustack.ai/model-instance-id"
            )
            and not (getattr(item, "labels", None) or {}).get(
                "cache-service-instance-id"
            )
        }
        if not ambiguous:
            return True
        response = self._clientset.model_instances.list(
            params={"page": -1}, use_cache=False
        )
        known = set()
        for other in response.items or []:
            if other.id == mi.id:
                continue
            metadata = other.get_deployment_metadata(self._worker_id)
            if metadata is not None:
                known.add(metadata.name)
            known.update(
                item.name
                for item in get_model_instance_workloads(other, self._worker_id) or []
            )
        return ambiguous <= known

    def _stop_legacy_guarded_model_instance(self, mi, delete_logs):
        """Prove old runtime absence before acknowledging the Server's stop token."""
        guard = model_instance_legacy_stop(mi)
        if guard is None or self._worker_id not in guard.worker_ids:
            return False
        key = (mi.id, mi.restart_count or 0)
        try:
            current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
        except NotFoundException:
            current = None
        if current is not None:
            latest = model_instance_legacy_stop(current)
            if (current.restart_count or 0) > guard.restart_count:
                return self._finish_legacy_stop(mi, delete_logs)
            if (
                latest is None
                or latest.token != guard.token
                or (current.restart_count or 0) != guard.restart_count
            ):
                return False
        with self._model_instance_start_lock:
            local = self._model_instance_by_instance_id.get(mi.id)
            if local is not None and (local.restart_count or 0) > guard.restart_count:
                return False
            self._managed_stopping[key] = delete_logs
            self._model_instance_by_instance_id[mi.id] = mi
        names = self._legacy_stop_names.setdefault((mi.id, str(guard.token)), set())
        if (
            current is not None
            and current.state == ModelInstanceStateEnum.ERROR
            and current.draining_since is None
            and self._worker_id in latest.confirmed_worker_ids
        ):
            # A persisted ACK also proves local cleanup after a Worker restart.
            names = self._legacy_stop_names[(mi.id, str(guard.token))] = None
        if names is not None:
            self._assigned_ports.setdefault(mi.id, set()).update(
                model_instance_ports(mi, self._worker_id)
            )
            for known in (mi, local):
                metadata = (
                    known.get_deployment_metadata(self._worker_id) if known else None
                )
                if metadata is not None:
                    names.add(metadata.name)
            persistence = self._log_persistence.get(mi.id)
            if getattr(persistence, 'workload_name', None):
                names.add(persistence.workload_name)
            if mi.id in self._starting_model_instances:
                return False
        return self._confirm_legacy_stop(mi, current, names, delete_logs)

    def _confirm_legacy_stop(self, mi, current, names, delete_logs):
        """Acknowledge local absence once and wait for the frozen member set."""
        guard = model_instance_legacy_stop(mi)
        latest = model_instance_legacy_stop(current) if current is not None else None
        try:
            confirmed = (
                mi.state == ModelInstanceStateEnum.ERROR
                and mi.draining_since is None
                and current is not None
                and self._worker_id in latest.confirmed_worker_ids
            )
            if not confirmed:
                if names is not None and not self._legacy_stop_runtime_absent(
                    mi, names
                ):
                    return False
                if current is None:
                    return self._finish_legacy_stop(mi, delete_logs)
                worker = self._clientset.workers.get(self._worker_id, use_cache=False)
                if worker.state not in (
                    WorkerStateEnum.READY,
                    WorkerStateEnum.MAINTENANCE,
                ) or getattr(worker, 'unreachable', False):
                    return False
                if (
                    names is not None
                    and mi.draining_since is None
                    and mi.state == ModelInstanceStateEnum.ERROR
                ):
                    # The Server may advance as soon as this ACK commits. Retire
                    # local logs and port claims before a new round can reserve them.
                    self._stop_container_log_persistence(mi.id)
                    self._forget_stopped_model_instance(mi, retain_instance=True)
                    self._legacy_stop_names[(mi.id, str(guard.token))] = None
                self._clientset.model_instances.confirm_stop(
                    id=mi.id,
                    report=ModelInstanceStopReport(
                        token=guard.token, restart_count=guard.restart_count
                    ),
                )
            current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
            latest = model_instance_legacy_stop(current)
            if (current.restart_count or 0) > guard.restart_count:
                return self._finish_legacy_stop(mi, delete_logs)
            if (
                latest is None
                or latest.token != guard.token
                or (current.restart_count or 0) != guard.restart_count
                or not set(latest.worker_ids) <= set(latest.confirmed_worker_ids)
            ):
                return False
        except NotFoundException:
            pass
        except Exception as exc:
            logger.warning('Cannot confirm legacy stop for %s: %s', mi.id, exc)
            return False
        return self._finish_legacy_stop(mi, delete_logs)

    def _finish_legacy_stop(self, mi, delete_logs):
        """Retire an acknowledged round without touching a newer local owner."""
        key = (mi.id, mi.restart_count or 0)
        guard = model_instance_legacy_stop(mi)
        record = (mi.id, str(guard.token))
        cleaned = (
            record in self._legacy_stop_names
            and self._legacy_stop_names[record] is None
        )
        local = self._model_instance_by_instance_id.get(mi.id)
        if local is None or (local.restart_count or 0) <= key[1]:
            if not cleaned:
                self._stop_container_log_persistence(mi.id)
                self._forget_stopped_model_instance(
                    mi, delete_logs=delete_logs, retain_instance=True
                )
            elif delete_logs:
                self._purge_instance_logs(mi.id)
        with self._model_instance_start_lock:
            local = self._model_instance_by_instance_id.get(mi.id)
            if local is not None and (local.restart_count or 0) <= key[1]:
                self._model_instance_by_instance_id.pop(mi.id, None)
            self._managed_stopping.pop(key, None)
        self._legacy_stop_names.pop(record, None)
        return True

    async def watch_models(self):
        """
        Loop to watch models to keep the cache updated.

        """

        logger.debug("Watching models.")

        while True:
            try:
                # Watch models without callback to keep the cache updated.
                await self._clientset.models.awatch(callback=None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching models: {e}")
                await asyncio.sleep(5)

    async def watch_model_instances_event(self):
        """
        Loop to watch model instances' event and handle.

        """

        logger.debug("Watching model instances event.")

        while True:
            try:
                await self._clientset.model_instances.awatch(
                    callback=self._handle_model_instance_event_async
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching model instances: {e}")
                await asyncio.sleep(5)

    async def watch_model_instances(self):
        """
        Loop to post process model instances, for example, restarting error instances.

        """

        logger.debug("Watching model instances.")

        while True:
            try:
                await asyncio.to_thread(self._retry_distributed_port_releases)
                for mi in list(self._error_model_instances.values()):
                    try:
                        await asyncio.to_thread(self._restart_error_model_instance, mi)
                    except Exception as e:
                        logger.warning(
                            "Failed to restart model instance %s (%s)",
                            mi.name,
                            type(e).__name__,
                        )
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error restarting model instances: {e}")
                await asyncio.sleep(5)

    def sync_model_instances_state(self):  # noqa: C901
        """
        Synchronize model instances' state.

        - If the model instance is scheduled but not initialized, skip.
        - If the provision process is still alive, skip.
        - If the workload is still launching, skip.
        - If the workload is not existed, unhealthy, inactive or failed, update the model instance state to ERROR.
        - If everything is fine, update the model instance state to RUNNING.
        """

        self._retry_managed_stops()

        # Snapshot local state BEFORE the list call. Reversing this order
        # races with CREATED events and reaps freshly-assigned instances.
        with self._model_instance_start_lock:
            local_instances = list(self._model_instance_by_instance_id.items())
        local_assigned_ids = {
            mid for mid, mi in local_instances if self._is_local_model_instance(mi)
        }

        # Read from the watch-backed cache. It holds the full set, so the
        # common (nothing-to-reap) path stays O(1) with no server/DB round
        # trip and scales independently of worker count — a direct per-worker
        # poll here is one SELECT over model_instances per worker every few
        # seconds, which does not scale to hundreds of workers.
        response = self._clientset.model_instances.list()
        all_items = response.items or []
        reap_ids = local_assigned_ids - {
            mi.id for mi in all_items if self._is_local_model_instance(mi)
        }

        # Optional periodic reconciliation against DB truth even when the cache
        # reports nothing reapable. The cache and local serving state are both
        # fed by the same awatch stream, so a ghost re-seeded into the cache
        # sits in both and `local - cache` never flags it; an independent DB
        # read is the only thing that catches that. Disabled by default (the
        # server-side cached_all fix plus reconnect-driven reaping cover the
        # realistic cases); opt in via the
        # GPUSTACK_MODEL_INSTANCE_STATE_RECONCILE_INTERVAL env var when a
        # coordinator may drop DELETEDs on a live stream.
        reconcile_interval = envs.MODEL_INSTANCE_STATE_RECONCILE_INTERVAL
        now = time.time()
        force_authoritative = (
            reconcile_interval > 0
            and now - self._last_state_reconcile_time >= reconcile_interval
        )

        # Reaping tears down live workloads, so it must act on an authoritative
        # snapshot. The cache is not one at every instant: awatch clears it and
        # flips _watch_started before the replay snapshot arrives, so a read
        # during a reconnect window can see an empty-but-"authoritative" cache
        # and surface healthy instances as stale. So confirm reap candidates
        # against a fresh, unpaginated, uncached fetch — this only touches the
        # DB when something looks reapable or on the periodic reconciliation.
        if reap_ids or force_authoritative:
            response = self._clientset.model_instances.list(
                params={"page": -1},
                use_cache=False,
            )
            # Any successful authoritative fetch counts as a reconciliation —
            # whether triggered by the timer or by a reap candidate — so reset
            # the timer here (list() raises on API failure, so a failed fetch
            # doesn't advance it and the next tick retries). Skipped when the
            # backstop is disabled to keep the timestamp meaningless-but-inert.
            if reconcile_interval > 0:
                self._last_state_reconcile_time = now
            all_items = response.items or []
            reap_ids = local_assigned_ids - {
                mi.id for mi in all_items if self._is_local_model_instance(mi)
            }

        # An empty authoritative result is legitimate (user stopped every
        # model); list() raises on API failure rather than returning empty, so
        # local_assigned_ids - ∅ correctly reaps everything still tracked.
        for stale_id in reap_ids:
            stale = self._model_instance_by_instance_id.get(stale_id)
            if stale is None or stale_id in self._starting_model_instances:
                # A start can still create a process or container. Keep its
                # reservations until the next pass can stop the finished start.
                continue
            logger.info(
                f"Reaping stale model instance {stale.name} (id={stale_id}); "
                f"server no longer reports it assigned to this worker."
            )
            try:
                # Equivalent to a dropped DELETED event, so purge logs too —
                # otherwise a reused id would inherit this instance's stale logs.
                if get_model_instance_workloads(stale) is not None:
                    self._schedule_managed_stop(stale, delete_logs=True)
                else:
                    self._stop_model_instance(stale, delete_logs=True)
            except Exception as e:
                logger.warning(f"Failed to reap stale model instance {stale.name}: {e}")

        if not all_items:
            # Nothing left to sync; reap pass above already handled stale
            # local state.
            return

        model_instances: List[ModelInstance] = []
        for model_instance in all_items:
            if has_model_instance_stop_barrier(model_instance):
                if self._is_local_model_instance(model_instance):
                    model_instances.append(model_instance)
                continue
            # if the model instance is assigned to this worker, it must be scheduled.
            # But we don't need to sync the scheduled model when it is not initialized yet.
            if (
                model_instance.worker_id == self._worker_id
                and model_instance.state != ModelInstanceStateEnum.SCHEDULED
            ):
                model_instances.append(model_instance)
            if (
                model_instance.distributed_servers
                and model_instance.distributed_servers.subordinate_workers
            ):
                for sw in model_instance.distributed_servers.subordinate_workers:
                    if sw.worker_id == self._worker_id:
                        model_instances.append(model_instance)
                        break

        # Retire log persistence the server no longer assigns here: a DELETED
        # event landing mid-pass leaves a generation nothing will ever stop, and
        # the persistence loop retries a missing workload forever.
        assigned_instance_ids = {mi.id for mi in model_instances}
        for stale_id in set(self._log_persistence) - assigned_instance_ids:
            logger.debug(
                f"Stopping log persistence for model instance {stale_id}, "
                f"no longer assigned to this worker"
            )
            self._stop_container_log_persistence(stale_id)

        # Reuse a fresh worker snapshot only within this sync pass.
        recovery_worker = None
        for model_instance in sorted(
            model_instances,
            key=lambda mi: self._rank_health_last_dispatch.get(mi.id, 0),
        ):
            # Isolate each instance so stale state or a failed report cannot block peers.
            try:
                if model_instance_legacy_stop(model_instance) is not None:
                    self._stop_model_instance(model_instance)
                    continue
                if get_model_instance_workloads(model_instance) is not None:
                    # A stop intent can outlive its local snapshot. Reconcile from
                    # Server authority so that cleanup can recover that ownership.
                    self._sync_managed_model_instance(model_instance)
                    continue
                # Skip if the provision process has not exited yet.
                if self._is_provisioning(model_instance):
                    logger.trace(
                        f"Model instance {model_instance.name} is provisioning. Skipping sync."
                    )
                    continue

                is_main_worker = model_instance.worker_id == self._worker_id

                # INITIALIZE_LATER followers have no workload until the main worker
                # finishes downloading. A missing container is expected while waiting.
                if (
                    not is_main_worker
                    and model_instance.distributed_servers.mode
                    == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                    and model_instance.state
                    not in (
                        ModelInstanceStateEnum.STARTING,
                        ModelInstanceStateEnum.RUNNING,
                        ModelInstanceStateEnum.ERROR,
                        ModelInstanceStateEnum.UNREACHABLE,
                    )
                ):
                    continue

                # Skip if the workload is still launching.
                # Use deployment metadata name for subordinate workers (e.g., "model-f0")
                # since their workload name differs from the model instance name.
                if is_main_worker:
                    workload = get_workload(model_instance.name)
                else:
                    deployment_metadata = model_instance.get_deployment_metadata(
                        self._worker_id
                    )
                    workload_name = (
                        deployment_metadata.name
                        if deployment_metadata
                        else model_instance.name
                    )
                    workload = get_workload(workload_name)

                if workload and workload.state in [
                    WorkloadStatusStateEnum.PENDING,
                    WorkloadStatusStateEnum.INITIALIZING,
                ]:
                    if is_main_worker:
                        subordinate_state = self._get_main_worker_distributed_state(
                            model_instance
                        )
                        if (
                            subordinate_state
                            and subordinate_state.get("state")
                            == ModelInstanceStateEnum.ERROR
                            and not self._auto_restart_enabled(model_instance)
                        ):
                            # The main engine may wait for a follower indefinitely.
                            # Surface a terminal follower failure instead of waiting
                            # for an HTTP listener that cannot finish initializing.
                            if subordinate_state["should_update"]:
                                self._report_model_instance(
                                    model_instance,
                                    state=subordinate_state['state'],
                                    state_message=subordinate_state['state_message'],
                                )
                            continue
                    # "Still launching" is also what a crash loop looks like. A
                    # container that binds, fails and is restarted never reaches
                    # FAILED, so the branch below never runs and the instance sits
                    # at `starting` for as long as anyone leaves it — which is the
                    # shape every port-level failure in a disaggregated deployment
                    # takes. Ask whether it is launching or looping before
                    # accepting the former.
                    if self._mark_crash_loop(model_instance, workload, is_main_worker):
                        continue
                    logger.trace(
                        f"Model instance {model_instance.name} workload is still launching. Skipping sync."
                    )
                    continue

                # Update model instance state to ERROR if the workload is not existed, unhealthy, inactive or failed.
                if not workload or workload.state in [
                    WorkloadStatusStateEnum.UNKNOWN,  # Rare, but possible, for example, leaving pause container.
                    WorkloadStatusStateEnum.UNHEALTHY,
                    WorkloadStatusStateEnum.INACTIVE,
                    WorkloadStatusStateEnum.FAILED,
                ]:
                    # Only if not in ERROR state yet.
                    if model_instance.state != ModelInstanceStateEnum.ERROR:
                        # Surface the workload's own diagnosis (e.g. a device-plugin
                        # admission rejection, an image-pull failure, an exit code)
                        # when available.
                        failure_message = _describe_workload_failure(workload)
                        # And then the log's, because the workload's is often just
                        # a number. A container that exits once and permanently —
                        # `exit code 127`, a command the image does not contain —
                        # never reaches the crash-loop path, so without this
                        # call the failure that can never recover would be the one
                        # explained least.
                        failure_message = self._append_log_diagnosis(
                            model_instance, failure_message
                        )
                        with contextlib.suppress(NotFoundException):
                            # Get patch dict for main worker.
                            if is_main_worker:
                                patch_dict = {
                                    "state": ModelInstanceStateEnum.ERROR,
                                    "state_message": failure_message,
                                }
                            # Get patch dict for subordinate worker.
                            else:
                                sw_pos = next(
                                    (
                                        i
                                        for i, sw in enumerate(
                                            model_instance.distributed_servers.subordinate_workers
                                        )
                                        if sw.worker_id == self._worker_id
                                    ),
                                )
                                sw = model_instance.distributed_servers.subordinate_workers[
                                    sw_pos
                                ]
                                sw.state = ModelInstanceStateEnum.ERROR
                                # Keep workload failures identifiable after the runtime
                                # clears its diagnosis, without losing the details.
                                if not failure_message.startswith(
                                    _WORKLOAD_FAILED_MESSAGE
                                ):
                                    failure_message = (
                                        f"{_WORKLOAD_FAILED_MESSAGE} {failure_message}"
                                    )
                                sw.state_message = failure_message
                                patch_dict = {
                                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                                }
                            # Update model instance.
                            self._report_model_instance(model_instance, **patch_dict)
                    continue

                # The workload is alive and this worker did not necessarily start it,
                # so reconcile the log persistence threads that belong to it.
                self._ensure_container_log_persistence(model_instance)

                # Otherwise, update model instance state to RUNNING if everything is fine.
                try:
                    model = self._get_model(model_instance)
                    if model.gpu_type_selector:
                        # vGPU: read back the real device allocation the operator
                        # device plugin wrote onto the workload's annotations.
                        self._sync_vgpu_allocation(
                            model_instance, workload, is_main_worker
                        )
                    if not model.backend_version:
                        # backend version may be empty on initialization.
                        # try to refresh to get updated model info on syncs.
                        model = self._refresh_model(model_instance)
                except NotFoundException:
                    # An instance can briefly outlive its deleted model in the watch cache.
                    continue

                backend = get_backend(model)
                health_check_path = self._get_health_check_path(
                    backend, model.owner_principal_id
                )
                if not health_check_path:
                    # A managed router runs as a custom backend, which registers no
                    # health path, and the generic probe reads that as "always
                    # ready". The recipe declares the router's own path, and the
                    # router is the group's only entrance -- the one member whose
                    # liveness must not be assumed.
                    health_check_path = managed_router_health_path(
                        model, model_instance.role
                    )
                if model.env and 'GPUSTACK_MODEL_HEALTH_CHECK_PATH' in model.env:
                    # A model-specific path can match a custom image whose health
                    # endpoint differs from the backend's default.
                    health_check_path = model.env['GPUSTACK_MODEL_HEALTH_CHECK_PATH']

                with contextlib.suppress(NotFoundException):
                    # Get patch dict for main worker.
                    if is_main_worker:
                        subordinate_state = self._get_main_worker_distributed_state(
                            model_instance
                        )
                        if subordinate_state is None:
                            if model_instance.state == ModelInstanceStateEnum.RUNNING:
                                self._restart_backoff_counts.pop(
                                    model_instance.id, None
                                )
                                self._error_model_instances.pop(model_instance.id, None)
                                self._pending_model_instance_errors.pop(
                                    model_instance.id, None
                                )
                                continue

                            # A recovered follower can clear its propagated failure;
                            # independent leader failures still require recovery.
                            if (
                                model_instance.state == ModelInstanceStateEnum.ERROR
                                and not (
                                    model_instance.distributed_servers
                                    and model_instance.distributed_servers.mode
                                    == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                                    and (model_instance.state_message or "").startswith(
                                        _SUBORDINATE_ERROR_PREFIX
                                    )
                                )
                            ):
                                continue
                            if not is_ready(
                                backend, model_instance, health_check_path, model
                            ):
                                continue

                            self._restart_backoff_counts.pop(model_instance.id, None)
                            # A member that served is not a member that never
                            # started: a later crash loop is a different failure
                            # and must not be reported as a bad configuration.
                            self._restart_tracker.observe_running(model_instance.id)
                            patch_dict = {
                                "state": ModelInstanceStateEnum.RUNNING,
                                "state_message": "",
                            }

                            # Fetch model meta once running.
                            meta = get_meta_from_running_instance(
                                model_instance, backend, model
                            )
                            if meta:
                                # Some meta is set in server evaluation and should be preserved, so we update meta instead of overwrite.
                                merged_meta = dict(model.meta or {})
                                merged_meta.update(meta)
                                if merged_meta != model.meta:
                                    self._update_model(model.id, meta=merged_meta)
                        elif subordinate_state["should_update"]:
                            patch_dict = {
                                "state": subordinate_state["state"],
                                "state_message": subordinate_state["state_message"],
                            }
                        else:
                            continue
                    # Get patch dict for subordinate worker.
                    else:
                        sw_pos = next(
                            (
                                i
                                for i, sw in enumerate(
                                    model_instance.distributed_servers.subordinate_workers
                                )
                                if sw.worker_id == self._worker_id
                            ),
                        )
                        sw = model_instance.distributed_servers.subordinate_workers[
                            sw_pos
                        ]
                        if sw.state == ModelInstanceStateEnum.RUNNING:
                            self._error_model_instances.pop(model_instance.id, None)
                            self._pending_model_instance_errors.pop(
                                model_instance.id, None
                            )
                            continue
                        if (
                            model_instance.distributed_servers.mode
                            == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                        ):
                            # Reconcile surviving workloads without starting work.
                            # The leader owns engine readiness; explicit backend and
                            # inference failures remain terminal.
                            if workload.state != WorkloadStatusStateEnum.RUNNING or (
                                sw.state == ModelInstanceStateEnum.ERROR
                                and not (sw.state_message or "").startswith(
                                    _WORKLOAD_FAILED_MESSAGE
                                )
                            ):
                                continue
                            if recovery_worker is None:
                                recovery_worker = self._clientset.workers.get(
                                    self._worker_id, use_cache=False
                                )
                            if (
                                recovery_worker.state != WorkerStateEnum.READY
                                or recovery_worker.unreachable
                            ):
                                continue
                        # Do not mutate the watch cache before the update succeeds:
                        # a failed write must leave recovery eligible for retry.
                        sw = sw.model_copy(
                            update={
                                "state": ModelInstanceStateEnum.RUNNING,
                                "state_message": "",
                            }
                        )
                        patch_dict = {
                            f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                        }
                    # Update model instance.
                    self._report_model_instance(model_instance, **patch_dict)
                    if (
                        patch_dict.get("state") == ModelInstanceStateEnum.RUNNING
                        or not is_main_worker
                    ):
                        self._error_model_instances.pop(model_instance.id, None)
                        self._pending_model_instance_errors.pop(model_instance.id, None)
            except Exception as e:
                logger.warning(
                    "Failed to sync model instance %s (%s)",
                    model_instance.name,
                    type(e).__name__,
                )

    @staticmethod
    def _get_main_worker_distributed_state(
        model_instance: ModelInstance,
    ) -> Optional[dict]:
        subordinate_workers = (
            model_instance.distributed_servers.subordinate_workers
            if (
                model_instance.distributed_servers
                and model_instance.distributed_servers.subordinate_workers
            )
            else []
        )

        if not subordinate_workers:
            return None

        error_sw = None
        unreachable_sw = None
        all_running = True
        for sw in subordinate_workers:
            if sw.state == ModelInstanceStateEnum.ERROR:
                error_sw = sw
                break
            if (
                sw.state == ModelInstanceStateEnum.UNREACHABLE
                and unreachable_sw is None
            ):
                unreachable_sw = sw
            if sw.state != ModelInstanceStateEnum.RUNNING:
                all_running = False

        if error_sw:
            return {
                "should_update": model_instance.state != ModelInstanceStateEnum.ERROR,
                "state": ModelInstanceStateEnum.ERROR,
                "state_message": (
                    f"{_SUBORDINATE_ERROR_PREFIX}"
                    f"{error_sw.worker_ip}: {error_sw.state_message}."
                ),
            }

        if unreachable_sw:
            return {
                "should_update": model_instance.state
                != ModelInstanceStateEnum.UNREACHABLE,
                "state": ModelInstanceStateEnum.UNREACHABLE,
                "state_message": (
                    f"Distributed serving unreachable in subordinate worker "
                    f"{unreachable_sw.worker_ip}: {unreachable_sw.state_message}."
                ),
            }

        if not all_running:
            return {"should_update": False}

        return None

    def _sync_vgpu_allocation(  # noqa: C901
        self,
        model_instance: ModelInstance,
        workload,
        is_main_worker: bool,
    ):
        """
        Patch the instance with the real device allocation read back from the
        workload annotations: GPU UUID(s) into gpu_addresses, the allocated
        card's index into gpu_indexes (resolved via this worker's reported
        gpu_devices), and the claim's vram re-keyed from the placeholder to
        that index so worker_allocated_cache charges the partial card to the
        right index. No-op until the device plugin has allocated; deferred
        (not patched) while the allocated card is missing from the worker's
        reported devices.
        """
        accelerators = _parse_allocated_accelerators(
            getattr(workload, "annotations", None)
        )
        if not accelerators:
            return

        uuids = [a["id"] for a in accelerators]

        # Resolve the patch target first (local lookups only).
        sw_pos = None
        if not is_main_worker:
            if (
                not model_instance.distributed_servers
                or not model_instance.distributed_servers.subordinate_workers
            ):
                return
            sw_pos = next(
                (
                    i
                    for i, sw in enumerate(
                        model_instance.distributed_servers.subordinate_workers
                    )
                    if sw.worker_id == self._worker_id
                ),
                None,
            )
            if sw_pos is None:
                return
        target = (
            model_instance
            if is_main_worker
            else model_instance.distributed_servers.subordinate_workers[sw_pos]
        )

        # Steady-state no-op guard: once addresses, indexes and the re-keyed
        # claim all agree with the annotation, skip the workers API call that
        # index resolution needs — it would otherwise fire every sync cycle
        # for every vGPU instance.
        target_claim = target.computed_resource_claim
        if (
            target.gpu_addresses == uuids
            and target.gpu_indexes
            and target_claim is not None
            and target_claim.vram
            and set(target_claim.vram.keys()) == set(target.gpu_indexes)
        ):
            return

        # Resolve the allocated card's index from this worker's reported
        # devices (the detected set, not the static config, which is empty
        # for auto-detected workers).
        gpu_devices = []
        with contextlib.suppress(Exception):
            worker = self._clientset.workers.get(self._worker_id)
            gpu_devices = (worker.status and worker.status.gpu_devices) or []
        index_by_uuid = {d.uuid: d.index for d in gpu_devices if d.uuid}
        allocated_index = next(
            (index_by_uuid[u] for u in uuids if u in index_by_uuid), None
        )
        if allocated_index is None:
            # The card isn't in the worker's reported devices (yet): patching
            # now would leave the claim charged to a wrong placeholder index.
            # Retry on the next sync cycle.
            logger.debug(
                f"vgpu allocation UUIDs {uuids} not found in worker "
                f"{self._worker_id} reported devices, deferring sync"
            )
            return

        def rekey_claim(claim):
            if claim is None or not claim.vram:
                return None
            if set(claim.vram.keys()) == {allocated_index}:
                return None
            return claim.model_copy(
                update={"vram": {allocated_index: sum(claim.vram.values())}}
            )

        with contextlib.suppress(NotFoundException):
            if is_main_worker:
                patch_dict = {}
                if model_instance.gpu_addresses != uuids:
                    patch_dict["gpu_addresses"] = uuids
                if model_instance.gpu_indexes != [allocated_index]:
                    patch_dict["gpu_indexes"] = [allocated_index]
                new_claim = rekey_claim(model_instance.computed_resource_claim)
                if new_claim is not None:
                    patch_dict["computed_resource_claim"] = new_claim
                if patch_dict:
                    self._report_model_instance(model_instance, **patch_dict)
                return

            sw = target
            changed = False
            if sw.gpu_addresses != uuids:
                sw.gpu_addresses = uuids
                changed = True
            if sw.gpu_indexes != [allocated_index]:
                sw.gpu_indexes = [allocated_index]
                changed = True
            new_claim = rekey_claim(sw.computed_resource_claim)
            if new_claim is not None:
                sw.computed_resource_claim = new_claim
                changed = True
            if changed:
                if self._uses_group_recovery(model_instance):
                    self._report_model_instance(
                        model_instance,
                        gpu_addresses=sw.gpu_addresses,
                        gpu_indexes=sw.gpu_indexes,
                        computed_resource_claim=sw.computed_resource_claim,
                    )
                    return
                self._report_model_instance(
                    model_instance,
                    **{f'distributed_servers.subordinate_workers.{sw_pos}': sw},
                )

    @staticmethod
    def _serve_model_instance(
        mi: ModelInstance,
        backend: BackendEnum,
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
        worker_id: int,
        inference_backend: InferenceBackend,
        fallback_registry: Optional[str] = None,
        launch_connection: Optional[Connection] = None,
    ):
        """
        Serve model instance in a subprocess.
        Exits the subprocess when serving ends.

        Args:
            mi: The model instance to serve.
            backend: The backend of the model instance.
            client_headers: The headers for the clientset.
            log_file_path: The path to the log file.
            cfg: The configuration.
            worker_id: The ID of the worker.
            inference_backend: The inference backend configuration.
            fallback_registry: The fallback container registry to use if needed.
        """

        setproctitle.setproctitle(f"gpustack_model_instance_{mi.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.get_server_url(),
            headers=client_headers,
        )

        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        with CappedLogWriter(log_file_path) as log_file:
            with RedirectStdoutStderr(log_file):
                try:
                    server_cls = _SERVER_CLASS_MAPPING.get(backend, CustomServer)
                    server_ins = server_cls(
                        clientset,
                        mi,
                        cfg,
                        worker_id,
                        inference_backend,
                        fallback_registry,
                    )
                    if launch_connection is not None:
                        server_ins._workload_plan_callback = (
                            lambda workloads, plans: launch_connection.send(
                                (workloads, plans)
                            )
                        )
                    logger.info(f"Provisioning model instance {mi.name}")
                    server_ins.start()
                    logger.info(
                        "Finished %s model instance %s",
                        (
                            "preparing"
                            if launch_connection is not None
                            else "provisioning"
                        ),
                        mi.name,
                    )
                except Exception as e:
                    logger.exception(
                        f"Error provisioning model instance {mi.name}: {e}"
                    )
                    raise e
                finally:
                    if launch_connection is not None:
                        launch_connection.close()

    def _clear_rank_health(
        self, instance_id: int, restart_count: Optional[int] = None
    ) -> None:
        with self._rank_health_lock:
            for key in list(self._rank_health_checks):
                if key[0] == instance_id and (
                    restart_count is None or key[1] == restart_count
                ):
                    check = self._rank_health_checks.pop(key)
                    if check.future is not None:
                        check.future.cancel()

    def _rank_health_targets(self, mi: ModelInstance, model: Model) -> Dict:
        """Only local main APIs participate; headless liveness comes from runtime."""
        startup = mi.state != ModelInstanceStateEnum.RUNNING
        config = _get_inference_health_check_config(model, mi.role)
        targets = {}
        for workload in get_model_instance_workloads(mi, self._worker_id) or []:
            running = workload.state == ModelInstanceWorkloadStateEnum.RUNNING
            for placement in workload.dp_rank_placements:
                if placement.node_rank != 0:
                    continue
                key = (
                    mi.id,
                    mi.restart_count or 0,
                    workload.name,
                    placement.dp_rank,
                    placement.node_rank,
                )
                targets[(*key, 'get')] = (
                    placement,
                    envs.MODEL_INSTANCE_HEALTH_CHECK_INTERVAL if startup else 10,
                    1 if startup else 15,
                    3,
                    running,
                )
                if (
                    not startup
                    and config['enabled']
                    and mi.role not in ("prefill", "decode")
                ):
                    targets[(*key, 'inference')] = (
                        placement,
                        config['interval'],
                        config['timeout'],
                        config['threshold'],
                        running,
                    )
        return targets

    def _component_health_targets(self, mi, model=None):
        targets = {}
        for workload in get_model_instance_workloads(mi, self._worker_id) or []:
            component = workload.component
            if component.api_port is None or component.stop_requested:
                continue
            key = (
                mi.id,
                mi.restart_count or 0,
                workload.name,
                component.kind,
                component.attempt,
                'component',
            )
            check = self._rank_health_checks.get(key)
            startup = check.startup if check else not bool(component.ready)
            targets[key] = (
                component,
                envs.MODEL_INSTANCE_HEALTH_CHECK_INTERVAL if startup else 10,
                5 if startup else 15,
                3,
                workload.state == ModelInstanceWorkloadStateEnum.RUNNING,
            )
            if (
                component.kind == 'vllm_hybrid'
                and not startup
                and model is not None
                and mi.role is None
            ):
                config = _get_inference_health_check_config(model, mi.role)
                if config['enabled']:
                    targets[(*key[:5], 'inference')] = (
                        component,
                        config['interval'],
                        config['timeout'],
                        config['threshold'],
                        workload.state == ModelInstanceWorkloadStateEnum.RUNNING,
                    )
        return targets

    def _collect_component_health(self, report, targets, completed):
        diagnostic, processed = {}, []
        by_name = {w.name: w for w in report}
        for key in completed:
            check = self._rank_health_checks.get(key)
            if check is None or key not in targets:
                continue
            _, _, _, threshold, running = targets[key]
            target = by_name[key[2]]
            if not running or target.state != ModelInstanceWorkloadStateEnum.RUNNING:
                processed.append((check, check.failures))
                continue
            healthy, reason = check.future.result()
            failures = 0 if healthy or check.startup else check.failures + 1
            target.component.ready = (
                healthy if check.startup or healthy else target.component.ready
            )
            if not healthy and failures >= threshold:
                target.component.ready = False
                target.state = ModelInstanceWorkloadStateEnum.FAILED
            if not healthy:
                message = (
                    f'{target.component.kind} health on Worker {self._worker_id}: {reason}; '
                    + (
                        'waiting for readiness'
                        if check.startup
                        else f'consecutive failures {failures}/{threshold}'
                    )
                )
                if not check.startup and failures >= threshold:
                    diagnostic['state_message'] = message
                    logger.warning(message)
                else:
                    logger.debug(message)
            processed.append((check, failures))
        return diagnostic, processed

    def _sync_component_health(self, current, model, completed):
        targets = self._component_health_targets(current, model)
        with self._rank_health_lock:
            for key, check in list(self._rank_health_checks.items()):
                if key[0] == current.id and key not in targets:
                    self._rank_health_checks.pop(key)
                    if check.future is not None:
                        check.future.cancel()
            report = [
                w.model_copy(deep=True)
                for w in get_model_instance_workloads(current, self._worker_id)
            ]
            diagnostic, processed = self._collect_component_health(
                report, targets, completed
            )
        if diagnostic or report != get_model_instance_workloads(
            current, self._worker_id
        ):
            current = self._report_managed_workloads(current, report, **diagnostic)
            self._hold_managed_instance(current)
        with self._rank_health_lock:
            for check, failures in processed:
                check.failures, check.last_check, check.future = (
                    failures,
                    time.monotonic(),
                    None,
                )
            # Startup is local to each component, not to the instance's SMG
            # recovery state. Keep the Engine's running failure threshold.
            for key in targets:
                check = self._rank_health_checks.get(key)
                target = next(w for w in report if w.name == key[2])
                if check is not None and target.component.ready:
                    check.startup = False
        worker = self._clientset.workers.get(id=self._worker_id)
        path = '/health'
        if self._has_hybrid_components(current):
            path = (
                self._get_health_check_path(
                    get_backend(current), model.owner_principal_id
                )
                or '/health'
            )
            if model.env and 'GPUSTACK_MODEL_HEALTH_CHECK_PATH' in model.env:
                path = model.env['GPUSTACK_MODEL_HEALTH_CHECK_PATH']
        self._schedule_rank_health(
            current,
            model,
            self._component_health_targets(current, model),
            worker.ip,
            path,
        )

    def _collect_rank_health(self, report, targets, completed):
        """Stage results without consuming them until the fenced report succeeds."""
        diagnostic, processed = {}, []
        placements = {
            (w.name, p.dp_rank, p.node_rank): p
            for w in report
            for p in w.dp_rank_placements
        }
        for key in completed:
            check = self._rank_health_checks.get(key)
            if check is None or key not in targets:
                continue
            placement, _, _, threshold, running = targets[key]
            if not running:
                processed.append((check, check.failures))
                continue
            healthy, reason = check.future.result()
            failures = 0 if healthy or check.startup else check.failures + 1
            target = placements[key[2:5]]
            first_fault = failures >= threshold and 'state' not in diagnostic
            if check.startup:
                target.ready = healthy
            elif healthy and key[5] == 'get':
                target.ready = True
            elif not healthy and failures >= threshold:
                target.ready = False
                diagnostic['state'] = ModelInstanceStateEnum.ERROR
            if not healthy:
                message = (
                    f"Rank health: dp_rank={key[3]} node_rank={key[4]} "
                    f"Worker={self._worker_id} port={placement.api_port} "
                    f"{key[5]}: {reason}; "
                    + (
                        'waiting for readiness'
                        if check.startup
                        else f'consecutive failures {failures}/{threshold}'
                    )
                )
                if not check.startup and failures >= threshold:
                    if 'state_message' not in diagnostic or first_fault:
                        diagnostic['state_message'] = message
                    logger.warning(message)
                else:
                    logger.debug(message)
            processed.append((check, failures))
        return diagnostic, processed

    @partial(_model_lifecycle, blocking=False)
    def _sync_managed_rank_health(
        self, mi: ModelInstance, *, startup_only: bool = False
    ) -> None:
        """Collect completed probes and submit bounded work; never wait for HTTP."""
        if (
            startup_only
            and mi.state == ModelInstanceStateEnum.RUNNING
            and not self._has_sglang_components(mi)
            and not self._has_hybrid_components(mi)
        ):
            return
        try:
            # Sample completion before refreshing: the authority check must be
            # later than the HTTP result, including after Worker adoption.
            with self._rank_health_lock:
                completed = [
                    key
                    for key, check in self._rank_health_checks.items()
                    if key[0] == mi.id
                    and check.future is not None
                    and check.future.done()
                ]
            current = ModelInstance.model_validate(
                self._clientset.model_instances.get(id=mi.id, use_cache=False)
            )
            if (
                current.restart_count != mi.restart_count
                or self._managed_stop_requested(current)
                or current.draining_since is not None
                or (current.id, current.restart_count or 0) in self._managed_stopping
                or not self._is_local_model_instance(current)
            ):
                self._clear_rank_health(mi.id, mi.restart_count or 0)
                return
            model = self._get_model(current)
            if self._has_sglang_components(current) or self._has_hybrid_components(
                current
            ):
                self._sync_component_health(current, model, completed)
                return
            if (
                current.role not in (None, "prefill", "decode")
                or get_backend(current) != BackendEnum.VLLM
                or get_model_instance_workloads(current) is None
            ):
                self._clear_rank_health(mi.id)
                return
            targets = self._rank_health_targets(current, model)
            startup = current.state != ModelInstanceStateEnum.RUNNING
            with self._rank_health_lock:
                for key, check in list(self._rank_health_checks.items()):
                    if key[0] == mi.id and (
                        key not in targets or check.startup != startup
                    ):
                        self._rank_health_checks.pop(key)
                        if check.future is not None:
                            check.future.cancel()
                report = [
                    w.model_copy(deep=True)
                    for w in get_model_instance_workloads(current, self._worker_id)
                ]
                diagnostic, processed = self._collect_rank_health(
                    report, targets, completed
                )
            if diagnostic or report != get_model_instance_workloads(
                current, self._worker_id
            ):
                current = self._report_managed_workloads(current, report, **diagnostic)
                self._hold_managed_instance(current)
            with self._rank_health_lock:
                for check, failures in processed:
                    check.failures = failures
                    check.last_check = time.monotonic()
                    check.future = None
            if self._managed_stop_requested(current):
                self._clear_rank_health(mi.id)
                self._schedule_managed_stop(current)
                return
            if targets:
                worker = self._clientset.workers.get(id=self._worker_id)
                path = (
                    (model.env or {}).get('GPUSTACK_MODEL_HEALTH_CHECK_PATH')
                    or self._get_health_check_path(
                        get_backend(current), model.owner_principal_id
                    )
                    or '/health'
                )
                self._schedule_rank_health(current, model, targets, worker.ip, path)
        except NotFoundException:
            self._clear_rank_health(mi.id)
        except Exception as exc:
            logger.warning('Cannot check managed rank health for %s: %s', mi.id, exc)

    def _schedule_rank_health(self, mi, model, targets, worker_ip, path):
        with self._rank_health_lock:
            # Invalidated running requests still occupy a slot until they end;
            # generation churn must not create an unbounded executor queue.
            self._rank_health_futures = {
                key: future
                for key, future in self._rank_health_futures.items()
                if not future.done()
            }
            for key in targets:
                self._rank_health_checks.setdefault(
                    key,
                    _RankHealthCheck(
                        startup=(
                            not bool(targets[key][0].ready)
                            if key[5] == 'component'
                            else mi.state != ModelInstanceStateEnum.RUNNING
                        )
                    ),
                )
            ordered = sorted(
                targets,
                key=lambda key: (self._rank_health_checks[key].last_check or 0)
                + targets[key][1],
            )
            for key in ordered:
                check = self._rank_health_checks[key]
                placement, interval, timeout, _, running = targets[key]
                if (
                    not running
                    or key in self._rank_health_futures
                    or check.future is not None
                    or (
                        check.last_check is not None
                        and time.monotonic() - check.last_check < interval
                    )
                ):
                    continue
                if len(self._rank_health_futures) >= _RANK_HEALTH_CONCURRENCY:
                    break
                if self._rank_health_executor is None:
                    self._rank_health_executor = ThreadPoolExecutor(
                        max_workers=_RANK_HEALTH_CONCURRENCY,
                        thread_name_prefix='rank-health',
                    )
                target = mi.model_copy(deep=True)
                target.worker_ip, target.port = worker_ip, placement.api_port
                if key[5] == 'component' and key[3] != 'vllm_hybrid':
                    main = next(
                        (
                            w.component
                            for w in get_model_instance_workloads(mi)
                            if w.component is not None
                            and w.component.kind == 'sglang_engine'
                            and w.component.api_port is not None
                        ),
                        None,
                    )
                    if main is None:
                        logger.debug(
                            "Waiting for the Engine API manifest before probing SMG"
                        )
                        continue
                    check.future = self._rank_health_executor.submit(
                        _probe_sglang_component_health,
                        target,
                        key[3],
                        main.dp_size,
                        f'http://{mi.worker_ip}:{main.api_port}',
                        timeout,
                    )
                else:
                    check.future = self._rank_health_executor.submit(
                        _probe_rank_health,
                        target,
                        model,
                        path,
                        'get' if key[5] == 'component' else key[5],
                        timeout,
                    )
                self._rank_health_futures[key] = check.future
                self._rank_health_last_dispatch[mi.id] = time.monotonic()

    def _probe_running_instance(
        self, mi: ModelInstance, model: Model, timeout: int
    ) -> bool:
        """Is this member still serving? Asked of a member already RUNNING.

        The readiness probe answers this question once, on the way into
        RUNNING, and is never asked again -- past that point an engine that is
        listening but no longer answering is invisible to everything except
        the container's own state, which stays healthy through a hang. So this
        is the only thing that finds one, and every member needs an answer.

        What differs per member is *which* question is worth its cost:

        * An engine role -- a prefill, a decode, a role-less replica sharing
          the group's engine -- is asked the same thing the readiness probe
          asks, on the backend's own health path. One GET. It catches the
          failure that matters here (the process lives, the server does not)
          without spending a real request, and it cannot be answered wrongly
          by a member that is healthy in isolation.

        * The router, and a plain deployment, get the real inference request.
          For the router that is the whole point: it is where a request meets
          both roles, so its answer covers the KV path between them, which is
          the failure disaggregation adds and the one no single member's
          health endpoint can see.

        Deliberately NOT a real request to a prefill or a decode. Measured on
        a live 1P1D: each answers a chat completion on its own in ~0.1s,
        because each is a complete engine that merely also carries KV
        transfer. So the request costs a real generation and the KV cache
        behind it, and buys a verdict on the member alone -- which the GET
        already gives.
        """
        role = getattr(mi, "role", None)
        if not role or role == RoleNameEnum.ROUTER.value:
            return is_inference_ready(mi, model, timeout=timeout)

        backend = get_backend(model)
        health_check_path = self._get_health_check_path(
            backend, model.owner_principal_id
        )
        if model.env and 'GPUSTACK_MODEL_HEALTH_CHECK_PATH' in model.env:
            health_check_path = model.env['GPUSTACK_MODEL_HEALTH_CHECK_PATH']
        # The probe's own timeout, not the readiness default: a failure
        # here retires the member, so it has to outlast a busy engine.
        return is_ready(backend, mi, health_check_path, model, timeout=timeout)

    def sync_model_instances_inference_health(self):
        """
        Synchronize model instances' inference health by sending actual inference requests.

        Per-model configuration, read from model.env and nowhere else -- there
        is no server- or worker-wide setting behind any of these, so a
        deployment that sets none of them is not probed at all:
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED: "true"/"false" (default: false)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL: seconds (default: 300)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT: seconds (default: 15,
          or 25 for SGLang PD engine roles)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD: count (default: 3)

        If the model has received successful inference traffic recently
        (within the configured interval), the active health check is skipped.
        """

        with self._model_instance_start_lock:
            local_instances = list(self._model_instance_by_instance_id.values())
        for mi in sorted(
            local_instances,
            key=lambda mi: self._rank_health_last_dispatch.get(mi.id, 0),
        ):
            if (
                get_model_instance_workloads(mi) is not None
                and mi.state == ModelInstanceStateEnum.RUNNING
            ):
                self._sync_managed_rank_health(mi)

        # Use the event-driven local cache instead of an API call.
        model_instances = [
            mi
            for mi in local_instances
            if mi.worker_id == self._worker_id
            and mi.state == ModelInstanceStateEnum.RUNNING
            and get_model_instance_workloads(mi) is None
        ]
        if not model_instances:
            return

        now = time.time()

        for model_instance in model_instances:
            try:
                model = self._get_model(model_instance)
                if not model:
                    continue

                # Read per-model config from model.env.
                config = _get_inference_health_check_config(model, model_instance.role)
                if not config["enabled"]:
                    continue

                interval = config["interval"]
                timeout = config["timeout"]
                threshold = config["threshold"]

                # Skip if the model is still provisioning.
                if self._is_provisioning(model_instance):
                    continue

                # Skip if not enough time has passed since last check.
                last_check = self._last_health_check_time.get(model_instance.id, 0)
                if now - last_check < interval:
                    continue

                self._last_health_check_time[model_instance.id] = now

                # Skip if recent successful inference was observed for this instance.
                last_success = self._last_successful_inference.get(model_instance.id, 0)
                if last_success > now - interval:
                    logger.debug(
                        f"Model instance {model_instance.name} had recent successful "
                        f"inference, skipping health check."
                    )
                    # Reset failure count since real traffic is succeeding.
                    self._inference_health_check_failures.pop(model_instance.id, None)
                    continue

                # Perform inference health check.
                if not self._probe_running_instance(model_instance, model, timeout):
                    failure_count = self._inference_health_check_failures.get(
                        model_instance.id, 0
                    )
                    failure_count += 1
                    self._inference_health_check_failures[model_instance.id] = (
                        failure_count
                    )

                    if failure_count >= threshold:
                        logger.warning(
                            f"Model instance {model_instance.name} inference health check failed "
                            f"{failure_count} times, updating state to ERROR."
                        )
                        patch_dict = {
                            "state": ModelInstanceStateEnum.ERROR,
                            "state_message": _INFERENCE_HEALTH_CHECK_FAILED_MESSAGE,
                        }
                        self._report_model_instance(model_instance, **patch_dict)
                        # Reset failure count after marking as error.
                        del self._inference_health_check_failures[model_instance.id]
                    else:
                        logger.debug(
                            f"Model instance {model_instance.name} inference health check failed "
                            f"{failure_count}/{threshold} times."
                        )
                else:
                    # Reset failure count on success.
                    self._inference_health_check_failures.pop(model_instance.id, None)
            except Exception as e:
                logger.warning(
                    "Failed to sync inference health for model instance %s (%s)",
                    model_instance.name,
                    type(e).__name__,
                )

    def _handle_model_instance_event(self, event: Event):
        """Handle a model instance event without ever crashing the watch stream.

        The awatch callback runs inline in the watch loop, so any exception
        that escapes here tears the stream down and forces a full reconnect
        plus cache reload. Swallow and log (with traceback) instead; the next
        event or the periodic state sync recovers the instance.
        """
        try:
            self._dispatch_model_instance_event(event)
        except Exception:
            logger.exception(
                f"Failed to handle {event.type} event for model instance "
                f"{getattr(event, 'id', None)}"
            )

    async def _handle_model_instance_event_async(self, event: Event) -> None:
        # Port coordination may call back into this worker's HTTP API. Keep
        # its event loop available while synchronous launch work waits on RPC.
        await asyncio.to_thread(self._handle_model_instance_event, event)

    def _dispatch_model_instance_event(self, event: Event):  # noqa: C901
        """
        Handle model instance events.

        Args:
            event: The model instance event to handle.

        """
        mi = ModelInstance.model_validate(event.data)

        if model_instance_legacy_stop(mi) is not None:
            if self._is_local_model_instance(mi):
                self._stop_model_instance(
                    mi, delete_logs=event.type == EventType.DELETED
                )
            return
        if get_model_instance_workloads(mi) is not None:
            self._sync_managed_model_instance(mi)
            return

        logger.trace(
            f"Received event: {str(event.type)}, id: {mi.id}, name: {mi.name}, state: {str(mi.state)}"
        )

        is_main_worker = mi.worker_id == self._worker_id

        if is_main_worker:
            with self._model_instance_start_lock:
                self._model_instance_by_instance_id[mi.id] = mi
            # Return if all subordinate workers aren't running.
            if (
                mi.distributed_servers
                and mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.RUN_FIRST
                and mi.distributed_servers.subordinate_workers
            ):
                ready = all(
                    sw.state == ModelInstanceStateEnum.RUNNING
                    for sw in mi.distributed_servers.subordinate_workers
                )
                if not ready:
                    logger.info(
                        f"Model instance {mi.name} waits for all subordinate workers to be ready."
                    )
                    return
        else:
            # Return if it isn't a distribution serving.
            if not mi.distributed_servers:
                return
            # Return if it's a delegated distribution,
            # which means the main worker is responsible for serving.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.DELEGATED
            ):
                return
            # Return if it isn't the member of the distribution serving.
            joined = any(
                sw.worker_id == self._worker_id
                for sw in mi.distributed_servers.subordinate_workers or []
            )
            if not joined:
                return
            # Track local followers even while they wait to start. Reaping
            # needs their workload metadata after the server deletes the row.
            with self._model_instance_start_lock:
                self._model_instance_by_instance_id[mi.id] = mi
            # Return if the main worker isn't initialized.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                and (
                    mi.state
                    not in [
                        ModelInstanceStateEnum.STARTING,
                        ModelInstanceStateEnum.RUNNING,
                        ModelInstanceStateEnum.ERROR,
                    ]
                )
            ):
                logger.info(
                    f"Model instance {mi.name} waits for main worker {mi.worker_ip} to be initialized."
                )
                return
            # FIXME: This is a temporary solution to prevent the main worker from being unable to start due to phantom reads.
            #        We confirm whether the operation should be performed by checking the state of the earlier subordinate worker.
            for sw in mi.distributed_servers.subordinate_workers:
                if sw.worker_id == self._worker_id:
                    break
                if sw.state not in [
                    ModelInstanceStateEnum.RUNNING,
                    ModelInstanceStateEnum.ERROR,
                ]:
                    logger.info(
                        f"Model instance {mi.name} waits for previous subordinate worker {sw.worker_ip} to be ready."
                    )
                    return

        if event.type == EventType.DELETED:
            # Teardown is left to the periodic reap in sync_model_instances_state,
            # which is the authoritative reconciler and must run anyway to catch
            # DELETEDs missed during a watch disconnect. Tearing down here too
            # would give a second concurrent caller racing the reap on
            # delete_workload, so just let the reap reap it (within one tick).
            logger.trace(
                f"DELETED event for model instance {mi.name}; "
                "teardown deferred to reap."
            )
            return

        if event.type == EventType.UPDATED:
            # Keep errors retryable even if the live recovery policy is unavailable.
            # The background restart handler checks that policy before restarting.
            if mi.state == ModelInstanceStateEnum.ERROR:
                self._error_model_instances[mi.id] = mi
                return

            if mi.state == ModelInstanceStateEnum.RUNNING:
                self._error_model_instances.pop(mi.id, None)
                self._pending_model_instance_errors.pop(mi.id, None)

            # Restart if scheduled and this is the assigned worker.
            if is_main_worker and mi.state == ModelInstanceStateEnum.SCHEDULED:
                self._restart_model_instance(mi)
                logger.trace(
                    f"UPDATED event: restarted scheduled model instance {mi.name}."
                )

            # Start on subordinate worker if not started yet, or restart if failed.
            if not is_main_worker:
                self._start_or_restart_subordinate(mi)

            return

        if event.type == EventType.CREATED:
            # Only handle CREATED if this is the assigned worker
            if not is_main_worker:
                return
            if mi.state == ModelInstanceStateEnum.RUNNING:
                logger.warning(
                    f"Model instance {mi.name} is already running. Skipping start."
                )
                return
            self._start_model_instance(mi)
            logger.trace(f"CREATED event: started created model instance {mi.name}.")

    def _start_or_restart_subordinate(self, mi: ModelInstance) -> None:
        """Start a follower, respecting the deployment's recovery policy."""
        if get_model_instance_workloads(mi) is not None:
            self._sync_managed_model_instance(mi)
            return
        deployment_metadata = mi.get_deployment_metadata(self._worker_id)
        workload_name = deployment_metadata.name if deployment_metadata else mi.name
        workload = get_workload(workload_name)
        if workload and workload.state not in (
            WorkloadStatusStateEnum.UNKNOWN,
            WorkloadStatusStateEnum.UNHEALTHY,
            WorkloadStatusStateEnum.INACTIVE,
            WorkloadStatusStateEnum.FAILED,
        ):
            return

        subordinate = next(
            worker
            for worker in mi.distributed_servers.subordinate_workers
            if worker.worker_id == self._worker_id
        )
        initial_start = not workload and subordinate.state in (
            ModelInstanceStateEnum.PENDING,
            ModelInstanceStateEnum.SCHEDULED,
            ModelInstanceStateEnum.DOWNLOADING,
        )
        if self._uses_group_recovery(mi) and not initial_start:
            if self._is_provisioning(mi):
                return
            if subordinate.state != ModelInstanceStateEnum.ERROR:
                position = mi.distributed_servers.subordinate_workers.index(subordinate)
                self._report_model_instance(
                    mi,
                    **{
                        f'distributed_servers.subordinate_workers.{position}': subordinate.model_copy(
                            update={
                                'state': ModelInstanceStateEnum.ERROR,
                                'state_message': _describe_workload_failure(workload),
                            }
                        )
                    },
                )
            return
        # The parent can still be STARTING after this follower fails. Its
        # local state distinguishes first launch from recovery on later events.
        if not initial_start and not self._auto_restart_enabled(
            mi, use_cache=not bool(mi.group_id)
        ):
            logger.debug(
                "Skipping recovery of model instance %s on worker %s: "
                "automatic recovery is not allowed.",
                mi.name,
                self._worker_id,
            )
            return

        if workload:
            self._stop_model_instance(mi, clear_restart_backoff=False)
        self._start_model_instance(mi)
        logger.trace(
            f"UPDATED event: started model instance {mi.name} on subordinate worker."
        )

    def _get_numbered_log_path(self, mi: ModelInstance) -> str:
        """Get log file path with restart count.

        Args:
            mi: The model instance.

        Returns:
            Log file path with format:
            {log_dir}/{name}.{model_instance_id}/{restart_count}/main.log
        """
        restart_count = mi.restart_count or 0
        return str(
            main_log_path(Path(self._serve_log_dir), mi.name, mi.id, restart_count)
        )

    def _discover_sidecar_logs(
        self,
        mi_id: int,
        mi_name: str,
        workload_name: str,
        restart_count: int,
        persistence: _LogPersistence,
        resume: bool = False,
    ):
        """Background thread that waits for sidecar containers to appear.

        Polls get_workload() until sidecar containers are found in the
        loggable list, then starts log persistence threads for each.
        Exits when sidecars are found or stop_event is set.

        Args:
            mi_id: Model instance ID
            mi_name: Model instance name, for the log file names
            workload_name: Workload name
            restart_count: Current restart count for log file naming
            persistence: The generation record this thread belongs to.
            resume: Append to sidecar log files left behind by a previous worker
                process instead of rewriting them.
        """
        stop_event = persistence.stop_event
        while not stop_event.is_set():
            try:
                workload = get_workload(workload_name)
                if stop_event.is_set():
                    return
                if workload and workload.loggable:
                    sidecars = [op for op in workload.loggable if op.name != "default"]
                    if sidecars:
                        self._start_sidecar_log_threads(
                            mi_id,
                            mi_name,
                            workload_name,
                            workload.loggable,
                            restart_count,
                            persistence,
                            resume=resume,
                        )
                        logger.debug(f"Sidecar discovery for {workload_name} complete")
                        return
            except Exception:
                pass
            stop_event.wait(timeout=2)

    def _start_sidecar_log_threads(
        self,
        mi_id: int,
        mi_name: str,
        workload_name: str,
        loggable_ops: list,
        restart_count: int,
        persistence: _LogPersistence,
        resume: bool = False,
    ):
        """Start additional log persistence threads for sidecar containers.

        Called from the sidecar discovery thread once the workload is available
        and multiple loggable containers are discovered.

        Args:
            mi_id: Model instance ID
            mi_name: Model instance name, for the log file names
            workload_name: Workload name
            loggable_ops: List of WorkloadStatusOperation from workload.loggable
            restart_count: Current restart count for log file naming
            persistence: The generation record these threads belong to.
            resume: Append to sidecar log files left behind by a previous worker
                process instead of rewriting them.
        """
        names = []
        for op in loggable_ops:
            if op.name == "default":
                continue  # Main container handled by caller thread

            log_path = str(
                sidecar_container_log_path(
                    Path(self._serve_log_dir),
                    mi_name,
                    mi_id,
                    op.name,
                    restart_count,
                )
            )
            stop_event = threading.Event()

            thread = threading.Thread(
                target=self._persist_container_logs,
                args=(workload_name, log_path, stop_event, op.token),
                kwargs={"resume": resume},
                daemon=True,
                name=f"log-persist-{workload_name}-{op.name}",
            )
            # Tracked on this generation's record, never on the shared registry.
            if not persistence.start_aux_thread(thread, stop_event):
                break
            names.append(op.name)

        if names:
            logger.debug(
                f"Started sidecar log persistence threads for {workload_name}: "
                f"{names}"
            )

    def _start_container_log_persistence(self, mi: ModelInstance, resume: bool = False):
        """Start a background thread to persist container logs.

        Starts a single "main" log persistence thread. The thread will
        automatically discover sidecar containers (e.g., Ray head) once
        the workload is created, and spawn additional threads for each.

        Args:
            mi: The model instance.
            resume: Append to log files left behind by a previous worker process
                instead of rewriting them.
        """
        targets = engine_log_targets(mi, self._worker_id)
        if not targets:
            self._record_log_sources(mi)
            return
        workload_name = targets[0][0]

        restart_count = mi.restart_count or 0
        log_path = str(
            container_log_path(Path(self._serve_log_dir), mi.name, mi.id, restart_count)
        )

        stop_event = threading.Event()

        # Main container log thread.
        thread = threading.Thread(
            target=self._persist_container_logs,
            args=(workload_name, log_path, stop_event),
            kwargs={"resume": resume},
            daemon=True,
            name=f"log-persist-{workload_name}",
        )
        persistence = _LogPersistence(stop_event, thread, restart_count)
        persistence.workload_name = workload_name

        # Sidecar discovery thread — polls until sidecar containers appear,
        # then starts additional log threads for each.
        discovery_thread = threading.Thread(
            target=self._discover_sidecar_logs,
            args=(mi.id, mi.name, workload_name, restart_count, persistence),
            kwargs={"resume": resume},
            daemon=True,
            name=f"log-discover-{workload_name}",
        )

        # Retire, register and start in one critical section: a concurrent start
        # would otherwise leave one generation with nothing left to signal it.
        with self._log_persistence_lock:
            current = self._log_persistence.get(mi.id)
            if current and current.restart_count > restart_count:
                return
            if not self._retire_log_persistence(mi.id):
                stuck = self._log_persistence[mi.id]
                if stuck.restart_count == restart_count:
                    # The previous generation still holds these files, and a
                    # second writer truncates the shards it is filling. The next
                    # sync tries again, by when it has come back from the runtime.
                    logger.warning(
                        f"Deferred container log persistence for {mi.name}: the "
                        f"previous one is still running"
                    )
                    return
                # An earlier restart's generation writes other files, and it
                # has been told to stop. Waiting on it would lose this restart's
                # logs: nothing retries a start for a container not yet running.
                logger.warning(
                    f"Container log persistence for {mi.name} restart "
                    f"{stuck.restart_count} is still stopping; starting restart "
                    f"{restart_count} beside it"
                )
            self._log_persistence[mi.id] = persistence
            thread.start()
            persistence.start_aux_thread(discovery_thread)
        self._record_log_sources(mi, persistence)
        self._sync_workload_log_persistence(mi, persistence)
        self._sync_component_log_persistence(mi, persistence)
        logger.debug(f"Started container log persistence thread for {mi.name}")

    def _ensure_container_log_persistence(self, mi: ModelInstance):
        """Re-attach container log persistence to an instance being adopted.

        The threads live in this process, so a worker restart ends them while the
        workload keeps running, and the replayed CREATED event returns early for
        an already-RUNNING instance. Runs every sync, so it must leave a live
        thread alone.

        Args:
            mi: The model instance.
        """
        # Only the main log thread counts: the sidecar discovery thread beside it
        # polls forever on a single-container workload.
        restart = mi.restart_count or 0
        targets = engine_log_targets(mi, self._worker_id)
        if not targets:
            self._record_log_sources(mi)
            return
        managed = get_model_instance_workloads(mi) is not None
        with self._log_persistence_lock:
            persistence = self._log_persistence.get(mi.id)
            if persistence and persistence.restart_count > restart:
                return
            reuse = (
                persistence is not None
                and persistence.restart_count == restart
                and persistence.main_thread.is_alive()
                and (not managed or persistence.workload_name == targets[0][0])
            )
            if reuse and persistence.stop_event.is_set():
                return
            if not reuse:
                # Hold the epoch check through local directory maintenance.
                self._align_instance_log_dir(mi)
                self._align_legacy_main_log(mi)
                self._migrate_flat_logs(mi)
        if reuse:
            self._record_log_sources(mi, persistence)
            self._sync_workload_log_persistence(mi, persistence)
            self._sync_component_log_persistence(mi, persistence)
            return

        logger.info(
            f"Re-attaching container log persistence for adopted model instance "
            f"{mi.name}"
        )
        self._start_container_log_persistence(mi, resume=True)

    def _sync_workload_log_persistence(self, mi, persistence) -> None:
        """Attach each additional Engine to its own archive and reconnect cursor."""
        with self._log_persistence_lock:
            if (
                self._log_persistence.get(mi.id) is not persistence
                or persistence.restart_count != (mi.restart_count or 0)
                or persistence.stop_event.is_set()
            ):
                return
            for name, archive_name in engine_log_targets(mi, self._worker_id)[1:]:
                previous = persistence.workload_writers.get(name)
                if previous:
                    thread, stopped = previous
                    if thread.is_alive():
                        continue
                    persistence.discard_aux_thread(thread, stopped)
                path = sidecar_container_log_path(
                    Path(self._serve_log_dir),
                    mi.name,
                    mi.id,
                    archive_name,
                    mi.restart_count or 0,
                )
                stopped = threading.Event()
                thread = threading.Thread(
                    target=self._persist_container_logs,
                    args=(name, str(path), stopped),
                    kwargs={'resume': True},
                    daemon=True,
                    name=f'log-persist-{name}',
                )
                if persistence.start_aux_thread(thread, stopped):
                    persistence.workload_writers[name] = (thread, stopped)

    def _record_log_sources(self, mi: ModelInstance, persistence=None) -> None:
        """Publish accepted identities without making logging an Engine fault."""
        try:
            with self._log_persistence_lock:
                current = self._log_persistence.get(mi.id)
                if (persistence is not None and current is not persistence) or (
                    current is not None
                    and current.restart_count != (mi.restart_count or 0)
                ):
                    return
                bands = mi.named_ports or {}
                multi_port_external = None
                if '_vllm_multi_port_external_supervisor' in bands:
                    multi_port_external = True
                elif all(
                    name in bands
                    for name in (
                        '_vllm_external_api',
                        '_vllm_external_rpc',
                        '_vllm_mp_connecting',
                    )
                ):
                    multi_port_external = False
                write_log_sources(
                    Path(self._serve_log_dir), mi, self._worker_id, multi_port_external
                )
        except Exception as exc:
            logger.warning("Cannot persist log sources for instance %s: %s", mi.id, exc)

    def _sync_component_log_persistence(self, mi, persistence) -> None:
        """Switch only the SMG writer, after its old writer confirms termination."""
        task = next(
            (
                w
                for w in get_model_instance_workloads(mi, self._worker_id) or []
                if w.component and w.component.kind == 'sglang_smg'
            ),
            None,
        )
        if task is None:
            return
        with self._log_persistence_lock:
            if (
                self._log_persistence.get(mi.id) is not persistence
                or persistence.restart_count != (mi.restart_count or 0)
                or persistence.stop_event.is_set()
                or task.component.attempt < persistence.component_attempt
            ):
                return
            previous = persistence.component_writer
            if previous:
                name, thread, stopped = previous
                if name == task.name and thread.is_alive():
                    return
                stopped.set()
                thread.join(timeout=0)
                if thread.is_alive():
                    return
                persistence.discard_aux_thread(thread, stopped)
            if task.state in (
                ModelInstanceWorkloadStateEnum.STOPPED,
                ModelInstanceWorkloadStateEnum.FAILED,
            ):
                return
            root = find_instance_log_dir(
                Path(self._serve_log_dir), mi.id
            ) or instance_log_dir(Path(self._serve_log_dir), mi.name, mi.id)
            path = root / str(mi.restart_count or 0) / 'component.sglang_smg.log'
            stopped = threading.Event()
            thread = threading.Thread(
                target=self._persist_container_logs,
                args=(task.name, str(path), stopped),
                kwargs=dict(resume=True, lineage=task.name),
                daemon=True,
                name=f'log-component-{task.name}',
            )
            if persistence.start_aux_thread(thread, stopped):
                persistence.component_writer = (task.name, thread, stopped)
                persistence.component_attempt = task.component.attempt

    def _align_legacy_main_log(self, mi: ModelInstance):
        """Rename a pre-v2.2.0 main log to carry the current restart_count.

        A {id}.log counts as restart 0, so the log viewer would otherwise file it
        under a different restart than the container log adoption creates.

        Args:
            mi: The model instance.
        """
        try:
            legacy_log = legacy_main_log_path(Path(self._serve_log_dir), mi.id)
            numbered_log = Path(self._get_numbered_log_path(mi))
            if not legacy_log.exists() or numbered_log.exists():
                return
            numbered_log.parent.mkdir(parents=True, exist_ok=True)
            legacy_log.rename(numbered_log)
            logger.info(f"Renamed legacy serve log {legacy_log} to {numbered_log}")
        except Exception as e:
            logger.warning(f"Failed to align legacy serve log for {mi.name}: {e}")

    def _align_instance_log_dir(self, mi: ModelInstance):
        """Rename an instance's log directory after its current name.

        One rename moves every restart with it. Opportunistic: a directory that
        cannot be renamed keeps its old name, which discovery still finds by id.

        Args:
            mi: The model instance.
        """
        log_dir = Path(self._serve_log_dir)
        target = instance_log_dir(log_dir, mi.name, mi.id)
        try:
            current = find_instance_log_dir(log_dir, mi.id)
            if current is None or current == target or target.exists():
                return
            current.rename(target)
            logger.info(f"Renamed serve log directory {current.name} to {target.name}")
        except OSError as e:
            logger.warning(f"Failed to rename serve log directory to {target}: {e}")

    def _migrate_flat_logs(self, mi: ModelInstance):
        """Move logs an earlier release wrote into this instance's directory.

        Safe under a live writer: the rename is atomic and an open handle
        follows the inode. Opportunistic -- a file that cannot be moved keeps
        its old place, which the reader still finds. The pre-v2.2.0 {id}.log is
        left to _align_legacy_main_log, which files it under the current
        restart rather than restart 0.

        Args:
            mi: The model instance.
        """
        log_dir = Path(self._serve_log_dir)
        for path, parsed in flat_instance_logs(log_dir, mi.id):
            if parsed.legacy:
                continue
            target = canonical_log_path(log_dir, mi.name, parsed)
            if target == path or target.exists():
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
                logger.info(f"Moved serve log {path.name} to {target}")
            except OSError as e:
                logger.warning(f"Failed to move serve log {path} to {target}: {e}")

    def _stop_container_log_persistence(
        self, model_instance_id: int, timeout: float = 2.0
    ):
        """Stop all container log persistence threads for a model instance.

        Args:
            model_instance_id: The model instance ID
            timeout: Maximum time to wait for each thread to stop (seconds)
        """
        # A thread still inside the runtime stays registered and exits on its
        # stop event once the stream returns; nothing retries the stop.
        with self._log_persistence_lock:
            self._retire_log_persistence(model_instance_id, timeout)

    def _retire_log_persistence(
        self, model_instance_id: int, timeout: float = 2.0
    ) -> bool:
        """Drop a model instance's log persistence generation and wait it out.

        The caller must hold ``_log_persistence_lock``; joining under it is safe
        because no log persistence thread ever takes it.

        Args:
            model_instance_id: The model instance ID
            timeout: Maximum time to wait for each thread to stop (seconds)

        Returns:
            Whether nothing of the generation is left running.
        """
        persistence = self._log_persistence.get(model_instance_id)
        if persistence is None:
            return True
        stopped = persistence.stop(model_instance_id, timeout)
        if stopped:
            del self._log_persistence[model_instance_id]
        # A generation that outlasted the wait stays registered: it is still
        # holding the log files, and dropping it here would leave the next
        # start with nothing to find and nothing to wait for.
        return stopped

    def _cleanup_old_logs(self, model_instance_id: int, current_restart_count: int):
        """Keep serve logs for restart_count in {R, R-1}.

        R==0 is a fresh lifecycle start, so all existing files are purged (any
        present belong to a previous owner of a reused id).
        """
        if current_restart_count == 0:
            self._purge_instance_logs(model_instance_id)
            return

        keep = {current_restart_count, current_restart_count - 1}
        log_dir = Path(self._serve_log_dir)

        # Runs before every start, and a failed listing must not stop one.
        try:
            restart_dirs = restart_log_dirs(log_dir, model_instance_id)
            flat_logs = flat_instance_logs(log_dir, model_instance_id)
        except Exception as e:
            logger.error(f"Failed to cleanup old logs for {model_instance_id}: {e}")
            return

        for restart_count, path in restart_dirs.items():
            if restart_count in keep:
                continue
            try:
                shutil.rmtree(path)
                logger.info(f"Deleted serve logs of restart {restart_count}: {path}")
            except OSError as e:
                logger.warning(f"Failed to delete serve log directory {path}: {e}")

        # Whatever an earlier release left flat is retired by the same window.
        # An unreadable name never reaches here, so no window can retire it.
        for path, parsed in flat_logs:
            if parsed.restart_count in keep:
                continue
            try:
                path.unlink()
                logger.info(f"Deleted old serve log file: {path}")
            except OSError as e:
                logger.warning(f"Failed to delete serve log file {path}: {e}")

    def _purge_instance_logs(self, model_instance_id: int):
        """Delete every serve log of a model instance id, wherever it lives.

        Membership is decided by the parsed id, never by a glob: this feeds a
        delete, so a name the layout cannot read is left alone.
        """
        log_dir = Path(self._serve_log_dir)

        # Runs before a fresh start, and a failed listing must not stop it.
        try:
            instance_dir = find_instance_log_dir(log_dir, model_instance_id)
            flat_logs = flat_instance_logs(log_dir, model_instance_id)
        except Exception as e:
            logger.error(
                f"Failed to purge logs for model instance {model_instance_id}: {e}"
            )
            return

        if instance_dir is not None:
            try:
                shutil.rmtree(instance_dir)
                logger.info(f"Deleted serve log directory: {instance_dir}")
            except OSError as e:
                logger.warning(
                    f"Failed to delete serve log directory {instance_dir}: {e}"
                )

        for path, _parsed in flat_logs:
            try:
                path.unlink()
                logger.info(f"Deleted serve log file: {path}")
            except OSError as e:
                logger.warning(f"Failed to delete serve log file {path}: {e}")

    def _start_model_instance(self, mi: ModelInstance):
        """Start once, deferring periodic reaping until process creation finishes."""
        if get_model_instance_workloads(mi) is not None:
            self._sync_managed_model_instance(mi)
            return
        with self._model_instance_start_lock:
            if mi.id in self._starting_model_instances:
                return
            self._starting_model_instances.add(mi.id)
        try:
            self._start_assigned_model_instance(mi)
        finally:
            with self._model_instance_start_lock:
                self._starting_model_instances.discard(mi.id)

    @_model_lifecycle
    def _start_assigned_model_instance(self, mi: ModelInstance) -> None:
        # A watch event may arrive after the instance has been deleted.
        # Bypass the event cache before allocating ports or creating work.
        try:
            current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
            if model_instance_legacy_stop(current) is not None:
                self._stop_model_instance(ModelInstance.model_validate(current))
                return
            if self._uses_group_recovery(mi) and not self._legacy_start_allowed(
                mi, current
            ):
                return
        except NotFoundException:
            logger.debug(
                "Skipping start of deleted model instance %s (id=%s).", mi.name, mi.id
            )
            self._stop_model_instance(mi, delete_logs=True)
            return
        except Exception as e:
            self._handle_model_instance_launch_failure(mi, e)
            return
        try:
            if mi.group_id:
                model = self._clientset.models.get(mi.model_id, use_cache=False)
                if model.stale is True:
                    raise RuntimeError(PD_CONFIGURATION_CHANGED_MESSAGE)
        except Exception as e:
            self._handle_model_instance_launch_failure(mi, e)
            return
        self._launch_model_instance(mi)

    def _legacy_start_allowed(self, mi, current) -> bool:
        """Accept an initial local launch only in the authoritative active round."""
        if (
            (current.restart_count or 0) != (mi.restart_count or 0)
            or current.state == ModelInstanceStateEnum.ERROR
            or current.draining_since is not None
            or self._worker_id not in member_worker_ids(current)
        ):
            return False
        if current.worker_id == self._worker_id:
            return current.state == ModelInstanceStateEnum.SCHEDULED
        subordinate = next(
            w
            for w in current.distributed_servers.subordinate_workers
            if w.worker_id == self._worker_id
        )
        return subordinate.state in (
            ModelInstanceStateEnum.PENDING,
            ModelInstanceStateEnum.SCHEDULED,
            ModelInstanceStateEnum.DOWNLOADING,
        )

    def _launch_model_instance(self, mi: ModelInstance):  # noqa: C901
        """
        Start model instance through a subprocess.

        Args:
            mi: The model instance to start.

        """
        managed = get_model_instance_workloads(mi) is not None
        if managed and mi.worker_id != self._worker_id and not mi.ports:
            return
        if managed and get_model_instance_workloads(mi, self._worker_id):
            self._sync_managed_model_instance(mi)
            return
        if self._is_provisioning(mi):
            logger.warning(f"Model instance {mi.name} is provisioning. Skipping start.")
            return

        # Restart teardown removes the local entry. Restore it before taking
        # ports so failed starts and missed delete events can still be reaped.
        with self._model_instance_start_lock:
            self._model_instance_by_instance_id[mi.id] = mi

        # Follow a rename before anything writes under the new name, or the
        # instance's logs end up split across two directories.
        self._align_instance_log_dir(mi)
        # Clean up old log files before starting
        self._cleanup_old_logs(mi.id, mi.restart_count or 0)

        is_main_worker = mi.worker_id == self._worker_id

        log_file_path = self._get_numbered_log_path(mi)

        sw_pos: Optional[int] = None
        sw: Optional[ModelInstanceSubordinateWorker] = None
        if not is_main_worker:
            sw_pos = next(
                (
                    i
                    for i, sw in enumerate(mi.distributed_servers.subordinate_workers)
                    if sw.worker_id == self._worker_id
                ),
            )
            sw = mi.distributed_servers.subordinate_workers[sw_pos]

        ports_assigned = False
        try:
            if model_instance_legacy_stop(mi) is not None:
                self._stop_model_instance(mi)
                return
            model = self._get_model(mi)
            backend = get_backend(model)

            if managed:
                if not self._managed_launcher_available(mi, model):
                    raise ValueError(
                        "No managed workload launcher is available for this deployment."
                    )
                if backend == BackendEnum.VLLM:
                    self._assign_external_ports(mi, model)
                elif mi.spans_workers and not mi.ports:
                    self._assign_distributed_ports(mi, model, backend)
                else:
                    self._assign_ports(mi, model, backend)
            else:
                self._assign_ports(mi, model, backend)
            ports_assigned = True

            if managed:
                fields = {"state": ModelInstanceStateEnum.INITIALIZING}
                if is_main_worker:
                    fields.update(
                        port=mi.port, ports=mi.ports, named_ports=mi.named_ports
                    )
                self._report_model_instance(mi, **fields)

            logger.debug(
                f"Starting model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

            fallback_registry = (
                registration.determine_default_registry(
                    self._config.system_default_container_registry
                )
                if is_built_in_backend(backend)
                else None
            )

            sender = None
            if managed:
                receiver, sender = multiprocessing.Pipe(duplex=False)
                self._managed_launches[(mi.id, mi.restart_count or 0)] = receiver
            process = multiprocessing.Process(
                target=ServeManager._serve_model_instance,
                args=(
                    mi,
                    backend,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                    self._worker_id,
                    self._inference_backend_manager.get_backend_by_name(
                        backend, model.owner_principal_id
                    ),
                    fallback_registry,
                )
                + ((sender,) if managed else ()),
            )
            process.daemon = False
            try:
                process.start()
            finally:
                if sender is not None:
                    sender.close()
            self._provisioning_processes[mi.id] = process

            # Start container log persistence for containerized backends
            self._start_container_log_persistence(mi)

            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.INITIALIZING,
                    "port": mi.port,
                    "ports": mi.ports,
                    # The bands are allocated on the worker but read on the
                    # server (the router's peer config), so they persist the
                    # same way `port`/`ports` do.
                    "named_ports": mi.named_ports,
                    "pid": process.pid,
                }
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.INITIALIZING
                # For initialize later mode, the state is set to RUNNING directly,
                # which means the subordinate worker doesn't need to wait for the main worker to be healthy.
                if (
                    mi.distributed_servers.mode
                    == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                ):
                    sw.state = ModelInstanceStateEnum.RUNNING
                sw.pid = process.pid
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            if managed:
                self._report_model_instance(
                    mi, state=ModelInstanceStateEnum.INITIALIZING, pid=process.pid
                )
            else:
                self._report_model_instance(mi, **patch_dict)
            self._error_model_instances.pop(mi.id, None)
            self._pending_model_instance_errors.pop(mi.id, None)
            logger.info(
                f"Started model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

        except Exception as e:
            # Reservations can exist before process creation succeeds. Delete
            # any workload before releasing them, even if no PID was recorded.
            if ports_assigned or mi.id in self._provisioning_processes:
                try:
                    self._stop_model_instance(mi)
                except Exception as cleanup_error:
                    logger.warning(
                        "Failed to clean up model instance %s after launch failure: %s",
                        mi.name,
                        cleanup_error,
                    )

            self._handle_model_instance_launch_failure(mi, e)

    def _assign_external_ports(self, mi: ModelInstance, model: Model) -> None:
        """Reserve all local rank APIs and the shared coordination ports atomically."""
        if mi.spans_workers and not mi.ports:
            self._assign_distributed_ports(mi, model, BackendEnum.VLLM)
            return
        with self._port_reservations.allocation(
            self._clientset, self._worker_id, model_instance_id=mi.id
        ) as unavailable:
            if mi.ports:
                taken = set(mi.ports)
                for band in (mi.named_ports or {}).values():
                    taken.update(range(band.base, band.base + band.count))
                if taken & unavailable:
                    raise RuntimeError(
                        "Managed External-LB ports are reserved by another instance."
                    )
            else:
                self._select_external_ports(
                    mi, model, unavailable, self._config.service_port_range
                )
                taken = model_instance_ports(mi, self._worker_id)
            self._assigned_ports[mi.id] = taken

    def _select_external_ports(self, mi, model, unavailable, port_range) -> None:
        counts = [len(mi.gpu_indexes or [])] + [
            len(w.gpu_indexes or [])
            for w in mi.distributed_servers.subordinate_workers or []
        ]
        requests = model_port_requests(model, mi.role, counts, managed=True)
        bands = self._allocate_port_plan(mi, requests, unavailable, port_range)
        mi.port = bands["_vllm_external_api"].base
        api_count = bands["_vllm_external_api"].count
        connector_names = [spec.name for spec in band_specs_for(model, mi.role)]
        mi.ports = [
            *range(mi.port, mi.port + api_count),
            *[
                band.base
                for name, band in bands.items()
                if name != "_vllm_external_api" and name not in connector_names
            ],
            *[
                port
                for name in connector_names
                if name in bands
                for port in range(
                    bands[name].base, bands[name].base + bands[name].count
                )
            ],
        ]
        mi.named_ports = bands

    @staticmethod
    def _allocate_port_plan(mi, requests, unavailable, port_range):
        """Select every band in the shared plan without publishing partial ports."""
        bands = {}
        for name, count, fixed in requests:
            if fixed is not None:
                low, high = network.parse_port_range(port_range)
                if (
                    not low <= fixed <= high
                    or fixed in unavailable
                    or not network.is_port_available(fixed, host=mi.worker_ip)
                ):
                    raise ValueError(
                        "Multi-port External-LB supervisor port is unavailable."
                    )
                base = fixed
            else:
                try:
                    base = network.get_free_band(
                        port_range=port_range,
                        count=count,
                        unavailable_ports=unavailable,
                        host=mi.worker_ip,
                    )
                except network.PortRangeExhaustedError as error:
                    raise network.PortRangeExhaustedError(
                        f"Model instance {mi.name} (role '{mi.role}') needs a "
                        f"{count}-port band for '{name}': {error}"
                    ) from error
            unavailable.update(range(base, base + count))
            bands[name] = PortBand(base=base, count=count)
        return bands

    def _component_preparer_handed_off(self, mi):
        if (mi.id, mi.restart_count or 0) in self._managed_stopping:
            return False
        try:
            current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
        except Exception:
            tracked = self._model_instance_by_instance_id.get(mi.id)
            return self._has_sglang_components(mi) or (
                tracked is not None and self._has_sglang_components(tracked)
            )
        if not self._has_sglang_components(current):
            return False
        if not self._same_component_epoch(mi, current):
            return True
        return any(
            w.component.kind == 'sglang_engine'
            and w.state
            not in (
                ModelInstanceWorkloadStateEnum.FAILED,
                ModelInstanceWorkloadStateEnum.STOPPED,
            )
            for w in get_model_instance_workloads(current, self._worker_id)
        )

    def _handle_model_instance_launch_failure(
        self, mi: ModelInstance, error: Exception
    ) -> None:
        """Retain launch failures until error reporting and policy reads succeed."""
        if self._component_preparer_handed_off(mi):
            logger.warning(
                'Ignoring retired component preparer failure for %s: %s', mi.id, error
            )
            return
        message = f"Failed to start model instance: {error}"
        if (
            mi.worker_id == self._worker_id
            or get_model_instance_workloads(mi) is not None
        ):
            patch_dict = {
                "state": ModelInstanceStateEnum.ERROR,
                "state_message": message,
            }
        else:
            sw_pos, sw = next(
                (i, sw)
                for i, sw in enumerate(mi.distributed_servers.subordinate_workers)
                if sw.worker_id == self._worker_id
            )
            sw.state = ModelInstanceStateEnum.ERROR
            sw.state_message = message
            patch_dict = {f"distributed_servers.subordinate_workers.{sw_pos}": sw}

        with self._model_instance_start_lock:
            self._model_instance_by_instance_id[mi.id] = mi
        self._error_model_instances[mi.id] = mi
        self._pending_model_instance_errors[mi.id] = patch_dict
        logger.error(f"Failed to start model instance {mi.name}: {error}")
        self._report_model_instance(mi, **patch_dict)
        self._pending_model_instance_errors.pop(mi.id, None)

    def _assign_ports(
        self,
        mi: ModelInstance,
        model: Model,
        backend: BackendEnum,
    ) -> None:
        """
        Assign ports to the model instance.

        This method is thread-safe and allocates ports for:
        - Main serving port
        - RPC port for vLLM DP communication (if applicable)
        - Connecting port for subordinate workers (if applicable)
        - Named connector bands declared by the instance's PD role

        Args:
            mi: The model instance to assign ports to.
            model: The model associated with the instance.
            backend: The backend type (e.g., vLLM, SGLang).
        """
        if (
            not mi.port
            and mi.worker_id == self._worker_id
            and mi.distributed_servers
            and mi.distributed_servers.mode
            == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
            and mi.distributed_servers.subordinate_workers
        ):
            self._assign_distributed_ports(mi, model, backend)
            return
        cleanup_waits = 0
        while True:
            pending_cleanup = False
            try:
                with self._port_reservations.allocation(
                    self._clientset, self._worker_id, model_instance_id=mi.id
                ) as unavailable_ports:
                    pending_cleanup = unavailable_ports.pending_cleanup
                    self._assign_local_ports(mi, model, backend, unavailable_ports)
                return
            except network.PortRangeExhaustedError:
                if not self._wait_for_port_cleanup(mi, pending_cleanup, cleanup_waits):
                    raise
                cleanup_waits += 1

    def _assign_local_ports(
        self,
        mi: ModelInstance,
        model: Model,
        backend: BackendEnum,
        unavailable_ports: PortReservationSnapshot,
    ) -> None:
        if mi.port:
            # Published ports stay stable across restarts, but cannot be
            # reused if another workload also claims them.
            conflicts = model_instance_ports(mi, self._worker_id) & unavailable_ports
            if conflicts:
                logger.warning(
                    "Port reservation conflict on worker %s for model instance %s: %s",
                    self._worker_id,
                    mi.id,
                    unavailable_ports.describe(conflicts),
                )
                raise RuntimeError(
                    f"Model instance ports {sorted(conflicts)} are reserved "
                    "by another instance. Recreate the conflicting instance."
                )
            self._register_assigned_ports(mi)
            return

        self._select_ports(
            mi, model, backend, unavailable_ports, self._config.service_port_range
        )
        self._register_assigned_ports(mi)

    def _wait_for_port_cleanup(
        self, mi: ModelInstance, pending_cleanup: bool, waits: int
    ) -> bool:
        """Retry temporary exhaustion without releasing a stopping owner's ports."""
        if not pending_cleanup or waits >= _PORT_CLEANUP_RETRIES:
            return False
        logger.info(
            "Waiting for old workload port cleanup before starting model instance %s "
            "(%s/%ss)",
            mi.name,
            (waits + 1) * _PORT_CLEANUP_RETRY_INTERVAL,
            _PORT_CLEANUP_RETRIES * _PORT_CLEANUP_RETRY_INTERVAL,
        )
        time.sleep(_PORT_CLEANUP_RETRY_INTERVAL)
        self._clientset.model_instances.get(id=mi.id, use_cache=False)
        return True

    def _port_coordinator(self) -> DistributedPortClient:
        if self._distributed_port_client is None:
            self._distributed_port_client = DistributedPortClient(self._clientset)
        return self._distributed_port_client

    def _assign_distributed_ports(
        self, mi: ModelInstance, model: Model, backend: BackendEnum
    ) -> None:
        """Choose a shared layout and fence it on every member before launch."""
        coordinator = self._port_coordinator()
        self._release_distributed_ports(mi.id)
        rejected: Set[int] = set()
        attempts = cleanup_waits = 0
        while attempts < 5:
            snapshots = coordinator.availability(mi.id)
            if {item.worker_id for item in snapshots} != set(member_worker_ids(mi)):
                raise RuntimeError("Incomplete worker port availability snapshot")
            ranges = [network.parse_port_range(item.port_range) for item in snapshots]
            start, end = max(item[0] for item in ranges), min(
                item[1] for item in ranges
            )
            if start > end:
                raise network.PortRangeExhaustedError(
                    "Participating workers have no common service port range"
                )
            unavailable = set(rejected)
            for snapshot in snapshots:
                unavailable.update(snapshot.unavailable_ports)
            candidate = mi.model_copy(deep=True)
            try:
                self._select_ports(
                    candidate, model, backend, unavailable, f"{start}-{end}"
                )
            except network.PortRangeExhaustedError:
                if not self._wait_for_port_cleanup(
                    mi, any(item.pending_cleanup for item in snapshots), cleanup_waits
                ):
                    raise
                cleanup_waits += 1
                rejected.clear()
                continue
            attempts += 1
            primary = next(item for item in snapshots if item.worker_id == mi.worker_id)
            claim = ModelPortClaim(
                token=uuid4(),
                restart_count=(
                    (mi.restart_count or 0)
                    if get_model_instance_workloads(mi) is not None
                    else None
                ),
                generations={item.worker_id: item.generation for item in snapshots},
                # A primary without a tentative claim can recover followers left
                # behind by its restart, using their exact observed revisions.
                replaces={
                    item.worker_id: item.claim_version
                    for item in snapshots
                    if primary.claim_version is None and item.claim_version is not None
                },
                ports=sorted(model_instance_ports(candidate, self._worker_id)),
            )
            with self._distributed_port_claim_lock:
                self._distributed_port_claims[mi.id] = claim.token
            try:
                result = coordinator.reserve(mi.id, claim)
                if result.accepted:
                    self._commit_distributed_ports(mi, candidate, claim)
                    return
            except _LocalPortReservationChanged as conflict:
                self._release_distributed_ports(mi.id, claim.token)
                rejected.update(conflict.ports)
                continue
            except Exception:
                try:
                    self._release_distributed_ports(mi.id, claim.token)
                except Exception:
                    logger.warning(
                        "Deferred distributed port cleanup for model instance %s",
                        mi.id,
                    )
                raise
            self._forget_distributed_claim(mi.id, claim.token)
            rejected.update(result.unavailable_ports)
        raise RuntimeError("Model ports changed during cross-worker allocation; retry")

    def _commit_distributed_ports(
        self, mi: ModelInstance, candidate: ModelInstance, claim: ModelPortClaim
    ) -> None:
        """Publish only a claim still owned after every worker acknowledged it."""
        with self._port_reservations.allocation(
            self._clientset, self._worker_id, model_instance_id=mi.id
        ) as taken:
            if self._port_reservations.model_claims.get(mi.id) != (
                str(claim.token),
                set(claim.ports),
            ):
                raise RuntimeError(
                    "Distributed model port claim changed before publication"
                )
            conflicts = set(claim.ports) & taken
            if conflicts:
                raise _LocalPortReservationChanged(conflicts)
            # Registration and the final ownership check share a lock. Recovery
            # cannot replace a claim once these local workload ports are held.
            self._register_assigned_ports(candidate)
            mi.port = candidate.port
            mi.ports = candidate.ports
            mi.named_ports = candidate.named_ports

    def _forget_distributed_claim(self, instance_id: int, token: UUID) -> None:
        with self._distributed_port_claim_lock:
            if self._distributed_port_claims.get(instance_id) == token:
                self._distributed_port_claims.pop(instance_id)

    def _release_distributed_ports(
        self, instance_id: int, token: Optional[UUID] = None
    ) -> None:
        with self._distributed_port_claim_lock:
            token = token or self._distributed_port_claims.get(instance_id)
            if token is not None:
                self._pending_distributed_port_releases.add((instance_id, token))
        if token is not None:
            self._port_coordinator().release(instance_id, token)
            self._forget_distributed_claim(instance_id, token)
            with self._distributed_port_claim_lock:
                self._pending_distributed_port_releases.discard((instance_id, token))

    def _retry_distributed_port_releases(self) -> None:
        with self._distributed_port_claim_lock:
            pending = list(self._pending_distributed_port_releases)
        for instance_id, token in pending:
            try:
                self._release_distributed_ports(instance_id, token)
            except Exception:
                logger.warning(
                    "Deferred distributed port cleanup for model instance %s",
                    instance_id,
                )

    @staticmethod
    def _select_sglang_ports(mi, model, unavailable, port_range) -> None:
        from gpustack.utils.model_instance_workers import sglang_smg_mode

        if mi.role is not None or not sglang_smg_mode(model):
            raise ValueError(
                "Managed components require ordinary multi-rank SGLang DP/DPA"
            )
        counts = [len(mi.gpu_indexes or [])] + [
            len(worker.gpu_indexes or [])
            for worker in mi.distributed_servers.subordinate_workers or []
        ]
        bands = ServeManager._allocate_port_plan(
            mi, model_port_requests(model, mi.role, counts), unavailable, port_range
        )
        mi.port = bands["_sglang_engine_api"].base
        mi.ports = [band.base for band in bands.values()]
        mi.named_ports = bands

    def _select_ports(
        self,
        mi: ModelInstance,
        model: Model,
        backend: BackendEnum,
        unavailable_ports: Set[int],
        port_range: str,
    ) -> None:
        """Select a complete layout without publishing or reserving partial results."""
        if get_model_instance_workloads(mi) is not None and backend == BackendEnum.VLLM:
            self._select_external_ports(mi, model, unavailable_ports, port_range)
            return
        if (
            get_model_instance_workloads(mi) is not None
            and backend == BackendEnum.SGLANG
        ):
            self._select_sglang_ports(mi, model, unavailable_ports, port_range)
            return
        counts = (
            [len(mi.gpu_indexes or [])]
            + [
                len(worker.gpu_indexes or [])
                for worker in (mi.distributed_servers.subordinate_workers or [])
            ]
            if mi.distributed_servers
            else [len(mi.gpu_indexes or [])]
        )
        requests = model_port_requests(model, mi.role, counts, managed=False)
        bands = self._allocate_port_plan(mi, requests, unavailable_ports, port_range)
        connector_names = [spec.name for spec in band_specs_for(model, mi.role)]
        named = dict(mi.named_ports or {})
        named.update({name: bands[name] for name in connector_names if name in bands})
        api = bands["_api"].base
        ports = [api]
        ports.extend(
            band.base for name, band in bands.items() if name.startswith("_cross_")
        )
        ports.extend(
            port
            for name in connector_names
            if name in bands
            for port in range(bands[name].base, bands[name].base + bands[name].count)
        )
        connecting = bands.get("_connecting")
        if connecting is not None:
            # Backends consume the connecting base as ports[-1]; derived
            # protection ports belong only in the named reservation metadata.
            ports.append(connecting.base)
            if connecting.count > 1:
                named[_VLLM_MP_CONNECTING_BAND_NAME] = connecting
        mi.port, mi.ports = api, ports
        if named:
            mi.named_ports = named

    def _mark_crash_loop(
        self, mi: ModelInstance, workload, is_main_worker: bool
    ) -> bool:
        """Turn a member that keeps restarting without serving into an ERROR.

        Returns True once it has been marked, so the caller stops treating the
        workload as merely slow to start.

        Only the main worker's row is written. A subordinate worker's state
        lives inside `distributed_servers`, and the existing failure path above
        already owns that shape; duplicating it here to catch a loop would mean
        two writers for one field.
        """
        if not is_main_worker or mi.state == ModelInstanceStateEnum.ERROR:
            return False

        restarts = max(
            (
                exit_.restart_count or 0
                for exit_ in (getattr(workload, "exits", None) or [])
            ),
            default=0,
        )
        now = datetime.now(timezone.utc)
        if not self._restart_tracker.observe_restart_count(mi.id, restarts, now):
            return False

        # The reason comes from the log rather than from the workload, because
        # the workload has none: a container that exited and was restarted
        # reports no message on Kubernetes and no exit code worth surfacing.
        # The log is where the root cause was written, several lines before the
        # exception that ended the process.
        diagnosis = diagnose(self._read_container_log(mi), mi.named_ports)
        message = f"Restarted {restarts} times without serving. " + (
            f"{diagnosis.summary} Log: {diagnosis.line}"
            if diagnosis
            else "No known failure signature was found in its log; check "
            "the instance log for the first error, not the last."
        )
        with contextlib.suppress(NotFoundException):
            self._report_model_instance(
                mi, state=ModelInstanceStateEnum.ERROR, state_message=message
            )
        logger.warning(f"Model instance {mi.name} is crash-looping: {message}")
        return True

    def _append_log_diagnosis(self, mi: ModelInstance, message: str) -> str:
        """Add what the log says to what the runtime says, when it knows more.

        The runtime's account of a failure is frequently just an exit code, and
        an exit code names the symptom. The log holds the cause, usually
        several lines before whatever ended the process — which is why the
        signature scan reports the earliest match rather than the last.

        Best-effort in both directions: no log, no match, or a read that throws
        all leave the message exactly as it arrived. A less specific failure
        message is a much smaller problem than an instance that fails to be
        marked failed.
        """
        try:
            diagnosis = diagnose(self._read_container_log(mi), mi.named_ports)
        except Exception:
            return message
        if not diagnosis:
            return message
        return f"{message} {diagnosis.summary} Log: {diagnosis.line}"

    def _read_container_log(self, mi: ModelInstance, limit: int = 256_000) -> str:
        """The tail of this instance's most recent container log.

        A tail rather than the whole file: an engine's startup log is large and
        the signatures being looked for are startup-time. Failures to read are
        swallowed — a missing log makes the diagnosis less specific, and must
        not stop the instance being marked failed.
        """
        try:
            log_dir = Path(self._serve_log_dir)
            candidates = sorted(
                log_dir.glob(f"{mi.id}.container.*.log"),
                key=lambda p: p.stat().st_mtime,
            )
            if not candidates:
                return ""
            path = candidates[-1]
            size = path.stat().st_size
            with open(path, "r", errors="replace") as f:
                if size > limit:
                    f.seek(size - limit)
                return f.read()
        except Exception as e:
            logger.debug(f"Failed to read the container log for {mi.name}: {e}")
            return ""

    def _register_assigned_ports(self, mi: ModelInstance) -> None:
        """Re-register an instance's already-persisted ports as taken.

        Covers both indexes: the named bands are whole runs, and only their
        base is stored, so the fence has to be re-expanded from `count`.
        Callers must use the shared port reservation allocation context.
        """
        taken = model_instance_ports(mi, self._worker_id)
        if taken:
            self._assigned_ports[mi.id] = taken

    def _restart_model_instance(self, mi: ModelInstance):
        """
        Restart model instance.

        Args:
            mi: The model instance to restart.
        """

        if model_instance_legacy_stop(mi) is not None:
            self._stop_model_instance(mi)
            return
        if get_model_instance_workloads(mi) is not None:
            self._sync_managed_model_instance(mi)
            return
        if self._uses_group_recovery(mi):
            current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
            if (
                (current.restart_count or 0) == (mi.restart_count or 0)
                and current.state == ModelInstanceStateEnum.SCHEDULED
                and not self._is_provisioning(mi)
                and get_workload(mi.get_deployment_metadata(self._worker_id).name)
                is None
            ):
                self._start_model_instance(mi)
            return
        self._stop_model_instance(mi, clear_restart_backoff=False)
        self._start_model_instance(mi)

    def _update_model(self, id: int, **kwargs):
        """
        Update model instance with given fields.

        Args:
            id: The ID of the model instance to update.
            **kwargs: The fields to update, group by field name and value.
        """

        try:
            m_public = self._clientset.models.get(id=id)

            m = ModelUpdate(**m_public.model_dump())
            for key, value in kwargs.items():
                set_attr(m, key, value)

            self._clientset.models.update(id=id, model_update=m)
        except NotFoundException:
            logger.warning(f"Model with ID {id} not found when trying to update.")

    def _report_model_instance(self, mi: ModelInstance, **fields):
        """Keep the observed round attached to local runtime state reports."""
        if self._uses_group_recovery(mi):
            return self._clientset.model_instances.report_workloads(
                id=mi.id,
                report=build_member_state_report(mi, self._worker_id, **fields),
            )
        return self._update_model_instance(mi.id, **fields)

    def _uses_group_recovery(self, mi: ModelInstance) -> bool:
        """Read the Server's round policy without consulting mutable model inputs."""
        if calculate_group_recovery_policy(mi, None) is None:
            return False
        if mi.distributed_servers.group_recovery is None:
            current = self._clientset.model_instances.get(id=mi.id, use_cache=False)
            if (current.restart_count or 0) != (mi.restart_count or 0):
                raise RuntimeError('Instance generation changed during recovery')
            mi.distributed_servers = current.distributed_servers.model_copy(deep=True)
        if mi.distributed_servers.group_recovery is None:
            raise RuntimeError('Instance recovery policy is not initialized')
        return requires_group_recovery(mi)

    def _update_model_instance(self, id: int, **kwargs):
        """
        Update model instance with given fields.

        Args:
            id: The ID of the model instance to update.
            **kwargs: The fields to update, group by field name and value.
        """

        try:
            mi_public = self._clientset.model_instances.get(id=id)
            if get_model_instance_workloads(mi_public) is not None:
                current = self._clientset.model_instances.get(id=id, use_cache=False)
                tracked = self._model_instance_by_instance_id.get(id)
                if tracked and (current.restart_count or 0) != (
                    tracked.restart_count or 0
                ):
                    raise RuntimeError("Instance generation changed during preparation")
                return self._clientset.model_instances.report_workloads(
                    id=id,
                    report=managed_instance_update(current, **kwargs),
                )

            mi = ModelInstanceUpdate(**mi_public.model_dump())
            for key, value in kwargs.items():
                set_attr(mi, key, value)

            self._clientset.model_instances.update(id=id, model_update=mi)
        except NotFoundException:
            logger.warning(
                f"Model instance with ID {id} not found when trying to update."
            )

    def _stop_model_instance(
        self,
        mi: ModelInstance,
        clear_restart_backoff: bool = True,
        delete_logs: bool = False,
    ):
        """
        Stop model instance and clean up.

        Args:
            mi: The model instance to stop.
            clear_restart_backoff: Whether to clear transient restart backoff state.
            delete_logs: Whether to remove the instance's serve log files. Only set
                on permanent teardown (DELETED); a restart must keep them so the
                log viewer can still show the previous run.
        """

        if get_model_instance_workloads(mi) is not None:
            key = (mi.id, mi.restart_count or 0)
            with self._model_instance_start_lock:
                self._managed_stopping[key] = (
                    self._managed_stopping.get(key, False) or delete_logs
                )
        return self._stop_assigned_model_instance(
            mi, clear_restart_backoff, delete_logs
        )

    @_model_lifecycle
    def _stop_assigned_model_instance(
        self, mi: ModelInstance, clear_restart_backoff: bool, delete_logs: bool
    ):
        if model_instance_legacy_stop(mi) is not None:
            return self._stop_legacy_guarded_model_instance(mi, delete_logs)
        if get_model_instance_workloads(mi) is not None:
            key = (mi.id, mi.restart_count or 0)
            return self._stop_managed_model_instance(
                mi, self._managed_stopping.get(key, delete_logs)
            )
        logger.debug(f"Stopping model instance {mi.name or mi.id}")

        # Stop container log persistence thread
        self._stop_container_log_persistence(mi.id)

        if delete_logs:
            self._purge_instance_logs(mi.id)

        # Teardown provisioning process if still alive.
        if self._is_provisioning(mi):
            terminate_multiprocessing_process(self._provisioning_processes[mi.id])

        # Delete workload.
        deployment_metadata = mi.get_deployment_metadata(self._worker_id)
        if deployment_metadata:
            delete_workload(deployment_metadata.name)

        self._forget_stopped_model_instance(mi, clear_restart_backoff, delete_logs)

    def _forget_stopped_model_instance(
        self,
        mi: ModelInstance,
        clear_restart_backoff: bool = True,
        delete_logs: bool = False,
        *,
        retain_instance: bool = False,
    ) -> None:
        """Release local state after workload cleanup.

        Managed stop finalization retains the instance until all cleanup intent is consumed.
        """
        try:
            self._release_distributed_ports(mi.id)
        except Exception:
            logger.warning(
                "Remote port cleanup pending for stopped model instance %s", mi.id
            )
        self._provisioning_processes.pop(mi.id, None)
        self._port_reservations.release_model(mi.id, forget_generation=delete_logs)
        self._error_model_instances.pop(mi.id, None)
        self._pending_model_instance_errors.pop(mi.id, None)
        self._model_cache_by_instance.pop(mi.id, None)
        with self._model_instance_start_lock:
            if not retain_instance:
                self._model_instance_by_instance_id.pop(mi.id, None)
        if clear_restart_backoff:
            self._restart_backoff_counts.pop(mi.id, None)
            # Same condition as the backoff on purpose. The crash-loop verdict
            # and the restart backoff answer the same question — "is this one
            # still worth retrying" — so a stop that keeps the backoff (the
            # restart path) has to keep the loop history too, or a member being
            # restarted for the fourth time looks like one being started for
            # the first.
            self._restart_tracker.forget(mi.id)
        self._clear_rank_health(mi.id)
        self._inference_health_check_failures.pop(mi.id, None)
        self._last_health_check_time.pop(mi.id, None)
        self._last_successful_inference.pop(mi.id, None)
        self._rank_health_last_dispatch.pop(mi.id, None)

        logger.info(f"Stopped model instance {mi.name or mi.id}")

    def _restart_error_model_instance(self, mi: ModelInstance):
        """
        Restart error model instance with exponential backoff,
        maximum delay 5 minutes.

        Args:
            mi: The model instance to restart.
        """
        if model_instance_legacy_stop(mi) is not None:
            self._stop_model_instance(mi)
            return
        if get_model_instance_workloads(mi) is not None:
            self._sync_managed_model_instance(mi)
            return
        if patch_dict := self._pending_model_instance_errors.get(mi.id):
            self._report_model_instance(mi, **patch_dict)
            self._pending_model_instance_errors.pop(mi.id, None)

        if self._uses_group_recovery(mi):
            return

        if self._is_provisioning(mi):
            logger.debug(f"Model instance {mi.name} is provisioning. Skipping restart.")
            return

        backoff_count = self._restart_backoff_counts.get(mi.id, 0)
        current_time = datetime.now(timezone.utc)
        delay = min(10 * (2 ** (backoff_count - 1)), 300) if backoff_count > 0 else 0
        if self._restart_backoff_pending(mi, delay, current_time):
            return

        # Watch events and the restart loop's queue snapshot can lag recovery.
        # Bypass the watch cache to check whether an error still exists.
        try:
            mi = self._clientset.model_instances.get(id=mi.id, use_cache=False)
        except NotFoundException:
            self._error_model_instances.pop(mi.id, None)
            self._pending_model_instance_errors.pop(mi.id, None)
            return

        if self._worker_id not in member_worker_ids(mi):
            self._error_model_instances.pop(mi.id, None)
            self._pending_model_instance_errors.pop(mi.id, None)
            return

        subordinate_error = mi.distributed_servers and any(
            sw.worker_id == self._worker_id and sw.state == ModelInstanceStateEnum.ERROR
            for sw in mi.distributed_servers.subordinate_workers or []
        )
        if mi.state != ModelInstanceStateEnum.ERROR and not subordinate_error:
            self._error_model_instances.pop(mi.id, None)
            self._pending_model_instance_errors.pop(mi.id, None)
            return

        # Another worker may have restarted the instance since it was queued.
        if self._restart_backoff_pending(mi, delay, current_time):
            return

        # Only a PD recovery that is due needs current model configuration.
        if not self._auto_restart_enabled(mi, use_cache=not bool(mi.group_id)):
            self._error_model_instances.pop(mi.id, None)
            return

        logger.info(
            f"Restarting model instance {mi.name} "
            f"(attempt {backoff_count + 1}) after {delay} seconds delay."
        )

        with contextlib.suppress(NotFoundException):
            self._restart_backoff_counts[mi.id] = backoff_count + 1
            self._report_model_instance(
                mi,
                restart_count=(mi.restart_count or 0) + 1,
                last_restart_time=current_time,
                state=ModelInstanceStateEnum.SCHEDULED,
                state_message='',
            )

        # Pop from error model instances,
        # if failed to restart next time, it will be added again in watch_model_instance_events().
        self._error_model_instances.pop(mi.id, None)

    def _restart_backoff_pending(
        self, mi: ModelInstance, delay: float, current_time: datetime
    ) -> bool:
        """Check retry timing locally before reading instance or model state."""
        last_restart_time = mi.last_restart_time or mi.updated_at
        if not delay or not last_restart_time:
            return False
        remaining = delay - (current_time - last_restart_time).total_seconds()
        if remaining <= 0:
            return False
        logger.trace(f"Delaying restart of {mi.name} for {remaining:.2f} seconds.")
        return True

    def _auto_restart_enabled(self, mi: ModelInstance, use_cache: bool = True) -> bool:
        """Use the watch cache for polling and refresh before PD recovery."""
        try:
            model = self._clientset.models.get(mi.model_id, use_cache=use_cache)
        except NotFoundException:
            return False
        return bool(model.restart_on_error) and not (
            mi.group_id and model.stale is True
        )

    def _get_model(self, mi: ModelInstance) -> Model:
        """
        Efficiently get model related to the model instance with caching.

        Args:
            mi: The model instance whose model to get.
        """
        if model := self._model_cache_by_instance.get(mi.id):
            return model

        # Project the role's overrides here too. This is a third cluster of
        # worker-side readers, and every one of them wants the role's value:
        # the vGPU type and backend version at sync, `env` for the health-check
        # config, and the backend that decides the port band and the fallback
        # registry at start. Without the projection this manager would size and
        # probe an instance from the Model-level spec while the child process
        # runs from the role's — the two would disagree on the same instance.
        # The cache is already keyed per instance, so projecting inside it is
        # exactly per-role.
        model = role_effective_model(self._clientset.models.get(mi.model_id), mi.role)
        # A managed router is a custom-backend image plus a command, and this
        # manager has to know that before the child process exists: the backend
        # is what picks the port band and the fallback registry. No peers are
        # passed — the command they render into is the child's business, and
        # rendering it here would only be rendering it twice.
        model = apply_managed_router(model, mi.role)
        self._model_cache_by_instance[mi.id] = model
        return model

    def _refresh_model(self, mi: ModelInstance) -> Model:
        """
        Refresh the model information from the server.

        Args:
            mi: The model instance whose model to refresh.

        Returns:
            The refreshed model.
        """
        logger.debug(f"Refreshing model {mi.model_name} information from server.")
        refreshed_model = role_effective_model(
            self._clientset.models.get(mi.model_id), mi.role
        )
        refreshed_model = apply_managed_router(refreshed_model, mi.role)
        self._model_cache_by_instance[mi.id] = refreshed_model
        return refreshed_model

    def _is_provisioning(self, mi: ModelInstance) -> bool:
        """
        Check if the model instance is still provisioning.

        Args:
            mi: The model instance to check.
        """
        if process := self._provisioning_processes.get(mi.id):
            if process.is_alive():
                process.join(timeout=0)
                return process.is_alive()
        return False

    def _get_health_check_path(
        self, backend: str, owner_principal_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Get health check path for the given backend.

        Args:
            backend: The backend name.
            owner_principal_id: Owner of the model being served, used to
                resolve an Org-scoped backend row over the Platform one.
        Returns:
            The health check path if exists, else None.
        """
        inference_backend = self._inference_backend_manager.get_backend_by_name(
            backend, owner_principal_id
        )

        return inference_backend.health_check_path if inference_backend else None

    def get_instance_port_by_model_instance_id(
        self, model_instance_id: int
    ) -> Optional[int]:
        """Return the current local serving port for a single-entry instance."""
        endpoint = self.get_instance_endpoint_by_model_instance_id(model_instance_id)
        return endpoint.port if endpoint is not None else None

    def get_instance_endpoint_by_model_instance_id(
        self,
        model_instance_id: int,
        *,
        model_id: Optional[int] = None,
        dp_rank: Optional[int] = None,
        worker_id: Optional[int] = None,
    ) -> Optional[ModelInstanceEndpoint]:
        """Resolve a stable target while fencing local and watched launch state."""
        local = self._model_instance_by_instance_id.get(model_instance_id)
        instance = local
        if (
            model_id is not None
            or dp_rank is not None
            or worker_id is not None
            or (local is not None and get_model_instance_workloads(local) is not None)
        ):
            instance = self._clientset.model_instances.get_cached(model_instance_id)
        if instance is None:
            return None
        if get_model_instance_workloads(instance) is None and not instance.ports:
            return None
        try:
            endpoint = resolve_model_instance_endpoint(
                instance,
                ModelInstanceTarget(
                    instance.model_id if model_id is None else model_id,
                    model_instance_id,
                    dp_rank=dp_rank,
                    worker_id=worker_id,
                ),
                worker_id=self._worker_id,
            )
        except (ValueError, NotFoundException):
            return None
        if get_model_instance_workloads(instance) is not None:
            if (model_instance_id, instance.restart_count or 0) in getattr(
                self, "_managed_stopping", {}
            ):
                return None
            # Watch delivery and local launch observations advance independently.
            # Resolve the same target locally before forwarding to a current port.
            if local is not None:
                try:
                    observed = resolve_model_instance_endpoint(
                        local,
                        ModelInstanceTarget(
                            instance.model_id,
                            model_instance_id,
                            dp_rank=dp_rank,
                            worker_id=worker_id,
                        ),
                        worker_id=self._worker_id,
                    )
                except ValueError:
                    return None
                if observed != endpoint:
                    return None
        return endpoint


def is_ready(
    backend: str,
    mi: ModelInstance,
    health_check_path: Optional[str] = None,
    model: Model = None,
    timeout: Optional[int] = None,
) -> bool:
    """
    Access the health endpoint of the given model instance to check if it is servable.

    A readiness probe waits five seconds for SGLang's /health endpoint and one
    second for other paths. A caller that turns a failure into ERROR passes its
    own timeout so a busy engine is not retired for answering late.
    """
    is_built_in = is_built_in_backend(backend)
    custom_image = bool(
        model
        and backend in (BackendEnum.VLLM, BackendEnum.SGLANG)
        and is_built_in_backend_custom_version(
            backend, model.backend_version, model.image_name
        )
    )
    allow_missing_health_route = custom_image and not health_check_path
    if (not is_built_in or backend == BackendEnum.CUSTOM) and (not health_check_path):
        # If custom backend does not have health check path, consider it always ready.
        return True

    if backend == BackendEnum.ASCEND_MINDIE and not health_check_path:
        # Ref: https://www.hiascend.com/document/detail/zh/mindie/21RC2/mindieservice/servicedev/mindie_service0066.html
        # /info provides metadata information and requires more time to respond. Use it for health check.
        health_check_path = "/info"
    elif (
        backend == BackendEnum.SGLANG
        and model
        and CategoryEnum.IMAGE in model.categories
        and not health_check_path
    ):
        if custom_image:
            # A custom image's version key does not establish which health
            # endpoints its server implements.
            health_check_path = "/health"
        elif not model.backend_version:
            # version may be empty at initialization, consider it not ready.
            return False
        elif compare_versions(model.backend_version, "0.5.5.post3") >= 0:
            # SGLang Diffusion supported health check path at v0.5.5.post3
            health_check_path = "/health"
        else:
            # Older versions do not support health check, consider it always ready.
            return True
    elif backend in (BackendEnum.VLLM, BackendEnum.SGLANG) and not health_check_path:
        health_check_path = "/health"
    elif is_built_in and backend != BackendEnum.CUSTOM and not health_check_path:
        # VoxBox uses the OpenAI model list as its default health endpoint.
        health_check_path = "/v1/models"

    if timeout is None:
        timeout = (
            5 if backend == BackendEnum.SGLANG and health_check_path == "/health" else 1
        )

    try:
        # Use the worker IP instead of localhost for health check.
        # Reasons:
        # 1. Connectivity to the loopback address does not work with Ascend MindIE.
        # 2. More adaptable to container networks.
        health_check_url = f"http://{mi.worker_ip}:{mi.port}{health_check_path}"
        response = requests.get(health_check_url, timeout=timeout)
        if response.status_code == 200:
            return True
        if allow_missing_health_route and response.status_code == 404:
            logger.warning(
                "Model instance %s uses a custom image without %s; accepting "
                "HTTP 404 only confirms the server responds and does not "
                "confirm model readiness. Set GPUSTACK_MODEL_HEALTH_CHECK_PATH "
                "to an endpoint that reports model readiness.",
                mi.name,
                health_check_path,
            )
            return True
    except Exception as e:
        logger.debug(f"Error checking model instance {mi.name} health: {e}")
        pass
    return False


def _get_inference_endpoint_and_payload(model: Model) -> tuple[str, dict] | None:
    """
    Get inference endpoint and payload for the model.
    Returns None if the model type should skip health check.
    """
    skip_categories = {
        CategoryEnum.IMAGE,
        CategoryEnum.SPEECH_TO_TEXT,
        CategoryEnum.TEXT_TO_SPEECH,
        CategoryEnum.UNKNOWN,
        # A decision deployment answers the Jev /v1/systemone contract,
        # not the generation/embedding/rerank endpoints this probe imitates;
        # probing /v1/chat/completions would mark it unavailable.
        CategoryEnum.DECISION,
    }
    if not skip_categories.isdisjoint(model.categories):
        return None

    # Return endpoint and payload based on model type (priority order)
    if CategoryEnum.EMBEDDING in model.categories:
        return "/v1/embeddings", {"model": model.name, "input": "test"}

    if CategoryEnum.RERANKER in model.categories:
        return "/v1/rerank", {
            "model": model.name,
            "query": "test",
            "documents": ["test"],
        }

    return "/v1/chat/completions", {
        "model": model.name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "max_completion_tokens": 1,
    }


def _get_inference_health_check_config(
    model: Model, role: Optional[str] = None
) -> dict:
    """Read per-model inference health check config from model.env."""
    env = model.env or {}
    enabled = env.get(
        "GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED", "false"
    ).lower() in (
        "true",
        "1",
    )
    interval = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL"),
        300,
    )
    # SGLang's /health can wait up to SGLANG_HEALTH_CHECK_TIMEOUT
    # (20s by default) for a scheduler response before returning 503.
    # Periodic PD probes allow 25s so the server can return its verdict.
    # Ref: https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/entrypoints/http_server.py#L193
    default_timeout = (
        25
        if get_backend(model) == BackendEnum.SGLANG
        and model.disaggregation is not None
        and role
        and role != RoleNameEnum.ROUTER.value
        else 15
    )
    timeout = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT"), default_timeout
    )
    threshold = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD"),
        3,
    )
    return {
        "enabled": enabled,
        "interval": interval,
        "timeout": timeout,
        "threshold": threshold,
    }


def is_inference_ready(
    mi: ModelInstance, model: Model, timeout: int = 15, *, allow_redirects: bool = True
) -> bool:
    """
    Send a minimal inference request to verify the inference capability is working.
    """
    # Check Custom backend (no standard inference API). A managed router is the
    # exception: it is only *launched* as a custom backend, and it serves the
    # group's OpenAI API -- which is exactly what this probe is for.
    if is_custom_backend(model.backend) and not is_managed_router(model, mi.role):
        return True

    # Check port assignment
    if not mi.port:
        logger.debug(f"Model instance {mi.name} does not have port assigned yet.")
        return False

    # Get endpoint and payload, None means skip health check
    result = _get_inference_endpoint_and_payload(model)
    if not result:
        logger.debug(f"Skipping inference check for {mi.name}")
        return True

    endpoint_path, payload = result
    inference_url = f"http://{mi.worker_ip}:{mi.port}{endpoint_path}"

    try:
        options = {} if allow_redirects else {"allow_redirects": False}
        response = requests.post(
            inference_url, json=payload, timeout=timeout, **options
        )
        if response.status_code == 200:
            return True
        else:
            logger.warning(
                f"Model instance {mi.name} inference health check failed "
                f"with status {response.status_code} for endpoint {endpoint_path}"
            )
    except Exception as e:
        logger.debug(
            f"Error checking model instance {mi.name} inference at {endpoint_path}: {e}"
        )

    return False


def _probe_rank_health(mi, model, path, kind, timeout) -> Tuple[bool, str]:
    """A direct rank probe; HTTP 200 does not prove generation can progress."""
    try:
        if kind == 'inference':
            healthy = is_inference_ready(
                mi, model, timeout=timeout, allow_redirects=False
            )
            return healthy, 'inference request failed' if not healthy else ''
        url = f"http://{mi.worker_ip}:{mi.port}{path}"
        response = requests.get(url, timeout=timeout, allow_redirects=False)
        try:
            reason = f'HTTP {response.status_code}'
            if response.status_code == 404:
                reason += '; configure GPUSTACK_MODEL_HEALTH_CHECK_PATH for this rank'
            return response.status_code == 200, reason
        finally:
            response.close()
    except requests.Timeout:
        return False, f'timeout ({timeout}s)'
    except requests.ConnectionError:
        return False, 'connection failed'
    except Exception as exc:
        return False, type(exc).__name__


def _component_get(url: str, deadline: float) -> requests.Response:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise requests.Timeout('component probe budget exhausted')
    response = requests.get(url, timeout=remaining, allow_redirects=False)
    if time.monotonic() >= deadline:
        response.close()
        raise requests.Timeout('component probe budget exhausted')
    return response


def _component_json(url: str, deadline: float) -> dict:
    response = _component_get(url, deadline)
    try:
        if response.status_code != 200:
            raise ValueError(f'HTTP {response.status_code}')
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    finally:
        response.close()


def _probe_sglang_component_health(
    mi: ModelInstance, kind: str, expected_dp: int, engine_url: str, timeout: float
) -> Tuple[bool, str]:
    """Probe pinned native response contracts within one absolute time budget."""
    deadline = time.monotonic() + timeout
    url = f'http://{mi.worker_ip}:{mi.port}'
    try:
        if kind == 'sglang_engine':
            health = _component_get(f'{url}/health', deadline)
            try:
                if health.status_code != 200:
                    return False, f'Engine health HTTP {health.status_code}'
            finally:
                health.close()
            response = _component_get(f'{url}/server_info', deadline)
            try:
                fallback = response.status_code == 404
                if not fallback and response.status_code != 200:
                    return False, f'Engine server_info HTTP {response.status_code}'
                info = None if fallback else response.json()
            finally:
                response.close()
            if fallback:
                info = _component_json(f'{url}/get_server_info', deadline)
            actual = info.get('dp_size') if isinstance(info, dict) else None
            attention_dp = (
                info.get('attn_dp_size', 1) if isinstance(info, dict) else None
            )
            # Each replica exposes one routing target per attention DP rank.
            return (
                type(actual) is int
                and type(attention_dp) is int
                and actual >= 1
                and attention_dp >= 1
                and actual * attention_dp == expected_dp
            ), 'Engine dp_size mismatch or invalid server_info'
        # Publication follows complete-group recovery, even when the native
        # router could still reach a healthy subset of logical ranks.
        ready = _component_json(f'{url}/readiness', deadline)
        if ready.get('status') != 'ready' or any(
            type(ready.get(k)) is not int or ready[k] != expected_dp
            for k in ('healthy_workers', 'total_workers')
        ):
            return False, 'SMG discovery not ready'
        data = _component_json(f'{url}/workers', deadline)
        workers = data.get('workers')
        if (
            type(data.get('total')) is not int
            or data['total'] != expected_dp
            or not isinstance(workers, list)
            or len(workers) != expected_dp
        ):
            return False, 'SMG worker count mismatch'
        expected = {f'{engine_url}@{rank}' for rank in range(expected_dp)}
        actual = []
        for worker in workers:
            if (
                not isinstance(worker, dict)
                or worker.get('worker_type') != 'regular'
                or worker.get('connection_mode') != 'HTTP'
                or worker.get('is_healthy') is not True
                or not isinstance(worker.get('url'), str)
            ):
                return False, 'SMG worker metadata or health invalid'
            actual.append(worker['url'])
        return (
            len(set(actual)) == expected_dp and set(actual) == expected,
            'SMG logical worker identities mismatch',
        )
    except requests.Timeout:
        return False, f'timeout ({timeout}s)'
    except requests.ConnectionError:
        return False, 'connection failed'
    except Exception as exc:
        return False, type(exc).__name__
