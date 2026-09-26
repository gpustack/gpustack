import asyncio
from collections import deque
import contextlib
from datetime import datetime, timezone
import json
import multiprocessing
import re
import threading
import time

import requests
import setproctitle
import os
from typing import Dict, Optional, Set, List, Callable
from pathlib import Path
import logging

from gpustack_runtime.deployer import (
    get_workload,
    WorkloadStatusStateEnum,
    delete_workload,
    logs_workload,
)
from gpustack_runtime.deployer.__utils__ import compare_versions

from gpustack import envs
from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.config import registration
from gpustack.logging import (
    RedirectStdoutStderr,
)
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    is_built_in_backend,
    is_custom_backend,
)
from gpustack.utils import network
from gpustack.utils.convert import safe_int
from gpustack.utils.attrs import set_attr
from gpustack.utils.command import find_int_parameter
from gpustack.utils.process import terminate_process_tree, add_signal_handlers
from gpustack.worker.backends.ascend_mindie import AscendMindIEServer
from gpustack.worker.backends.sglang import SGLangServer
from gpustack.utils.command import resolve_executor_backend
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.worker.backends.vox_box import VoxBoxServer
from gpustack.worker.backends.custom import CustomServer
from gpustack.worker.log_sources import (
    existing_legacy_main_log,
    extract_container_restart_count,
    extract_restart_count,
    legacy_main_log_path,
)
from gpustack.worker.model_meta import get_meta_from_running_instance
from gpustack.client import ClientSet
from gpustack.worker.pd_router import (
    apply_managed_router,
    is_managed_router,
    managed_router_health_path,
)
from gpustack.worker.pd_diagnostics import RestartTracker, diagnose

from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelUpdate,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    get_backend,
    DistributedServerCoordinateModeEnum,
    ModelInstanceSubordinateWorker,
    CategoryEnum,
    PortBand,
    RoleNameEnum,
    role_effective_model,
)
from gpustack.schemas.pd_modes import PDPortScopeEnum
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.worker.pd_injection import (
    ACCELERATOR_COUNT_KEY,
    band_count_key,
    band_specs_for,
    band_width,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.inference_backend_manager import InferenceBackendManager

logger = logging.getLogger(__name__)

# Inference health check error message
_INFERENCE_HEALTH_CHECK_FAILED_MESSAGE = "Inference health check failed."

# Last-resort message for a workload that stopped serving without saying why.
_WORKLOAD_FAILED_MESSAGE = "Inference server exited or unhealthy."

# One health-check cycle (+2s margin) to let a container return after a stream
# EOF; beyond that gpustack marks it ERROR and takes over recovery.
LOG_RECONNECT_GRACE_SECONDS = envs.MODEL_INSTANCE_HEALTH_CHECK_INTERVAL + 2

# Tail read when seeding a resumed anchor; widened up to the max for a record
# longer than one read.
_LOG_TAIL_CHUNK_SIZE = 8192
_LOG_TAIL_MAX_READ = 1 << 20

# How long a seeded anchor may hold back a followed stream before rewriting.
# A line count cannot bound this: live output reaches the same skip as replay.
_LOG_RESUME_SKIP_TIMEOUT = 30.0

# Global lock for port assignment to avoid pickle serialization issues
_port_lock = threading.Lock()

# The `count` spelling that means "as many ports as this member has cards".

# vLLM's mp path does not bind only VLLM_DP_MASTER_PORT: it derives nine more
# ports from it (one per DP init attempt), so the connecting port is the *base*
# of a band and every port in that band has to be probed, fenced and recorded
# like any other allocation. Ten is vLLM's number, not ours.
_VLLM_MP_CONNECTING_BAND = 10

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


class PDPortScopeUnsupportedError(Exception):
    """A band declares a scope this allocator cannot honour.

    Raised rather than degraded to per-instance. `PDPortScopeEnum.ROLE` means
    "every member of this role shares one band", which needs a registry keyed
    by (group, role) that does not exist yet. Allocating such a band per
    instance produces the one thing worse than an unimplemented feature: a
    band of the right *width* at the wrong *base* on each member, so the
    connector that was supposed to meet on it only fails at handshake, long
    after the instances report healthy.
    """


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


def _tail_lines(log_path: str, count: int) -> List[str]:
    """The last `count` complete lines of a log file, or [] if unreadable.

    Splits on '\\n' alone to match the runtime's log framing: str.splitlines()
    also splits on '\\r', so one progress-bar line would become several pieces
    that can never equal one streamed line.

    Args:
        log_path: Path to the log file.
        count: Maximum number of lines to return.

    Returns:
        The trailing lines, newlines included, oldest first.
    """
    try:
        with open(log_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = _LOG_TAIL_CHUNK_SIZE
            while True:
                f.seek(max(0, size - read_size))
                text = f.read().decode('utf-8', errors='replace')

                # Drop whatever follows the final '\n', and the first line when
                # the read boundary cut it.
                lines = [line + '\n' for line in text.split('\n')[:-1]]
                if read_size < size and lines:
                    lines.pop(0)

                # Widen instead of reporting "no anchor": the caller would then
                # reopen in 'w' and delete the history it is adopting.
                if lines or read_size >= size or read_size >= _LOG_TAIL_MAX_READ:
                    return lines[-count:]
                read_size *= 2
    except OSError:
        return []


def _drop_partial_last_line(log_path: str):
    """Truncate a log file's trailing line when it carries no newline.

    A worker killed mid-write leaves a fragment; the runtime replays that line
    whole, so appending after the fragment would join the two.

    Args:
        log_path: Path to the log file.
    """
    try:
        with open(log_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return
            f.seek(size - 1)
            if f.read(1) == b'\n':
                return
            f.seek(max(0, size - _LOG_TAIL_CHUNK_SIZE))
            chunk = f.read()

        last_newline = chunk.rfind(b'\n')
        if last_newline < 0:
            # No boundary within reach; resume degrades to a rewrite anyway.
            return
        os.truncate(log_path, size - (len(chunk) - last_newline - 1))
    except OSError as e:
        logger.warning(f"Failed to trim the partial last line of {log_path}: {e}")


class _LogPersistence:
    """The threads and stop events of one generation of a model instance's log
    persistence.

    A sidecar thread only mutates its own generation's record, never the shared
    registry -- which is what lets the registry be swapped under a lock while
    threads are joined, with no joined thread waiting on that lock.
    """

    def __init__(self, stop_event: threading.Event, main_thread: threading.Thread):
        self.stop_event = stop_event
        self.main_thread = main_thread
        self._lock = threading.Lock()
        self._stopping = False
        self._aux_threads: List[threading.Thread] = []
        self._aux_stop_events: List[threading.Event] = []

    def add_aux_thread(
        self, thread: threading.Thread, stop_event: Optional[threading.Event] = None
    ):
        """Track a thread of this generation, or stop it if teardown has begun.

        Args:
            thread: The thread to track.
            stop_event: Its own stop event, if it has one.
        """
        with self._lock:
            if not self._stopping:
                self._aux_threads.append(thread)
                if stop_event is not None:
                    self._aux_stop_events.append(stop_event)
                return
        if stop_event is not None:
            stop_event.set()

    def stop(self, model_instance_id: int, timeout: float):
        """Signal every thread of this generation and wait for it to finish.

        Args:
            model_instance_id: The model instance ID, for logging.
            timeout: Maximum time to wait for each thread (seconds).
        """
        self.stop_event.set()
        with self._lock:
            self._stopping = True
            stop_events = list(self._aux_stop_events)
            threads = [self.main_thread] + list(self._aux_threads)

        for stop_event in stop_events:
            stop_event.set()

        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(
                        f"Log persistence thread {thread.name} for model instance "
                        f"{model_instance_id} did not stop within {timeout}s"
                    )


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


class ServeManager:
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
    ):
        self._worker_id_getter = worker_id_getter
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._clientset_getter = clientset_getter

        self._provisioning_processes = {}
        self._log_persistence = {}
        self._log_persistence_lock = threading.Lock()
        self._error_model_instances = {}
        self._model_cache_by_instance = {}
        self._model_instance_by_instance_id = {}

        # Instance-level port tracking to avoid conflicts
        self._assigned_ports: Dict[int, Set[int]] = {}
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

        # Timestamp of the last authoritative (uncached) DB reconciliation in
        # the state sync, for the optional periodic backstop. Starts "now" so
        # the first forced reconciliation is one full period in, not immediate.
        self._last_state_reconcile_time: float = time.time()

        os.makedirs(self._serve_log_dir, exist_ok=True)

    def record_successful_inference(self, instance_id: int):
        """Called by worker proxy on successful inference response."""
        self._last_successful_inference[instance_id] = time.time()

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
                    callback=self._handle_model_instance_event
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
                for mi in list(self._error_model_instances.values()):
                    self._restart_error_model_instance(mi)
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

        # Snapshot local state BEFORE the list call. Reversing this order
        # races with CREATED events and reaps freshly-assigned instances.
        local_assigned_ids = {
            mid
            for mid, mi in self._model_instance_by_instance_id.items()
            if mi.get_deployment_metadata(self._worker_id) is not None
        }

        # Read from the watch-backed cache. It holds the full set, so the
        # common (nothing-to-reap) path stays O(1) with no server/DB round
        # trip and scales independently of worker count — a direct per-worker
        # poll here is one SELECT over model_instances per worker every few
        # seconds, which does not scale to hundreds of workers.
        response = self._clientset.model_instances.list()
        all_items = response.items or []
        reap_ids = local_assigned_ids - {
            mi.id
            for mi in all_items
            if mi.get_deployment_metadata(self._worker_id) is not None
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
                mi.id
                for mi in all_items
                if mi.get_deployment_metadata(self._worker_id) is not None
            }

        # An empty authoritative result is legitimate (user stopped every
        # model); list() raises on API failure rather than returning empty, so
        # local_assigned_ids - ∅ correctly reaps everything still tracked.
        for stale_id in reap_ids:
            stale = self._model_instance_by_instance_id.get(stale_id)
            if stale is None:
                continue
            logger.info(
                f"Reaping stale model instance {stale.name} (id={stale_id}); "
                f"server no longer reports it assigned to this worker."
            )
            try:
                # Equivalent to a dropped DELETED event, so purge logs too —
                # otherwise a reused id would inherit this instance's stale logs.
                self._stop_model_instance(stale, delete_logs=True)
            except Exception as e:
                logger.warning(f"Failed to reap stale model instance {stale.name}: {e}")

        if not all_items:
            # Nothing left to sync; reap pass above already handled stale
            # local state.
            return

        model_instances: List[ModelInstance] = []
        for model_instance in all_items:
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
        for model_instance in model_instances:
            # Skip if the provision process has not exited yet.
            if self._is_provisioning(model_instance):
                logger.trace(
                    f"Model instance {model_instance.name} is provisioning. Skipping sync."
                )
                continue

            is_main_worker = model_instance.worker_id == self._worker_id

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
                            sw.state_message = failure_message
                            patch_dict = {
                                f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                            }
                        # Update model instance.
                        self._update_model_instance(model_instance.id, **patch_dict)
                continue

            # The workload is alive and this worker did not necessarily start it,
            # so reconcile the log persistence threads that belong to it.
            self._ensure_container_log_persistence(model_instance)

            # Otherwise, update model instance state to RUNNING if everything is fine.
            model = self._get_model(model_instance)
            if model.gpu_type_selector:
                # vGPU: read back the real device allocation the operator
                # device plugin wrote onto the workload's annotations.
                self._sync_vgpu_allocation(model_instance, workload, is_main_worker)
            if not model.backend_version:
                # backend version may be empty on initialization.
                # try to refresh to get updated model info on syncs.
                model = self._refresh_model(model_instance)

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
                # NOTE: There is no known use case for now. Keep this in case the built-in backends
                # introduce breaking changes and the default health check path no longer works.
                health_check_path = model.env['GPUSTACK_MODEL_HEALTH_CHECK_PATH']

            with contextlib.suppress(NotFoundException):
                # Get patch dict for main worker.
                if is_main_worker:
                    subordinate_state = self._get_main_worker_distributed_state(
                        model_instance
                    )
                    if subordinate_state is None:
                        if model_instance.state == ModelInstanceStateEnum.RUNNING:
                            self._restart_backoff_counts.pop(model_instance.id, None)
                            continue

                        if (
                            model_instance.state == ModelInstanceStateEnum.ERROR
                            or not is_ready(
                                backend, model_instance, health_check_path, model
                            )
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
                    sw = model_instance.distributed_servers.subordinate_workers[sw_pos]
                    if sw.state == ModelInstanceStateEnum.RUNNING:
                        continue
                    if (
                        model_instance.distributed_servers.mode
                        == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                    ):
                        # Startup sets RUNNING directly in this mode. Only repair
                        # an unreachable subordinate with a surviving workload;
                        # the main worker still owns the engine readiness check.
                        if sw.state != ModelInstanceStateEnum.UNREACHABLE:
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
                self._update_model_instance(model_instance.id, **patch_dict)

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
                    f"Distributed serving error in subordinate worker "
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
                    self._update_model_instance(model_instance.id, **patch_dict)
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
                self._update_model_instance(
                    model_instance.id,
                    **{f"distributed_servers.subordinate_workers.{sw_pos}": sw},
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

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
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
                    logger.info(f"Provisioning model instance {mi.name}")
                    server_ins.start()
                    logger.info(f"Finished provisioning model instance {mi.name}")
                except Exception as e:
                    logger.exception(
                        f"Error provisioning model instance {mi.name}: {e}"
                    )
                    raise e

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
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT: seconds (default: 15)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD: count (default: 3)

        If the model has received successful inference traffic recently
        (within the configured interval), the active health check is skipped.
        """

        # Use the event-driven local cache instead of an API call.
        model_instances = [
            mi
            for mi in self._model_instance_by_instance_id.values()
            if mi.state == ModelInstanceStateEnum.RUNNING
        ]
        if not model_instances:
            return

        now = time.time()

        for model_instance in model_instances:
            model = self._get_model(model_instance)
            if not model:
                continue

            # Read per-model config from model.env.
            config = _get_inference_health_check_config(model)
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
                self._inference_health_check_failures[model_instance.id] = failure_count

                if failure_count >= threshold:
                    logger.warning(
                        f"Model instance {model_instance.name} inference health check failed "
                        f"{failure_count} times, updating state to ERROR."
                    )
                    patch_dict = {
                        "state": ModelInstanceStateEnum.ERROR,
                        "state_message": _INFERENCE_HEALTH_CHECK_FAILED_MESSAGE,
                    }
                    self._update_model_instance(model_instance.id, **patch_dict)
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

    def _dispatch_model_instance_event(self, event: Event):  # noqa: C901
        """
        Handle model instance events.

        Args:
            event: The model instance event to handle.

        """
        mi = ModelInstance.model_validate(event.data)

        logger.trace(
            f"Received event: {str(event.type)}, id: {mi.id}, name: {mi.name}, state: {str(mi.state)}"
        )

        is_main_worker = mi.worker_id == self._worker_id

        if is_main_worker:
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
            # Caching matched ERROR instances for restart handling.
            if mi.state == ModelInstanceStateEnum.ERROR:
                model = self._get_model(mi)
                if model.restart_on_error:
                    self._error_model_instances[mi.id] = mi
                    logger.trace(
                        f"UPDATED event: cached error model instance {mi.name} for restart."
                    )
                return

            # Restart if scheduled and this is the assigned worker.
            if is_main_worker and mi.state == ModelInstanceStateEnum.SCHEDULED:
                self._restart_model_instance(mi)
                logger.trace(
                    f"UPDATED event: restarted scheduled model instance {mi.name}."
                )

            # Start on subordinate worker if not started yet, or restart if failed.
            if not is_main_worker:
                deployment_metadata = mi.get_deployment_metadata(self._worker_id)
                workload_name = (
                    deployment_metadata.name if deployment_metadata else mi.name
                )
                workload = get_workload(workload_name)
                if not workload:
                    self._start_model_instance(mi)
                    logger.trace(
                        f"UPDATED event: started model instance {mi.name} on subordinate worker."
                    )
                elif workload.state in [
                    WorkloadStatusStateEnum.UNKNOWN,
                    WorkloadStatusStateEnum.UNHEALTHY,
                    WorkloadStatusStateEnum.INACTIVE,
                    WorkloadStatusStateEnum.FAILED,
                ]:
                    self._stop_model_instance(mi, clear_restart_backoff=False)
                    self._start_model_instance(mi)
                    logger.trace(
                        f"UPDATED event: restarted failed model instance {mi.name} on subordinate worker."
                    )

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

    def _get_numbered_log_path(self, mi: ModelInstance) -> str:
        """Get log file path with restart count.

        Args:
            mi: The model instance.

        Returns:
            Log file path with format: {log_dir}/{model_instance_id}.{restart_count}.log
        """
        restart_count = mi.restart_count or 0
        return f"{self._serve_log_dir}/{mi.id}.{restart_count}.log"

    def _persist_container_logs(  # noqa: C901
        self,
        workload_name: str,
        log_path: str,
        stop_event: threading.Event,
        token: Optional[str] = None,
        resume: bool = False,
    ):
        """Persist container logs to local file (runs in a separate thread).

        Reconnects on stream EOF while the workload is still alive, resuming by
        skipping already-written history (matched by an anchor window of the
        last lines written). A manual/runtime restart briefly looks terminated
        at EOF, so it waits a grace window for the container to return before
        giving up. Exits only if the container stays terminated for that whole
        window or the thread is asked to stop.

        Args:
            workload_name: Name of the container workload
            log_path: Path to save container logs
            stop_event: Event to signal thread to stop
            token: Operation token identifying a specific container in the workload.
                If None, logs from the default (index=0) container are fetched.
                resolve it, at the cost of a cluster-wide lookup per attempt.
            resume: Adopt a log file a previous worker process left behind,
                appending to it instead of rewriting it from the runtime's replay.
        """
        retry_count = 0
        first_connect = True
        # Anchor: a window of the last lines written. Matching a run of lines
        # (not one) avoids false-matching a repeated line during replay.
        anchor_window = deque(maxlen=5)

        # Only an anchor seeded from the file may be absent from the replay, so
        # only it needs a deadline. Cleared once it matches or is retired, and
        # each connection derives its deadline from it.
        anchor_is_seeded = False

        if resume:
            # Seed the anchor from the file's tail so this connection behaves
            # like a reconnect; an empty or unreadable file keeps first_connect.
            _drop_partial_last_line(log_path)
            anchor_window.extend(_tail_lines(log_path, anchor_window.maxlen))
            first_connect = not anchor_window
            anchor_is_seeded = not first_connect

        while not stop_event.is_set():
            try:
                log_stream = logs_workload(
                    name=workload_name,
                    token=token,
                    tail=-1,
                    follow=True,
                )

                if hasattr(log_stream, '__iter__'):
                    # On reconnect the runtime replays history from the start;
                    # skip it until the anchor window matches.
                    anchor = list(anchor_window)
                    skip_until_anchor = not first_connect and bool(anchor)
                    replayed = deque(maxlen=len(anchor)) if anchor else None
                    received_lines = False
                    # A followed stream never EOFs while the container lives, so
                    # an unreplayable anchor would hold live output back forever.
                    skip_deadline = (
                        time.monotonic() + _LOG_RESUME_SKIP_TIMEOUT
                        if anchor_is_seeded
                        else None
                    )
                    with open(
                        log_path,
                        'w' if first_connect else 'a',
                        buffering=1,
                        encoding='utf-8',
                    ) as f:
                        first_connect = False
                        for line in log_stream:
                            received_lines = True
                            if stop_event.is_set():
                                break

                            if isinstance(line, bytes):
                                line = line.decode('utf-8', errors='replace')
                            else:
                                line = str(line)

                            if skip_until_anchor:
                                replayed.append(line)
                                if list(replayed) == anchor:
                                    skip_until_anchor = False
                                    anchor_is_seeded = False
                                elif (
                                    skip_deadline is not None
                                    and time.monotonic() > skip_deadline
                                ):
                                    break
                                continue

                            f.write(line)
                            f.flush()
                            anchor_window.append(line)
                    retry_count = 0

                    # Anchor never matched -> rotated out; restart fresh. An
                    # empty reconnect must NOT reset, or the next round reopens
                    # in 'w' and truncates the saved log.
                    if skip_until_anchor and received_lines:
                        first_connect = True
                        anchor_window.clear()
                        # The rewrite rebuilds the anchor from the stream.
                        anchor_is_seeded = False

                # A restart briefly looks terminated at EOF; wait for the
                # container to return before giving up, so logs aren't dropped.
                if stop_event.is_set() or not self._wait_for_container_recovery(
                    workload_name, stop_event
                ):
                    break
                logger.debug(
                    f"Log stream for {workload_name} ended while workload still "
                    f"running; reconnecting"
                )
                stop_event.wait(timeout=1)

            except Exception as e:
                if stop_event.is_set():
                    break
                retry_count += 1
                logger.debug(
                    f"Container not ready for {workload_name}, retrying "
                    f"(attempt {retry_count}): {e}"
                )
                stop_event.wait(timeout=2)

        logger.debug(f"Log persistence thread for {workload_name} exiting")

    def _container_still_running(self, workload_name: str) -> bool:
        """Whether the workload is still alive (a dead stream should reconnect
        rather than exit)."""
        try:
            workload = get_workload(workload_name)
        except Exception:
            return True  # transient query failure: reconnect, don't drop logs
        return bool(workload) and workload.state in (
            WorkloadStatusStateEnum.PENDING,
            WorkloadStatusStateEnum.INITIALIZING,
            WorkloadStatusStateEnum.RUNNING,
        )

    def _wait_for_container_recovery(
        self,
        workload_name: str,
        stop_event: threading.Event,
        grace_seconds: float = LOG_RECONNECT_GRACE_SECONDS,
        poll_interval: float = 1.0,
    ) -> bool:
        """Poll until the workload is alive again (True -> reconnect) or the
        grace window elapses / stop_event fires (False -> give up). A restart
        momentarily looks terminated at EOF, which a single check can't tell
        apart from a real termination.
        """
        attempts = max(1, int(grace_seconds / poll_interval))
        for _ in range(attempts):
            if stop_event.is_set():
                return False
            if self._container_still_running(workload_name):
                return True
            stop_event.wait(timeout=poll_interval)
        return False

    def _discover_sidecar_logs(
        self,
        mi_id: int,
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
                if workload and workload.loggable:
                    sidecars = [op for op in workload.loggable if op.name != "default"]
                    if sidecars:
                        self._start_sidecar_log_threads(
                            mi_id,
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

            log_path = (
                f"{self._serve_log_dir}/{mi_id}.container."
                f"{op.name}.{restart_count}.log"
            )
            stop_event = threading.Event()

            thread = threading.Thread(
                target=self._persist_container_logs,
                args=(workload_name, log_path, stop_event, op.token),
                kwargs={"resume": resume},
                daemon=True,
                name=f"log-persist-{workload_name}-{op.name}",
            )
            thread.start()

            # Tracked on this generation's record, never on the shared registry.
            persistence.add_aux_thread(thread, stop_event)
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
        # Use deployment metadata name for the actual workload name,
        # which differs for subordinate workers (e.g., "model-f0").
        deployment_metadata = mi.get_deployment_metadata(self._worker_id)
        workload_name = deployment_metadata.name if deployment_metadata else mi.name

        restart_count = mi.restart_count or 0
        log_path = f"{self._serve_log_dir}/{mi.id}.container.{restart_count}.log"

        stop_event = threading.Event()

        # Main container log thread.
        thread = threading.Thread(
            target=self._persist_container_logs,
            args=(workload_name, log_path, stop_event),
            kwargs={"resume": resume},
            daemon=True,
            name=f"log-persist-{workload_name}",
        )
        persistence = _LogPersistence(stop_event, thread)

        # Sidecar discovery thread — polls until sidecar containers appear,
        # then starts additional log threads for each.
        discovery_thread = threading.Thread(
            target=self._discover_sidecar_logs,
            args=(mi.id, workload_name, restart_count, persistence),
            kwargs={"resume": resume},
            daemon=True,
            name=f"log-discover-{workload_name}",
        )
        persistence.add_aux_thread(discovery_thread)

        # Retire, register and start in one critical section: a concurrent start
        # would otherwise leave one generation with nothing left to signal it.
        with self._log_persistence_lock:
            self._retire_log_persistence(mi.id)
            self._log_persistence[mi.id] = persistence
            thread.start()
            discovery_thread.start()
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
        with self._log_persistence_lock:
            persistence = self._log_persistence.get(mi.id)
        if persistence and persistence.main_thread.is_alive():
            return

        self._align_legacy_main_log(mi)
        logger.info(
            f"Re-attaching container log persistence for adopted model instance "
            f"{mi.name}"
        )
        self._start_container_log_persistence(mi, resume=True)

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
            legacy_log.rename(numbered_log)
            logger.info(f"Renamed legacy serve log {legacy_log} to {numbered_log}")
        except Exception as e:
            logger.warning(f"Failed to align legacy serve log for {mi.name}: {e}")

    def _stop_container_log_persistence(
        self, model_instance_id: int, timeout: float = 2.0
    ):
        """Stop all container log persistence threads for a model instance.

        Args:
            model_instance_id: The model instance ID
            timeout: Maximum time to wait for each thread to stop (seconds)
        """
        with self._log_persistence_lock:
            self._retire_log_persistence(model_instance_id, timeout)

    def _retire_log_persistence(self, model_instance_id: int, timeout: float = 2.0):
        """Drop a model instance's log persistence generation and wait it out.

        The caller must hold ``_log_persistence_lock``; joining under it is safe
        because no log persistence thread ever takes it.

        Args:
            model_instance_id: The model instance ID
            timeout: Maximum time to wait for each thread to stop (seconds)
        """
        persistence = self._log_persistence.pop(model_instance_id, None)
        if persistence:
            persistence.stop(model_instance_id, timeout)

    def _cleanup_old_logs(self, model_instance_id: int, current_restart_count: int):
        """Keep serve logs for restart_count in {R, R-1}.

        R==0 is a fresh lifecycle start, so all existing files are purged (any
        present belong to a previous owner of a reused id).
        """
        if current_restart_count == 0:
            self._purge_instance_logs(model_instance_id)
            return

        try:
            log_dir = Path(self._serve_log_dir)

            # Separate main logs, container logs, and sidecar container logs
            main_log_pattern = f"{model_instance_id}.*.log"
            all_main_logs = [
                f for f in log_dir.glob(main_log_pattern) if '.container.' not in f.name
            ]

            # The glob cannot match {id}.log; it takes part as restart 0.
            legacy_log = existing_legacy_main_log(log_dir, model_instance_id)
            if legacy_log:
                all_main_logs.append(legacy_log)

            container_log_pattern = f"{model_instance_id}.container.*.log"
            all_container_files = list(log_dir.glob(container_log_pattern))

            # Split into default container logs (e.g., 42.container.0.log)
            # and sidecar container logs (e.g., 42.container.ray-head.0.log)
            default_container_logs = [
                f
                for f in all_container_files
                if extract_container_restart_count(f.name) > 0
                or re.match(rf'{model_instance_id}\.container\.\d+\.log', f.name)
            ]
            sidecar_container_logs = [
                f for f in all_container_files if f not in default_container_logs
            ]

            self._cleanup_log_type(all_main_logs, current_restart_count, "main")
            self._cleanup_log_type(
                default_container_logs, current_restart_count, "container"
            )
            self._cleanup_log_type(
                sidecar_container_logs, current_restart_count, "sidecar_container"
            )

        except Exception as e:
            logger.error(f"Failed to cleanup old logs for {model_instance_id}: {e}")

    def _cleanup_log_type(
        self,
        log_files: List[Path],
        current_restart_count: int,
        log_type: str,
    ):
        """Delete log files whose restart_count is not current or previous."""

        keep = {current_restart_count}
        if current_restart_count > 0:
            keep.add(current_restart_count - 1)

        def _extract_sidecar_restart_count(filename: str) -> int:
            """Extract restart count from {id}.container.{name}.{restart_count}.log"""
            match = re.match(r'\d+\.container\.[^.]+\.(\d+)\.log', filename)
            return int(match.group(1)) if match else 0

        extract_fns = {
            "main": extract_restart_count,
            "container": extract_container_restart_count,
            "sidecar_container": _extract_sidecar_restart_count,
        }
        extract_fn = extract_fns.get(log_type, extract_container_restart_count)

        for f in log_files:
            rc = extract_fn(f.name)
            if rc in keep:
                continue
            try:
                f.unlink()
                logger.info(f"Deleted old {log_type} log file: {f}")
            except Exception as e:
                logger.warning(f"Failed to delete {log_type} log file {f}: {e}")

    def _purge_instance_logs(self, model_instance_id: int):
        """Delete all serve logs (main/container/sidecar) for a model instance id."""
        try:
            log_dir = Path(self._serve_log_dir)
            files = list(log_dir.glob(f"{model_instance_id}.*.log"))

            # The glob cannot match {id}.log, which a reused id would inherit.
            legacy_log = existing_legacy_main_log(log_dir, model_instance_id)
            if legacy_log:
                files.append(legacy_log)

            for f in files:
                try:
                    f.unlink()
                    logger.info(f"Deleted serve log file: {f}")
                except Exception as e:
                    logger.warning(f"Failed to delete serve log file {f}: {e}")
        except Exception as e:
            logger.error(
                f"Failed to purge logs for model instance {model_instance_id}: {e}"
            )

    def _start_model_instance(self, mi: ModelInstance):  # noqa: C901
        """
        Start model instance through a subprocess.

        Args:
            mi: The model instance to start.

        """
        if self._is_provisioning(mi):
            logger.warning(f"Model instance {mi.name} is provisioning. Skipping start.")
            return

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

        try:
            model = self._get_model(mi)
            backend = get_backend(model)

            self._assign_ports(mi, model, backend)

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
                ),
            )
            process.daemon = False
            process.start()
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

            self._update_model_instance(mi.id, **patch_dict)
            logger.info(
                f"Started model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

        except Exception as e:
            # Clean up provisioning process if started.
            if mi.id in self._provisioning_processes:
                self._stop_model_instance(mi)

            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.ERROR,
                    "state_message": f"Failed to start model instance: {e}",
                }
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.ERROR
                sw.state_message = f"Failed to start model instance: {e}"
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.error(f"Failed to start model instance {mi.name}: {e}")

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
        with _port_lock:
            if mi.port:
                # Ports already assigned (a restart reusing what was
                # persisted), so nothing to allocate — but they still have to
                # be re-registered. This process's view of what is taken lives
                # only in `_assigned_ports`, so returning without refilling it
                # leaves the whole band looking free to the next instance
                # started here. Harmless while an instance held one port;
                # with a band of `1 + Σcount` it hands the same ports out
                # twice after a worker restart.
                self._register_assigned_ports(mi)
                return

            if self._assigned_ports:
                unavailable_ports = set.union(*self._assigned_ports.values())
            else:
                unavailable_ports = set()

            # Main serving port
            mi.port = network.get_free_port(
                port_range=self._config.service_port_range,
                unavailable_ports=unavailable_ports,
                host=mi.worker_ip,
            )
            mi.ports = [mi.port]
            unavailable_ports.add(mi.port)

            # Additional ports for distributed servers (mp path allocates all):
            #   ports[0]: HTTP API (always)
            #   ports[1]: --data-parallel-rpc-port (DP coordinator ZMQ)
            #   ports[2]: --master-port (PyTorch distributed TCP store)
            #   ports[3]: env VLLM_PORT (dp_only only; reserved but unused otherwise)
            #   ports[4:-1]: named bands, then the connecting band's derived
            #                ports — fenced, addressed by nobody
            #   ports[-1]: connecting port (= VLLM_DP_MASTER_PORT for dp_only/nested)
            # Ray path: only ports[1] (DP RPC), when user dp > 1.
            connecting_port: Optional[int] = None
            connecting_band: List[int] = []
            if mi.distributed_servers and mi.distributed_servers.subordinate_workers:
                executor_backend = (
                    resolve_executor_backend(
                        model.backend_parameters, model.backend_version
                    )
                    if backend == BackendEnum.VLLM
                    else None
                )
                # The connecting port doubles as VLLM_DP_MASTER_PORT, and on the
                # mp path vLLM derives nine more ports from it, so what has to
                # be free is a run of ten — not one port with nine unprobed
                # neighbours, which is all a `get_free_port` here can promise.
                # Allocated first so the cross ports (incl. VLLM_PORT)
                # land outside the band rather than inside it.
                band_count = _VLLM_MP_CONNECTING_BAND if executor_backend == "mp" else 1
                connecting_port = network.get_free_band(
                    port_range=self._config.service_port_range,
                    count=band_count,
                    unavailable_ports=unavailable_ports,
                    host=mi.worker_ip,
                )
                connecting_band = list(
                    range(connecting_port, connecting_port + band_count)
                )
                unavailable_ports |= set(connecting_band)

                cross_ports: List[int] = []
                if backend == BackendEnum.VLLM:
                    if executor_backend == "mp":
                        # DP RPC + PyTorch master + VLLM_PORT.
                        cross_port_count = 3
                    else:
                        dps = find_int_parameter(
                            model.backend_parameters,
                            ["data-parallel-size", "dp"],
                        )
                        cross_port_count = 1 if dps and dps > 1 else 0
                    for _ in range(cross_port_count):
                        cross_port = network.get_free_port(
                            port_range=self._config.service_port_range,
                            unavailable_ports=unavailable_ports,
                            host=mi.worker_ip,
                        )
                        cross_ports.append(cross_port)
                        unavailable_ports.add(cross_port)

                mi.ports.extend(cross_ports)

            # Named connector bands, appended to `mi.ports` as well as written
            # to `mi.named_ports`. `named_ports` is a *new* index, not a
            # migration: the positional convention above stays exactly as it
            # is. The append matters on its own — under hostNetwork the
            # runtime declares every entry of `mi.ports` as a hostPort
            # (`_get_configured_ports()` never looks at `named_ports`), and
            # that declaration is the only thing that turns a same-host port
            # collision into a schedulable-Pending with an event instead of a
            # container crash-looping forever in `starting`.
            named_band_ports = self._assign_named_ports(mi, model, unavailable_ports)
            mi.ports.extend(named_band_ports)

            if connecting_port is not None:
                # Only the base goes into `mi.ports`, and it goes last: the
                # distributed backends read the connecting port as `ports[-1]`
                # (VLLM_DP_MASTER_PORT / VLLM_PORT), so nothing may be appended
                # after it.
                mi.ports.append(connecting_port)
                if len(connecting_band) > 1:
                    # The nine ports vLLM derives from the base are recorded in
                    # `named_ports` rather than in `mi.ports`, and the
                    # distinction is deliberate. Both indexes persist and both
                    # are re-fenced on a worker restart, so either one fixes
                    # what was broken here: ten ports that were bound but
                    # neither probed nor registered. Only `mi.ports` is turned
                    # into host ports by the runtime — so putting them there
                    # would also rewrite the container spec of every existing
                    # multi-worker vLLM deployment, which is a change to
                    # non-disaggregated behaviour and belongs to whoever
                    # decides to make it, not to this one. What that costs:
                    # a collision with a process outside this worker still
                    # surfaces as a crash loop rather than as an unschedulable
                    # Pod. Within one worker the fence now prevents it, which
                    # is where two members of one deployment actually collide.
                    named_ports = dict(mi.named_ports or {})
                    named_ports[_VLLM_MP_CONNECTING_BAND_NAME] = PortBand(
                        base=connecting_port, count=len(connecting_band)
                    )
                    mi.named_ports = named_ports

            self._assigned_ports[mi.id] = set(mi.ports)
            for band in (mi.named_ports or {}).values():
                self._assigned_ports[mi.id] |= set(
                    range(band.base, band.base + max(band.count, 1))
                )

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
            self._update_model_instance(
                mi.id,
                state=ModelInstanceStateEnum.ERROR,
                state_message=message,
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
        Callers must hold `_port_lock`.
        """
        taken: Set[int] = set(mi.ports or [])
        if mi.port:
            taken.add(mi.port)
        for band in (mi.named_ports or {}).values():
            taken |= set(range(band.base, band.base + max(band.count, 1)))
        if taken:
            self._assigned_ports[mi.id] = taken

    def _assign_named_ports(
        self,
        mi: ModelInstance,
        model: Model,
        unavailable_ports: Set[int],
    ) -> List[int]:
        """Allocate the port bands this instance's PD role declares.

        The widths come from `pd-modes.yaml` and nothing else: measured, one
        connector wants one port per data-parallel replica and another wants
        one per tensor-parallel rank, so there is no platform-side formula to
        compute them from. Writes `mi.named_ports` and returns every port of
        every band, in order, for the caller to append to `mi.ports`.

        Mutates `unavailable_ports` with the *whole* band, not just its base:
        the ports a connector derives from the base are bound just as surely
        as the base is, and the existing vLLM mp path fencing off ten ports
        around the connecting port is the same idea with the width hardcoded.

        Every band is allocated per instance, which is what
        `PDPortScopeEnum.INSTANCE` means and what every band shipped today
        declares. A `role`-scoped band would need a registry shared across the
        instances of one role, which does not exist yet; nothing declares one,
        so allocating it per instance is strictly safer than pretending.

        Callers must hold `_port_lock`.
        """
        mode_name = getattr(getattr(model, "disaggregation", None), "mode", None)
        specs = band_specs_for(model, mi.role, context=f"Model instance {mi.name}")
        if not specs:
            # Not a PD instance, or a role that declares no band. Either way,
            # today's path byte for byte.
            return []

        named_ports: Dict[str, PortBand] = dict(mi.named_ports or {})
        band_ports: List[int] = []
        for spec in specs:
            if spec.scope == PDPortScopeEnum.ROLE:
                # Not implemented, and therefore refused. See
                # PDPortScopeUnsupportedError: per-instance allocation of a
                # role-scoped band is not a partial implementation of it, it is
                # a different band on every member.
                raise PDPortScopeUnsupportedError(
                    f"Model instance {mi.name} (role '{mi.role}', PD mode "
                    f"'{mode_name}') declares port band '{spec.name}' with "
                    f"scope '{PDPortScopeEnum.ROLE.value}'. Role-scoped bands "
                    "need a registry shared by the members of one role, which "
                    "does not exist yet, and allocating one per instance would "
                    "give each member a different base for a band they are "
                    "supposed to meet on."
                )
            count = band_width(
                spec,
                cards=len(mi.gpu_indexes or []),
                backend_parameters=model.backend_parameters,
            )
            if count is None:
                # Skipping the band leaves `{{ports.<name>}}` unresolved in the
                # launch, which the renderer logs and the engine rejects by
                # name; allocating a guessed width would instead look like it
                # worked. Which of the two reasons it was decides where the
                # operator looks next, so it is named.
                why = (
                    "has no accelerators assigned to size it from"
                    if band_count_key(spec) == ACCELERATOR_COUNT_KEY
                    else "declares no parallelism parameter answering to that name"
                )
                logger.warning(
                    f"Model instance {mi.name} (role '{mi.role}') declares port "
                    f"band '{spec.name}' with count '{spec.count}' but {why}. "
                    "No ports are reserved for it, so the placeholder reaches "
                    "the engine verbatim and the launch fails naming it."
                )
                continue
            try:
                base = network.get_free_band(
                    port_range=self._config.service_port_range,
                    count=count,
                    unavailable_ports=unavailable_ports,
                    host=mi.worker_ip,
                )
            except network.PortRangeExhaustedError as e:
                # Re-raise with the role and band named. The allocator knows
                # the arithmetic but not who was asking, and "which role of
                # which deployment" is the first thing an operator needs.
                raise network.PortRangeExhaustedError(
                    f"Model instance {mi.name} (role '{mi.role}', PD mode "
                    f"'{mode_name}') needs a {count}-port band for "
                    f"'{spec.name}': {e}"
                ) from e
            band = list(range(base, base + count))
            unavailable_ports |= set(band)
            named_ports[spec.name] = PortBand(base=base, count=count)
            band_ports.extend(band)

        mi.named_ports = named_ports
        return band_ports

    def _restart_model_instance(self, mi: ModelInstance):
        """
        Restart model instance.

        Args:
            mi: The model instance to restart.
        """

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

    def _update_model_instance(self, id: int, **kwargs):
        """
        Update model instance with given fields.

        Args:
            id: The ID of the model instance to update.
            **kwargs: The fields to update, group by field name and value.
        """

        try:
            mi_public = self._clientset.model_instances.get(id=id)

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

        logger.debug(f"Stopping model instance {mi.name or mi.id}")

        # Stop container log persistence thread
        self._stop_container_log_persistence(mi.id)

        if delete_logs:
            self._purge_instance_logs(mi.id)

        # Teardown provisioning process if still alive.
        if self._is_provisioning(mi):
            terminate_process_tree(self._provisioning_processes[mi.id].pid)

        # Delete workload.
        deployment_metadata = mi.get_deployment_metadata(self._worker_id)
        if deployment_metadata:
            delete_workload(deployment_metadata.name)

        # Cleanup internal states.
        self._provisioning_processes.pop(mi.id, None)
        self._assigned_ports.pop(mi.id, None)
        self._error_model_instances.pop(mi.id, None)
        self._model_cache_by_instance.pop(mi.id, None)
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
        self._inference_health_check_failures.pop(mi.id, None)
        self._last_health_check_time.pop(mi.id, None)
        self._last_successful_inference.pop(mi.id, None)

        logger.info(f"Stopped model instance {mi.name or mi.id}")

    def _restart_error_model_instance(self, mi: ModelInstance):
        """
        Restart error model instance with exponential backoff,
        maximum delay 5 minutes.

        Args:
            mi: The model instance to restart.
        """
        if self._is_provisioning(mi):
            logger.debug(f"Model instance {mi.name} is provisioning. Skipping restart.")
            return

        restart_count = mi.restart_count or 0
        backoff_count = self._restart_backoff_counts.get(mi.id, 0)
        last_restart_time = mi.last_restart_time or mi.updated_at

        current_time = datetime.now(timezone.utc)
        delay = min(10 * (2 ** (backoff_count - 1)), 300) if backoff_count > 0 else 0
        if backoff_count > 0 and last_restart_time:
            elapsed_time = (current_time - last_restart_time).total_seconds()
            if elapsed_time < delay:
                logger.trace(
                    f"Delaying restart of {mi.name} for {delay - elapsed_time:.2f} seconds."
                )
                return

        logger.info(
            f"Restarting model instance {mi.name} "
            f"(attempt {backoff_count + 1}) after {delay} seconds delay."
        )

        with contextlib.suppress(NotFoundException):
            self._restart_backoff_counts[mi.id] = backoff_count + 1
            self._update_model_instance(
                mi.id,
                restart_count=restart_count + 1,
                last_restart_time=current_time,
                state=ModelInstanceStateEnum.SCHEDULED,
                state_message="",
            )

        # Pop from error model instances,
        # if failed to restart next time, it will be added again in watch_model_instance_events().
        self._error_model_instances.pop(mi.id, None)

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
        refreshed_model = self._clientset.models.get(mi.model_id)
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
        """
        Get the port of the model instance related to the given model instance ID.

        Args:
            model_instance_id: The model instance ID to get the port for.

        Returns:
            The port of the model instance if it exists and is running, else None.
        """
        instance = self._model_instance_by_instance_id.get(
            model_instance_id
        )  # Ensure the model instance is cached.
        return (
            instance.ports[0]
            if instance and instance.state == ModelInstanceStateEnum.RUNNING
            else None
        )


def is_ready(
    backend: str,
    mi: ModelInstance,
    health_check_path: Optional[str] = None,
    model: Model = None,
    timeout: int = 1,
) -> bool:
    """
    Access the health endpoint of the given model instance to check if it is servable.

    ``timeout`` defaults to the second that suits the caller this was written
    for: a member on its way into RUNNING, polled every pass, where a slow
    answer costs nothing but another pass. A caller that turns a failure into
    ERROR must pass its own -- under load a busy engine answers late, and one
    second of patience would retire a member that is merely working.
    """
    is_built_in = is_built_in_backend(backend)
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
    ):
        if not model.backend_version:
            # version may be empty at initialization, consider it not ready.
            return False
        elif compare_versions(model.backend_version, "0.5.5.post3") >= 0:
            # SGLang Diffusion supported health check path at v0.5.5.post3
            health_check_path = "/health"
        else:
            # Older versions do not support health check, consider it always ready.
            return True
    elif is_built_in and backend != BackendEnum.CUSTOM and not health_check_path:
        # Built-in backends (vLLM, SGLang, vox-box) except (Custom, MindIE) use /v1/models as health check path.
        health_check_path = "/v1/models"

    try:
        # Use the worker IP instead of localhost for health check.
        # Reasons:
        # 1. Connectivity to the loopback address does not work with Ascend MindIE.
        # 2. More adaptable to container networks.
        health_check_url = f"http://{mi.worker_ip}:{mi.port}{health_check_path}"
        response = requests.get(health_check_url, timeout=timeout)
        if response.status_code == 200:
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


def _get_inference_health_check_config(model: Model) -> dict:
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
    timeout = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT"),
        15,
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


def is_inference_ready(mi: ModelInstance, model: Model, timeout: int = 15) -> bool:
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
        response = requests.post(inference_url, json=payload, timeout=timeout)
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
