import asyncio
import logging
import shlex
import socket
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import httpx

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    ContainerResources,
    WorkloadPlan,
    WorkloadStatusStateEnum,
    create_workload,
    delete_workload,
    get_workload,
)
from gpustack_runtime.detector import detect_backend, detect_devices
from gpustack_runtime.envs import (
    GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY,
    to_bool,
)

from gpustack import envs
from gpustack.api.exceptions import NotFoundException
from gpustack.client import ClientSet
from gpustack.config import registration
from gpustack.config.config import Config
from gpustack.schemas.cache_providers import (
    CUSTOM_VERSION,
    CacheProvider,
    CacheProviderHealthCheck,
    CacheProviderVersionConfig,
    render_l2_adapter,
    render_template,
)
from gpustack.schemas.cache_services import (
    CacheServiceInstance,
    CacheServiceInstanceUpdate,
    CacheServicePublic,
    CacheServiceStateEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.cache_provider_catalog import get_cache_provider
from gpustack.utils import network
from gpustack.utils.attrs import set_attr
from gpustack.utils.command import (
    drop_empty_flag_values,
    extract_flag_arguments,
    merge_flag_arguments,
)
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.utils.runtime import transform_workload_plan

logger = logging.getLogger(__name__)

HEALTH_PROBE_TIMEOUT_SECONDS = 2

MAX_CONSECUTIVE_RESTARTS = 5
"""Consecutive crashes tolerated before the instance is parked in ERROR."""

RESTART_BACKOFF_BASE_SECONDS = 30
RESTART_BACKOFF_MAX_SECONDS = 300

RESTART_COUNT_RESET_SECONDS = 600
"""How long an instance must stay healthy after its last restart before the
consecutive-restart budget is cleared."""

PENDING_START_GRACE_SECONDS = 60
"""How long an instance may stay PENDING before the sync pass re-drives its
start. The start is normally driven by the instance's PENDING event; this
window bounds how long a start that never took effect — a missed event, a
worker restart mid-start, a state write-back that did not reach the server —
keeps the instance stuck."""


class CacheServiceManager:
    """
    Runs managed cache service instances on this worker: launches the
    provider's container for each instance scheduled here, and keeps the
    instance's state in sync with the workload and its health probe.
    Rendering inputs (provider, version, config) come from the instance's
    parent cache service.
    """

    _port_lock = threading.Lock()
    _start_lock = threading.Lock()

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

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    """
    The clientset to access the API server.
    """

    _starting: Set[int]
    """
    IDs of instances whose start is in flight in this process, so the sync
    pass does not re-drive a start that is still running (pulling the
    provider image can take minutes). Guarded by _start_lock.
    """

    _last_start_attempt: Dict[int, datetime]
    """
    When this process last started each instance, so an instance that stays
    PENDING because its start keeps failing to take effect is retried on the
    PENDING_START_GRACE_SECONDS cadence rather than on every sync pass.
    Guarded by _start_lock.
    """

    _assigned_ports: Dict[int, Tuple[int, int]]
    """
    (port, metrics_port) pairs allocated in this process, keyed by cache
    service instance ID. Guarded by _port_lock so concurrent starts can't
    hand out the same port.
    """

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
    ):
        self._worker_id_getter = worker_id_getter
        self._clientset_getter = clientset_getter
        self._config = cfg

        self._assigned_ports = {}
        self._starting = set()
        self._last_start_attempt = {}

    async def watch_cache_service_instances_event(self):
        """
        Loop to watch cache service instances' event and handle.
        """
        logger.info("Watching cache service instances event.")
        while True:
            try:
                await self._clientset.cache_service_instances.awatch(
                    callback=self._handle_cache_service_instance_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching cache service instances: {e}")
                await asyncio.sleep(5)

    def _handle_cache_service_instance_event(self, event: Event):
        """
        Handle cache service instance events.

        Args:
            event: The cache service instance event to handle.
        """
        instance = CacheServiceInstance.model_validate(event.data)
        if instance.worker_id != self._worker_id:
            return

        logger.trace(
            f"Received event: {str(event.type)}, instance id: {instance.id}, "
            f"cache service id: {instance.cache_service_id}, "
            f"state: {str(instance.state)}"
        )

        if event.type == EventType.DELETED:
            self._stop_cache_service_instance(instance)
            return

        if instance.state == CacheServiceStateEnum.PENDING:
            self._schedule_start(instance)

    def _schedule_start(self, instance: CacheServiceInstance):
        """
        Run the blocking workload creation off the watch event loop.
        Without a running loop (direct invocation), run inline. A start
        already in flight for this instance is not started a second time.
        """
        if not self._claim_start(instance.id):
            logger.debug(
                f"Skipped starting cache service instance {instance.id}: "
                "a start is already in flight"
            )
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._start_cache_service_instance(instance)
            return
        loop.run_in_executor(None, self._start_cache_service_instance, instance)

    def _claim_start(self, instance_id: int) -> bool:
        """Mark a start as in flight; False when one already is."""
        with CacheServiceManager._start_lock:
            if instance_id in self._starting:
                return False
            self._starting.add(instance_id)
            self._last_start_attempt[instance_id] = datetime.now(timezone.utc)
            return True

    def _release_start(self, instance_id: int):
        with CacheServiceManager._start_lock:
            self._starting.discard(instance_id)

    def _forget_start(self, instance_id: int):
        with CacheServiceManager._start_lock:
            self._starting.discard(instance_id)
            self._last_start_attempt.pop(instance_id, None)

    def _start_cache_service_instance(self, instance: CacheServiceInstance):
        """
        Start the managed cache server container for a cache service
        instance.

        Args:
            instance: The cache service instance to start.
        """
        try:
            try:
                cache_service = self._clientset.cache_services.get(
                    id=instance.cache_service_id
                )
            except NotFoundException:
                self._update_cache_service_instance(
                    instance.id,
                    state=CacheServiceStateEnum.ERROR,
                    state_message=(
                        f"Parent cache service {instance.cache_service_id} "
                        "not found."
                    ),
                )
                return

            provider = get_cache_provider(cache_service.provider_name)
            if provider is None:
                self._update_cache_service_instance(
                    instance.id,
                    state=CacheServiceStateEnum.ERROR,
                    state_message=f"Unknown cache provider: {cache_service.provider_name}",
                )
                return

            version_config, resolved_version, source_image = (
                self._resolve_version_config(cache_service, provider)
            )

            # Starting is idempotent: a stale workload left over from a
            # previous run of this instance (crash, manual restart) is removed
            # first, so restart and first start share this code path.
            deployment_metadata = instance.get_deployment_metadata()
            try:
                delete_workload(deployment_metadata.name)
            except Exception as e:
                # The workload may not exist yet.
                logger.debug(
                    f"Skipped deleting workload {deployment_metadata.name} "
                    f"before start: {e}"
                )
            self._release_ports(instance.id)

            port, metrics_port = self._allocate_ports(instance)
            params = self._build_template_params(
                cache_service, provider, port, metrics_port
            )

            # A declared run command is the whole argument vector and takes
            # the image's ENTRYPOINT slot; run_args instead keeps the
            # image's own entrypoint and rides as the CMD arguments
            # appended to it (container semantics: args alone append, a
            # command replaces). The user parameters and L2 flags below
            # join whichever vector the version declared.
            overrides_entrypoint = bool(version_config.run_command)
            launch_template = version_config.run_command or version_config.run_args
            argv: Optional[List[str]] = None
            if launch_template:
                # Render per token so an optional placeholder resolving to
                # None yields an empty token that is dropped together with
                # the flag it belongs to.
                rendered_tokens = [
                    render_template(token, params)
                    for token in shlex.split(launch_template)
                ]
                argv = drop_empty_flag_values(rendered_tokens)
            user_parameters = (
                cache_service.config.parameters if cache_service.config else None
            )
            if user_parameters:
                argv = (
                    merge_flag_arguments(argv, user_parameters)
                    if argv
                    else list(user_parameters)
                )

            # L2 storage config renders after the user-parameters merge so
            # the structured config always wins over a hand-written flag.
            argv, l2_env = self._apply_l2_storage(cache_service, provider, argv)

            # Provider env templates render first; entries rendering empty are
            # dropped so unset optional parameters don't produce invalid
            # config. Service-level env overrides provider defaults, and the
            # L2 storage credentials override both.
            env: Dict[str, str] = {}
            for key, value in (version_config.env or {}).items():
                rendered = render_template(value, params)
                if rendered:
                    env[key] = rendered
            if cache_service.config and cache_service.config.env:
                env.update(cache_service.config.env)
            env.update(l2_env)

            fallback_registry = registration.determine_default_registry(
                self._config.system_default_container_registry
            )
            image = apply_registry_override_to_image(
                self._config, source_image, fallback_registry
            )
            if not image:
                raise ValueError(
                    f"Failed to resolve image for cache provider "
                    f"{cache_service.provider_name} version {resolved_version}"
                )

            run_container = Container(
                image=image,
                name="default",
                profile=ContainerProfileEnum.RUN,
                execution=ContainerExecution(
                    privileged=False,
                    command=argv if overrides_entrypoint else None,
                    args=None if overrides_entrypoint else argv,
                ),
                envs=[
                    ContainerEnv(name=name, value=value) for name, value in env.items()
                ],
                resources=self._gpu_resources(),
            )
            workload_plan = WorkloadPlan(
                name=deployment_metadata.name,
                host_network=True,
                # Shares the host IPC namespace with the engine containers
                # so the cache server can import their KV buffers by CUDA
                # IPC handle (the lmcache_driven zero-copy path). Same
                # escape hatch as the engine side: service env, then the
                # worker-global GPUSTACK_HOST_IPC, overrides the default —
                # e.g. Kubernetes PodSecurity baseline rejects hostIPC
                # pods, and the CPU host-copy path works without it.
                host_ipc=self._host_ipc_enabled(cache_service),
                containers=[run_container],
                labels=deployment_metadata.labels,
            )
            logger.info(
                f"Creating cache service workload {deployment_metadata.name} "
                f"with image {image} on port {port}"
            )
            create_workload(
                transform_workload_plan(self._config, workload_plan, fallback_registry)
            )

            if self._update_cache_service_instance(
                instance.id,
                state=CacheServiceStateEnum.STARTING,
                port=port,
                metrics_port=metrics_port,
                state_message="",
            ):
                logger.info(
                    f"Started cache service {cache_service.name} instance "
                    f"(id={instance.id}) on port {port}"
                )
            else:
                # The container is up but the server still sees the instance
                # as PENDING; the sync pass re-drives the start rather than
                # leaving a running cache server nothing points at.
                logger.error(
                    f"Started cache service workload {deployment_metadata.name} "
                    f"but failed to mark instance {instance.id} as starting"
                )
        except Exception as e:
            self._release_ports(instance.id)
            self._update_cache_service_instance(
                instance.id,
                state=CacheServiceStateEnum.ERROR,
                state_message=str(e),
            )
            logger.error(
                f"Failed to start cache service instance {instance.id} "
                f"(service id={instance.cache_service_id}): {e}"
            )
        finally:
            self._release_start(instance.id)

    def _gpu_resources(self) -> ContainerResources:
        """
        Expose every local GPU to the cache server so the CUDA-IPC transfer
        path (LMCache's lmcache_driven mode) can map the KV buffers of the
        co-located engines: importing an IPC handle needs a CUDA context on
        the same device, and a per-node server attaches to engines on any of
        the node's GPUs. Empty on a worker with no detected accelerator, so
        the server stays CPU-only there (auto mode falls back to a host-copy
        transfer).
        """
        resources = ContainerResources()
        backend = detect_backend()
        if isinstance(backend, str) and backend:
            key = GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY.get(backend)
            if key:
                resources[key] = "all"
        return resources

    def _resolve_version_config(
        self,
        cache_service: CacheServicePublic,
        provider: CacheProvider,
    ) -> Tuple[CacheProviderVersionConfig, Optional[str], str]:
        """
        Resolve the (version config, version identifier, container image)
        the instance runs with. The reserved "custom" version keeps the
        provider's run command and env templates but takes the image from
        the service config, so the image must be command-compatible with
        that declaration. Raises ValueError when the catalog or the service
        config cannot serve the request.
        """
        if cache_service.provider_version == CUSTOM_VERSION:
            if not provider.custom_version:
                raise ValueError(
                    f"Cache provider {cache_service.provider_name} does not "
                    f"allow the custom version"
                )
            version_config = provider.custom_version_config()
            if version_config is None:
                raise ValueError(
                    f"Cache provider {cache_service.provider_name} has no "
                    f"default version to template the custom version"
                )
            image = cache_service.config.image if cache_service.config else None
            if not image:
                raise ValueError(
                    f"config.image is required when provider_version is "
                    f"'{CUSTOM_VERSION}'"
                )
            return version_config, CUSTOM_VERSION, image

        version_config, resolved_version = provider.get_version_config(
            cache_service.provider_version
        )
        if version_config is None:
            raise ValueError(
                f"Unknown version '{resolved_version}' for cache provider "
                f"{cache_service.provider_name}"
            )
        backend, runtime_version = self._detect_runtime()
        # Fail fast on an unsupported accelerator: falling back to the
        # plain image (built for another accelerator family) would only
        # crash-loop the container without ever naming the real cause.
        if not version_config.supports_runtime(backend):
            raise ValueError(
                f"Cache provider {cache_service.provider_name} "
                f"({resolved_version}) has no image for {backend} workers; "
                f"scope the service to supported workers via the worker "
                f"selector"
            )
        return (
            version_config,
            resolved_version,
            version_config.resolve_image(backend, runtime_version),
        )

    def _detect_runtime(self) -> Tuple[Optional[str], Optional[str]]:
        """
        This node's (accelerator backend, runtime version), e.g.
        ("cuda", "13.0") — the key into a version's runtime_images.
        (None, None) on accelerator-less workers, where the plain image
        serves.
        """
        backend = detect_backend()
        if not (isinstance(backend, str) and backend):
            return None, None
        version = None
        try:
            version = next(
                (
                    device.runtime_version
                    for device in detect_devices()
                    if device.runtime_version
                ),
                None,
            )
        except Exception as e:
            logger.warning(f"Failed to detect accelerator runtime version: {e}")
        return backend, version

    @staticmethod
    def _host_ipc_enabled(cache_service: CacheServicePublic) -> bool:
        """Host IPC defaults on for cache servers (the CUDA-IPC transfer
        path needs it) but stays overridable: the service's env, then the
        worker-global GPUSTACK_HOST_IPC, wins over the default."""
        service_env = (cache_service.config.env if cache_service.config else None) or {}
        if envs.HOST_IPC_ENV in service_env:
            return to_bool(service_env[envs.HOST_IPC_ENV])
        if envs.HOST_IPC is not None:
            return to_bool(envs.HOST_IPC)
        return True

    @staticmethod
    def _build_template_params(
        cache_service: CacheServicePublic,
        provider: CacheProvider,
        port: int,
        metrics_port: int,
    ) -> Dict[str, Any]:
        """
        Build the placeholder namespace the version templates render
        against: the reserved platform keys, extended by the provider's
        declared fields carrying configured values (falling back to
        declared defaults). The platform keys win over a same-named field.
        """
        params: Dict[str, Any] = {
            "host": "0.0.0.0",
            "port": port,
            "metrics_port": metrics_port,
            "ram_size": (
                cache_service.config.ram_size if cache_service.config else None
            ),
            "chunk_size": (
                cache_service.config.chunk_size if cache_service.config else None
            ),
        }
        field_values = (
            cache_service.config.fields if cache_service.config else None
        ) or {}
        for field in provider.managed_fields:
            value = field_values.get(field.name, field.default)
            if isinstance(value, bool):
                value = str(value).lower()
            params.setdefault(field.name, value)
        return params

    def _apply_l2_storage(
        self,
        cache_service: CacheServicePublic,
        provider: CacheProvider,
        argv: Optional[List[str]],
    ) -> Tuple[Optional[List[str]], Dict[str, str]]:
        """
        Attach the service's L2 storage config to the cache server argument
        vector: each entry renders as one occurrence of the provider-declared
        flag carrying its adapter JSON, appended in declared order — the cache
        server prefers the earliest tier for reads and writes to all of them.
        A version running the image's own entrypoint has no vector of its
        own; the flags become its arguments.
        Secret-bearing fields go to the returned env; because env vars are
        process-global, two entries delivering a value through the same env
        var cannot coexist. Hand-written occurrences of the flag in the
        user parameters stay usable as an escape hatch for adapter types
        the declaration doesn't cover: they re-append after the structured
        entries, so the UI-visible order keeps the higher read priority.
        Raises ValueError when the provider can't serve the config.
        """
        l2_storages = cache_service.config.l2_storages if cache_service.config else None
        if not l2_storages:
            return argv, {}

        l2_args: List[str] = []
        l2_env: Dict[str, str] = {}
        env_sources: Dict[str, str] = {}
        for l2_storage in l2_storages:
            entry_args, entry_env = render_l2_adapter(
                provider,
                l2_storage.backend,
                l2_storage.params or {},
                l2_storage.adapter_flag_enabled,
            )
            for env_name, value in entry_env.items():
                if env_name in env_sources:
                    raise ValueError(
                        f"L2 storage entries '{env_sources[env_name]}' and "
                        f"'{l2_storage.backend}' both deliver the env var "
                        f"'{env_name}'; only one entry may set it"
                    )
                env_sources[env_name] = l2_storage.backend
                l2_env[env_name] = value
            l2_args.extend(entry_args)

        remaining, hand_written = extract_flag_arguments(
            argv or [], provider.l2_adapter_flag
        )
        if hand_written:
            logger.info(
                f"Cache service {cache_service.name}"
                f"(id={cache_service.id}) also passes "
                f"{provider.l2_adapter_flag} via parameters; appending those "
                f"adapters after the structured entries"
            )
        return remaining + l2_args + hand_written, l2_env

    def _allocate_ports(self, instance: CacheServiceInstance) -> Tuple[int, int]:
        """
        Allocate the instance's (port, metrics_port) pair on this worker.

        Ports already handed out by this process and ports recorded on other
        cache service instances of this worker are both treated as
        unavailable, so a restarted worker can't re-issue a port an existing
        instance holds. The metrics port additionally excludes the service
        port picked just before it.
        """
        with CacheServiceManager._port_lock:
            unavailable_ports = {
                port for pair in self._assigned_ports.values() for port in pair
            }
            try:
                instances_page = self._clientset.cache_service_instances.list(
                    # page=-1 disables pagination: a truncated page would
                    # blind the conflict check to the ports it dropped.
                    params={"worker_id": self._worker_id, "page": -1}
                )
                for existing in instances_page.items or []:
                    if existing.id == instance.id:
                        continue
                    if existing.port:
                        unavailable_ports.add(existing.port)
                    if existing.metrics_port:
                        unavailable_ports.add(existing.metrics_port)
            except Exception as e:
                logger.warning(
                    f"Failed to list cache service instances for port "
                    f"allocation: {e}"
                )

            # Prefer the ports already recorded on the instance: engines
            # attached to this cache server carry them in denormalized
            # snapshots that nothing refreshes, so a restart that changed
            # ports would strand every running deployment on a dead
            # endpoint until its model instances are recreated.
            if (
                instance.port
                and instance.metrics_port
                and instance.port not in unavailable_ports
                and instance.metrics_port not in unavailable_ports
                and network.is_port_available(instance.port)
                and network.is_port_available(instance.metrics_port)
            ):
                self._assigned_ports[instance.id] = (
                    instance.port,
                    instance.metrics_port,
                )
                return instance.port, instance.metrics_port

            port = network.get_free_port(
                port_range=self._config.service_port_range,
                unavailable_ports=unavailable_ports,
            )
            unavailable_ports.add(port)
            metrics_port = network.get_free_port(
                port_range=self._config.service_port_range,
                unavailable_ports=unavailable_ports,
            )
            self._assigned_ports[instance.id] = (port, metrics_port)
            return port, metrics_port

    def _release_ports(self, instance_id: int):
        with CacheServiceManager._port_lock:
            self._assigned_ports.pop(instance_id, None)

    def sync_cache_service_instances_state(self):
        """
        Synchronize managed cache service instances' state on this worker:
        - PENDING past PENDING_START_GRACE_SECONDS -> start (the start path
          is idempotent), recovering starts that never took effect.
        - Workload missing, failed, unhealthy or exited -> restart with
          exponential backoff; after MAX_CONSECUTIVE_RESTARTS crashes -> ERROR.
        - Health probe passes -> RUNNING (healthy).
        - Health probe fails after RUNNING -> UNREACHABLE.
        - STARTING with a failing probe is left alone (still booting).
        """
        instances_page = self._clientset.cache_service_instances.list(
            # page=-1 disables pagination: instances beyond a page would
            # never be synced or restarted.
            params={"worker_id": self._worker_id, "page": -1}
        )
        # Prune start bookkeeping for rows that no longer exist (a missed
        # DELETED event would otherwise accumulate entries forever).
        listed_ids = {instance.id for instance in instances_page.items or []}
        with CacheServiceManager._start_lock:
            for stale_id in set(self._last_start_attempt) - listed_ids:
                self._last_start_attempt.pop(stale_id, None)
        if not instances_page.items:
            return

        # Parent services are fetched once per sync pass: siblings of a
        # per-node service share the same parent row.
        parent_services: Dict[int, Optional[CacheServicePublic]] = {}
        for instance in instances_page.items:
            if instance.worker_id != self._worker_id:
                continue
            if instance.state == CacheServiceStateEnum.PENDING:
                self._start_stale_pending_instance(instance)
                continue
            if instance.state not in (
                CacheServiceStateEnum.STARTING,
                CacheServiceStateEnum.RUNNING,
                CacheServiceStateEnum.UNREACHABLE,
            ):
                continue
            try:
                cache_service = self._get_parent_service(
                    parent_services, instance.cache_service_id
                )
                if cache_service is None:
                    # The parent is gone; the instance row is about to be
                    # cascade-deleted, so there is nothing to sync against.
                    logger.debug(
                        f"Skipped syncing cache service instance {instance.id}: "
                        f"parent service {instance.cache_service_id} not found"
                    )
                    continue
                self._sync_single_cache_service_instance_state(instance, cache_service)
            except Exception as e:
                logger.error(
                    f"Failed to sync cache service instance {instance.id} "
                    f"(service id={instance.cache_service_id}) state: {e}"
                )

    def _start_stale_pending_instance(self, instance: CacheServiceInstance):
        """
        Start a PENDING instance whose start never took effect.

        Starts are triggered by the instance's PENDING event, so a start that
        was lost — event never delivered, worker restarted mid-start, the
        STARTING write-back never reached the server — would otherwise leave
        the instance PENDING forever, with or without a running container.
        Instances PENDING for less than PENDING_START_GRACE_SECONDS are left
        to the event path, starts still in flight are not duplicated, and one
        that already ran that recently is not repeated — a start whose
        write-back never lands would otherwise recreate the container on every
        sync pass.
        """
        now = datetime.now(timezone.utc)
        updated_at = instance.updated_at
        if (
            updated_at is not None
            and (now - updated_at).total_seconds() < PENDING_START_GRACE_SECONDS
        ):
            return

        with CacheServiceManager._start_lock:
            if instance.id in self._starting:
                return
            last_attempt = self._last_start_attempt.get(instance.id)
        if (
            last_attempt is not None
            and (now - last_attempt).total_seconds() < PENDING_START_GRACE_SECONDS
        ):
            return

        logger.info(
            f"Starting cache service instance {instance.id} "
            f"(service id={instance.cache_service_id}): still pending "
            f"after {PENDING_START_GRACE_SECONDS}s"
        )
        self._schedule_start(instance)

    def _get_parent_service(
        self,
        cache: Dict[int, Optional[CacheServicePublic]],
        cache_service_id: int,
    ) -> Optional[CacheServicePublic]:
        if cache_service_id not in cache:
            try:
                cache[cache_service_id] = self._clientset.cache_services.get(
                    id=cache_service_id
                )
            except NotFoundException:
                cache[cache_service_id] = None
        return cache[cache_service_id]

    def _sync_single_cache_service_instance_state(
        self,
        instance: CacheServiceInstance,
        cache_service: CacheServicePublic,
    ):
        """Synchronize a single cache service instance's state."""
        deployment_metadata = instance.get_deployment_metadata()
        workload = get_workload(deployment_metadata.name)

        if not workload or workload.state in [
            WorkloadStatusStateEnum.FAILED,
            WorkloadStatusStateEnum.UNHEALTHY,
            WorkloadStatusStateEnum.INACTIVE,
        ]:
            self._restart_crashed_cache_service_instance(
                instance, cache_service, deployment_metadata.name
            )
            return

        ready = self._probe_ready(instance, cache_service.provider_name)
        now = datetime.now(timezone.utc)
        if ready:
            updates = {}
            if (
                instance.state != CacheServiceStateEnum.RUNNING
                or instance.healthy is not True
            ):
                updates.update(
                    state=CacheServiceStateEnum.RUNNING,
                    healthy=True,
                    last_check_at=now,
                    state_message="",
                )
            # An instance that has stayed healthy past the reset window has
            # broken out of its crash loop; clear the consecutive-restart
            # budget so a much later crash gets a fresh set of attempts.
            if (
                (instance.restart_count or 0) > 0
                and instance.last_restart_time is not None
                and (now - instance.last_restart_time).total_seconds()
                >= RESTART_COUNT_RESET_SECONDS
            ):
                updates["restart_count"] = 0
            if updates:
                self._update_cache_service_instance(instance.id, **updates)
            return

        if instance.state == CacheServiceStateEnum.RUNNING:
            self._update_cache_service_instance(
                instance.id,
                state=CacheServiceStateEnum.UNREACHABLE,
                healthy=False,
                last_check_at=now,
            )
        # STARTING with a failing probe: the server is still booting, leave it
        # for the next sync round.

    def _restart_crashed_cache_service_instance(
        self,
        instance: CacheServiceInstance,
        cache_service: CacheServicePublic,
        workload_name: str,
    ):
        """
        Recover a cache service instance whose workload is missing or dead by
        re-entering PENDING (which retriggers the normal start path) with
        exponential backoff. After MAX_CONSECUTIVE_RESTARTS consecutive
        crashes — or immediately when the parent service disables
        restart_on_error — the instance is parked in ERROR until a manual
        restart.
        """
        if cache_service.restart_on_error is False:
            if instance.state != CacheServiceStateEnum.ERROR:
                self._update_cache_service_instance(
                    instance.id,
                    state=CacheServiceStateEnum.ERROR,
                    state_message=(
                        "Cache server exited. Automatic restart is disabled "
                        "for this service; restart it manually."
                    ),
                    healthy=False,
                )
            return

        restart_count = instance.restart_count or 0
        if restart_count >= MAX_CONSECUTIVE_RESTARTS:
            self._update_cache_service_instance(
                instance.id,
                state=CacheServiceStateEnum.ERROR,
                state_message=(
                    f"Cache server keeps crashing "
                    f"({MAX_CONSECUTIVE_RESTARTS} restarts attempted). "
                    "Check the service logs for the failure cause."
                ),
                healthy=False,
            )
            return

        now = datetime.now(timezone.utc)
        delay = min(
            RESTART_BACKOFF_BASE_SECONDS * 2**restart_count,
            RESTART_BACKOFF_MAX_SECONDS,
        )
        last_restart_time = instance.last_restart_time
        if (
            last_restart_time is not None
            and (now - last_restart_time).total_seconds() < delay
        ):
            # Within the backoff window; retry on a later sync round.
            return

        try:
            delete_workload(workload_name)
        except Exception as e:
            # The workload may already be gone.
            logger.debug(
                f"Skipped deleting crashed cache service workload "
                f"{workload_name}: {e}"
            )

        attempt = restart_count + 1
        logger.info(
            f"Restarting crashed cache service {cache_service.name} instance "
            f"(id={instance.id}), attempt {attempt}/{MAX_CONSECUTIVE_RESTARTS}"
        )
        self._update_cache_service_instance(
            instance.id,
            state=CacheServiceStateEnum.PENDING,
            restart_count=attempt,
            last_restart_time=now,
            state_message=(
                f"Cache server exited; restarting "
                f"(attempt {attempt}/{MAX_CONSECUTIVE_RESTARTS})."
            ),
            healthy=False,
        )

    def _probe_ready(self, instance: CacheServiceInstance, provider_name: str) -> bool:
        """
        Probe the cache server per the provider's health check declaration.
        Managed cache servers run with host networking on this worker, so
        loopback reaches them directly.
        """
        provider = get_cache_provider(provider_name)
        health_check = provider.health_check if provider else CacheProviderHealthCheck()
        host = "127.0.0.1"
        port = (
            instance.metrics_port if health_check.target == "metrics" else instance.port
        )
        if not port:
            return False

        if health_check.scheme == "http":
            path = health_check.path or "/"
            if not path.startswith("/"):
                path = "/" + path
            try:
                resp = httpx.get(
                    f"http://{host}:{port}{path}",
                    timeout=HEALTH_PROBE_TIMEOUT_SECONDS,
                )
                return resp.status_code < 400
            except Exception:
                return False

        try:
            with socket.create_connection(
                (host, port), timeout=HEALTH_PROBE_TIMEOUT_SECONDS
            ):
                return True
        except Exception:
            return False

    def _stop_cache_service_instance(self, instance: CacheServiceInstance):
        """
        Stop the instance's workload and free its tracked ports.

        Args:
            instance: The cache service instance to stop.
        """
        deployment_metadata = instance.get_deployment_metadata()
        try:
            delete_workload(deployment_metadata.name)
        except Exception as e:
            # The workload may already be gone (never created or cleaned up).
            logger.warning(
                f"Failed to delete cache service workload "
                f"{deployment_metadata.name}: {e}"
            )
        self._release_ports(instance.id)
        self._forget_start(instance.id)
        logger.info(
            f"Stopped cache service instance {instance.id} "
            f"(service id={instance.cache_service_id})"
        )

    def _update_cache_service_instance(self, id: int, **kwargs) -> bool:
        """
        Update cache service instance with given fields.

        Args:
            id: The ID of the cache service instance to update.
            **kwargs: The fields to update, group by field name and value.

        Returns:
            Whether the update was applied. A failed write-back is reported
            rather than raised: callers run on the watch event loop or the
            sync thread, where an exception would be dropped, and the sync
            pass re-drives what the lost update would have set.
        """
        try:
            instance_public = self._clientset.cache_service_instances.get(id=id)

            instance = CacheServiceInstanceUpdate(**instance_public.model_dump())
            for key, value in kwargs.items():
                set_attr(instance, key, value)

            self._clientset.cache_service_instances.update(id=id, model_update=instance)
            return True
        except NotFoundException:
            logger.warning(
                f"Cache service instance with ID {id} not found when trying "
                "to update."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to update cache service instance {id}: {e}")
            return False
