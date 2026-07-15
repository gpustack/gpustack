import asyncio
from multiprocessing import Process
import os
import re
import shutil
import subprocess
import tempfile
import threading
import importlib.util
import aiohttp

import uvicorn
from fastapi import FastAPI
import logging
import secrets
import tenacity
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.gpu_instances import sync_builtin_templates_to_db
from gpustack.logging import setup_logging
from gpustack.schemas.users import (
    User,
    get_default_cluster_principal,
    default_cluster_principal_name,
)
from gpustack.schemas.principals import (
    Principal,
    PrincipalType,
    init_authenticated_principal_id,
    init_platform_principal_id,
    platform_principal_id,
)
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster, ClusterProvider, ClusterStateEnum
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget
from gpustack.schemas.model_provider import ModelProvider
from gpustack.security import (
    generate_secure_password,
    get_secret_hash,
    API_KEY_PREFIX,
)
from gpustack.server.app import create_app
from gpustack.server.passwords import set_password
from gpustack.server.services import provision_bootstrap_admin_orgs
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.config import registration
from gpustack.server.catalog import init_model_catalog
from gpustack.server.controllers import (
    ModelController,
    ModelFileController,
    ModelInstanceController,
    WorkerController,
    ClusterController,
    WorkerPoolController,
    InferenceBackendController,
    ModelRouteController,
    ModelRouteTargetController,
    ModelProviderController,
)
from gpustack.gpu_instances.controllers import (
    GPUInstanceController,
    GPUInstancePersistentVolumeController,
    GPUInstancePersistentVolumeTypeController,
)
from gpustack.server.db import async_session
from gpustack.server.lora_model_routes import (
    cleanup_orphan_lora_routes,
    create_lora_model_routes,
)
from gpustack.utils.lora_model_source import normalized_lora_list
from gpustack.server.init_db import init_db, get_query_count
from gpustack.scheduler.scheduler import Scheduler
from gpustack.server.system_load import SystemLoadCollector
from gpustack.server.update_check import UpdateChecker
from gpustack.server.worker_status_buffer import flush_worker_status_to_db
from gpustack.server.metrics_collector import flush_gateway_metrics_to_db
from gpustack.server.usage_details_archiver import UsageDetailsArchiver
from gpustack.server.resource_event_logger import ResourceEventLogger
from gpustack.server.resource_usage_collector import ResourceUsageCollector
from gpustack.server.storage_usage_collector import StorageUsageCollector
from gpustack import envs
from gpustack.server.usage_archiver import TableArchiver
from gpustack.schemas.metered_usage import MeteredUsage, MeteredUsageArchive
from gpustack.schemas.resource_events import ResourceEvent, ResourceEventArchive
from gpustack.server.worker_instance_cleaner import WorkerInstanceCleaner
from gpustack.server.worker_syncer import WorkerSyncer
from gpustack.utils.process import add_signal_handlers_in_loop
from gpustack.config.registration import write_registration_token
from gpustack.exporter.exporter import MetricExporter
from gpustack.gateway.utils import (
    model_ingress_prefix,
    model_route_ingress_prefix,
    model_route_ingress_name,
    fallback_ingress_name,
    cleanup_ingresses,
    cleanup_model_mapper,
    cleanup_fallback_filters,
    cleanup_ai_proxy_config,
    cleanup_generic_proxy_router,
    cleanup_mcpbridge_registry,
    resolve_instance_address_from_model_header,
)
from gpustack.gateway import get_async_k8s_config
from gpustack.envs import (
    GATEWAY_PORT_CHECK_INTERVAL,
    GATEWAY_PORT_CHECK_RETRY_COUNT,
    DEFAULT_CLUSTER_KUBERNETES,
)
from gpustack.server.coordinator import LocalCoordinator
from gpustack.server.coordinator.cache import preload_cache
from gpustack.server.coordinator.models import get_model_for_topic
from gpustack.server import bus
from gpustack.server import cache as cache_module
from alembic import command
from alembic.config import Config as AlembicConfig

from gpustack.websocket_proxy.proxy_server import HTTPSProxyServer
from gpustack.api.auth import (
    authenticate_worker_by_request_headers,
)

from gpustack.gpu_instances import gateway_client
from gpustack.gpu_instances.gateway import (
    reconcile_gpustack_operator_subscription,
)

logger = logging.getLogger(__name__)


class _ExternalSubprocess:
    """Adapter that exposes a ``subprocess.Popen`` as a ``multiprocessing.Process``-like
    handle, so it can live alongside the worker process in ``_sub_processes`` and be
    driven by ``_start_sub_processes`` / ``_monitor_sub_processes``.

    Stdout and stderr are captured and pumped line-by-line into the server logger
    by background daemon threads, so the operator's output shows up in the same
    place as the rest of the server log.
    """

    def __init__(self, args, name="", log=None, on_exit=None):
        self._args = args
        self.name = name or args[0]
        self._popen = None
        self._logger = log or logger
        self._log_threads = []
        self._on_exit = on_exit
        self._exit_callback_invoked = False
        self._exit_lock = threading.Lock()

    def start(self):
        self._popen = subprocess.Popen(
            self._args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._spawn_log_pump(self._popen.stdout, logging.INFO)
        self._spawn_log_pump(self._popen.stderr, logging.INFO)
        if self._on_exit is not None:
            self._spawn_exit_watcher()

    def stop(self):
        if self._popen:
            self._popen.terminate()
            try:
                self._popen.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._popen.kill()

    def _spawn_exit_watcher(self):
        def watch():
            try:
                self._popen.wait()
            finally:
                self._invoke_on_exit()

        t = threading.Thread(
            target=watch,
            name=f"{self.name}-exit-watcher",
            daemon=True,
        )
        t.start()

    def _invoke_on_exit(self):
        with self._exit_lock:
            if self._exit_callback_invoked:
                return
            self._exit_callback_invoked = True
        try:
            self._on_exit()
        except Exception:
            self._logger.exception("[%s] on_exit callback failed", self.name)

    def _spawn_log_pump(self, stream, level):
        def pump():
            try:
                for line in iter(stream.readline, ""):
                    line = line.rstrip()
                    if line:
                        self._logger.log(level, "[%s] %s", self.name, line)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t = threading.Thread(
            target=pump,
            name=f"{self.name}-log-{'out' if level == logging.INFO else 'err'}",
            daemon=True,
        )
        t.start()
        self._log_threads.append(t)

    def is_alive(self) -> bool:
        if self._popen is None:
            return False
        return self._popen.poll() is None

    @property
    def exitcode(self):
        if self._popen is None:
            return None
        return self._popen.returncode


class Server:
    def __init__(self, config: Config, worker_process: Process):
        self._config: Config = config
        self._before_healthy_sub_processes = []
        self._after_healthy_sub_processes = []
        self._async_tasks = []
        self._worker_process = worker_process
        # Coordination components
        self._coordinator = None
        self._leader_election_task = None

    @property
    def all_processes(self):
        return self._before_healthy_sub_processes + self._after_healthy_sub_processes

    def _create_async_task(self, coro):
        self._async_tasks.append(asyncio.create_task(coro))

    @property
    def config(self):
        return self._config

    async def start(self):
        logger.info("Starting GPUStack server.")

        if self._config.external_auth_insecure_skip_tls_verify:
            logger.warning(
                "external_auth_insecure_skip_tls_verify is enabled: TLS "
                "verification is DISABLED for the external-auth IdP handshake."
            )

        add_signal_handlers_in_loop()

        self._run_migrations()
        await self._prepare_data()

        init_model_catalog(self._config.model_catalog_file)
        # it's safe to determine server_role after migration
        if self._config.server_role() == Config.ServerRole.BOTH:
            self._after_healthy_sub_processes.append(self._worker_process)

        self._enqueue_operator_process()

        # Create FastAPI app. Plugin ``__init__(app, cfg)`` runs here and
        # may attach a distributed-mode coordinator to the plugin instance.
        app = create_app(self._config)
        self._app = app

        # Initialize coordinator from plugin instances (LocalCoordinator if
        # none supplied). Must run before the event bus goes online so any
        # early publishes are routed correctly.
        await self._init_coordinator(app)

        # Preload change-detection cache after the coordinator is up.
        # Required in distributed mode so the first cross-instance event
        # on each topic carries accurate ``changed_fields``.
        await self._preload_change_detector_cache()

        self._start_sub_processes()

        # Start Leader-Only tasks (includes scheduler and controllers)
        # In single-node mode, they start immediately.
        # In distributed mode, they start only when this node becomes leader.
        await self._start_leader_only_tasks()

        # These tasks can run on all instances
        self._start_worker_status_flusher()
        self._start_gateway_metrics_flusher()
        self._start_metrics_exporter()
        self._start_query_count_logger()
        self._start_default_registry_checker()
        self._start_proxy_servers(app)
        self._start_extension_plugins(app)
        self._start_gpustack_operator_subscription()

        serving_host = (
            "127.0.0.1"
            if self._config.gateway_mode == GatewayModeEnum.embedded
            else "0.0.0.0"
        )

        config = uvicorn.Config(
            app,
            host=serving_host,
            port=self._config.get_api_port(),
            access_log=False,
            log_level="error",
        )

        setup_logging()
        logger.info(f"Gateway mode: {self._config.gateway_mode.value}.")
        serving_api_message = f"Serving GPUStack API on {config.host}:{config.port}."
        if self._config.gateway_mode == GatewayModeEnum.embedded:
            logger.debug(serving_api_message)
            logger.info(
                f"GPUStack Server will serve on 0.0.0.0:{self._config.get_gateway_port()}."
            )
            if self._config.get_tls_secret_name() is not None:
                logger.info(
                    f"GPUStack Server will serve TLS on 0.0.0.0:{self._config.tls_port}."
                )
        else:
            logger.info(serving_api_message)

        server = uvicorn.Server(config)
        self._create_async_task(server.serve())

        await asyncio.gather(*self._async_tasks)

    def _start_default_registry_checker(self):
        registration.determine_default_registry(
            self._config.system_default_container_registry,
        ),

    def _run_migrations(self):
        logger.info("Running database migration.")

        spec = importlib.util.find_spec("gpustack")
        if spec is None:
            raise ImportError("The 'gpustack' package is not found.")

        pkg_path = spec.submodule_search_locations[0]
        alembic_cfg = AlembicConfig()
        alembic_cfg.set_main_option(
            "script_location", os.path.join(pkg_path, "migrations")
        )

        db_url = self._config.get_database_url()
        # Use the pymysql driver to execute migrations to avoid compatibility issues between asynchronous drivers and Alembic.
        if db_url.startswith("mysql://"):
            db_url = re.sub(r'^mysql://', 'mysql+pymysql://', db_url)
        db_url_escaped = db_url.replace("%", "%%")
        alembic_cfg.set_main_option("sqlalchemy.url", db_url_escaped)
        try:
            command.upgrade(alembic_cfg, "head")
        except Exception as e:
            raise RuntimeError(f"Database migration failed: {e}") from e
        logger.info("Database migration completed.")

    async def _prepare_data(self):
        self._setup_data_dir(self._config.data_dir)

        await init_db(self._config.get_database_url())

        async with async_session() as session:
            await self._init_data(session)

        logger.debug("Data initialization completed.")

    def _start_scheduler(self):
        """Start the scheduler and return the task."""
        scheduler = Scheduler(self._config)
        task = asyncio.create_task(scheduler.start())
        logger.debug("Scheduler started.")
        return task

    def _start_controllers(self):
        """Start all controllers and return the list of tasks."""
        tasks = []

        model_provider_controller = ModelProviderController(self._config)
        tasks.append(asyncio.create_task(model_provider_controller.start()))

        model_route_target_controller = ModelRouteTargetController(self._config)
        tasks.append(asyncio.create_task(model_route_target_controller.start()))

        model_route_controller = ModelRouteController(self._config)
        tasks.append(asyncio.create_task(model_route_controller.start()))

        model_controller = ModelController(self._config)
        tasks.append(asyncio.create_task(model_controller.start()))

        model_instance_controller = ModelInstanceController(self._config)
        tasks.append(asyncio.create_task(model_instance_controller.start()))

        worker_controller = WorkerController(self._config)
        tasks.append(asyncio.create_task(worker_controller.start()))

        model_file_controller = ModelFileController()
        tasks.append(asyncio.create_task(model_file_controller.start()))

        cluster_controller = ClusterController(self._config)
        tasks.append(asyncio.create_task(cluster_controller.start()))

        worker_pool_controller = WorkerPoolController()
        tasks.append(asyncio.create_task(worker_pool_controller.start()))

        inference_backend_controller = InferenceBackendController()
        tasks.append(asyncio.create_task(inference_backend_controller.start()))

        gpu_instance_controller = GPUInstanceController(self._config)
        tasks.append(asyncio.create_task(gpu_instance_controller.start()))

        gpu_instance_pv_controller = GPUInstancePersistentVolumeController(self._config)
        tasks.append(asyncio.create_task(gpu_instance_pv_controller.start()))

        gpu_instance_pvt_controller = GPUInstancePersistentVolumeTypeController(
            self._config
        )
        tasks.append(asyncio.create_task(gpu_instance_pvt_controller.start()))

        logger.debug("Controllers started.")
        return tasks

    def _start_system_load_collector(self):
        collector = SystemLoadCollector()
        self._create_async_task(collector.start())

        logger.debug("System load collector started.")

    def _start_worker_syncer(self, app: FastAPI):
        worker_syncer = WorkerSyncer(
            lambda: getattr(app.state, "http_client", None),
            lambda: getattr(app.state, "http_client_no_proxy", None),
        )
        self._create_async_task(worker_syncer.start())

        logger.debug("Worker syncer started.")

    def _start_worker_status_flusher(self):
        self._create_async_task(flush_worker_status_to_db())

        logger.debug("Worker status flusher started.")

    def _start_gpustack_operator_subscription(self):
        """Keep the in-process gpustack-operator gateway in sync with Cluster events.

        Runs on every server instance — each server spawns its own
        ``gpustack-operator`` subprocess and must feed it the current cluster
        set, so this cannot be a leader-only task.
        """
        if shutil.which("gpustack-operator") is None:
            return
        self._create_async_task(reconcile_gpustack_operator_subscription())

        logger.debug("GPUStack operator subscription started.")

    def _start_gateway_metrics_flusher(self):
        # Always start — both the gateway report endpoint and the in-process
        # ModelUsageMiddleware feed the same buffer, so the flusher must run
        # even when the external gateway is disabled.
        self._create_async_task(flush_gateway_metrics_to_db())

        logger.debug("Gateway metrics flusher started.")

    def _start_worker_instance_cleaner(self):
        worker_instance_cleaner = WorkerInstanceCleaner()
        self._create_async_task(worker_instance_cleaner.start())

        logger.debug("Worker instance cleaner started.")

    def _start_usage_details_archiver(self):
        # Construction can fail on schema drift between hot/archive tables or
        # an invalid cron expression. Surface that loudly and skip launching
        # the loop so the rest of the leader tasks (and the leader-election
        # retry) aren't taken down with it. Without the archiver the
        # model_usage_details hot table will grow unbounded — operators must
        # see this in logs rather than have it buried as "Leader election
        # error" by the outer election handler.
        try:
            archiver = UsageDetailsArchiver()
        except Exception:
            logger.critical(
                "Usage details archiver failed to initialize — archival is "
                "DISABLED. The model_usage_details hot table will grow "
                "unbounded until this is resolved.",
                exc_info=True,
            )
            return
        self._create_async_task(archiver.start())

        logger.debug("Usage details archiver started.")

    def _start_resource_usage(self):
        """Start the resource-metering pipeline: event logger → collectors →
        events archiver. Leader-only — the logger writes one resource_events row
        per lifecycle transition, so concurrent loggers would double-write.

        The events archiver construction can fail on schema drift / bad cron;
        surface that and skip just the archiver, keeping the logger + collectors
        (and the rest of the leader tasks) running.
        """
        self._create_async_task(ResourceEventLogger().start())
        self._create_async_task(ResourceUsageCollector().start())
        self._create_async_task(StorageUsageCollector().start())

        # Hot/cold archivers for metered_usage + resource_events. Construction can
        # fail on schema drift / bad cron; surface that and skip just the failed
        # archiver, keeping the logger + collectors (and other leader tasks).
        for archiver_factory, name in (
            (
                lambda: TableArchiver(
                    MeteredUsage,
                    MeteredUsageArchive,
                    anchor_col="bucket_start",
                    retention_months=envs.METERED_USAGE_RETENTION_MONTHS,
                    cron=envs.METERED_USAGE_ARCHIVE_CRON,
                    batch_size=envs.METERED_USAGE_ARCHIVE_BATCH_SIZE,
                    label="metered_usage",
                ),
                "metered_usage",
            ),
            (
                lambda: TableArchiver(
                    ResourceEvent,
                    ResourceEventArchive,
                    anchor_col="occurred_at",
                    retention_months=envs.USAGE_EVENTS_RETENTION_MONTHS,
                    cron=envs.USAGE_EVENTS_ARCHIVE_CRON,
                    batch_size=envs.USAGE_EVENTS_ARCHIVE_BATCH_SIZE,
                    label="resource_events",
                ),
                "resource_events",
            ),
        ):
            try:
                archiver = archiver_factory()
            except Exception:
                logger.critical(
                    "%s archiver failed to initialize — archival is DISABLED; "
                    "the hot table will grow unbounded until resolved.",
                    name,
                    exc_info=True,
                )
            else:
                self._create_async_task(archiver.start())

        logger.debug("Resource usage metering started.")

    def _start_update_checker(self):
        """Start update checker."""
        if self._config.disable_update_check:
            return
        update_checker = UpdateChecker(update_check_url=self._config.update_check_url)
        self._create_async_task(update_checker.start())
        logger.debug("Update checker started.")

    async def _monitor_sub_processes(self):
        while self.all_processes:
            for process in self.all_processes[:]:
                if not process.is_alive():
                    if process.exitcode != 0:
                        raise RuntimeError(
                            f"Sub process {process.name} died with exit code {process.exitcode}"
                        )
                    if process in self._before_healthy_sub_processes:
                        self._before_healthy_sub_processes.remove(process)
                    elif process in self._after_healthy_sub_processes:
                        self._after_healthy_sub_processes.remove(process)
            await asyncio.sleep(5)

    def _enqueue_operator_process(self):
        """Queue the ``gpustack-operator wg`` subprocess for startup.

        Skipped silently when the binary is not on PATH so deployments without
        the operator addon installed don't fail.

        The worker gateway is bound to a process-private unix socket whose
        path is randomized for each server startup. This narrows the
        socket's discoverability so unrelated local processes cannot easily
        target it, and the path is handed to the in-process gateway client
        so other server code can reach the operator's ``/apis/*`` endpoints.
        """
        if shutil.which("gpustack-operator") is None:
            logger.debug("gpustack-operator binary not found on PATH, skipping start.")
            return

        socket_dir = tempfile.mkdtemp(prefix="gpustack-operator-")
        os.chmod(socket_dir, 0o700)

        socket_path = os.path.join(socket_dir, f"{secrets.token_hex(8)}.sock")
        gateway_client.set_gateway_unix_path(socket_path)

        args = [
            "gpustack-operator",
            "wg",
            "--worker-conn-gpustack-api-port",
            str(self._config.get_api_port()),
            "--bind-unix-path",
            socket_path,
        ]
        if self._config.debug:
            args.append("--v=4")

        def _cleanup_socket_dir():
            shutil.rmtree(socket_dir, ignore_errors=True)

        self._before_healthy_sub_processes.append(
            _ExternalSubprocess(
                args, name="gpustack-operator", on_exit=_cleanup_socket_dir
            )
        )

    def _start_sub_processes(self):
        # Start before-healthy subprocesses synchronously so they are guaranteed
        # to be spawned before any leader task (scheduler, controllers, ...) is
        # scheduled. Otherwise the coroutine would only be queued and would not
        # actually run until the first ``await`` in ``start()``, by which time
        # controllers have already been created.
        for process in self._before_healthy_sub_processes:
            process.start()

        async def wait_and_start_after_healthy():
            # Wait for the API to be healthy before starting the rest of the processes,
            # so that they can rely on the API being available when they start.
            api_url = f"http://127.0.0.1:{self._config.api_port}/healthz"
            async with aiohttp.ClientSession() as session:
                while True:
                    try:
                        await asyncio.sleep(2)
                        async with session.get(api_url) as response:
                            if response.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    except asyncio.CancelledError:
                        return

            # Start the rest of the processes after the API is ready.
            for process in self._after_healthy_sub_processes:
                process.start()

            await self._monitor_sub_processes()

        if len(self.all_processes) == 0:
            return

        self._create_async_task(wait_and_start_after_healthy())

    async def _wait_for_gateway_ready(self):
        if self._config.gateway_mode != GatewayModeEnum.embedded:
            return
        # http port is always started
        ports = [self._config.port]
        if self._config.get_tls_secret_name() is not None:
            ports.append(self._config.tls_port)
        logger.info(f"Waiting for ports {ports} of GPUStack to be ready...")
        # wait for gateway ready for about 60s
        await self._check_ports_ready(*ports)
        logger.info("GPUStack Server is ready.")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(GATEWAY_PORT_CHECK_RETRY_COUNT),
        wait=tenacity.wait_fixed(GATEWAY_PORT_CHECK_INTERVAL),
        reraise=True,
        before_sleep=lambda retry_state: logger.debug(
            f"Waiting for ports {retry_state.args[1]} to be healthy (attempt {retry_state.attempt_number}) due to: {retry_state.outcome.exception()}"
        ),
    )
    async def _check_ports_ready(self, *ports: int):
        for port in ports:
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
            except Exception:
                raise RuntimeError(f"Port {port} is not healthy or not listening")

    def _start_metrics_exporter(self):
        if self._config.disable_metrics:
            return

        exporter = MetricExporter(cfg=self._config)
        self._create_async_task(exporter.generate_metrics_cache())
        self._create_async_task(exporter.start())

    def _start_query_count_logger(self):
        """Start a background task to log query count periodically."""

        async def log_query_count():
            while True:
                await asyncio.sleep(60)  # Log every minute
                count = get_query_count()
                logger.debug(f"[DB QUERY COUNT] Total queries since startup: {count}")

        self._create_async_task(log_query_count())

    @staticmethod
    def _setup_data_dir(data_dir: str):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    async def _init_data(self, session: AsyncSession):
        # Resolve PLATFORM_PRINCIPAL_ID first: the identity-consolidation
        # migration renumbers the platform principal above
        # MAX(users.id), so every downstream init step that defaults to
        # ``platform_principal_id()`` needs the live value bound.
        await self._init_platform_principal_id(session)
        # ``system/authenticated`` is seeded lazily on first call —
        # bind it at startup so the resolver returns a real id before
        # the first request lands on ``_accessible_clusters``.
        await self._init_authenticated_principal_id(session)

        init_data_funcs = [
            self._init_user,
            self._init_default_cluster,
            self._migrate_legacy_token,
            self._migrate_legacy_workers,
            self._ensure_registration_token,
            self._cleanup_orphaned_gateway_data,
            sync_builtin_templates_to_db,
            self._reconcile_lora_model_routes,
        ]
        for init_data_func in init_data_funcs:
            await init_data_func(session)

    async def _reconcile_lora_model_routes(self, session: AsyncSession):
        """Bring per-LoRA ModelRoute rows in sync with each Model.lora_list:
        create missing ones and remove orphans (LoRA child routes whose
        name `<base>:<lora>` is no longer in lora_list and not mounted
        on any RUNNING instance).
        """
        all_models = await Model.all_by_fields(session, fields={"deleted_at": None})
        models_with_lora = [m for m in all_models if normalized_lora_list(m)]
        for model in models_with_lora:
            await create_lora_model_routes(
                session,
                model,
                access_policy=model.access_policy,
                generic_proxy=model.generic_proxy,
            )
            await cleanup_orphan_lora_routes(session, model)

        await session.commit()
        logger.info(
            f"LoRA model route reconcile: {len(models_with_lora)} models with lora_list scanned"
        )

    async def _init_platform_principal_id(self, session: AsyncSession):
        await init_platform_principal_id(session)

    async def _init_authenticated_principal_id(self, session: AsyncSession):
        await init_authenticated_principal_id(session)

    async def _init_user(self, session: AsyncSession):
        # Skip bootstrap when any non-system admin already exists, so that
        # renaming the default "admin" account does not cause a duplicate
        # admin to be regenerated on master restart.
        existing_admin = await User.first_by_fields(
            session=session,
            fields={
                "is_admin": True,
                "kind": PrincipalType.USER,
                "is_active": True,
            },
        )
        if existing_admin:
            return

        # A machine-generated bootstrap password forces a first-login change
        # and surfaces the retrieval guide; an operator-supplied one does not.
        # The machine-generated password lives in the ``initial_admin_password``
        # file under ``data_dir``: in HA the Helm chart mounts a shared Secret
        # there so every replica reads the same value, and single-node installs
        # generate it locally and write it there. An explicit
        # ``--bootstrap-password`` takes precedence and is not force-changed.
        bootstrap_password = self._config.bootstrap_password
        require_password_change = False
        password_file = os.path.join(self._config.data_dir, "initial_admin_password")
        if not bootstrap_password:
            require_password_change = True
            if os.path.exists(password_file):
                try:
                    with open(password_file, encoding="utf-8") as file:
                        bootstrap_password = file.read().strip()
                except (OSError, ValueError) as e:
                    # ValueError covers UnicodeDecodeError on a corrupted /
                    # non-UTF-8 file, which is not an OSError.
                    logger.warning(
                        f"Failed to read initial admin password from {password_file}: {e}"
                    )
            # An empty / whitespace-only / unreadable file (interrupted write,
            # empty Secret mount, permission issue) must never yield an empty
            # admin password — fall back to a freshly generated one.
            if bootstrap_password:
                logger.info(f"Using initial admin password from {password_file}.")
            else:
                bootstrap_password = generate_secure_password()
                try:
                    with open(password_file, "w", encoding="utf-8") as file:
                        file.write(bootstrap_password + "\n")
                    logger.info(
                        "Generated initial admin password. "
                        f"You can get it from {password_file}"
                    )
                except OSError as e:
                    logger.warning(
                        "Generated initial admin password but could not persist it "
                        f"to {password_file}: {e}"
                    )

        user = User(
            name="admin",
            display_name="Default System Admin",
            is_admin=True,
        )
        user = await User.create(session, user, auto_commit=False)
        await set_password(
            session,
            user.id,
            bootstrap_password,
            require_password_change=require_password_change,
            auto_commit=False,
        )
        await provision_bootstrap_admin_orgs(session, user)
        await session.commit()

    async def _migrate_legacy_token(self, session: AsyncSession):
        if not self._config.token:
            return
        # this should be created from sql migration script.
        cluster_principal = await get_default_cluster_principal(session)
        if cluster_principal is None or cluster_principal.cluster is None:
            logger.debug(
                "Default cluster user not exist, skipping legacy token migration."
            )
            return

        default_cluster = cluster_principal.cluster
        if not default_cluster:
            logger.debug(
                "Default cluster does not exist, skipping legacy token migration."
            )
            return
        if default_cluster.registration_token:
            return
        try:
            default_cluster.registration_token = self._config.token
            await default_cluster.update(session=session, auto_commit=False)

            if default_cluster.system_principal_id is None:
                raise RuntimeError("Default cluster has no system principal.")
            default_cluster_principal = await Principal.one_by_id(
                session=session, id=default_cluster.system_principal_id
            )
            if default_cluster_principal is None:
                raise RuntimeError("Default cluster user does not exist.")
            if len(default_cluster_principal.api_keys) > 0:
                raise RuntimeError(
                    "Default cluster user already has API keys, cannot migrate legacy token."
                )

            new_key = ApiKey(
                name="Legacy Cluster Token",
                access_key="",
                hashed_secret_key=get_secret_hash(self._config.token),
                user_id=default_cluster_principal.id,
                user=default_cluster_principal,
            )
            await ApiKey.create(session, new_key, auto_commit=False)
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to migrate legacy token: {e}")
            await session.rollback()
            raise e

    async def _migrate_legacy_workers(self, session: AsyncSession):
        # Use hardcode cluster 1 to make sure the cluster is created in migration step
        default_cluster = await Cluster.one_by_id(session=session, id=1)
        if not default_cluster:
            logger.debug(
                "Default cluster does not exist, skipping legacy worker migration."
            )
            return
        workers = await Worker.all_by_fields(
            session=session,
            fields={
                "cluster_id": default_cluster.id,
                "token": None,
            },
        )
        if len(workers) == 0:
            return
        system_name_prefix = "system/worker"
        # ``worker.system_principal_id`` is the post-inversion link. For
        # legacy workers it'll be NULL until we provision one below.
        for worker in workers:
            try:
                worker_principal = None
                if worker.system_principal_id is not None:
                    worker_principal = await Principal.one_by_id(
                        session=session, id=worker.system_principal_id
                    )
                if not worker_principal:
                    to_create_principal = Principal(
                        name=f'{system_name_prefix}-{worker.id}',
                        kind=PrincipalType.SYSTEM,
                    )
                    worker_principal = await Principal.create(
                        session, to_create_principal, auto_commit=False
                    )
                    worker.system_principal_id = worker_principal.id
                    await worker.save(session=session, auto_commit=False)
                    access_key = secrets.token_hex(8)
                    secret_key = secrets.token_hex(16)
                    to_create_apikey = ApiKey(
                        name=worker_principal.name,
                        access_key=access_key,
                        hashed_secret_key=get_secret_hash(secret_key),
                        user=worker_principal,
                        user_id=worker_principal.id,
                    )
                    await ApiKey.create(session, to_create_apikey, auto_commit=False)
                    await worker.update(
                        session=session,
                        source={"token": f"{API_KEY_PREFIX}_{access_key}_{secret_key}"},
                        auto_commit=False,
                    )
                    await session.commit()
            except Exception as e:
                logger.error(
                    f"Failed to migrate worker {worker.id} ({worker.name}): {e}"
                )
                await session.rollback()
                raise e

    async def _ensure_registration_token(self, session: AsyncSession):
        cluster_principal = await get_default_cluster_principal(session)
        if cluster_principal is None or cluster_principal.cluster is None:
            logger.debug(
                "Default cluster user not exist, skipping registration token generation."
            )
            return
        # Hold a local reference: ``ApiKey.create`` triggers
        # ``ActiveRecordMixin._refresh_related_objects`` which calls
        # ``session.refresh(cluster_principal)``, expiring its eagerly-loaded
        # ``cluster`` attribute. With ``User.cluster`` set to
        # ``lazy="noload"``, accessing ``cluster_principal.cluster``
        # afterwards returns ``None`` and the subsequent update would
        # blow up.
        cluster = cluster_principal.cluster
        token = cluster.registration_token
        if not token:
            try:
                access_key = secrets.token_hex(8)
                secret_key = secrets.token_hex(16)
                new_key = ApiKey(
                    name="Default Cluster Token",
                    access_key=access_key,
                    hashed_secret_key=get_secret_hash(secret_key),
                    user_id=cluster_principal.id,
                    user=cluster_principal,
                )
                await ApiKey.create(session, new_key, auto_commit=False)
                token = f"{API_KEY_PREFIX}_{access_key}_{secret_key}"
                await cluster.update(
                    session=session,
                    source={"registration_token": token},
                    auto_commit=False,
                )
                await session.commit()
            except Exception as e:
                logger.error(f"Failed to ensure registration token: {e}")
                await session.rollback()
                raise e

        write_registration_token(
            data_dir=self._config.data_dir,
            token=token,
        )

    async def _cleanup_orphaned_gateway_data(self, session: AsyncSession):
        if self.config.gateway_mode == GatewayModeEnum.disabled:
            return
        # Remove the orphaned ingresses of model routes
        model_routes = await ModelRoute.all_by_field(
            session=session, field="deleted_at", value=None
        )
        route_targets = await ModelRouteTarget.all_by_fields(
            session=session,
            fields={"deleted_at": None},
        )
        providers = await ModelProvider.all_by_fields(
            session=session,
            fields={"deleted_at": None},
        )
        model_instances = await ModelInstance.all_by_fields(
            session=session,
            fields={"deleted_at": None},
        )
        workers = await Worker.all_by_fields(
            session=session,
            fields={"deleted_at": None},
        )
        # Needed so cleanup can tell which instances serve their subordinates'
        # own API (hybrid/external-LB) and which LoRA aliases to keep.
        models = await Model.all_by_fields(
            session=session,
            fields={"deleted_at": None},
        )
        fallback_route_ids = [
            ep.route_id
            for ep in route_targets
            if ep.fallback_status_codes is not None
            and len(ep.fallback_status_codes) > 0
        ]
        expected_ingress_names = [
            model_route_ingress_name(model_route.id) for model_route in model_routes
        ]
        expected_names = expected_ingress_names + [
            fallback_ingress_name(model_route_ingress_name(id))
            for id in fallback_route_ids
        ]

        k8s_config = get_async_k8s_config(cfg=self.config)
        await cleanup_ingresses(
            namespace=self.config.get_namespace(),
            expected_names=expected_names,
            config=k8s_config,
            cleanup_prefix=model_route_ingress_prefix,
            reason="orphaned",
        )
        await cleanup_ingresses(
            namespace=self.config.get_namespace(),
            expected_names=expected_names,
            config=k8s_config,
            cleanup_prefix=model_ingress_prefix,
            reason="legacy",
        )
        await cleanup_model_mapper(
            namespace=self.config.gateway_namespace,
            expected_ingresses=expected_ingress_names,
            config=k8s_config,
        )
        await cleanup_fallback_filters(
            namespace=self.config.get_namespace(),
            expected_names=expected_names,
            cleanup_prefix=model_route_ingress_prefix,
            reason="orphaned",
            k8s_config=k8s_config,
        )
        await cleanup_ai_proxy_config(
            namespace=self.config.gateway_namespace,
            providers=providers,
            routes=model_routes,
            k8s_config=k8s_config,
        )
        await cleanup_generic_proxy_router(
            routes=model_routes,
            k8s_config=k8s_config,
            namespace=self.config.gateway_namespace,
        )
        await cleanup_mcpbridge_registry(
            providers=providers,
            namespace=self.config.gateway_namespace,
            model_instances=model_instances,
            workers=workers,
            models=models,
            k8s_config=k8s_config,
        )

    def _should_create_default_cluster(self) -> bool:
        # only server or both will get into this logic
        if self._config.server_role() == Config.ServerRole.BOTH:
            return True
        if self._config.token:
            return True
        return False

    async def _init_default_cluster(self, session: AsyncSession):
        if not self._should_create_default_cluster():
            return
        default_cluster_principal = await get_default_cluster_principal(session)
        if default_cluster_principal:
            return
        user_defined_default_cluster = await self.user_defined_default_cluster(session)
        set_default = user_defined_default_cluster is None
        logger.info("Creating default cluster...")
        provider = ClusterProvider.Docker
        if DEFAULT_CLUSTER_KUBERNETES:
            provider = ClusterProvider.Kubernetes
        hashed_suffix = secrets.token_hex(6)
        default_cluster = Cluster(
            name="Default Cluster",
            description="The default cluster for GPUStack",
            provider=provider,
            state=ClusterStateEnum.READY,
            hashed_suffix=hashed_suffix,
            registration_token="",
            is_default=set_default,
            owner_principal_id=platform_principal_id(),
        )
        default_cluster = await Cluster.create(
            session, default_cluster, auto_commit=False
        )

        default_cluster_principal = Principal(
            name=default_cluster_principal_name,
            kind=PrincipalType.SYSTEM,
        )
        default_cluster_principal = await Principal.create(
            session, default_cluster_principal, auto_commit=False
        )
        default_cluster.system_principal_id = default_cluster_principal.id
        await default_cluster.save(session=session, auto_commit=False)

        # No cluster_access grant needed: the cluster's `owner_principal_id`
        # already binds it to the platform Org, whose members are
        # implicit USER-level consumers. cluster_access rows are only
        # for cross-Org / group / user borrowing.

        await session.commit()
        logger.debug("Default cluster created.")

    async def user_defined_default_cluster(self, session: AsyncSession) -> Cluster:
        # Used during initial bootstrap to decide whether to create a
        # platform-Org default — only need to check the platform Org slot
        # since per-Org defaults are independent.
        cluster = await Cluster.one_by_fields(
            session=session,
            fields={
                "is_default": True,
                "owner_principal_id": platform_principal_id(),
                "deleted_at": None,
            },
        )
        return cluster

    def _start_proxy_servers(self, app: FastAPI) -> None:
        _proxy_server = HTTPSProxyServer(
            host=self._config.get_proxy_listen_address(),
            port=self._config.get_proxy_port(),
            connection_manager_getter=app.state.message_server_handler.get_connection_manager,
            authenticator=lambda headers: authenticate_worker_by_request_headers(
                headers, validate_proxy=None
            ),
            header_router=resolve_instance_address_from_model_header,
        )
        self._create_async_task(_proxy_server.start())

    def _start_extension_plugins(self, app: FastAPI) -> None:
        for plugin in getattr(app.state, "extension_plugins", []):
            try:
                for coro in plugin.async_tasks():
                    self._create_async_task(coro)
            except Exception:
                logger.exception(
                    "Failed to start async tasks from extension plugin %s",
                    type(plugin).__name__,
                )

    async def _init_coordinator(self, app: FastAPI):
        """Pick a coordinator from extension plugins (if any) and start it.

        Plugins attach a ``Coordinator`` to ``self.coordinator`` inside
        their ``__init__(app, cfg)``. We scan ``app.state.extension_plugins``
        after ``create_app`` has run and take the first non-None one. If
        no plugin supplies one, we fall back to ``LocalCoordinator``.
        """
        coordinator = None
        for plugin in getattr(app.state, "extension_plugins", []):
            candidate = getattr(plugin, "coordinator", None)
            if candidate is not None:
                coordinator = candidate
                logger.info(f"Coordinator provided by plugin: {type(plugin).__name__}")
                break

        if coordinator is None:
            coordinator = LocalCoordinator(self._config)
            logger.debug("Using LocalCoordinator")

        self._coordinator = coordinator
        await self._coordinator.start()

        # Set up bus and cache to use coordinator
        bus.set_coordinator(coordinator)
        await bus.event_bus.start()
        cache_module.set_coordinator(coordinator)

        await self._prepare_jwt_secret_key()

    async def _preload_change_detector_cache(self):
        if isinstance(self._coordinator, LocalCoordinator):
            return

        topics = [
            "worker",
            "model",
            "modelinstance",
            "modelroute",
            "modelroutetarget",
            "workerpool",
            "inferencebackend",
        ]
        async with async_session() as session:
            for topic in topics:
                model_class = get_model_for_topic(topic)
                if model_class is None:
                    continue
                try:
                    await preload_cache(topic, model_class, session)
                except Exception as e:
                    logger.warning(
                        f"Failed to preload change-detection cache for {topic}: {e}"
                    )

    async def _prepare_jwt_secret_key(self):
        """Enforce that distributed deployments use an explicit JWT secret.

        ``Config`` auto-generates a local ``jwt_secret_key`` file during init
        so early startup paths (e.g. ``initialize_gateway``) have a usable key.
        That auto-generated value is safe only in single-node mode; distributed
        instances must share the SAME secret or JWTs signed by one instance
        won't verify on another. We rely on the ``_jwt_secret_key_user_provided``
        flag (set from --jwt-secret-key / GPUSTACK_JWT_SECRET_KEY / config file)
        rather than the current value, since the value is always populated by
        the time this runs.
        """
        if self._config._jwt_secret_key_user_provided:
            return

        if isinstance(self._coordinator, LocalCoordinator):
            return

        raise RuntimeError(
            "jwt_secret_key must be explicitly set in distributed mode. "
            "Mount a Kubernetes Secret or pass it via the --jwt-secret-key flag "
            "or set the GPUSTACK_JWT_SECRET_KEY environment variable."
        )

    async def _start_leader_only_tasks(self):
        """Start tasks that should only run on the Leader instance."""
        if isinstance(self._coordinator, LocalCoordinator):
            # Local mode: start leader tasks directly (always run)
            self._start_leader_tasks()
            return

        # Distributed mode: start leader election loop
        logger.info("Starting leader election loop...")
        self._leader_election_task = asyncio.create_task(self._leader_election_loop())

    async def _leader_election_loop(self):
        """Main leader election loop using coordinator."""
        server_id = self._config.server_id
        ttl = self._coordinator.leader_election_ttl
        renew_interval = self._coordinator.leader_election_renew_interval
        is_first_attempt = True

        while True:
            try:
                if not self._coordinator.is_leader():
                    # Try to acquire leadership
                    if is_first_attempt:
                        logger.info(
                            f"Server {server_id} attempting to acquire leadership..."
                        )
                    acquired = await self._coordinator.acquire_leadership(ttl)
                    if acquired:
                        logger.info(
                            f"Server {server_id} became leader, starting scheduler and controllers"
                        )
                        # Start leader-only tasks
                        self._start_leader_tasks()
                    elif is_first_attempt:
                        logger.info(
                            f"Server {server_id} is standby, waiting for leadership..."
                        )
                        is_first_attempt = False
                else:
                    # Renew leadership
                    renewed = await self._coordinator.renew_leadership(ttl)
                    if not renewed:
                        logger.error(
                            f"Server {server_id} lost leadership, exiting for restart"
                        )
                        # Hard exit to prevent split-brain: os._exit bypasses
                        # cleanup so the process stops immediately and the
                        # container runtime can restart it as a standby.
                        os._exit(1)

                await asyncio.sleep(renew_interval)
            except Exception as e:
                logger.error(f"Leader election error: {e}")
                await asyncio.sleep(5)

    def _start_leader_tasks(self):
        """Start tasks that run only on the leader.

        Note: If leadership is lost, the process exits directly (os._exit),
        so we don't need to track and cancel these tasks.
        """
        # Scheduler
        self._start_scheduler()

        # Controllers
        self._start_controllers()

        # System Load Collector
        self._start_system_load_collector()

        # Update Checker
        self._start_update_checker()

        # Worker Instance Cleaner
        self._start_worker_instance_cleaner()

        # Usage Details Archiver (move aged rows to archive table)
        self._start_usage_details_archiver()

        # Resource usage metering (event logger → collectors → archiver)
        self._start_resource_usage()

        # Worker Syncer (checks worker reachability and updates states)
        self._start_worker_syncer(self._app)
