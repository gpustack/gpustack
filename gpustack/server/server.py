import asyncio
from multiprocessing import Process
import os
import re
import aiohttp

import uvicorn
import logging
import secrets
import tenacity
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.logging import setup_logging
from gpustack.schemas.users import (
    User,
    UserRole,
    get_default_cluster_user,
    default_cluster_user_name,
)
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster, ClusterProvider, ClusterStateEnum
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget
from gpustack.schemas.model_provider import ModelProvider
from gpustack.security import (
    JWTManager,
    generate_secure_password,
    get_secret_hash,
    API_KEY_PREFIX,
)
from gpustack.server.app import create_app
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
from gpustack.server.db import async_session
from gpustack.server.init_db import init_db, get_query_count
from gpustack.scheduler.scheduler import Scheduler
from gpustack.server.system_load import SystemLoadCollector
from gpustack.server.update_check import UpdateChecker
from gpustack.server.usage_buffer import flush_usage_to_db
from gpustack.server.heartbeat_buffer import flush_heartbeats_to_db
from gpustack.server.worker_instance_cleaner import WorkerInstanceCleaner
from gpustack.server.worker_syncer import WorkerSyncer
from gpustack.server.metrics_collector import GatewayMetricsCollector
from gpustack.utils.process import add_signal_handlers_in_loop
from gpustack.config.registration import write_registration_token
from gpustack.exporter.exporter import MetricExporter
from gpustack.gateway.utils import (
    model_ingress_prefix,
    model_route_ingress_prefix,
    model_route_selector,
    model_route_ingress_name,
    fallback_ingress_name,
    cleanup_ingresses,
    cleanup_selected_wasm_plugins,
    cleanup_fallback_filters,
    cleanup_ai_proxy_config,
    cleanup_mcpbridge_registry,
)
from gpustack.gateway import get_async_k8s_config
from gpustack.envs import (
    GATEWAY_PORT_CHECK_INTERVAL,
    GATEWAY_PORT_CHECK_RETRY_COUNT,
    DEFAULT_CLUSTER_KUBERNETES,
)

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, config: Config, worker_process: Process):
        self._config: Config = config
        self._sub_processes = []
        self._async_tasks = []
        self._worker_process = worker_process

    @property
    def all_processes(self):
        return self._sub_processes

    def _create_async_task(self, coro):
        self._async_tasks.append(asyncio.create_task(coro))

    @property
    def config(self):
        return self._config

    async def start(self):
        logger.info("Starting GPUStack server.")

        add_signal_handlers_in_loop()

        self._run_migrations()
        await self._prepare_data()

        init_model_catalog(self._config.model_catalog_file)
        # it's safe to determine server_role after migration
        if self._config.server_role() == Config.ServerRole.BOTH:
            self._sub_processes.append(self._worker_process)

        self._start_sub_processes()
        self._start_scheduler()
        self._start_controllers()
        self._start_system_load_collector()
        self._start_worker_syncer()
        self._start_update_checker()
        self._start_model_usage_flusher()
        self._start_heartbeat_flusher()
        self._start_worker_instance_cleaner()
        self._start_metrics_exporter()
        self._start_gateway_metrics_collector()
        self._start_query_count_logger()
        self._start_default_registry_checker()

        jwt_manager = JWTManager(self._config.jwt_secret_key)
        # Start FastAPI server
        app = create_app(self._config)
        app.state.server_config = self._config
        app.state.jwt_manager = jwt_manager
        if self._config.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self._config.allow_origins,
                allow_credentials=self._config.allow_credentials,
                allow_methods=self._config.allow_methods,
                allow_headers=self._config.allow_headers,
            )

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

        from alembic import command
        from alembic.config import Config as AlembicConfig
        import importlib.util

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
        scheduler = Scheduler(self._config)
        self._create_async_task(scheduler.start())

        logger.debug("Scheduler started.")

    def _start_controllers(self):
        model_provider_controller = ModelProviderController(self._config)
        self._create_async_task(model_provider_controller.start())

        model_route_target_controller = ModelRouteTargetController(self._config)
        self._create_async_task(model_route_target_controller.start())

        model_route_controller = ModelRouteController(self._config)
        self._create_async_task(model_route_controller.start())

        model_controller = ModelController(self._config)
        self._create_async_task(model_controller.start())

        model_instance_controller = ModelInstanceController(self._config)
        self._create_async_task(model_instance_controller.start())

        worker_controller = WorkerController(self._config)
        self._create_async_task(worker_controller.start())

        model_file_controller = ModelFileController()
        self._create_async_task(model_file_controller.start())

        cluster_controller = ClusterController(self._config)
        self._create_async_task(cluster_controller.start())

        worker_pool_controller = WorkerPoolController()
        self._create_async_task(worker_pool_controller.start())

        inference_backend_controller = InferenceBackendController()
        self._create_async_task(inference_backend_controller.start())

        logger.debug("Controllers started.")

    def _start_system_load_collector(self):
        collector = SystemLoadCollector()
        self._create_async_task(collector.start())

        logger.debug("System load collector started.")

    def _start_worker_syncer(self):
        worker_syncer = WorkerSyncer()
        self._create_async_task(worker_syncer.start())

        logger.debug("Worker syncer started.")

    def _start_model_usage_flusher(self):
        self._create_async_task(flush_usage_to_db())

        logger.debug("Model usage flusher started.")

    def _start_heartbeat_flusher(self):
        self._create_async_task(flush_heartbeats_to_db())

        logger.debug("Heartbeat flusher started.")

    def _start_worker_instance_cleaner(self):
        worker_instance_cleaner = WorkerInstanceCleaner()
        self._create_async_task(worker_instance_cleaner.start())

        logger.debug("Worker instance cleaner started.")

    def _start_gateway_metrics_collector(self):
        if self._config.gateway_mode not in [
            GatewayModeEnum.embedded,
            GatewayModeEnum.external,
        ]:
            return
        collector = GatewayMetricsCollector(cfg=self._config)

        async def _start_collector_after_port_ready():
            await self._wait_for_gateway_ready()
            await collector.start()
            logger.debug("Gateway metrics collector started.")

        self._create_async_task(_start_collector_after_port_ready())

    def _start_update_checker(self):
        if self._config.disable_update_check:
            return

        update_checker = UpdateChecker(update_check_url=self._config.update_check_url)
        self._create_async_task(update_checker.start())

        logger.debug("Update checker started.")

    def _start_sub_processes(self):
        async def start_process_after_api_ready():
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

            for process in self._sub_processes:
                process.start()

        if len(self._sub_processes) == 0:
            return
        self._create_async_task(start_process_after_api_ready())

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
        init_data_funcs = [
            self._init_user,
            self._init_default_cluster,
            self._migrate_legacy_token,
            self._migrate_legacy_workers,
            self._ensure_registration_token,
            self._cleanup_orphaned_gateway_data,
        ]
        for init_data_func in init_data_funcs:
            await init_data_func(session)

    async def _init_user(self, session: AsyncSession):
        user = await User.first_by_field(
            session=session, field="username", value="admin"
        )
        if not user:
            bootstrap_password = self._config.bootstrap_password
            require_password_change = False
            if not bootstrap_password:
                require_password_change = True
                bootstrap_password = generate_secure_password()
                bootstrap_password_file = os.path.join(
                    self._config.data_dir, "initial_admin_password"
                )
                with open(bootstrap_password_file, "w") as file:
                    file.write(bootstrap_password + "\n")
                logger.info(
                    "Generated initial admin password. "
                    f"You can get it from {bootstrap_password_file}"
                )

            user = User(
                username="admin",
                full_name="Default System Admin",
                hashed_password=get_secret_hash(bootstrap_password),
                is_admin=True,
                require_password_change=require_password_change,
            )
            await User.create(session, user)

    async def _migrate_legacy_token(self, session: AsyncSession):
        if not self._config.token:
            return
        # this should be created from sql migration script.
        cluster_user = await get_default_cluster_user(session)
        if cluster_user is None or cluster_user.cluster is None:
            logger.debug(
                "Default cluster user not exist, skipping legacy token migration."
            )
            return

        default_cluster = cluster_user.cluster
        if not default_cluster:
            logger.debug(
                "Default cluster does not exist, skipping legacy token migration."
            )
            return
        if default_cluster.registration_token != "":
            return
        try:
            default_cluster.registration_token = self._config.token
            await default_cluster.update(session=session, auto_commit=False)

            default_cluster_user = await User.one_by_fields(
                session=session,
                fields={
                    "cluster_id": default_cluster.id,
                    "is_system": True,
                    "role": UserRole.Cluster,
                },
            )
            if default_cluster_user is None:
                raise RuntimeError("Default cluster user does not exist.")
            if len(default_cluster_user.api_keys) > 0:
                raise RuntimeError(
                    "Default cluster user already has API keys, cannot migrate legacy token."
                )

            new_key = ApiKey(
                name="Legacy Cluster Token",
                access_key="",
                hashed_secret_key=get_secret_hash(self._config.token),
                user_id=default_cluster_user.id,
                user=default_cluster_user,
            )
            await ApiKey.create(session, new_key, auto_commit=False)
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to migrate legacy token: {e}")
            session.rollback()
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
        worker_ids = [worker.id for worker in workers]
        worker_users = await User.all_by_fields(
            session=session,
            fields={
                "cluster_id": default_cluster.id,
                "is_system": True,
                "role": UserRole.Worker,
            },
            extra_conditions=[User.worker_id.in_(worker_ids)],
        )
        user_by_worker_id = {user.worker_id: user for user in worker_users}
        for worker in workers:
            try:
                worker_user = user_by_worker_id.get(worker.id, None)
                if not worker_user:
                    to_create_user = User(
                        username=f'{system_name_prefix}-{worker.id}',
                        is_system=True,
                        role=UserRole.Worker,
                        hashed_password="",
                        cluster=default_cluster,
                        cluster_id=default_cluster.id,
                        worker=worker,
                        worker_id=worker.id,
                    )
                    worker_user = await User.create(
                        session=session, source=to_create_user, auto_commit=False
                    )
                    access_key = secrets.token_hex(8)
                    secret_key = secrets.token_hex(16)
                    to_create_apikey = ApiKey(
                        name=worker_user.username,
                        access_key=access_key,
                        hashed_secret_key=get_secret_hash(secret_key),
                        user=worker_user,
                        user_id=worker_user.id,
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
                session.rollback()
                raise e

    async def _ensure_registration_token(self, session: AsyncSession):
        cluster_user = await get_default_cluster_user(session)
        if cluster_user is None or cluster_user.cluster is None:
            logger.debug(
                "Default cluster user not exist, skipping registration token generation."
            )
            return
        token = cluster_user.cluster.registration_token
        if token == "":
            try:
                access_key = secrets.token_hex(8)
                secret_key = secrets.token_hex(16)
                new_key = ApiKey(
                    name="Default Cluster Token",
                    access_key=access_key,
                    hashed_secret_key=get_secret_hash(secret_key),
                    user_id=cluster_user.id,
                    user=cluster_user,
                )
                await ApiKey.create(session, new_key, auto_commit=False)
                token = f"{API_KEY_PREFIX}_{access_key}_{secret_key}"
                cluster = cluster_user.cluster
                await cluster.update(
                    session=session,
                    source={"registration_token": token},
                    auto_commit=False,
                )
                await session.commit()
            except Exception as e:
                logger.error(f"Failed to ensure registration token: {e}")
                session.rollback()
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
        fallback_route_ids = [
            ep.route_id
            for ep in route_targets
            if ep.fallback_status_codes is not None
            and len(ep.fallback_status_codes) > 0
        ]
        expected_names = [
            model_route_ingress_name(model_route.id) for model_route in model_routes
        ]
        expected_names.extend(
            [
                fallback_ingress_name(model_route_ingress_name(id))
                for id in fallback_route_ids
            ]
        )
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
        await cleanup_selected_wasm_plugins(
            namespace=self.config.gateway_namespace,
            expected_names=expected_names,
            config=k8s_config,
            extra_labels=model_route_selector,
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
        await cleanup_mcpbridge_registry(
            providers=providers,
            namespace=self.config.gateway_namespace,
            model_instances=model_instances,
            workers=workers,
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
        default_cluster_user = await get_default_cluster_user(session)
        if default_cluster_user:
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
        )
        default_cluster = await Cluster.create(
            session, default_cluster, auto_commit=False
        )

        default_cluster_user = User(
            username=default_cluster_user_name,
            is_system=True,
            is_admin=False,
            require_password_change=False,
            role=UserRole.Cluster,
            hashed_password="",
            cluster=default_cluster,
        )
        await User.create(session, default_cluster_user, auto_commit=False)

        await session.commit()
        logger.debug("Default cluster created.")

    async def user_defined_default_cluster(self, session: AsyncSession) -> Cluster:
        cluster = await Cluster.first_by_field(
            session=session, field="is_default", value=True
        )
        return cluster
