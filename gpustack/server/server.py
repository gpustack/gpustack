import asyncio
from multiprocessing import Process
import os
import re
from typing import List
import uvicorn
import logging
import secrets
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.logging import setup_logging
from gpustack.schemas.users import User, UserRole, system_name_prefix
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster, ClusterProvider, ClusterStateEnum
from gpustack.schemas.models import Model
from gpustack.security import (
    JWTManager,
    generate_secure_password,
    get_secret_hash,
    API_KEY_PREFIX,
)
from gpustack.server.app import create_app
from gpustack.config.config import Config, GatewayModeEnum
from gpustack.server.catalog import init_model_catalog
from gpustack.server.controllers import (
    ModelController,
    ModelFileController,
    ModelInstanceController,
    WorkerController,
    ClusterController,
    WorkerPoolController,
    InferenceBackendController,
)
from gpustack.server.db import get_engine, init_db
from gpustack.scheduler.scheduler import Scheduler
from gpustack.server.system_load import SystemLoadCollector
from gpustack.server.update_check import UpdateChecker
from gpustack.server.usage_buffer import flush_usage_to_db
from gpustack.server.worker_instance_cleaner import WorkerInstanceCleaner
from gpustack.server.worker_syncer import WorkerSyncer
from gpustack.server.metrics_collector import GatewayMetricsCollector
from gpustack.utils.process import add_signal_handlers_in_loop
from gpustack.config.registration import write_registration_token
from gpustack.exporter.exporter import MetricExporter
from gpustack.gateway.utils import cleanup_orphaned_model_ingresses

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, config: Config, sub_processes: List[Process] = None):
        if sub_processes is None:
            sub_processes = []
        self._config: Config = config
        self._sub_processes = sub_processes
        self._async_tasks = []

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

        self._start_sub_processes()
        self._start_scheduler()
        self._start_controllers()
        self._start_system_load_collector()
        self._start_worker_syncer()
        self._start_update_checker()
        self._start_model_usage_flusher()
        self._start_worker_instance_cleaner()
        self._start_metrics_exporter()
        self._start_gateway_metrics_collector()

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
        ssl_keyfile = None
        ssl_certfile = None
        if self._config.gateway_mode == GatewayModeEnum.disabled:
            ssl_keyfile = self._config.ssl_keyfile
            ssl_certfile = self._config.ssl_certfile

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self._config.get_api_port(),
            access_log=False,
            log_level="error",
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
        )

        setup_logging()

        logger.info(f"Serving on {config.host}:{config.port}.")
        logger.info(f"Gateway mode: {self._config.gateway_mode.value}.")
        if self._config.gateway_mode == GatewayModeEnum.embedded:
            logger.info(
                f"Embedded gateway will serve on {self._config.get_gateway_port()}."
            )
            if self._config.get_tls_secret_name() is not None:
                logger.info(
                    f"Embedded gateway will serve TLS on port {self._config.tls_port}."
                )
        server = uvicorn.Server(config)
        self._create_async_task(server.serve())

        await asyncio.gather(*self._async_tasks)

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

        db_url = self._config.database_url
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

        await init_db(self._config.database_url)

        engine = get_engine()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            await self._init_data(session)

        logger.debug("Data initialization completed.")

    def _start_scheduler(self):
        scheduler = Scheduler(self._config)
        self._create_async_task(scheduler.start())

        logger.debug("Scheduler started.")

    def _start_controllers(self):
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

    def _start_worker_instance_cleaner(self):
        worker_instance_cleaner = WorkerInstanceCleaner()
        self._create_async_task(worker_instance_cleaner.start())

        logger.debug("Worker instance cleaner started.")

    def _start_gateway_metrics_collector(self):
        if self._config.gateway_mode != GatewayModeEnum.embedded:
            return
        collector = GatewayMetricsCollector(cfg=self._config)

        self._create_async_task(collector.start())
        logger.debug("Gateway metrics collector started.")

    def _start_update_checker(self):
        if self._config.disable_update_check:
            return

        update_checker = UpdateChecker(update_check_url=self._config.update_check_url)
        self._create_async_task(update_checker.start())

        logger.debug("Update checker started.")

    def _start_sub_processes(self):
        for process in self._sub_processes:
            process.start()

    def _start_metrics_exporter(self):
        if self._config.disable_metrics:
            return

        exporter = MetricExporter(cfg=self._config)
        self._create_async_task(exporter.generate_metrics_cache())
        self._create_async_task(exporter.start())

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

    async def _init_default_cluster(self, session: AsyncSession):
        cluster_count = await Cluster.count(session)
        if cluster_count == 0:
            hashed_suffix = secrets.token_hex(6)
            default_cluster = Cluster(
                name="Default Cluster",
                description="The default cluster for GPUStack",
                provider=ClusterProvider.Docker,
                state=ClusterStateEnum.READY,
                hashed_suffix=hashed_suffix,
                registration_token="",
            )
            default_cluster = await Cluster.create(
                session, default_cluster, auto_commit=False
            )

            default_cluster_user = User(
                username=f"system/cluster-{default_cluster.id}",
                is_system=True,
                is_admin=False,
                require_password_change=False,
                role=UserRole.Cluster,
                hashed_password="",
                cluster=default_cluster,
            )
            await User.create(session, default_cluster_user, auto_commit=False)

            if default_cluster.id != 1:
                # _migrate_legacy_token handles legacy token for cluster with ID 1.
                # For other cluster IDs, we create an API key like normal.
                access_key = secrets.token_hex(8)
                secret_key = secrets.token_hex(16)
                to_create_apikey = ApiKey(
                    name=f'{system_name_prefix}-{hashed_suffix}',
                    access_key=access_key,
                    hashed_secret_key=get_secret_hash(secret_key),
                    user=default_cluster_user,
                )
                await ApiKey.create(session, to_create_apikey, auto_commit=False)

            await session.commit()
            logger.debug("Default cluster created.")

    async def _migrate_legacy_token(self, session: AsyncSession):
        """
        Migrate legacy tokens to the new format.
        This is a placeholder for future migration logic.
        """
        if not self._config.token or self._config.token.startswith(API_KEY_PREFIX):
            return
        default_cluster = await Cluster.one_by_id(session=session, id=1)
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
        cluster_user = await User.first_by_field(
            session=session, field="username", value="system/cluster-1"
        )
        if not cluster_user or not cluster_user.cluster:
            logger.info("Cluster doesn't exist, skipping writing registration token.")
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
        # Remove the orphaned ingresses of model
        models = await Model.all_by_field(
            session=session, field="deleted_at", value=None
        )
        model_ids = [model.id for model in models]
        k8s_config = self.config.get_async_k8s_config()
        await cleanup_orphaned_model_ingresses(
            namespace=self.config.get_gateway_namespace(),
            existing_model_ids=model_ids,
            config=k8s_config,
        )
