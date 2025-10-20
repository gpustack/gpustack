import asyncio
from multiprocessing import Process
import os
import re
from typing import List
import uvicorn
import logging
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.logging import setup_logging
from gpustack.schemas.users import User
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.workers import Worker
from gpustack.security import (
    JWTManager,
    generate_secure_password,
    get_secret_hash,
    API_KEY_PREFIX,
)
from gpustack.server.app import create_app
from gpustack.config import Config
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
from gpustack.server.worker_syncer import WorkerSyncer
from gpustack.utils.process import add_signal_handlers_in_loop
from gpustack.utils.task import run_periodically_in_thread
from gpustack.worker.registration import write_registration_token
from gpustack.exporter.exporter import MetricExporter

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
        self._start_metrics_exporter()

        port = 80
        if self._config.port:
            port = self._config.port
        elif self._config.ssl_certfile and self._config.ssl_keyfile:
            port = 443
        host = "0.0.0.0"
        if self._config.host:
            host = self._config.host

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
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            access_log=False,
            log_level="error",
            ssl_certfile=self._config.ssl_certfile,
            ssl_keyfile=self._config.ssl_keyfile,
        )

        setup_logging()

        logger.info(f"Serving on {config.host}:{config.port}.")
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
        async with AsyncSession(engine) as session:
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

        cluster_controller = ClusterController()
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

        # Start the metric exporter with retry.
        run_periodically_in_thread(exporter.start, 15, 5)

    @staticmethod
    def _setup_data_dir(data_dir: str):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    async def _init_data(self, session: AsyncSession):
        init_data_funcs = [
            self._init_user,
            self._migrate_legacy_token,
            self._init_default_cluster_token,
            self._ensure_registration_token,
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
        """
        Migrate legacy tokens to the new format.
        This is a placeholder for future migration logic.
        """
        if not self._config.token or self._config.token.startswith(API_KEY_PREFIX):
            return
        default_worker = await User.first_by_field(
            session=session, field="username", value="system/worker-0"
        )
        existing_key = await ApiKey.first_by_field(
            session=session, field="user_id", value=default_worker.id
        )
        if existing_key:
            logger.info("Legacy token already migrated.")
        else:
            new_key = ApiKey(
                name="Legacy Worker Token",
                access_key="",
                hashed_secret_key=get_secret_hash(self._config.token),
                user_id=default_worker.id,
                user=default_worker,
            )
            await ApiKey.create(session, new_key)
        fields = {
            "deleted_at": None,
            "token": None,
        }
        legacy_workers = await Worker.all_by_fields(session, fields)
        for worker in legacy_workers:
            worker.token = self._config.token
            await worker.update(session)

    async def _init_default_cluster_token(self, session: AsyncSession):
        """
        Initialize the default cluster token.
        """
        cluster_user = await User.first_by_field(
            session=session, field="username", value="system/cluster-1"
        )
        # the cluster_user is created in the migration, so it should always exist
        if not cluster_user or not cluster_user.cluster_id:
            return
        existing_key = await ApiKey.first_by_field(
            session=session, field="user_id", value=cluster_user.id
        )
        if existing_key:
            logger.info("Default cluster token already exists.")
            return
        tokens = cluster_user.cluster.registration_token.split("_", 2)
        access_key = tokens[1]
        secret_key = tokens[2]
        new_key = ApiKey(
            name="Default Cluster Token",
            access_key=access_key,
            hashed_secret_key=get_secret_hash(secret_key),
            user=cluster_user,
            user_id=cluster_user.id,
        )
        await ApiKey.create(session, new_key)

    async def _ensure_registration_token(self, session: AsyncSession):
        cluster_user = await User.first_by_field(
            session=session, field="username", value="system/cluster-1"
        )
        if not cluster_user or not cluster_user.cluster:
            logger.info("Cluster doesn't exist, skipping writing registration token.")
            return
        write_registration_token(
            data_dir=self._config.data_dir,
            token=cluster_user.cluster.registration_token,
        )
