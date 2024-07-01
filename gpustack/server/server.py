import asyncio
import atexit
from multiprocessing import Process
import os
from typing import List
import uvicorn
import logging
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.logging import setup_logging
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.security import generate_secure_password, get_secret_hash
from gpustack.server.app import app
from gpustack.config import Config
from gpustack.server.controller import ModelController
from gpustack.server.db import get_engine, init_db
from gpustack.scheduler.scheduler import Scheduler
from gpustack.server.data_collector import DataCollector


logger = logging.getLogger(__name__)


class Server:
    def __init__(self, config: Config, sub_processes: List[Process] = None):
        if sub_processes is None:
            sub_processes = []
        self._config: Config = config
        self._sub_processes = sub_processes

        atexit.register(self.at_exit)

    @property
    def all_processes(self):
        return self._sub_processes

    @property
    def config(self):
        return self._config

    async def start(self):
        logger.info("Starting GPUStack server.")

        self._run_migrations()
        await self._prepare_data()

        self._start_sub_processes()
        self._start_scheduler()
        self._start_controllers()
        self._start_system_load_collector()

        port = 80
        if self._config.port:
            port = self._config.port
        elif self._config.ssl_certfile and self._config.ssl_keyfile:
            port = 443
        host = "0.0.0.0"
        if self._config.host:
            host = self._config.host

        # Start FastAPI server
        app.state.server_config = self._config
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
        await server.serve()

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
        alembic_cfg.set_main_option("sqlalchemy.url", self._config.database_url)
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migration completed.")

    async def _prepare_data(self):
        self._setup_data_dir(self._config.data_dir)

        await init_db(self._config.database_url)

        engine = get_engine()
        async with AsyncSession(engine) as session:
            await self._init_data(session)

        logger.debug("Data initialization completed.")

    def _start_scheduler(self):
        scheduler = Scheduler(system_reserved=self._config.system_reserved)
        asyncio.create_task(scheduler.start())

        logger.debug("Scheduler started.")

    def _start_controllers(self):
        controller = ModelController()
        asyncio.create_task(controller.start())

        logger.debug("Controller started.")

    def _start_system_load_collector(self):
        collector = DataCollector()
        asyncio.create_task(collector.start())

        logger.debug("System load collector started.")

    def _start_sub_processes(self):
        for process in self._sub_processes:
            process.start()

    @staticmethod
    def _setup_data_dir(data_dir: str):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    async def _init_data(self, session: AsyncSession):
        init_data_funcs = [self._init_model, self._init_user]
        for init_data_func in init_data_funcs:
            await init_data_func(session)

    async def _init_model(self, session: AsyncSession):
        if not self._config.serve_default_models:
            return

        data = [
            {
                "name": "Qwen1.5-0.5B-Chat",
                "source": "huggingface",
                "huggingface_repo_id": "Qwen/Qwen1.5-0.5B-Chat-GGUF",
                "huggingface_filename": "qwen1_5-0_5b-chat-q5_k_m.gguf",
            },
            {
                "name": "Llama-3-8B-Instruct",
                "source": "huggingface",
                "huggingface_repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2",
                "huggingface_filename": "Meta-Llama-3-8B-Instruct-v2.Q5_K_M.gguf",
            },
        ]

        default_models: List[Model] = [Model(**item_data) for item_data in data]

        for model in default_models:
            existing = await Model.first_by_field(session, "name", model.name)
            if existing:
                continue
            await Model.create(session, model)

        logger.debug("Created default models.")

    async def _init_user(self, session: AsyncSession):
        user = await User.first_by_field(
            session=session, field="username", value="admin"
        )
        if not user:
            bootstrap_password = self._config.bootstrap_password
            if not bootstrap_password:
                bootstrap_password = generate_secure_password()
                bootstrap_password_file = os.path.join(
                    self._config.data_dir, "initial_admin_password"
                )
                with open(bootstrap_password_file, "w") as file:
                    file.write(bootstrap_password)
                logger.info(
                    "Generated initial admin password. "
                    f"You can get it from {bootstrap_password_file}."
                )

            user = User(
                username="admin",
                full_name="Default System Admin",
                hashed_password=get_secret_hash(bootstrap_password),
                is_admin=True,
                require_password_change=True,
            )
            await User.create(session, user)

    def at_exit(self):
        logger.info("Stopping GPUStack server.")
        for process in self._sub_processes:
            process.terminate()
        logger.info("Stopped all processes.")
