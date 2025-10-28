import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
import os
import logging
import socket
from typing import Optional

import aiohttp
from fastapi import FastAPI
import setproctitle
import tenacity
import uvicorn

from gpustack.api import exceptions
from gpustack.config import Config
from gpustack.config.envs import TCP_CONNECTOR_LIMIT
from gpustack.routes import debug, probes
from gpustack.routes.worker import logs, proxy
from gpustack.server import catalog
from gpustack.utils.network import get_first_non_loopback_ip, get_ifname_by_ip
from gpustack.client import ClientSet
from gpustack.logging import setup_logging
from gpustack.utils.process import add_signal_handlers_in_loop
from gpustack.utils.system_check import check_glibc_version
from gpustack.utils.task import run_periodically_in_thread
from gpustack.worker.inference_backend_manager import InferenceBackendManager
from gpustack.worker.model_file_manager import ModelFileManager
from gpustack.worker.runtime_metrics_aggregator import RuntimeMetricsAggregator
from gpustack.worker.serve_manager import ServeManager
from gpustack.worker.exporter import MetricExporter
from gpustack.worker.tools_manager import ToolsManager
from gpustack.worker.worker_manager import WorkerManager
from gpustack.worker.collector import WorkerStatusCollector
from gpustack.config.registration import read_worker_token

logger = logging.getLogger(__name__)


class Worker:
    _clientset: ClientSet
    _register_clientset: ClientSet
    _status_collector: WorkerStatusCollector
    _worker_manager: WorkerManager
    _config: Config
    _worker_ip: Optional[str] = None
    _worker_ifname: Optional[str] = None
    _worker_id: Optional[int] = None

    def worker_ip(self) -> str:
        return self._config.worker_ip or self._worker_ip

    def worker_ifname(self) -> str:
        return self._config.worker_ifname or self._worker_ifname

    def worker_id(self) -> int:
        return self._worker_id

    def clientset(self) -> ClientSet:
        return self._clientset

    def __init__(self, cfg: Config, is_embedded: bool = False):
        self._config = cfg
        self._is_embedded = is_embedded
        self._log_dir = cfg.log_dir
        self._address = "0.0.0.0"
        self._exporter_enabled = not cfg.disable_worker_metrics
        self._async_tasks = []
        self._worker_ip, self._worker_ifname = get_first_non_loopback_ip()

        self._worker_name = cfg.worker_name
        if self._worker_name is None:
            self._worker_name = self._get_worker_name()
        self._runtime_metrics_cache = defaultdict()

        self._status_collector = WorkerStatusCollector(
            cfg=cfg,
            worker_ip_getter=self.worker_ip,
            worker_ifname_getter=self.worker_ifname,
            worker_id_getter=self.worker_id,
        )

        self._worker_manager = WorkerManager(
            cfg=cfg,
            is_embedded=is_embedded,
            collector=self._status_collector,
            worker_name=self._worker_name,
        )

        self._exporter = MetricExporter(
            cfg=cfg,
            worker_name=self._worker_name,
            collector=self._status_collector,
            worker_ip_getter=self.worker_ip,
            worker_id_getter=self.worker_id,
            clientset_getter=self.clientset,
            cache=self._runtime_metrics_cache,
        )

    def _get_worker_name(self):
        # Hostname might change with the network, so we store the worker name in a file.
        # It avoids creating multiple workers for the same node.
        # This is useful when running standalone on a PC.
        worker_name_path = os.path.join(self._config.data_dir, "worker_name")
        if os.path.exists(worker_name_path):
            with open(worker_name_path, "r") as file:
                worker_name = file.read().strip()
        else:
            worker_name = socket.gethostname()
            with open(worker_name_path, "w") as file:
                file.write(worker_name)

        return worker_name

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_fixed(2),
        reraise=True,
        before_sleep=lambda retry_state: logger.debug(
            f"Retrying to get worker ID (attempt {retry_state.attempt_number}) due to: {retry_state.outcome.exception()}"
        ),
    )
    def _register(self):
        self._clientset = self._worker_manager.register_with_server()
        # Worker ID is available after the worker registration.
        worker_list = self._clientset.workers.list(
            params={"me": 'true'},
        )
        if len(worker_list.items) != 1:
            raise Exception(f"Worker {self._worker_name} not registered.")
        self._worker_id = worker_list.items[0].id

    def _create_async_task(self, coro):
        self._async_tasks.append(asyncio.create_task(coro))

    def start(self):
        setup_logging(self._config.debug)

        if self._is_embedded:
            setproctitle.setproctitle("gpustack_worker")

        check_glibc_version()

        tools_manager = ToolsManager(
            tools_download_base_url=self._config.tools_download_base_url,
            pipx_path=self._config.pipx_path,
            data_dir=self._config.data_dir,
            bin_dir=self._config.bin_dir,
        )
        tools_manager.prepare_tools()
        catalog.prepare_chat_templates(self._config.data_dir)

        try:
            asyncio.run(self.start_async())
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.error(f"Error serving worker APIs: {e}")
        finally:
            logger.info("Worker has shut down.")

    async def start_async(self):
        """
        Start the worker.
        """

        logger.info("Starting GPUStack worker.")

        add_signal_handlers_in_loop()

        self._register()
        if self._exporter_enabled:
            # Start the runtime metrics cacher.
            _runtime_metrics_aggregator = RuntimeMetricsAggregator(
                cache=self._runtime_metrics_cache,
                worker_id_getter=self.worker_id,
                clientset=self._clientset,
            )
            run_periodically_in_thread(_runtime_metrics_aggregator.aggregate, 3, 30)

            # Start the metric exporter with retry.
            run_periodically_in_thread(self._exporter.start, 15)

        # Monitor the ip change, if not fixed.
        if not self._config.worker_ip:
            # Check worker ip change every 15 seconds.
            run_periodically_in_thread(self._check_worker_ip_change, 15)
        # Fill the worker ifname if not set.
        elif not self._config.worker_ifname:
            self._worker_ifname = get_ifname_by_ip(self._config.worker_ip)

        # Report the worker node status to the server every 30 seconds.
        run_periodically_in_thread(self._worker_manager.sync_worker_status, 30)

        # Start the worker server to expose APIs.
        self._create_async_task(self._serve_apis())

        inference_backend_manager = InferenceBackendManager(self._clientset)
        # Start InferenceBackend listener to cache backend data
        self._create_async_task(inference_backend_manager.start_listener())

        serve_manager = ServeManager(
            worker_id=self._worker_id,
            clientset=self._clientset,
            cfg=self._config,
            inference_backend_manager=inference_backend_manager,
        )
        # Check serving model instances' health every 3 seconds.
        run_periodically_in_thread(serve_manager.sync_model_instances_state, 3)
        self._create_async_task(serve_manager.watch_model_instances_event())
        self._create_async_task(serve_manager.watch_model_instances())

        model_file_manager = ModelFileManager(
            worker_id=self._worker_id, clientset=self._clientset, cfg=self._config
        )
        self._create_async_task(model_file_manager.watch_model_files())

        await asyncio.gather(*self._async_tasks)

    async def _serve_apis(self):
        """
        Start the worker server to expose APIs.
        """

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            connector = aiohttp.TCPConnector(
                limit=TCP_CONNECTOR_LIMIT,
                force_close=True,
            )
            app.state.http_client = aiohttp.ClientSession(
                connector=connector, trust_env=True
            )
            yield
            await app.state.http_client.close()

        app = FastAPI(
            title="GPUStack Worker",
            response_model_exclude_unset=True,
            lifespan=lifespan,
        )
        app.state.config = self._config
        app.state.token = read_worker_token(self._config.data_dir)

        app.include_router(debug.router, prefix="/debug")
        app.include_router(probes.router)
        app.include_router(logs.router)
        app.include_router(proxy.router)
        exceptions.register_handlers(app)

        config = uvicorn.Config(
            app,
            host=self._address,
            port=self._config.worker_port,
            access_log=False,
            log_level="error",
        )

        setup_logging()
        logger.info(f"Serving worker APIs on {config.host}:{config.port}.")
        server = uvicorn.Server(config)

        await server.serve()

    def _check_worker_ip_change(self):
        """
        Detect if the worker IP has changed. If so, delete legacy model
        instances so they can be recreated with the new worker IP.
        """

        new_ip, new_ifname = get_first_non_loopback_ip()
        old_ip, old_ifname = self.worker_ip(), self.worker_ifname()
        if new_ip == old_ip and new_ifname == old_ifname:
            return

        logger.info(
            f"Worker IP changed from {old_ip}({old_ifname}) to {new_ip}{new_ifname}"
        )
        self._worker_ip = new_ip
        self._worker_ifname = new_ifname
        self._worker_manager.sync_worker_status()

        for instance in self._clientset.model_instances.list(
            params={"worker_id": str(self._worker_id)}
        ).items:
            self._clientset.model_instances.delete(instance.id)
