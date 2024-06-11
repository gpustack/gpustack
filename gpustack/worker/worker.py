import asyncio
import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import setproctitle
import uvicorn

from gpustack.config import Config
from gpustack.worker.logs import LogOptionsDep
from gpustack.worker.node_manager import NodeManager
from gpustack.worker.serve_manager import ServeManager
from gpustack.client import ClientSet
from gpustack.logging import setup_logging
from gpustack.utils import run_periodically_async
from gpustack.worker.exporter import MetricExporter
from gpustack.worker.logs import log_generator


logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, cfg: Config):
        clientset = ClientSet(base_url=cfg.server_url)
        self._node_manager = NodeManager(node_ip=cfg.node_ip, clientset=clientset)
        self._serve_manager = ServeManager(
            server_url=cfg.server_url, log_dir=cfg.log_dir, clientset=clientset
        )

        self._log_dir = cfg.log_dir
        self._address = "0.0.0.0"
        self._port = 10050
        self._exporter_enabled = cfg.enable_metrics
        self._exporter = MetricExporter(node_ip=cfg.node_ip, port=cfg.metrics_port)

    def start(self, is_multiprocessing=False):
        if is_multiprocessing:
            setproctitle.setproctitle("gpustack_worker")

        asyncio.run(self.start_async())

    async def start_async(self):
        """
        Start the worker.
        """

        logger.info("Starting GPUStack worker.")

        # Start the metric exporter.
        if self._exporter_enabled:
            asyncio.create_task(self._exporter.start())

        # Report the node status to the server periodically.
        run_periodically_async(self._node_manager.sync_node_status, 5 * 60)

        # watch model instances and handle them.
        asyncio.create_task(self._serve_model_instances())

        # Start the worker server to expose APIs.
        await self._serve_apis()

    async def _serve_model_instances(self):
        logger.info("Start serving model instances.")

        while True:
            await self._do_serve_model_instances()
            await asyncio.sleep(5)  # rewatch if it fails

    async def _do_serve_model_instances(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._serve_manager.watch_model_instances)
        await loop.run_in_executor(None, self._serve_manager.monitor_processes)

    async def _serve_apis(self):
        """
        Start the worker server to expose APIs.
        """

        app = FastAPI(title="GPUStack Worker", response_model_exclude_unset=True)

        @app.get("/serveLogs/{id}")
        async def get_serve_logs(id: int, log_options: LogOptionsDep):
            path = f"{self._log_dir}/serve/{id}.log"
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail="Logs not found")

            return StreamingResponse(
                log_generator(path, log_options), media_type="text/plain"
            )

        config = uvicorn.Config(
            app,
            host=self._address,
            port=self._port,
            access_log=False,
            log_level="error",
        )

        setup_logging()
        logger.info(f"Serving worker APIs on {config.host}:{config.port}.")
        server = uvicorn.Server(config)
        await server.serve()
