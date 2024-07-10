import asyncio
import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import setproctitle
import uvicorn

from gpustack.config import Config
from gpustack.worker.logs import LogOptionsDep
from gpustack.worker.worker_manager import WorkerManager
from gpustack.worker.serve_manager import ServeManager
from gpustack.client import ClientSet
from gpustack.logging import setup_logging
from gpustack.utils.task import run_periodically_in_thread
from gpustack.worker.exporter import MetricExporter
from gpustack.worker.logs import log_generator


logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, cfg: Config):
        clientset = ClientSet(
            base_url=cfg.server_url,
            username=f"system/worker/{cfg.worker_ip}",
            password=cfg.token,
        )
        self._worker_manager = WorkerManager(
            worker_ip=cfg.worker_ip, clientset=clientset
        )
        self._serve_manager = ServeManager(
            server_url=cfg.server_url,
            clientset=clientset,
            log_dir=cfg.log_dir,
            data_dir=cfg.data_dir,
        )

        self._log_dir = cfg.log_dir
        self._address = "0.0.0.0"
        self._port = cfg.worker_port
        self._exporter_enabled = cfg.enable_metrics
        self._exporter = MetricExporter(
            worker_ip=cfg.worker_ip, port=cfg.metrics_port, clientset=clientset
        )

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

        # Report the worker node status to the server every 30 seconds.
        run_periodically_in_thread(
            self._worker_manager.sync_worker_status,
            interval=30,
            initial_delay=2,
        )
        # Monitor the processes of model instances every 60 seconds.
        run_periodically_in_thread(self._serve_manager.monitor_processes, 60)
        # Watch model instances with retry.
        run_periodically_in_thread(self._serve_manager.watch_model_instances, 5)

        # Start the worker server to expose APIs.
        await self._serve_apis()

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
        try:
            await server.serve()
        finally:
            logger.info("Worker stopped.")
