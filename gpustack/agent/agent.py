import asyncio
import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import setproctitle
import uvicorn

from gpustack.agent.config import AgentConfig
from gpustack.agent.logs import LogOptionsDep
from gpustack.agent.node_manager import NodeManager
from gpustack.agent.serve_manager import ServeManager
from gpustack.client import ClientSet
from gpustack.logging import setup_logging
from gpustack.utils import run_periodically_async
from gpustack.agent.exporter import MetricExporter
from gpustack.agent.logs import log_generator


logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, cfg: AgentConfig):
        clientset = ClientSet(base_url=cfg.server)
        self._node_manager = NodeManager(node_ip=cfg.node_ip, clientset=clientset)
        self._serve_manager = ServeManager(
            server_url=cfg.server, log_dir=cfg.log_dir, clientset=clientset
        )

        self._log_dir = cfg.log_dir
        self._address = "0.0.0.0"
        self._port = 10050
        self._exporter_enabled = cfg.metric_enabled
        self._exporter = MetricExporter(node_ip=cfg.node_ip, port=cfg.metrics_port)

    def start(self, is_multiprocessing=False):
        if is_multiprocessing:
            setproctitle.setproctitle("gpustack_agent")

        asyncio.run(self.start_async())

    async def start_async(self):
        """
        Start the agent.
        """

        logger.info("Starting GPUStack agent.")

        # Start the metric exporter.
        if self._exporter_enabled:
            asyncio.create_task(self._exporter.start())

        # Report the node status to the server periodically.
        run_periodically_async(self._node_manager.sync_node_status, 5 * 60)

        # watch model instances and handle them.
        asyncio.create_task(self._serve_model_instances())

        # Start the agent server to expose APIs.
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
        Start the agent server to expose APIs.
        """

        app = FastAPI(title="GPUStackAgent", response_model_exclude_unset=True)

        @app.get("/serveLogs/{id}")
        async def get_serve_logs(id: int, log_options: LogOptionsDep):
            path = f"{self._log_dir}/serve/{id}.log"
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail="Logs not found")

            return StreamingResponse(
                log_generator(path, log_options), media_type="text/plain"
            )

        @app.get("/")
        async def debug():
            return {"message": "Hello from agent"}

        config = uvicorn.Config(
            app,
            host=self._address,
            port=self._port,
            access_log=False,
            log_level="error",
        )

        setup_logging()
        logger.info(f"Serving agent APIs on {config.host}:{config.port}.")
        server = uvicorn.Server(config)
        await server.serve()
