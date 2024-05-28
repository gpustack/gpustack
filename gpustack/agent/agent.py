import asyncio
import os
import multiprocessing
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

from gpustack.agent.config import AgentConfig
from gpustack.agent.logs import LogOptionsDep
from gpustack.agent.node_manager import NodeManager
from gpustack.agent.task_manager import TaskManager
from gpustack.logging import setup_logging
from gpustack.utils import run_periodically_async
from gpustack.generated_client.client import Client
from gpustack.agent.logs import log_generator


logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, cfg: AgentConfig):
        client = Client(base_url=cfg.server)
        self._node_manager = NodeManager(node_ip=cfg.node_ip, client=client)
        self._task_manager = TaskManager(server_url=cfg.server, log_dir=cfg.log_dir)

        self._log_dir = cfg.log_dir
        self._address = "0.0.0.0"
        self._port = 10050

    def start(self):
        asyncio.run(self.start_async())

    async def start_async(self):
        """
        Start the agent.
        """

        logger.info("Starting GPUStack agent.")

        # Report the node status to the server periodically.
        run_periodically_async(self._node_manager.sync_node_status, 5 * 60)

        # watch tasks and handle them.
        asyncio.create_task(self._watch_tasks())

        # Start the agent server to expose APIs.
        await self._serve_apis()

    async def _watch_tasks(self):
        logger.info("Start watching tasks.")

        while True:
            await self._do_watch_tasks()
            await asyncio.sleep(5)  # rewatch tasks if it fails

    async def _do_watch_tasks(self):
        loop = asyncio.get_running_loop()
        pool = multiprocessing.Pool()
        try:
            await loop.run_in_executor(None, self._task_manager.watch_tasks, pool)
        except Exception as e:
            logger.error(f"Error watching tasks: {e}")
        finally:
            pool.terminate()

    async def _serve_apis(self):
        """
        Start the agent server to expose APIs.
        """

        app = FastAPI(title="GPUStackAgent", response_model_exclude_unset=True)

        @app.get("/taskLogs/{task_id}")
        async def get_task_logs(task_id: int, log_options: LogOptionsDep):
            path = f"{self._log_dir}/tasks/{task_id}.log"
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
