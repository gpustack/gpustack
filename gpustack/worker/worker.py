import asyncio
import os
import logging
import socket
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import setproctitle
import uvicorn
from pathlib import Path

from gpustack.config import Config
from gpustack.schemas.workers import SystemReserved, WorkerUpdate
from gpustack.server import catalog
from gpustack.utils import file, platform
from gpustack.utils.network import get_first_non_loopback_ip
from gpustack.client import ClientSet
from gpustack.logging import setup_logging
from gpustack.utils.process import add_signal_handlers_in_loop
from gpustack.utils.task import run_periodically_in_thread
from gpustack.worker.logs import LogOptionsDep
from gpustack.worker.serve_manager import ServeManager
from gpustack.worker.exporter import MetricExporter
from gpustack.worker.logs import log_generator
from gpustack.worker.tools_manager import ToolsManager
from gpustack.worker.worker_manager import WorkerManager


logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, cfg: Config):
        self._config = cfg
        self._log_dir = cfg.log_dir
        self._address = "0.0.0.0"
        self._port = cfg.worker_port
        self._exporter_enabled = not cfg.disable_metrics
        self._enable_worker_ip_monitor = False
        self._system_reserved = SystemReserved(ram=0, vram=0)

        if cfg.system_reserved is not None:
            # GB to Bytes
            self._system_reserved.ram = (
                (cfg.system_reserved.get("ram") or cfg.system_reserved.get("memory", 2))
                * 1024
                * 1024
                * 1024
            )
            self._system_reserved.vram = (
                (
                    cfg.system_reserved.get("vram")
                    or cfg.system_reserved.get("gpu_memory", 0)
                )
                * 1024
                * 1024
                * 1024
            )

        self._worker_ip = cfg.worker_ip
        if self._worker_ip is None:
            self._worker_ip = get_first_non_loopback_ip()
            self._enable_worker_ip_monitor = True

        self._worker_name = cfg.worker_name
        if self._worker_name is None:
            self._worker_name = self._get_worker_name()

        self._clientset = ClientSet(
            base_url=cfg.server_url,
            username=f"system/worker/{self._worker_name}",
            password=cfg.token,
        )
        self._worker_manager = WorkerManager(
            worker_ip=self._worker_ip,
            worker_name=self._worker_name,
            system_reserved=self._system_reserved,
            clientset=self._clientset,
            cfg=cfg,
        )
        self._serve_manager = ServeManager(
            clientset=self._clientset,
            worker_name=self._worker_name,
            cfg=cfg,
        )
        self._exporter = MetricExporter(
            worker_ip=self._worker_ip,
            worker_name=self._worker_name,
            port=cfg.metrics_port,
            clientset=self._clientset,
            cfg=cfg,
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

    def start(self, is_multiprocessing=False):
        setup_logging(self._config.debug)

        if is_multiprocessing:
            setproctitle.setproctitle("gpustack_worker")

        tools_manager = ToolsManager(
            tools_download_base_url=self._config.tools_download_base_url,
            pipx_path=self._config.pipx_path,
            device=self.get_device_by_gpu_devices(),
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

    def get_device_by_gpu_devices(self) -> Optional[str]:
        gpu_devices = self._config.get_gpu_devices()
        if gpu_devices:
            vendor = gpu_devices[0].vendor
            return platform.device_type_from_vendor(vendor)
        return None

    async def start_async(self):
        """
        Start the worker.
        """

        logger.info("Starting GPUStack worker.")

        add_signal_handlers_in_loop()

        if self._exporter_enabled:
            # Start the metric exporter with retry.
            run_periodically_in_thread(self._exporter.start, 15)

        if self._enable_worker_ip_monitor:
            # Check worker ip change every 15 seconds.
            run_periodically_in_thread(self._check_worker_ip_change, 15)

        # Report the worker node status to the server every 30 seconds.
        run_periodically_in_thread(self._worker_manager.sync_worker_status, 30)

        if not self._config.disable_rpc_servers:
            # Start rpc server instances with restart.
            run_periodically_in_thread(self._worker_manager.start_rpc_servers, 20, 3)

        # Check serving model instances' health every 3 seconds.
        run_periodically_in_thread(
            self._serve_manager.health_check_serving_instances, 3
        )

        asyncio.create_task(self._serve_manager.watch_model_instances())

        # Start the worker server to expose APIs.
        await self._serve_apis()

    async def _serve_apis(self):
        """
        Start the worker server to expose APIs.
        """

        app = FastAPI(title="GPUStack Worker", response_model_exclude_unset=True)

        @app.get("/serveLogs/{id}")
        async def get_serve_logs(id: int, log_options: LogOptionsDep):
            path = Path(self._log_dir) / "serve" / f"{id}.log"

            try:
                file.check_file_with_retries(path)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Log file not found")

            return StreamingResponse(
                log_generator(path, log_options), media_type="text/plain"
            )

        @app.get("/healthz")
        async def healthz():
            return "ok"

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

    def _check_worker_ip_change(self):
        """
        Detect if the worker IP has changed. If so, delete legacy model
        instances so they can be recreated with the new worker IP.
        """

        worker = None
        workers = self._clientset.workers.list(params={"name": self._worker_name})
        if workers is not None and len(workers.items) != 0:
            worker = workers.items[0]

        current_ip = get_first_non_loopback_ip()
        if current_ip != self._worker_ip:
            logger.info(f"Worker IP changed from {self._worker_ip} to {current_ip}")
            if worker is None:
                raise Exception(f"Worker {self._worker_name} not found")

            self.update_worker_ip(worker, current_ip)
        elif worker and current_ip != worker.ip:
            logger.info(f"Worker IP changed from {worker.ip} to {current_ip}")
            self.update_worker_ip(worker, current_ip)

    def update_worker_ip(self, worker, current_ip: str):
        self._worker_ip = current_ip
        self._worker_manager._worker_ip = current_ip
        self._exporter._worker_ip = current_ip

        worker_update: WorkerUpdate = WorkerUpdate.model_validate(worker)
        worker_update.ip = current_ip
        self._clientset.workers.update(worker.id, worker_update)

        for instance in self._clientset.model_instances.list(
            params={"worker_id": worker.id}
        ).items:
            self._clientset.model_instances.delete(instance.id)
