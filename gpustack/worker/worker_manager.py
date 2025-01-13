from datetime import datetime, timezone
import multiprocessing
import os
import logging
from typing import Dict

import psutil

from gpustack.api.exceptions import (
    AlreadyExistsException,
)

from gpustack.client import ClientSet
from gpustack.config.config import Config
from gpustack.detectors.custom.custom import Custom
from gpustack.schemas.workers import (
    SystemReserved,
    Worker,
)
from gpustack.utils import network
from gpustack.utils import platform
from gpustack.utils.process import terminate_process_tree
from gpustack.worker.collector import WorkerStatusCollector
from gpustack.worker.rpc_server import RPCServer, RPCServerProcessInfo
from gpustack.detectors.detector_factory import DetectorFactory


logger = logging.getLogger(__name__)


class WorkerManager:
    def __init__(
        self,
        worker_ip: str,
        worker_name: str,
        system_reserved: SystemReserved,
        clientset: ClientSet,
        cfg: Config,
    ):
        self._registration_completed = False
        self._worker_name = worker_name
        self._worker_ip = worker_ip
        self._worker_port = cfg.worker_port
        self._clientset = clientset
        self._system_reserved = system_reserved
        self._rpc_servers: Dict[int, RPCServerProcessInfo] = {}
        self._rpc_server_log_dir = f"{cfg.log_dir}/rpc_server"
        self._gpu_devices = cfg.get_gpu_devices()

        os.makedirs(self._rpc_server_log_dir, exist_ok=True)

    def sync_worker_status(self):
        """
        Should be called periodically to sync the worker node status with the server.
        It registers the worker node with the server if necessary.
        """

        # Register the worker node with the server.
        self.register_with_server()
        self._update_worker_status()

    def _update_worker_status(self):
        collector = WorkerStatusCollector(
            worker_ip=self._worker_ip,
            worker_name=self._worker_name,
            worker_port=self._worker_port,
            clientset=self._clientset,
            worker_manager=self,
            gpu_devices=self._gpu_devices,
        )

        try:
            worker = collector.collect()
            result = self._clientset.workers.list(params={"name": self._worker_name})
        except Exception as e:
            logger.error(f"Failed to update status for worker {self._worker_name}: {e}")
            return

        if result is None or len(result.items) == 0:
            logger.error(f"Worker {self._worker_name} not found")
            return

        now = datetime.now(timezone.utc).replace(microsecond=0)

        current = result.items[0]
        worker.name = current.name
        worker.id = current.id
        worker.labels = current.labels
        worker.state = current.state
        worker.unreachable = current.unreachable
        worker.system_reserved = self._system_reserved
        worker.heartbeat_time = now

        try:
            result = self._clientset.workers.update(id=current.id, model_update=worker)
        except Exception as e:
            logger.error(f"Failed to update worker {self._worker_name} status: {e}")

    def register_with_server(self):
        if self._registration_completed:
            return

        worker = self._initialize_worker()
        if worker is None:
            return
        self._register_worker(worker)
        self._registration_completed = True

    def _register_worker(self, worker: Worker):
        logger.info(
            f"Registering worker: {worker.name}",
        )

        try:
            self._clientset.workers.create(worker)
        except AlreadyExistsException:
            logger.debug(f"Worker {worker.name} already exists, skip registration.")
            return
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            return

        logger.info(f"Worker {worker.name} registered.")

    def _initialize_worker(self):
        try:
            collector = WorkerStatusCollector(
                worker_ip=self._worker_ip,
                worker_name=self._worker_name,
                worker_port=self._worker_port,
                clientset=self._clientset,
                worker_manager=self,
                gpu_devices=self._gpu_devices,
            )
            worker = collector.collect()

            worker.system_reserved = self._system_reserved
            worker.labels = {
                "os": platform.system(),
                "arch": platform.arch(),
            }

            return worker
        except Exception as e:
            logger.error(f"Failed to initialize worker: {e}")
            return

    def _register_shutdown_hooks(self):
        pass

    def start_rpc_servers(self):
        try:
            self._start_rpc_servers()
        except Exception as e:
            logger.error(f"Failed to start rpc servers: {e}")
            return

    def _start_rpc_servers(self):
        try:
            detector_factory = (
                DetectorFactory("custom", {"custom": [Custom(self._gpu_devices)]})
                if self._gpu_devices
                else DetectorFactory()
            )
            gpu_devices = detector_factory.detect_gpus()
        except Exception as e:
            logger.error(f"Failed to get GPU devices while start rpc servers: {e}")
            return

        for gpu_device in gpu_devices:
            if gpu_device.index is None:
                logger.warning(
                    f"GPU device {gpu_device.name} does not have an index. Skipping start rpc server."
                )
                continue

            current = self._rpc_servers.get(gpu_device.index)
            if current:
                if current.process.is_alive():
                    continue

                pid = current.process.pid
                logger.warning(
                    f"RPC server for GPU {gpu_device.index} is not running, pid {pid}, restarting."
                )
                try:
                    terminate_process_tree(pid)
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    logger.error(f"Failed to terminate process {pid}: {e}")
                self._rpc_servers.pop(gpu_device.index)

            log_file_path = f"{self._rpc_server_log_dir}/gpu-{gpu_device.index}.log"
            port = network.get_free_port(start=50000, end=51024)
            process = multiprocessing.Process(
                target=RPCServer.start,
                args=(port, gpu_device.index, gpu_device.vendor, log_file_path),
            )

            process.daemon = True
            process.start()

            self._rpc_servers[gpu_device.index] = RPCServerProcessInfo(
                process=process, port=port, gpu_index=gpu_device.index
            )
            logger.info(
                f"Started RPC server for GPU {gpu_device.index} on port {port}, pid {process.pid}"
            )

    def get_rpc_servers(self) -> Dict[int, RPCServerProcessInfo]:
        return self._rpc_servers
