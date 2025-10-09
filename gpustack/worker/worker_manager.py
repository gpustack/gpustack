import multiprocessing
import os
import logging
from typing import Dict, Optional

import psutil

from gpustack.client import ClientSet
from gpustack.client.worker_manager_clients import (
    WorkerStatusClient,
    WorkerRegistrationClient,
)
from gpustack.config.config import Config
from gpustack.detectors.custom.custom import Custom
from gpustack.schemas.workers import (
    WorkerCreate,
    WorkerUpdate,
)
from gpustack.utils import network
from gpustack.utils import platform
from gpustack.utils.process import terminate_process_tree
from gpustack.worker.collector import WorkerStatusCollector
from gpustack.worker.rpc_server import RPCServer, RPCServerProcessInfo
from gpustack.worker.registration import registration_client
from gpustack.detectors.detector_factory import DetectorFactory
from gpustack.utils.profiling import time_decorator
from gpustack.security import API_KEY_PREFIX
from gpustack.utils.uuid import get_legacy_uuid

logger = logging.getLogger(__name__)


class WorkerManager:
    _is_embedded: bool
    _collector: WorkerStatusCollector
    _clientset: Optional[ClientSet] = None
    _registration_client: WorkerRegistrationClient
    _status_client: WorkerStatusClient
    # worker name is used for logging
    _worker_name: str

    def __init__(
        self,
        cfg: Config,
        is_embedded: bool,
        collector: WorkerStatusCollector,
        worker_name: str,
    ):
        self._is_embedded = is_embedded
        self._cfg = cfg
        self._collector = collector
        self._worker_name = worker_name
        self._rpc_servers: Dict[int, RPCServerProcessInfo] = {}
        self._rpc_server_log_dir = f"{cfg.log_dir}/rpc_server"
        self._rpc_server_cache_dir = f"{cfg.cache_dir}/rpc_server/"
        self._rpc_server_args = cfg.rpc_server_args
        os.makedirs(self._rpc_server_log_dir, exist_ok=True)

        if self._cfg.token and self._cfg.server_url:
            self._prepare_clients(self._cfg.token)

    def _prepare_clients(self, token: str):
        if self._clientset is not None and self._status_client is not None:
            return
        if not token.startswith(API_KEY_PREFIX):
            legacy_uuid = get_legacy_uuid(self._cfg.data_dir)
            if not legacy_uuid:
                raise ValueError(
                    "Legacy UUID not found, please re-register the worker."
                )
            token = f"{API_KEY_PREFIX}_{legacy_uuid}_{token}"
        self._clientset = ClientSet(
            base_url=self._cfg.server_url,
            api_key=token,
        )
        self._status_client = WorkerStatusClient(self._clientset.http_client)

    def sync_worker_status(self):
        """
        Should be called periodically to sync the worker node status with the server.
        It registers the worker node with the server if necessary.
        """
        if self._status_client is None:
            return
        try:
            workerStatus = self._collector.collect(self._clientset)
        except Exception as e:
            logger.error(f"Failed to collect status for worker: {e}")
            return
        try:
            self._status_client.create(workerStatus)
        except Exception as e:
            logger.error(f"Failed to update worker status: {e}")

    def register_with_server(self) -> ClientSet:
        # If the worker has been registered, self._clientset should be valid.
        # the clientset is built in WorkerManager.__init__ if cfg.token is set.
        if self._clientset:
            return self._clientset
        try:
            token = self._register_worker()
            with open(os.path.join(self._cfg.data_dir, 'token'), 'w') as f:
                f.write(token + "\n")
            self._prepare_clients(token)
            return self._clientset
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            raise

    def _register_worker(self) -> str:
        logger.info(
            f"Registering worker: {self._worker_name}",
        )
        self._registration_client = registration_client(self._cfg, self._is_embedded)
        external_id = None
        external_id_path = os.path.join(self._cfg.data_dir, 'external_id')
        if os.path.exists(external_id_path):
            with open(os.path.join(self._cfg.data_dir, 'external_id'), 'r') as f:
                external_id = f.read()

        @time_decorator
        def timed_collect():
            return self._collector.collect(initial=True)

        workerStatus = timed_collect()
        workerUpdate = WorkerUpdate(
            name=self._worker_name,
            labels=ensure_builtin_labels(self._worker_name),
        )
        to_register = WorkerCreate.model_validate(
            {
                **workerStatus.model_dump(),
                **workerUpdate.model_dump(),
                "external_id": external_id,
            }
        )
        created = self._registration_client.create(to_register)
        logger.info(
            f"Worker {self._worker_name} registered with worker_id {created.id}."
        )
        return created.token

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
                DetectorFactory(
                    "custom", {"custom": [Custom(self._collector.gpu_devices)]}
                )
                if self._collector.gpu_devices
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
            port = network.get_free_port(
                port_range=self._cfg.rpc_server_port_range,
                unavailable_ports=self.get_occupied_ports(),
            )
            process = multiprocessing.Process(
                target=RPCServer.start,
                args=(
                    port,
                    gpu_device.index,
                    gpu_device.vendor,
                    log_file_path,
                    self._rpc_server_cache_dir,
                    self._cfg.bin_dir,
                    self._rpc_server_args,
                ),
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

    def get_occupied_ports(self) -> set[int]:
        return {server.port for server in self._rpc_servers.values()}


def ensure_builtin_labels(worker_name: str) -> dict:
    return {
        "os": platform.system(),
        "arch": platform.arch(),
        "worker-name": worker_name,
    }
