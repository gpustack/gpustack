import os
import logging
from typing import Optional

from gpustack.client import ClientSet
from gpustack.client.worker_manager_clients import (
    WorkerStatusClient,
    WorkerRegistrationClient,
)
from gpustack.config.config import Config
from gpustack.schemas.workers import (
    WorkerCreate,
    WorkerUpdate,
)
from gpustack.utils import platform
from gpustack.worker.collector import WorkerStatusCollector
from gpustack.config.registration import (
    registration_client,
    read_worker_token,
    write_worker_token,
)
from gpustack.utils.profiling import time_decorator

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
        worker_token = read_worker_token(self._cfg.data_dir)
        if worker_token:
            self._prepare_clients(worker_token)

    def _prepare_clients(self, token: str):
        if self._clientset is not None and self._status_client is not None:
            return
        self._clientset = ClientSet(
            base_url=self._cfg.get_server_url(),
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
        # the clientset is built in WorkerManager.__init__ if cfg._worker_token is stored.
        if self._clientset:
            return self._clientset
        try:
            token = self._register_worker()
            write_worker_token(self._cfg.data_dir, token)
            self._prepare_clients(token)
            return self._clientset
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            raise

    def _register_worker(self) -> str:
        logger.info(
            f"Registering worker: {self._worker_name}",
        )
        self._registration_client = registration_client(
            data_dir=self._cfg.data_dir,
            server_url=self._cfg.get_server_url(),
            registration_token=self._cfg.token,
            wait_token_file=self._is_embedded,
        )
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


def ensure_builtin_labels(worker_name: str) -> dict:
    return {
        "os": platform.system(),
        "arch": platform.arch(),
        "worker-name": worker_name,
    }
