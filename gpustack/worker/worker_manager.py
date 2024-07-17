import platform
import logging

from gpustack.api.exceptions import (
    AlreadyExistsException,
)

from gpustack.client import ClientSet
from gpustack.schemas.workers import SystemReserved, Worker, WorkerStateEnum
from gpustack.worker.collector import WorkerStatusCollector

logger = logging.getLogger(__name__)


class WorkerManager:
    def __init__(
        self,
        worker_ip: str,
        worker_name: str,
        system_reserved: SystemReserved,
        clientset: ClientSet,
    ):
        self._registration_completed = False
        self._worker_name = worker_name
        self._worker_ip = worker_ip
        self._clientset = clientset
        self._system_reserved = system_reserved

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
            clientset=self._clientset,
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

        current = result.items[0]
        worker.id = current.id
        worker.state = WorkerStateEnum.READY
        worker.system_reserved = self._system_reserved

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
                clientset=self._clientset,
            )
            worker = collector.collect()

            os_info = platform.uname()
            arch_info = platform.machine()

            worker.system_reserved = self._system_reserved
            worker.labels = {
                "os": os_info.system,
                "arch": arch_info,
            }

            return worker
        except Exception as e:
            logger.error(f"Failed to initialize worker: {e}")
            return

    def _register_shutdown_hooks(self):
        pass
