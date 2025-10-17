import logging
from typing import List, Tuple

from gpustack.policies.base import WorkerFilter
from gpustack.schemas.models import Model, get_backend
from gpustack.schemas.workers import Worker
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    is_built_in_backend,
)
from gpustack.server.db import get_engine
from gpustack_runner import list_service_runners
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession


logger = logging.getLogger(__name__)


class BackendFrameworkFilter(WorkerFilter):
    """
    Filter workers based on whether the inference_backend corresponding to the model's backend
    supports the worker's runtime_framework.
    """

    def __init__(self, model: Model):
        self.model = model
        self.backend_name = get_backend(model)

    async def _get_backend_supported_frameworks(self, backend_name: str) -> List[str]:
        """
        Get supported frameworks for a given backend.
        This method supports both built-in backends (via gpustack-runner) and custom backends (via database).

        Args:
            backend_name: The name of the backend service

        Returns:
            List of supported framework names
        """
        supported_frameworks = []

        # First, try to get frameworks from database (for both built-in and custom backends)
        try:
            engine = get_engine()
            async with AsyncSession(engine) as session:
                statement = select(InferenceBackend).where(
                    InferenceBackend.backend_name == backend_name
                )
                result = await session.exec(statement)
                backend = result.first()

                if backend and backend.version_configs and backend.version_configs.root:
                    for version_config in backend.version_configs.root.values():
                        # For custom backends, use custom_framework
                        if version_config.custom_framework:
                            supported_frameworks.append(version_config.custom_framework)
                        # For built-in backends, use built_in_frameworks
                        elif version_config.built_in_frameworks:
                            supported_frameworks.extend(
                                version_config.built_in_frameworks
                            )

                    # Remove duplicates while preserving order
                    supported_frameworks = list(dict.fromkeys(supported_frameworks))

        except Exception as e:
            logger.warning(
                f"Failed to get frameworks from database for backend {backend_name}: {e}"
            )

        if is_built_in_backend(backend_name):
            runners_list = list_service_runners(service=backend_name.lower())

            if runners_list and len(runners_list) > 0:
                for version in runners_list[0].versions:
                    if version.version and version.backends:
                        for backend_runner in version.backends:
                            if backend_runner.backend:
                                supported_frameworks.append(backend_runner.backend)

            # Remove duplicates while preserving order
            supported_frameworks = list(dict.fromkeys(supported_frameworks))

        return supported_frameworks

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter workers based on backend framework compatibility.

        Args:
            workers: List of workers to filter

        Returns:
            Tuple of (filtered_workers, filter_messages)
        """
        if not self.backend_name:
            logger.warning(
                "Could not determine backend for model, skipping framework compatibility filter"
            )
            return workers, []

        # Get supported frameworks for the model's backend
        supported_frameworks = await self._get_backend_supported_frameworks(
            self.backend_name
        )

        if not supported_frameworks:
            logger.info(
                f"No framework restrictions found for backend {self.backend_name}, allowing all workers"
            )
            return workers, []

        filtered_workers = []
        filtered_messages = []

        for worker in workers:
            # Get all runtime frameworks from worker's GPU devices
            worker_frameworks = set()
            worker_frameworks.add("cpu")
            if worker.status and worker.status.gpu_devices:
                for gpu_device in worker.status.gpu_devices:
                    worker_frameworks.add(gpu_device.type)

            # Check if any worker framework is supported by the backend
            compatible_frameworks = worker_frameworks.intersection(
                set(supported_frameworks)
            )

            if compatible_frameworks:
                filtered_workers.append(worker)
            else:
                worker_frameworks_str = (
                    ", ".join(sorted(worker_frameworks))
                    if worker_frameworks
                    else "none"
                )
                supported_frameworks_str = ", ".join(sorted(supported_frameworks))
                filtered_messages.append(
                    f"Worker {worker.name} filtered out: runtime frameworks [{worker_frameworks_str}] "
                    f"not compatible with backend {self.backend_name} supported frameworks [{supported_frameworks_str}]"
                )

        if filtered_messages:
            logger.info(
                f"BackendFrameworkCompatibilityFilter: {len(filtered_messages)} workers filtered out"
            )

        return filtered_workers, filtered_messages
