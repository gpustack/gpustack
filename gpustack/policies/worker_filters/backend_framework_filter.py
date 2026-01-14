import logging
from typing import List, Tuple, Optional

from gpustack.policies.base import WorkerFilter
from gpustack.schemas.models import Model, get_backend, BackendEnum
from gpustack.schemas.workers import Worker
from gpustack.schemas.inference_backend import (
    InferenceBackend,
)
from gpustack.server.db import get_engine
from gpustack_runner import list_service_runners
from gpustack_runtime.deployer.__utils__ import compare_versions
from gpustack_runtime.detector.ascend import get_ascend_cann_variant
from gpustack_runtime.detector import ManufacturerEnum
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
        self._engine = get_engine()

    def _get_gpu_query_conditions(
        self, worker: Worker
    ) -> List[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
        """
        Extract query conditions from worker's GPU devices.

        Args:
            worker: Worker to extract GPU information from

        Returns:
            List of tuples (gpu_type, runtime_version, backend_version, variant)
        """
        query_conditions = set()
        if worker.status and worker.status.gpu_devices:
            for gpu in worker.status.gpu_devices:
                variant = None
                if gpu.vendor == ManufacturerEnum.ASCEND and gpu.arch_family:
                    variant = get_ascend_cann_variant(gpu.arch_family).lower()
                query_conditions.add(
                    (gpu.type, gpu.runtime_version, self.model.backend_version, variant)
                )
        if not query_conditions:
            query_conditions.add(("cpu", None, self.model.backend_version, None))
        return list(query_conditions)

    async def _has_lower_runners(self, **kwargs) -> bool:
        backend_version = kwargs.get("backend_version")
        if backend_version:
            kwargs.pop("backend_version")
        # Since backend versions are backward compatible,
        # if an exact version match cannot be found, we can try to see if a lower version is available.
        runners_list = list_service_runners(**kwargs)
        for runner in runners_list:
            if not runner.versions or len(runner.versions) == 0:
                continue
            try:
                runner_version = runner.versions[0].backends[0].versions[0].version
                if compare_versions(runner_version, backend_version) <= 0:
                    return True
            except Exception:
                pass
        return False

    async def _has_supported_runners(
        self,
        gpu_type: str,
        runtime_version: Optional[str],
        backend_version: Optional[str],
        variant: Optional[str],
    ) -> bool:
        """
        Get supported runner versions for given GPU configuration.

        Args:
            gpu_type: GPU type (cuda, rocm, cann)
            runtime_version: GPU runtime version (e.g., "12.4")
            backend_version: Inference Backend version (e.g., "0.11.0")
            variant: Variant for Ascend GPUs (CANN version)

        Returns:
            True if any runner is compatible with given GPU configuration, False otherwise
        """

        async with AsyncSession(self._engine) as session:
            statement = select(InferenceBackend).where(
                InferenceBackend.backend_name == self.backend_name
            )
            result = await session.exec(statement)
            backend = result.first()

            if backend and backend.version_configs and backend.version_configs.root:
                for version, version_config in backend.version_configs.root.items():
                    if backend_version and backend_version != version:
                        continue

                    # Check if gpu_type is supported by custom_framework
                    is_custom_supported = version_config.custom_framework and (
                        version_config.custom_framework == "cpu"
                        or gpu_type == version_config.custom_framework
                    )

                    # Check if gpu_type is supported by built_in_frameworks
                    is_built_in_supported = version_config.built_in_frameworks and (
                        "cpu" in version_config.built_in_frameworks
                        and gpu_type in version_config.built_in_frameworks
                    )

                    # GPU is supported if either custom or built-in framework supports it
                    if is_custom_supported or is_built_in_supported:
                        return True

        kwargs = {
            "backend": gpu_type,
            "service": self.backend_name.lower(),
        }
        if runtime_version:
            kwargs["backend_version"] = runtime_version
        if variant:
            kwargs["backend_variant"] = variant

        # If model does not specify backend_version, exclude deprecated runners
        if not self.model.backend_version:
            kwargs["with_deprecated"] = False

        # If model specifies backend_version, use it as service_version filter
        if backend_version:
            kwargs["service_version"] = backend_version

        runners_list = list_service_runners(**kwargs)
        if runners_list and len(runners_list) > 0:
            return True

        return await self._has_lower_runners(**kwargs)

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter workers based on backend framework and version compatibility.
        Try using each GPU type and version from the input workers to query for available runners.

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

        if self.model.backend == BackendEnum.CUSTOM:
            return workers, []

        filtered_workers = []
        filtered_messages = []

        # Check if model has specified backend_version
        has_backend_version = self.model.backend_version is not None

        for worker in workers:
            # Get and deduplicate query conditions from worker's GPU devices
            query_conditions = self._get_gpu_query_conditions(worker)

            # Check if any GPU condition is compatible
            is_compatible = False
            incompatible_reasons = []

            for gpu_type, runtime_version, backend_version, variant in query_conditions:
                # Check framework compatibility

                # Get supported runners for this GPU configuration
                is_supported = await self._has_supported_runners(
                    gpu_type, runtime_version, backend_version, variant
                )

                # Check version compatibility
                if has_backend_version:
                    # Mode 1: Version matching - check if GPU supports the specified backend_version
                    if is_supported:
                        is_compatible = True
                        logger.debug(
                            f"Worker {worker.name} supports backend version {self.model.backend_version}"
                        )
                        break
                    else:
                        incompatible_reasons.append(
                            f"Worker {worker.name} does not support backend version {self.model.backend_version} or the backend version not exists. "
                        )
                else:
                    # Mode 2: Auto matching - check if there are any available backend versions
                    if is_supported:
                        is_compatible = True
                        break
                    else:
                        reason_text = f"GPU {gpu_type} (runtime: {runtime_version}, variant: {variant}) "
                        if gpu_type == "cpu":
                            reason_text = "CPU device "
                        reason_text += "has no available backend versions"
                        incompatible_reasons.append(reason_text)

            if is_compatible:
                filtered_workers.append(worker)
            else:
                reason = "; ".join(incompatible_reasons)
                filtered_messages.append(f"Worker {worker.name} filtered out: {reason}")

        if filtered_messages:
            logger.info(
                f"BackendFrameworkCompatibilityFilter: {len(filtered_messages)} workers filtered out"
            )

        return filtered_workers, filtered_messages
