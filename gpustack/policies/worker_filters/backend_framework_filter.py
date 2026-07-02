import logging
from typing import Dict, List, Tuple, Optional

from gpustack.policies.base import WorkerFilter
from gpustack.schemas.models import Model, get_backend, BackendEnum
from gpustack.schemas.workers import Worker
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    VersionConfig,
)
from gpustack.server.db import async_session
from gpustack_runner import list_service_runners
from gpustack_runtime.detector.ascend import get_ascend_cann_variant
from gpustack_runtime.detector import ManufacturerEnum


logger = logging.getLogger(__name__)


class BackendFrameworkFilter(WorkerFilter):
    """
    Filter workers based on whether the inference_backend corresponding to the model's backend
    supports the worker's runtime_framework.
    """

    def __init__(self, model: Model):
        self.model = model
        self.backend_name = get_backend(model)

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
        if not query_conditions or self.model.cpu_offloading:
            query_conditions.add(("cpu", None, self.model.backend_version, None))
        return list(query_conditions)

    def _visible_version_configs(
        self,
        inference_backends: List[InferenceBackend],
    ) -> Dict[str, VersionConfig]:
        """
        Version configs of this model's backend visible under the Hybrid
        scope: the Platform row (owner_principal_id IS NULL) merged with the
        model owner's row, whose keys override the Platform ones. Rows owned
        by other principals are ignored.
        """
        # getattr, NOT direct access: the scheduler deploys persisted
        # Model rows, but the evaluator drives this same filter through
        # find_candidate with a ModelSpec (compatibility checks before a
        # model exists), which carries no owner_principal_id — pydantic
        # raises AttributeError on direct access. Unowned specs fall back
        # to Platform-only visibility.
        owner_id = getattr(self.model, "owner_principal_id", None)
        platform_row = None
        owner_row = None
        for b in inference_backends:
            if b.backend_name != self.backend_name:
                continue
            if b.owner_principal_id is None:
                platform_row = b
            elif b.owner_principal_id == owner_id:
                owner_row = b
        merged: Dict[str, VersionConfig] = {}
        for row in (platform_row, owner_row):
            if row and row.version_configs and row.version_configs.root:
                merged.update(row.version_configs.root)
        return merged

    async def _has_supported_runners(
        self,
        inference_backends: List[InferenceBackend],
        gpu_type: str,
        backend_version: Optional[str],
        variant: Optional[str],
    ) -> bool:
        """
        Check whether a supported runner exists for the given GPU configuration.

        Args:
            gpu_type: GPU type (cuda, rocm, cann)
            backend_version: Inference Backend version (e.g., "0.11.0")
            variant: Variant for Ascend GPUs (CANN version)

        Returns:
            True if a supported runner exists, False otherwise.
        """
        version_configs = self._visible_version_configs(inference_backends)
        for version, version_config in version_configs.items():
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
                or gpu_type in version_config.built_in_frameworks
            )

            # GPU is supported if either custom or built-in framework supports it
            if is_custom_supported or is_built_in_supported:
                return True

        kwargs = {
            "backend": gpu_type,
            "service": self.backend_name.lower(),
        }
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

        return False

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

        async with async_session() as session:
            inference_backends = await InferenceBackend.all(session)

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
                    inference_backends,
                    gpu_type,
                    backend_version,
                    variant,
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
                        reason_text = "No backend versions are available for "
                        if gpu_type == "cpu":
                            reason_text += "CPU device"
                        else:
                            reason_text += (
                                f"GPU device ({gpu_type} {runtime_version or ''})"
                            )
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
