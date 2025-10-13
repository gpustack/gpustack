import asyncio
import logging
import os
from typing import Dict, List, Optional

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
)
from gpustack.policies.event_recorder.recorder import EventCollector
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
    get_local_model_weight_size,
)
from gpustack.schemas.models import (
    CategoryEnum,
    ComputedResourceClaim,
    Model,
    SourceEnum,
)
from gpustack.schemas.workers import Worker
from gpustack.config import Config
from gpustack.utils.hub import get_model_weight_size
from gpustack.utils.unit import byte_to_gib

logger = logging.getLogger(__name__)

EVENT_ACTION_DEFAULT = "default_scheduling_msg"
EVENT_ACTION_RESOURCE_ESTIMATION = "custom_backend_resource_estimation_msg"


async def estimate_custom_backend_vram(
    model: Model, token: Optional[str] = None
) -> int:
    """
    Estimate the VRAM requirement in bytes for custom backends.

    Formula:
        VRAM = WEIGHT * 1.2 + FRAMEWORK_FOOTPRINT

    This follows the same approach as vLLM but with configurable framework overhead.
    """
    if model.env and 'GPUSTACK_MODEL_VRAM_CLAIM' in model.env:
        # Use as a potential workaround if the empirical vram estimation is far beyond the expected value.
        return int(model.env['GPUSTACK_MODEL_VRAM_CLAIM'])

    # Custom backends may have different framework overhead
    # Default to a conservative estimate
    framework_overhead = (
        1 * 1024**3  # 1 GiB for most custom backends
        if not model.categories or CategoryEnum.LLM in model.categories
        else 256 * 1024**2  # 256 MiB for non-LLM models
    )

    weight_size = 0
    timeout_in_seconds = 15

    try:
        if (
            model.source == SourceEnum.HUGGING_FACE
            or model.source == SourceEnum.MODEL_SCOPE
        ):
            weight_size = await asyncio.wait_for(
                asyncio.to_thread(get_model_weight_size, model, token),
                timeout=timeout_in_seconds,
            )
        elif model.source == SourceEnum.LOCAL_PATH and os.path.exists(model.local_path):
            weight_size = get_local_model_weight_size(model.local_path)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout when getting weight size for model {model.name}")
    except Exception as e:
        logger.warning(f"Cannot get weight size for model {model.name}: {e}")

    # Reference: https://blog.eleuther.ai/transformer-math/#total-inference-memory
    return int(weight_size * 1.2 + framework_overhead)


class CustomBackendResourceFitSelector(ScheduleCandidatesSelector):
    """
    Resource fit selector for custom backends.

    This selector:
    - Estimates resource requirements for custom backends
    - Finds suitable workers based on resource availability
    - Supports both GPU and CPU-only deployments
    """

    def __init__(self, cfg: Config, model: Model):
        self._cfg = cfg
        self._model = model
        self._event_collector = EventCollector(model)
        self._messages = []

        # Estimated resource requirements
        self._vram_claim = 0
        self._ram_claim = 0

        # Whether the backend supports CPU-only inference
        self._supports_cpu_only = getattr(model, 'supports_cpu_only', True)

    def get_messages(self) -> List[str]:
        """Get scheduling messages."""
        return self._messages

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement.
        """
        # Estimate VRAM requirements using actual model weight
        self._vram_claim = await estimate_custom_backend_vram(
            self._model, self._cfg.huggingface_token
        )

        # Estimate RAM requirements (conservative estimate)
        self._ram_claim = max(
            int(self._vram_claim * 0.1), 2 * 1024**3
        )  # At least 2GB RAM

        logger.info(
            f"Calculated resource claim for model {self._model.readable_source}, "
            f"VRAM claim: {self._vram_claim}, RAM claim: {self._ram_claim}"
        )

        # Set default message
        default_msg = f"The model requires approximately {byte_to_gib(self._vram_claim)} GiB of VRAM."
        self._messages.append(default_msg)

        # Try different candidate selection strategies
        candidate_functions = [
            self._find_single_worker_single_gpu_candidates,
            self._find_single_worker_multi_gpu_candidates,
        ]

        # Add CPU-only candidates if supported
        if self._supports_cpu_only:
            candidate_functions.append(self._find_cpu_only_candidates)

        for candidate_func in candidate_functions:
            logger.debug(
                f"Custom backend for model {self._model.readable_source}, "
                f"trying candidate selector: {candidate_func.__name__}"
            )

            candidates = await candidate_func(workers)
            if candidates:
                return candidates

        # No suitable candidates found
        self._add_no_candidates_message()
        return []

    async def _find_single_worker_single_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates using a single GPU on a single worker.
        """
        candidates = []

        for worker in workers:
            if not worker.status or not worker.status.gpu_devices:
                continue

            allocatable = await get_worker_allocatable_resource(worker)

            for gpu_device in worker.status.gpu_devices:
                gpu_index = gpu_device.index

                # Check if GPU has enough VRAM
                if gpu_index in allocatable.vram:
                    available_vram = allocatable.vram[gpu_index]

                    if available_vram >= self._vram_claim:
                        # Check RAM requirement
                        if allocatable.ram >= self._ram_claim:
                            candidate = self._create_single_gpu_candidate(
                                worker, [gpu_index], {gpu_index: self._vram_claim}
                            )
                            candidates.append(candidate)

        return candidates

    async def _find_single_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates using multiple GPUs on a single worker.
        """
        candidates = []

        for worker in workers:
            if not worker.status or not worker.status.gpu_devices:
                continue

            if len(worker.status.gpu_devices) < 2:
                continue  # Need at least 2 GPUs for multi-GPU

            allocatable = await get_worker_allocatable_resource(worker)

            # Try to distribute VRAM across multiple GPUs
            available_gpus = []
            total_available_vram = 0

            for gpu_device in worker.status.gpu_devices:
                gpu_index = gpu_device.index
                if gpu_index in allocatable.vram:
                    available_vram = allocatable.vram[gpu_index]
                    available_gpus.append((gpu_index, available_vram))
                    total_available_vram += available_vram

            # Check if total VRAM is sufficient
            if total_available_vram >= self._vram_claim and len(available_gpus) >= 2:
                # Check RAM requirement
                if allocatable.ram >= self._ram_claim:
                    # Distribute VRAM evenly across GPUs
                    gpu_indexes = [gpu[0] for gpu in available_gpus]
                    vram_per_gpu = self._vram_claim // len(gpu_indexes)
                    vram_distribution = {idx: vram_per_gpu for idx in gpu_indexes}

                    candidate = self._create_multi_gpu_candidate(
                        worker, gpu_indexes, vram_distribution
                    )
                    candidates.append(candidate)

        return candidates

    async def _find_cpu_only_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates for CPU-only inference.
        """
        candidates = []

        for worker in workers:
            allocatable = await get_worker_allocatable_resource(worker)

            # Check if worker has enough RAM for CPU inference
            if allocatable.ram >= self._ram_claim:
                candidate = self._create_cpu_only_candidate(worker)
                candidates.append(candidate)

        return candidates

    def _create_single_gpu_candidate(
        self, worker: Worker, gpu_indexes: List[int], vram_distribution: Dict[int, int]
    ) -> ModelInstanceScheduleCandidate:
        """
        Create a single GPU candidate.
        """
        computed_resource_claim = ComputedResourceClaim(
            ram=self._ram_claim,
            vram=vram_distribution,
        )

        return ModelInstanceScheduleCandidate(
            worker=worker,
            gpu_indexes=gpu_indexes,
            computed_resource_claim=computed_resource_claim,
        )

    def _create_multi_gpu_candidate(
        self, worker: Worker, gpu_indexes: List[int], vram_distribution: Dict[int, int]
    ) -> ModelInstanceScheduleCandidate:
        """
        Create a multi-GPU candidate.
        """
        computed_resource_claim = ComputedResourceClaim(
            ram=self._ram_claim,
            vram=vram_distribution,
        )

        return ModelInstanceScheduleCandidate(
            worker=worker,
            gpu_indexes=gpu_indexes,
            computed_resource_claim=computed_resource_claim,
        )

    def _create_cpu_only_candidate(
        self, worker: Worker
    ) -> ModelInstanceScheduleCandidate:
        """
        Create a CPU-only candidate.
        """
        computed_resource_claim = ComputedResourceClaim(
            ram=self._ram_claim,
            vram={},  # No VRAM for CPU-only
        )

        return ModelInstanceScheduleCandidate(
            worker=worker,
            gpu_indexes=None,  # No GPUs for CPU-only
            computed_resource_claim=computed_resource_claim,
        )

    def _add_no_candidates_message(self):
        """
        Add message when no suitable candidates are found.
        """
        vram_gb = byte_to_gib(self._vram_claim)
        ram_gb = byte_to_gib(self._ram_claim)

        message = (
            f"No suitable workers found. "
            f"Required: VRAM {vram_gb:.1f} GiB, RAM {ram_gb:.1f} GiB."
        )

        self._messages.append(message)
