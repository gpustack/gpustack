import asyncio
import logging
import os
from typing import Dict, List, Optional

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
)
from gpustack.policies.event_recorder.recorder import EventCollector, EventLevelEnum
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
    get_local_model_weight_size,
    ListMessageBuilder,
)
from gpustack.schemas.models import (
    CategoryEnum,
    ComputedResourceClaim,
    Model,
    SourceEnum,
)
from gpustack.schemas.workers import Worker
from gpustack.config import Config
from gpustack.server.db import get_engine
from gpustack.utils.hub import get_model_weight_size
from gpustack.utils.unit import byte_to_gib
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_ids_by_worker, parse_gpu_id

logger = logging.getLogger(__name__)

EVENT_ACTION_DEFAULT = "default_scheduling_msg"
EVENT_ACTION_RESOURCE_ESTIMATION = "custom_backend_resource_estimation_msg"
EVENT_ACTION_MANUAL_MULTI = "custom_backend_manual_gpu_scheduling_msg"
EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU = (
    "custom_backend_auto_single_worker_multi_gpu_scheduling_msg"
)
EVENT_ACTION_AUTO_SINGLE_GPU = "custom_backend_auto_single_gpu_scheduling_msg"
EVENT_ACTION_CPU_ONLY = "custom_backend_cpu_only_scheduling_msg"


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
        self._event_collector = EventCollector(model, logger)
        self._messages = []
        self._engine = get_engine()

        # Estimated resource requirements
        self._vram_claim = 0
        self._ram_claim = 0

        # Whether the backend supports CPU-only inference
        self._supports_cpu_only = getattr(model, 'supports_cpu_only', True)

        # Manual GPU selection (worker -> [gpu_indexes])
        self._selected_gpu_workers: Optional[List[str]] = None
        self._selected_gpu_worker_count: int = 0
        self._selected_gpu_indexes_by_worker: Dict[str, List[int]] = {}
        # Multi-worker is not supported yet, but the code structure remains consistent with other backends for easier future adjustments.

        if self._model.gpu_selector and self._model.gpu_selector.gpu_ids:
            gpu_ids_by_worker = parse_gpu_ids_by_worker(
                self._model.gpu_selector.gpu_ids
            )
            self._selected_gpu_workers = list(gpu_ids_by_worker.keys())
            self._selected_gpu_worker_count = len(self._selected_gpu_workers)
            for worker_name, gpu_ids in gpu_ids_by_worker.items():
                gpu_indexes: List[int] = []
                for gpu_id in gpu_ids:
                    valid, matched = parse_gpu_id(gpu_id)
                    if valid:
                        gpu_index = safe_int(matched.get("gpu_index"))
                        gpu_indexes.append(gpu_index)
                self._selected_gpu_indexes_by_worker[worker_name] = gpu_indexes

    def get_messages(self) -> List[str]:
        """Get scheduling messages."""
        return self._messages

    def _set_messages(self):
        """Aggregate event messages into a compact diagnostic string list (similar to vLLM, without utilization)."""
        if self._messages:
            return

        event_messages = {
            EVENT_ACTION_DEFAULT: "",
            EVENT_ACTION_MANUAL_MULTI: "",
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU: "",
            EVENT_ACTION_AUTO_SINGLE_GPU: "",
            EVENT_ACTION_CPU_ONLY: "",
            EVENT_ACTION_RESOURCE_ESTIMATION: "",
        }

        for event in self._event_collector.events:
            if event.action in event_messages:
                event_messages[event.action] = event.message

        messages = event_messages[EVENT_ACTION_DEFAULT] + "\n"
        for action in [
            EVENT_ACTION_MANUAL_MULTI,
            EVENT_ACTION_CPU_ONLY,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            EVENT_ACTION_AUTO_SINGLE_GPU,
            # Fallback when no specific path message exists
            EVENT_ACTION_RESOURCE_ESTIMATION,
        ]:
            if event_messages[action]:
                messages += event_messages[action]
                break

        self._messages.append(messages)

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

        # Default message (VRAM/RAM claims)
        default_msg_list = ListMessageBuilder(
            f"The model requires approximately {byte_to_gib(self._vram_claim)} GiB of VRAM"
            f" and {byte_to_gib(self._ram_claim)} GiB of RAM."
        )
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_DEFAULT,
            str(default_msg_list),
        )

        # Try different candidate selection strategies
        candidate_functions = [
            self._find_manual_gpu_selection_candidates,
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
                # Prepare diagnostic messages for the user
                self._set_messages()
                return candidates

        # No suitable candidates found
        self._add_no_candidates_message()
        self._set_messages()
        return []

    async def _find_manual_gpu_selection_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates for manual GPU selection based on user-specified GPU IDs.
        """
        # Skip if no manual GPU selection is specified
        if not self._selected_gpu_workers:
            return []

        logger.debug(
            f"Custom backend manual GPU selection: workers={self._selected_gpu_workers}, "
            f"worker_count={self._selected_gpu_worker_count}, "
            f"gpu_indexes_by_worker={self._selected_gpu_indexes_by_worker}"
        )

        candidates: List[ModelInstanceScheduleCandidate] = []

        # Handle single worker scenarios
        if self._selected_gpu_worker_count == 1:
            selected_worker_name = self._selected_gpu_workers[0]
            selected_gpu_indexes = self._selected_gpu_indexes_by_worker[
                selected_worker_name
            ]

            # Find the worker
            target_worker = None
            for worker in workers:
                if worker.name == selected_worker_name:
                    target_worker = worker
                    break

            if not target_worker:
                self._messages.append(
                    f"Selected worker '{selected_worker_name}' not found."
                )
                return []

            allocatable = await get_worker_allocatable_resource(
                self._engine, target_worker
            )

            # Single GPU selection
            if len(selected_gpu_indexes) == 1:
                gpu_index = selected_gpu_indexes[0]
                available_vram = allocatable.vram.get(gpu_index, 0)

                overcommit = False
                if (
                    available_vram < self._vram_claim
                    or allocatable.ram < self._ram_claim
                ):
                    overcommit = True

                vram_claim = {gpu_index: int(self._vram_claim)}
                candidates.append(
                    ModelInstanceScheduleCandidate(
                        worker=target_worker,
                        gpu_indexes=[gpu_index],
                        computed_resource_claim=ComputedResourceClaim(
                            vram=vram_claim,
                            ram=self._ram_claim,
                        ),
                        overcommit=overcommit,
                    )
                )

                # Add manual selection diagnostics
                scheduling_msg = ListMessageBuilder(
                    [
                        f"Selected GPU {gpu_index} has {byte_to_gib(available_vram)} GiB allocatable VRAM.",
                        f"Required VRAM {byte_to_gib(self._vram_claim)} GiB, RAM {byte_to_gib(self._ram_claim)} GiB.",
                    ]
                )
                self._event_collector.add(
                    EventLevelEnum.INFO,
                    EVENT_ACTION_MANUAL_MULTI,
                    str(scheduling_msg),
                )

            # Multi GPU selection on single worker
            elif len(selected_gpu_indexes) > 1:
                # Check if total VRAM across selected GPUs is sufficient
                total_available_vram = 0
                for idx in selected_gpu_indexes:
                    total_available_vram += allocatable.vram.get(idx, 0)

                overcommit = False
                if (
                    total_available_vram < self._vram_claim
                    or allocatable.ram < self._ram_claim
                ):
                    overcommit = True

                # Distribute VRAM evenly across selected GPUs (bounded by allocatable VRAM)
                vram_per_gpu = max(
                    int(self._vram_claim // len(selected_gpu_indexes)), 1
                )
                vram_distribution: Dict[int, int] = {}
                for idx in selected_gpu_indexes:
                    vram_distribution[idx] = min(
                        vram_per_gpu, allocatable.vram.get(idx, 0)
                    )

                candidates.append(
                    ModelInstanceScheduleCandidate(
                        worker=target_worker,
                        gpu_indexes=selected_gpu_indexes,
                        computed_resource_claim=ComputedResourceClaim(
                            vram=vram_distribution,
                            ram=self._ram_claim,
                        ),
                        overcommit=overcommit,
                    )
                )

                # Add manual selection diagnostics
                scheduling_msg = ListMessageBuilder(
                    [
                        f"Selected GPUs have {byte_to_gib(total_available_vram)} GiB allocatable VRAM across {len(selected_gpu_indexes)} GPUs.",
                        f"Required VRAM {byte_to_gib(self._vram_claim)} GiB, RAM {byte_to_gib(self._ram_claim)} GiB.",
                    ]
                )
                self._event_collector.add(
                    EventLevelEnum.INFO,
                    EVENT_ACTION_MANUAL_MULTI,
                    str(scheduling_msg),
                )

        # Multi-worker scenarios are not supported for custom backends
        elif self._selected_gpu_worker_count > 1:
            # Record unsupported manual multi-worker selection
            self._event_collector.add(
                EventLevelEnum.ERROR,
                EVENT_ACTION_MANUAL_MULTI,
                str(
                    ListMessageBuilder(
                        "Manual GPU selection across multiple workers is not supported for custom backends."
                    )
                ),
            )
            return []

        return candidates

    async def _find_single_worker_single_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates using a single GPU on a single worker.
        """
        candidates = []
        largest_vram = 0

        for worker in workers:
            if not worker.status or not worker.status.gpu_devices:
                continue

            allocatable = await get_worker_allocatable_resource(self._engine, worker)

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
                    largest_vram = max(largest_vram, available_vram)

        if not candidates:
            # Add diagnostic message similar to vLLM single GPU path (without utilization)
            event_msg = ListMessageBuilder(
                f"The current available GPU only has {byte_to_gib(largest_vram)} GiB allocatable VRAM."
            )
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_AUTO_SINGLE_GPU,
                str(event_msg),
            )
        return candidates

    async def _find_single_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates using multiple GPUs on a single worker.
        """
        candidates = []
        largest_worker_vram = 0
        largest_worker_gpu_count = 0

        for worker in workers:
            if not worker.status or not worker.status.gpu_devices:
                continue

            if len(worker.status.gpu_devices) < 2:
                continue  # Need at least 2 GPUs for multi-GPU

            allocatable = await get_worker_allocatable_resource(self._engine, worker)

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
            # Track largest worker VRAM for diagnostics
            if total_available_vram > largest_worker_vram:
                largest_worker_vram = total_available_vram
                largest_worker_gpu_count = len(available_gpus)

        if not candidates:
            # Add diagnostic message similar to vLLM multi-GPU path (without utilization)
            event_msg_list = ListMessageBuilder(
                f"The largest available worker has {byte_to_gib(largest_worker_vram)} GiB allocatable VRAM across {largest_worker_gpu_count} GPUs."
            )
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
                str(event_msg_list),
            )
        return candidates

    async def _find_cpu_only_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates for CPU-only inference.
        """
        candidates = []

        for worker in workers:
            allocatable = await get_worker_allocatable_resource(self._engine, worker)

            # Check if worker has enough RAM for CPU inference
            if allocatable.ram >= self._ram_claim:
                candidate = self._create_cpu_only_candidate(worker)
                candidates.append(candidate)
            else:
                # Add CPU-only diagnostic message
                event_msg = ListMessageBuilder(
                    f"CPU-only inference is supported. Requires at least {byte_to_gib(self._ram_claim)} GiB RAM."
                )
                self._event_collector.add(
                    EventLevelEnum.INFO,
                    EVENT_ACTION_CPU_ONLY,
                    str(event_msg),
                )

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
            f"Required VRAM {vram_gb:.1f} GiB, RAM {ram_gb:.1f} GiB."
        )
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_RESOURCE_ESTIMATION,
            str(ListMessageBuilder(message)),
        )
