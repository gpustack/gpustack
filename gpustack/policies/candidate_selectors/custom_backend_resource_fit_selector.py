import logging
from typing import Dict, List, Optional

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
)
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    EVENT_ACTION_AUTO_SINGLE_GPU,
    EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
    EVENT_ACTION_DEFAULT,
    EVENT_ACTION_MANUAL_MULTI,
    RequestEstimateUsage,
    ScheduleCandidatesSelector,
)
from gpustack.policies.event_recorder.recorder import EventCollector, EventLevelEnum
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
    ListMessageBuilder,
    group_workers_by_gpu_type,
    estimate_model_vram,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
)
from gpustack.schemas.workers import Worker
from gpustack.config import Config
from gpustack.utils.unit import byte_to_gib

logger = logging.getLogger(__name__)

EVENT_ACTION_RESOURCE_ESTIMATION = "backend_resource_estimation_msg"
EVENT_ACTION_CPU_ONLY = "backend_cpu_only_scheduling_msg"


class CustomBackendResourceFitSelector(ScheduleCandidatesSelector):
    """
    Resource fit selector for custom backends.

    This selector:
    - Estimates resource requirements for custom backends
    - Finds suitable workers based on resource availability
    - Supports both GPU and CPU-only deployments
    """

    def __init__(self, cfg: Config, model: Model, model_instances: List[ModelInstance]):
        super().__init__(cfg, model, model_instances)
        self._event_collector = EventCollector(model, logger)
        self._messages = []

        # Estimated resource requirements
        self._vram_claim = 0
        self._ram_claim = 0

        self._set_gpu_count()

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
        self._vram_claim = await estimate_model_vram(
            self._model, self._config.huggingface_token, workers
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
            self.find_manual_gpu_selection_candidates,
            self.find_single_worker_single_gpu_candidates,
            self.find_single_worker_multi_gpu_candidates,
        ]

        # Add CPU-only candidates if supported
        if self._model.cpu_offloading:
            candidate_functions.append(self._find_cpu_only_candidates)

        for candidate_func in candidate_functions:
            if self.should_skip_candidate_func(candidate_func):
                continue

            logger.debug(
                f"Custom backend for model {self._model.readable_source}, "
                f"trying candidate selector: {candidate_func.__name__}"
            )

            candidates = candidate_func(workers)
            if candidates:
                # Prepare diagnostic messages for the user
                self._set_messages()
                return candidates

        # No suitable candidates found
        self._add_no_candidates_message()
        self._set_messages()
        return []

    def should_skip_candidate_func(self, candidate_func) -> bool:
        # Skip conditions for manual GPU selection.
        if (
            self._selected_gpu_workers
            and candidate_func != self.find_manual_gpu_selection_candidates
        ):
            return True

        return False

    def find_manual_gpu_selection_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        # Single worker scenarios
        if self._selected_gpu_workers and len(self._selected_gpu_workers) == 1:
            request = RequestEstimateUsage(
                ram=self._ram_claim,
                vram=self._vram_claim,
            )
            return self._find_manual_gpu_selection_candidates(
                workers, {"*": 0}, request, self._event_collector
            )

        # Multi-worker scenarios are not supported for custom backends
        elif self._selected_gpu_workers and len(self._selected_gpu_workers) > 1:
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

    def find_single_worker_single_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates using a single GPU on a single worker.
        """
        candidates = []
        largest_vram = 0

        workers_by_gpu_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_by_gpu_type.items():
            for worker in workers_of_type:
                if not worker.status or not worker.status.gpu_devices:
                    continue

                allocatable = get_worker_allocatable_resource(
                    self._model_instances, worker, gpu_type
                )

                for gpu_device in worker.status.gpu_devices:
                    gpu_index = gpu_device.index

                    # Check if GPU has enough VRAM
                    if gpu_index in allocatable.vram:
                        available_vram = allocatable.vram[gpu_index]

                        if available_vram >= self._vram_claim:
                            # Check RAM requirement
                            if allocatable.ram >= self._ram_claim:
                                candidate = self._create_single_gpu_candidate(
                                    worker,
                                    [gpu_index],
                                    {gpu_index: self._vram_claim},
                                    gpu_type,
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

    def find_single_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates using multiple GPUs on a single worker.
        """
        candidates = []
        largest_worker_vram = 0
        largest_worker_gpu_count = 0

        workers_by_gpu_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_by_gpu_type.items():
            for worker in workers_of_type:
                if not worker.status or not worker.status.gpu_devices:
                    continue

                if len(worker.status.gpu_devices) < 2:
                    continue  # Need at least 2 GPUs for multi-GPU

                allocatable = get_worker_allocatable_resource(
                    self._model_instances, worker, gpu_type
                )

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
                if (
                    total_available_vram >= self._vram_claim
                    and len(available_gpus) >= 2
                ):
                    # Check RAM requirement
                    if allocatable.ram >= self._ram_claim:
                        # Distribute VRAM evenly across GPUs
                        gpu_indexes = [gpu[0] for gpu in available_gpus]
                        vram_per_gpu = self._vram_claim // len(gpu_indexes)
                        vram_distribution = {idx: vram_per_gpu for idx in gpu_indexes}

                        candidate = self._create_multi_gpu_candidate(
                            worker, gpu_indexes, vram_distribution, gpu_type
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

    def _find_cpu_only_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates for CPU-only inference.
        """
        candidates = []

        for worker in workers:
            allocatable = get_worker_allocatable_resource(self._model_instances, worker)

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
        self,
        worker: Worker,
        gpu_indexes: List[int],
        vram_distribution: Dict[int, int],
        gpu_type: Optional[str] = None,
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
            gpu_type=gpu_type,
            gpu_indexes=gpu_indexes,
            computed_resource_claim=computed_resource_claim,
        )

    def _create_multi_gpu_candidate(
        self,
        worker: Worker,
        gpu_indexes: List[int],
        vram_distribution: Dict[int, int],
        gpu_type: Optional[str] = None,
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
            gpu_type=gpu_type,
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
