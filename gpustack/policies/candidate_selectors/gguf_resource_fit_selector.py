import itertools
import logging
import os
import copy
import time
from typing import Any, Dict, List, Optional, Tuple

from gpustack.policies.event_recorder.recorder import EventCollector, EventLevelEnum
from gpustack.policies.utils import get_worker_allocatable_resource, ListMessageBuilder
from gpustack.scheduler.calculator import (
    GPUOffloadEnum,
    ModelResourceClaim,
    Estimate,
    MemoryEstimate,
    calculate_gguf_model_resource_claim,
)
from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScheduleCandidate,
)
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    ScheduleCandidatesSelector,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
    ModelInstanceSubordinateWorker,
    is_image_model,
)
from gpustack.schemas.workers import Worker
from gpustack.utils.command import find_parameter
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_id, group_gpu_ids_by_worker
from gpustack.utils.unit import byte_to_gib, byte_to_kib


logger = logging.getLogger(__name__)

DEFAULT_MAX_RPC_SERVER_COUNT = 8
DEFAULT_MAX_RPC_COMBINATION_GENERATE_GPU_COUNT = 16
default_max_rpc_server_count = int(
    os.getenv("DEFAULT_MAX_RPC_SERVER_COUNT", DEFAULT_MAX_RPC_SERVER_COUNT)
)
default_max_rpc_combination_generate_gpu_count = int(
    os.getenv(
        "DEFAULT_MAX_RPC_COMBINATION_GENERATE_GPU_COUNT",
        DEFAULT_MAX_RPC_COMBINATION_GENERATE_GPU_COUNT,
    )
)

# event reasons
EVENT_REASON_INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES"
EVENT_REASON_MAX_RPC_COMBINATION_GENERATE_GPU_COUNT_EXCEED = (
    "MAX_RPC_COMBINATION_GENERATE_GPU_COUNT_EXCEED"
)
EVENT_REASON_INSUFFICIENT_RESOURCES_GPU_SELECTED = "INSUFFICIENT_RESOURCES_GPU_SELECTED"
EVENT_REASON_SELECTED_INVALID_GPU = "SELECTED_INVALID_GPU"
EVENT_REASON_INVALID_BACKEND_PARAMETER = "INVALID_BACKEND_PARAMETER"

# event action
EVENT_ACTION_SINGLE_WORKER_SINGLE_GPU_FULL_OFFLOADING = (
    "Single-Worker Single-GPU Full Offloading"
)
EVENT_ACTION_SINGLE_WORKER_MULTI_GPU_FULL_OFFLOADING = (
    "Single-Worker Multi-GPU Full Offloading"
)
EVENT_ACTION_DISTRIBUTED_DEPLOYMENT = "Distributed Deployment"
EVENT_ACTION_SINGLE_WORKER_PARTIAL_OFFLOADING = "Single-Worker Partial Offloading"
EVENT_ACTION_CPU_OFFLOADING = "CPU Offloading"
EVENT_ACTION_PRE_CHECK = "Pre-Check"


class GGUFResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        model: Model,
        model_instances: List[ModelInstance],
        cache_dir: Optional[str] = None,
    ):
        self._initialize_basic_data(model, model_instances, cache_dir)
        self._initialize_cached_claim_data()
        self._initialize_model_parameters(model)
        self._initialize_selected_gpu_ids()

    def _initialize_basic_data(
        self,
        model: Model,
        model_instances: List[ModelInstance],
        cache_dir: Optional[str],
    ):
        """Initialize basic data."""
        self._model = model
        self._model_instances = model_instances
        self._cache_dir = cache_dir
        self._workers = []  # Initialize workers list for remote parsing

        self._workers_allocatable_resource = {}
        self._gpus_allocatable_vram = []
        self._workers_allocatable_vram = []

        self._worker_name_to_worker: Dict[str, Worker] = {}
        self._worker_id_to_worker: Dict[int, Worker] = {}

        self._max_gpu_vram = 0
        self._approximate_full_offload_required_gpu_number = 1
        self._allocatable_gpu_count = 0
        self._gpu_count = 0
        self._allocatable_worker_count = 0

        self._messages = []
        self._event_collector = EventCollector(self._model, logger)

    def _initialize_cached_claim_data(self):
        """Initialize cached claim data."""

        # Cached simple claim data.
        self._full_offload_resource_claim = None
        self._partial_offload_resource_claim = None
        self._disable_offload_resource_claim = None
        self._total_layers = 0
        self._non_uma_single_gpu_full_offload_vram = 0
        self._uma_single_gpu_full_offload_vram = 0
        self._non_uma_single_layer_vram = 0
        self._uma_single_layer_vram = 0
        self._rpc_non_uma_single_layer_vram = 0
        self._rpc_uma_single_layer_vram = 0

        # Cached complex claim data.
        self._multi_workers_multi_gpus_partial_offload_resource_claim_cache = {}
        self._single_worker_multi_gpus_partial_offload_resource_claim_cache = {}
        self._cache_max_size = 50

    def _initialize_model_parameters(self, model: Model):
        """Initialize model parameters."""
        self._param_tensor_split = None
        self._param_gpu_layers = None
        self._param_ctx_size_in_kib = 8192
        if model.backend_parameters:
            self._param_tensor_split = find_parameter(
                model.backend_parameters, ["ts", "tensor-split"]
            )

            _param_gpu_layers = find_parameter(
                model.backend_parameters, ["ngl", "gpu-layers", "n-gpu-layers"]
            )
            if _param_gpu_layers:
                self._param_gpu_layers = safe_int(_param_gpu_layers, default=None)

            _param_ctx_size = find_parameter(
                self._model.backend_parameters, ["ctx-size", "c"]
            )
            if _param_ctx_size:
                self._param_ctx_size_in_kib = byte_to_kib(safe_int(_param_ctx_size))

    def _initialize_selected_gpu_ids(self):
        """Initialize selected GPU IDs."""
        self._max_rpc_server_count = default_max_rpc_server_count
        self._selected_gpu_ids_by_worker = {}
        self._selected_gpu_ids = []
        if self._model.gpu_selector and self._model.gpu_selector.gpu_ids:
            self._selected_gpu_ids_by_worker = group_gpu_ids_by_worker(
                self._model.gpu_selector.gpu_ids
            )
            self._selected_gpu_ids = sorted(self._model.gpu_selector.gpu_ids)
            self._max_rpc_server_count = len(self._selected_gpu_ids)

        if self._param_tensor_split:
            # ignore the gpu_selector if tensor split is set.
            logger.info(
                f"Model {self._model.name} has tensor-split, ignore the gpu_selector"
            )
            self._selected_gpu_ids_by_worker = {}
            self._selected_gpu_ids = []

    def _get_worker_allocatable_resource(self, worker: Worker) -> Allocatable:
        if self._workers_allocatable_resource.get(worker.id):
            return self._workers_allocatable_resource.get(worker.id)

        return get_worker_allocatable_resource(self._model_instances, worker)

    def _get_claim_with_layers(
        self, layers: int, is_uma: bool = False
    ) -> Tuple[int, int]:
        vram_claim = 0
        ram_claim = 0
        for memory in self._partial_offload_resource_claim.items:
            if memory.offloadLayers == layers:
                vram_claim = memory.vrams[0].nonuma
                ram_claim = memory.ram.nonuma
                if is_uma:
                    vram_claim = memory.vrams[0].uma
                    ram_claim = memory.ram.uma
                break
        return vram_claim, ram_claim

    def _get_tensor_split_claim_with_layers(
        self,
        layers: int,
        is_uma: bool = False,
        claim_items: List[MemoryEstimate] = None,
    ) -> Tuple[List[int], int]:
        vram_claims = []
        ram_claim = 0
        for memory in claim_items:
            if memory.offloadLayers == layers:
                if is_uma:
                    vram_claims = [vram.uma for vram in memory.vrams]
                    ram_claim = memory.ram.uma
                else:
                    vram_claims = [vram.nonuma for vram in memory.vrams]
                    ram_claim = memory.ram.nonuma
                break
        return vram_claims, ram_claim

    def _filter_selected_gpus_in_workers(self, workers: List[Worker]):
        """
        Filter the selected GPUs in workers.
        """
        if (
            self._model.gpu_selector is None
            or self._model.gpu_selector.gpu_ids is None
            or len(self._model.gpu_selector.gpu_ids) == 0
        ):
            return workers, []

        for worker in workers:
            gpu_candidates = []
            for gpu in worker.status.gpu_devices:
                id = f"{worker.name}:{gpu.type}:{gpu.index}"
                if id not in self._model.gpu_selector.gpu_ids:
                    continue

                gpu_candidates.append(gpu)

            worker.status.gpu_devices = gpu_candidates

    def _set_workers_allocatable_resource(self, workers: List[Worker]):
        workers_allocatable, workers_allocatable_vram, workers_gpus_allocatable_vram = (
            self._generate_workers_and_gpus_allocatable_resources(workers)
        )

        sorted_workers_allocatable_vram, sorted_gpus_allocatable_vram = (
            _sort_and_group_worker_gpu_vram(
                workers_allocatable_vram, workers_gpus_allocatable_vram
            )
        )

        self._workers_allocatable_resource = workers_allocatable
        self._gpus_allocatable_vram = sorted_gpus_allocatable_vram
        self._workers_allocatable_vram = sorted_workers_allocatable_vram

        self._allocatable_gpu_count = len(self._gpus_allocatable_vram)
        self._gpu_count = sum(
            len(worker.status.gpu_devices)
            for worker in workers
            if worker.status.gpu_devices
        )

        if len(self._gpus_allocatable_vram) > 0:
            self._max_gpu_vram = self._gpus_allocatable_vram[0][2]
            self._approximate_full_offload_required_gpu_number = (
                self._estimate_approximate_required_gpu_number(
                    self._non_uma_single_gpu_full_offload_vram
                )
            )

        for worker in workers:
            self._worker_id_to_worker[worker.id] = worker
            self._worker_name_to_worker[worker.name] = worker

        self._allocatable_worker_count = len(self._workers_allocatable_resource.keys())

    def _set_single_layer_vram(
        self, result: ModelResourceClaim, rpc_result: ModelResourceClaim
    ):
        for item in result.resource_claim_estimate.items:
            if item.vrams[0].handleLayers == 1:
                self._non_uma_single_layer_vram = item.vrams[0].nonuma
                self._uma_single_layer_vram = item.vrams[0].uma
                break

        for item in rpc_result.resource_claim_estimate.items:
            if item.vrams[0].handleLayers == 1:
                self._rpc_non_uma_single_layer_vram = item.vrams[0].nonuma
                self._rpc_uma_single_layer_vram = item.vrams[0].uma
                break

    async def _set_offload_resource_claim(self):
        result = await self._calculate_model_resource_claim()

        rpc_result = await self._calculate_model_resource_claim(
            tensor_split=[1, 1],
            rpc=["host:80"],
        )

        disable_offload_result = copy.deepcopy(result)
        disable_offload_result.resource_claim_estimate.items = [
            result.resource_claim_estimate.items[0]
        ]

        full_offload_result = copy.deepcopy(result)
        full_offload_result.resource_claim_estimate.items = [
            result.resource_claim_estimate.items[-1]
        ]
        full_offload_item = full_offload_result.resource_claim_estimate.items[0]

        self._full_offload_resource_claim = full_offload_result.resource_claim_estimate
        self._partial_offload_resource_claim = result.resource_claim_estimate
        self._disable_offload_result_claim = (
            disable_offload_result.resource_claim_estimate
        )
        self._total_layers = full_offload_item.offloadLayers
        self._uma_single_gpu_full_offload_vram = full_offload_item.vrams[0].uma
        self._non_uma_single_gpu_full_offload_vram = full_offload_item.vrams[0].nonuma
        self._uma_single_gpu_full_offload_ram = full_offload_item.ram.uma
        self._non_uma_single_gpu_full_offload_ram = full_offload_item.ram.nonuma

        self._set_single_layer_vram(result, rpc_result)

    def _set_model_parameters(self):
        if self._param_gpu_layers and self._param_gpu_layers > self._total_layers:
            self._param_gpu_layers = self._total_layers

    def _set_messages(self, candidates: List[ModelInstanceScheduleCandidate]):
        event_messages = {
            EventLevelEnum.ERROR: [],
            EventLevelEnum.WARNING: [],
            EVENT_ACTION_PRE_CHECK: "",
            EVENT_ACTION_SINGLE_WORKER_SINGLE_GPU_FULL_OFFLOADING: "",
            EVENT_ACTION_SINGLE_WORKER_MULTI_GPU_FULL_OFFLOADING: "",
            EVENT_ACTION_DISTRIBUTED_DEPLOYMENT: "",
            EVENT_ACTION_SINGLE_WORKER_PARTIAL_OFFLOADING: "",
            EVENT_ACTION_CPU_OFFLOADING: "",
        }

        for event in self._event_collector.events:
            if event.level == EventLevelEnum.ERROR:
                event_messages[EventLevelEnum.ERROR].append(event.message)
            elif event.level == EventLevelEnum.WARNING:
                event_messages[EventLevelEnum.WARNING].append(event.message)
            else:
                event_messages[event.action] = event.message

        if event_messages[EventLevelEnum.ERROR]:
            self._messages = event_messages[EventLevelEnum.ERROR]
            return

        if not candidates:
            for action in [
                EVENT_ACTION_PRE_CHECK,
                EVENT_ACTION_SINGLE_WORKER_MULTI_GPU_FULL_OFFLOADING,
                EVENT_ACTION_SINGLE_WORKER_SINGLE_GPU_FULL_OFFLOADING,
                EVENT_ACTION_DISTRIBUTED_DEPLOYMENT,
                EVENT_ACTION_SINGLE_WORKER_PARTIAL_OFFLOADING,
                EVENT_ACTION_CPU_OFFLOADING,
            ]:
                if event_messages[action]:
                    self._messages.append(event_messages[action])
                    break

        self._messages.extend(event_messages[EventLevelEnum.WARNING])

    def get_messages(self) -> List[str]:
        return self._messages

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates by the resource fit claim.
        """

        if not workers:
            return []

        # Save workers reference for remote parsing
        self._workers = workers

        # reset the data with input workers.
        await self._set_offload_resource_claim()
        self._filter_selected_gpus_in_workers(workers)
        self._set_workers_allocatable_resource(workers)
        self._set_model_parameters()

        sorted_workers = self._sort_workers_by_allocatable_vram(workers)
        candidates = await self._filter_in_sequence(sorted_workers)
        return candidates

    async def _filter_in_sequence(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Filter the workers with the full offloading claim.
        """
        candidates = []
        candidate_functions = [
            self.find_single_worker_single_gpu_full_offloading_candidates,
            self.find_single_worker_multi_gpu_full_offloading_candidates,
            self.find_multi_worker_multi_gpu_candidates,
            self.find_single_worker_partial_offloading_candidates,
            self.find_single_worker_cpu_candidates,
        ]

        overall_start_time = time.time()
        for candidate_func in candidate_functions:
            if self._should_skip_candidate_func(candidate_func):
                continue

            func_start_time = time.time()
            logger.info(
                f"Begin filter candidates with resource fit selector: "
                f"{candidate_func.__name__}, model {self._model.name or self._model.readable_source}",
            )

            candidates = await candidate_func(workers)
            func_latency = time.time() - func_start_time
            logger.info(
                f"Finished filter candidates with resource fit selector: "
                f"{candidate_func.__name__}, model {self._model.name or self._model.readable_source}, "
                f"latency: {func_latency:.2f}s, candidates: {len(candidates)}",
            )
            if candidates is not None and len(candidates) > 0:
                break

        if (not candidates or len(candidates) == 0) and len(
            self._event_collector.events
        ) == 0:
            msg = ListMessageBuilder(
                f"The model requires approximately {byte_to_gib(self._non_uma_single_gpu_full_offload_vram)} GiB VRAM and {byte_to_gib(self._non_uma_single_gpu_full_offload_ram)} GiB RAM."
            )
            if self._selected_gpu_ids:
                msg.append(
                    "Selected GPUs do not meet the requirements to run the model."
                )
            else:
                msg.append(
                    "Cannot find a suitable worker combination to run the model."
                )
            self._event_collector.add(
                EventLevelEnum.ERROR,
                EVENT_ACTION_PRE_CHECK,
                str(msg),
                reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
            )

        overall_latency = time.time() - overall_start_time
        logger.info(
            f"Finished resource fit selector found {len(candidates)} candidates, model {self._model.name or self._model.readable_source}, "
            f"latency: {overall_latency:.2f}s",
        )

        self._set_messages(candidates)
        return candidates

    def _should_skip_candidate_func(self, candidate_func) -> bool:

        # Skip conditions for CPU offloading.
        if not self._model.cpu_offloading and candidate_func in [
            self.find_single_worker_partial_offloading_candidates,
            self.find_single_worker_cpu_candidates,
        ]:
            return True

        # Skip conditions for distributed inference.
        if (
            not self._model.distributed_inference_across_workers
            and candidate_func == self.find_multi_worker_multi_gpu_candidates
        ):
            return True

        # Skip conditions for image models.
        if (
            is_image_model(self._model)
            and candidate_func
            != self.find_single_worker_single_gpu_full_offloading_candidates
        ):
            # Only full offloading is supported for image models.
            return True

        if self._should_skip_for_params(candidate_func):
            return True

        if self._should_skip_for_manual_scheduling(candidate_func):
            return True

        if self._should_skip_for_allocatable_resource(candidate_func):
            return True

    def _should_skip_for_params(self, candidate_func) -> bool:
        # Skip conditions for param gpu layers.
        if self._param_gpu_layers or self._param_gpu_layers == 0:
            if (
                self._param_gpu_layers > 0
                and self._param_gpu_layers < self._total_layers
                and not self._model.cpu_offloading
            ):
                self._event_collector.add(
                    EventLevelEnum.ERROR,
                    EVENT_ACTION_PRE_CHECK,
                    f"The parameter --gpu-layers {self._param_gpu_layers} requires CPU offloading. Please allow CPU offloading in the model configuration.",
                    reason=EVENT_REASON_INVALID_BACKEND_PARAMETER,
                )
                return True

            if (
                self._param_gpu_layers == 0
                and candidate_func != self.find_single_worker_cpu_candidates
            ):
                # User specified full CPU offloading.
                return True

            if (
                self._param_gpu_layers != 0
                and candidate_func == self.find_single_worker_cpu_candidates
            ):
                # User specified GPU offloading.
                return True

            if self._param_gpu_layers != self._total_layers and candidate_func in [
                self.find_single_worker_single_gpu_full_offloading_candidates,
                self.find_single_worker_multi_gpu_full_offloading_candidates,
            ]:
                # User specified partial offloading.
                return True

            if self._param_gpu_layers == self._total_layers and candidate_func in [
                self.find_single_worker_partial_offloading_candidates,
                self.find_single_worker_cpu_candidates,
            ]:
                # User specified full offloading.
                return True

        # Skip conditions for param tensor_split.
        if self._param_tensor_split:
            if (
                candidate_func
                == self.find_single_worker_single_gpu_full_offloading_candidates
            ):
                return True

    def _should_skip_for_manual_scheduling(self, candidate_func) -> bool:
        # Skip conditions for manual scheduling.
        if self._selected_gpu_ids:
            if candidate_func == self.find_single_worker_cpu_candidates:
                return True

            worker_num = len(self._selected_gpu_ids_by_worker)
            if (
                worker_num > 1
                and candidate_func != self.find_multi_worker_multi_gpu_candidates
            ):
                return True

            if worker_num == 1:
                selected_worker_name = next(
                    iter(self._selected_gpu_ids_by_worker.keys())
                )
                selected_gpu_count = len(
                    self._selected_gpu_ids_by_worker.get(selected_worker_name)
                )

                if (
                    candidate_func == self.find_multi_worker_multi_gpu_candidates
                    or (
                        selected_gpu_count > 1
                        and candidate_func
                        == self.find_single_worker_single_gpu_full_offloading_candidates
                    )
                    or (
                        selected_gpu_count == 1
                        and candidate_func
                        == self.find_single_worker_multi_gpu_full_offloading_candidates
                    )
                ):
                    return True

    def _should_skip_for_allocatable_resource(  # noqa: C901
        self, candidate_func
    ) -> bool:
        # Skip conditions for worker allocatable resources.
        if self._allocatable_worker_count == 0:
            self._event_collector.add(
                EventLevelEnum.ERROR,
                EVENT_ACTION_PRE_CHECK,
                "Insufficient resources for the model. Please check the available VRAM and RAM for the workers.",
                reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
            )
            return True

        if self._allocatable_gpu_count == 0:
            if candidate_func == self.find_single_worker_cpu_candidates:
                return False

            if self._selected_gpu_ids:
                self._event_collector.add(
                    EventLevelEnum.ERROR,
                    EVENT_ACTION_PRE_CHECK,
                    f"Selected GPUs need at least {byte_to_gib(self._non_uma_single_layer_vram)} GiB of available VRAM.",
                    reason=EVENT_REASON_INSUFFICIENT_RESOURCES_GPU_SELECTED,
                )
                return True

            if self._gpu_count != 0:
                self._event_collector.add(
                    EventLevelEnum.INFO,
                    EVENT_ACTION_PRE_CHECK,
                    "No available resources for workers. Please check workers's available VRAM.",
                    reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
                )
                return True

        if (
            self._allocatable_worker_count < 2
            and candidate_func == self.find_multi_worker_multi_gpu_candidates
        ):
            return True

        if self._allocatable_gpu_count < 2 and candidate_func in [
            self.find_single_worker_multi_gpu_full_offloading_candidates,
            self.find_multi_worker_multi_gpu_candidates,
        ]:
            return True

        if self._selected_gpu_ids_by_worker:
            for (
                worker_name,
                selected_gpu_ids,
            ) in self._selected_gpu_ids_by_worker.items():
                worker = self._worker_name_to_worker.get(worker_name)
                if not worker:
                    self._event_collector.add(
                        EventLevelEnum.ERROR,
                        EVENT_ACTION_PRE_CHECK,
                        f"Selected GPUs's worker {worker_name} not found in the workers list.",
                        reason=EVENT_REASON_SELECTED_INVALID_GPU,
                    )
                    return True

                is_unified_memory = worker.status.memory.is_unified_memory

                worker_allocatable = self._workers_allocatable_resource.get(worker.id)
                if not worker_allocatable:
                    continue

                for selected_gpu_id in selected_gpu_ids:
                    valid, matched = parse_gpu_id(selected_gpu_id)
                    if not valid:
                        continue

                    selected_gpu_index = safe_int(matched.get("gpu_index"))
                    vram = worker_allocatable.vram.get(selected_gpu_index, 0)
                    vram_claim = self._get_single_layer_vram(is_unified_memory, False)
                    if vram < vram_claim:
                        self._event_collector.add(
                            EventLevelEnum.ERROR,
                            EVENT_ACTION_PRE_CHECK,
                            f"Selected GPU {selected_gpu_id} lacks enough VRAM. At least {byte_to_gib(vram_claim)} GiB is required.",
                            reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
                        )
                        return True

    async def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with workers.
        """

        logger.debug(f"Input {len(self._gpus_allocatable_vram)} candidates")

        if not self._gpus_allocatable_vram:
            return []

        final_candidates = []
        for gpu_allocatable_vram in self._gpus_allocatable_vram:
            result, should_continue = (
                await self._find_single_worker_single_gpu_full_offloading_candidates(
                    gpu_allocatable_vram
                )
            )
            if result:
                final_candidates.extend(result)

            if not should_continue:
                # Skip subsequent gpus because they have less vram
                break

        self._advise_for_find_single_worker_single_gpu_full_offloading_candidates(
            final_candidates, self._gpus_allocatable_vram[0]
        )

        logger.debug(f"Qualified {len(final_candidates)} candidates")

        return final_candidates

    async def _find_single_worker_single_gpu_full_offloading_candidates(
        self, gpu_allocatable_vram: Tuple[int, int, int]
    ) -> Tuple[List[ModelInstanceScheduleCandidate], bool]:
        """
        Find single worker single gpu full offloading candidates for the model instance with gpu_allocatable_vram.
        Args:
            gpu_allocatable_vram (Tuple[int, int, int]): Tuple of worker_id, gpu_index, gpu_allocatable_vram.

        Returns:
            Tuple[List[ModelInstanceScheduleCandidate], bool]: List of model instance schedule candidates and should_continue flag.
        """

        candidates = []
        worker = self._worker_id_to_worker.get(gpu_allocatable_vram[0])
        is_unified_memory = worker.status.memory.is_unified_memory

        vram_claim = self._non_uma_single_gpu_full_offload_vram
        ram_claim = self._non_uma_single_gpu_full_offload_ram
        if is_unified_memory:
            vram_claim = self._uma_single_gpu_full_offload_vram
            ram_claim = self._uma_single_gpu_full_offload_ram

        allocatable_vram = gpu_allocatable_vram[2]
        allocatable_ram = self._workers_allocatable_resource.get(worker.id).ram

        if is_unified_memory:
            # For UMA, we need to remove the claim of gpu memory before check the memory.
            if vram_claim > allocatable_vram:
                return None, True

            if ram_claim > allocatable_ram - vram_claim:
                return None, False
        else:
            if vram_claim > allocatable_vram:
                return None, True

            if ram_claim > allocatable_ram:
                return None, False

        gpu_index = gpu_allocatable_vram[1]
        satisfied_candidate = self._create_candidate(
            worker,
            self._total_layers,
            ram_claim,
            {gpu_index: vram_claim},
            [gpu_index],
        )
        candidates.append(satisfied_candidate)

        logger.debug(
            f"Found intermediate candidate: {satisfied_candidate.to_log_string()}"
        )

        return candidates, True

    async def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """Find single worker multi gpu full offloading candidates for the model instance with workers.

        Args:
            workers (List[Worker]): workers sorted by allocatable vram resource.

        Returns:
            List[ModelInstanceScheduleCandidate]: List of model instance schedule candidates.
        """

        candidates = []
        advise_use_worker = None
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            result = (
                await self._find_single_worker_multi_gpu_full_offloading_candidates(
                    worker
                )
            )
            if result:
                candidates.extend(result)
            elif advise_use_worker is None:
                advise_use_worker = worker

        logger.debug(f"Found {len(candidates)} intermediate candidates")

        min_gpu_count = -1
        final_candidates = []
        if candidates:
            min_gpu_count = min(len(candidate.gpu_indexes) for candidate in candidates)
            final_candidates = [
                candidate
                for candidate in candidates
                if len(candidate.gpu_indexes) == min_gpu_count
            ]

        await self._advise_for_find_single_worker_multi_gpus_full_offloading_candidates(
            final_candidates, advise_use_worker
        )
        return final_candidates

    async def _find_single_worker_multi_gpu_full_offloading_candidates(  # noqa: C901
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi gpu full offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """

        is_unified_memory = worker.status.memory.is_unified_memory
        allocatable = self._get_worker_allocatable_resource(worker)

        # Pre filter
        logger.debug(f"Pre candidates filter for worker: {worker.name}")
        total_gpu = len(allocatable.vram.keys())
        if total_gpu < 2:
            return None

        if is_unified_memory:
            if allocatable.ram < self._uma_single_gpu_full_offload_ram:
                return None
        else:
            if allocatable.ram < self._non_uma_single_gpu_full_offload_ram:
                return None

        candidates = []
        begin_gpu_count = max(2, self._approximate_full_offload_required_gpu_number)
        for gpu_count in range(begin_gpu_count, total_gpu + 1):
            gpu_combinations, equal_vram = (
                self._generate_combinations_for_single_worker_multi_gpus(
                    allocatable, worker, gpu_count
                )
            )

            if not gpu_combinations:
                continue

            logger.debug(
                f"Input {len(gpu_combinations)} intermediate candidates for combinations with {gpu_count} gpus for worker: {worker.name}"
            )

            for i, gpu_combination in enumerate(gpu_combinations):
                satisfied_candidate = await self._find_single_worker_multi_gpu_full_offloading_candidates_with_combinations(
                    worker, allocatable, gpu_combination, is_unified_memory
                )

                if satisfied_candidate:
                    candidates.append(satisfied_candidate)
                    logger.debug(
                        f"Found intermediate candidate: {satisfied_candidate.to_log_string()}"
                    )

                if equal_vram and i > 0:
                    # Skip subsequent combinations because they have same vram
                    break

            # clear cache each count
            self._single_worker_multi_gpus_partial_offload_resource_claim_cache.clear()

            if candidates:
                break

        logger.debug(
            f"Qualified {len(candidates)} candidates for worker: {worker.name}"
        )
        return candidates

    async def _find_single_worker_multi_gpu_full_offloading_candidates_with_combinations(
        self,
        worker: Worker,
        allocatable: Allocatable,
        gpu_combination: Tuple[Tuple[int]],
        is_unified_memory: bool,
    ) -> ModelInstanceScheduleCandidate:

        # Check the resource claim should at least satisfy the minium resource claim(single gpu full offload).
        vram_sum = sum([value[-1] for value in gpu_combination])
        if (
            is_unified_memory and vram_sum < self._uma_single_gpu_full_offload_vram
        ) or (
            not is_unified_memory
            and vram_sum < self._non_uma_single_gpu_full_offload_vram
        ):
            # Skip subsequent combinations with same gpu count because they have less vram
            return None

        cache_key = self._cache_key_for_single_worker_multi_gpus_combination(
            gpu_combination
        )
        tensor_splitting = [value[-1] for value in gpu_combination]
        estimate = await self._get_or_calculate_model_resource_claim(
            self._single_worker_multi_gpus_partial_offload_resource_claim_cache,
            cache_key,
            tensor_splitting,
        )
        full_offload_item = estimate.items[-1]

        # ram
        ram_claim = full_offload_item.ram.nonuma
        if is_unified_memory:
            ram_claim = full_offload_item.ram.uma

        if ram_claim > allocatable.ram:
            return None

        # vram
        vram_claim_matched = True
        vram_claim = {}
        for gci in range(len(gpu_combination)):
            estimate_gpu_index = gci
            real_gpu_index = gpu_combination[gci][0]
            gpu_allocatable = allocatable.vram[real_gpu_index]

            single_gpu_vram_claim = full_offload_item.vrams[estimate_gpu_index].nonuma
            if is_unified_memory:
                single_gpu_vram_claim = full_offload_item.vrams[estimate_gpu_index].uma

            if single_gpu_vram_claim > gpu_allocatable:
                vram_claim_matched = False
                break

            vram_claim[real_gpu_index] = single_gpu_vram_claim

        if not vram_claim_matched:
            # stop to check other combinations have the same gpu count.
            return None

        gpu_indexes = [value[0] for value in gpu_combination]
        satisfied_candidate = self._create_candidate(
            worker,
            self._total_layers,
            ram_claim,
            vram_claim,
            gpu_indexes,
            tensor_splitting,
        )
        return satisfied_candidate

    async def find_single_worker_partial_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu partial offloading candidates for the model instance.
        """
        max_offload_layers = 0
        if self._param_gpu_layers:
            max_offload_layers = self._param_gpu_layers

        single_gpu_partial_offloading_candidates = []
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            result = (
                await self._find_single_worker_single_gpu_partial_offloading_candidates(
                    worker,
                    max_offload_layers,
                )
            )
            if (
                result
                and result.computed_resource_claim.offload_layers >= max_offload_layers
            ):
                max_offload_layers = result.computed_resource_claim.offload_layers
                single_gpu_partial_offloading_candidates.append(result)

        logger.debug(
            f"Found {len(single_gpu_partial_offloading_candidates)} intermediate candidates for single_worker_single_gpu_partial_offloading_candidates, max_offload_layers: {max_offload_layers}"
        )

        multi_gpu_partial_offloading_candidates = []
        multi_gpu_max_offload_layers = 0
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            results = (
                await self._find_single_worker_multi_gpu_partial_offloading_candidates(
                    worker, max_offload_layers
                )
            )
            if results:
                if _get_max_offload_layers(results) >= max_offload_layers:
                    multi_gpu_max_offload_layers = _get_max_offload_layers(results)
                    max_offload_layers = multi_gpu_max_offload_layers
                    multi_gpu_partial_offloading_candidates.extend(results)

        logger.debug(
            f"Found {len(multi_gpu_partial_offloading_candidates)} intermediate candidates for find_single_worker_multi_gpu_partial_offloading_candidates, max_offload_layers: {multi_gpu_max_offload_layers}"
        )

        intermediate_candidates = (
            single_gpu_partial_offloading_candidates
            + multi_gpu_partial_offloading_candidates
        )
        final_candidates = _filter_candidates_by_max_offload_layers(
            intermediate_candidates, max_offload_layers
        )

        await self._advise_for_find_single_worker_partial_offloading_candidates(
            final_candidates, workers[0].id
        )

        return final_candidates

    async def _find_single_worker_single_gpu_partial_offloading_candidates(  # noqa: C901
        self, worker: Worker, current_max_offload_layers: int = 0
    ) -> ModelInstanceScheduleCandidate:
        """
        Find single worker single gpu partial offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None

        Args:
            worker (Worker): Worker instance.
            current_max_offload_layers (int): Current max offload layers, if user specified param gpu laysers, the value would be the same as self._param_gpu_layers.
        """

        logger.debug(
            f"Input {len(worker.status.gpu_devices)} candidates for worker {worker.name}, with resource fit selector find_single_worker_single_gpu_partial_offloading_candidates, input max_offload_layers: {current_max_offload_layers}"
        )

        logger.debug(f"Pre candidates filter for worker: {worker.name}")
        if self._param_tensor_split:
            return None

        if self._selected_gpu_ids_by_worker:
            if worker.name not in self._selected_gpu_ids_by_worker:
                return None
            elif len(self._selected_gpu_ids_by_worker.get(worker.name)) > 1:
                return None

        is_unified_memory = worker.status.memory.is_unified_memory
        estimate = self._partial_offload_resource_claim

        allocatable = self._get_worker_allocatable_resource(worker)
        worker_allocatable_vram = sum(allocatable.vram.values())
        worker_allocatable_ram = allocatable.ram

        if not self._can_offload_at_least_one_layer(
            worker_allocatable_vram, self._get_single_layer_vram(is_unified_memory)
        ):
            return None

        sorted_gpus_memory = sorted(
            allocatable.vram.items(), key=lambda item: item[1], reverse=True
        )

        (
            vram_claim_for_current_max_offload_layers,
            ram_claim_for_current_max_offload_layers,
        ) = self._get_claim_with_layers(current_max_offload_layers, is_unified_memory)

        if worker_allocatable_vram < vram_claim_for_current_max_offload_layers:
            return None

        # NB(thxCode): There is a special case that when the worker is unified memory(usually, we assume it is an Apple macOS worker),
        # the true estimated usage includes both the vram and ram claims.
        # Since the `worker_allocatable_ram` is guaranteed to reflect the remaining unified memory of the worker,
        # we can check if the Apple macOS worker can serve the estimated usage or not.
        if is_unified_memory:
            if (
                worker_allocatable_ram
                < vram_claim_for_current_max_offload_layers
                + ram_claim_for_current_max_offload_layers
            ):
                return None

        # User specified gpu layers
        if self._param_gpu_layers:
            if worker_allocatable_ram < ram_claim_for_current_max_offload_layers:
                return None

            for gpu in sorted_gpus_memory:
                gpu_index = gpu[0]
                gpu_memory = gpu[1]
                if gpu_memory > vram_claim_for_current_max_offload_layers:
                    satisfied_candidate = self._create_candidate(
                        worker,
                        current_max_offload_layers,
                        ram_claim_for_current_max_offload_layers,
                        {gpu_index: vram_claim_for_current_max_offload_layers},
                        [gpu_index],
                    )
                    logger.debug(
                        f"Found intermediate candidate: {satisfied_candidate.to_log_string()}"
                    )
                    return satisfied_candidate
                else:
                    # Skip subsequent gpus because they have less vram
                    break
            return None

        # Normal case, without user specified gpu layers
        arr = []
        estimate_arr = []
        for memory in estimate.items:
            if memory.fullOffloaded:
                continue

            if (
                current_max_offload_layers
                and memory.offloadLayers < current_max_offload_layers
            ):
                continue

            vram_claim = memory.vrams[0].nonuma
            ram_claim = memory.ram.nonuma
            if is_unified_memory:
                vram_claim = memory.vrams[0].uma
                ram_claim = memory.ram.uma

            arr.append(vram_claim)
            estimate_arr.append(
                {
                    "vram": vram_claim,
                    "ram": ram_claim,
                    "offload_layers": memory.offloadLayers,
                }
            )

        for gpu in sorted_gpus_memory:
            gpu_index = gpu[0]
            gpu_memory = gpu[1]

            if self._selected_gpu_ids:
                valid, matched = parse_gpu_id(self._selected_gpu_ids[0])
                is_selected_gpu = valid and matched.get("gpu_index") == str(gpu_index)
                if not is_selected_gpu:
                    continue

            if not self._can_offload_at_least_one_layer(
                gpu_memory, self._get_single_layer_vram(is_unified_memory)
            ):
                continue

            index = binary_search(arr, gpu_memory)
            if index == -1:
                # Skip subsequent gpus because they have less vram
                break

            if (
                is_unified_memory
                # For UMA, we need to remove the claim of gpu memory before check if the memory.
                and (estimate_arr[index]["ram"] > allocatable.ram - arr[index])
                or (estimate_arr[index]["ram"] > allocatable.ram)
            ):
                continue

            offload_layers = estimate_arr[index]["offload_layers"]
            if offload_layers >= current_max_offload_layers:
                current_max_offload_layers = offload_layers
                satisfied_candidate = self._create_candidate(
                    worker,
                    offload_layers,
                    estimate_arr[index]["ram"],
                    {gpu_index: estimate_arr[index]["vram"]},
                    [gpu_index],
                )
                logger.debug(
                    f"Found intermediate candidate: {satisfied_candidate.to_log_string()}"
                )

                return satisfied_candidate
            else:
                # Skip subsequent gpus because they have less vram
                break

        logger.debug(f"Qualified 0 candidates for worker: {worker.name}")
        return None

    async def _find_single_worker_multi_gpu_partial_offloading_candidates(  # noqa: C901
        self, worker: Worker, current_max_offload_layers: int = 0
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi gpu partial offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """

        logger.debug(
            f"Input {len(worker.status.gpu_devices)} candidates for worker {worker.name}, with resource fit selector find_single_worker_multi_gpu_partial_offloading_candidates, input max_offload_layers: {current_max_offload_layers}"
        )

        logger.debug(f"Pre candidates filter for worker: {worker.name}")

        total_gpu = len(worker.status.gpu_devices) if worker.status.gpu_devices else 0
        if total_gpu < 2:
            return []

        if self._selected_gpu_ids_by_worker:
            if worker.name not in self._selected_gpu_ids_by_worker:
                return []
            elif len(self._selected_gpu_ids_by_worker.get(worker.name)) < 2:
                return []

        is_unified_memory = worker.status.memory.is_unified_memory
        allocatable = self._get_worker_allocatable_resource(worker)
        worker_allocatable_vram = sum(allocatable.vram.values())
        if not self._can_offload_at_least_one_layer(
            worker_allocatable_vram,
            self._get_single_layer_vram(is_unified_memory),
        ):
            return []

        (
            vram_claim_for_current_max_offload_layers,
            ram_claim_for_current_max_offload_layers,
        ) = self._get_claim_with_layers(current_max_offload_layers, is_unified_memory)
        if worker_allocatable_vram < vram_claim_for_current_max_offload_layers:
            return []

        candidates: List[ModelInstanceScheduleCandidate] = []
        previous_max_offload_layers = current_max_offload_layers
        for gpu_count in range(2, total_gpu + 1):
            gpu_combinations, equal_vram = (
                self._generate_combinations_for_single_worker_multi_gpus(
                    allocatable,
                    worker,
                    gpu_count,
                    vram_claim_for_current_max_offload_layers,
                )
            )

            if not gpu_combinations:
                continue

            logger.debug(
                f"Input {len(gpu_combinations)} intermediate candidates for combinations with {gpu_count} gpus for worker: {worker.name}, max_offload_layers: {current_max_offload_layers}"
            )

            for i, gpu_combination in enumerate(gpu_combinations):
                sum_vram = sum([value[-1] for value in gpu_combination])
                if sum_vram < vram_claim_for_current_max_offload_layers:
                    # Skip subsequent combinations with same gpu count because they have less vram
                    break

                satisfied_candidate = await self._find_single_worker_multi_gpu_partial_offloading_candidates_with_combination(
                    worker, gpu_combination, current_max_offload_layers
                )

                if satisfied_candidate and (
                    satisfied_candidate.computed_resource_claim.offload_layers
                    >= current_max_offload_layers
                ):
                    current_max_offload_layers = (
                        satisfied_candidate.computed_resource_claim.offload_layers
                    )
                    vram_claim_for_current_max_offload_layers, _ = (
                        self._get_claim_with_layers(
                            current_max_offload_layers, is_unified_memory
                        )
                    )

                    candidates.append(satisfied_candidate)

                    logger.debug(
                        f"Found intermediate candidate: {satisfied_candidate.to_log_string()}"
                    )

                if equal_vram and i > 0:
                    # Skip subsequent combinations because they have same vram
                    break

            if (
                previous_max_offload_layers
                and current_max_offload_layers <= previous_max_offload_layers
            ):
                # Skip subsequent gpu count because they need more gpus.
                break
            previous_max_offload_layers = current_max_offload_layers

        if not candidates:
            logger.debug(f"Qualified 0 candidates for worker: {worker.name}")
            return None

        max_offload_candidates = _get_max_offload_layers_candidates(candidates)

        logger.debug(
            f"Found {len(candidates)} intermediate candidates for worker {worker.name}, max_offload_layers of them: {current_max_offload_layers}"
        )

        min_gpu_count = min(
            len(candidate.gpu_indexes) for candidate in max_offload_candidates
        )

        final_candiates = [
            candidate
            for candidate in max_offload_candidates
            if len(candidate.gpu_indexes) == min_gpu_count
        ]

        logger.debug(
            f"Qualified {len(final_candiates)} candidates for worker: {worker.name}"
        )
        return final_candiates

    async def _find_single_worker_multi_gpu_partial_offloading_candidates_with_combination(  # noqa: C901
        self,
        worker: Worker,
        gpu_combination: Tuple[Tuple[int]],
        max_offload_layers: int = 0,
    ) -> ModelInstanceScheduleCandidate:
        """
        Find max offload layers for gpu combination.

        Args:
            worker (Worker): The worker instance containing GPU information.
            gpu_combination (List[Tuple[int]]): A list of tuples, each containing GPU index and it's vram (e.g., [(0, 106), (1, 98)])
            max_offload_layers (int): The current maximum offload layers, only consider candiate that offload layers equal or greater then it,
              if user specified param gpu layers, the value of max_offload_layers is the same as self._param_gpu_layers.
        """

        is_unified_memory = worker.status.memory.is_unified_memory
        allocatable = self._get_worker_allocatable_resource(worker)

        tensor_splitting = [value[-1] for value in gpu_combination]
        result = await self._calculate_model_resource_claim(
            tensor_split=tensor_splitting,
        )
        estimate = result.resource_claim_estimate

        gpu_indexes_mapping = [value[0] for value in gpu_combination]

        final_offload_layers = -1
        final_ram_claim = -1
        final_gpu_claims = {}
        final_gpu_indexes = []

        estimate_items = estimate.items[::-1]

        # User specified gpu layers
        if self._param_gpu_layers:
            if len(estimate.items) - 1 < self._param_gpu_layers:
                logger.error(
                    f"Invalid param gpu layers: {self._param_gpu_layers}, max layers is {len(estimate.items) - 1}, model {self._model.name or self._model.readable_source}"
                )
                return None
            estimate_items = estimate.items[
                self._param_gpu_layers : self._param_gpu_layers + 1
            ]

        for item in estimate_items:
            if item.fullOffloaded:
                continue
            if item.offloadLayers < max_offload_layers:
                break

            if (
                is_unified_memory
                and (item.ram.uma > (allocatable.ram - sum(g.uma for g in item.vrams)))
                or (item.ram.nonuma > allocatable.ram)
            ):
                continue

            vram_not_matched = False
            gpu_indexes = []
            gpu_claims = {}
            for vram_index, vram_claim in enumerate(item.vrams):
                real_gpu_index = gpu_indexes_mapping[vram_index]
                if (
                    is_unified_memory
                    and vram_claim.uma > allocatable.vram[real_gpu_index]
                ) or (vram_claim.nonuma > allocatable.vram[real_gpu_index]):
                    vram_not_matched = True
                    break

                gpu_indexes.append(real_gpu_index)
                gpu_claims[real_gpu_index] = vram_claim.nonuma
                if is_unified_memory:
                    gpu_claims[real_gpu_index] = vram_claim.uma

            if vram_not_matched:
                continue

            final_offload_layers = item.offloadLayers
            final_gpu_claims = gpu_claims
            final_gpu_indexes = gpu_indexes
            final_ram_claim = item.ram.nonuma
            if is_unified_memory:
                final_ram_claim = item.ram.uma

            break

        if final_offload_layers == -1:
            return None

        return self._create_candidate(
            worker,
            final_offload_layers,
            final_ram_claim,
            final_gpu_claims,
            final_gpu_indexes,
            tensor_splitting,
        )

    async def find_single_worker_cpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker without offloading candidates for the model instance with workers.
        """

        logger.debug(f"Input {len(workers)} candidates")

        final_candidates = []
        sorted_workers = self._sort_workers_by_allocatable_ram(workers)

        for worker in sorted_workers:
            result = await self._find_single_worker_with_cpu_candidates(worker)
            if result:
                final_candidates.extend(result)
            else:
                break
                # Skip subsequent workers because they have less ram

        self._advise_for_find_single_worker_cpu_candidates(
            final_candidates, sorted_workers[0].id
        )

        logger.debug(f"Qualified {len(final_candidates)} candidates")

        return final_candidates

    async def _find_single_worker_with_cpu_candidates(
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker without offloading candidates for the model instance.
        """

        allocatable = self._get_worker_allocatable_resource(worker)
        is_unified_memory = worker.status.memory.is_unified_memory
        estimate = self._disable_offload_result_claim

        ram_claim = estimate.items[0].ram.nonuma
        if is_unified_memory:
            ram_claim = estimate.items[0].ram.uma

        if ram_claim > allocatable.ram:
            return []

        satisfied_candidate = self._create_candidate(worker, 0, ram_claim, None, None)

        logger.debug(
            f"Found intermediate candidate: {satisfied_candidate.to_log_string()}"
        )

        return [satisfied_candidate]

    async def find_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:

        combinations = self._generate_combinations_for_worker_with_rpcs(workers)

        if combinations is None:
            return []

        combinations_count = sum(len(value) for value in combinations.values())
        combinations_sorted_keys = sorted(combinations.keys())

        logger.debug(
            f"Input {combinations_count} intermediate candidates from combinations: 1 main + rpcs({combinations_sorted_keys[0] - 1} to {combinations_sorted_keys[-1] - 1})"
        )

        candidates = []
        is_full_offloading = False
        max_offload_layers = -1
        for count in combinations_sorted_keys:

            logger.debug(
                f"Input {len(combinations[count])} intermediate candidates for combinations: 1 main + {count - 1} rpcs"
            )

            # Pre filter.
            sum_first_vram = sum([value[-1] for value in combinations[count][0]])

            begin_layers, end_layers = (
                self._find_multi_worker_multi_gpu_candidates_determine_layer_range(
                    sum_first_vram, max_offload_layers
                )
            )

            # Skip since the pre filter can't find begin layers for this combinations.
            if begin_layers == -1:
                continue

            # Checking combinations.
            logger.debug(
                f"Checking combinations: 1 main + {count - 1} rpcs, begin_layers: {begin_layers}, end_layers: {end_layers}"
            )

            for combination in combinations[count]:
                satisfied_candidate = (
                    await self._find_multi_worker_multi_gpu_candidate_with_combination(
                        combination,
                        begin_layers,
                        end_layers,
                    )
                )

                if not satisfied_candidate:
                    continue

                if not is_full_offloading:
                    is_full_offloading = (
                        satisfied_candidate.computed_resource_claim.offload_layers
                        == self._total_layers
                    )

                if (
                    satisfied_candidate.computed_resource_claim.offload_layers
                    >= max_offload_layers
                ):
                    max_offload_layers = (
                        satisfied_candidate.computed_resource_claim.offload_layers
                    )
                    candidates.append(satisfied_candidate)

                    logger.debug(
                        f"Found intermediate candidate: {satisfied_candidate.to_log_string()}"
                    )

            # Clean cache after each count.
            self._multi_workers_multi_gpus_partial_offload_resource_claim_cache.clear()

            if self._param_gpu_layers and len(candidates) > 0:
                # Skip subsequent counts because they use more rpc servers to offload same layers.
                break

            if is_full_offloading:
                # Skip subsequent counts because they use more rpc servers.
                break

        final_candidates = (
            self._find_multi_worker_multi_gpu_candidates_finalize_candidates(
                candidates, max_offload_layers
            )
        )
        return final_candidates

    def _find_multi_worker_multi_gpu_candidates_finalize_candidates(
        self, candidates, max_offload_layers
    ):
        logger.debug(f"Found {len(candidates)} intermediate candidates")

        final_candidates = []
        if candidates:
            if (
                not self._model.cpu_offloading
                and max_offload_layers != self._total_layers
            ):
                final_candidates = []
            else:
                final_candidates = _filter_candidates_by_max_offload_layers(
                    candidates, max_offload_layers
                )

        logger.debug(
            f"Qualified candidates: {len(final_candidates)}, max_offload_layers: {max_offload_layers}"
        )
        return final_candidates

    def _find_multi_worker_multi_gpu_candidates_determine_layer_range(
        self, sum_first_vram, max_offload_layers
    ) -> Tuple[int, int]:
        """
        Determine the layer range for multi worker multi gpu candidates.
        """

        begin_layers = -1
        end_layers = -1
        if self._param_gpu_layers:
            vram_claim_with_with_layers, _ = self._get_claim_with_layers(
                self._param_gpu_layers, True
            )
            if sum_first_vram < vram_claim_with_with_layers:
                return begin_layers, end_layers

            begin_layers = self._param_gpu_layers
            end_layers = self._param_gpu_layers
        else:

            for index, item in enumerate(
                reversed(self._partial_offload_resource_claim.items)
            ):
                if not self._model.cpu_offloading:
                    if index > 0:
                        break

                    if sum_first_vram > item.vrams[0].uma:
                        begin_layers = self._total_layers
                        end_layers = self._total_layers - 1
                else:
                    if item.offloadLayers < max_offload_layers:
                        continue

                    if sum_first_vram > item.vrams[0].uma:
                        begin_layers = item.offloadLayers
                        end_layers = 0
                        break

        return begin_layers, end_layers

    async def _find_multi_worker_multi_gpu_candidate_with_combination(  # noqa: C901
        self,
        combination,
        begin_layers,
        end_layers,
    ) -> ModelInstanceScheduleCandidate:
        """
        find multi worker multi gpu candidate with combination.
        combination example: ( ($worker_id, $worker_allocatable_vram), ($worker_id, $gpu_index, $gpu_allocatable), ($worker_id, $gpu_index, $gpu_allocatable) )
        """

        main_worker_id = combination[0][0]
        main_worker = self._worker_id_to_worker.get(main_worker_id)
        main_worker_is_unified_memory = main_worker.status.memory.is_unified_memory
        main_worker_gpus = [
            [value[1], value[2]]
            for value in self._gpus_allocatable_vram
            if value[0] == main_worker_id
        ]
        main_worker_gpu_indexes = [value[0] for value in main_worker_gpus]

        flag_tensor_spliting = []
        flag_rpc_servers = []
        for i in range(1, len(combination)):
            c_worker_id = combination[i][0]

            flag_rpc_servers.append(
                f"{self._worker_id_to_worker.get(c_worker_id).name}:{50052 + i}"
            )
            flag_tensor_spliting.append(combination[i][2])

        flag_tensor_spliting.extend([value[1] for value in main_worker_gpus])

        cache_key = self._cache_key_for_main_with_rpcs_combination(combination)
        estimate_result: Estimate = await self._get_or_calculate_model_resource_claim(  # type: ignore
            self._multi_workers_multi_gpus_partial_offload_resource_claim_cache,
            cache_key,
            flag_tensor_spliting,
            flag_rpc_servers,
        )

        satisfied_candidate = None
        estimate_items: List[MemoryEstimate] = sorted(
            estimate_result.items,
            key=lambda x: x.offloadLayers,
            reverse=True,
        )

        for e in estimate_items:
            if e.offloadLayers > begin_layers or e.offloadLayers < end_layers:
                continue

            # main worker checking.
            main_worker_ram_claim = e.ram.nonuma
            if main_worker_is_unified_memory:
                main_worker_ram_claim = e.ram.uma

            if (
                main_worker_ram_claim
                > self._workers_allocatable_resource.get(main_worker_id).ram
            ):
                continue

            main_worker_vram_claim = {}
            main_worker_satisfied = False
            for (
                main_worker_gpu_index,
                main_worker_gpu_allocatable,
            ) in self._workers_allocatable_resource.get(main_worker_id).vram.items():
                if main_worker_gpu_index not in main_worker_gpu_indexes:
                    continue

                # vrams: [rpc_server1, rpc_server2, ..., main_worker]
                position = len(flag_rpc_servers) + main_worker_gpu_indexes.index(
                    main_worker_gpu_index
                )

                claim = e.vrams[position].nonuma
                if main_worker_is_unified_memory:
                    claim = e.vrams[position].uma

                if claim > main_worker_gpu_allocatable:
                    main_worker_satisfied = False
                    break

                main_worker_satisfied = True
                main_worker_vram_claim[main_worker_gpu_index] = claim

            if not main_worker_satisfied:
                continue

            subordinate_workers = self._check_combination_rpcs(
                combination, e, self._total_layers
            )
            if not subordinate_workers:
                continue

            satisfied_candidate = self._create_candidate(
                main_worker,
                e.offloadLayers,
                main_worker_ram_claim,
                main_worker_vram_claim,
                main_worker_gpu_indexes,
                flag_tensor_spliting,
                subordinate_workers,
            )
            break

        return satisfied_candidate

    def _generate_workers_and_gpus_allocatable_resources(
        self, workers: List[Worker]
    ) -> Tuple[
        Dict[str, Allocatable], List[Tuple[int, int]], List[Tuple[int, int, int]]
    ]:
        """
        Generate allocatable resources for workers and their GPUs.

        Args:
            workers (List[Worker]): List of workers.

        Returns:
            Tuple containing workers' allocatable resources, workers' allocatable VRAM, and GPUs' allocatable VRAM.
        """

        workers_allocatable = {}
        workers_allocatable_vram = []
        workers_gpus_allocatable_vram = []

        for worker in workers:
            result = self._get_worker_allocatable_resource(worker)
            workers_allocatable[worker.id] = result

            if len(result.vram.keys()) == 0:
                continue

            worker_allocatable_vram = sum(result.vram.values())
            if worker_allocatable_vram > 0:
                workers_allocatable_vram.append([worker.id, worker_allocatable_vram])

            for gpu_device in worker.status.gpu_devices:
                if gpu_device.index is None:
                    logger.warning(
                        f"gpu index is not found for {worker.name} {gpu_device.name}"
                    )

                gpu_allocatable_vram = result.vram.get(gpu_device.index)
                if gpu_allocatable_vram is not None and gpu_allocatable_vram > 0:
                    workers_gpus_allocatable_vram.append(
                        [worker.id, gpu_device.index, gpu_allocatable_vram]
                    )

        return (
            workers_allocatable,
            workers_allocatable_vram,
            workers_gpus_allocatable_vram,
        )

    def _sort_workers_by_allocatable_vram(self, workers: List[Worker]) -> List[Worker]:
        worker_vram_totals = {
            worker.id: sum(self._workers_allocatable_resource[worker.id].vram.values())
            for worker in workers
        }

        sorted_workers = sorted(
            workers, key=lambda worker: worker_vram_totals[worker.id], reverse=True
        )
        return sorted_workers

    def _sort_workers_by_allocatable_ram(self, workers: List[Worker]) -> List[Worker]:
        sorted_workers = sorted(
            workers,
            key=lambda worker: self._workers_allocatable_resource[worker.id].ram,
            reverse=True,
        )
        return sorted_workers

    def _can_offload_at_least_one_layer(
        self, allocatable_vram: int, single_layer_vram: int
    ) -> bool:
        """Check if there is enough VRAM to offload at least one layer."""
        return allocatable_vram >= single_layer_vram

    def _get_single_layer_vram(self, is_unified_memory: bool, rpc: bool = False) -> int:
        """Get the VRAM required for a single layer based on memory type and RPC."""
        if rpc:
            return (
                self._rpc_uma_single_layer_vram
                if is_unified_memory
                else self._rpc_non_uma_single_layer_vram
            )
        return (
            self._uma_single_layer_vram
            if is_unified_memory
            else self._non_uma_single_layer_vram
        )

    def _generate_combinations_given_tensor_split(
        self,
    ) -> dict[Tuple[Tuple[int]]]:
        """
        Generate gpu combinations given tensor split.
        Example:
            Given: tensor_split = "1,5,8"
            Output: [((0, 1), (1, 5), (2, 8))]
        """
        tensor_splits = [int(x) for x in self._param_tensor_split.split(",")]
        n_split = len(tensor_splits)

        split_by_index = []
        for i in range(n_split):
            split_by_index.append((i, tensor_splits[i]))
        gpu_combinations = list(itertools.combinations(split_by_index, n_split))
        return gpu_combinations

    def _generate_combinations_for_single_worker_multi_gpus(
        self,
        allocatable: Allocatable,
        worker: Worker,
        gpu_count: int,
        at_least_vram: Optional[int] = None,
    ) -> Tuple[List[Tuple[Tuple[int]]], bool]:

        if self._param_tensor_split:
            # use specified tensor split when the param is set.
            total_gpu = len(worker.status.gpu_devices) or len(self._selected_gpu_ids)
            if total_gpu < len(self._param_tensor_split.split(",")):
                return None, False
            gpu_combinations = self._generate_combinations_given_tensor_split()
            return gpu_combinations, False

        if self._selected_gpu_ids_by_worker.get(worker.name):
            if len(self._selected_gpu_ids) != gpu_count:
                return None, False

            select_gpu_combinations = self._generate_combinations_with_selected_gpus(
                worker
            )
            gpu_combinations = [select_gpu_combinations]
            return gpu_combinations, False

        filterd_gpus = []
        equal_vram = True
        for gpu_index, vram in allocatable.vram.items():
            if not self._can_offload_at_least_one_layer(
                vram,
                self._get_single_layer_vram(worker.status.memory.is_unified_memory),
            ):
                continue
            filterd_gpus.append((gpu_index, vram))
            if vram != list(allocatable.vram.values())[0]:
                equal_vram = False

        total_gpu = len(filterd_gpus)
        sorted_gpus_memory = sorted(
            filterd_gpus, key=lambda item: item[1], reverse=True
        )

        if at_least_vram:
            if (
                sum([value[1] for value in sorted_gpus_memory[:gpu_count]])
                < at_least_vram
            ):
                return None, False

        gpu_combinations = list(itertools.combinations(sorted_gpus_memory, gpu_count))

        # gpu_combinations examples:
        # (($gpu_index, $gpu_allocatable), ($gpu_index, $gpu_allocatable))
        return gpu_combinations, equal_vram

    def _generate_combinations_with_selected_gpus(
        self, worker: Worker
    ) -> dict[Tuple[Tuple[int]]]:

        gpu_combinations = []
        selected_gpu_ids = self._selected_gpu_ids_by_worker.get(worker.name)
        allocatable = self._get_worker_allocatable_resource(worker)
        for selected_gpu_id in selected_gpu_ids:
            valid, matched = parse_gpu_id(selected_gpu_id)
            if not valid:
                continue

            selected_gpu_index = safe_int(matched.get("gpu_index"))
            vram = allocatable.vram.get(selected_gpu_index, 0)
            gpu_combinations.append((selected_gpu_index, vram))

        sorted_gpu_combinations = sorted(
            gpu_combinations, key=lambda item: item[1], reverse=True
        )
        return sorted_gpu_combinations

    def _generate_combinations_for_worker_with_rpcs(
        self,
        workers: List[Worker],
    ) -> Dict:

        combinations = {}
        if self._selected_gpu_ids:
            combinations = (
                self._generate_combinations_for_worker_with_rpcs_with_selected_gpu_ids(
                    workers,
                )
            )
        else:
            combinations = (
                self._generate_combinations_for_worker_with_rpcs_without_selected_gpu_ids()
            )

        # combinations examples:
        # [( ($worker_id, $worker_allocatable_vram), ($worker_id, $gpu_index, $gpu_allocatable), ($worker_id, $gpu_index, $gpu_allocatable) )]
        return combinations

    def _generate_combinations_for_worker_with_rpcs_with_selected_gpu_ids(  # noqa: C901
        self,
        workers: List[Worker],
    ) -> Dict:
        if not self._selected_gpu_ids:
            return None

        selected_workers_allocatable_vram = []
        worker_name_id_map = {worker.name: worker.id for worker in workers}

        worker_names = list(self._selected_gpu_ids_by_worker.keys())
        worker_ids = [worker.id for worker in workers if worker.name in worker_names]
        for w in self._workers_allocatable_vram:
            if w[0] in worker_ids:
                selected_workers_allocatable_vram.append(w)

        selected_workers_gpus_allocatable_vram = []
        for selected_gpu_id in self._selected_gpu_ids:
            valid, matched = parse_gpu_id(selected_gpu_id)
            if not valid:
                continue

            selected_worker_name = matched.get("worker_name")
            selected_gpu_index = matched.get("gpu_index")
            selected_worker_id = worker_name_id_map.get(selected_worker_name)
            for w in self._gpus_allocatable_vram:
                if w[0] == selected_worker_id and str(w[1]) == selected_gpu_index:
                    selected_workers_gpus_allocatable_vram.append(w)

        sorted_workers, sorted_gpus = _sort_and_group_worker_gpu_vram(
            selected_workers_allocatable_vram,
            selected_workers_gpus_allocatable_vram,
        )

        main_worker_vram, main_worker = self._get_main_for_combination(
            sorted_workers,
        )

        if not main_worker_vram:
            self._event_collector.add(
                EventLevelEnum.ERROR,
                EVENT_ACTION_DISTRIBUTED_DEPLOYMENT,
                str(
                    ListMessageBuilder(
                        [
                            f"Selected GPUs need at least {byte_to_gib(self._non_uma_single_layer_vram)} GiB of VRAM.",
                            "No suitable GPU found for the main server.",
                        ]
                    )
                ),
                reason=EVENT_REASON_INSUFFICIENT_RESOURCES_GPU_SELECTED,
            )
            return None

        # Check if the rpc gpus can offload even one layer.
        filtered_gpus = [gpu for gpu in sorted_gpus if gpu[0] != main_worker_vram[0]]

        for gpu in filtered_gpus:
            rpc_at_least_vram = self._get_single_layer_vram(
                self._worker_id_to_worker.get(gpu[0]).status.memory.is_unified_memory,
                True,
            )

            if not self._can_offload_at_least_one_layer(
                gpu[2],
                rpc_at_least_vram,
            ):
                key = f"{self._worker_id_to_worker.get(gpu[0]).name}:{gpu[2]}"

                self._event_collector.add(
                    EventLevelEnum.ERROR,
                    EVENT_ACTION_DISTRIBUTED_DEPLOYMENT,
                    str(
                        ListMessageBuilder(
                            [
                                f"The model in RPC requires approximately {byte_to_gib(rpc_at_least_vram)} GiB VRAM.",
                                f"Selected GPU {key} lacks enough VRAM to serve as an RPC server.",
                            ]
                        )
                    ),
                    reason=EVENT_REASON_INSUFFICIENT_RESOURCES_GPU_SELECTED,
                )
                return None

        c = [
            (
                main_worker_vram,
                *[gpu for gpu in filtered_gpus if gpu[0] != main_worker_vram[0]],
            )
        ]

        combinations = {}
        for item in c:
            key = len(item)
            if key not in combinations:
                combinations[key] = []

            combinations[key].append(item)

        logger.debug(
            f"Generated combinations with main: {main_worker.name} and rpcs number: {len(c) - 1}"
        )
        return combinations

    def _generate_combinations_for_worker_with_rpcs_without_selected_gpu_ids(
        self,
    ) -> Dict:

        main_worker_vram, main_worker = self._get_main_for_combination(
            self._workers_allocatable_vram,
        )

        if not main_worker_vram:
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_DISTRIBUTED_DEPLOYMENT,
                str(
                    ListMessageBuilder(
                        [
                            f"The model requires approximately {byte_to_gib(self._non_uma_single_layer_vram)} GiB VRAM.",
                            "No suitable main server found, current GPUs lack enough VRAM to offload a single layer.",
                        ]
                    )
                ),
                reason=EVENT_REASON_INSUFFICIENT_RESOURCES_GPU_SELECTED,
            )
            return None

        # Skip the gpus if the rpc gpus can't offload even one layer.
        filtered_gpus = [
            gpu for gpu in self._gpus_allocatable_vram if gpu[0] != main_worker_vram[0]
        ]
        filtered_gpus = [
            gpu
            for gpu in filtered_gpus
            if self._can_offload_at_least_one_layer(
                gpu[2],
                self._get_single_layer_vram(
                    self._worker_id_to_worker.get(
                        gpu[0]
                    ).status.memory.is_unified_memory,
                    True,
                ),
            )
        ]
        if len(filtered_gpus) == 0:
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_DISTRIBUTED_DEPLOYMENT,
                str(
                    ListMessageBuilder(
                        [
                            f"The model requires approximately {byte_to_gib(self._rpc_non_uma_single_layer_vram)} GiB VRAM.",
                            "Current GPUs lacks enough VRAM to serve as an RPC server.",
                        ]
                    )
                ),
                reason=EVENT_REASON_INSUFFICIENT_RESOURCES_GPU_SELECTED,
            )
            return None

        # Limit the number of gpus for generate rpc combination.
        if len(filtered_gpus) > default_max_rpc_combination_generate_gpu_count:
            self._event_collector.add(
                EventLevelEnum.WARNING,
                EVENT_ACTION_DISTRIBUTED_DEPLOYMENT,
                str(
                    ListMessageBuilder(
                        f"Too many candidate RPC servers (max {default_max_rpc_combination_generate_gpu_count} supported), skipping distributed deployment. Use manual scheduling to select GPUs if needed."
                    )
                ),
                reason=EVENT_REASON_MAX_RPC_COMBINATION_GENERATE_GPU_COUNT_EXCEED,
            )
            return None

        combinations = {}
        key_range = min(len(filtered_gpus), self._max_rpc_server_count)
        for i in range(1, (key_range + 1)):
            c = [
                (main_worker_vram, *v) for v in itertools.combinations(filtered_gpus, i)
            ]

            key = i + 1
            if key not in combinations:
                combinations[key] = []

            combinations[key].extend(c)

        logger.debug(
            f"Generated combinations with main: {main_worker.name} and rpcs number: 1-{len(filtered_gpus)}"
        )
        return combinations

    def _check_combination_rpcs(
        self,
        combination,
        e: MemoryEstimate,
        total_layers: int,
    ) -> List[ModelInstanceSubordinateWorker]:
        """
        Check the rpc servers resource satisfied with combination.
        combination example: ( ($worker_id, $worker_allocatable_vram), ($worker_id, $gpu_index, $gpu_allocatable), ($worker_id, $gpu_index, $gpu_allocatable) )
        """

        subordinate_workers: List[ModelInstanceSubordinateWorker] = []

        for i in range(1, len(combination)):
            r_worker_id = combination[i][0]
            r_gpu_index = combination[i][1]
            r_allocatable = combination[i][2]
            r_worker = self._worker_id_to_worker.get(r_worker_id)
            r_is_unified_memory = r_worker.status.memory.is_unified_memory

            position = i - 1
            r_vram_claim = e.vrams[position].nonuma
            if r_is_unified_memory:
                r_vram_claim = e.vrams[position].uma

            if r_vram_claim > r_allocatable:
                break

            subordinate_workers.append(
                ModelInstanceSubordinateWorker(
                    worker_id=r_worker_id,
                    worker_name=r_worker.name,
                    gpu_indexes=[r_gpu_index],
                    computed_resource_claim=ComputedResourceClaim(
                        is_unified_memory=r_is_unified_memory,
                        offload_layers=e.vrams[position].handleLayers,
                        vram={r_gpu_index: r_vram_claim},
                        ram=0,
                        total_layers=total_layers,
                    ),
                )
            )

        if len(subordinate_workers) != len(combination) - 1:
            return []

        return subordinate_workers

    def _get_main_for_combination(
        self,
        workers_allocations_vrams: Tuple[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], Worker]:
        """
        Get the worker with the most allocatable vram as main

        Args:
            workers_allocations_vrams (Tuple[Tuple[int, int]]): each tuple example ($worker_id, $worker_allocatable_vram)
            workers_allocatable (Dict[int, Allocatable]): workers allocatable resources

        Returns:
            Tuple[Tuple[int, int], Worker]: main worker vram and worker instance
        """

        max_worker_can_offload_vram = 0
        main_worker = None
        main_worker_vram = None

        for sw in workers_allocations_vrams:
            is_uma = self._worker_id_to_worker.get(
                sw[0]
            ).status.memory.is_unified_memory
            single_layer_vram = self._get_single_layer_vram(is_uma)

            if not self._can_offload_at_least_one_layer(
                sw[1],
                single_layer_vram,
            ):
                continue

            sum_can_offload_at_least_one_layer_vram = 0
            for ga in self._workers_allocatable_resource.get(sw[0]).vram.values():
                if self._can_offload_at_least_one_layer(
                    ga,
                    single_layer_vram,
                ):
                    sum_can_offload_at_least_one_layer_vram += ga

            if sum_can_offload_at_least_one_layer_vram > max_worker_can_offload_vram:
                max_worker_can_offload_vram = sum_can_offload_at_least_one_layer_vram
                main_worker = self._worker_id_to_worker.get(sw[0])
                main_worker_vram = sw

        return main_worker_vram, main_worker

    def _update_cache_for_multi_workers_multi_gpus_partial_offload_resource_claim(
        self, key: str, value: Any
    ):
        if (
            len(self._multi_workers_multi_gpus_partial_offload_resource_claim_cache)
            < self._cache_max_size
        ):
            self._multi_workers_multi_gpus_partial_offload_resource_claim_cache[key] = (
                value
            )

    def _update_cache_for_single_worker_multi_gpus_partial_offload_resource_claim(
        self, key: str, value: Any
    ):
        if (
            len(self._single_worker_multi_gpus_partial_offload_resource_claim_cache)
            < self._cache_max_size
        ):
            self._single_worker_multi_gpus_partial_offload_resource_claim_cache[key] = (
                value
            )

    def _cache_key_for_single_worker_multi_gpus_combination(self, combination):
        # combination example:
        # ( ($gpu_index, $gpu_allocatable), ($gpu_index, $gpu_allocatable) )
        values = [str(item[-1]) for item in combination]
        key = '|'.join(values)
        return key

    def _cache_key_for_main_with_rpcs_combination(self, combination):
        # combination example:
        # ( ($worker_id, $worker_allocatable_vram), ($worker_id, $gpu_index, $gpu_allocatable), ($worker_id, $gpu_index, $gpu_allocatable) )
        values = [str(item[-1]) for item in combination]
        key = '|'.join(values)
        return key

    async def _calculate_model_resource_claim(
        self, offload: GPUOffloadEnum = GPUOffloadEnum.Partial, **kwargs
    ) -> ModelResourceClaim:
        return await calculate_gguf_model_resource_claim(
            self._model,
            offload,
            workers=self._workers,
            cache_dir=self._cache_dir,
            **kwargs,
        )

    async def _get_or_calculate_model_resource_claim(
        self, cache, cache_key, tensor_split=None, rpc=None
    ) -> Estimate:
        """
        Get the resource claim estimate from cache or calculate it if not present in cache.
        """
        if cache_key in cache:
            return cache[cache_key]

        result = await self._calculate_model_resource_claim(
            tensor_split=tensor_split, rpc=rpc
        )
        estimate = result.resource_claim_estimate
        cache[cache_key] = estimate
        return estimate

    def _create_candidate(
        self,
        worker: Worker,
        offload_layers: int,
        ram_claim: int,
        vram_claim: Dict[int, int],
        gpu_indexes: Optional[List[int]] = None,
        tensor_split: Optional[List[int]] = None,
        subordinate_workers: Optional[List[ModelInstanceSubordinateWorker]] = None,
    ) -> ModelInstanceScheduleCandidate:
        """
        Create a ModelInstanceScheduleCandidate object.
        """
        candidate = ModelInstanceScheduleCandidate(
            worker=worker,
            gpu_indexes=gpu_indexes,
            computed_resource_claim=ComputedResourceClaim(
                is_unified_memory=worker.status.memory.is_unified_memory,
                offload_layers=offload_layers,
                vram=vram_claim,
                ram=ram_claim,
                total_layers=self._total_layers,
                tensor_split=tensor_split,
            ),
            subordinate_workers=subordinate_workers,
        )

        return candidate

    def _estimate_approximate_required_gpu_number(self, vram: int) -> int:
        """Estimate the approximate required number of GPUs based on the VRAM offload rate."""
        if vram == 0:
            return -1
        vram_offload_rate = self._max_gpu_vram / vram
        for i in range(1, 17):
            if vram_offload_rate > 1 / i:
                return i
        return -1

    def _advise_for_find_single_worker_single_gpu_full_offloading_candidates(
        self,
        candidates: List[ModelInstanceScheduleCandidate],
        gpu_vram: Tuple[int, int, int],
    ):
        if candidates:
            return

        ram = self._workers_allocatable_resource.get(gpu_vram[0]).ram
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_SINGLE_WORKER_SINGLE_GPU_FULL_OFFLOADING,
            str(
                ListMessageBuilder(
                    [
                        f"The model requires approximately {byte_to_gib(self._non_uma_single_gpu_full_offload_vram)} GiB VRAM and {byte_to_gib(self._non_uma_single_gpu_full_offload_ram)} GiB RAM.",
                        f"The available GPU has {byte_to_gib(gpu_vram[2])} GiB VRAM and {byte_to_gib(ram)} GiB RAM.",
                    ]
                )
            ),
            reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
        )

    async def _advise_for_find_single_worker_multi_gpus_full_offloading_candidates(
        self, candidates: List[ModelInstanceScheduleCandidate], worker: Worker
    ):
        """
        Generate advise for single worker multi gpus.
        """

        if candidates:
            return

        if not worker:
            return

        worker_allocatable = self._workers_allocatable_resource.get(worker.id)
        worker_gpus_allocatable_vram = sum(worker_allocatable.vram.values())

        tensor_split = list(worker_allocatable.vram.values())
        result = await self._calculate_model_resource_claim(
            offload=GPUOffloadEnum.Full, tensor_split=tensor_split
        )
        estimate = result.resource_claim_estimate
        sum_gpu_vram_claim = sum([vram.nonuma for vram in estimate.items[0].vrams])
        ram_claim = estimate.items[0].ram.nonuma

        if (
            worker_gpus_allocatable_vram > sum_gpu_vram_claim
            and worker_allocatable.ram > ram_claim
        ):
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_SINGLE_WORKER_MULTI_GPU_FULL_OFFLOADING,
                str(
                    ListMessageBuilder(
                        [
                            f"The model requires approximately {byte_to_gib(sum_gpu_vram_claim)} GiB VRAM and {byte_to_gib(ram_claim)} GiB RAM.",
                            f"The largest available worker provides {byte_to_gib(worker_gpus_allocatable_vram)} GiB VRAM and {byte_to_gib(worker_allocatable.ram)} GiB RAM.",
                            "Unable to find a combination of GPUs that satisfies the requirements when splitting by layers.",
                        ]
                    )
                ),
                reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
            )
        else:
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_SINGLE_WORKER_MULTI_GPU_FULL_OFFLOADING,
                str(
                    ListMessageBuilder(
                        [
                            f"The model requires approximately {byte_to_gib(sum_gpu_vram_claim)} GiB VRAM and {byte_to_gib(ram_claim)} GiB RAM.",
                            f"The largest available worker provides {byte_to_gib(worker_gpus_allocatable_vram)} GiB VRAM and {byte_to_gib(worker_allocatable.ram)} GiB RAM.",
                        ]
                    )
                ),
                reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
            )

    async def _advise_for_find_single_worker_partial_offloading_candidates(
        self, candidates: List[ModelInstanceScheduleCandidate], worker_id: int
    ):
        if candidates:
            return

        worker_allocatable = self._workers_allocatable_resource.get(worker_id)
        worker_gpus_allocatable_vram = sum(worker_allocatable.vram.values())

        if self._param_gpu_layers:
            # Multi-GPU
            tensor_split = list(worker_allocatable.vram.values())
            result = await self._calculate_model_resource_claim(
                offload=GPUOffloadEnum.Partial, tensor_split=tensor_split
            )
            estimate = result.resource_claim_estimate
            vram_claims, ram_claim = self._get_tensor_split_claim_with_layers(
                self._param_gpu_layers, False, estimate.items
            )
            sum_gpu_vram_claim = sum(vram_claims)

            if (
                worker_gpus_allocatable_vram > sum_gpu_vram_claim
                and worker_allocatable.ram > ram_claim
            ):
                self._event_collector.add(
                    EventLevelEnum.INFO,
                    EVENT_ACTION_SINGLE_WORKER_PARTIAL_OFFLOADING,
                    str(
                        ListMessageBuilder(
                            [
                                f"The model requires approximately {byte_to_gib(sum_gpu_vram_claim)} GiB VRAM and {byte_to_gib(ram_claim)} GiB RAM to offload {self._param_gpu_layers} layers.",
                                f"The largest available worker provides {byte_to_gib(worker_gpus_allocatable_vram)} GiB VRAM and {byte_to_gib(worker_allocatable.ram)} GiB RAM but unable to find a combination of GPUs that satisfies the requirements when splitting by layers. Try offloading fewer layers to lower resource required.",
                            ]
                        )
                    ),
                    reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
                )
            else:
                self._event_collector.add(
                    EventLevelEnum.INFO,
                    EVENT_ACTION_SINGLE_WORKER_PARTIAL_OFFLOADING,
                    str(
                        ListMessageBuilder(
                            [
                                f"The model requires approximately {byte_to_gib(sum_gpu_vram_claim)} GiB VRAM and {byte_to_gib(ram_claim)} GiB RAM to offload {self._param_gpu_layers} layers.",
                                f"The largest available worker provides {byte_to_gib(worker_gpus_allocatable_vram)} GiB VRAM and {byte_to_gib(worker_allocatable.ram)} GiB RAM. Try offloading fewer layers to lower resource required.",
                            ]
                        )
                    ),
                    reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
                )

    def _advise_for_find_single_worker_cpu_candidates(
        self, candidates: List[ModelInstanceScheduleCandidate], worker_id: int
    ):
        if candidates:
            return

        worker_allocatable = self._workers_allocatable_resource.get(worker_id)
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_CPU_OFFLOADING,
            str(
                ListMessageBuilder(
                    [
                        f"The model requires approximately {byte_to_gib(self._disable_offload_result_claim.items[0].ram.nonuma)} GiB RAM.",
                        f"The largest available worker provides {byte_to_gib(worker_allocatable.ram)} GiB RAM.",
                    ]
                )
            ),
            reason=EVENT_REASON_INSUFFICIENT_RESOURCES,
        )


def binary_search(arr, target):
    """
    Binary search the target in the arr.
    If the target is found, return the index of the target.

    Args:
        arr (List[int]): The input list, is a sorted list from smallest to largest.
        target (int): The target number.
    """
    if len(arr) == 0:
        return -1

    if arr[0] > target:
        return -1

    if arr[-1] < target:
        return len(arr) - 1

    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return high


def _get_max_offload_layers(candidates: List[ModelInstanceScheduleCandidate]) -> int:
    if not candidates:
        return 0

    return max(
        candidate.computed_resource_claim.offload_layers for candidate in candidates
    )


def _get_max_offload_layers_candidates(
    candidates: List[ModelInstanceScheduleCandidate],
) -> List[ModelInstanceScheduleCandidate]:
    if not candidates:
        return 0

    max_offload_layers = _get_max_offload_layers(candidates)
    return _filter_candidates_by_max_offload_layers(candidates, max_offload_layers)


def _filter_candidates_by_max_offload_layers(
    candidates: List[ModelInstanceScheduleCandidate], max_offload_layers
) -> List[ModelInstanceScheduleCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.computed_resource_claim.offload_layers == max_offload_layers
    ]


def _sort_and_group_worker_gpu_vram(
    workers_vram: List[Tuple[int, int]],
    gpus_allocatable_vram: List[Tuple[int, int, int]],
):
    sorted_workers = sorted(workers_vram, key=lambda item: item[1], reverse=True)
    sorted_gpus = sorted(gpus_allocatable_vram, key=lambda item: item[2], reverse=True)
    return sorted_workers, sorted_gpus
