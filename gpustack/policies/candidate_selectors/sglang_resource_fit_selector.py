import logging
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from transformers.utils import strtobool

from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
    EVENT_ACTION_AUTO_SINGLE_GPU,
    EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
    EVENT_ACTION_DEFAULT,
    EVENT_ACTION_MANUAL_MULTI,
    ModelParameters,
    RequestEstimateUsage,
    ScheduleCandidatesSelector,
)
from gpustack.policies.event_recorder.recorder import EventCollector, EventLevelEnum
from gpustack.policies.utils import (
    get_computed_ram_claim,
    ListMessageBuilder,
    group_worker_gpu_by_memory,
    WorkerGPUInfo,
    estimate_model_vram,
    get_model_ram_claim,
    group_workers_by_gpu_type,
    ram_not_enough,
    sort_workers_by_gpu_count,
    estimate_diffusion_model_vram,
    get_worker_allocatable_resource,
    sort_gpu_indexes_by_allocatable_rate,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
    ModelInstanceSubordinateWorker,
    CategoryEnum,
)
from gpustack.schemas.workers import Worker
from gpustack.config import Config
from gpustack.utils.command import (
    find_bool_parameter,
    find_parameter,
    find_int_parameter,
)
from gpustack.utils.unit import byte_to_gib, byte_to_mib

logger = logging.getLogger(__name__)


class SGLangResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        cfg: Config,
        model: Model,
        model_instances: List[ModelInstance],
    ):
        super().__init__(cfg, model, model_instances)

        self._vram_claim = 0
        self._ram_claim = 0
        self._messages = []
        self._event_collector = EventCollector(self._model, logger)

        self._param_mem_fraction_static = 0
        self._mem_fraction_static_by_gpu_type = {"*": 0}

        self._tp_size = 1
        self._pp_size = 1
        self._dp_size = 1
        self._chunked_prefill_size = None
        self._cuda_graph_max_bs = None

        # for multi worker schedule
        self._largest_multi_gpu_vram: int = 0
        self._largest_multi_gpu_total = 0
        self._largest_multi_gpu_utilization_satisfied_count = 0
        self._unsatisfied_gpu_messages: Dict[str, List[int]] = {}
        self._is_diffusion = CategoryEnum.IMAGE in self._model.categories

        world_size, strategies = (
            SGLangResourceFitSelector.get_world_size_from_backend_parameters(model)
        )
        self._set_gpu_count(world_size, strategies)

    async def _init_model_parameters(self, workers: List[Worker]):
        await super()._init_model_parameters(workers)
        self._validate_and_set_arguments()

    @staticmethod
    def get_world_size_from_backend_parameters(
        model: Model,
    ) -> Tuple[Optional[int], Optional[List[str]]]:
        tp = find_int_parameter(
            model.backend_parameters, ["tp-size", "tensor-parallel-size"]
        )
        pp = find_int_parameter(
            model.backend_parameters, ["pp-size", "pipeline-parallel-size"]
        )
        dp = find_int_parameter(
            model.backend_parameters, ["dp-size", "data-parallel-size"]
        )
        dp_attention = find_bool_parameter(
            model.backend_parameters, ["enable-dp-attention"]
        )
        if dp_attention:
            # For DP attention, it's using the TP group GPUs for data parallelism.
            # So we don't need to consider DP size for GPU count calculation.
            dp = None

        if tp or pp or dp:
            world_size = 1
            strategies = []
            if tp:
                strategies.append("tp")
                world_size *= tp
            if pp:
                strategies.append("pp")
                world_size *= pp
            if dp:
                strategies.append("dp")
                world_size *= dp

            return world_size, strategies

        return None, None

    def _get_nnodes(self) -> int:
        if self._model.backend_parameters:
            nnodes_param = find_parameter(self._model.backend_parameters, ["nnodes"])
            if nnodes_param:
                return int(nnodes_param)

        return 1

    def _validate_and_set_arguments(self):
        model = self._model
        self._tp_size = (
            find_int_parameter(
                model.backend_parameters, ["tp-size", "tensor-parallel-size"]
            )
            or 1
        )
        num_attention_heads = self._model_params.num_attention_heads
        if (
            self._tp_size
            and num_attention_heads
            and num_attention_heads % self._tp_size != 0
        ):
            raise ValueError(
                f"Total number of attention heads ({num_attention_heads})"
                " must be divisible by tp-size "
                f"({self._tp_size})."
            )

        self._pp_size = (
            find_int_parameter(
                model.backend_parameters, ["pp-size", "pipeline-parallel-size"]
            )
            or 1
        )
        self._dp_size = (
            find_int_parameter(
                model.backend_parameters, ["dp-size", "data-parallel-size"]
            )
            or 1
        )

        enable_mixed_chunk_param = find_parameter(
            model.backend_parameters, ["enable-mixed-chunk"]
        )
        enable_mixed_chunk = (
            strtobool(enable_mixed_chunk_param)
            if enable_mixed_chunk_param is not None
            else False
        )

        nnodes = self._get_nnodes()
        if self._tp_size or self._pp_size:
            world_size = int(self._tp_size or 1) * int(self._pp_size or 1)
            if world_size % nnodes != 0:
                raise ValueError(f"tp-size {world_size} must be divisible by nnodes")

        _speculative_algorithm = find_parameter(
            model.backend_parameters, ["speculative-algorithm"]
        )
        if self._pp_size and int(self._pp_size) > 1:
            if _speculative_algorithm is not None or enable_mixed_chunk:
                # We don't need to check overlap schedule. SGLang ignore this conflict and proceed.
                # Ref: https://github.com/sgl-project/sglang/blob/64480ec7124b8c23d9560746ca20415bfaf97a8e/python/sglang/srt/server_args.py#L1548-L1553
                raise ValueError(
                    "Pipeline parallelism is not compatible with overlap schedule, speculative decoding, mixed chunked prefill."
                )

        _enable_dp_attention = find_bool_parameter(
            model.backend_parameters, ["enable-dp-attention"]
        )
        if self._dp_size and int(self._dp_size) > 1 and nnodes != 1:
            if not _enable_dp_attention:
                raise ValueError(
                    "multi-node data parallel is not supported unless dp attention!"
                )

        if _speculative_algorithm is not None and enable_mixed_chunk:
            raise ValueError("enable_mixed_chunk is required for speculative decoding")

        if message := self._check_tp_size_divisibility(self._tp_size):
            raise ValueError(message + " Consider adjusting your tp-size value.")

        mem_fraction_static = find_parameter(
            self._model.backend_parameters, ["mem-fraction-static"]
        )
        if mem_fraction_static:
            self._param_mem_fraction_static = float(mem_fraction_static)

    def _cal_effective_vram(self, gpu_type: str) -> float:
        """Calculate effective VRAM considering SGLang's memory management."""
        if (
            self._is_diffusion
            or self._mem_fraction_static_by_gpu_type.get(gpu_type) == 0
        ):
            return self._vram_claim
        return self._vram_claim / self._mem_fraction_static_by_gpu_type.get(gpu_type)

    def _set_messages(self):
        """Set scheduling messages for SGLang."""
        if self._messages:
            return

        event_messages = {
            EVENT_ACTION_DEFAULT: "",
            EVENT_ACTION_MANUAL_MULTI: "",
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU: "",
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU: "",
            EVENT_ACTION_AUTO_SINGLE_GPU: "",
        }

        for event in self._event_collector.events:
            event_messages[event.action] = event.message

        messages = event_messages[EVENT_ACTION_DEFAULT] + "\n"
        for action in [
            EVENT_ACTION_MANUAL_MULTI,
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            EVENT_ACTION_AUTO_SINGLE_GPU,
        ]:
            if event_messages[action]:
                messages += event_messages[action]
                break

        self._messages.append(messages)

    def _add_message(self, message: str):
        self._messages.append(message)

    def get_messages(self) -> List[str]:
        return self._messages

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement for SGLang.
        """

        # Initialize model parameters.
        await self._init_model_parameters(workers)

        if self._is_diffusion:
            self._vram_claim = await estimate_diffusion_model_vram(
                self._model, self._config.huggingface_token, workers
            )
        else:
            self.cal_mem_fraction_static(workers)
            self._vram_claim = await estimate_model_vram(
                self._model, self._config.huggingface_token, workers
            )
        self._ram_claim = get_model_ram_claim(self._model)

        logger.info(
            f"Calculated SGLang resource claim for model {self._model.readable_source}, "
            f"VRAM claim: {self._vram_claim}, RAM claim: {self._ram_claim}"
        )

        default_msg_list = ListMessageBuilder(
            f"The model requires approximately {byte_to_gib(self._vram_claim)} GiB of VRAM"
            f"{f' and {byte_to_gib(self._ram_claim)} GiB of RAM' if self._ram_claim > 0 else ''}."
        )
        max_mem_fraction_static = (
            max(self._mem_fraction_static_by_gpu_type.values())
            if self._mem_fraction_static_by_gpu_type
            else 0
        )
        if max_mem_fraction_static != 0 and not self._is_diffusion:
            default_msg_list.append(
                f"With --mem-fraction-static={max_mem_fraction_static}, "
                f"all GPUs combined need to provide at least {byte_to_gib(int(self._vram_claim / max_mem_fraction_static))} GiB of total VRAM "
                f"and each GPU needs {int(max_mem_fraction_static * 100)}% of allocatable VRAM."
            )
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_DEFAULT,
            str(default_msg_list),
        )

        candidate_functions = [
            self.find_manual_gpu_selection_candidates,
            self.find_single_worker_single_gpu_full_offloading_candidates,
            self.find_single_worker_multi_gpu_full_offloading_candidates,
            self.find_multi_worker_multi_gpu_candidates,
        ]

        sort_workers_by_gpu_count(workers)

        for candidate_func in candidate_functions:

            if self.should_skip_candidate_func(candidate_func):
                continue

            logger.debug(
                f"SGLang model {self._model.readable_source}, filter candidates with resource fit selector: {candidate_func.__name__}"
            )

            candidates = candidate_func(workers)

            if len(candidates) == 1 and candidates[0].overcommit:
                self._set_messages()

            if candidates:
                return candidates

        self._set_messages()
        return []

    def cal_mem_fraction_static(self, workers: List[Worker]):
        """Calculate mem_fraction_static for SGLang for all type gpus."""
        workers_by_gpu_type = group_workers_by_gpu_type(workers)
        valid_gpu_types = (
            set(self._selected_gpu_indexes_by_gpu_type_and_worker.keys())
            if self._selected_gpu_indexes_by_gpu_type_and_worker
            else set(workers_by_gpu_type.keys())
        )
        self._mem_fraction_static_by_gpu_type.update(
            {
                gpu_type: (
                    self._param_mem_fraction_static
                    if self._param_mem_fraction_static > 0
                    else MemFractionStaticCalculator(
                        self._model,
                        self._model_instances,
                        self._model_params,
                        gpu_type,
                        self._selected_gpu_indexes_by_gpu_type_and_worker,
                    )._cal_mem_fraction_static(workers_of_type)
                )
                for gpu_type, workers_of_type in workers_by_gpu_type.items()
                if gpu_type in valid_gpu_types
            }
        )

    def should_skip_candidate_func(self, candidate_func) -> bool:
        # Skip conditions for manual GPU selection.
        if (
            self._selected_gpu_workers
            and candidate_func != self.find_manual_gpu_selection_candidates
        ):
            return True

        # Skip conditions for distributed inference.
        if (
            not self._model.distributed_inference_across_workers
            and candidate_func == self.find_multi_worker_multi_gpu_candidates
        ):
            return True

        # SGLang Diffusion unsupported multi-worker
        if (
            CategoryEnum.IMAGE in self._model.categories
            and candidate_func == self.find_multi_worker_multi_gpu_candidates
        ):
            return True

        return False

    def find_manual_gpu_selection_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        request = RequestEstimateUsage(
            ram=self._ram_claim,
            vram=self._vram_claim,
        )
        if self._is_diffusion:
            request.vram = request.vram * self._gpu_count
        return self._find_manual_gpu_selection_candidates(
            workers,
            self._mem_fraction_static_by_gpu_type,
            request,
            self._event_collector,
        )

    def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single GPU candidates for SGLang.
        This function only handles automatic GPU selection.
        """
        if self._gpu_count is not None and self._gpu_count > 1:
            # Skip multi-GPU selection
            return []

        # Auto selection only
        candidates = []
        workers_by_gpu_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_by_gpu_type.items():
            for worker in workers_of_type:
                worker_candidates = (
                    self._find_single_worker_single_gpu_full_offloading_candidates(
                        worker, gpu_type
                    )
                )
                candidates.extend(worker_candidates)

        return candidates

    def _find_single_worker_single_gpu_full_offloading_candidates(
        self, worker: Worker, gpu_type: Optional[str] = None
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with worker.
        requires: worker.status.gpu_devices is not None
        """

        candidates = []

        allocatable = self.get_worker_allocatable_resource(worker, gpu_type)

        if ram_not_enough(self._ram_claim, allocatable):
            return []

        if not worker.status.gpu_devices:
            return []

        largest_single_gpu_vram = 0
        largest_single_gpu_utilization = 0

        for _, gpu in enumerate(worker.status.gpu_devices):
            gpu_index = gpu.index
            allocatable_vram = allocatable.vram.get(gpu_index, 0)
            allocatable_gpu_utilization = allocatable_vram / gpu.memory.total

            if allocatable_vram > largest_single_gpu_vram:
                largest_single_gpu_vram = allocatable_vram
                largest_single_gpu_utilization = allocatable_gpu_utilization
            exceeds_vram = self._vram_claim > gpu.memory.total * (
                self._mem_fraction_static_by_gpu_type.get(gpu_type)
                if not self._is_diffusion
                else 1
            )
            exceeds_memory_utilization = (
                not self._is_diffusion
                and allocatable_gpu_utilization
                < self._mem_fraction_static_by_gpu_type.get(gpu_type)
            )
            if exceeds_vram or exceeds_memory_utilization:
                continue

            request_usage = (
                RequestEstimateUsage(ram=self._ram_claim, vram=self._vram_claim)
                if self._is_diffusion
                else None
            )
            vram_claim = self._get_worker_resource_claim(
                worker,
                [gpu_index],
                (
                    self._mem_fraction_static_by_gpu_type.get(gpu_type)
                    if not self._is_diffusion
                    else 0
                ),
                request=request_usage,
                gpu_type=gpu_type,
            )
            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_type=gpu.type,
                    gpu_indexes=[gpu_index],
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=get_computed_ram_claim(self._model, vram_claim),
                        vram_utilization=self._mem_fraction_static_by_gpu_type.get(
                            gpu_type
                        ),
                    ),
                )
            )

        if not candidates:
            event_msg = f"The current available GPU only has {byte_to_gib(largest_single_gpu_vram)} GiB allocatable VRAM."
            if (
                self._mem_fraction_static_by_gpu_type.get(gpu_type) != 0
                and not self._is_diffusion
            ):
                event_msg = (
                    event_msg.rstrip(".")
                    + f" ({(largest_single_gpu_utilization * 100):.2f}%)."
                )
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_AUTO_SINGLE_GPU,
                str(ListMessageBuilder(event_msg)),
            )

        return candidates

    def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi GPU candidates for SGLang.
        This function only handles automatic GPU selection.
        """
        candidates = []

        workers_by_gpu_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_by_gpu_type.items():
            for worker in workers_of_type:
                worker_candidates = (
                    self._find_single_worker_multi_gpu_full_offloading_candidates(
                        worker, gpu_type
                    )
                )
                candidates.extend(worker_candidates)

        return candidates

    def _find_single_worker_multi_gpu_full_offloading_candidates(  # noqa: C901
        self, worker: Worker, gpu_type: Optional[str] = None
    ) -> List[ModelInstanceScheduleCandidate]:
        """Find single worker multi GPU candidates for a specific worker."""
        if not worker.status.gpu_devices or len(worker.status.gpu_devices) < 2:
            return []

        # SGLang performs VRAM balancing checks. We group all GPUs based on available VRAM capacity
        gpu_group = group_worker_gpu_by_memory(
            [worker],
            model_instances=self._model_instances,
            ram_claim=self._ram_claim,
            gpu_type=gpu_type,
        )

        for info in gpu_group:
            gpu_list = info
            if not self._is_diffusion and any(
                gpu.allocatable_vram / gpu.gpu_device.memory.total
                < self._mem_fraction_static_by_gpu_type.get(gpu_type)
                for gpu in gpu_list
            ):
                continue
            if self._is_diffusion and any(
                gpu.allocatable_vram < self._cal_effective_vram(gpu_type)
                for gpu in gpu_list
            ):
                continue
            total_allocatable_vram = sum(gpu.allocatable_vram for gpu in gpu_list)

            if not gpu_list or len(gpu_list) < 2:
                continue

            if total_allocatable_vram > self._largest_multi_gpu_vram:
                self._largest_multi_gpu_vram = total_allocatable_vram
                self._largest_multi_gpu_utilization_satisfied_count = len(gpu_list)
                self._largest_multi_gpu_total = len(worker.status.gpu_devices)

            if total_allocatable_vram < self._cal_effective_vram(gpu_type):
                continue

            # Sort by vram in descending order
            sorted_gpu_devices: list[WorkerGPUInfo] = sorted(
                gpu_list,
                key=lambda gpu: gpu.allocatable_vram,
                reverse=True,
            )

            vram_sum = 0
            gpu_sum = 0
            gpu_indexes = []
            vram_claim: Dict[int, int] = {}
            found_candidate = False
            for _, gpu_device in enumerate(sorted_gpu_devices):
                gpu = gpu_device.gpu_device
                gpu_indexes.append(gpu.index)
                vram_claim[gpu.index] = int(
                    gpu_device.allocatable_vram
                    if self._is_diffusion
                    else int(
                        gpu.memory.total
                        * self._mem_fraction_static_by_gpu_type.get(gpu_type)
                    )
                )
                gpu_sum += 1
                vram_sum += vram_claim[gpu.index]

                if not self._is_tp_size_divisible(gpu_sum):
                    continue

                if self._gpu_count and gpu_sum >= self._gpu_count:
                    if vram_sum >= self._vram_claim:
                        found_candidate = True
                    # if self._gpu_count is set, cannot return more than gpu_count
                    break

                if (not self._gpu_count) and vram_sum >= self._vram_claim:
                    found_candidate = True
                    break

            if found_candidate:
                return [
                    ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_type=gpu_type,
                        gpu_indexes=gpu_indexes,
                        computed_resource_claim=ComputedResourceClaim(
                            vram=vram_claim,
                            ram=get_computed_ram_claim(self._model, vram_claim),
                            vram_utilization=self._mem_fraction_static_by_gpu_type.get(
                                gpu_type
                            ),
                        ),
                    )
                ]
        event_msg_list = []
        if message := self._check_tp_size_divisibility(
            self._largest_multi_gpu_utilization_satisfied_count
        ):
            event_msg_list.append(message)

        if len(event_msg_list) == 0:
            event_msg = (
                f"The largest available worker has {byte_to_gib(self._largest_multi_gpu_vram):.2f} GiB allocatable VRAM."
                if not self._is_diffusion
                else f"SGLang Diffusion requires each GPU to provide {byte_to_gib(int(self._cal_effective_vram(gpu_type)))} GiB of allocatable VRAM when running in parallel."
            )
            if (
                self._mem_fraction_static_by_gpu_type.get(gpu_type) != 0
                and not self._is_diffusion
            ):
                effective_vram = (
                    byte_to_gib(
                        int(
                            self._largest_multi_gpu_vram
                            * self._mem_fraction_static_by_gpu_type.get(gpu_type)
                            * self._largest_multi_gpu_utilization_satisfied_count
                            / self._largest_multi_gpu_total
                        )
                    )
                    if self._largest_multi_gpu_total > 0
                    else 0
                )
                event_msg = (
                    event_msg.rstrip(".")
                    + f", {self._largest_multi_gpu_utilization_satisfied_count}/{self._largest_multi_gpu_total} of GPUs meet the VRAM utilization ratio, providing {effective_vram} GiB of allocatable VRAM."
                )
            event_msg_list.append(event_msg)

        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            str(ListMessageBuilder(event_msg_list)),
        )

        return []

    def find_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker], gpu_type: Optional[str] = None
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find multi worker multi GPU candidates for SGLang.
        This function only handles automatic multi-worker selection.
        """

        candidates = []

        workers_by_gpu_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_by_gpu_type.items():
            gpu_group = group_worker_gpu_by_memory(
                workers_of_type,
                model_instances=self._model_instances,
                ram_claim=self._ram_claim,
                gpu_type=gpu_type,
            )

            for gpu_list in gpu_group:
                if len(gpu_list) <= 1:
                    continue
                if any(
                    not self._is_diffusion
                    and gpu.allocatable_vram / gpu.gpu_device.memory.total
                    < self._mem_fraction_static_by_gpu_type.get(gpu_type)
                    for gpu in gpu_list
                ):
                    continue

                worker_gpu_cnt = {}
                for gpu_info in gpu_list:
                    if not worker_gpu_cnt.get(gpu_info.worker_id):
                        worker_gpu_cnt[gpu_info.worker_id] = 0
                    worker_gpu_cnt[gpu_info.worker_id] += 1
                first_cnt = 0
                # workers must with the same number of GPUs
                for gpu_cnt in worker_gpu_cnt.values():
                    if not first_cnt:
                        first_cnt = gpu_cnt
                    if first_cnt != gpu_cnt:
                        first_cnt = 0
                        break
                if not first_cnt:
                    continue

                workers_candidates = self.auto_select_multi_worker_multi_gpu_candidates(
                    workers_of_type, gpu_type
                )
                if workers_candidates:
                    candidates.extend(workers_candidates)
                    break  # only need one group since gpu_list is sorted by vram in ascending order

        return candidates

    def auto_select_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker], gpu_type: Optional[str] = None
    ) -> List[ModelInstanceScheduleCandidate]:
        """Auto select multi worker multi GPU candidates for SGLang."""
        candidates = []

        if not workers or len(workers) < 2:
            return candidates

        sort_workers_by_gpu_count(workers)

        workers_by_gpu_count_dict = defaultdict(list)
        for worker in workers:
            if not worker.status or not worker.status.gpu_devices:
                continue

            workers_by_gpu_count_dict[len(worker.status.gpu_devices)].append(worker)

        # Store the optimal combination info to show
        workers_combination: List[Worker] = []
        largest_vram = 0
        worker_count = 0
        device_count_per_worker = 0

        # Loop through worker groups with the same number of GPUs.
        for gpu_count, worker_group in workers_by_gpu_count_dict.items():
            if len(worker_group) < 2:
                continue

            selected_workers: List[Worker] = []
            gpu_sum = 0
            vram_sum = 0
            for worker in worker_group:
                allocatable = self.get_worker_allocatable_resource(worker, gpu_type)

                if ram_not_enough(self._ram_claim, allocatable):
                    # The RAM resource(for extended KV cache) is required per worker.
                    # Skip the worker if it does not satisfy the RAM requirement.
                    continue

                if not self._is_diffusion and any(
                    allocatable.vram.get(gpu.index, 0) / gpu.memory.total
                    < self._mem_fraction_static_by_gpu_type.get(gpu_type)
                    for gpu in worker.status.gpu_devices
                ):
                    continue
                selected_workers.append(worker)
                gpu_sum += gpu_count
                vram_sum += sum(
                    int(
                        int(
                            gpu.memory.total
                            * self._mem_fraction_static_by_gpu_type.get(gpu_type)
                        )
                        if not self._is_diffusion
                        else allocatable.vram.get(gpu.index, 0)
                    )
                    for gpu in worker.status.gpu_devices
                )

                if not self._is_tp_size_divisible(gpu_count):
                    continue

                if vram_sum >= self._vram_claim:
                    return [
                        _create_candidate(
                            self._model,
                            selected_workers,
                            self._mem_fraction_static_by_gpu_type.get(gpu_type),
                        )
                    ]
            if vram_sum > largest_vram:
                workers_combination = selected_workers
                largest_vram = vram_sum
                worker_count = len(worker_group)
                device_count_per_worker = gpu_count

        # Nothing can be return, construct scheduling message
        event_message = ListMessageBuilder([])
        if workers_combination:
            worker_names = [worker.name for worker in workers_combination]
            worker_names_msg = (
                str(worker_names[:3]).rstrip("]")
                + f"...(more {len(worker_names) - 3})]"
                if len(worker_names) > 3
                else str(worker_names)
            )
            message = f"The optimal combination {worker_names_msg} provides {byte_to_gib(largest_vram)} GiB of allocatable VRAM."
            if worker_count - len(workers_combination) > 0:
                message += f" There are {worker_count - len(workers_combination)} {'workers' if worker_count - len(workers_combination) > 1 else 'worker'} that can provide {device_count_per_worker} {'GPUs' if device_count_per_worker > 1 else 'GPU'}, as the workers in the combination, but some GPUs among them fail to meet requirements."
            event_message.append(message)

        event_message.append(
            "Cannot find a suitable worker combination to run the model in distributed mode. "
            "If you are confident that the resources are sufficient, you may manually schedule the model by selecting the workers and GPUs."
        )
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
            str(event_message),
        )

        return []


def _create_candidate(
    model: Model,
    selected_workers: List[Worker],
    mem_fraction_static: float = 0.9,
) -> ModelInstanceScheduleCandidate:
    """Create a candidate with SGLang-specific parameters and primary node confirmation."""
    if not selected_workers:
        raise ValueError("No workers provided for candidate creation")

    # Primary worker is the first one (with most GPUs)
    primary_worker = selected_workers[0]
    subordinate_workers = []

    # Calculate primary worker resources
    primary_gpu_indexes = []
    primary_vram_claim = {}
    gpu_type = None

    for gpu in primary_worker.status.gpu_devices:
        primary_gpu_indexes.append(gpu.index)
        primary_vram_claim[gpu.index] = int(gpu.memory.total * mem_fraction_static)
        gpu_type = gpu.type

    # Process subordinate workers if any
    if len(selected_workers) > 1:
        for worker in selected_workers[1:]:
            gpu_indexes = []
            vram_claim = {}

            if worker.status.gpu_devices:
                for gpu in worker.status.gpu_devices:
                    gpu_indexes.append(gpu.index)
                    vram_claim[gpu.index] = int(gpu.memory.total * mem_fraction_static)

            subordinate_worker = ModelInstanceSubordinateWorker(
                worker_id=worker.id,
                worker_name=worker.name,
                worker_ip=worker.ip,
                worker_ifname=worker.ifname,
                total_gpus=len(gpu_indexes),
                gpu_type=gpu_type,
                gpu_indexes=gpu_indexes,
                computed_resource_claim=ComputedResourceClaim(
                    vram=vram_claim,
                    ram=get_computed_ram_claim(model, vram_claim),
                    vram_utilization=mem_fraction_static,
                ),
            )
            subordinate_workers.append(subordinate_worker)

    computed_resource_claim = ComputedResourceClaim(
        vram=primary_vram_claim,
        ram=get_computed_ram_claim(model, primary_vram_claim),
    )

    return ModelInstanceScheduleCandidate(
        worker=primary_worker,
        gpu_type=gpu_type,
        gpu_indexes=primary_gpu_indexes,
        computed_resource_claim=computed_resource_claim,
        subordinate_workers=subordinate_workers if subordinate_workers else None,
    )


class MemFractionStaticCalculator:
    _model: Model
    _model_instances: List[ModelInstance]
    _gpu_type: Optional[str]
    _chunked_prefill_size: Optional[int]
    _cuda_graph_max_bs: Optional[int]
    _enable_dp_attention: Optional[bool]
    _speculative_algorithm: Optional[str]
    _tp_size: int
    _pp_size: int
    _dp_size: int
    _max_pp_size: int

    # Model hyperparameters.
    _model_params: ModelParameters

    _selected_gpu_indexes_by_gpu_type_and_worker: Dict[str, Dict[int, List[int]]] = {}

    def __init__(
        self,
        model: Model,
        model_instances: List[ModelInstance],
        model_params: ModelParameters,
        gpu_type: str,
        selected_gpu_indexes_by_gpu_type_and_worker: Dict[str, Dict[int, List[int]]],
    ) -> None:
        self._model = model
        self._model_instances = model_instances
        self._model_params = model_params
        self._gpu_type = gpu_type
        self._selected_gpu_indexes_by_gpu_type_and_worker = (
            selected_gpu_indexes_by_gpu_type_and_worker
        )

        self._chunked_prefill_size = find_int_parameter(
            self._model.backend_parameters, ["chunked-prefill-size"]
        )
        self._cuda_graph_max_bs = find_int_parameter(
            self._model.backend_parameters, ["cuda-graph-max-bs"]
        )
        self._enable_dp_attention = find_bool_parameter(
            model.backend_parameters, ["enable-dp-attention"]
        )
        self._speculative_algorithm = find_parameter(
            model.backend_parameters, ["speculative-algorithm"]
        )
        self._tp_size = (
            find_int_parameter(
                model.backend_parameters, ["tp-size", "tensor-parallel-size"]
            )
            or 1
        )
        self._pp_size = (
            find_int_parameter(
                model.backend_parameters, ["pp-size", "pipeline-parallel-size"]
            )
            or 1
        )
        self._dp_size = (
            find_int_parameter(
                model.backend_parameters, ["dp-size", "data-parallel-size"]
            )
            or 1
        )

        self._max_pp_size = max(
            self._pp_size,
            len(
                self._selected_gpu_indexes_by_gpu_type_and_worker.get(
                    self._gpu_type, {}
                )
            ),
        )

        self._select_tp_size = max(
            [self._tp_size]
            + [
                len(gpu_indexes)
                for gpu_indexes in self._selected_gpu_indexes_by_gpu_type_and_worker.get(
                    self._gpu_type, {}
                ).values()
            ]
        )

    def _cal_mem_fraction_static(self, workers: List[Worker]) -> float:  # noqa: C901
        """
        Adapted from sglang's server_args memory fraction logic.
        Logic of SGLang set default mem_fraction_static:
        https://github.com/sgl-project/sglang/blob/037c3982af4a996f41b38cacf59f0be24b8699f8/python/sglang/srt/server_args.py#L751-L919
        note: we largely maintained the same code structure and logic, except we removed some assignments unrelated to the calculation of mem_fraction_static.
        Args:
            workers: List of workers used to determine GPU memory characteristics, the input workers should only contain same type GPUs.
        """
        if find_parameter(self._model.backend_parameters, ["mem-fraction-static"]):
            return

        is_npu = self._is_npu(workers)
        gpu_mem_bytes = self._get_min_gpu_sum(workers)
        gpu_mem = byte_to_mib(gpu_mem_bytes)
        # Step 1: Use the minimum GPU memory of all workers to calculate _chunked_prefill_size and _cuda_graph_max_bs.
        if gpu_mem:
            if gpu_mem < 20 * 1024:
                # T4, 4080
                # (_chunked_prefill_size 2k, _cuda_graph_max_bs 8)
                if self._chunked_prefill_size is None:
                    self._chunked_prefill_size = 2048
                if self._cuda_graph_max_bs is None:
                    self._cuda_graph_max_bs = 8
            elif is_npu and gpu_mem < 32 * 1024:
                # Atlas A2B4
                # (_chunked_prefill_size 32k, _cuda_graph_max_bs 16 if tp < 4 else 64)
                if self._chunked_prefill_size is None:
                    self._chunked_prefill_size = 32768
                if self._cuda_graph_max_bs is None:
                    if self._get_max_tp_size() < 4:
                        self._cuda_graph_max_bs = 16
                    else:
                        self._cuda_graph_max_bs = 64
            elif gpu_mem < 35 * 1024:
                # A10, 4090, 5090
                # (_chunked_prefill_size 2k, _cuda_graph_max_bs 24 if tp < 4 else 80)
                if self._chunked_prefill_size is None:
                    self._chunked_prefill_size = 2048
                if self._cuda_graph_max_bs is None:
                    # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM < 35GB, you can either disable cuda graph or set `_cuda_graph_max_bs` to a very small value to reduce the memory overhead of creating cuda graphs, with almost no impact on performance.
                    # However, when serving models with TP4 or TP8, we need to enable cuda graph to maintain high performance. In this case, we can set `_cuda_graph_max_bs` to 80 (half of the default value 160) to reduce the memory overhead of creating cuda graphs. Looking at the logs
                    # from TP4 serving of qwen2-72b, a value of 80 is sufficient and can reduce the memory overhead of creating cuda graphs on lower-end GPUs compared to the original 160, avoiding OOM issues.
                    if self._get_max_tp_size() < 4:
                        self._cuda_graph_max_bs = 24
                    else:
                        self._cuda_graph_max_bs = 80
            elif gpu_mem < 60 * 1024:
                # A100 (40GB), L40,
                # (_chunked_prefill_size 4k, _cuda_graph_max_bs 32 if tp < 4 else 160)
                if self._chunked_prefill_size is None:
                    self._chunked_prefill_size = 4096
                if self._cuda_graph_max_bs is None:
                    if self._get_max_tp_size() < 4:
                        self._cuda_graph_max_bs = 32
                    else:
                        self._cuda_graph_max_bs = 160
            elif is_npu and gpu_mem < 64 * 1024:
                # Atlas A2 and Atlas A3
                # (_chunked_prefill_size 32k, _cuda_graph_max_bs 64 if tp < 4 else 128)
                if self._chunked_prefill_size is None:
                    self._chunked_prefill_size = 32768
                if self._cuda_graph_max_bs is None:
                    if self._get_max_tp_size() < 4:
                        self._cuda_graph_max_bs = 64
                    else:
                        self._cuda_graph_max_bs = 128
            elif gpu_mem < 90 * 1024:
                # H100, A100
                # (_chunked_prefill_size 8k, _cuda_graph_max_bs 256 if tp < 4 else 512)
                if self._chunked_prefill_size is None:
                    self._chunked_prefill_size = 8192
                if self._cuda_graph_max_bs is None:
                    if self._get_max_tp_size() < 4:
                        self._cuda_graph_max_bs = 256
                    else:
                        self._cuda_graph_max_bs = 512
            elif gpu_mem < 160 * 1024:
                # H20, H200
                # (_chunked_prefill_size 8k, _cuda_graph_max_bs 256 if tp < 4 else 512)
                if self._chunked_prefill_size is None:
                    self._chunked_prefill_size = 8192
                if self._cuda_graph_max_bs is None:
                    if self._get_max_tp_size() < 4:
                        self._cuda_graph_max_bs = 256
                    else:
                        self._cuda_graph_max_bs = 512
            else:
                # B200, MI300
                # (_chunked_prefill_size 16k, _cuda_graph_max_bs 512)
                if self._chunked_prefill_size is None:
                    self._chunked_prefill_size = 16384
                if self._cuda_graph_max_bs is None:
                    self._cuda_graph_max_bs = 512
        else:
            # Fallback defaults when gpu_mem is None
            if self._chunked_prefill_size is None:
                self._chunked_prefill_size = 4096
            if self._cuda_graph_max_bs is None:
                self._cuda_graph_max_bs = 160

        # Step 2: Calculate reserved memory by other configs
        # Constant meta data (e.g., from attention backend)
        reserved_mem = 512
        # For activation during large prefill
        if self._chunked_prefill_size > 0:
            reserved_mem += max(self._chunked_prefill_size, 2048) * 1.5

        # For cuda graphs
        reserved_mem += self._cuda_graph_max_bs * 2
        # Some adjustments for large parallel size
        reserved_mem += self._get_max_tp_size() * self._max_pp_size / 8 * 1024

        if self._enable_dp_attention:
            # DP attention needs more padding for some operations
            reserved_mem += self._cuda_graph_max_bs * self._dp_size * 3

            # DP attention uses much more memory for large cuda graph max bs,
            # likely due to some inefficiencies in torch allocator or our implementation.
            # So we need to reserve more memory.
            if self._cuda_graph_max_bs > 300:
                reserved_mem += self._cuda_graph_max_bs * self._dp_size * 1.5

        if gpu_mem is not None and gpu_mem > 60 * 1024:
            reserved_mem = max(reserved_mem, 10 * 1024)

        if self._speculative_algorithm is not None:
            if self._speculative_algorithm == "STANDALONE":
                # standalonedraft model and cuda graphs
                reserved_mem += 6 * 1024
            elif self._speculative_algorithm != "NGRAM":
                # eagle draft models and cuda graphs
                reserved_mem += 2 * 1024

        # For piecewise cuda graphs
        enable_piecewise_cuda_graph = find_parameter(
            self._model.backend_parameters, ["enable-piecewise-cuda-graph"]
        )
        if enable_piecewise_cuda_graph:
            piecewise_cuda_graph_max_tokens = (
                find_int_parameter(
                    self._model.backend_parameters, ["piecewise-cuda-graph-max-tokens"]
                )
                or 0
            )
            reserved_mem += piecewise_cuda_graph_max_tokens // 4

        _mem_fraction_static = (
            round((gpu_mem - reserved_mem) / gpu_mem, 3)
            if gpu_mem is not None and gpu_mem > 0
            else 0.88
        )

        # Step 3: adjust mem_fraction_static for VL models
        # Multimodal models need more memory for the image processing,
        # so we adjust the mem_fraction_static accordingly.
        model_config = self._model_params
        vision_config = getattr(model_config, "vision_config", None)
        if vision_config is not None:
            # roughly reduce the mem_fraction_static base on params of Vit
            original_server_arg_mem_fraction = _mem_fraction_static
            # a base mem_fraction_static factor for regular Vit
            base_mem_fraction_reduction_ratio = 0.95

            vit_num_layers = getattr(vision_config, "num_hidden_layers", 24)
            vit_hidden_size = getattr(vision_config, "hidden_size", 1024)

            # baseline ViT params (ViT-L/14)
            baseline_vit_layers = 24
            baseline_vit_hidden_size = 1024

            # weight params count
            current_complexity_score = vit_num_layers * (vit_hidden_size**2)
            baseline_complexity_score = baseline_vit_layers * (
                baseline_vit_hidden_size**2
            )
            complexity_ratio = (
                current_complexity_score / baseline_complexity_score
                if baseline_complexity_score > 0
                else 1.0
            )

            # every time the complexity grows 100%, adjust final factor for 10%
            sensitivity_scale = 0.1
            dynamic_adjustment_factor = 1.0 - sensitivity_scale * (
                complexity_ratio - 1.0
            )
            dynamic_adjustment_factor = max(0.8, min(1.05, dynamic_adjustment_factor))

            final_overall_factor = (
                base_mem_fraction_reduction_ratio * dynamic_adjustment_factor
            )

            _mem_fraction_static = round(
                original_server_arg_mem_fraction * final_overall_factor, 3
            )

        return _mem_fraction_static

    def _is_npu(self, workers: List[Worker]) -> bool:
        """
        Check if the selected GPU type is NPU
        """
        is_npu = self._gpu_type == 'cann' or any(
            ((gpu.vendor or '').lower() == 'ascend')
            or ((gpu.type or '').lower() == 'cann')
            for w in workers
            for gpu in ((w.status and w.status.gpu_devices) or [])
        )
        return is_npu

    def _get_min_gpu_sum(self, workers: List[Worker]) -> int:
        """
        Get the min GPU memory of input workers
        """
        use_manual = self._model.gpu_selector is not None
        totals: List[int] = []
        for worker in workers:
            if not worker.status or not worker.status.gpu_devices:
                continue

            # If gpus_per_replica is set, choose top-N GPUs by allocatable VRAM
            selected = None
            if use_manual:
                # Pre-selected GPU indexes for this worker when manual selection is used
                selected = self._selected_gpu_indexes_by_gpu_type_and_worker.get(
                    self._gpu_type, {}
                ).get(worker.name)

                if self._model.gpu_selector.gpus_per_replica:
                    allocatable = get_worker_allocatable_resource(
                        self._model_instances, worker, self._gpu_type
                    )
                    sorted_gpu_indexes = [
                        idx
                        for idx in sort_gpu_indexes_by_allocatable_rate(
                            worker, allocatable.vram, gpu_type=self._gpu_type
                        )
                        if idx in selected
                    ]

                    selected = sorted_gpu_indexes[
                        : self._model.gpu_selector.gpus_per_replica
                    ]

            # Traverse GPUs for this worker and respect manual selection if present
            for gpu in worker.status.gpu_devices:
                if selected is not None and gpu.index not in selected:
                    continue
                total = gpu.memory.total if (gpu.memory and gpu.memory.total) else 0
                totals.append(total)
        return min(totals) if totals else 0

    def _get_max_tp_size(self) -> int:
        cmp_list = [self._tp_size, self._select_tp_size]
        if self._model.gpu_selector and self._model.gpu_selector.gpus_per_replica:
            cmp_list.append(self._model.gpu_selector.gpus_per_replica)
        return max(cmp_list)
