import json
from collections import defaultdict
import logging
import re
from typing import Dict, List, Optional, Tuple
from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScheduleCandidate,
)
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
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
    estimate_model_vram,
    ram_not_enough,
    get_model_ram_claim,
    get_computed_ram_claim,
    sort_workers_by_gpu_count,
)
from gpustack.schemas.models import (
    CategoryEnum,
    ComputedResourceClaim,
    Model,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.workers import GPUDevicesInfo, Worker
from gpustack.config import Config
from gpustack.utils.command import find_parameter, find_int_parameter
from gpustack.utils.unit import byte_to_gib

logger = logging.getLogger(__name__)


def parse_model_size_by_name(model_name: str) -> int:
    """
    Parse the model size from the model name.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model_name)
    if match:
        size_in_billions = float(match.group(1))
        return int(size_in_billions * 1e9)
    else:
        raise ValueError(f"Cannot parse model size from model name: {model_name}")


class VLLMResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        cfg: Config,
        model: Model,
    ):
        super().__init__(cfg, model)

        self._vram_claim = 0
        self._ram_claim = 0
        self._largest_single_gpu_vram = 0
        self._largest_single_gpu_vram_utilization = 0
        self._largest_multi_gpu_vram = 0
        self._largest_multi_gpu_total = 0
        self._largest_multi_gpu_utilization_satisfied_count = 0

        self._messages = []
        self._event_collector = EventCollector(self._model, logger)
        self._workers_allocatable_resource: Dict[int, Allocatable] = {}
        self._worker_name_to_vram: Dict[str, Dict[int, int]] = {}

        self._unsatisfied_gpu_messages: Dict[str, List[int]] = {}

        world_size, strategies = (
            VLLMResourceFitSelector.get_world_size_from_backend_parameters(model)
        )
        self._set_gpu_count(world_size, strategies)
        self._set_gpu_memory_utilization()

        self._validate_arguments()

    @staticmethod
    def get_world_size_from_backend_parameters(
        model: Model,
    ) -> Tuple[Optional[int], Optional[List[str]]]:
        tp = find_int_parameter(
            model.backend_parameters, ["tensor-parallel-size", "tp"]
        )
        pp = find_int_parameter(
            model.backend_parameters, ["pipeline-parallel-size", "pp"]
        )
        dp = find_int_parameter(model.backend_parameters, ["data-parallel-size", "dp"])
        dpl = find_int_parameter(
            model.backend_parameters, ["--data-parallel-size-local", "dpl"]
        )

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
                if dpl:
                    # NB(thxCode): Indicate how many DP groups(each group owns `-dp` number devices) are there in one worker.
                    world_size *= dpl

            return world_size, strategies

        return None, None

    def _set_gpu_memory_utilization(self):
        self._gpu_memory_utilization = 0.9
        model = self._model
        if self._disable_gpu_memory_utilization():
            # gpu memory utilization is not used for non-LLM models
            self._gpu_memory_utilization = 0

        self._gpu_memory_utilization_parameter_name = "gpu-memory-utilization"
        gmu = find_parameter(
            model.backend_parameters, [self._gpu_memory_utilization_parameter_name]
        )
        if gmu:
            self._gpu_memory_utilization = float(gmu)

    def _disable_gpu_memory_utilization(self) -> bool:
        """
        Determine whether GPU memory utilization should be disabled. vLLM does not use --gpu-memory-utilization for non-LLM models
        like embedding and reranker, except for some specific models like Qwen3-Embedding and Qwen3-Reranker.

        Rules:
        1. For non-LLM models, GPU memory utilization is DISABLED (return True) unless they are in the exception list.
        2. Otherwise, GPU memory utilization is ENABLED (return False).
        """
        if not self._model.categories:
            return False

        architectures = self._model_params.architectures or []

        # Non-LLM models that vLLM still uses GPU memory utilization
        NON_LLM_GMU_EXCEPTIONS = {
            "Qwen3ForCausalLM",
            "Qwen3ForSequenceClassification",  # Qwen3-Embedding & Qwen3-Reranker
        }

        if CategoryEnum.LLM not in self._model.categories:
            # Disable for non-LLM models unless they are in the exception list
            return not any(arch in NON_LLM_GMU_EXCEPTIONS for arch in architectures)

        return False

    def _set_model_parameters(self):
        super()._set_model_parameters()

        # Get the architectures from hf-overrides. This helps make resource allocation
        # decisions for specific models like Qwen3-Embedding and Qwen3-Reranker.
        hf_overrides = find_parameter(self._model.backend_parameters, ["hf-overrides"])
        if hf_overrides:
            overrides_dict = json.loads(hf_overrides)
            if isinstance(overrides_dict, dict) and "architectures" in overrides_dict:
                self._model_params.architectures = overrides_dict["architectures"]

        self._num_attention_heads = self._model_params.num_attention_heads

    def _cal_effective_vram(self) -> float:
        if self._largest_multi_gpu_total == 0:
            return 0.0
        return (
            byte_to_gib(self._largest_multi_gpu_vram)
            * self._gpu_memory_utilization
            * self._largest_multi_gpu_utilization_satisfied_count
            / self._largest_multi_gpu_total
        )

    def _set_messages(self):
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

    async def _get_worker_allocatable_resource(self, worker: Worker):
        if worker.id in self._workers_allocatable_resource:
            return self._workers_allocatable_resource[worker.id]

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        self._workers_allocatable_resource[worker.id] = allocatable
        return allocatable

    def _get_worker_vram(self, worker: Worker) -> Dict[int, int]:
        if worker.name in self._worker_name_to_vram:
            return self._worker_name_to_vram[worker.name]

        if worker.status is None or not worker.status.gpu_devices:
            return {}

        vram_total_by_index = {}
        for gpu in worker.status.gpu_devices:
            total = gpu.memory.total if gpu.memory else 0
            vram_total_by_index[gpu.index] = total

        self._worker_name_to_vram[worker.name] = vram_total_by_index
        return vram_total_by_index

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement.
        """

        self._vram_claim = await estimate_model_vram(
            self._model, self._config.huggingface_token
        )
        self._ram_claim = get_model_ram_claim(self._model)
        logger.info(
            f"Calculated resource claim for model {self._model.readable_source}, "
            f"VRAM claim: {self._vram_claim}, RAM claim: {self._ram_claim}"
        )

        default_msg_list = ListMessageBuilder(
            f"The model requires approximately {byte_to_gib(self._vram_claim)} GiB of VRAM"
            f"{f' and {byte_to_gib(self._ram_claim)} GiB of RAM' if self._ram_claim > 0 else ''}."
        )
        if self._gpu_memory_utilization != 0:
            default_msg_list.append(
                f"With --{self._gpu_memory_utilization_parameter_name}={self._gpu_memory_utilization}, "
                f"all GPUs combined need to provide at least {byte_to_gib(int(self._vram_claim / self._gpu_memory_utilization))} GiB of total VRAM "
                f"and each GPU needs {int(self._gpu_memory_utilization * 100)}% of allocatable VRAM."
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

        for candidate_func in candidate_functions:
            if self.should_skip_candidate_func(candidate_func):
                continue

            logger.debug(
                f"model {self._model.readable_source}, filter candidates with resource fit selector: {candidate_func.__name__}"
            )

            candidates = await candidate_func(workers)

            if len(candidates) >= 1 and candidates[0].overcommit:
                # Manually selected candidate with overcommit. Also add the message.
                # It's useful for compatibility check.
                self._set_messages()

            if candidates:
                return candidates

        self._set_messages()
        return []

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

        return False

    async def find_manual_gpu_selection_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        request = RequestEstimateUsage(
            ram=self._ram_claim,
            vram=self._vram_claim,
        )

        return await self._find_manual_gpu_selection_candidates(
            workers,
            self._gpu_memory_utilization,
            request,
            self._event_collector,
        )

    async def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with workers.
        """
        if self._gpu_count is not None and self._gpu_count > 1:
            # Skip multi-GPU selection
            return []

        if (
            self._selected_gpu_worker_count > 1
            and self._gpu_count
            and self._gpu_count > 1
        ):
            # Skip multi-worker selection
            return []

        selected_gpu_worker = None
        filtered_workers = []

        if self._selected_gpu_workers:
            # Handle manual scheduling
            for worker in workers:
                if worker.name in self._selected_gpu_workers:
                    filtered_worker = worker
                    filtered_devices = []
                    indexes = self._selected_gpu_indexes_by_worker.get(worker.name, [])
                    for gpu in worker.status.gpu_devices:
                        if gpu.index in indexes:
                            filtered_devices.append(gpu)
                    filtered_worker.status.gpu_devices = filtered_devices
                    filtered_workers.append(filtered_worker)
        else:
            filtered_workers = workers

        candidates = []
        for worker in filtered_workers:
            if not worker.status.gpu_devices:
                continue

            if selected_gpu_worker and worker.name != selected_gpu_worker:
                continue

            result = (
                await self._find_single_worker_single_gpu_full_offloading_candidates(
                    worker,
                )
            )
            if result:
                candidates.extend(result)

        if (
            self._model.replicas > 1
            and self._selected_gpu_workers
            and len(candidates) > 0
        ):
            candidates = [
                candidate for candidate in candidates if not candidate.overcommit
            ]
            if len(candidates) == 0:
                self._event_collector.add(
                    EventLevelEnum.INFO,
                    EVENT_ACTION_MANUAL_MULTI,
                    str(
                        ListMessageBuilder(
                            "Manual scheduling for multi-replica model instances does not allow overcommit or heterogeneous deployment topologies."
                        )
                    ),
                )

        return candidates

    async def _find_single_worker_single_gpu_full_offloading_candidates(
        self,
        worker: Worker,
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with worker.
        requires: worker.status.gpu_devices is not None
        """

        candidates = []

        allocatable = await self._get_worker_allocatable_resource(worker)

        if ram_not_enough(self._ram_claim, allocatable):
            return []

        if not worker.status.gpu_devices:
            return []

        for _, gpu in enumerate(worker.status.gpu_devices):

            gpu_index = gpu.index
            allocatable_vram = allocatable.vram.get(gpu_index, 0)
            allocatable_gpu_memory_utilization = allocatable_vram / gpu.memory.total

            if allocatable_vram > self._largest_single_gpu_vram:
                self._largest_single_gpu_vram = allocatable_vram
                self._largest_single_gpu_vram_utilization = (
                    allocatable_gpu_memory_utilization
                )

            if gpu.memory is None or gpu.memory.total == 0:
                continue

            overcommit = False
            exceeds_vram = (
                self._vram_claim > gpu.memory.total * self._gpu_memory_utilization
                if self._gpu_memory_utilization > 0  # LLMs
                else self._vram_claim > allocatable_vram  # non LLMs
            )
            exceeds_memory_utilization = (
                self._gpu_memory_utilization > 0
                and allocatable_gpu_memory_utilization < self._gpu_memory_utilization
            )
            if exceeds_vram or exceeds_memory_utilization:
                if self._selected_gpu_worker_count != 0:
                    overcommit = True
                else:
                    continue

            vram_claim_bytes = (
                int(gpu.memory.total * self._gpu_memory_utilization)
                if self._gpu_memory_utilization > 0  # LLMs
                else int(self._vram_claim)  # non LLMs
            )

            vram_claim = {gpu_index: vram_claim_bytes}
            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=[gpu_index],
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=get_computed_ram_claim(self._model, vram_claim),
                    ),
                    overcommit=overcommit,
                )
            )

        if not candidates or (len(candidates) == 1 and candidates[0].overcommit):
            event_msg = f"The current available GPU only has {byte_to_gib(self._largest_single_gpu_vram)} GiB allocatable VRAM."
            if self._gpu_memory_utilization != 0:
                event_msg = (
                    event_msg.rstrip(".")
                    + f" ({(self._largest_single_gpu_vram_utilization * 100):.2f}%)."
                )
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_AUTO_SINGLE_GPU,
                str(ListMessageBuilder(event_msg)),
            )

        return candidates

    async def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        if self._gpu_count == 1:
            return []

        should_skip_worker = 0
        if (
            self._selected_gpu_worker_count > 1
            and self._model.gpu_selector
            and self._model.gpu_selector.gpus_per_replica
        ):
            for (
                worker_name,
                selected_gpus,
            ) in self._selected_gpu_indexes_by_worker.items():
                if len(selected_gpus) < self._model.gpu_selector.gpus_per_replica:
                    should_skip_worker += 1
            if should_skip_worker == self._selected_gpu_worker_count:
                return []

        candidates = []
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

        if not candidates:
            return []

        min_gpu_count = min(len(candidate.gpu_indexes) for candidate in candidates)
        final_candidates = [
            candidate
            for candidate in candidates
            if len(candidate.gpu_indexes) == min_gpu_count
        ]
        return final_candidates

    async def _find_single_worker_multi_gpu_full_offloading_candidates(  # noqa: C901
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi gpu full offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """

        total_gpu = len(worker.status.gpu_devices)
        if total_gpu < 2:
            return None

        allocatable = await self._get_worker_allocatable_resource(worker)

        if ram_not_enough(self._ram_claim, allocatable):
            return []

        gpu_list = []
        total_allocatable_vram = 0
        satisfied_gpu_count = 0

        for gpu in worker.status.gpu_devices:
            if gpu.memory is None or gpu.memory.total is None:
                continue

            allocatable_vram = allocatable.vram.get(gpu.index, 0)
            total_allocatable_vram += allocatable_vram

            if allocatable_vram / gpu.memory.total > self._gpu_memory_utilization:
                satisfied_gpu_count += 1
                gpu_list.append(gpu)

        if total_allocatable_vram > self._largest_multi_gpu_total:
            self._largest_multi_gpu_vram = total_allocatable_vram
            self._largest_multi_gpu_utilization_satisfied_count = satisfied_gpu_count
            self._largest_multi_gpu_total = len(worker.status.gpu_devices)

        # Sort by vram in descending order
        sorted_gpu_devices: GPUDevicesInfo = sorted(
            gpu_list,
            key=lambda gpu: allocatable.vram.get(gpu.index, 0),
            reverse=True,
        )

        vram_sum = 0
        gpu_sum = 0
        gpu_indexes = []
        vram_claim: Dict[int, int] = {}
        found_candidate = False
        for _, gpu in enumerate(sorted_gpu_devices):
            gpu_indexes.append(gpu.index)
            vram_claim[gpu.index] = (
                int(gpu.memory.total * self._gpu_memory_utilization)
                if self._gpu_memory_utilization > 0  # LLMs
                else allocatable.vram.get(gpu.index, 0)  # non LLMs
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
                    gpu_indexes=gpu_indexes,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=get_computed_ram_claim(self._model, vram_claim),
                    ),
                )
            ]
        event_msg_list = []
        if msg := self._check_tp_size_divisibility(
            self._largest_multi_gpu_utilization_satisfied_count
        ):
            event_msg_list.append(msg)
        event_msg = f"The largest available worker has {byte_to_gib(self._largest_multi_gpu_vram)} GiB allocatable VRAM."
        if self._gpu_memory_utilization != 0:
            event_msg = (
                event_msg.rstrip(".")
                + f", {self._largest_multi_gpu_utilization_satisfied_count}/{self._largest_multi_gpu_total} of GPUs meet the VRAM utilization ratio, providing {self._cal_effective_vram():.2f} GiB of allocatable VRAM."
            )
        event_msg_list.append(event_msg)

        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            str(ListMessageBuilder(event_msg_list)),
        )

        return []

    async def find_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        if self._model.distributed_inference_across_workers:
            return await self.auto_select_multi_worker_multi_gpu_candidates(workers)

        return []

    async def auto_select_multi_worker_multi_gpu_candidates(  # noqa: C901
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Auto select multi worker multi gpu candidates.
        Currently, a candidate should match the following conditions:
        1. Workers in the candidate have the same number of GPUs.
        2. All GPUs in the worker satisfy the gpu_memory_utilization requirement.
        3. TP size can be divided by the number of attention heads.
        4. The total VRAM claim is greater than the estimated VRAM claim.
        5. If gpu_count is set via parallelism, the total GPU count should be equal to gpu_count.
        """

        if not workers or len(workers) < 2:
            return []

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

            if not self._is_tp_size_divisible(gpu_count):
                continue

            selected_workers: List[Worker] = []
            gpu_sum = 0
            vram_sum = 0
            for worker in worker_group:
                allocatable = await self._get_worker_allocatable_resource(worker)

                if ram_not_enough(self._ram_claim, allocatable):
                    # The RAM resource(for extended KV cache) is required per worker.
                    # Skip the worker if it does not satisfy the RAM requirement.
                    continue

                if any(
                    gpu.memory is None
                    or gpu.memory.total is None
                    or (
                        allocatable.vram.get(gpu.index, 0) / gpu.memory.total
                        < self._gpu_memory_utilization
                    )
                    for gpu in worker.status.gpu_devices
                ):
                    # Skip the worker if any GPU does not satisfy the gpu_memory_utilization requirement.
                    continue
                selected_workers.append(worker)
                gpu_sum += gpu_count
                vram_sum += sum(
                    int(gpu.memory.total * (self._gpu_memory_utilization or 1))
                    for gpu in worker.status.gpu_devices
                )

                if self._gpu_count:
                    # Parallelism is set. Proceed until we match the exact GPU count.
                    if gpu_sum < self._gpu_count:
                        continue
                    elif gpu_sum > self._gpu_count:
                        break

                if vram_sum >= self._vram_claim:
                    return [
                        _create_candidate(
                            self._model,
                            selected_workers,
                            self._gpu_memory_utilization,
                        )
                    ]
            if vram_sum > largest_vram:
                workers_combination = selected_workers
                largest_vram = vram_sum
                worker_count = len(worker_group)
                device_count_per_worker = gpu_count

        # Nothing can be return, construct scheduling message
        event_message = ListMessageBuilder([])
        if self._gpu_memory_utilization == 0:
            event_message.append(
                f"The largest available worker has {byte_to_gib(largest_vram)} GiB of VRAM."
            )
        elif workers_combination:
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

    def _validate_arguments(self):
        tp = find_int_parameter(
            self._model.backend_parameters, ["tensor-parallel-size", "tp"]
        )
        if msg := self._check_tp_size_divisibility(tp):
            raise ValueError(
                msg + " Consider adjusting your tensor-parallel-size value."
            )


def _create_candidate(
    model: Model,
    selected_workers: List[Worker],
    gpu_memory_utilization: float = 0.9,
) -> ModelInstanceScheduleCandidate:
    """
    Create a candidate with all GPUs from the selected workers.
    """
    main_worker = selected_workers[0]
    vram_claim_main = {
        gpu.index: int(gpu.memory.total * gpu_memory_utilization)
        for gpu in main_worker.status.gpu_devices
    }
    candidate = ModelInstanceScheduleCandidate(
        worker=main_worker,
        gpu_indexes=[gpu.index for gpu in main_worker.status.gpu_devices],
        computed_resource_claim=ComputedResourceClaim(
            vram=vram_claim_main,
            ram=get_computed_ram_claim(model, vram_claim_main),
        ),
    )
    candidate.subordinate_workers = []
    for worker in selected_workers[1:]:
        vram_claim_subworker = {
            gpu.index: int(gpu.memory.total * gpu_memory_utilization)
            for gpu in worker.status.gpu_devices
        }
        candidate.subordinate_workers.append(
            ModelInstanceSubordinateWorker(
                worker_id=worker.id,
                worker_name=worker.name,
                worker_ip=worker.ip,
                worker_ifname=worker.ifname,
                total_gpus=len(worker.status.gpu_devices),
                gpu_indexes=[gpu.index for gpu in worker.status.gpu_devices],
                computed_resource_claim=ComputedResourceClaim(
                    vram=vram_claim_subworker,
                    ram=get_computed_ram_claim(model, vram_claim_subworker),
                ),
            )
        )

    return candidate
