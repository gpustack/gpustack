import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from gpustack_runtime.deployer.__utils__ import compare_versions

from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScheduleCandidate,
)
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    ModelAttentionTypeEnum,
    RequestEstimateUsage,
    ScheduleCandidatesSelector,
)
from gpustack.policies.event_recorder.recorder import EventCollector
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
    ListMessageBuilder,
    get_local_model_weight_size,
)
from gpustack.scheduler.model_registry import is_multimodal_model
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
    SourceEnum,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.workers import GPUDeviceStatus, Worker
from gpustack.config import Config
from gpustack.utils.convert import safe_int
from gpustack.utils.hub import (
    get_model_weight_size,
)
from gpustack.utils.unit import byte_to_gib
from gpustack.worker.backends.ascend_mindie import AscendMindIEParameters

logger = logging.getLogger(__name__)


class AscendMindIEResourceFitSelector(ScheduleCandidatesSelector):
    """
    Ascend MindIE Resource Fit Selector,
    which is designed to select deployment candidates.
    """

    def __init__(  # noqa: C901
        self,
        config: Config,
        model: Model,
        model_instances: List[ModelInstance],
    ):
        super().__init__(config, model, model_instances)

        # Diagnostic message to be set to the model instance.
        self._diagnostic_messages: List[str] = []
        # Serving parameters.
        self._serving_params = AscendMindIEParameters()
        # Temporary indexer for caching worker's allocatable, avoiding redundant database queries: {Worker ID: Allocatable}.
        self.__worker_alloc_idx: Dict[int, Allocatable] = {}
        # Temporary indexer for caching worker's devices that sort by VRAM size: {Worker ID: sorted([Device 0, Device 1, ...])}.
        self.__worker_sorted_devices_idx: Dict[int, List[GPUDeviceStatus]] = {}

        # Store and format the abnormal message during scheduling, finally it will be extended to self._diagnostic_messages.
        self._scheduling_messages: ListMessageBuilder = ListMessageBuilder([])

    async def _init_model_parameters(self, workers: List[Worker] = None):
        await super()._init_model_parameters(workers)

        # Parse model params.
        model = self._model
        max_seq_len = self._model_params.derived_max_seq_len
        if max_seq_len > 8192:
            max_seq_len = 8192
        self._serving_params.max_seq_len = max_seq_len
        try:
            self._serving_params.from_args_and_envs(
                model.backend_parameters or [], model.env or {}
            )
        except Exception as e:
            raise ValueError(
                f"Failed to parse model {model.name} serve parameters: {e}"
            )

        world_size, strategies = (
            AscendMindIEResourceFitSelector.get_world_size_from_backend_parameters(
                model
            )
        )
        self._set_gpu_count(world_size, strategies)

    @staticmethod
    def get_world_size_from_backend_parameters(
        model: Model,
    ) -> Tuple[Optional[int], Optional[List[str]]]:
        if model.backend_parameters is None:
            return None, None

        serving_params = AscendMindIEParameters()
        serving_params.from_args_and_envs(model.backend_parameters, model.env or {})

        pp, tp, dp, cp, sp, moe_tp, moe_ep, ws = (
            serving_params.pipeline_parallel_size,
            serving_params.tensor_parallel_size,
            serving_params.data_parallel_size,
            serving_params.context_parallel_size,
            serving_params.sequence_parallel_size,
            serving_params.moe_tensor_parallel_size,
            serving_params.moe_expert_parallel_size,
            serving_params.world_size,
        )

        if pp > 1 or tp > 0 or dp > 0 or cp > 0 or sp > 0 or moe_tp > 0 or moe_ep > 0:
            world_size = ws
            strategies = []
            if pp > 1:
                strategies.append("pp")
            if tp > 0:
                strategies.append("tp")
            if dp > 0:
                strategies.append("dp")
            if cp > 0:
                strategies.append("cp")
            if sp > 0:
                strategies.append("sp")
            if moe_tp > 0:
                strategies.append("moe_tp")
            if moe_ep > 0:
                strategies.append("moe_ep")

            return world_size, strategies

        return None, None

    def get_messages(self) -> List[str]:
        return self._diagnostic_messages

    async def select_candidates(
        self,
        workers: List[Worker],
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Select available deployment candidates based on the resource requirements.

        The selection process involves:
        1. Estimating the resource requirements.
        2. Constructing candidates that can accommodate the estimated resource requirements.
        """

        # Initialize model parameters.
        await self._init_model_parameters(workers)

        # Estimate resource usage.
        estimated_usage = await self._estimate_usage(workers)

        # Filter workers based on estimated usage.
        candidates = await self._construct_candidates(estimated_usage, workers)
        if (not candidates) or (len(candidates) == 1 and candidates[0].overcommit):
            logger.error(
                f"No suitable candidates found for the model {self._model.name}."
            )
            self._diagnostic_messages.append(str(self._scheduling_messages))
        else:
            logger.info(
                f"Found {len(candidates)} candidates for the model {self._model.name}."
            )
        return candidates

    async def _estimate_usage(  # noqa: C901
        self, workers: Optional[List[Worker]] = None
    ) -> RequestEstimateUsage:
        """
        Estimate the resource usage of the model instance.

        Formula:

            RAM  = FOOTPRINT + KV_CACHE_SWAPPABLE
            VRAM = FOOTPRINT + WEIGHT + KV_CACHE + COMPUTATION
        """

        # A potential workaround if the empirical resource estimation is biased greatly.
        if self._model.env:
            reqeust_ram = safe_int(self._model.env.get("GPUSTACK_MODEL_RAM_CLAIM", 0))
            request_vram = safe_int(self._model.env.get("GPUSTACK_MODEL_VRAM_CLAIM", 0))
            if request_vram > 0:
                return RequestEstimateUsage(reqeust_ram, request_vram)

        is_multimodal = (
            is_multimodal_model(self._model_params.architectures)
            and compare_versions(self._model.backend_version or "0.0.0", "2.2.rc1") >= 0
        )

        """
        RAM
        """

        # Hardcode the RAM footprint for now.
        ram_footprint = 512 * 1024**2

        # Get KV cache swappable size.
        ram_kv_cache_swappable = self._serving_params.cpu_mem_size * 1024**3

        """
        VRAM
        """

        # Treat the VRAM footprint as 3 GiB by default.
        vram_footprint = 3 * 1024**3

        # Override vram_footprint if RESERVED_MEMORY_GB is set.
        if "RESERVED_MEMORY_GB" in (self._model.env or {}):
            reserved_memory_gb = safe_int(self._model.env.get("RESERVED_MEMORY_GB"))
            if reserved_memory_gb > 0:
                vram_footprint = reserved_memory_gb * 1024**3

        # Get model weight size.
        vram_weight = 0
        try:
            source = self._model.source
            if source in [SourceEnum.HUGGING_FACE, SourceEnum.MODEL_SCOPE]:
                vram_weight = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_model_weight_size,
                        self._model,
                        self._config.huggingface_token,
                    ),
                    timeout=15,
                )
            elif source == SourceEnum.LOCAL_PATH:
                local_path = self._model.local_path
                vram_weight = await get_local_model_weight_size(local_path, workers)
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when getting weight size for model {self._model.name}"
            )
        except Exception as e:
            logger.warning(f"Cannot get weight size for model {self._model.name}: {e}")

        n_tokens = self._serving_params.max_seq_len
        n_layers = self._model_params.num_hidden_layers or 0
        n_heads = 1
        d_head = self._model_params.head_dim or 0
        t_size = 4 if self._model_params.torch_dtype in ["float32", "float"] else 2

        # Get KV cache size,
        # according to the model's attention type,
        # see the "Comparison of Key-Value Cache" chapter of
        # https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V2/main/deepseek-v2-tech-report.pdf.
        vram_kv_cache = 0
        if attention_type := self._model_params.get_attention_type():
            # Get cache type size,
            # see https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0292.html.
            ct_size = t_size
            if self._model_params.quantization_config:
                kv_quant_type = self._model_params.quantization_config.get(
                    "kv_quant_type", ""
                )
                if (
                    kv_quant_type
                    and isinstance(kv_quant_type, str)
                    and kv_quant_type.lower() == "c8"
                ):
                    ct_size = 1

            if attention_type == ModelAttentionTypeEnum.MHA:
                n_heads = 2 * self._model_params.num_attention_heads
            elif attention_type == ModelAttentionTypeEnum.GQA:
                n_heads = 2 * self._model_params.num_key_value_heads
            elif attention_type == ModelAttentionTypeEnum.MQA:
                n_heads = 2
            elif attention_type == ModelAttentionTypeEnum.MLA:
                d_head = (
                    self._model_params.qk_nope_head_dim
                    + self._model_params.qk_nope_head_dim
                )

            vram_kv_cache = n_heads * d_head * n_layers * n_tokens * ct_size

        # Correct multimodal vram_kv_cache.
        if is_multimodal:
            vram_kv_cache = self._serving_params.max_prefill_tokens * (
                n_layers * d_head * 4
            )

        # Get cache type size,
        # see https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0288.html.
        at_size = t_size
        if (
            self._model_params.quantize
            and self._model_params.quantize.lower().startswith("w8a8")
        ):
            at_size = 1

        # Get computation size,
        # do inverting calculation according to https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindieservice/servicedev/mindie_service0105.html.
        #
        # We assume the given max_batch_size is the maximum batch size,
        # so the original formula:
        #   max_batch_size   = FLOOR(total_block_num / minium_block_num)
        #   minium_block_num = CEIL(input_tokens / cache_block_size)
        #   total_block_num  = FLOOR(computation_size / (n_layers * cache_block_size * n_heads * d_head * t_size))
        #
        # But we don't know the input_tokens, we can make an assumption that the input_tokens is equal to the max_seq_len,
        # so the formula can be simplified to:
        # computation_size = max_batch_size * (cache_block_size) * (n_layers * n_heads * d_head * t_size)

        vram_computation = (
            self._serving_params.max_batch_size
            * self._serving_params.cache_block_size
            * (n_layers * n_heads * d_head * at_size)
        )

        # Correct multimodal vram_computation.
        if is_multimodal:
            vram_computation = (
                self._serving_params.max_batch_size
                * self._serving_params.max_iter_times
                * (n_layers * d_head * 4)
            )

        ram = ram_footprint + ram_kv_cache_swappable
        vram = vram_footprint + vram_weight + vram_kv_cache + vram_computation

        logger.info(
            f"Estimated model resource usage: "
            f"RAM: footprint = {byte_to_gib(ram_footprint)} GiB, kv_cache_swappable = {byte_to_gib(ram_kv_cache_swappable)} GiB, total = {byte_to_gib(ram)} GiB, "
            f"VRAM: footprint = {byte_to_gib(vram_footprint)} GiB, kv_cache = {byte_to_gib(vram_kv_cache)} GiB, weight = {byte_to_gib(vram_weight)} GiB, computation = {byte_to_gib(vram_computation)} GiB, total = {byte_to_gib(vram)} GiB"
        )

        return RequestEstimateUsage(ram, vram)

    async def _validate_parallelized(self) -> bool:  # noqa: C901
        # Validate whether model can be parallelized.
        if num_attention_heads := self._model_params.num_attention_heads:
            if self._serving_params.pipeline_parallel_size > 1:
                if num_attention_heads % self._serving_params.tensor_parallel_size != 0:
                    self._diagnostic_messages.append(
                        f"Model's attention heads ({num_attention_heads}) must be divisible by "
                        f"the --tensor-parallel-size ({self._serving_params.tensor_parallel_size})."
                    )
                    return False
            else:
                if num_attention_heads % self._serving_params.world_size != 0:
                    if self._serving_params.data_parallel_size > 1:
                        self._diagnostic_messages.append(
                            f"Model's attention heads ({num_attention_heads}) must be divisible by "
                            f"the world size ({self._serving_params.world_size}), "
                            f"multiplied by --data-parallel-size ({self._serving_params.data_parallel_size}) "
                            f"and --tensor-parallel-size ({self._serving_params.tensor_parallel_size})."
                        )
                    elif self._serving_params.context_parallel_size > 1:
                        self._diagnostic_messages.append(
                            f"Model's attention heads ({num_attention_heads}) must be divisible by "
                            f"the world size ({self._serving_params.world_size}), "
                            f"multiplied by --context-parallel-size ({self._serving_params.context_parallel_size}) "
                            f"and --tensor-parallel-size ({self._serving_params.tensor_parallel_size})."
                        )
                    elif self._serving_params.moe_expert_parallel_size > 1:
                        self._diagnostic_messages.append(
                            f"Model's attention heads ({num_attention_heads}) must be divisible by "
                            f"the world size ({self._serving_params.world_size}), "
                            f"multiplied by --moe-expert-parallel-size ({self._serving_params.moe_expert_parallel_size}) "
                            f"and --moe-tensor-parallel-size ({self._serving_params.moe_tensor_parallel_size})."
                        )
                    else:
                        self._diagnostic_messages.append(
                            f"Model's attention heads ({num_attention_heads}) must be divisible by "
                            f"the --tensor-parallel-size ({self._serving_params.tensor_parallel_size})."
                        )
                    return False
                if self._serving_params.moe_expert_parallel_size > 1:
                    if moe_num_experts := self._model_params.moe_num_experts:
                        if (
                            moe_num_experts
                            % self._serving_params.moe_expert_parallel_size
                            != 0
                        ):
                            self._diagnostic_messages.append(
                                f"Model's MoE experts ({moe_num_experts}) must be divisible by "
                                f"the --moe-expert-parallel-size ({self._serving_params.moe_expert_parallel_size})."
                            )
                            return False
                if self._serving_params.moe_tensor_parallel_size > 1:
                    if moe_inter_size := self._model_params.moe_intermediate_size:
                        if (
                            moe_inter_size
                            % self._serving_params.moe_tensor_parallel_size
                            != 0
                        ):
                            self._diagnostic_messages.append(
                                f"Model's MoE intermediate size ({moe_inter_size}) must be divisible by "
                                f"the --moe-tensor-parallel-size ({self._serving_params.moe_tensor_parallel_size})."
                            )
                            return False
        if vocab_size := self._model_params.vocab_size:
            if vocab_size % self._serving_params.tensor_parallel_size != 0:
                self._diagnostic_messages.append(
                    f"Model's vocabulary size ({vocab_size}) must be divisible by "
                    f"the --tensor-parallel-size ({self._serving_params.tensor_parallel_size})."
                )
                return False
        return True

    async def _get_available_worker_devices_idx(  # noqa: C901
        self, workers, ram_request
    ) -> Dict[Worker, Dict[int, GPUDeviceStatus]]:
        available_worker_devices_idx: Dict[Worker, Dict[int, GPUDeviceStatus]] = {}
        for worker in workers:
            # Skip if the worker does not have devices.
            if not worker.status.gpu_devices:
                continue

            worker_alloc = await self.__get_worker_alloc(worker)

            # Skip if the worker does not have enough RAM.
            ram_allocate = worker_alloc.ram
            if ram_request > ram_allocate:
                continue

            # Skip if VRAM request exceeds the main device allocatable.
            device = self.__worker_sorted_devices_idx[worker.id][0]
            device_vram_request = int(
                device.memory.total * self._serving_params.npu_memory_fraction
            )
            device_vram_allocate = worker_alloc.vram.get(device.index, 0)
            if device_vram_request > device_vram_allocate:
                continue

            # Get selected devices of the worker: {Device Index: Device}.
            selected_devices_idx: Dict[int, GPUDeviceStatus] = {
                device.index: device
                for device in self.__worker_sorted_devices_idx[worker.id]
                if device.type == "cann"
            }

            # Index the worker and its devices.
            available_worker_devices_idx[worker] = selected_devices_idx

        # Validate if there are available workers,
        # if not, return.
        if not available_worker_devices_idx:
            self._diagnostic_messages.append("No available workers found.")
            return available_worker_devices_idx

        # Sort available worker devices by allocatable resources,
        # make sure the worker with the largest VRAM is in first position.
        if len(available_worker_devices_idx) > 1:
            available_worker_devices_idx = dict(
                sorted(
                    available_worker_devices_idx.items(),
                    key=lambda item: self.__worker_alloc_idx[item[0].id].vram.get(
                        self.__worker_sorted_devices_idx[item[0].id][0].index, 0
                    ),
                    reverse=True,
                )
            )
        return available_worker_devices_idx

    def _manual_select_candidates(
        self, workers: List[Worker], request_usage: RequestEstimateUsage
    ) -> List[ModelInstanceScheduleCandidate]:
        event_collector = EventCollector(self._model, logger)
        candidates = self._find_manual_gpu_selection_candidates(
            workers,
            {"*": self._serving_params.npu_memory_fraction},
            request_usage,
            event_collector,
        )

        for event in event_collector.events:
            self._scheduling_messages.append(event.message.removeprefix("- "))
        return candidates

    async def _select_single_worker(  # noqa: C901
        self,
        available_worker_devices_idx,
        request_usage,
    ):
        candidates: List[ModelInstanceScheduleCandidate] = []
        largest_vram = 0
        total_devices_count = 0
        satisfied_devices_count = 0
        # Iterate over the workers.
        for worker, devices in available_worker_devices_idx.items():
            # Skip if the worker does not have enough devices.
            if 0 < self._serving_params.world_size > len(devices):
                continue

            if len(devices) == 0:
                continue

            # Construct candidate by the worker.
            candidate = ModelInstanceScheduleCandidate(
                worker=worker,
                gpu_indexes=[],
                gpu_addresses=[],
                computed_resource_claim=ComputedResourceClaim(
                    ram=request_usage.ram,
                    vram={},
                ),
            )

            # Worker allocation.
            worker_alloc = await self.__get_worker_alloc(worker)
            # Record the remaining VRAM request in the worker level,
            # which is used to determine if enough devices are selected.
            worker_vram_request_remain = request_usage.vram
            # Record the remaining count needed to be selected,
            # which is used to determine if enough devices are selected.
            world_size_remain = self._serving_params.world_size

            # Iterate over the worker's devices from the largest VRAM.
            for device in self.__worker_sorted_devices_idx[worker.id]:
                # Break if selected devices are satisfied in semi-automatic mode.
                if world_size_remain == 0:
                    break

                # Break if selected devices or requested VRAM are satisfied in automatic mode.
                if world_size_remain < 0 and worker_vram_request_remain <= 0:
                    if world_size_remain < -1:
                        if self._is_tp_size_divisible(abs(world_size_remain + 1)):
                            break
                    else:
                        break

                # Calculate device VRAM request.
                device_vram_request = int(
                    devices[device.index].memory.total
                    * self._serving_params.npu_memory_fraction
                )
                device_vram_allocate = worker_alloc.vram.get(device.index, 0)
                # Skip if the device VRAM request exceeds the allocatable.
                if device_vram_request > device_vram_allocate:
                    continue

                # Append the device to the candidate.
                candidate.gpu_indexes.append(device.index)
                candidate.gpu_addresses.append(
                    device.network.inet
                    if device.network
                    and device.network.status == 'up'
                    and device.network.inet
                    else "-.-.-.-"
                )
                candidate.computed_resource_claim.vram[device.index] = (
                    device_vram_request
                )

                # Stats
                worker_vram_request_remain -= device_vram_request
                world_size_remain -= 1

            if request_usage.vram - worker_vram_request_remain > largest_vram:
                largest_vram = request_usage.vram - worker_vram_request_remain
                satisfied_devices_count = (
                    self._serving_params.world_size - world_size_remain
                )
                total_devices_count = len(self.__worker_sorted_devices_idx[worker.id])

            # Skip if selected devices are not satisfied in semi-automatic mode.
            if world_size_remain > 0:
                continue

            # Skip if attention heads cannot be divided by the selected devices count in automatic mode.
            elif world_size_remain < -1:
                if not self._is_tp_size_divisible(abs(world_size_remain + 1)):
                    continue

            # Skip if the worker does not have enough VRAM.
            if worker_vram_request_remain > 0:
                continue

            # Append the candidate if satisfied the requested resources.
            candidates.append(candidate)
            logger.debug(f"Found intermediate candidate: {candidate.to_log_string()}")

        if (
            len(candidates) == 0
            and not self._model.distributed_inference_across_workers
        ):
            effective_vram = (
                byte_to_gib(largest_vram)
                * self._serving_params.npu_memory_fraction
                * satisfied_devices_count
                / total_devices_count
                if total_devices_count
                else 0.0
            )
            event_msg = (
                f"The largest available worker has {byte_to_gib(largest_vram)} GiB allocatable VRAM, "
                f"{satisfied_devices_count}/{total_devices_count} of GPUs meet the VRAM utilization ratio, providing {effective_vram:.2f} GiB of allocatable VRAM."
            )
            self._scheduling_messages.append(event_msg)

        return candidates

    async def _select_multi_workers(  # noqa: C901
        self, available_worker_devices_idx, request_usage
    ):
        if not self._model.distributed_inference_across_workers:
            return []
        candidates: List[ModelInstanceScheduleCandidate] = []
        # Store the optimal combination info to show
        largest_vram = 0
        workers_combination: List[Worker] = []
        worker_count = 0
        device_count_per_worker = 0
        # Group workers by device count: {Devices Count: [Worker, ...]}.
        device_count_worker_group_idx: Dict[int, List[Worker]] = {}
        for worker in available_worker_devices_idx.keys():
            devices_count = len(self.__worker_sorted_devices_idx[worker.id])
            if devices_count not in device_count_worker_group_idx:
                device_count_worker_group_idx[devices_count] = []
            device_count_worker_group_idx[devices_count].append(worker)

        # Sort workers by device count in descending order,
        # and refill lower device count workers with higher device count workers.
        # For example, the original indexer looks like:
        # {
        #   16: [Worker A],
        #    8: [Worker B],
        #    2: [Worker E],
        #    4: [Worker C, Worker D],
        # },
        # and moodify to:
        # {
        #   16: [Worker A],
        #    8: [Worker B, Worker A],
        #    4: [Worker C, Worker D, Worker B, Worker A],
        #    2: [Worker E, Worker C, Worker D, Worker B, Worker A],
        # }
        if len(device_count_worker_group_idx) > 1:
            device_count_worker_group_idx = dict(
                sorted(
                    device_count_worker_group_idx.items(),
                    key=lambda item: item[0],
                    reverse=True,
                )
            )
            keys = list(device_count_worker_group_idx.keys())
            for i in range(0, len(keys) - 1):
                key, next_key = keys[i], keys[i + 1]
                device_count_worker_group_idx[next_key].extend(
                    device_count_worker_group_idx[key]
                )

        # Iterate over the workers from the largest device count.
        for device_count, worker_group in device_count_worker_group_idx.items():
            # Skip if the worker group is smaller.
            if len(worker_group) < 2:
                continue

            # Skip if the worker group does not have enough devices in semi-automatic mode.
            if (
                0 < self._serving_params.world_size > device_count * len(worker_group)
                or 0 < self._serving_params.local_world_size > device_count
            ):
                continue

            # Get suggested local world size group to guide the selection.
            local_world_size_group = []
            # If in semi-automatic mode, use the given local world size.
            if self._serving_params.local_world_size > 0:
                local_world_size_group.append(self._serving_params.local_world_size)
            # Else, find a suitable local world size based on the device count.
            elif self._serving_params.local_world_size < 0:
                local_world_size = (
                    device_count
                    if self._serving_params.world_size < 0
                    else min(device_count, self._serving_params.world_size)
                )
                while local_world_size > 1:
                    # Skip if the local world size is not power of 2.
                    if local_world_size & (local_world_size - 1) != 0:
                        local_world_size -= 1
                        continue
                    if not self._is_tp_size_divisible(local_world_size):
                        local_world_size -= 1
                        continue
                    # Found a valid local world size.
                    local_world_size_group.append(local_world_size)
                    local_world_size -= 1
                local_world_size_group.append(1)
                local_world_size_group.reverse()

            # Iterate over the local world sizes to find candidates.
            for local_world_size in local_world_size_group:
                candidate: Optional[ModelInstanceScheduleCandidate] = None
                subworker: Optional[ModelInstanceSubordinateWorker] = None
                subworker_index = -1

                # Record the remaining VRAM request in the worker group level,
                # which is used to determine if enough devices are selected.
                worker_group_vram_request_remain = request_usage.vram
                # Record the remaining count needed to be selected,
                # which is used to determine if enough devices are selected.
                world_size_remain = self._serving_params.world_size

                selected_workers: List[Worker] = []

                for worker in worker_group:
                    # Break if selected devices are satisfied in semi-automatic mode.
                    if world_size_remain == 0:
                        break

                    # Construct candidate by the main worker.
                    if subworker_index < 0:
                        candidate = ModelInstanceScheduleCandidate(
                            worker=worker,
                            gpu_indexes=[],
                            gpu_addresses=[],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=request_usage.ram,
                                vram={},
                            ),
                            subordinate_workers=[],
                        )
                    # Increase subordinate workers.
                    else:
                        subworker = ModelInstanceSubordinateWorker(
                            worker_id=worker.id,
                            worker_name=worker.name,
                            worker_ip=worker.ip,
                            worker_ifname=worker.ifname,
                            total_gpus=len(worker.status.gpu_devices),
                            gpu_indexes=[],
                            gpu_addresses=[],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=request_usage.ram,
                                vram={},
                            ),
                        )

                    # Worker allocation.
                    worker_alloc = await self.__get_worker_alloc(worker)
                    # Record the remaining VRAM request in the devices level,
                    # which is used to determine if enough devices are selected.
                    worker_vram_request_remain = worker_group_vram_request_remain
                    # Record the remaining count needed to be selected,
                    # which is used to determine if enough devices are selected.
                    local_world_size_remain = local_world_size

                    # Iterate over the worker's devices from the largest VRAM.
                    for device in self.__worker_sorted_devices_idx[worker.id]:
                        # Break if selected devices are satisfied in semi-automatic mode.
                        if local_world_size_remain == 0:
                            break

                        if device.type != "cann":
                            continue

                        # Calculate device VRAM request.
                        device_vram_request = int(
                            available_worker_devices_idx[worker][
                                device.index
                            ].memory.total
                            * self._serving_params.npu_memory_fraction
                        )
                        device_vram_allocate = worker_alloc.vram.get(device.index, 0)
                        # Skip if the device VRAM request exceeds the allocatable.
                        if device_vram_request > device_vram_allocate:
                            continue

                        # Skip if the device doesn't have network address.
                        if (
                            not device.network
                            or device.network.status != 'up'
                            or not device.network.inet
                        ):
                            continue

                        # Increase main worker's devices.
                        if subworker_index < 0:
                            candidate.gpu_type = device.type
                            candidate.gpu_indexes.append(device.index)
                            candidate.gpu_addresses.append(device.network.inet)
                            candidate.computed_resource_claim.vram[device.index] = (
                                device_vram_request
                            )
                        # Increase subordinate worker's devices.
                        else:
                            subworker.gpu_type = device.type
                            subworker.gpu_indexes.append(device.index)
                            subworker.gpu_addresses.append(device.network.inet)
                            subworker.computed_resource_claim.vram[device.index] = (
                                device_vram_request
                            )

                        # Stats
                        worker_vram_request_remain -= device_vram_request
                        local_world_size_remain -= 1

                    # Skip if selected devices are not satisfied in semi-automatic mode.
                    if local_world_size_remain > 0:
                        continue

                    # Append the subordinate worker to the candidate.
                    if subworker_index >= 0:
                        candidate.subordinate_workers.append(subworker)

                    # Stats
                    subworker_index += 1
                    worker_group_vram_request_remain = worker_vram_request_remain
                    world_size_remain -= local_world_size
                    selected_workers.append(worker)

                if request_usage.vram - worker_group_vram_request_remain > largest_vram:
                    largest_vram = request_usage.vram - worker_group_vram_request_remain
                    workers_combination = selected_workers
                    worker_count = len(worker_group)
                    device_count_per_worker = device_count

                # Skip if selected devices are not satisfied in semi-automatic mode.
                if world_size_remain > 0:
                    continue

                # Skip if attention heads cannot be divided by the selected devices count in automatic mode.
                elif world_size_remain < -1:
                    if not self._is_tp_size_divisible(abs(world_size_remain + 1)):
                        continue

                # Skip if the worker group does not have enough VRAM.
                if worker_group_vram_request_remain > 0:
                    continue

                # Skip if the subordinate workers are empty.
                if not candidate.subordinate_workers:
                    continue

                # Append the candidate if satisfied the requested resources.
                candidates.append(candidate)
                logger.debug(
                    f"Found intermediate candidate: {candidate.to_log_string()}"
                )

        if len(candidates) == 0:
            # Nothing can be return, construct scheduling message
            worker_names = [worker.name for worker in workers_combination]
            worker_names_msg = (
                str(worker_names[:3]).rstrip("]")
                + f"...(more {len(worker_names) - 3})]"
                if len(worker_names) > 3
                else str(worker_names)
            )
            message = f"The optimal combination {worker_names_msg} provides {byte_to_gib(largest_vram)} GiB of allocatable VRAM."
            if worker_count - len(workers_combination) > 0:
                message += f" There are {worker_count - len(workers_combination)} {'workers' if worker_count - len(workers_combination) > 1 else 'worker'} that can provide {device_count_per_worker} {'devices' if device_count_per_worker > 1 else 'device'}, as the workers in the combination, but some devices among them fail to meet requirements."
            self._scheduling_messages.append(message)
            self._scheduling_messages.append(
                "Cannot find a suitable worker combination to run the model in distributed mode. "
                "If you are confident that the resources are sufficient, you may manually schedule the model by selecting the workers and devices."
            )

        return candidates

    async def _construct_candidates(  # noqa: C901
        self,
        request_usage: RequestEstimateUsage,
        workers: List[Worker],
    ) -> List[ModelInstanceScheduleCandidate]:

        # Result.
        candidates: List[ModelInstanceScheduleCandidate] = []

        # Validate
        if not await self._validate_parallelized():
            return []

        # Set default scheduling message
        self._scheduling_messages.append(
            f"The model requires approximately {byte_to_gib(request_usage.vram)} GiB VRAM and {byte_to_gib(request_usage.ram)} GiB RAM."
        )
        if self._serving_params.npu_memory_fraction:
            self._scheduling_messages.append(
                f"With --npu-memory-fraction={self._serving_params.npu_memory_fraction}, "
                f"all GPUs combined need to provide at least {(byte_to_gib(request_usage.vram) / self._serving_params.npu_memory_fraction):.2f} GiB of total VRAM "
                f"and each GPU needs {int(self._serving_params.npu_memory_fraction * 100)}% of allocatable VRAM."
            )

        # Statisfy the selected devices count, if specified.
        if self._model.gpu_selector and self._model.gpu_selector.gpu_ids:
            return self._manual_select_candidates(
                workers,
                request_usage,
            )

        # Available worker devices: {Worker: {Device Index: Device}},
        # all devices are in sorted.
        available_worker_devices_idx = await self._get_available_worker_devices_idx(
            workers, request_usage.ram
        )
        if not available_worker_devices_idx:
            return candidates

        # Try to find a single worker that can satisfy the requested resources.
        candidates = await self._select_single_worker(
            available_worker_devices_idx,
            request_usage,
        )

        # Return if found candidates.
        if candidates:
            return candidates

        # Try to find multiple workers that can satisfy the requested resources.
        candidates = await self._select_multi_workers(
            available_worker_devices_idx,
            request_usage,
        )

        return candidates

    async def __get_worker_alloc(self, worker: Worker) -> Allocatable:
        """
        Get the allocatable resources of the worker.
        This method caches the allocatable resources to avoid redundant database queries.
        """

        if worker.id in self.__worker_alloc_idx:
            return self.__worker_alloc_idx[worker.id]

        worker_alloc = get_worker_allocatable_resource(self._model_instances, worker)
        if not worker_alloc:
            logger.warning(f"Worker {worker.name} has no allocatable resources.")
            worker_alloc = Allocatable(ram=0, vram={0: 0})

        self.__worker_alloc_idx[worker.id] = worker_alloc
        self.__worker_sorted_devices_idx[worker.id] = sorted(
            worker.status.gpu_devices,
            key=lambda device: worker_alloc.vram.get(device.index, 0),
            reverse=True,
        )
        return worker_alloc
