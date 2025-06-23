import asyncio
import dataclasses
import enum
import logging
import os
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncEngine

from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
)
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    get_local_model_weight_size,
)
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    SourceEnum,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.workers import GPUDeviceInfo, Worker
from gpustack.config import Config
from gpustack.server.db import get_engine
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_id, parse_gpu_ids_by_worker
from gpustack.utils.hub import (
    get_model_weight_size,
    get_pretrained_config,
    get_hf_text_config,
    get_max_model_len,
)
from gpustack.utils.unit import byte_to_gib
from gpustack.worker.backends.ascend_mindie import AscendMindIEParameters

logger = logging.getLogger(__name__)


class ModelAttentionTypeEnum(enum.Enum):
    UNK = "unknown"
    MHA = "multi_head_attention"
    GQA = "grouped_query_attention"
    MQA = "multi_query_attention"
    MLA = "multi_head_latent_attention"


@dataclasses.dataclass
class ModelParameters:
    derived_max_seq_len: int = 0
    num_hidden_layers: int = 0
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: int = 1
    n_group: Optional[int] = None
    head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    torch_dtype: str = "bfloat16"
    quantize: Optional[str] = None
    quantization_config: Optional[Dict] = None

    def from_model(self, model: Model):  # noqa: C901
        """
        Parse the model's (hyper)parameters from the model.
        """

        # Parse
        pretrained_config = get_pretrained_config(model, trust_remote_code=True)
        if not pretrained_config:
            raise ValueError(f"Failed to get model {model.name} pretrained config")
        for attr_name in [attr.name for attr in dataclasses.fields(self.__class__)]:
            try:
                if attr_value := getattr(pretrained_config, attr_name):
                    setattr(self, attr_name, attr_value)
            except AttributeError:
                # If reach here, that means the field is an internal property,
                # which would not register in the argument parser.
                pass

        # Get maximum sequence length.
        try:
            pretrained_config = get_pretrained_config(model)
            pretrained_or_hf_text_config = get_hf_text_config(pretrained_config)
            self.derived_max_seq_len = get_max_model_len(pretrained_or_hf_text_config)
        except Exception as e:
            raise ValueError(
                f"Failed to get model {model.name} maximum sequence length: {e}"
            ) from e

        # Default
        if not self.num_attention_heads:
            # For backward compatibility, try to get num_attention_heads from llm_config.
            llm_config = getattr(pretrained_config, "llm_config", None)
            if llm_config:
                self.num_attention_heads = getattr(
                    llm_config, "num_attention_heads", None
                )
        if not self.qk_nope_head_dim and self.hidden_size and self.num_attention_heads:
            self.head_dim = self.hidden_size // self.num_attention_heads

    def get_attention_type(self) -> ModelAttentionTypeEnum:
        """
        Get the attention type based on the hyperparameters.
        """

        if self.num_attention_heads:
            if self.num_key_value_heads == 1:
                return ModelAttentionTypeEnum.MQA
            elif (
                1 < self.num_key_value_heads < self.num_attention_heads
                and self.num_attention_heads % self.num_key_value_heads == 0
            ):
                return ModelAttentionTypeEnum.GQA
            elif self.num_key_value_heads == self.num_attention_heads:
                if self.n_group and self.n_group > 1:
                    return ModelAttentionTypeEnum.MLA
                return ModelAttentionTypeEnum.MHA
        return ModelAttentionTypeEnum.UNK


class AscendMindIEResourceFitSelector(ScheduleCandidatesSelector):
    """
    Ascend MindIE Resource Fit Selector,
    which is designed to select deployment candidates.
    """

    def __init__(  # noqa: C901
        self,
        config: Config,
        model: Model,
    ):
        # GPUStack configuration.
        self._config: Config = config
        # Model instance.
        self._model: Model = model
        # Diagnostic message to be set to the model instance.
        self._diagnostic_messages: List[str] = []
        # Database engine.
        self._engine: AsyncEngine = get_engine()
        # Model's hyperparameters.
        self._model_params = ModelParameters()
        # Serving parameters.
        self._serving_params = AscendMindIEParameters()
        # Indexer of selected worker name and its device indexes: {Worker Name: [Device Index 0, Device Index 1, ...]}.
        self._selected_worker_name_devices_idx: Dict[str, List[int]] = {}
        # Temporary indexer for caching worker's allocatable, avoiding redundant database queries: {Worker ID: Allocatable}.
        self.__worker_alloc_idx: Dict[int, Allocatable] = {}
        # Temporary indexer for caching worker's devices that sort by VRAM size: {Worker ID: sorted([Device 0, Device 1, ...])}.
        self.__worker_sorted_devices_idx: Dict[int, List[GPUDeviceInfo]] = {}

        # Expand selected devices.
        if model.gpu_selector and model.gpu_selector.gpu_ids:
            selected_devices_cnt = 0
            worker_device_ids_map = parse_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            for worker_name, device_ids in worker_device_ids_map.items():
                device_indexes = []
                for device_id in device_ids:
                    valid, matched = parse_gpu_id(device_id)
                    if valid:
                        device_index = safe_int(matched.get("gpu_index"))
                        device_indexes.append(device_index)
                        selected_devices_cnt += 1
                self._selected_worker_name_devices_idx[worker_name] = device_indexes
            # Configure serving parameters to help following process(including selecting and parsing).
            # If the given serving parameters are not matched,
            # the parsing logic will raise an error.
            self._serving_params.world_size = selected_devices_cnt

        # Parse model config.
        try:
            self._model_params.from_model(model)
        except Exception as e:
            raise ValueError(f"Failed to parse model {model.name} hyperparameters: {e}")

        # Parse model params.
        if model.backend_parameters:
            max_seq_len = self._model_params.derived_max_seq_len
            if max_seq_len > 8192:
                max_seq_len = 8192
            self._serving_params.max_seq_len = max_seq_len
            try:
                self._serving_params.from_args(model.backend_parameters)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse model {model.name} serve parameters: {e}"
                )
            # Set selected device count based on the parallelism settings.
            if self._serving_params.tensor_parallel_size > 0:
                world_size = self._serving_params.tensor_parallel_size
                if self._serving_params.data_parallel_size > 0:
                    world_size = (
                        self._serving_params.tensor_parallel_size
                        * self._serving_params.data_parallel_size
                    )
                # Configure world size to help following process: selecting.
                self._serving_params.world_size = world_size
            elif self._serving_params.data_parallel_size > 0:
                world_size = self._serving_params.data_parallel_size
                # Configure world size to help following process: selecting.
                self._serving_params.world_size = world_size

    def get_messages(self) -> List[str]:
        return self._diagnostic_messages

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Select available deployment candidates based on the resource requirements.

        The selection process involves:
        1. Estimating the resource requirements.
        2. Constructing candidates that can accommodate the estimated resource requirements.
        """

        # Estimate resource usage.
        estimated_usage = await self._estimate_usage()

        # Filter workers based on estimated usage.
        candidates = await self._construct_candidates(estimated_usage, workers)
        if not candidates:
            logger.error(
                f"No suitable candidates found for the model {self._model.name}."
            )
            if len(self._diagnostic_messages) == 0:
                message = f"Model requires RAM {byte_to_gib(estimated_usage[0])} GiB and VRAM {byte_to_gib(estimated_usage[1])} GiB, "
                if self._serving_params.world_size > 0:
                    message += (
                        f"with {self._serving_params.world_size} devices selected, "
                    )
                message += f"but {'available' if self._selected_worker_name_devices_idx else 'selected'} workers do not satisfy the requirements."
                self._diagnostic_messages.append(message)
            return []

        return candidates

    async def _estimate_usage(self) -> Optional[Tuple[int, int]]:  # noqa: C901
        """
        Estimate the resource usage of the model instance.

        Formula:

            RAM  = FOOTPRINT + KV_CACHE_SWAPPABLE
            VRAM = FOOTPRINT + WEIGHT + KV_CACHE + COMPUTATION
        """

        # A potential workaround if the empirical resource estimation is biased greatly.
        if self._model.env:
            env = self._model.env
            reqeust_ram = safe_int(env.get("GPUSTACK_MODEL_RAM_CLAIM", 0))
            request_vram = safe_int(env.get("GPUSTACK_MODEL_VRAM_CLAIM", 0))
            if reqeust_ram or request_vram:
                return reqeust_ram, request_vram

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

        # Hardcode the VRAM footprint for now.
        vram_footprint = 3 * 1024**3

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
                if os.path.exists(local_path):
                    vram_weight = get_local_model_weight_size(local_path)
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when getting weight size for model {self._model.name}"
            )
        except Exception as e:
            logger.warning(f"Cannot get weight size for model {self._model.name}: {e}")

        n_tokens = self._serving_params.max_seq_len
        n_layers = self._model_params.num_hidden_layers
        n_heads = 1
        d_head = self._model_params.head_dim
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
                ).lower()
                if kv_quant_type == "c8":
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

        ram = ram_footprint + ram_kv_cache_swappable
        vram = vram_footprint + vram_weight + vram_kv_cache + vram_computation

        logger.info(
            f"Estimated model resource usage: "
            f"RAM: footprint = {byte_to_gib(ram_footprint)} GiB, kv_cache_swappable = {byte_to_gib(ram_kv_cache_swappable)} GiB, total = {byte_to_gib(ram)} GiB, "
            f"VRAM: footprint = {byte_to_gib(vram_footprint)} GiB, kv_cache = {byte_to_gib(vram_kv_cache)} GiB, weight = {byte_to_gib(vram_weight)} GiB, computation = {byte_to_gib(vram_computation)} GiB, total = {byte_to_gib(vram)} GiB"
        )

        return ram, vram

    async def _construct_candidates(  # noqa: C901
        self,
        estimated_usage: Tuple[int, int],
        workers: List[Worker],
        quick_fit: bool = True,
    ) -> List[ModelInstanceScheduleCandidate]:

        # Result.
        candidates: List[ModelInstanceScheduleCandidate] = []

        """
        Validate
        """

        if self._selected_worker_name_devices_idx:
            # Record single worker selected devices count,
            # which is used to validate the number of selected devices in single worker.
            selected_worker_devices_cnt = 0

            # Validate whether all selected workers have the same number of devices.
            for _, device_indexes in self._selected_worker_name_devices_idx.items():
                if selected_worker_devices_cnt == 0:
                    selected_worker_devices_cnt = len(device_indexes)
                elif selected_worker_devices_cnt != len(device_indexes):
                    self._diagnostic_messages.append(
                        "All selected workers must choose the same number of devices."
                    )
                    return candidates
            # Validate whether the number of selected devices is power of 2.
            if selected_worker_devices_cnt & (selected_worker_devices_cnt - 1) != 0:
                self._diagnostic_messages.append(
                    f"Number of selected devices must be power of 2, but got {selected_worker_devices_cnt}."
                )
                return candidates
            # Validate whether the number of attention heads is divisible by the selected device count.
            if (
                self._model_params.num_attention_heads
                and self._model_params.num_attention_heads % selected_worker_devices_cnt
                != 0
            ):
                self._diagnostic_messages.append(
                    f"Model's attention heads ({self._model_params.num_attention_heads}) must be divisible by "
                    f"each worker's selected device count ({selected_worker_devices_cnt})."
                )
                return candidates
        # Validate whether model can be parallelized.
        if self._model_params.num_attention_heads:
            if (
                self._model_params.num_attention_heads
                % self._serving_params.tensor_parallel_size
                != 0
            ):
                self._diagnostic_messages.append(
                    f"Model's attention heads ({self._model_params.num_attention_heads}) must be divisible by "
                    f"the tensor parallel size ({self._serving_params.tensor_parallel_size})."
                )
                return candidates
            if (
                self._model_params.num_attention_heads
                % self._serving_params.data_parallel_size
                != 0
            ):
                self._diagnostic_messages.append(
                    f"Model's attention heads ({self._model_params.num_attention_heads}) must be divisible by "
                    f"the data parallel size ({self._serving_params.data_parallel_size})."
                )
                return candidates

        """
        Get available workers.
        """

        ram_request, vram_request = estimated_usage[0], estimated_usage[1]

        # Get available worker devices: {Worker: {Device Index: Device}},
        # all devices are in sorted.
        available_worker_devices_idx: Dict[Worker, Dict[int, GPUDeviceInfo]] = {}
        # Record the remaining count needed to be selected,
        # which is used to determine if enough devices are selected.
        selected_devices_cnt_remain = self._serving_params.world_size

        for worker in workers:
            # Choice devices of the worker based on the selected worker names and devices.
            if self._selected_worker_name_devices_idx:
                # Break if selected devices are enough,
                # if the value is negative, it means not selected.
                if selected_devices_cnt_remain == 0:
                    break

                # Skip if the worker is not in the selected worker names.
                if worker.name not in self._selected_worker_name_devices_idx:
                    continue

                _ = await self.__get_worker_alloc(worker)

                # Get selected devices of the worker: {Device Index: Device}.
                selected_devices_idx: Dict[int, GPUDeviceInfo] = {
                    device.index: device
                    for device in self.__worker_sorted_devices_idx[worker.id]
                    if device.index
                    in self._selected_worker_name_devices_idx[worker.name]
                }

                # Return if the selected devices aren't matched with the worker's available devices.
                if len(selected_devices_idx) < len(
                    self._selected_worker_name_devices_idx[worker.name]
                ):
                    self._diagnostic_messages.append(
                        f"Worker {worker.name} does not have enough devices to satisfy the selected devices: "
                        f"{self._selected_worker_name_devices_idx[worker.name]}."
                    )
                    return candidates

                # Stats
                selected_devices_cnt_remain -= len(selected_devices_idx)

            # Otherwise, use all devices of the worker.
            else:
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
                selected_devices_idx: Dict[int, GPUDeviceInfo] = {
                    device.index: device
                    for device in self.__worker_sorted_devices_idx[worker.id]
                }

            # Index the worker and its devices.
            available_worker_devices_idx[worker] = selected_devices_idx

        # Return if no available workers found.
        if not available_worker_devices_idx:
            self._diagnostic_messages.append("No available workers found.")
            return candidates

        # Return if the selected workers aren't matched with the available workers.
        if self._selected_worker_name_devices_idx:
            if len(self._selected_worker_name_devices_idx) > len(
                available_worker_devices_idx
            ):
                unavailable_workers = set(
                    self._selected_worker_name_devices_idx.keys()
                ) - set(worker.name for worker in available_worker_devices_idx.keys())
                if len(unavailable_workers) > 1:
                    self._diagnostic_messages.append(
                        f"Selected workers {', '.join(unavailable_workers)} are not available."
                    )
                else:
                    self._diagnostic_messages.append(
                        f"Selected worker {unavailable_workers.pop()} is not available."
                    )
                return candidates

        # Sort available worker devices by allocatable resources,
        # make sure the worker with the largest VRAM is first.
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

        """
        Statisfy the selected device count, if specified.
        """

        if self._selected_worker_name_devices_idx:
            candidate: Optional[ModelInstanceScheduleCandidate] = None
            subworker: Optional[ModelInstanceSubordinateWorker] = None
            subworker_index = -1

            # Record the remaining VRAM request in the worker group level,
            # which is used to determine if enough devices are selected.
            worker_group_vram_request_remain = vram_request

            for worker, devices in available_worker_devices_idx.items():
                worker_alloc = await self.__get_worker_alloc(worker)

                # Construct candidate by the main worker.
                if subworker_index < 0:
                    candidate = ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_indexes=[],
                        gpu_addresses=[],
                        computed_resource_claim=ComputedResourceClaim(
                            ram=ram_request,
                            vram={},
                        ),
                        subordinate_workers=[],
                    )
                # Increase subordinate workers.
                else:
                    subworker = ModelInstanceSubordinateWorker(
                        worker_id=worker.id,
                        worker_ip=worker.ip,
                        total_gpus=len(worker.status.gpu_devices),
                        gpu_indexes=[],
                        gpu_addresses=[],
                        computed_resource_claim=ComputedResourceClaim(
                            ram=ram_request,
                            vram={},
                        ),
                    )

                # Record the remaining VRAM request in the devices level,
                # which is used to determine if enough devices are selected.
                devices_vram_request_remain = worker_group_vram_request_remain

                # Iterate over the worker's devices from the largest VRAM.
                for device in self.__worker_sorted_devices_idx[worker.id]:
                    # Skip if the device is not in the selected worker's devices.
                    if device.index not in devices:
                        continue

                    # Calculate device VRAM request/allocate.
                    device_vram_request = int(
                        devices[device.index].memory.total
                        * self._serving_params.npu_memory_fraction
                    )
                    device_vram_allocate = worker_alloc.vram.get(device.index, 0)

                    # Increase main worker's devices.
                    if subworker_index < 0:
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
                    # Increase subordinate worker's devices.
                    else:
                        subworker.gpu_indexes.append(device.index)
                        subworker.gpu_addresses.append(
                            device.network.inet
                            if device.network
                            and device.network.status == 'up'
                            and device.network.inet
                            else "-.-.-.-"
                        )
                        subworker.computed_resource_claim.vram[device.index] = (
                            device_vram_request
                        )

                    # Validate
                    if device_vram_request > device_vram_allocate:
                        candidate.overcommit = True
                        self._diagnostic_messages.append(
                            f"Worker {worker.name} device {device.index} requests VRAM {byte_to_gib(device_vram_request)} GiB, "
                            f"exceeds the allocatable {byte_to_gib(device_vram_allocate)} GiB."
                        )

                    # Stats
                    devices_vram_request_remain -= device_vram_request

                # Validate
                ram_allocate = worker_alloc.ram
                if ram_request > ram_allocate:
                    candidate.overcommit = True
                    self._diagnostic_messages.append(
                        f"Worker {worker.name} requests RAM {byte_to_gib(ram_request)} GiB, "
                        f"exceeds the allocatable {byte_to_gib(ram_allocate)} GiB."
                    )

                # Append the subordinate worker to the candidate.
                if subworker_index >= 0:
                    candidate.subordinate_workers.append(subworker)

                # Stats
                subworker_index += 1
                worker_group_vram_request_remain = devices_vram_request_remain

            # Validate
            if worker_group_vram_request_remain > 0:
                candidate.overcommit = True
                self._diagnostic_messages.append(
                    f"Selected devices do not have enough VRAM to satisfy the request {byte_to_gib(vram_request)} GiB, "
                    f"still exceeds VRAM {byte_to_gib(worker_group_vram_request_remain)} GiB."
                )

            # Return one candidate for selected workers.
            candidates.append(candidate)
            logger.debug(f"Found intermediate candidate: {candidate.to_log_string()}")
            return candidates

        """
        Try to find a single worker that can satisfy the requested resources.
        """

        # Iterate over the workers.
        for worker, devices in available_worker_devices_idx.items():
            # Skip if the worker does not have enough devices.
            if 0 < self._serving_params.world_size > len(devices):
                continue

            worker_alloc = await self.__get_worker_alloc(worker)

            # Construct candidate by the worker.
            candidate = ModelInstanceScheduleCandidate(
                worker=worker,
                gpu_indexes=[],
                gpu_addresses=[],
                computed_resource_claim=ComputedResourceClaim(
                    ram=ram_request,
                    vram={},
                ),
            )

            # Record the remaining count needed to be selected,
            # which is used to determine if enough devices are selected.
            selected_devices_cnt_remain = self._serving_params.world_size
            # Record single worker selected devices count,
            # which is used to determine if enough devices are selected.
            selected_worker_devices_cnt = 0
            # Record the remaining VRAM request in the devices level,
            # which is used to determine if enough devices are selected.
            devices_vram_request_remain = vram_request

            # Iterate over the worker's devices from the largest VRAM.
            for device in self.__worker_sorted_devices_idx[worker.id]:
                # Break if selected devices are satisfied in selected mode.
                if selected_devices_cnt_remain == 0:
                    break

                # Break if selected devices or requested VRAM are satisfied in automatic mode.
                if selected_devices_cnt_remain < 0 and devices_vram_request_remain <= 0:
                    if self._model_params.num_attention_heads:
                        # Validate if attention heads can be divided by the selected devices count.
                        if selected_worker_devices_cnt and (
                            self._model_params.num_attention_heads
                            % selected_worker_devices_cnt
                            == 0
                        ):
                            break
                        # Otherwise, find at least one device.
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
                devices_vram_request_remain -= device_vram_request
                selected_devices_cnt_remain -= 1
                selected_worker_devices_cnt += 1

            # Skip if the worker does not have enough VRAM.
            if devices_vram_request_remain > 0:
                continue

            # Skip if no devices are selected from the worker.
            if selected_worker_devices_cnt == 0:
                continue

            # Skip if attention heads cannot be divided by the selected devices count.
            if (
                self._model_params.num_attention_heads
                and self._model_params.num_attention_heads % selected_worker_devices_cnt
                != 0
            ):
                continue

            # Skip if the total devices count is not the same as the requested world size.
            if 0 < self._serving_params.world_size != selected_worker_devices_cnt:
                continue

            # Append the candidate if satisfied the requested resources.
            candidates.append(candidate)
            logger.debug(f"Found intermediate candidate: {candidate.to_log_string()}")

        # Return if enabled quick fit and found candidates.
        if candidates and quick_fit:
            return candidates

        """
        Try to find multiple workers that can satisfy the requested resources.
        """

        # Group workers by device count: {Devices Count: [Worker, ...]}.
        device_count_worker_group_idx: Dict[int, List[Worker]] = {}
        for worker in available_worker_devices_idx.keys():
            devices_count = len(self.__worker_sorted_devices_idx[worker.id])
            if devices_count not in device_count_worker_group_idx:
                device_count_worker_group_idx[devices_count] = []
            device_count_worker_group_idx[devices_count].append(worker)

        # Sort workers by device count in descending order.
        device_count_worker_group_idx = dict(
            sorted(
                device_count_worker_group_idx.items(),
                key=lambda item: item[0],
                reverse=True,
            )
        )

        # Iterate over the workers from the largest device count.
        for device_count, worker_group in device_count_worker_group_idx.items():
            worker_group_size = len(worker_group)
            # Skip if the worker group is smaller.
            if worker_group_size < 2:
                continue
            # Skip if the worker group does not have enough devices.
            if 0 < self._serving_params.world_size > device_count * worker_group_size:
                continue

            candidate: Optional[ModelInstanceScheduleCandidate] = None
            subworker: Optional[ModelInstanceSubordinateWorker] = None
            subworker_index = -1

            # Record previous single worker selected devices count,
            # which is used to determine if enough devices are selected.
            selected_worker_devices_cnt_previous = 0
            # Record the remaining VRAM request in the worker group level,
            # which is used to determine if enough devices are selected.
            worker_group_vram_request_remain = vram_request

            for worker in worker_group:
                worker_alloc = await self.__get_worker_alloc(worker)

                # Construct candidate by the main worker.
                if subworker_index < 0:
                    candidate = ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_indexes=[],
                        gpu_addresses=[],
                        computed_resource_claim=ComputedResourceClaim(
                            ram=ram_request,
                            vram={},
                        ),
                        subordinate_workers=[],
                    )
                # Increase subordinate workers.
                else:
                    subworker = ModelInstanceSubordinateWorker(
                        worker_id=worker.id,
                        worker_ip=worker.ip,
                        total_gpus=len(worker.status.gpu_devices),
                        gpu_indexes=[],
                        gpu_addresses=[],
                        computed_resource_claim=ComputedResourceClaim(
                            ram=ram_request,
                            vram={},
                        ),
                    )

                # Record the remaining count needed to be selected,
                # which is used to determine if enough devices are selected.
                selected_devices_cnt_remain = self._serving_params.world_size
                # Record single worker selected devices count,
                # which is used to determine if enough devices are selected.
                selected_worker_devices_cnt = 0
                # Record the remaining VRAM request in the devices level,
                # which is used to determine if enough devices are selected.
                devices_vram_request_remain = worker_group_vram_request_remain

                # Pad selected devices count remain,
                # make sure the selected devices count is not less than the previous worker in automatic mode.
                if (
                    self._serving_params.world_size
                    < 0
                    < selected_worker_devices_cnt_previous
                ):
                    selected_devices_cnt_remain = selected_worker_devices_cnt_previous

                # Iterate over the worker's devices from the largest VRAM.
                for device in self.__worker_sorted_devices_idx[worker.id]:
                    # Break if selected devices are satisfied in selected mode.
                    if selected_devices_cnt_remain == 0:
                        break

                    # Break if selected devices or requested VRAM are satisfied in automatic mode.
                    if (
                        selected_devices_cnt_remain < 0
                        and devices_vram_request_remain <= 0
                    ):
                        if self._model_params.num_attention_heads:
                            # Validate if attention heads can be divided by the selected devices count.
                            if selected_worker_devices_cnt and (
                                self._model_params.num_attention_heads
                                % selected_worker_devices_cnt
                                == 0
                            ):
                                break
                            # Otherwise, find at least one device.
                        else:
                            break

                    # Calculate device VRAM request.
                    device_vram_request = int(
                        available_worker_devices_idx[worker][device.index].memory.total
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
                        candidate.gpu_indexes.append(device.index)
                        candidate.gpu_addresses.append(device.network.inet)
                        candidate.computed_resource_claim.vram[device.index] = (
                            device_vram_request
                        )
                    # Increase subordinate worker's devices.
                    else:
                        subworker.gpu_indexes.append(device.index)
                        subworker.gpu_addresses.append(device.network.inet)
                        subworker.computed_resource_claim.vram[device.index] = (
                            device_vram_request
                        )

                    # Stats
                    devices_vram_request_remain -= device_vram_request
                    selected_devices_cnt_remain -= 1
                    selected_worker_devices_cnt += 1

                # Skip if no devices are selected from the worker.
                if selected_worker_devices_cnt == 0:
                    continue

                # Skip if attention heads cannot be divided by the selected devices count.
                if (
                    self._model_params.num_attention_heads
                    and self._model_params.num_attention_heads
                    % selected_worker_devices_cnt
                    != 0
                ):
                    continue

                # Skip if selected devices count is not the same as the previous worker.
                if self._serving_params.world_size < 0 and (
                    0
                    < selected_worker_devices_cnt_previous
                    != selected_worker_devices_cnt
                ):
                    continue

                # Append the subordinate worker to the candidate.
                if subworker_index >= 0:
                    candidate.subordinate_workers.append(subworker)

                # Stats
                subworker_index += 1
                selected_worker_devices_cnt_previous = selected_worker_devices_cnt
                worker_group_vram_request_remain = devices_vram_request_remain

            # Skip if the worker group does not have enough VRAM.
            if worker_group_vram_request_remain > 0:
                continue

            # Skip if no devices are selected from this worker group.
            selected_workers_devices_cnt = selected_worker_devices_cnt_previous * (
                subworker_index + 1
            )
            if selected_workers_devices_cnt == 0:
                continue

            # Skip if attention heads cannot be divided by the selected devices count.
            if (
                self._model_params.num_attention_heads
                and self._model_params.num_attention_heads
                % selected_workers_devices_cnt
                != 0
            ):
                continue

            # Skip if the total devices count is not the same as the requested world size.
            if 0 < self._serving_params.world_size != selected_workers_devices_cnt:
                continue

            # Append the candidate if satisfied the requested resources.
            if candidate and candidate.subordinate_workers:
                logger.debug(
                    f"Found intermediate candidate: {candidate.to_log_string()}"
                )
                candidates.append(candidate)

        return candidates

    async def __get_worker_alloc(self, worker: Worker) -> Allocatable:
        """
        Get the allocatable resources of the worker.
        This method caches the allocatable resources to avoid redundant database queries.
        """

        if worker.id in self.__worker_alloc_idx:
            return self.__worker_alloc_idx[worker.id]

        worker_alloc = await get_worker_allocatable_resource(self._engine, worker)
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
