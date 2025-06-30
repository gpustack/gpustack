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
    moe_num_experts: Optional[int] = None
    moe_num_shared_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None

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
        if not self.moe_num_experts:
            for key in [
                "n_routed_experts",
                "num_local_experts",
                "num_experts",
            ]:
                if value := getattr(pretrained_config, key, None):
                    setattr(self, "moe_num_experts", value)
                    break
        if self.moe_num_experts and not self.moe_num_shared_experts:
            for key in [
                "n_shared_experts",
                "num_shared_experts",
            ]:
                if value := getattr(pretrained_config, key, None):
                    setattr(self, "moe_num_shared_experts", value)
                    break

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
            worker_device_ids_map = parse_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            selected_worker_devices_cnt = 0
            selected_devices_cnt = 0
            for worker_name, device_ids in worker_device_ids_map.items():
                device_indexes = []
                for device_id in device_ids:
                    valid, matched = parse_gpu_id(device_id)
                    if valid:
                        device_index = safe_int(matched.get("gpu_index"))
                        device_indexes.append(device_index)
                if selected_worker_devices_cnt == 0:
                    selected_worker_devices_cnt = len(device_indexes)
                elif selected_worker_devices_cnt != len(device_indexes):
                    raise ValueError(
                        f"Selected devices count for worker {worker_name} is not matched with the previous worker."
                    )
                if selected_worker_devices_cnt & (selected_worker_devices_cnt - 1) != 0:
                    raise ValueError(
                        f"Selected devices count for worker {worker_name} is not power of 2."
                    )
                self._selected_worker_name_devices_idx[worker_name] = device_indexes
                selected_devices_cnt += len(device_indexes)
            # Configure serving parameters to help following process.
            # If the given serving parameters are not matched,
            # the parsing logic will raise an error.
            self._serving_params.local_world_size = selected_worker_devices_cnt
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
            reqeust_ram = safe_int(self._model.env.get("GPUSTACK_MODEL_RAM_CLAIM", 0))
            request_vram = safe_int(self._model.env.get("GPUSTACK_MODEL_VRAM_CLAIM", 0))
            if request_vram > 0:
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
        if self._model.env:
            reserved_memory_gb = safe_int(
                self._model.env.get("RESERVED_MEMORY_GB", "3")
            )
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

        # Validate whether model can be parallelized.
        if num_attention_heads := self._model_params.num_attention_heads:
            if self._selected_worker_name_devices_idx:
                if num_attention_heads % self._serving_params.local_world_size != 0:
                    self._diagnostic_messages.append(
                        f"Model's attention heads ({num_attention_heads}) must be divisible by "
                        f"the selected devices count ({self._serving_params.local_world_size}) per worker."
                    )
                    return candidates
            else:
                if self._serving_params.pipeline_parallel_size > 1:
                    if (
                        num_attention_heads % self._serving_params.tensor_parallel_size
                        != 0
                    ):
                        self._diagnostic_messages.append(
                            f"Model's attention heads ({num_attention_heads}) must be divisible by "
                            f"the --tensor-parallel-size ({self._serving_params.tensor_parallel_size})."
                        )
                        return candidates
                else:
                    if num_attention_heads % self._serving_params.world_size != 0:
                        if self._serving_params.data_parallel_size > 1:
                            self._diagnostic_messages.append(
                                f"Model's attention heads ({num_attention_heads}) must be divisible by "
                                f"the world size ({self._serving_params.world_size}), "
                                f"multiplied by --data-parallel-size ({self._serving_params.data_parallel_size}) "
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
                        return candidates
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
                                return candidates
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
                                return candidates

        """
        Get available workers.
        """

        ram_request, vram_request = estimated_usage[0], estimated_usage[1]

        # Available worker devices: {Worker: {Device Index: Device}},
        # all devices are in sorted.
        available_worker_devices_idx: Dict[Worker, Dict[int, GPUDeviceInfo]] = {}

        # Get available worker devices for manual mode.
        if self._selected_worker_name_devices_idx:
            # Record the remaining count needed to be selected,
            # which is used to determine if enough devices are selected.
            world_size_remain = self._serving_params.world_size

            for worker in workers:
                # Break if selected devices are enough,
                # if the value is negative, it means not selected.
                if world_size_remain == 0:
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
                        f"Worker {worker.name} does not satisfy the selected devices: "
                        f"{self._selected_worker_name_devices_idx[worker.name]}."
                    )
                    return candidates

                # Stats
                world_size_remain -= len(selected_devices_idx)

                # Index the worker and its devices.
                available_worker_devices_idx[worker] = selected_devices_idx

            # Validate if the selected workers are matched with the available workers,
            # if not, return.
            if len(self._selected_worker_name_devices_idx) > len(
                available_worker_devices_idx
            ):
                unavailable_workers = set(
                    self._selected_worker_name_devices_idx.keys()
                ) - set(worker.name for worker in available_worker_devices_idx.keys())
                if len(unavailable_workers) > 1:
                    self._diagnostic_messages.append(
                        f"Unavailable selected workers [{', '.join(unavailable_workers)}]."
                    )
                else:
                    self._diagnostic_messages.append(
                        f"Unavailable selected worker {unavailable_workers.pop()}."
                    )
                return candidates

        # Otherwise, get available worker devices for semi-automatic or automatic mode.
        else:
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
                selected_devices_idx: Dict[int, GPUDeviceInfo] = {
                    device.index: device
                    for device in self.__worker_sorted_devices_idx[worker.id]
                }

                # Index the worker and its devices.
                available_worker_devices_idx[worker] = selected_devices_idx

            # Validate if there are available workers,
            # if not, return.
            if not available_worker_devices_idx:
                self._diagnostic_messages.append("No available workers found.")
                return candidates

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

        """
        Statisfy the selected devices count, if specified.
        """

        if self._selected_worker_name_devices_idx:
            candidate: Optional[ModelInstanceScheduleCandidate] = None
            subworker: Optional[ModelInstanceSubordinateWorker] = None
            subworker_index = -1

            # Record the remaining VRAM request in the worker group level,
            # which is used to determine if enough devices are selected.
            workers_vram_request_remain = vram_request

            for worker, devices in available_worker_devices_idx.items():
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

                # Worker allocation.
                worker_alloc = await self.__get_worker_alloc(worker)
                # Record the remaining VRAM request in the worker level,
                # which is used to determine if enough devices are selected.
                worker_vram_request_remain = workers_vram_request_remain

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
                    worker_vram_request_remain -= device_vram_request

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
                workers_vram_request_remain = worker_vram_request_remain

            # Validate
            if workers_vram_request_remain > 0:
                candidate.overcommit = True
                self._diagnostic_messages.append(
                    f"Selected devices do not have enough VRAM to satisfy the request {byte_to_gib(vram_request)} GiB, "
                    f"still exceeds VRAM {byte_to_gib(workers_vram_request_remain)} GiB."
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
            # Break if enabled quick fit and found candidates.
            if candidates and quick_fit:
                break

            # Skip if the worker does not have enough devices.
            if 0 < self._serving_params.world_size > len(devices):
                continue

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

            # Worker allocation.
            worker_alloc = await self.__get_worker_alloc(worker)
            # Record the remaining VRAM request in the worker level,
            # which is used to determine if enough devices are selected.
            worker_vram_request_remain = vram_request
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
                    if self._model_params.num_attention_heads:
                        # Validate if attention heads can be divided by the selected devices count.
                        if world_size_remain < -1 and (
                            self._model_params.num_attention_heads
                            % abs(world_size_remain + 1)
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
                worker_vram_request_remain -= device_vram_request
                world_size_remain -= 1

            # Skip if selected devices are not satisfied in semi-automatic mode.
            if world_size_remain > 0:
                continue

            # Skip if attention heads cannot be divided by the selected devices count in automatic mode.
            elif world_size_remain < -1:
                if (
                    self._model_params.num_attention_heads
                    and self._model_params.num_attention_heads
                    % abs(world_size_remain + 1)
                    != 0
                ):
                    continue

            # Skip if the worker does not have enough VRAM.
            if worker_vram_request_remain > 0:
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
            # Break if enabled quick fit and found candidates.
            if candidates and quick_fit:
                break

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
                    # Skip if the attention heads can be divided by the selected devices count.
                    if (
                        self._model_params.num_attention_heads
                        and self._model_params.num_attention_heads % local_world_size
                        != 0
                    ):
                        local_world_size -= 1
                        continue
                    # Found a valid local world size.
                    local_world_size_group.append(local_world_size)
                    local_world_size -= 1
                local_world_size_group.append(1)
                local_world_size_group.reverse()

            # Iterate over the local world sizes to find candidates.
            for local_world_size in local_world_size_group:
                # Break if enabled quick fit and found candidates.
                if candidates and quick_fit:
                    break

                candidate: Optional[ModelInstanceScheduleCandidate] = None
                subworker: Optional[ModelInstanceSubordinateWorker] = None
                subworker_index = -1

                # Record the remaining VRAM request in the worker group level,
                # which is used to determine if enough devices are selected.
                worker_group_vram_request_remain = vram_request
                # Record the remaining count needed to be selected,
                # which is used to determine if enough devices are selected.
                world_size_remain = self._serving_params.world_size

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

                # Skip if selected devices are not satisfied in semi-automatic mode.
                if world_size_remain > 0:
                    continue

                # Skip if attention heads cannot be divided by the selected devices count in automatic mode.
                elif world_size_remain < -1:
                    if (
                        self._model_params.num_attention_heads
                        and self._model_params.num_attention_heads
                        % abs(world_size_remain + 1)
                        != 0
                    ):
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
