import asyncio
import logging
import os
from typing import Callable, Dict, List, Optional

from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.client.worker_filesystem_client import WorkerFilesystemClient
from gpustack.policies.base import (
    Allocatable,
    Allocated,
)
from gpustack.schemas.models import (
    ModelInstance,
    Model,
    CategoryEnum,
    SourceEnum,
    is_llm_model,
)
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.workers import Worker, GPUDevicesStatus, GPUDeviceStatus
from pydantic import BaseModel

from gpustack.server.services import ModelFileService
from gpustack.utils.hub import get_model_weight_size, get_diffusion_model_weight_size

logger = logging.getLogger(__name__)


class WorkerGPUInfo(BaseModel):
    """
    Data structure to represent a GPU device with its associated worker information.
    """

    worker_id: int
    worker_name: str
    gpu_device: GPUDeviceStatus
    allocatable_vram: int  # in bytes


def get_worker_allocatable_resource(  # noqa: C901
    all_model_instances: List[ModelInstance],
    worker: Worker,
    gpu_type: Optional[str] = None,
) -> Allocatable:
    """
    Get the worker with the latest allocatable resources, if gpu_type is provided, only consider the GPUs of that type.
    """

    def update_allocated_vram(allocated, resource_claim):
        for gpu_index, vram in resource_claim.vram.items():
            allocated.vram[gpu_index] = allocated.vram.get(gpu_index, 0) + vram

    is_unified_memory = worker.status.memory.is_unified_memory
    model_instances = get_worker_model_instances(all_model_instances, worker)
    allocated = Allocated(ram=0, vram={})

    for model_instance in model_instances:
        # Handle resource allocation for main worker
        if model_instance.worker_id == worker.id and (
            gpu_type is None
            or model_instance.gpu_type is None
            or model_instance.gpu_type == gpu_type
        ):
            allocated.ram += model_instance.computed_resource_claim.ram or 0
            if model_instance.gpu_indexes:
                update_allocated_vram(allocated, model_instance.computed_resource_claim)

        # Handle resource allocation for subordinate workers
        if (
            model_instance.distributed_servers
            and model_instance.distributed_servers.subordinate_workers
        ):
            for (
                subordinate_worker
            ) in model_instance.distributed_servers.subordinate_workers:
                if subordinate_worker.worker_id != worker.id:
                    continue

                if subordinate_worker.computed_resource_claim and (
                    gpu_type is None or model_instance.gpu_type == gpu_type
                ):
                    # rpc server only consider the vram
                    update_allocated_vram(
                        allocated, subordinate_worker.computed_resource_claim
                    )

    allocatable = Allocatable(ram=0, vram={})
    if worker.status.gpu_devices:
        for _, gpu in enumerate(worker.status.gpu_devices):
            gpu_index = gpu.index

            if (
                gpu.memory is None
                or gpu.memory.total is None
                or (gpu_type is not None and gpu.type != gpu_type)
            ):
                continue
            allocatable_vram = max(
                (
                    gpu.memory.total
                    - allocated.vram.get(gpu_index, 0)
                    - worker.system_reserved.vram
                ),
                0,
            )
            allocatable.vram[gpu_index] = allocatable_vram

    allocatable.ram = max(
        (worker.status.memory.total - allocated.ram - worker.system_reserved.ram), 0
    )

    if is_unified_memory:
        allocatable.ram = max(
            allocatable.ram
            - worker.system_reserved.vram
            - sum(allocated.vram.values()),
            0,
        )

        # For UMA, we need to set the gpu memory to the minimum of
        # the calculated with max allow gpu memory and the allocatable memory.
        if allocatable.vram:
            allocatable.vram[0] = min(allocatable.ram, allocatable.vram[0])

    logger.debug(
        f"Worker {worker.name} gpu_type {gpu_type}, "
        f"reserved memory: {worker.system_reserved.ram}, "
        f"reserved gpu memory: {worker.system_reserved.vram}, "
        f"allocatable memory: {allocatable.ram}, "
        f"allocatable gpu memory: {allocatable.vram}"
    )
    return allocatable


def group_gpu_devices_by_memory(
    gpu_devices: GPUDevicesStatus,
) -> List[GPUDevicesStatus]:
    """
    Group GPU devices by allocatable memory size with the constraint that the minimum
    allocatable GPU memory in each group should not be less than 0.9 times the
    allocatable memory of other GPUs in the same group.

    Args:
        gpu_devices: List of GPU device information

    Returns:
        List of GPU device groups, where each group is a list of GPU devices

    Example:
        If we have GPUs with allocatable memory [8GB, 8.5GB, 16GB, 16.5GB, 32GB],
        they might be grouped as:
        - Group 1: [8GB, 8.5GB] (8GB >= 8.5GB * 0.9 = 7.65GB)
        - Group 2: [16GB, 16.5GB] (16GB >= 16.5GB * 0.9 = 14.85GB)
        - Group 3: [32GB]
    """
    if not gpu_devices:
        return []

    def get_allocatable_memory(gpu: GPUDeviceStatus) -> Optional[int]:
        """Calculate allocatable memory (total - allocated)"""
        if not gpu.memory or gpu.memory.total is None:
            return None
        allocated = gpu.memory.allocated or 0
        return gpu.memory.total - allocated

    # Filter out GPUs without valid memory information
    valid_gpus = []
    for gpu in gpu_devices:
        allocatable_memory = get_allocatable_memory(gpu)
        if allocatable_memory is not None and allocatable_memory > 0:
            valid_gpus.append(gpu)

    if not valid_gpus:
        return []

    # Sort GPUs by allocatable memory size (ascending order)
    sorted_gpus = sorted(valid_gpus, key=lambda gpu: get_allocatable_memory(gpu))

    groups = []
    current_group = []

    for gpu in sorted_gpus:
        if not current_group:
            # Start a new group
            current_group = [gpu]
        else:
            # Check if this GPU can be added to the current group
            min_allocatable_memory = get_allocatable_memory(
                current_group[0]
            )  # First GPU has minimum allocatable memory
            current_gpu_allocatable_memory = get_allocatable_memory(gpu)

            # Check if min_allocatable_memory >= current_gpu_allocatable_memory * 0.9
            # This ensures the minimum allocatable memory is not less than 0.9 times any other allocatable memory in the group
            if min_allocatable_memory >= current_gpu_allocatable_memory * 0.9:
                current_group.append(gpu)
            else:
                # Cannot add to current group, start a new group
                groups.append(current_group)
                current_group = [gpu]

    # Add the last group
    if current_group:
        groups.append(current_group)

    return groups


def group_workers_by_gpu_type(workers: List[Worker]) -> Dict[str, List[Worker]]:
    """
    Group workers by their GPU types.

    Args:
        workers: List of workers containing GPU devices

    Returns:
        Dictionary mapping GPU type to list of workers that with gpus of that type
    """
    gpu_type_to_workers: Dict[str, List[Worker]] = {}

    for worker in workers:
        if not worker.status or not worker.status.gpu_devices:
            gpu_type_to_workers.setdefault(None, []).append(worker)
            continue

        # Collect unique GPU types for this worker
        gpus: Dict[str, GPUDevicesStatus] = {}
        for gpu in worker.status.gpu_devices:
            gpus.setdefault(gpu.type, []).append(gpu)

        # Add worker to each GPU type group
        for gpu_type in gpus.keys():
            w = worker.model_copy()
            w.status.gpu_devices = gpus[gpu_type]
            gpu_type_to_workers.setdefault(gpu_type, []).append(w)

    return gpu_type_to_workers


def get_vram_claim_from_model_env(model: Model) -> Optional[int]:
    """
    Get the VRAM claim from model environment variable 'GPUSTACK_MODEL_VRAM_CLAIM' if set.
    """
    if model.env and 'GPUSTACK_MODEL_VRAM_CLAIM' in model.env:
        try:
            return int(model.env['GPUSTACK_MODEL_VRAM_CLAIM'])
        except ValueError:
            logger.warning(
                f"Invalid VRAM claim value for model {model.name}: {model.env['GPUSTACK_MODEL_VRAM_CLAIM']}"
            )

    return None


async def _get_cached_model_size(
    session: Optional[AsyncSession],
    model: Model,
) -> Optional[int]:
    """
    Get cached file size from downloaded ModelFile.
    """
    if not session:
        return None

    source_index = model.model_source_index
    if not source_index:
        return None

    model_files = await ModelFileService(session).get_by_source_index(source_index)
    if not model_files:
        return None

    # Find READY files with size
    for mf in model_files:
        if mf.state == ModelFileStateEnum.READY and mf.size:
            logger.info(f"Using cached size {mf.size} from ModelFile {mf.id}")
            return mf.size

    return None


async def estimate_model_vram(
    model: Model,
    token: Optional[str] = None,
    workers: Optional[List[Worker]] = None,
    session: Optional[AsyncSession] = None,
) -> int:
    """
    Estimate the vram requirement in bytes heuristically.
    This is the minimum requirement to help us decide how many GPUs are needed for the model.
    If users explicitly set parameters like tp & pp, this estimation is not needed.

    Formula for Diffusion (Image) models:

        VRAM = WEIGHT_SIZE

    Formula for LLM models:

        VRAM = WEIGHT_SIZE * 1.2 + RESERVED_FOOTPRINT

    Reference for the 20% overhead: https://blog.eleuther.ai/transformer-math/#total-inference-memory

    For example, using bfloat16,
    - 0.5B requires 3.1 GiB
    - 3B requires 8.9 GiB
    - 7B requires 19.0 GiB
    - 72B requires 164.5 GiB

    """
    env_vram_claim = get_vram_claim_from_model_env(model)
    if env_vram_claim is not None:
        # Use as a potential workaround if the empirical vram estimation is far beyond the expected value.
        return env_vram_claim

    weight_size = 0
    timeout_in_seconds = 15

    try:
        if model.categories and CategoryEnum.IMAGE in model.categories:
            weight_size = await asyncio.wait_for(
                estimate_diffusion_model_vram(model, token, workers, session),
                timeout=timeout_in_seconds,
            )
            return weight_size
        elif (
            model.source == SourceEnum.HUGGING_FACE
            or model.source == SourceEnum.MODEL_SCOPE
        ):
            weight_size = await asyncio.wait_for(
                asyncio.to_thread(get_model_weight_size, model, token),
                timeout=timeout_in_seconds,
            )
        elif model.source == SourceEnum.LOCAL_PATH:
            # Try to get cached size from ModelFile first
            cached_size = await _get_cached_model_size(session, model)
            if cached_size:
                weight_size = cached_size
            else:
                # Fall back to querying workers
                weight_size = await get_local_model_weight_size(
                    model.local_path, workers
                )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout when getting weight size for model {model.name}")
    except Exception as e:
        logger.warning(f"Cannot get weight size for model {model.name}: {e}")

    # Reference: https://blog.eleuther.ai/transformer-math/#total-inference-memory
    activation_overhead_factor = 1.2
    if model.categories and CategoryEnum.TEXT_TO_SPEECH in model.categories:
        # Emperical factor base on Qwen3-TTS
        activation_overhead_factor = 3

    # CUDA graphs can take additional 1~3 GiB memory
    # https://github.com/vllm-project/vllm/blob/v0.6.1/vllm/worker/model_runner.py#L1313
    # For non-LLM models like embedding, set a smaller overhead
    framework_overhead = 2 * 1024**3 if is_llm_model(model) else 512 * 1024**2

    return int(weight_size * activation_overhead_factor) + framework_overhead


async def estimate_diffusion_model_vram(
    model: Model,
    token: Optional[str] = None,
    workers: Optional[List[Worker]] = None,
    session: Optional[AsyncSession] = None,
) -> int:
    """ """
    if model.env and 'GPUSTACK_MODEL_VRAM_CLAIM' in model.env:
        # Use as a potential workaround if the empirical vram estimation is far beyond the expected value.
        return int(model.env['GPUSTACK_MODEL_VRAM_CLAIM'])
    weight_size = 0
    timeout_in_seconds = 15
    try:
        if (
            model.source == SourceEnum.HUGGING_FACE
            or model.source == SourceEnum.MODEL_SCOPE
        ):
            weight_size = await asyncio.wait_for(
                asyncio.to_thread(get_diffusion_model_weight_size, model, token),
                timeout=timeout_in_seconds,
            )
        elif model.source == SourceEnum.LOCAL_PATH:
            # Try to get cached size from ModelFile first
            cached_size = await _get_cached_model_size(session, model)
            if cached_size:
                weight_size = cached_size
            else:
                # Fall back to querying workers
                weight_size = await get_local_model_weight_size(
                    model.local_path, workers
                )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout when getting weight size for model {model.name}")
    except Exception as e:
        logger.warning(f"Cannot get weight size for model {model.name}: {e}")

    return weight_size


def get_worker_model_instances(
    all_model_instances: List[ModelInstance], worker: Worker
) -> List[ModelInstance]:
    """
    Get all model instances related to the worker, including:
    1. Model instances assigned to this worker (main worker)
    2. Model instances that use this worker as a subordinate worker in distributed inference
    """
    # Filter to get only the relevant instances:
    # 1. Instances assigned to this worker (main worker)
    # 2. Instances that use this worker as a subordinate worker
    relevant_instances = []
    for model_instance in all_model_instances:
        # Check if this is a main worker instance
        if model_instance.worker_id == worker.id:
            relevant_instances.append(model_instance)
        # Check if this worker is used as a subordinate worker
        elif (
            model_instance.distributed_servers
            and model_instance.distributed_servers.subordinate_workers
        ):
            for (
                subordinate_worker
            ) in model_instance.distributed_servers.subordinate_workers:
                if subordinate_worker.worker_id == worker.id:
                    relevant_instances.append(model_instance)
                    break

    return relevant_instances


class ListMessageBuilder:
    def __init__(self, messages: Optional[str | List[str]]):
        if not messages:
            self._messages = []
        self._messages = messages if isinstance(messages, list) else [messages]

    def append(self, message: str):
        self._messages.append(message)

    def extend(self, message: List[str]):
        self._messages.extend(message)

    def __str__(self) -> str:
        return "\n".join([f"- {line}" for line in self._messages])


def get_model_num_attention_heads(pretrained_config) -> Optional[int]:
    """
    Get the number of attention heads in the model.
    Priority: llm_config > text_config > root-level setting > thinker_config.text_config
    """

    num_attention_heads = None
    try:
        # Helper to get num_attention_heads from config
        def get_heads_from(cfg, key="num_attention_heads"):
            value = getattr(cfg, key, None)
            if isinstance(value, int) and value > 0:
                return value
            return None

        thinker_config = getattr(pretrained_config, "thinker_config", None)
        thinker_text_config = (
            getattr(thinker_config, "text_config", None) if thinker_config else None
        )

        configs_by_priority = [
            getattr(pretrained_config, "llm_config", None),
            getattr(pretrained_config, "text_config", None),
            pretrained_config,
            thinker_text_config,
        ]

        for config in configs_by_priority:
            if not config:
                continue
            heads = get_heads_from(config)
            if heads is not None:
                num_attention_heads = heads
                break

    except Exception as e:
        logger.warning(f"Cannot get num_attention_heads: {e}")

    return num_attention_heads


async def get_local_model_weight_size(
    local_path: str, workers: Optional[List[Worker]] = None
) -> int:
    """
    Get the local model weight size in bytes. Estimate by the total size of files in the top-level (depth 1) of the directory.

    If the model exists locally (on the server), calculate it locally.
    Otherwise, if workers are provided, check if the model exists on any worker and get the size from there.

    Args:
        local_path: Path to the model directory
        workers: Optional list of workers to check

    Returns:
        Total size in bytes
    """
    weight_file_extensions = (".safetensors", ".bin", ".pt", ".pth")

    if os.path.exists(local_path):
        if not os.path.isdir(local_path):
            raise NotADirectoryError(
                f"The specified path '{local_path}' is not a directory."
            )

        total_size = 0
        try:
            with os.scandir(local_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(weight_file_extensions):
                        total_size += entry.stat().st_size
            return total_size
        except PermissionError:
            raise PermissionError(f"Permission denied when accessing '{local_path}'.")
        except Exception as e:
            logger.error(f"Failed to calculate size locally for {local_path}: {e}")
            raise e
    if not workers:
        raise FileNotFoundError(f"The specified path '{local_path}' does not exist.")

    async def try_get_size_from_worker(worker: Worker) -> Optional[int]:
        """Try to get model weight size from a single worker."""
        try:
            async with WorkerFilesystemClient() as fs_client:
                size = await fs_client.get_model_weight_size(worker, local_path)
                if isinstance(size, int):
                    logger.info(
                        f"Successfully got model weight size from worker {worker.id}: {size} bytes"
                    )
                    return size
                return None
        except Exception as e:
            logger.debug(
                f"Failed to get model weight size from worker {worker.id}: {e}"
            )
            return None

    # Concurrently try all workers and return the first successful result
    logger.info(f"Broadcasting model weight size request to {len(workers)} workers")
    tasks = [try_get_size_from_worker(worker) for worker in workers]

    # Use as_completed to get results as they finish
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        if result is not None:
            return result


def group_worker_gpu_by_memory(
    workers: List[Worker],
    model_instances: List[ModelInstance],
    ram_claim: int = 0,
    gpu_type: Optional[str] = None,
) -> List[List[WorkerGPUInfo]]:
    """
    Group GPU devices from multiple workers by allocatable memory size with the constraint
    that the minimum allocatable GPU memory in each group should not be less than 0.9 times
    the allocatable memory of other GPUs in the same group.

    Args:
        engine: Database engine for calculating allocatable resources
        workers: List of workers containing GPU devices
        ram_claim: RAM claim in bytes to filter out workers that do not have enough RAM

    Returns:
        List of GPU device groups, where each group is a list of WorkerGPUInfo objects
        containing worker information and GPU device details

    Example:
        If we have GPUs from different workers with allocatable memory [8GB, 8.5GB, 16GB, 16.5GB, 32GB],
        they might be grouped as:
        - Group 1: [WorkerGPUInfo(worker1, gpu1, 8GB), WorkerGPUInfo(worker2, gpu2, 8.5GB)]
        - Group 2: [WorkerGPUInfo(worker1, gpu3, 16GB), WorkerGPUInfo(worker3, gpu1, 16.5GB)]
        - Group 3: [WorkerGPUInfo(worker2, gpu4, 32GB)]
    """
    if not workers:
        return []

    # Collect all GPU devices with their worker information and allocatable VRAM
    worker_gpu_infos = []

    for worker in workers:
        if not worker.status or not worker.status.gpu_devices:
            continue

        # Get allocatable resources for this worker
        allocatable = get_worker_allocatable_resource(model_instances, worker, gpu_type)

        if ram_not_enough(ram_claim, allocatable):
            continue

        for gpu_device in worker.status.gpu_devices:
            if gpu_device.index is None:
                continue
            if not gpu_device.memory or gpu_device.memory.total == 0:
                continue

            # Get allocatable VRAM for this specific GPU
            gpu_index = gpu_device.index
            allocatable_vram = allocatable.vram.get(gpu_index, 0)

            # Only include GPUs with positive allocatable VRAM
            if allocatable_vram > 0:
                worker_gpu_info = WorkerGPUInfo(
                    worker_id=worker.id,
                    worker_name=worker.name,
                    gpu_device=gpu_device,
                    allocatable_vram=allocatable_vram,
                )
                worker_gpu_infos.append(worker_gpu_info)

    if not worker_gpu_infos:
        return []

    # Sort GPUs by allocatable VRAM size (ascending order)
    sorted_worker_gpu_infos = sorted(
        worker_gpu_infos, key=lambda info: info.allocatable_vram
    )

    groups = []
    current_group = []

    for worker_gpu_info in sorted_worker_gpu_infos:
        if not current_group:
            # Start a new group
            current_group = [worker_gpu_info]
        else:
            # Check if this GPU can be added to the current group
            min_allocatable_vram = current_group[
                0
            ].allocatable_vram  # First GPU has minimum allocatable VRAM
            current_allocatable_vram = worker_gpu_info.allocatable_vram

            # Check if min_allocatable_vram >= current_allocatable_vram * 0.9
            # This ensures the minimum allocatable VRAM is not less than 0.9 times any other allocatable VRAM in the group
            if min_allocatable_vram >= current_allocatable_vram * 0.9:
                current_group.append(worker_gpu_info)
            else:
                # Cannot add to current group, start a new group
                groups.append(current_group)
                current_group = [worker_gpu_info]

    # Add the last group
    if current_group:
        groups.append(current_group)

    return groups


def ram_not_enough(ram_claim: int, allocatable: Allocatable) -> bool:
    """
    Check if the allocatable RAM is not enough for the claimed RAM.
    """
    if ram_claim <= 0:
        return False
    return allocatable.ram < ram_claim


def get_model_ram_claim(model: Model) -> int:
    """
    Get the RAM requirement for the model in bytes.
    """
    extended_kv_cache = model.extended_kv_cache
    if (
        extended_kv_cache
        and extended_kv_cache.enabled
        and extended_kv_cache.ram_size
        and extended_kv_cache.ram_size > 0
    ):
        # When extended kv cache is enabled, reserve the ram for KV cache.
        return extended_kv_cache.ram_size * 1024**3
    return 0


def get_computed_ram_claim(
    model: Model, vram_claim: Dict[int, int], static_ram: Optional[int] = None
) -> Optional[int]:
    """
    Get the computed RAM claim for the model based on the provided model and vram_claim.
    The priority is as follows:
    1. If static_ram is provided, use it.
    2. If RAM size for extended KV cache is available, use it.
    3. If RAM ratio for extended KV cache is set and vram_claim is available, calculate RAM as ram_ratio * total_vram_claim.
    4. If neither is available, return None.
    """
    if static_ram:
        return static_ram

    ext = model.extended_kv_cache
    if not ext or not ext.enabled:
        return None

    claim = None
    # static ram size
    if ext.ram_size and ext.ram_size > 0:
        claim = ext.ram_size * 1024**3

    # ram ratio to vram
    elif ext.ram_ratio and ext.ram_ratio > 0 and vram_claim:
        total_vram_claim = sum(vram_claim.values())
        claim = int(total_vram_claim * ext.ram_ratio)

    return claim


def sort_workers_by_gpu_count(workers: List[Worker]):
    """
    Sort workers by the number of GPUs.
    """
    workers.sort(
        key=lambda worker: (
            len(worker.status.gpu_devices)
            if worker.status and worker.status.gpu_devices
            else 0
        ),
        reverse=True,
    )


def sort_gpu_indexes_by_allocatable_rate(
    worker: Worker, allocatable: dict, gpu_type: Optional[str] = None
) -> List[int]:
    """
    Sort GPU indexes of a worker by allocatable VRAM rate (allocatable_vram / total_vram), ascending.
    """
    allocatable_rate = {
        gpu.index: (
            allocatable.get(gpu.index, 0) / gpu.memory.total
            if gpu.memory
            and gpu.memory.total
            and (gpu_type is None or gpu.type == gpu_type)
            else 0
        )
        for gpu in worker.status.gpu_devices
    }

    # return a list of gpu indexes sorted by allocatable rate in ascending order
    return sorted(allocatable_rate, key=lambda idx: allocatable_rate[idx], reverse=True)


def sort_selected_workers_by_gpu_type_and_resource(
    workers: List[Worker],
    selected_gpu_indexes_by_gpu_type_and_worker: Dict[str, Dict[str, List[int]]],
    get_worker_allocatable_resource: Callable[[Worker, Optional[str]], Allocatable],
) -> Dict[str, List[Worker]]:
    """
    Filter and sort selected workers by their GPU resource availability.

    This function selects workers whose names are in `selected_gpu_workers`,
    then sorts their GPU devices by allocatable VRAM rate (ascending),
    and finally sorts the workers by the number of GPUs (descending).
    """
    selected_workers_by_gpu_type = {
        gpu_type: [] for gpu_type in selected_gpu_indexes_by_gpu_type_and_worker.keys()
    }
    selected_gpu_types = list(selected_gpu_indexes_by_gpu_type_and_worker.keys())
    for gpu_type in selected_gpu_types:
        for worker in workers:
            # Skip invalid
            selected_gpu_indexes = selected_gpu_indexes_by_gpu_type_and_worker.get(
                gpu_type, {}
            ).get(worker.name)
            if (
                not worker.status
                or not worker.status.gpu_devices
                or not selected_gpu_indexes
            ):
                continue

            # Sort selected GPUs by allocatable rate
            allocatable = get_worker_allocatable_resource(worker, gpu_type)
            sorted_gpu_indexes = [
                idx
                for idx in sort_gpu_indexes_by_allocatable_rate(
                    worker, allocatable.vram
                )
                if idx in selected_gpu_indexes
            ]
            sorted_gpus = [
                gpu
                for idx in sorted_gpu_indexes
                for gpu in worker.status.gpu_devices
                if gpu.index == idx and (gpu_type is None or gpu.type == gpu_type)
            ]

            # Create a copy of the worker with sorted GPUs
            w = worker.model_copy(deep=True)
            w.status.gpu_devices = sorted_gpus
            selected_workers_by_gpu_type[gpu_type].append(w)

    # Sort workers by GPU count
    for gpu_type in selected_workers_by_gpu_type:
        sort_workers_by_gpu_count(selected_workers_by_gpu_type[gpu_type])
    return selected_workers_by_gpu_type
