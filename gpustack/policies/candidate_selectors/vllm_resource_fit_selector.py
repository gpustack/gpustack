import asyncio
import logging
import os
import re
from typing import Dict, List

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
)
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
    SourceEnum,
)
from gpustack.schemas.workers import GPUDevicesInfo, Worker
from huggingface_hub import HfApi

from gpustack.server.db import get_engine
from gpustack.utils.command import find_parameter

logger = logging.getLogger(__name__)


async def estimate_model_vram(model: Model) -> int:
    """
    Estimate the vram requirement in bytes heuristically.
    This is the minimum requirement to help us decide how many GPUs are needed for the model.
    If users explicitly set parameters like tp & pp, this estimation is not needed.

    Formula:

        VRAM = WEIGHT * 1.2 + RESERVERD_FOOTPRINT

    Reference for the 20% overhead: https://blog.eleuther.ai/transformer-math/#total-inference-memory

    For example, using bfloat16,
    - 0.5B requires 3.1 GiB
    - 3B requires 8.9 GiB
    - 7B requires 19.0 GiB
    - 72B requires 164.5 GiB

    """
    # CUDA graphs can take additional 1~3 GiB memory
    # https://github.com/vllm-project/vllm/blob/v0.6.1/vllm/worker/model_runner.py#L1313
    cuda_graph_size = 2 * 1024**3
    weight_size = 0
    timeout_in_seconds = 15

    try:
        if model.source == SourceEnum.HUGGING_FACE:
            weight_size = await asyncio.wait_for(
                asyncio.to_thread(get_hf_model_weight_size, model.huggingface_repo_id),
                timeout=timeout_in_seconds,
            )
        elif model.source == SourceEnum.MODEL_SCOPE:
            trust_remote_code = False
            if (
                model.backend_parameters
                and "--trust-remote-code" in model.backend_parameters
            ):
                trust_remote_code = True
            weight_size = await asyncio.wait_for(
                asyncio.to_thread(
                    get_ms_model_weight_size,
                    model.model_scope_model_id,
                    trust_remote_code=trust_remote_code,
                ),
                timeout=timeout_in_seconds,
            )
        elif model.source == SourceEnum.LOCAL_PATH and os.path.exists(model.local_path):
            weight_size = get_local_model_weight_size(model.local_path)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout when getting weight size for model {model.name}")
    except Exception as e:
        logger.warning(f"Cannot get weight size for model {model.name}: {e}")

    return weight_size * 1.2 + cuda_graph_size


def get_hf_model_weight_size(repo_id: str) -> int:
    """
    Get the model weight size in bytes from the hugging face model_info API.
    """
    api = HfApi()
    model_info = api.model_info(repo_id)
    safetensors_info = model_info.safetensors
    if not safetensors_info or not safetensors_info.parameters:
        raise ValueError("No safetensors information found.")

    d_type, total_params = next(iter(model_info.safetensors.parameters.items()))

    # https://github.com/huggingface/huggingface_hub/blob/v0.25.1/src/huggingface_hub/utils/_safetensors.py#L10
    dtype_to_bytes = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }

    bytes_per_param = dtype_to_bytes.get(d_type, 2)
    total_weight_size = total_params * bytes_per_param
    return int(total_weight_size)


def get_ms_model_weight_size(model_id: str, trust_remote_code: bool = False) -> int:
    """
    Get the modelscope model weight size in bytes.
    """
    from modelscope import AutoConfig

    # ModelScope does not provide the info in the API. Infer from the model name.
    total_params = parse_model_size_by_name(model_id)

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    torch_dtype = getattr(config, 'torch_dtype', "float16")
    dtype_to_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
    }
    bytes_per_param = dtype_to_bytes.get(torch_dtype, 2)
    total_weight_size = total_params * bytes_per_param
    return int(total_weight_size)


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


def get_local_model_weight_size(local_path: str) -> int:
    """
    Get the local model weight size in bytes. Estimate by the total size of files in the top-level (depth 1) of the directory.
    """
    total_size = 0

    try:
        with os.scandir(local_path) as entries:
            for entry in entries:
                if entry.is_file():
                    total_size += entry.stat().st_size
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified path '{local_path}' does not exist.")
    except NotADirectoryError:
        raise NotADirectoryError(
            f"The specified path '{local_path}' is not a directory."
        )
    except PermissionError:
        raise PermissionError(f"Permission denied when accessing '{local_path}'.")

    return total_size


class VLLMResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        model: Model,
        model_instance: ModelInstance,
    ):
        self._engine = get_engine()
        self._model = model
        self._model_instance = model_instance
        self._gpu_count = None

        # When tp/pp is set, the gpu count is calculated by tp * pp.
        # Pick the candidate with satisfied gpu count.
        # Otherwise, estimate gpu count by vram requirement heuristically.
        tp = find_parameter(model.backend_parameters, ["tensor-parallel-size", "tp"])
        pp = find_parameter(model.backend_parameters, ["pipeline-parallel-size", "pp"])
        if tp:
            self._gpu_count = int(tp)
            self._vram_claim = 0
            if pp:
                self._gpu_count *= int(pp)

        self._gpu_memory_utilization = 0.9
        gmu = find_parameter(model.backend_parameters, ["gpu-memory-utilization"])
        if gmu:
            self._gpu_memory_utilization = float(gmu)

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement.
        """

        if not self._gpu_count:
            self._vram_claim = await estimate_model_vram(self._model)
            logger.info(
                f"Calculated resource claim for model instance {self._model_instance.name}, "
                f"claim: {self._vram_claim}"
            )

        candidate_functions = [
            self.find_single_worker_single_gpu_full_offloading_candidates,
            self.find_single_worker_multi_gpu_full_offloading_candidates,
        ]

        for candidate_func in candidate_functions:
            logger.debug(
                f"model {self._model.name}, filter candidates with resource fit selector: {candidate_func.__name__}, instance {self._model_instance.name}"
            )

            candidates = await candidate_func(workers)
            if candidates:
                return candidates

        return []

    async def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with workers.
        """
        if self._gpu_count is not None and self._gpu_count > 1:
            return []

        candidates = []
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            result = (
                await self._find_single_worker_single_gpu_full_offloading_candidates(
                    worker
                )
            )
            if result:
                candidates.extend(result)

        return candidates

    async def _find_single_worker_single_gpu_full_offloading_candidates(
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with worker.
        requires: worker.status.gpu_devices is not None
        """
        candidates = []

        allocatable = await get_worker_allocatable_resource(self._engine, worker)

        if worker.status.gpu_devices:
            for _, gpu in enumerate(worker.status.gpu_devices):
                gpu_index = gpu.index
                allocatable_vram = allocatable.vram.get(gpu_index, 0)

                if gpu.memory is None or gpu.memory.total == 0:
                    continue

                if (self._vram_claim > allocatable_vram) or (
                    allocatable_vram / gpu.memory.total < self._gpu_memory_utilization
                ):
                    continue

                candidates.append(
                    ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_indexes=[gpu_index],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={
                                gpu_index: int(
                                    gpu.memory.total * self._gpu_memory_utilization
                                )
                            },
                        ),
                    )
                )

        return candidates

    async def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        if self._gpu_count == 1:
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

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        sorted_gpu_devices: GPUDevicesInfo = sorted(
            [
                gpu
                for gpu in worker.status.gpu_devices
                if gpu.memory is not None
                and gpu.memory.total is not None
                and (
                    allocatable.vram.get(gpu.index, 0) / gpu.memory.total
                    > self._gpu_memory_utilization
                )
            ],
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
            vram_claim[gpu.index] = int(gpu.memory.total * self._gpu_memory_utilization)
            gpu_sum += 1
            vram_sum += vram_claim[gpu.index]

            if self._gpu_count and gpu_sum >= self._gpu_count:
                found_candidate = True
                break

            if self._gpu_count is None and vram_sum >= self._vram_claim:
                found_candidate = True
                break

        if found_candidate:
            return [
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=gpu_indexes,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                    ),
                )
            ]

        return []
