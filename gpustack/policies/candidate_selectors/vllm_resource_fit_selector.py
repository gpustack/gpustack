import asyncio
from collections import defaultdict
import logging
import os
import re
from typing import Dict, List, Optional
from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
)
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
    get_worker_model_instances,
)
from gpustack.schemas.models import (
    CategoryEnum,
    ComputedResourceClaim,
    Model,
    RayActor,
    SourceEnum,
)
from gpustack.schemas.workers import GPUDevicesInfo, Worker
from gpustack.config import Config
from gpustack.server.db import get_engine
from gpustack.utils.command import find_parameter
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_id, parse_gpu_ids_by_worker
from gpustack.utils.hub import get_model_weight_size, get_pretrained_config
from gpustack.utils.unit import byte_to_gib

logger = logging.getLogger(__name__)


async def estimate_model_vram(model: Model, token: Optional[str] = None) -> int:
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
    # For non-LLM models like embedding, set a smaller overhead
    framework_overhead = (
        2 * 1024**3
        if not model.categories or CategoryEnum.LLM in model.categories
        else 512 * 1024**2
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
    return weight_size * 1.2 + framework_overhead


def get_model_num_attention_heads(model: Model) -> Optional[int]:
    """
    Get the number of attention heads in the model.
    """

    num_attention_heads = None
    try:
        config = get_pretrained_config(model, trust_remote_code=True)
        num_attention_heads = getattr(config, "num_attention_heads", None)
        if not num_attention_heads:
            llm_config = getattr(config, "llm_config", None)
            if llm_config:
                num_attention_heads = getattr(llm_config, "num_attention_heads", None)
    except Exception as e:
        logger.warning(f"Cannot get num_attention_heads for model {model.name}: {e}")

    return num_attention_heads


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
        cfg: Config,
        model: Model,
    ):
        self._engine = get_engine()
        self._cfg = cfg
        self._model = model
        self._gpu_count = None
        self._vram_claim = 0
        self._largest_single_gpu_vram = 0
        self._largest_single_gpu_vram_utilization = 0
        self._largest_multi_gpu_vram = 0
        self._largest_multi_gpu_total = 0
        self._largest_multi_gpu_utilization_satisfied_count = 0

        self._num_attention_heads = None
        self._messages = []
        self._workers_allocatable_resource: Dict[int, Allocatable] = {}

        self._selected_gpu_workers: List[str] = None
        self._selected_gpu_worker_count = 0
        self._selected_gpu_indexes_by_worker: Dict[str, List[int]] = {}

        if self._model.gpu_selector and self._model.gpu_selector.gpu_ids:
            gpu_ids_by_worker = parse_gpu_ids_by_worker(
                self._model.gpu_selector.gpu_ids
            )
            self._selected_gpu_workers = list(gpu_ids_by_worker.keys())
            self._selected_gpu_worker_count = len(self._selected_gpu_workers)
            for worker_name, gpu_ids in gpu_ids_by_worker.items():
                gpu_indexes = []
                for gpu_id in gpu_ids:
                    valid, matched = parse_gpu_id(gpu_id)
                    if valid:
                        gpu_index = safe_int(matched.get("gpu_index"))
                        gpu_indexes.append(gpu_index)
                self._selected_gpu_indexes_by_worker[worker_name] = gpu_indexes

            # When user defined gpu selector, we use the gpu count from it.
            self._gpu_count = len(self._model.gpu_selector.gpu_ids)

        # When tp/pp is set, the gpu count is calculated by tp * pp.
        # Pick the candidate with satisfied gpu count.
        # Otherwise, estimate gpu count by vram requirement heuristically.
        tp = find_parameter(model.backend_parameters, ["tensor-parallel-size", "tp"])
        pp = find_parameter(model.backend_parameters, ["pipeline-parallel-size", "pp"])
        if tp or pp:
            world_size = int(tp or 1) * int(pp or 1)

            if self._gpu_count and self._gpu_count != world_size:
                # Both gpu selector and tp/pp are set, validate they match.
                raise ValueError(
                    f"Model {model.name} has -tp/-pp set, but the selected gpu count ({self._gpu_count}) does not match the world size ({world_size})."
                )
            else:
                self._gpu_count = world_size
                self._vram_claim = 0

        self._gpu_memory_utilization = 0.9
        if model.categories and CategoryEnum.LLM not in model.categories:
            # gpu memory utilization is not used for non-LLM models
            self._gpu_memory_utilization = 0

        gmu = find_parameter(model.backend_parameters, ["gpu-memory-utilization"])
        if gmu:
            self._gpu_memory_utilization = float(gmu)

        self._num_attention_heads = get_model_num_attention_heads(model)
        if (
            self._gpu_count
            and self._num_attention_heads
            and self._num_attention_heads % self._gpu_count != 0
        ):
            raise ValueError(
                f"Total number of attention heads ({self._num_attention_heads})"
                " must be divisible by gpu count "
                f"({self._gpu_count})."
            )

    def _set_messages(self):
        if self._messages:
            return

        messages = [
            f"The model requires {self._gpu_memory_utilization * 100}%(--gpu-memory-utilization={self._gpu_memory_utilization}) VRAM for each GPU that satisfies {byte_to_gib(self._vram_claim)} GiB VRAM in total."
        ]
        if self._largest_multi_gpu_vram > 0 and self._gpu_memory_utilization > 0:
            messages = [
                f"The model requires {self._gpu_memory_utilization * 100}% (--gpu-memory-utilization={self._gpu_memory_utilization}) VRAM for each GPU, with a total VRAM requirement of {byte_to_gib(self._vram_claim)} GiB VRAM. The largest available worker provides {byte_to_gib(self._largest_multi_gpu_vram)} GiB VRAM, and {self._largest_multi_gpu_utilization_satisfied_count}/{self._largest_multi_gpu_total} of GPUs meet the VRAM utilization ratio."
            ]
        elif self._largest_single_gpu_vram > 0 and self._gpu_memory_utilization > 0:
            messages = [
                f"The model requires {self._gpu_memory_utilization * 100}% (--gpu-memory-utilization={self._gpu_memory_utilization}) VRAM on a GPU with {byte_to_gib(self._vram_claim)} GiB VRAM. The available GPU has {byte_to_gib(self._largest_single_gpu_vram)} GiB VRAM and {self._largest_single_gpu_vram_utilization * 100:.2f}% allocatable VRAM ratio."
            ]
        elif self._largest_multi_gpu_vram > 0 and self._gpu_memory_utilization == 0:
            # Non-LLM models
            messages = [
                f"The model requires a total VRAM requirement of {byte_to_gib(self._vram_claim)} GiB VRAM. The largest available worker provides {byte_to_gib(self._largest_multi_gpu_vram)} GiB VRAM."
            ]
        elif self._largest_single_gpu_vram > 0 and self._gpu_memory_utilization == 0:
            # Non-LLM models
            messages = [
                f"The model requires a GPU with {byte_to_gib(self._vram_claim)} GiB VRAM. The available GPU has {byte_to_gib(self._largest_single_gpu_vram)} GiB VRAM."
            ]
        elif self._gpu_memory_utilization == 0:
            # Non-LLM models
            messages = [
                f"The model requires {byte_to_gib(self._vram_claim)} GiB VRAM in total."
            ]

        if self._cfg.enable_ray and self._model.distributed_inference_across_workers:
            messages.append(
                "Cannot find a suitable worker combination to run the model in distributed mode. If you are confident that the resources are sufficient, you may manually schedule the model by selecting the workers and GPUs."
            )

        self._messages = messages

    def _add_message(self, message: str):
        self._messages.append(message)

    def get_messages(self) -> str:
        return self._messages

    async def _get_worker_allocatable_resource(self, worker: Worker):
        if worker.id in self._workers_allocatable_resource:
            return self._workers_allocatable_resource[worker.id]

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        self._workers_allocatable_resource[worker.id] = allocatable
        return allocatable

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement.
        """

        if not self._gpu_count:
            self._vram_claim = await estimate_model_vram(
                self._model, self._cfg.huggingface_token
            )
            logger.info(
                f"Calculated resource claim for model {self._model.readable_source}, "
                f"claim: {self._vram_claim}"
            )

        candidate_functions = [
            self.find_single_worker_single_gpu_full_offloading_candidates,
            self.find_single_worker_multi_gpu_full_offloading_candidates,
            self.find_multi_worker_multi_gpu_candidates,
        ]

        for candidate_func in candidate_functions:
            logger.debug(
                f"model {self._model.readable_source}, filter candidates with resource fit selector: {candidate_func.__name__}"
            )

            candidates = await candidate_func(workers)
            if candidates:
                return candidates

        self._set_messages()
        return []

    async def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with workers.
        """
        if self._gpu_count is not None and self._gpu_count > 1:
            # Skip multi-GPU selection
            return []

        if self._selected_gpu_worker_count > 1:
            # Skip multi-worker selection
            return []

        selected_gpu_worker = None
        selected_gpu_index = None
        if self._selected_gpu_workers:
            # Handle manual scheduling
            selected_gpu_worker = self._selected_gpu_workers[0]
            selected_gpu_indexes = self._selected_gpu_indexes_by_worker[
                selected_gpu_worker
            ]
            selected_gpu_index = (
                selected_gpu_indexes[0] if selected_gpu_indexes else None
            )

        candidates = []
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            if selected_gpu_worker and worker.name != selected_gpu_worker:
                continue

            result = (
                await self._find_single_worker_single_gpu_full_offloading_candidates(
                    worker, selected_gpu_index
                )
            )
            if result:
                candidates.extend(result)

        return candidates

    async def _find_single_worker_single_gpu_full_offloading_candidates(
        self, worker: Worker, selected_gpu_index: Optional[int]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with worker.
        requires: worker.status.gpu_devices is not None
        """

        candidates = []

        allocatable = await self._get_worker_allocatable_resource(worker)

        if worker.status.gpu_devices:
            for _, gpu in enumerate(worker.status.gpu_devices):

                if selected_gpu_index and gpu.index != selected_gpu_index:
                    continue

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

                if (self._vram_claim > allocatable_vram) or (
                    self._gpu_memory_utilization > 0
                    and allocatable_gpu_memory_utilization
                    < self._gpu_memory_utilization
                ):
                    continue

                vram_claim = (
                    int(gpu.memory.total * self._gpu_memory_utilization)
                    if self._gpu_memory_utilization > 0  # LLMs
                    else int(self._vram_claim)  # non LLMs
                )

                candidates.append(
                    ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_indexes=[gpu_index],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={
                                gpu_index: vram_claim,
                            },
                        ),
                    )
                )

        return candidates

    async def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        if self._gpu_count == 1 or self._selected_gpu_worker_count > 1:
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
            self._largest_multi_gpu_total = total_allocatable_vram
            self._largest_multi_gpu_utilization_satisfied_count = satisfied_gpu_count
            self._largest_multi_gpu_total = len(worker.status.gpu_devices)

        # Sort by vram in descending order
        sorted_gpu_devices: GPUDevicesInfo = sorted(
            gpu_list,
            key=lambda gpu: allocatable.vram.get(gpu.index, 0),
            reverse=True,
        )

        if self._selected_gpu_workers and len(self._selected_gpu_workers) == 1:
            selected_gpu_worker = self._selected_gpu_workers[0]
            selected_gpu_indexes = self._selected_gpu_indexes_by_worker[
                selected_gpu_worker
            ]
            if worker.name != selected_gpu_worker:
                return []

            vram_claim = {}
            for gpu_index in selected_gpu_indexes:
                vram_claim[gpu_index] = (
                    int(
                        allocatable.vram.get(gpu_index, 0)
                        * self._gpu_memory_utilization
                    )
                    if self._gpu_memory_utilization > 0  # LLMs
                    else int(self._vram_claim / len(selected_gpu_indexes))  # non LLMs
                )

            if sum(vram_claim.values()) < self._vram_claim:
                return []

            return [
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=selected_gpu_indexes,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                    ),
                )
            ]

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

            if self._num_attention_heads and self._num_attention_heads % gpu_sum != 0:
                continue

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

    async def find_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        if not self._cfg.enable_ray:
            return []

        if self._selected_gpu_workers:
            return await self.manual_select_multi_worker_multi_gpu_candidates(workers)

        return await self.auto_select_multi_worker_multi_gpu_candidates(workers)

    async def auto_select_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Auto select multi worker multi gpu candidates.
        Currently, a candidate should match the following conditions:
        1. Workers in the candidate have the same number of GPUs.
        2. All GPUs in the worker satisfy the gpu_memory_utilization requirement.
        3. The total number of GPUs can be divided by the number of attention heads.
        4. The total VRAM claim is greater than the estimated VRAM claim.
        """

        if not workers:
            return []

        sort_workers_by_gpu_count(workers)

        workers_by_gpu_count_dict = defaultdict(list)
        for worker in workers:
            if not worker.status or not worker.status.gpu_devices:
                continue

            workers_by_gpu_count_dict[len(worker.status.gpu_devices)].append(worker)

        # Loop through worker groups with the same number of GPUs.
        for gpu_count, worker_group in workers_by_gpu_count_dict.items():
            if len(worker_group) < 2:
                continue

            selected_workers: List[Worker] = []
            gpu_sum = 0
            vram_sum = 0
            for worker in worker_group:
                allocatable = await self._get_worker_allocatable_resource(worker)
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
                    int(gpu.memory.total * self._gpu_memory_utilization)
                    for gpu in worker.status.gpu_devices
                )

                if (
                    self._num_attention_heads
                    and self._num_attention_heads % gpu_sum == 0
                ) and (vram_sum >= self._vram_claim):
                    return [
                        _create_candidate(
                            selected_workers, self._gpu_memory_utilization
                        )
                    ]

        return []

    async def manual_select_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get manual selected multi worker multi gpu candidates.
        """
        if not self._selected_gpu_workers or len(self._selected_gpu_workers) < 2:
            return []

        if not workers:
            return []

        sort_workers_by_gpu_count(workers)

        main_worker = workers[0]
        main_worker_name = main_worker.name
        main_gpu_indexes = self._selected_gpu_indexes_by_worker[main_worker_name]
        main_vram_claim = await self._get_worker_vram_claim(
            main_worker, main_gpu_indexes, self._gpu_memory_utilization
        )

        ray_actors: List[RayActor] = []
        for worker in workers:
            if worker.name not in self._selected_gpu_workers:
                continue

            if not await self._validate_distributed_vllm_limit_per_worker(worker):
                return []

            if worker.name == main_worker_name:
                continue

            gpu_indexes = self._selected_gpu_indexes_by_worker[worker.name]
            vram_claim = await self._get_worker_vram_claim(
                worker, gpu_indexes, self._gpu_memory_utilization
            )
            ray_actors.append(
                RayActor(
                    worker_id=worker.id,
                    worker_ip=worker.ip,
                    total_gpus=len(worker.status.gpu_devices),
                    gpu_indexes=gpu_indexes,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                    ),
                )
            )

        return [
            ModelInstanceScheduleCandidate(
                worker=main_worker,
                gpu_indexes=main_gpu_indexes,
                computed_resource_claim=ComputedResourceClaim(
                    vram=main_vram_claim,
                ),
                ray_actors=ray_actors,
            )
        ]

    async def _validate_distributed_vllm_limit_per_worker(self, worker: Worker) -> bool:
        """
        Validate that there is no more than one distributed vLLM instance per worker.
        """
        instances = await get_worker_model_instances(self._engine, worker)
        for instance in instances:
            if instance.distributed_servers and instance.distributed_servers.ray_actors:
                self._messages = [
                    f"Each worker can run only one distributed vLLM instance. Worker '{worker.name}' already has '{instance.name}'."
                ]
                return False

        return True

    async def _get_worker_vram_claim(
        self,
        worker: Worker,
        gpu_indexes: List[int],
        gpu_memory_utilization: float = 0.9,
    ) -> Dict[int, int]:
        """
        Given a worker and gpu indexes, get the vram claim according to gpu_memory_utilization.
        Returns a dictionary of gpu index to vram claim in bytes.
        """
        vram_claim: Dict[int, int] = {}

        allocatable = await self._get_worker_allocatable_resource(worker)
        for gpu in worker.status.gpu_devices:
            if gpu.index in gpu_indexes:
                gpu_vram_claim = int(gpu.memory.total * gpu_memory_utilization)
                allocatable_vram = allocatable.vram.get(gpu.index, 0)
                if gpu_vram_claim > allocatable_vram:
                    # Allocatable seems to be smaller than the expected.
                    # We claim the maximum allocatable vram and proceed.
                    gpu_vram_claim = allocatable_vram

                vram_claim[gpu.index] = gpu_vram_claim

        return vram_claim


def _create_candidate(
    selected_workers: List[Worker], gpu_memory_utilization: float = 0.9
) -> ModelInstanceScheduleCandidate:
    """
    Create a candidate with all GPUs from the selected workers.
    """
    main_worker = selected_workers[0]
    candidate = ModelInstanceScheduleCandidate(
        worker=main_worker,
        gpu_indexes=[gpu.index for gpu in main_worker.status.gpu_devices],
        computed_resource_claim=ComputedResourceClaim(
            vram={
                gpu.index: int(gpu.memory.total * gpu_memory_utilization)
                for gpu in main_worker.status.gpu_devices
            },
        ),
    )
    candidate.ray_actors = [
        RayActor(
            worker_id=worker.id,
            worker_ip=worker.ip,
            total_gpus=len(worker.status.gpu_devices),
            gpu_indexes=[gpu.index for gpu in worker.status.gpu_devices],
            computed_resource_claim=ComputedResourceClaim(
                vram={
                    gpu.index: int(gpu.memory.total * gpu_memory_utilization)
                    for gpu in worker.status.gpu_devices
                },
            ),
        )
        for worker in selected_workers[1:]
    ]
    return candidate


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
