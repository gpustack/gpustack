import itertools
import logging
from typing import Dict, List, Optional, Tuple

from gpustack.policies.utils import get_worker_allocatable_resource
from gpustack.scheduler.calculator import (
    GPUOffloadEnum,
    calculate_model_resource_claim,
    memoryEstimate,
)
from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
    ModelInstanceRPCServer,
    is_image_model,
)
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_engine
from gpustack.utils.command import find_parameter
from gpustack.utils.gpu import parse_gpu_id, parse_gpu_ids_by_worker


logger = logging.getLogger(__name__)

MAX_RPC_SERVER_COUNT = 3


class GGUFResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        model: Model,
        model_instance: ModelInstance,
        cache_dir: Optional[str] = None,
    ):
        self._engine = get_engine()
        self._model = model
        self._model_instance = model_instance
        self._cache_dir = cache_dir
        self._max_rpc_server_count = MAX_RPC_SERVER_COUNT

        self._param_tensor_split = find_parameter(
            model.backend_parameters, ["ts", "tensor-split"]
        )

        self._selected_gpu_ids_by_worker = {}
        self._selected_gpu_ids = []
        if self._model.gpu_selector and self._model.gpu_selector.gpu_ids:
            self._selected_gpu_ids_by_worker = parse_gpu_ids_by_worker(
                self._model.gpu_selector.gpu_ids
            )
            self._selected_gpu_ids = sorted(self._model.gpu_selector.gpu_ids)
            self._max_rpc_server_count = len(self._selected_gpu_ids)

        if self._param_tensor_split:
            # ignore the gpu_selector if tensor split is set.
            logger.info(f"Model {model.name} has tensor-split, ignore the gpu_selector")
            self._selected_gpu_ids_by_worker = {}
            self._selected_gpu_ids = []

    def _has_distributed_params(self):
        return self._param_tensor_split

    def _get_gpu_layers(self) -> Optional[str]:
        return find_parameter(
            self._model.backend_parameters, ["ngl", "gpu-layers", "n-gpu-layers"]
        )

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates by the resource fit claim.
        """
        candidates = await self._filter_in_sequence(workers)
        return candidates

    async def _filter_in_sequence(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Filter the workers with the full offloading claim.
        """

        candidate_functions = [
            self.find_single_worker_single_gpu_full_offloading_candidates,
            self.find_single_worker_multi_gpu_full_offloading_candidates,
            self.find_multi_worker_multi_gpu_candidates,
            self.find_single_worker_partial_offloading_candidates,
            self.find_single_worker_cpu_candidates,
        ]

        for candidate_func in candidate_functions:
            if self._should_skip_candidate_func(candidate_func):
                continue

            logger.debug(
                f"model {self._model.name}, filter candidates with resource fit selector: "
                f"{candidate_func.__name__}, instance {self._model_instance.name}",
            )

            candidates = await candidate_func(workers)
            if candidates:
                return candidates

        return []

    def _should_skip_candidate_func(self, candidate_func) -> bool:
        # Skip conditions for CPU offloading.
        if not self._model.cpu_offloading and candidate_func in [
            self.find_single_worker_partial_offloading_candidates,
            self.find_single_worker_cpu_candidates,
        ]:
            return True

        if (
            self._get_gpu_layers() == "0"
            and candidate_func != self.find_single_worker_cpu_candidates
        ):
            # User specified full CPU offloading.
            return True

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

        return False

    async def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with workers.
        """
        if self._has_distributed_params():
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

        is_unified_memory = worker.status.memory.is_unified_memory
        allocatable = await get_worker_allocatable_resource(self._engine, worker)

        result = await calculate_model_resource_claim(
            self._model_instance,
            self._model,
            GPUOffloadEnum.Full,
            cache_dir=self._cache_dir,
        )
        estimate = result.resource_claim_estimate
        total_layers = estimate.items[0].offloadLayers

        for gpu_index in allocatable.vram:
            vram_claim = estimate.items[0].vrams[0].nonuma
            ram_claim = estimate.items[0].ram.nonuma

            if is_unified_memory:
                vram_claim = estimate.items[0].vrams[0].uma
                ram_claim = estimate.items[0].ram.uma

                # For UMA, we need to remove the claim of gpu memory before check the memory.
                if (vram_claim > allocatable.vram[gpu_index]) or (
                    ram_claim > allocatable.ram - vram_claim
                ):
                    continue
            else:
                if (vram_claim > allocatable.vram[gpu_index]) or (
                    ram_claim > allocatable.ram
                ):
                    continue

            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=[gpu_index],
                    computed_resource_claim=ComputedResourceClaim(
                        is_unified_memory=is_unified_memory,
                        offload_layers=total_layers,
                        vram={gpu_index: vram_claim},
                        ram=ram_claim,
                        total_layers=total_layers,
                    ),
                )
            )
        return candidates

    async def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
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

        is_unified_memory = worker.status.memory.is_unified_memory
        total_gpu = len(worker.status.gpu_devices)
        if total_gpu < 2:
            return None

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        gpu_combinations = await self._generate_combinations_for_worker_gpu(
            allocatable, worker
        )

        if self._param_tensor_split:
            # use specified tensor split when the param is set.
            if total_gpu < len(self._param_tensor_split.split(",")):
                return None
            gpu_combinations = await self._generate_combinations_given_tensor_split()

        candidates = []
        for gpu_count in gpu_combinations:
            for gpu_combination in gpu_combinations[gpu_count]:
                tensor_splitting = [value[-1] for value in gpu_combination]

                result = await calculate_model_resource_claim(
                    self._model_instance,
                    self._model,
                    GPUOffloadEnum.Full,
                    cache_dir=self._cache_dir,
                    tensor_split=tensor_splitting,
                )
                estimate = result.resource_claim_estimate
                total_layers = estimate.items[0].offloadLayers

                # ram
                ram_claim = estimate.items[0].ram.nonuma
                if is_unified_memory:
                    ram_claim = estimate.items[0].ram.uma

                if ram_claim > allocatable.ram:
                    continue

                # vram
                vram_claim_matched = True
                vram_claim = {}
                for gci in range(len(gpu_combination)):
                    estimate_gpu_index = gci
                    real_gpu_index = gpu_combination[gci][0]
                    gpu_allocatable = allocatable.vram[real_gpu_index]

                    single_gpu_vram_claim = (
                        estimate.items[0].vrams[estimate_gpu_index].nonuma
                    )
                    if is_unified_memory:
                        single_gpu_vram_claim = (
                            estimate.items[0].vrams[estimate_gpu_index].uma
                        )

                    if single_gpu_vram_claim > gpu_allocatable:
                        vram_claim_matched = False
                        break

                    vram_claim[real_gpu_index] = single_gpu_vram_claim

                if not vram_claim_matched:
                    # stop to check other combinations have the same gpu count.
                    break

                gpu_indexes = [value[0] for value in gpu_combination]
                candidates.append(
                    ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_indexes=gpu_indexes,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=is_unified_memory,
                            offload_layers=total_layers,
                            vram=vram_claim,
                            ram=ram_claim,
                            total_layers=total_layers,
                        ),
                    )
                )

            if candidates:
                break

        return candidates

    async def find_single_worker_partial_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu partial offloading candidates for the model instance.
        """
        single_gpu_partial_offloading_candidates = []
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            result = (
                await self._find_single_worker_single_gpu_partial_offloading_candidates(
                    worker
                )
            )
            if result:
                single_gpu_partial_offloading_candidates.extend(result)

        single_gpu_partial_max_offload_layers = _get_max_offload_layers(
            single_gpu_partial_offloading_candidates
        )

        multi_gpu_partial_offloading_candidates = []
        for worker in workers:
            result = (
                await self._find_single_worker_multi_gpu_partial_offloading_candidates(
                    worker
                )
            )
            if result:
                multi_gpu_partial_offloading_candidates.extend(result)

        multi_gpu_partial_max_offload_layers = _get_max_offload_layers(
            multi_gpu_partial_offloading_candidates
        )

        final_candidates = []
        if (
            single_gpu_partial_max_offload_layers
            >= multi_gpu_partial_max_offload_layers
        ):
            final_candidates = _filter_candidates_by_max_offload_layers(
                single_gpu_partial_offloading_candidates,
                single_gpu_partial_max_offload_layers,
            )
        else:
            final_candidates = _filter_candidates_by_max_offload_layers(
                multi_gpu_partial_offloading_candidates,
                multi_gpu_partial_max_offload_layers,
            )
        return final_candidates

    async def _find_single_worker_single_gpu_partial_offloading_candidates(  # noqa: C901
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu partial offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """
        if self._has_distributed_params():
            return []

        if self._selected_gpu_ids_by_worker:
            if worker.name not in self._selected_gpu_ids_by_worker:
                return []
            elif len(self._selected_gpu_ids_by_worker.get(worker.name)) > 1:
                return []

        candidates = []

        is_unified_memory = worker.status.memory.is_unified_memory
        allocatable = await get_worker_allocatable_resource(self._engine, worker)

        result = await calculate_model_resource_claim(
            self._model_instance,
            self._model,
            GPUOffloadEnum.Partial,
            cache_dir=self._cache_dir,
        )
        estimate = result.resource_claim_estimate
        total_layers = estimate.items[-1].offloadLayers

        arr = []
        estimate_arr = []
        for memory in estimate.items:
            if memory.fullOffloaded:
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

        for gpu_index in allocatable.vram:
            if self._selected_gpu_ids:
                valid, matched = parse_gpu_id(self._selected_gpu_ids[0])
                is_selected_gpu = valid and matched.get("gpu_index") == str(gpu_index)
                if not is_selected_gpu:
                    continue

            index = binary_search(arr, allocatable.vram[gpu_index])
            if index == -1:
                continue

            if (
                is_unified_memory
                # For UMA, we need to remove the claim of gpu memory before check if the memory.
                and (estimate_arr[index]["ram"] > allocatable.ram - arr[index])
                or (estimate_arr[index]["ram"] > allocatable.ram)
            ):
                continue

            offload_layers = estimate_arr[index]["offload_layers"]
            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=[gpu_index],
                    computed_resource_claim=ComputedResourceClaim(
                        is_unified_memory=is_unified_memory,
                        offload_layers=offload_layers,
                        vram={gpu_index: estimate_arr[index]["vram"]},
                        ram=estimate_arr[index]["ram"],
                        total_layers=total_layers,
                    ),
                )
            )
        return candidates

    async def _find_single_worker_multi_gpu_partial_offloading_candidates_with_combination(  # noqa: C901
        self, worker: Worker, gpu_combination: Tuple[Tuple[int]]
    ) -> ModelInstanceScheduleCandidate:
        """
        Find max offload layers for gpu combination.

        Args:
            worker (Worker): The worker instance containing GPU information.
            gpu_combination (List[Tuple[int]]): A list of tuples, each containing GPU index and it's vram (e.g., [(0, 106), (1, 98)])
        """

        gpu_count = len(gpu_combination)
        is_unified_memory = worker.status.memory.is_unified_memory
        allocatable = await get_worker_allocatable_resource(self._engine, worker)

        tensor_splitting = [value[-1] for value in gpu_combination]
        result = await calculate_model_resource_claim(
            self._model_instance,
            self._model,
            GPUOffloadEnum.Partial,
            cache_dir=self._cache_dir,
            tensor_split=tensor_splitting,
        )
        estimate = result.resource_claim_estimate
        total_layers = estimate.items[-1].offloadLayers

        gpu_indexes_mapping = [value[0] for value in gpu_combination]
        gpu_offload_layers = {}
        for estimate_gpu_index in range(gpu_count):
            vram_claims = [
                (
                    memory.vrams[estimate_gpu_index].uma
                    if is_unified_memory
                    else memory.vrams[estimate_gpu_index].nonuma
                )
                for memory in estimate.items
                if not memory.fullOffloaded
            ]

            actual_gpu_index = gpu_indexes_mapping[estimate_gpu_index]
            index = binary_search(vram_claims, allocatable.vram[actual_gpu_index])
            if index <= 0:
                continue

            memory_estimate = estimate.items[index]
            if (
                is_unified_memory
                and memory_estimate.ram.uma
                > allocatable.ram
                - sum(vram.uma for vram in estimate.items[index].vrams)
            ) or (memory_estimate.ram.nonuma > allocatable.ram):
                continue

            actual_gpu_index = gpu_indexes_mapping[estimate_gpu_index]
            gpu_offload_layers[actual_gpu_index] = {
                "offload_layers": estimate.items[index].offloadLayers,
            }

        if len(gpu_offload_layers) != gpu_count:
            return None

        # Find the minimum offload layers among all gpus, offload layers less than it will need more ram and less vram.
        try_offload_layers = min(
            layers["offload_layers"] for layers in gpu_offload_layers.values()
        )

        final_offload_layers = -1
        final_gpu_claims = {}
        final_ram_claim = -1
        final_gpu_indexes = []
        for item in estimate.items[try_offload_layers::-1]:
            if (
                is_unified_memory
                and (item.ram.uma > allocatable.ram - sum(g.uma for g in item.vrams))
                or (item.ram.nonuma > allocatable.ram)
            ):
                continue

            final_offload_layers = item.offloadLayers
            final_ram_claim = item.ram.nonuma
            if is_unified_memory:
                final_ram_claim = item.ram.uma

            for index in range(len(item.vrams)):
                real_gpu_index = gpu_indexes_mapping[index]
                final_gpu_indexes.append(real_gpu_index)

                final_gpu_claims[real_gpu_index] = item.vrams[index].nonuma
                if is_unified_memory:
                    final_gpu_claims[real_gpu_index] = item.vrams[index].uma

            break

        if final_offload_layers == -1:
            return None

        return ModelInstanceScheduleCandidate(
            worker=worker,
            gpu_indexes=final_gpu_indexes,
            computed_resource_claim=ComputedResourceClaim(
                is_unified_memory=is_unified_memory,
                offload_layers=final_offload_layers,
                vram=final_gpu_claims,
                ram=final_ram_claim,
                total_layers=total_layers,
            ),
        )

    async def _find_single_worker_multi_gpu_partial_offloading_candidates(  # noqa: C901
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi gpu partial offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """

        total_gpu = len(worker.status.gpu_devices) if worker.status.gpu_devices else 0
        if total_gpu < 2:
            return []

        if self._selected_gpu_ids_by_worker:
            if worker.name not in self._selected_gpu_ids_by_worker:
                return []
            elif len(self._selected_gpu_ids_by_worker.get(worker.name)) < 2:
                return []

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        gpu_combinations = await self._generate_combinations_for_worker_gpu(
            allocatable, worker
        )

        if self._param_tensor_split:
            # use specified tensor split when the param is set.
            if total_gpu < len(self._param_tensor_split.split(",")):
                return None
            gpu_combinations = await self._generate_combinations_given_tensor_split()

        candidates: List[ModelInstanceScheduleCandidate] = []
        for gpu_count in gpu_combinations:
            for gpu_combination in gpu_combinations[gpu_count]:
                candidate = await self._find_single_worker_multi_gpu_partial_offloading_candidates_with_combination(
                    worker, gpu_combination
                )

                if candidate:
                    candidates.append(candidate)

        if not candidates:
            return None

        max_offload_layers = _get_max_offload_layers(candidates)
        max_offload_candidates = _filter_candidates_by_max_offload_layers(
            candidates, max_offload_layers
        )

        min_gpu_count = min(
            len(candidate.gpu_indexes) for candidate in max_offload_candidates
        )

        return [
            candidate
            for candidate in max_offload_candidates
            if len(candidate.gpu_indexes) == min_gpu_count
        ]

    async def find_single_worker_cpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker without offloading candidates for the model instance with workers.
        """
        candidates = []
        for worker in workers:
            result = await self._find_single_worker_with_cpu_candidates(worker)
            if result:
                candidates.extend(result)
        return candidates

    async def _find_single_worker_with_cpu_candidates(
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker without offloading candidates for the model instance.
        """

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        is_unified_memory = worker.status.memory.is_unified_memory

        result = await calculate_model_resource_claim(
            self._model_instance,
            self._model,
            GPUOffloadEnum.Disable,
            cache_dir=self._cache_dir,
        )
        estimate = result.resource_claim_estimate

        full_offload_result = await calculate_model_resource_claim(
            self._model_instance,
            self._model,
            GPUOffloadEnum.Full,
            cache_dir=self._cache_dir,
        )
        total_layers = full_offload_result.resource_claim_estimate.items[
            0
        ].offloadLayers

        ram_claim = estimate.items[0].ram.nonuma
        if is_unified_memory:
            ram_claim = estimate.items[0].ram.uma

        if ram_claim > allocatable.ram:
            return []

        return [
            ModelInstanceScheduleCandidate(
                worker=worker,
                gpu_indexes=None,
                computed_resource_claim=ComputedResourceClaim(
                    is_unified_memory=is_unified_memory,
                    offload_layers=0,
                    vram=None,
                    ram=ram_claim,
                    total_layers=total_layers,
                ),
            )
        ]

    async def find_multi_worker_multi_gpu_candidates(self, workers: List[Worker]):
        if not self._model.distributable:
            return []

        worker_map = {worker.id: worker for worker in workers}
        combinations, workers_allocatable, workers_gpus_allocatable = (
            await self._generate_combinations_for_worker_with_rpc_servers(workers)
        )

        candidates = []
        for count in combinations:
            for combination in combinations[count]:
                estimate_satisfied_candidate = (
                    await self._find_multi_worker_multi_gpu_candidate_with_combination(
                        combination,
                        worker_map,
                        workers_allocatable,
                        workers_gpus_allocatable,
                    )
                )

                if estimate_satisfied_candidate:
                    candidates.append(estimate_satisfied_candidate)

        if not candidates:
            return []

        max_offload_layers = _get_max_offload_layers(candidates)
        total_layers = candidates[0].computed_resource_claim.total_layers

        if not self._model.cpu_offloading and max_offload_layers != total_layers:
            return []

        final_candidates = _filter_candidates_by_max_offload_layers(
            candidates, max_offload_layers
        )
        return final_candidates

    async def _generate_combinations_for_worker_gpu(
        self, allocatable: Allocatable, worker: Worker
    ) -> dict[Tuple[Tuple[int]]]:
        total_gpu = len(worker.status.gpu_devices)
        sorted_gpus_memory = sorted(
            allocatable.vram.items(), key=lambda item: item[1], reverse=True
        )

        gpu_combinations = {}
        for i in range(2, total_gpu + 1):
            gpu_combinations[i] = list(itertools.combinations(sorted_gpus_memory, i))

        if self._selected_gpu_ids_by_worker.get(worker.name):
            filtered_gpu_combinations = (
                self._filter_selected_combinations_for_worker_gpu(
                    gpu_combinations, worker
                )
            )
            gpu_combinations = filtered_gpu_combinations

        # gpu_combinations examples:
        # { 2: (($gpu_index, $gpu_allocatable), ($gpu_index, $gpu_allocatable))}
        return gpu_combinations

    def _filter_selected_combinations_for_worker_gpu(
        self, gpu_combinations: dict[Tuple[Tuple[int]]], worker: Worker
    ) -> dict[Tuple[Tuple[int]]]:

        index_device_type = {}
        for device in worker.status.gpu_devices:
            index_device_type[device.index] = device.type

        filtered_gpu_combinations = {}
        selected_worker_gpu_ids = sorted(
            self._selected_gpu_ids_by_worker.get(worker.name)
        )
        selected_worker_gpu_count = len(selected_worker_gpu_ids)

        filtered_gpu_combination = gpu_combinations.get(selected_worker_gpu_count)
        if filtered_gpu_combination:
            for fc in filtered_gpu_combination:
                filtered_gpu_combination_ids = []
                for gpu_vram in fc:
                    gpu_type = index_device_type.get(gpu_vram[0])
                    gpu_id = f"{worker.name}:{gpu_type}:{gpu_vram[0]}"
                    filtered_gpu_combination_ids.append(gpu_id)

                if selected_worker_gpu_ids == sorted(filtered_gpu_combination_ids):
                    if selected_worker_gpu_count not in filtered_gpu_combinations:
                        filtered_gpu_combinations[selected_worker_gpu_count] = []

                    filtered_gpu_combinations[selected_worker_gpu_count].append(fc)

        return filtered_gpu_combinations

    async def _generate_combinations_given_tensor_split(
        self,
    ) -> dict[Tuple[Tuple[int]]]:
        """
        Generate gpu combinations given tensor split.
        Example:
            Given: tensor_split = "1,5,8"
            Output:
            {
                3: [
                    ((0, 1), (1, 5), (2, 8))
                ]
            }
        """
        tensor_splits = [int(x) for x in self._param_tensor_split.split(",")]
        n_split = len(tensor_splits)

        split_by_index = []
        for i in range(n_split):
            split_by_index.append((i, tensor_splits[i]))
        gpu_combinations = {
            n_split: list(itertools.combinations(split_by_index, n_split))
        }

        return gpu_combinations

    async def _generate_combinations_for_worker_with_rpc_servers(
        self, workers: List[Worker]
    ) -> tuple[Dict, Dict, List]:

        workers_allocatable = {}
        workers_allocatable_vram = []
        workers_gpus_allocatable = []

        workers_allocatable_gpu_ids = {}
        workers_gpu_indexes_type = {}
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            result = await get_worker_allocatable_resource(self._engine, worker)
            workers_allocatable[worker.id] = result
            worker_allocatable_vram = sum(result.vram.values())
            if worker_allocatable_vram > 0:
                workers_allocatable_vram.append([worker.id, worker_allocatable_vram])

            workers_allocatable_gpu_ids[worker.id] = []
            for gpu_device in worker.status.gpu_devices:
                worker_gpu_index_key = f"{worker.id}:{gpu_device.index}"
                workers_gpu_indexes_type[worker_gpu_index_key] = gpu_device.type

                if gpu_device.index is None:
                    logger.warning(
                        f"gpu index is not found for {worker.name} {gpu_device.name}"
                    )

                gpu_allocatable_vram = result.vram.get(gpu_device.index)
                if gpu_allocatable_vram is not None and gpu_allocatable_vram > 0:
                    workers_gpus_allocatable.append(
                        [worker.id, gpu_device.index, gpu_allocatable_vram]
                    )

                    gpu_id = f"{worker.name}:{gpu_device.type}:{gpu_device.index}"
                    workers_allocatable_gpu_ids[worker.id].append(gpu_id)

        sorted_workers = sorted(
            workers_allocatable_vram, key=lambda item: item[1], reverse=True
        )
        sorted_gpus = sorted(
            workers_gpus_allocatable, key=lambda item: item[2], reverse=True
        )

        combinations = {}
        for i in range(1, (MAX_RPC_SERVER_COUNT + 1)):
            c = [
                (r, *v)
                for r in sorted_workers
                for v in itertools.combinations(sorted_gpus, i)
            ]

            combinations[i + 1] = [
                r for r in c if all(r[0][0] != r[j][0] for j in range(1, len(r)))
            ]

        if self._selected_gpu_ids:
            filtered_combinations = (
                self._filter_selected_combinations_for_worker_with_rpc_servers(
                    combinations,
                    workers_allocatable_gpu_ids,
                    workers_gpu_indexes_type,
                    workers,
                )
            )
            combinations = filtered_combinations

        # combinations examples:
        # [( ($worker_id, $worker_allocatable_vram), ($worker_id, $gpu_index, $gpu_allocatable), ($worker_id, $gpu_index, $gpu_allocatable) )]
        return combinations, workers_allocatable, workers_gpus_allocatable

    def _filter_selected_combinations_for_worker_with_rpc_servers(
        self,
        combinations: Dict,
        workers_allocatable_gpu_ids,
        workers_gpu_indexes_type,
        workers: List[Worker],
    ) -> Dict:
        worker_map = {worker.id: worker for worker in workers}
        filtered_combinations = {}
        for count in combinations:
            for combination in combinations[count]:
                combination_gpu_ids = []

                main_worker_id = combination[0][0]
                main_worker_gpu_ids = workers_allocatable_gpu_ids.get(main_worker_id)
                combination_gpu_ids.extend(main_worker_gpu_ids)

                for i in range(1, len(combination)):
                    rpc_worker_id = combination[i][0]
                    rpc_worker_name = worker_map.get(rpc_worker_id).name

                    rpc_key = f"{combination[i][0]}:{combination[i][1]}"
                    rpc_gpu_type = workers_gpu_indexes_type.get(rpc_key)
                    rpc_gpu_id = f"{rpc_worker_name}:{rpc_gpu_type}:{combination[i][1]}"

                    combination_gpu_ids.append(rpc_gpu_id)

                if sorted(combination_gpu_ids) == sorted(self._selected_gpu_ids):
                    if count not in filtered_combinations:
                        filtered_combinations[count] = []

                    filtered_combinations[count].append(combination)

        return filtered_combinations

    async def _check_combination_rpc_servers(
        self,
        combination,
        worker_map: Dict[int, Worker],
        e: memoryEstimate,
        main_worker_gpu_indexes: List[int],
    ) -> List[ModelInstanceRPCServer]:
        """
        Check the rpc servers resource satisfied with combination.
        combination example: ( ($worker_id, $worker_allocatable_vram), ($worker_id, $gpu_index, $gpu_allocatable), ($worker_id, $gpu_index, $gpu_allocatable) )
        """

        rpc_servers: List[ModelInstanceRPCServer] = []

        for i in range(1, len(combination)):
            r_worker_id = combination[i][0]
            r_gpu_index = combination[i][1]
            r_allocatable = combination[i][2]
            r_is_unified_memory = worker_map.get(
                r_worker_id
            ).status.memory.is_unified_memory

            position = i - 1
            r_vram_claim = e.vrams[position].nonuma
            if r_is_unified_memory:
                r_vram_claim = e.vrams[position].uma

            if r_vram_claim > r_allocatable:
                break

            rpc_servers.append(
                ModelInstanceRPCServer(
                    worker_id=r_worker_id,
                    gpu_index=r_gpu_index,
                    computed_resource_claim=ComputedResourceClaim(
                        is_unified_memory=r_is_unified_memory,
                        offload_layers=e.vrams[position].handleLayers,
                        vram={r_gpu_index: r_vram_claim},
                        ram=0,
                        total_layers=e.offloadLayers,
                    ),
                )
            )

        if len(rpc_servers) != len(combination) - 1:
            return []

        return rpc_servers

    async def _find_multi_worker_multi_gpu_candidate_with_combination(
        self,
        combination,
        worker_map: Dict[int, Worker],
        workers_allocatable,
        workers_gpus_allocatable,
    ):
        """
        find multi worker multi gpu candidate with combination.
        combination example: ( ($worker_id, $worker_allocatable_vram), ($worker_id, $gpu_index, $gpu_allocatable), ($worker_id, $gpu_index, $gpu_allocatable) )
        """

        main_worker_id = combination[0][0]
        main_worker = worker_map.get(main_worker_id)
        main_worker_is_unified_memory = main_worker.status.memory.is_unified_memory
        main_worker_gpus = [
            [value[1], value[2]]
            for value in workers_gpus_allocatable
            if value[0] == main_worker_id
        ]
        main_worker_gpu_indexes = [value[0] for value in main_worker_gpus]

        flag_tensor_spliting = []
        flag_rpc_servers = []
        for i in range(1, len(combination)):
            c_worker_id = combination[i][0]

            flag_rpc_servers.append(f"{worker_map.get(c_worker_id).name}:{50052 + i}")
            flag_tensor_spliting.append(combination[i][2])

        flag_tensor_spliting.extend([value[1] for value in main_worker_gpus])

        result = await calculate_model_resource_claim(
            self._model_instance,
            self._model,
            GPUOffloadEnum.Partial,
            cache_dir=self._cache_dir,
            tensor_split=flag_tensor_spliting,
            rpc=flag_rpc_servers,
        )

        estimate_satisfied_candidate = None
        estimate = sorted(
            result.resource_claim_estimate.items,
            key=lambda x: x.offloadLayers,
            reverse=True,
        )
        total_layers = estimate[0].offloadLayers

        for e in estimate:
            # main worker checking.
            main_worker_ram_claim = e.ram.nonuma
            if main_worker_is_unified_memory:
                main_worker_ram_claim = e.ram.uma

            if main_worker_ram_claim > workers_allocatable.get(main_worker_id).ram:
                continue

            main_worker_vram_claim = {}
            main_worker_satisfied = False
            for (
                main_worker_gpu_index,
                main_worker_gpu_allocatable,
            ) in workers_allocatable.get(main_worker_id).vram.items():
                # vrams: [rpc_server1, rpc_server2, ..., main_worker]
                position = len(flag_rpc_servers) + main_worker_gpu_indexes.index(
                    main_worker_gpu_index
                )

                claim = e.vrams[position].nonuma
                if main_worker_is_unified_memory:
                    claim = e.vrams[position].uma

                if claim > main_worker_gpu_allocatable:
                    break

                main_worker_satisfied = True
                main_worker_vram_claim[main_worker_gpu_index] = claim

            if not main_worker_satisfied:
                continue

            rpc_servers = await self._check_combination_rpc_servers(
                combination, worker_map, e, main_worker_gpu_indexes
            )
            if not rpc_servers:
                continue

            estimate_satisfied_candidate = ModelInstanceScheduleCandidate(
                worker=main_worker,
                gpu_indexes=main_worker_gpu_indexes,
                computed_resource_claim=ComputedResourceClaim(
                    is_unified_memory=main_worker_is_unified_memory,
                    offload_layers=e.offloadLayers,
                    vram=main_worker_vram_claim,
                    ram=main_worker_ram_claim,
                    total_layers=total_layers,
                ),
                rpc_servers=rpc_servers,
            )
            break

        return estimate_satisfied_candidate


# arr is a sorted list from smallest to largest
def binary_search(arr, target):
    """
    Binary search the target in the arr.
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


def _filter_candidates_by_max_offload_layers(
    candidates: List[ModelInstanceScheduleCandidate], max_offload_layers
):
    return [
        candidate
        for candidate in candidates
        if candidate.computed_resource_claim.offload_layers == max_offload_layers
    ]
