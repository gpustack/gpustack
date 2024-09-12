import itertools
import logging
from typing import Dict, List, Optional, Tuple

from gpustack.policies.calculator import (
    GPUOffloadEnum,
    calculate_model_resource_claim,
    estimate,
    memoryEstimate,
)
from gpustack.policies.policy import (
    Allocatable,
    Allocated,
    ModelInstanceScheduleCandidate,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
    ModelInstanceRPCServer,
)
from gpustack.schemas.workers import Worker
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.server.db import get_engine
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

MAX_RPC_SERVER_COUNT = 3


class ResourceFitPolicy:
    def __init__(
        self,
        model: Model,
        model_instance: ModelInstance,
        cache_dir: Optional[str] = None,
        estimate: Optional[estimate] = None,
    ):
        self._engine = get_engine()
        self._model = model
        self._model_instance = model_instance
        self._cache_dir = cache_dir
        self._basic_estimate = estimate

    async def filter(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Filter the workers with the resource fit claim.
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
            self.find_single_worker_partial_offloading_candidates,
            self.find_multi_worker_multi_gpu_candidates,
            self.find_single_worker_cpu_candidates,
        ]

        for candidate_func in candidate_functions:
            if (not self._model.cpu_offloading) and (
                candidate_func == self.find_single_worker_partial_offloading_candidates
                or candidate_func == self.find_single_worker_cpu_candidates
            ):
                continue

            if (
                not self._model.distributed_inference_across_workers
            ) and candidate_func == self.find_multi_worker_multi_gpu_candidates:
                continue

            logger.debug(
                f"model {self._model.name}, filter candidates with resource fit policy: {candidate_func.__name__}, instance {self._model_instance.name}"
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
                    gpu_allocatable = gpu_combination[gci][1]

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

    async def _find_single_worker_single_gpu_partial_offloading_candidates(
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu partial offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """
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

    async def _find_single_worker_multi_gpu_partial_offloading_candidates(
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi gpu partial offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """

        total_gpu = len(worker.status.gpu_devices) if worker.status.gpu_devices else 0
        if total_gpu < 2:
            return []

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        gpu_combinations = await self._generate_combinations_for_worker_gpu(
            allocatable, worker
        )

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
        if not self._basic_estimate.distributable:
            return []

        # now single worker patial offloading has a higher priority than multi worker multi gpu.
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

        return gpu_combinations

    async def _generate_combinations_for_worker_with_rpc_servers(
        self, workers: List[Worker]
    ) -> tuple[Dict, Dict, List]:

        workers_allocatable = {}
        workers_allocatable_vram = []
        workers_gpus_allocatable = []
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            result = await get_worker_allocatable_resource(self._engine, worker)
            workers_allocatable[worker.id] = result
            workers_allocatable_vram.append([worker.id, sum(result.vram.values())])

            for gpu_device in worker.status.gpu_devices:
                if gpu_device.index is None:
                    logger.warning(
                        f"gpu index is not found for {worker.name} {gpu_device.name}"
                    )

                workers_gpus_allocatable.append(
                    [worker.id, gpu_device.index, result.vram.get(gpu_device.index)]
                )

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
                r for r in c if all(r[0][0] != r[i][0] for i in range(1, len(r)))
            ]

        # combinations examples:
        # [( ($worker_id, $worker_allocatable_vram), ($worker_id, $gpu_index, $gpu_allocatable), ($worker_id, $gpu_index, $gpu_allocatable) )]
        return combinations, workers_allocatable, workers_gpus_allocatable

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


async def get_worker_model_instances(
    engine: AsyncEngine, worker: Worker
) -> List[ModelInstance]:
    async with AsyncSession(engine) as session:
        model_instances = await ModelInstance.all_by_field(
            session, "worker_id", worker.id
        )
        return model_instances


async def get_worker_allocatable_resource(  # noqa: C901
    engine: AsyncEngine, worker: Worker
) -> Allocatable:
    """
    Get the worker with the latest allocatable resources.
    """

    def update_allocated_vram(allocated, resource_claim):
        for gpu_index, vram in resource_claim.vram.items():
            allocated.vram[gpu_index] = allocated.vram.get(gpu_index, 0) + vram

    is_unified_memory = worker.status.memory.is_unified_memory
    model_instances = await get_worker_model_instances(engine, worker)
    allocated = Allocated(ram=0, vram={})

    for model_instance in model_instances:
        if model_instance.worker_id != worker.id:
            continue
        allocated.ram += model_instance.computed_resource_claim.ram or 0
        if model_instance.gpu_indexes:
            update_allocated_vram(allocated, model_instance.computed_resource_claim)

        if (
            model_instance.distributed_servers
            and model_instance.distributed_servers.rpc_servers
        ):
            for rpc_server in model_instance.distributed_servers.rpc_servers:
                if rpc_server.computed_resource_claim:
                    # rpc server only consider the vram
                    update_allocated_vram(allocated, rpc_server.computed_resource_claim)

    allocatable = Allocatable(ram=0, vram={})
    if worker.status.gpu_devices:
        for _, gpu in enumerate(worker.status.gpu_devices):
            gpu_index = gpu.index

            if gpu.memory is None or gpu.memory.total is None:
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
        # the caculated with max allow gpu memory and the allocatable memory.
        if allocatable.vram:
            allocatable.vram[0] = min(allocatable.ram, allocatable.vram[0])

    logger.debug(
        f"Worker {worker.name} reserved memory: {worker.system_reserved.ram}, "
        f"reserved gpu memory: {worker.system_reserved.vram}, "
        f"allocatable memory: {allocatable.ram}, "
        f"allocatable gpu memory: {allocatable.vram}"
    )
    return allocatable


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
