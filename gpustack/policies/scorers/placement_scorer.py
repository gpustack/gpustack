from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, List, Optional

from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScheduleCandidate,
    ModelInstanceScore,
    ModelInstanceScorer,
    ScheduleCandidatesScorer,
)
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
    ModelInstanceSubordinateWorker,
    PlacementStrategyEnum,
)
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine

MaxScore = 100

logger = logging.getLogger(__name__)


@dataclass
class ResourceWeight:
    vram: int = 2
    ram: int = 1


@dataclass
class ModelWeight:
    current: int = 1
    others: int = 0.2


@dataclass
class InferenceServerTypeWeight:
    server: int = 5
    rpc_server: int = 1  # max rpc server count is 3


class ScaleTypeEnum(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


class PlacementScorer(ScheduleCandidatesScorer, ModelInstanceScorer):
    def __init__(
        self,
        model: Model,
        scale_type: ScaleTypeEnum = ScaleTypeEnum.SCALE_UP,
    ):
        self._engine = get_engine()
        self._model = model
        self._resource_weight = ResourceWeight()
        self._model_weight = ModelWeight()
        self._inference_server_type_weight = InferenceServerTypeWeight()
        self._scale_type = scale_type

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidate with placement strategy.
        """

        logger.debug(
            f"model {self._model.readable_source}, score canidates with {self._scale_type} placement policy"
        )

        if self._model.placement_strategy == PlacementStrategyEnum.SPREAD:
            return await self.score_spread(candidates)
        elif self._model.placement_strategy == PlacementStrategyEnum.BINPACK:
            return await self.score_binpack(candidates)
        else:
            raise ValueError(
                f"Invalid placement strategy {self._model.placement_strategy}"
            )

    async def score_instances(
        self, instances: List[ModelInstance]
    ) -> List[ModelInstanceScore]:
        """
        Score the instances with placement strategy.
        """

        logger.debug(
            f"model {self._model.name}, score instances with {self._scale_type} placement policy"
        )

        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session)
            worker_map = {worker.id: worker for worker in workers}

            if self._model.placement_strategy == PlacementStrategyEnum.SPREAD:
                return await self.score_spread_instances(instances, worker_map)
            elif self._model.placement_strategy == PlacementStrategyEnum.BINPACK:
                return await self.score_binpack_instances(instances, worker_map)
            else:
                raise ValueError(
                    f"Invalid placement strategy {self._model.placement_strategy}"
                )

    async def score_binpack(  # noqa: C901
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidates with the binpack strategy.
        """
        for candidate in candidates:
            allocatable = await get_worker_allocatable_resource(
                self._engine, candidate.worker
            )

            final_score = 0
            score = await self._score_binpack_item(
                candidate.gpu_indexes,
                candidate.computed_resource_claim,
                allocatable,
                self._scale_type,
            )
            final_score = score

            if candidate.subordinate_workers:
                rpc_server_score = await self._score_binpack_subordinate_workers(
                    candidate.subordinate_workers, self._scale_type
                )
                final_score = (
                    score * self._inference_server_type_weight.server
                    + rpc_server_score
                    * len(candidate.subordinate_workers)
                    * self._inference_server_type_weight.rpc_server
                ) / (
                    self._inference_server_type_weight.server
                    + self._inference_server_type_weight.rpc_server
                    * len(candidate.subordinate_workers)
                )

            candidate.score = final_score

        return candidates

    async def score_binpack_instances(  # noqa: C901
        self, instances: List[ModelInstance], worker_map: dict
    ) -> List[ModelInstanceScore]:
        """
        Score the candidates with the binpack strategy.
        """
        scored_instances = []

        for instance in instances:
            if instance.worker_id is None:
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            worker = worker_map.get(instance.worker_id)
            if worker is None:
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            allocatable = await get_worker_allocatable_resource(self._engine, worker)

            final_score = 0
            score = await self._score_binpack_item(
                instance.gpu_indexes,
                instance.computed_resource_claim,
                allocatable,
                self._scale_type,
            )
            final_score = score

            if (
                instance.distributed_servers
                and instance.distributed_servers.subordinate_workers
            ):
                subordinate_workers = instance.distributed_servers.subordinate_workers
                subordinate_worker_score = (
                    await self._score_binpack_subordinate_workers(
                        subordinate_workers, self._scale_type
                    )
                )
                final_score = (
                    score * self._inference_server_type_weight.server
                    + subordinate_worker_score
                    * len(subordinate_workers)
                    * self._inference_server_type_weight.rpc_server
                ) / (
                    self._inference_server_type_weight.server
                    + self._inference_server_type_weight.rpc_server
                    * len(subordinate_workers)
                )

            scored_instances.append(
                ModelInstanceScore(model_instance=instance, score=final_score)
            )

        return scored_instances

    async def score_spread(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidates with the spread strategy.
        """
        worker_model_instances_count_map = await self._get_worker_model_instance_count()

        for candidate in candidates:
            candidate.score = await self._score_spread_item(
                candidate.gpu_indexes,
                candidate.worker,
                worker_model_instances_count_map,
            )

        return candidates

    async def score_spread_instances(
        self, instances: List[ModelInstance], worker_map: dict
    ) -> List[ModelInstanceScore]:
        """
        Score the candidates with the spread strategy.
        """
        worker_model_instances_count_map = await self._get_worker_model_instance_count()

        scored_instances = []
        for instance in instances:
            if instance.worker_id is None:
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            worker = worker_map.get(instance.worker_id)
            score = await self._score_spread_item(
                instance.gpu_indexes, worker, worker_model_instances_count_map
            )
            scored_instances.append(
                ModelInstanceScore(model_instance=instance, score=score)
            )

        return scored_instances

    async def _score_spread_item(
        self,
        instance_gpu_indexes: List[int],
        worker: Worker,
        worker_model_instances_count_map: dict,
    ) -> int:
        """
        Score the candidates with the spread strategy.
        """

        if worker is None:
            return 0

        instance_worker_id = worker.id
        # level 1: max score, no model instances
        if instance_worker_id not in worker_model_instances_count_map:
            return MaxScore

        instance_count_map = worker_model_instances_count_map.get(
            instance_worker_id, {}
        )

        if instance_gpu_indexes is not None and len(instance_gpu_indexes) > 0:
            return await self._score_spread_gpu(
                instance_count_map,
                worker,
                instance_gpu_indexes,
            )
        else:
            return await self._score_spread_cpu(instance_count_map.get("total", {}))

    async def _score_binpack_item(  # noqa: C901
        self,
        gpu_indexes: List[int],
        computed_resource_claim: ComputedResourceClaim,
        allocatable: Allocatable,
        scale_type: str,
    ) -> int:
        score = 0
        gpu_count = len(gpu_indexes) if gpu_indexes else 0

        def calculate_score(
            ram_claim: Optional[int],
            ram_allocatable: Optional[int],
            vram_claim: Dict[int, int],
            vram_allocatable: Dict[int, int],
        ):
            if ram_claim is None or ram_allocatable is None or ram_allocatable == 0:
                ram_score = 0
            else:
                ram_score = (
                    ram_claim / ram_allocatable * MaxScore * self._resource_weight.ram
                )

            vram_score = (
                vram_claim / vram_allocatable * MaxScore * self._resource_weight.vram
            )
            return (ram_score + vram_score) / (
                self._resource_weight.ram + self._resource_weight.vram
            )

        if gpu_count == 0:
            # computed_resource_claim.ram must have value when running cpu only model instance
            if scale_type == ScaleTypeEnum.SCALE_UP:
                score = computed_resource_claim.ram / allocatable.ram * MaxScore
            elif scale_type == ScaleTypeEnum.SCALE_DOWN:
                score = (
                    computed_resource_claim.ram
                    / (allocatable.ram + computed_resource_claim.ram)
                    * MaxScore
                )
        elif gpu_count == 1:
            if scale_type == ScaleTypeEnum.SCALE_UP:
                score = calculate_score(
                    computed_resource_claim.ram,
                    allocatable.ram,
                    computed_resource_claim.vram[gpu_indexes[0]],
                    allocatable.vram[gpu_indexes[0]],
                )
            elif scale_type == ScaleTypeEnum.SCALE_DOWN:
                score = calculate_score(
                    computed_resource_claim.ram,
                    allocatable.ram + computed_resource_claim.ram or 0,
                    computed_resource_claim.vram[gpu_indexes[0]],
                    allocatable.vram[gpu_indexes[0]]
                    + computed_resource_claim.vram[gpu_indexes[0]],
                )
        else:
            for i in gpu_indexes:
                if scale_type == ScaleTypeEnum.SCALE_UP:
                    result = calculate_score(
                        computed_resource_claim.ram,
                        allocatable.ram,
                        computed_resource_claim.vram[i],
                        allocatable.vram[i],
                    )
                elif scale_type == ScaleTypeEnum.SCALE_DOWN:
                    result = calculate_score(
                        computed_resource_claim.ram,
                        allocatable.ram + (computed_resource_claim.ram or 0),
                        computed_resource_claim.vram[i],
                        allocatable.vram[i] + computed_resource_claim.vram[i],
                    )
                if result > score:
                    score = result

        return score

    async def _score_spread_gpu(
        self,
        instance_count_map: dict,
        worker: Worker,
        instance_gpu_indexes: List[int],
    ) -> int:
        score = 0
        worker_current_model_instance_count = instance_count_map.get("total", {}).get(
            "current", 0
        )
        worker_other_model_instance_count = instance_count_map.get("total", {}).get(
            "others", 0
        )

        worker_gpu_count = len(worker.status.gpu_devices)
        each_gpu_max_score = 10 / (worker_gpu_count + 1)
        gpu_map = instance_count_map.get("gpu", {})

        if (
            worker_current_model_instance_count == 0
            and worker_other_model_instance_count == 0
        ):
            score = MaxScore

        elif (
            worker_current_model_instance_count == 0
            and worker_other_model_instance_count > 0
        ):
            # level 2: 90 < score < 100, only have other model's instances
            score = 90

            for gpu_index in instance_gpu_indexes:
                if gpu_index not in gpu_map:
                    score += each_gpu_max_score / 1
                    continue
                count = gpu_map.get(gpu_index, {}).get("others", 0)
                score += each_gpu_max_score / (count + 1)

        elif (
            worker_current_model_instance_count > 0
            and worker_other_model_instance_count == 0
        ):
            # level 3: 80 < score < 90, only have current model's instances
            score = 80

            for gpu_index in instance_gpu_indexes:
                if gpu_index not in gpu_map:
                    score += each_gpu_max_score / 1
                    continue
                count = gpu_map.get(gpu_index, {}).get("current", 0)
                score += each_gpu_max_score / (count + 1)

        else:
            # level 4: 70 < score < 80, have both current model's instances and other model's instances
            score = 70

            for gpu_index in instance_gpu_indexes:
                if gpu_index not in gpu_map:
                    score += each_gpu_max_score / 1
                    continue
                current_count = gpu_map.get(gpu_index, {}).get("current", 0)
                others_count = gpu_map.get(gpu_index, {}).get("others", 0)
                score += each_gpu_max_score / (
                    (current_count + 1) + (others_count + 1) * self._model_weight.others
                )

        return score

    async def _score_spread_cpu(self, instance_count_map: dict) -> int:
        worker_current_model_instance_count = instance_count_map.get("current", 0)

        worker_others_model_instance_count = instance_count_map.get("others", 0)

        score = 0
        if (
            worker_current_model_instance_count == 0
            and worker_others_model_instance_count == 0
        ):
            # level 1: max score, no model instances
            score = MaxScore
        elif (
            worker_current_model_instance_count == 0
            and worker_others_model_instance_count > 0
        ):
            # level 2: 90 < score < 100, only have other model's instances
            score = 10 / (worker_others_model_instance_count + 1)
            score += 90
        elif (
            worker_current_model_instance_count > 0
            and worker_others_model_instance_count == 0
        ):
            # level 3: 80 < score < 90, only have current model's instances
            score = 10 / (worker_current_model_instance_count + 1)
            score += 80
        else:
            # level 4: 70 < score < 80, have both current model's instances and other model's instances
            score = 10 / (
                (worker_current_model_instance_count + 1)
                + (worker_others_model_instance_count + 1) * self._model_weight.others
            )
            score += 70

        return score

    async def _score_binpack_subordinate_workers(
        self, subordinate_workers: List[ModelInstanceSubordinateWorker], scale_type: str
    ) -> int:
        if subordinate_workers is None:
            return 0

        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session)
            worker_map = {worker.id: worker for worker in workers}

            score = 0
            for subordinate_worker in subordinate_workers:
                allocatable = await get_worker_allocatable_resource(
                    self._engine,
                    worker_map.get(subordinate_worker.worker_id),
                )

                score += await self._score_binpack_item(
                    subordinate_worker.gpu_indexes,
                    subordinate_worker.computed_resource_claim,
                    allocatable,
                    scale_type,
                )

            return score

    async def _get_worker_model_instance_count(self) -> dict:
        """
        Get current model and other models deployed model instance count for each worker/gpu.

        Returns:
            dict: A map of worker id to model instance count.

        Example:
            {
                "worker_1": {
                    "total": {"current": 2, "others": 3},
                    "gpu": {
                        0: {"current": 1, "others": 2},
                        1: {"current": 0, "others": 1}
                    }
                },
                "worker_2": {
                    "total": {"current": 1, "others": 1},
                    "gpu": {
                        0: {"current": 2, "others": 0}
                    }
                }
            }
        """

        if not hasattr(self._model, "id") or self._model.id is None:
            return {}

        model_id = self._model.id
        model_instances = await get_model_instances(self._engine)

        worker_model_instances_count_map = defaultdict(
            lambda: {
                "total": {"current": 0, "others": 0},
                "gpu": defaultdict(lambda: {"current": 0, "others": 0}),
            }
        )

        for model_instance in model_instances:
            if model_instance.worker_id is None:
                continue

            is_current_model = model_instance.model_id == model_id
            if model_instance.gpu_indexes:
                for gpu_index in model_instance.gpu_indexes:
                    update_count(
                        worker_model_instances_count_map,
                        model_instance.worker_id,
                        gpu_index,
                        is_current_model,
                    )
            else:
                update_count(
                    worker_model_instances_count_map,
                    model_instance.worker_id,
                    None,
                    is_current_model,
                )

            if (
                model_instance.distributed_servers
                and model_instance.distributed_servers.subordinate_workers
            ):
                for (
                    subordinate_worker
                ) in model_instance.distributed_servers.subordinate_workers:
                    for subordinate_gpu_index in subordinate_worker.gpu_indexes:
                        update_count(
                            worker_model_instances_count_map,
                            subordinate_worker.worker_id,
                            subordinate_gpu_index,
                            is_current_model,
                        )

        return worker_model_instances_count_map


def update_count(
    worker_model_instances_count_map, worker_id, gpu_index, is_current_model
):
    if gpu_index is not None:
        key = "current" if is_current_model else "others"
        worker_model_instances_count_map[worker_id]["gpu"][gpu_index][key] += 1

    key = "current" if is_current_model else "others"
    worker_model_instances_count_map[worker_id]["total"][key] += 1


async def get_model_instances(engine: AsyncEngine) -> List[ModelInstance]:
    async with AsyncSession(engine) as session:
        model_instances = await ModelInstance.all(session)
        return model_instances
