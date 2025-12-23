import logging
from typing import List, Optional

from gpustack.policies.base import ModelInstanceScore
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_engine
from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)

MaxScore = 100


class StatusScorer:
    def __init__(self, model: Model, model_instance: Optional[ModelInstance] = None):
        self._engine = get_engine()
        self._model = model
        self._model_instance = model_instance

    async def score_instances(
        self, instances: List[ModelInstance]
    ) -> List[ModelInstanceScore]:
        """
        Score the instances with the worker and instance status.
        """

        logger.debug(f"model {self._model.name}, score instances with status policy")

        scored_instances = []
        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session)
            worker_map = {worker.id: worker for worker in workers}

            for instance in instances:
                if instance.worker_id is None:
                    scored_instances.append(
                        ModelInstanceScore(model_instance=instance, score=0)
                    )
                    continue

                score = 0
                worker = worker_map.get(instance.worker_id)
                if worker is None:
                    scored_instances.append(
                        ModelInstanceScore(model_instance=instance, score=0)
                    )
                    continue

                if worker.state == WorkerStateEnum.NOT_READY:
                    score = 0
                elif instance.state == ModelInstanceStateEnum.ERROR:
                    score = 0
                elif (
                    worker.state == WorkerStateEnum.READY
                    and instance.state == ModelInstanceStateEnum.RUNNING
                ):
                    score = MaxScore
                else:
                    score = 50

                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=score)
                )

            return scored_instances
