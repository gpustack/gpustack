import asyncio
import logging
import queue
from typing import List
from sqlmodel.ext.asyncio.session import AsyncSession
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from gpustack.logging import setup_logging
from gpustack.scheduler.policy import (
    ModelInstanceScheduleCandidate,
    ResourceFitPolicy,
    SystemReservedResource,
)
from gpustack.scheduler.queue import AsyncUniqueQueue
from gpustack.schemas.workers import Worker
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.server.bus import EventType
from gpustack.server.db import get_engine
from gpustack.scheduler.calculator import (
    ModelInstanceResourceClaim,
    calculate_model_resource_claim,
)

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, check_interval: int = 30, system_reserved: dict = None):
        """
        Init the scheduler with queue and interval.
        """

        self._id = "model-instance-scheduler"
        self._check_interval = check_interval
        self._engine = get_engine()
        self._queue = AsyncUniqueQueue()
        self._system_reserved = SystemReservedResource(0, 0)

        if system_reserved is not None:
            memory = (
                int(system_reserved.get("memory", 0)) * 1024 * 1024 * 1024
            )  # GB to Bytes
            gpu_memory = (
                int(system_reserved.get("gpuMemory", 0)) * 1024 * 1024 * 1024
            )  # GB to Bytes
            self._system_reserved = SystemReservedResource(
                memory=memory, gpu_memory=gpu_memory
            )

    async def start(self):
        """
        Start the scheduler.
        """
        setup_logging()

        # scheduler queue.
        asyncio.create_task(self._schedule_cycle())

        # scheduler job trigger by time interval.
        trigger = IntervalTrigger(seconds=self._check_interval)
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self._enqueue_pending_instances,
            trigger=trigger,
            id=self._id,
            max_instances=1,
        )
        scheduler.start()

        # scheduler job trigger by event.
        async for event in ModelInstance.subscribe(self._engine):
            if event.type == EventType.DELETED:
                continue

            await self._enqueue_pending_instances()

        logger.info("Scheduler started.")

    async def _enqueue_pending_instances(self):
        """
        Get the pending model instances.
        """

        async with AsyncSession(self._engine) as session:
            instances = await ModelInstance.all_by_field(
                session, "state", ModelInstanceStateEnum.pending
            )

        tasks = []
        for instance in instances:
            if self._should_schedule(instance):
                task = asyncio.create_task(
                    self._process_calculate_model_resource_claim(instance)
                )
                tasks.append(task)

        await asyncio.gather(*tasks)

    async def _process_calculate_model_resource_claim(self, instance: ModelInstance):
        try:
            async with AsyncSession(self._engine) as session:
                model = await Model.one_by_id(session, instance.model_id)

            task_output = await calculate_model_resource_claim(instance, model)

            await self._queue.put(task_output)

        except Exception as e:
            logger.error(f"Failed to calculate model resource claim: {e}")

    def _should_schedule(self, instance: ModelInstance) -> bool:
        """
        Check if the model instance should be scheduled.
        Args:
            instance: ModelInstance to check.
        """

        return (
            instance.worker_id is None
            and instance.state == ModelInstanceStateEnum.pending
        )

    async def _schedule_cycle(self):
        while True:
            try:
                item = await self._queue.get()
                try:
                    await self._schedule_one(item)
                    self._queue.task_done()
                except Exception as e:
                    logger.error(f"Failed to schedule model instance: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Failed to get item from schedule queue: {e}")

    async def _schedule_one(self, item: ModelInstanceResourceClaim):
        """
        Schedule a model instance by picking one candidate.
        Args:
            item: ModelInstanceResourceClaim to schedule.
        """
        logger.debug(f"Scheduling model instance {item.model_instance.id}")

        state_message = ""
        instance = item.model_instance
        estimate = item.resource_claim_estimate
        candiate: ModelInstanceScheduleCandidate = None

        filterPolicies = [ResourceFitPolicy(estimate, self._system_reserved)]

        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session)

        if len(workers) != 0:
            try:
                candidates = await self._get_model_instance_schedule_candidates(workers)
                for policy in filterPolicies:
                    candidates = await policy.filter(candidates)
            except Exception as e:
                state_message = f"Failed to filter workers with policies: {e}"
                logger.error(state_message)

        model_instance = await ModelInstance.one_by_id(session, instance.id)
        if len(candidates) == 0:
            # update model instance.
            model_instance.state = ModelInstanceStateEnum.pending
            model_instance.state_message = "No fit worker"
            if state_message != "":
                model_instance.state_message = state_message

            await model_instance.update(session, model_instance)
            logger.debug(f"No fit worker for model instance {model_instance.id}")
        else:
            # pick the first candidate now, should scoring all the candidates later.
            candiate = candidates[0]

            # update model instance.
            model_instance.state = ModelInstanceStateEnum.scheduled
            model_instance.state_message = ""
            model_instance.worker_id = candiate.worker.id
            model_instance.worker_ip = candiate.worker.ip
            model_instance.computed_resource_claim = candiate.computed_resource_claim
            model_instance.gpu_index = candiate.gpu_index

            await model_instance.update(session, model_instance)

            logger.debug(
                f"Scheduled model instance {model_instance.id} to node "
                f"{candiate.worker.id} gpu {candiate.gpu_index}"
            )

    async def _get_model_instance_schedule_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Convert the workers to the candidates.
        """
        candiates = []
        for worker in workers:
            candiates.append(ModelInstanceScheduleCandidate(worker, None, None))
        return candiates
