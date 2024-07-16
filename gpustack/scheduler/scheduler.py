import asyncio
from datetime import datetime, timedelta, timezone
import logging
import queue
from sqlmodel.ext.asyncio.session import AsyncSession
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from gpustack.scheduler.policy import (
    ModelInstanceScheduleCandidate,
    ResourceFitPolicy,
    SystemReservedResource,
)
from gpustack.scheduler.queue import AsyncUniqueQueue
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
)
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
                int(system_reserved.get("gpu_memory", 0)) * 1024 * 1024 * 1024
            )  # GB to Bytes
            self._system_reserved = SystemReservedResource(
                memory=memory, gpu_memory=gpu_memory
            )

    async def start(self):
        """
        Start the scheduler.
        """

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
            if event.type != EventType.CREATED:
                continue

            await self._enqueue_pending_instances()

        logger.info("Scheduler started.")

    async def _enqueue_pending_instances(self):
        """
        Get the pending model instances.
        """
        try:
            async with AsyncSession(self._engine) as session:
                instances = await ModelInstance.all(session)
                tasks = []
                for instance in instances:
                    if self._should_schedule(instance):
                        task = asyncio.create_task(
                            self._process_calculate_model_resource_claim(instance)
                        )
                        tasks.append(task)

                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Failed to enqueue pending model instances: {e}")

    async def _process_calculate_model_resource_claim(self, instance: ModelInstance):
        async with AsyncSession(self._engine) as session:
            try:
                instance = await ModelInstance.one_by_id(session, instance.id)
                if instance.state != ModelInstanceStateEnum.ANALYZING:
                    instance.state = ModelInstanceStateEnum.ANALYZING
                    instance.state_message = "Evaluating resource requirements."
                    await instance.update(session)

                model = await Model.one_by_id(session, instance.model_id)
                if model is None:
                    raise Exception("Model not found.")

                task_output = await calculate_model_resource_claim(instance, model)

                await self._queue.put(task_output)

            except Exception as e:
                try:
                    instance.state = ModelInstanceStateEnum.ERROR
                    instance.state_message = (
                        f"Failed to calculate model resource claim: {e}"
                    )
                    await instance.update(session)
                except Exception as ue:
                    logger.error(
                        f"Failed to update model instance: {ue}. Original error: {e}"
                    )

    def _should_schedule(self, instance: ModelInstance) -> bool:
        """
        Check if the model instance should be scheduled.
        Args:
            instance: ModelInstance to check.
        """

        return (
            (
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.PENDING
            )
            or (
                # Reschedule while it stays in anayzing state for too long,
                # maybe the server is restarted.
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.ANALYZING
                and datetime.now(timezone.utc)
                - instance.updated_at.replace(tzinfo=timezone.utc)
                > timedelta(minutes=3)
            )
            or (
                # Reschedule while it stays in scheduled state for too long,
                # maybe the worker is down.
                instance.worker_id is not None
                and instance.state == ModelInstanceStateEnum.SCHEDULED
                and datetime.now(timezone.utc)
                - instance.updated_at.replace(tzinfo=timezone.utc)
                > timedelta(minutes=3)
            )
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
        logger.debug(f"Scheduling model instance {item.model_instance.name}")

        state_message = ""
        instance = item.model_instance
        estimate = item.resource_claim_estimate
        filterPolicies = [ResourceFitPolicy(estimate, self._system_reserved)]

        async with AsyncSession(self._engine) as session:
            workers = await Worker.all_by_field(session, "state", WorkerStateEnum.READY)

            candidates = []
            if len(workers) != 0:
                try:
                    for policy in filterPolicies:
                        candidates = await policy.filter(workers)
                except Exception as e:
                    state_message = f"Failed to filter workers with policies: {e}"
                    logger.error(state_message)

            model_instance = await ModelInstance.one_by_id(session, instance.id)
            if len(candidates) == 0:
                # update model instance.
                model_instance.state = ModelInstanceStateEnum.ERROR
                model_instance.state_message = "No suitable workers."
                if state_message != "":
                    model_instance.state_message = state_message

                await model_instance.update(session, model_instance)
                logger.debug(
                    f"No suitable workers for model instance {model_instance.name}"
                )
            else:
                # pick the highest offload layer, should scoring all the candidates later.
                candidate: ModelInstanceScheduleCandidate = candidates[0]
                if len(candidates) > 1:
                    for i in range(1, len(candidates)):
                        if (
                            candidates[i].computed_resource_claim.offload_layers
                            > candidate.computed_resource_claim.offload_layers
                        ):
                            candidate = candidates[i]

                # update model instance.
                model_instance.state = ModelInstanceStateEnum.SCHEDULED
                model_instance.state_message = ""
                model_instance.worker_id = candidate.worker.id
                model_instance.worker_name = candidate.worker.name
                model_instance.worker_ip = candidate.worker.ip
                model_instance.computed_resource_claim = (
                    candidate.computed_resource_claim
                )
                model_instance.gpu_index = candidate.gpu_index

                await model_instance.update(session, model_instance)

                logger.debug(
                    f"Scheduled model instance {model_instance.name} to worker "
                    f"{model_instance.worker_name} gpu {candidate.gpu_index}"
                )
