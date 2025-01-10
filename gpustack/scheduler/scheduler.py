import asyncio
from datetime import datetime, timedelta, timezone
import logging
import os
import queue
from typing import List, Tuple
from sqlmodel.ext.asyncio.session import AsyncSession
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from gpustack.policies.candidate_selectors.vox_box_resource_fit_selector import (
    VoxBoxResourceFitSelector,
)
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.config.config import Config
from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
    WorkerFilterChain,
)
from gpustack.policies.candidate_selectors.gguf_resource_fit_selector import (
    GGUFResourceFitSelector,
)
from gpustack.policies.worker_filters.label_matching_filter import LabelMatchingFilter
from gpustack.policies.worker_filters.gpu_matching_filter import GPUMatchingFilter
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    VLLMResourceFitSelector,
)
from gpustack.scheduler.queue import AsyncUniqueQueue
from gpustack.policies.worker_filters.status_filter import StatusFilter
from gpustack.schemas.workers import Worker
from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    SourceEnum,
    get_backend,
    is_gguf_model,
    is_audio_model,
)
from gpustack.server.bus import EventType
from gpustack.server.db import get_engine
from gpustack.scheduler.calculator import (
    GPUOffloadEnum,
    calculate_model_resource_claim,
)
from gpustack.utils.gpu import parse_gpu_ids_by_worker
from gpustack.utils.hub import get_pretrained_config
from gpustack.utils.task import run_in_thread

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, cfg: Config, check_interval: int = 180):
        """
        Init the scheduler with queue and interval.
        """

        self._id = "model-instance-scheduler"
        self._config = cfg
        self._check_interval = check_interval
        self._engine = get_engine()
        self._queue = AsyncUniqueQueue()
        self._cache_dir = None

        if self._config.cache_dir is not None:
            self._cache_dir = os.path.join(self._config.cache_dir, "gguf-parser")
            os.makedirs(self._cache_dir, exist_ok=True)

            self._vox_box_cache_dir = os.path.join(self._config.cache_dir, "vox-box")
            os.makedirs(self._vox_box_cache_dir, exist_ok=True)

    async def start(self):
        """
        Start the scheduler.
        """

        try:
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
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")

        logger.info("Scheduler started.")

        # scheduler job trigger by event.
        async for event in ModelInstance.subscribe(self._engine):
            if event.type != EventType.CREATED:
                continue

            await self._enqueue_pending_instances()

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
                        task = asyncio.create_task(self._evaluate(instance))
                        tasks.append(task)

                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Failed to enqueue pending model instances: {e}")

    async def _evaluate(self, instance: ModelInstance):  # noqa: C901
        """
        Evaluate the model instance's metadata.
        """
        async with AsyncSession(self._engine) as session:
            try:
                instance = await ModelInstance.one_by_id(session, instance.id)

                model = await Model.one_by_id(session, instance.model_id)
                if model is None:
                    raise Exception("Model not found.")

                if instance.state != ModelInstanceStateEnum.ANALYZING:
                    instance.state = ModelInstanceStateEnum.ANALYZING
                    instance.state_message = "Evaluating resource requirements"
                    await instance.update(session)

                if model.source == SourceEnum.LOCAL_PATH and not os.path.exists(
                    model.local_path
                ):
                    # The local path model is not accessible from the server, skip evaluation.
                    await self._queue.put(instance)
                    return

                if is_gguf_model(model):
                    await self._evaluate_gguf_model(session, model, instance)
                    if await self.check_model_distributability(
                        session, model, instance
                    ):
                        return

                elif is_audio_model(model):
                    await self._evaluate_audio_model(session, model)
                else:
                    await self._evaluate_pretrained_config(session, model)

                await self._queue.put(instance)
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

    async def check_model_distributability(
        self, session: AsyncSession, model: Model, instance: ModelInstance
    ):
        if (
            not model.distributable
            and model.gpu_selector
            and model.gpu_selector.gpu_ids
        ):
            worker_gpu_ids = parse_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            if len(worker_gpu_ids) > 1:
                instance.state = ModelInstanceStateEnum.ERROR
                instance.state_message = (
                    "The model is not distributable to multiple workers."
                )
                await instance.update(session)
                return True
        return False

    async def _evaluate_gguf_model(
        self,
        session: AsyncSession,
        model: Model,
        instance: ModelInstance,
    ):
        task_output = await calculate_model_resource_claim(
            instance,
            model,
            offload=GPUOffloadEnum.Full,
            cache_dir=self._cache_dir,
            ollama_library_base_url=self._config.ollama_library_base_url,
        )

        should_update = False
        if task_output.resource_claim_estimate.reranking and not model.categories:
            should_update = True
            model.categories = [CategoryEnum.RERANKER]
            model.reranker = True

        if task_output.resource_claim_estimate.embeddingOnly and not model.categories:
            should_update = True
            model.categories = [CategoryEnum.EMBEDDING]
            model.embedding_only = True

        if task_output.resource_claim_estimate.imageOnly and not model.categories:
            should_update = True
            model.categories = [CategoryEnum.IMAGE]
            model.image_only = True

        if not model.categories:
            should_update = True
            model.categories = [CategoryEnum.LLM]

        if (
            task_output.resource_claim_estimate.distributable
            and not model.distributable
        ):
            should_update = True
            model.distributable = True

        if model.gpu_selector and model.gpu_selector.gpu_ids:
            worker_gpu_ids = parse_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            if (
                len(worker_gpu_ids) > 1
                and model.distributable
                and not model.distributed_inference_across_workers
            ):
                should_update = True
                model.distributed_inference_across_workers = True

        if should_update:
            await model.update(session)

    async def _evaluate_pretrained_config(
        self,
        session: AsyncSession,
        model: Model,
    ):
        try:
            pretrained_config = await run_in_thread(
                get_pretrained_config, timeout=15, model=model
            )
        except Exception as e:
            # It's not always possible to get the pretrained config.
            # For example, some models require additional parameters to be passed.
            logger.debug(f"Cannot get pretrained config: {e}")
            return

        architectures = getattr(pretrained_config, "architectures", []) or []

        # https://docs.vllm.ai/en/latest/models/supported_models.html#text-embedding
        supported_embedding_architectures = [
            "BertModel",
            "Gemma2Model",
            "MistralModel",
            "LlamaModel",
            "Qwen2Model",
            "RobertaModel",
            "RobertaForMaskedLM",
            "XLMRobertaModel",
        ]
        is_embedding_model = False

        for architecture in architectures:
            if architecture in supported_embedding_architectures:
                is_embedding_model = True
                break

        should_update = False
        if is_embedding_model and not model.categories:
            should_update = True
            model.categories = [CategoryEnum.EMBEDDING]
            model.embedding_only = True

        if not model.categories:
            should_update = True
            model.categories = [CategoryEnum.LLM]

        if should_update:
            await model.update(session)

    async def _evaluate_audio_model(
        self,
        session: AsyncSession,
        model: Model,
    ):
        try:
            from vox_box.estimator.estimate import estimate_model
            from vox_box.config import Config as VoxBoxConfig
        except ImportError:
            raise Exception("vox_box is not installed.")

        cfg = VoxBoxConfig()
        cfg.cache_dir = self._vox_box_cache_dir
        cfg.model = model.local_path
        cfg.huggingface_repo_id = model.huggingface_repo_id
        cfg.model_scope_model_id = model.model_scope_model_id

        try:
            timeout_in_seconds = 15
            model_dict = await asyncio.wait_for(
                asyncio.to_thread(estimate_model, cfg),
                timeout=timeout_in_seconds,
            )
        except Exception as e:
            logger.error(f"Failed to estimate model {model.name}: {e}")
            return

        supported = model_dict.get("supported", False)
        if not supported:
            logger.error(f"Model {model.name} is not supported.")
            return

        task_type = model_dict.get("task_type")
        if task_type == "tts":
            model.categories = [CategoryEnum.TEXT_TO_SPEECH]
            model.text_to_speech = True
        elif task_type == "stt":
            model.categories = [CategoryEnum.SPEECH_TO_TEXT]
            model.speech_to_text = True

        if model.categories and (
            CategoryEnum.TEXT_TO_SPEECH in model.categories
            or CategoryEnum.SPEECH_TO_TEXT in model.categories
        ):
            await model.update(session)

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

    async def find_candidate(
        self,
        instance: ModelInstance,
        model: Model,
        workers: List[Worker],
    ) -> Tuple[ModelInstanceScheduleCandidate, List[str]]:
        """
        Find a schedule candidate for the model instance.
        :param instance: Model instance to schedule.
        :param model: Model to schedule.
        :param workers: List of workers to consider.
        :return: A tuple containing:
                 - The schedule candidate.
                 - A list of messages for the scheduling process.
        """
        try:
            filters = [
                GPUMatchingFilter(model, instance),
                LabelMatchingFilter(model, instance),
                StatusFilter(model, instance),
            ]

            worker_filter_chain = WorkerFilterChain(filters)
            workers, messages = await worker_filter_chain.filter(workers)

            candidates_selector: ScheduleCandidatesSelector = None
            if is_gguf_model(model):
                candidates_selector = GGUFResourceFitSelector(
                    model, instance, self._cache_dir
                )
            elif is_audio_model(model):
                candidates_selector = VoxBoxResourceFitSelector(
                    self._config, model, instance, self._vox_box_cache_dir
                )
            else:
                try:
                    candidates_selector = VLLMResourceFitSelector(model, instance)
                except Exception as e:
                    return None, [f"VLLM resource fit selector init failed: {e}"]

            candidates = await candidates_selector.select_candidates(workers)

            placement_scorer = PlacementScorer(model, instance)
            candidates = await placement_scorer.score(candidates)

            candidate = self.pick_highest_score_candidate(candidates)

            if candidate is None and len(workers) > 0:
                resource_fit_message = "No workers meet the resource requirements."
                if get_backend(model) == BackendEnum.VLLM:
                    resource_fit_message += " Consider adjusting parameters such as --gpu-memory-utilization (default: 0.9), --max-model-len, or --enforce-eager to lower the resource demands."
                messages.append(resource_fit_message)
            return candidate, messages
        except Exception as e:
            state_message = (
                f"Failed to find candidate for model instance {instance.name}: {e}"
            )
            logger.error(state_message)

    async def _schedule_one(self, instance: ModelInstance):
        """
        Schedule a model instance by picking one candidate.
        Args:
            item: Model instance to schedule.
        """
        logger.debug(f"Scheduling model instance {instance.name}")

        state_message = ""

        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session)
            if len(workers) == 0:
                state_message = "No available workers"

            model = await Model.one_by_id(session, instance.model_id)
            if model is None:
                state_message = "Model not found"

            candidate = None
            messages = []
            if workers and model:
                candidate, messages = await self.find_candidate(
                    instance, model, workers
                )

            model_instance = await ModelInstance.one_by_id(session, instance.id)
            if candidate is None:
                # update model instance.
                if model_instance.state in (
                    ModelInstanceStateEnum.SCHEDULED,
                    ModelInstanceStateEnum.ANALYZING,
                ):
                    model_instance.state = ModelInstanceStateEnum.PENDING
                    model_instance.state_message = (
                        "No suitable workers.\n"
                        "Details:\n" + "\n".join(f"- {msg}" for msg in messages)
                    )
                if state_message != "":
                    model_instance.state_message = state_message

                await model_instance.update(session, model_instance)
                logger.debug(
                    f"No suitable workers for model instance {model_instance.name}, state: {model_instance.state}"
                )
            else:
                # update model instance.
                model_instance.state = ModelInstanceStateEnum.SCHEDULED
                model_instance.state_message = ""
                model_instance.worker_id = candidate.worker.id
                model_instance.worker_name = candidate.worker.name
                model_instance.worker_ip = candidate.worker.ip
                model_instance.computed_resource_claim = (
                    candidate.computed_resource_claim
                )
                model_instance.gpu_indexes = candidate.gpu_indexes
                model_instance.distributed_servers = DistributedServers(
                    rpc_servers=candidate.rpc_servers
                )

                await model_instance.update(session, model_instance)

                logger.debug(
                    f"Scheduled model instance {model_instance.name} to worker "
                    f"{model_instance.worker_name} gpu {candidate.gpu_indexes}"
                )

    def pick_highest_score_candidate(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ):
        """
        Pick the most offload layers from candidates.
        Args:
            candidates: List of ModelInstanceScheduleCandidate.
        """

        logger.debug(f"Pick highest score candidate from {len(candidates)} candidates")

        if len(candidates) == 0:
            return None

        candidate = candidates[0]
        for i in range(1, len(candidates)):
            if candidates[i].score > candidate.score:
                candidate = candidates[i]

        return candidate
