"""
分布式调度器模块

提供多Server环境下的分布式调度功能，包括：
- 分布式锁管理
- 调度协调
- 跨Server状态同步
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import queue
from typing import List, Tuple, Optional
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.config.config import Config, get_global_config
from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    WorkerFilterChain,
)
from gpustack.policies.candidate_selectors import (
    AscendMindIEResourceFitSelector,
    GGUFResourceFitSelector,
    SGLangResourceFitSelector,
    VLLMResourceFitSelector,
    VoxBoxResourceFitSelector,
)
from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (
    CustomBackendResourceFitSelector,
)
from gpustack.policies.utils import ListMessageBuilder
from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
)
from gpustack.policies.worker_filters.label_matching_filter import LabelMatchingFilter
from gpustack.policies.worker_filters.gpu_matching_filter import GPUMatchingFilter
from gpustack.policies.worker_filters.cluster_filter import ClusterFilter
from gpustack.scheduler.model_registry import detect_model_type
from gpustack.scheduler.queue import AsyncUniqueQueue
from gpustack.policies.worker_filters.status_filter import StatusFilter
from gpustack.schemas.inference_backend import is_built_in_backend
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
    DistributedServerCoordinateModeEnum,
)
from gpustack.server.bus import EventType
from gpustack.server.db import async_session
from gpustack.scheduler.calculator import (
    GPUOffloadEnum,
    calculate_model_resource_claim,
)
from gpustack.server.services import ModelInstanceService, ModelService
from gpustack.utils.command import find_parameter
from gpustack.utils.gpu import group_gpu_ids_by_worker
from gpustack.utils.hub import (
    get_pretrained_config_with_fallback,
    has_diffusers_model_index,
)
from gpustack.utils.math import largest_power_of_2_leq
from sqlalchemy.orm.attributes import flag_modified

logger = logging.getLogger(__name__)


class DistributedScheduler:
    """
    分布式调度器
    
    在多Server环境下提供协调的模型实例调度功能。
    通过分布式锁防止重复调度，并通过协调服务实现跨Server的调度决策。
    """
    
    def __init__(self, cfg: Config, check_interval: int = 180):
        """
        初始化分布式调度器
        
        Args:
            cfg: GPUStack配置对象
            check_interval: 检查间隔（秒）
        """
        self._id = "model-instance-scheduler"
        self._config = cfg
        self._check_interval = check_interval
        self._queue = AsyncUniqueQueue()
        self._cache_dir = None
        self._coordinator = None
        self._server_id = None
        self._scheduling_mode = cfg.scheduling_mode or "auto"
        
        if self._config.cache_dir is not None:
            self._cache_dir = os.path.join(self._config.cache_dir, "gguf-parser")
            os.makedirs(self._cache_dir, exist_ok=True)
            
            self._vox_box_cache_dir = os.path.join(self._config.cache_dir, "vox-box")
            os.makedirs(self._vox_box_cache_dir, exist_ok=True)
    
    def set_coordinator(self, coordinator, server_id: str):
        """
        设置协调服务
        
        Args:
            coordinator: 协调服务实例
            server_id: 当前Server ID
        """
        self._coordinator = coordinator
        self._server_id = server_id
        logger.info(f"DistributedScheduler configured with coordinator, Server ID: {server_id}")
    
    async def start(self):
        """启动调度器"""
        try:
            # 调度队列任务
            asyncio.create_task(self._schedule_cycle())
            
            # 定时触发调度任务
            trigger = IntervalTrigger(
                seconds=self._check_interval, timezone=timezone.utc
            )
            scheduler = AsyncIOScheduler(timezone=timezone.utc)
            scheduler.add_job(
                self._enqueue_pending_instances,
                trigger=trigger,
                id=self._id,
                max_instances=1,
            )
            scheduler.start()
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
        
        logger.info("Distributed scheduler started.")
        
        # 通过事件触发调度
        async for event in ModelInstance.subscribe(source="scheduler"):
            if event.type != EventType.CREATED:
                continue
            
            await self._enqueue_pending_instances()
    
    async def _enqueue_pending_instances(self):
        """将待调度的模型实例加入队列"""
        try:
            async with async_session() as session:
                instances = await ModelInstance.all(session)
                tasks = []
                for instance in instances:
                    if self._should_schedule(instance):
                        task = asyncio.create_task(self._evaluate(instance))
                        tasks.append(task)
                
                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Failed to enqueue pending model instances: {e}")
    
    async def _evaluate(self, instance: ModelInstance):
        """评估模型实例的资源需求
        
        Evaluate resource requirements for model instance
        
        修复：确保该函数作为类方法正确缩进，以便访问类实例变量如 self._queue
        Fix: Ensure this function is properly indented as a class method to access class instance variables like self._queue
        """
        async with async_session() as session:
            try:
                instance = await ModelInstance.one_by_id(session, instance.id)
                
                model = await Model.one_by_id(session, instance.model_id)
                if model is None:
                    raise Exception("Model not found.")
                
                if instance.state != ModelInstanceStateEnum.ANALYZING:
                    instance.state = ModelInstanceStateEnum.ANALYZING
                    instance.state_message = "Evaluating resource requirements"
                    await ModelInstanceService(session).update(instance)
                
                if model.source == SourceEnum.LOCAL_PATH and not os.path.exists(
                    model.local_path
                ):
                    # 本地路径模型无法从Server访问，跳过评估
                    # Local path model cannot be accessed from Server, skip evaluation
                    await self._queue.put(instance)
                    return
                
                should_update_model = False
                try:
                    if is_gguf_model(model):
                        should_update_model = await self._evaluate_gguf_model(model)
                        if await self.check_model_distributability(session, model, instance):
                            return
                    elif model.backend == BackendEnum.VOX_BOX:
                        should_update_model = await self._evaluate_vox_box_model(model)
                    else:
                        should_update_model = await self._evaluate_pretrained_config(
                            model, session=session, raise_raw=True,
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to evaluate model {model.name or model.readable_source}: {e}"
                    )
                
                if should_update_model:
                    await ModelService(session).update(model)
                
                await self._queue.put(instance)
            except Exception as e:
                try:
                    instance.state = ModelInstanceStateEnum.ERROR
                    instance.state_message = str(e)
                    await ModelInstanceService(session).update(instance)
                except Exception as ue:
                    logger.error(
                        f"Failed to update model instance: {ue}. Original error: {e}"
                    )
    
    async def check_model_distributability(
        self, session: AsyncSession, model: Model, instance: ModelInstance
    ):
        """检查模型是否可分布式部署"""
        if (
            not model.distributable
            and model.gpu_selector
            and model.gpu_selector.gpu_ids
        ):
            worker_gpu_ids = group_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            if len(worker_gpu_ids) > 1:
                instance.state = ModelInstanceStateEnum.ERROR
                instance.state_message = (
                    "The model is not distributable to multiple workers."
                )
                await ModelInstanceService(session).update(instance)
                return True
        return False
    
    def _should_schedule(self, instance: ModelInstance) -> bool:
        """检查是否应该调度该实例"""
        newly_created = (instance.updated_at - instance.created_at) < timedelta(
            seconds=1
        )
        update_delta = datetime.now(timezone.utc) - instance.updated_at.replace(
            tzinfo=timezone.utc
        )
        return (
            (
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.PENDING
                and (newly_created or update_delta > timedelta(seconds=90))
            )
            or (
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.ANALYZING
                and update_delta > timedelta(minutes=3)
            )
            or (
                instance.worker_id is not None
                and instance.state == ModelInstanceStateEnum.SCHEDULED
                and update_delta > timedelta(minutes=3)
            )
        )
    
    async def _schedule_cycle(self):
        """调度循环"""
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
    
    async def _schedule_one(self, instance: ModelInstance):
        """
        调度单个模型实例
        
        通过分布式锁确保同一实例不会被多个Server同时调度
        """
        logger.debug(f"Scheduling model instance {instance.name}")
        
        # 确定调度模式
        use_distributed = (
            self._config.distributed_scheduling
            and self._coordinator
            and self._server_id
            and self._scheduling_mode in ["distributed", "auto"]
        )
        
        if use_distributed:
            # 尝试获取分布式锁
            lock_id = f"schedule_lock_{instance.id}"
            lock_timeout = getattr(self._config, 'schedule_lock_timeout', 60)
            
            acquired = await self._coordinator.acquire_lock(
                lock_id,
                self._server_id,
                timeout=lock_timeout
            )
            
            if not acquired:
                logger.debug(
                    f"Could not acquire schedule lock for instance {instance.id}, "
                    "another server may be scheduling it"
                )
                # 重新入队，稍后重试
                await asyncio.sleep(5)
                await self._queue.put(instance)
                return
            
            logger.debug(f"Acquired schedule lock for instance {instance.id}")
            
            try:
                await self._do_schedule(instance)
            finally:
                # 释放锁
                await self._coordinator.release_lock(lock_id, self._server_id)
                logger.debug(f"Released schedule lock for instance {instance.id}")
        else:
            # 回退到本地调度
            await self._do_schedule(instance)
    
    async def _do_schedule(self, instance: ModelInstance):
        """执行实际的调度逻辑"""
        logger.debug(f"Executing scheduling for model instance {instance.name}")
        
        state_message = ""
        
        async with async_session() as session:
            workers = await Worker.all(session)
            if len(workers) == 0:
                state_message = "No available workers"
            
            model = await Model.one_by_id(session, instance.model_id)
            if model is None:
                state_message = "Model not found"
            
            model_instance = await ModelInstance.one_by_id(session, instance.id)
            if model_instance is None:
                logger.debug(
                    f"Model instance(ID: {instance.id}) was deleted before scheduling"
                )
                return
            
            model_instances = await ModelInstance.all(
                session, options=[selectinload(ModelInstance.model)]
            )
            
            candidate = None
            messages = []
            if workers and model:
                try:
                    candidate, messages = await find_candidate(
                        self._config, model, workers, model_instances
                    )
                except Exception as e:
                    state_message = f"Failed to find candidate: {e}"
            
            if candidate is None:
                # 更新模型实例状态
                if model_instance.state in (
                    ModelInstanceStateEnum.SCHEDULED,
                    ModelInstanceStateEnum.ANALYZING,
                ):
                    model_instance.state = ModelInstanceStateEnum.PENDING
                    model_instance.state_message = (
                        "No suitable workers.\nDetails:\n" + "".join(messages)
                    )
                if state_message != "":
                    model_instance.state_message = state_message
                
                await ModelInstanceService(session).update(model_instance)
                logger.debug(
                    f"No suitable workers for model instance {model_instance.name}, state: {model_instance.state}"
                )
            else:
                # 更新模型实例状态
                model_instance.state = ModelInstanceStateEnum.SCHEDULED
                model_instance.state_message = ""
                model_instance.worker_id = candidate.worker.id
                model_instance.worker_name = candidate.worker.name
                model_instance.worker_ip = candidate.worker.ip
                model_instance.worker_advertise_address = (
                    candidate.worker.advertise_address
                )
                model_instance.worker_ifname = candidate.worker.ifname
                model_instance.computed_resource_claim = (
                    candidate.computed_resource_claim
                )
                model_instance.gpu_type = candidate.gpu_type
                model_instance.gpu_indexes = candidate.gpu_indexes
                model_instance.gpu_addresses = candidate.gpu_addresses
                model_instance.distributed_servers = DistributedServers(
                    subordinate_workers=candidate.subordinate_workers,
                )
                if get_backend(model) in (
                    BackendEnum.VLLM,
                    BackendEnum.ASCEND_MINDIE,
                    BackendEnum.SGLANG,
                ):
                    model_instance.distributed_servers.mode = (
                        DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                    )
                
                await ModelInstanceService(session).update(model_instance)
                
                logger.debug(
                    f"Scheduled model instance {model_instance.name} to worker "
                    f"{model_instance.worker_name} gpu {candidate.gpu_indexes}"
                )
    
    async def _evaluate_gguf_model(self, model: Model) -> bool:
        """评估GGUF模型"""
        task_output = await calculate_model_resource_claim(
            model,
            offload=GPUOffloadEnum.Full,
            cache_dir=self._cache_dir,
        )
        if (
            task_output.resource_architecture
            and not task_output.resource_architecture.is_deployable()
        ):
            raise ValueError(
                "Unsupported model. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
            )
        
        should_update = False
        if task_output.resource_claim_estimate.reranking and not model.categories:
            should_update = True
            model.categories = [CategoryEnum.RERANKER]
        
        if task_output.resource_claim_estimate.embeddingOnly and not model.categories:
            should_update = True
            model.categories = [CategoryEnum.EMBEDDING]
        
        if task_output.resource_claim_estimate.imageOnly and not model.categories:
            should_update = True
            model.categories = [CategoryEnum.IMAGE]
        
        if not model.categories:
            should_update = True
            model.categories = [CategoryEnum.LLM]
        
        if task_output.resource_claim_estimate.distributable and not model.distributable:
            should_update = True
            model.distributable = True
        
        if model.gpu_selector and model.gpu_selector.gpu_ids:
            worker_gpu_ids = group_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            if (
                len(worker_gpu_ids) > 1
                and model.distributable
                and not model.distributed_inference_across_workers
            ):
                should_update = True
                model.distributed_inference_across_workers = True
            
            gpus_per_replica_modified = self._set_model_gpus_per_replica(model)
            should_update = should_update or gpus_per_replica_modified
        
        return should_update
    
    def _set_model_gpus_per_replica(self, model: Model) -> bool:
        """设置模型的每副本GPU数量"""
        if not model.gpu_selector or not model.gpu_selector.gpu_ids:
            return False
        
        if model.gpu_selector.gpus_per_replica and model.gpu_selector.gpus_per_replica > 0:
            return False
        
        gpus_per_replica = largest_power_of_2_leq(
            len(model.gpu_selector.gpu_ids) // model.replicas
        )
        model.gpu_selector.gpus_per_replica = gpus_per_replica
        try:
            flag_modified(model, "gpu_selector")
        except AttributeError:
            pass
        return True
    
    async def _evaluate_vox_box_model(self, model: Model) -> bool:
        """评估VoxBox模型"""
        try:
            from vox_box.estimator.estimate import estimate_model
            from vox_box.config import Config as VoxBoxConfig
        except ImportError:
            raise Exception("vox_box is not installed.")
        
        cfg = VoxBoxConfig()
        cfg.cache_dir = os.path.join(self._config.cache_dir, "vox-box")
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
            raise Exception(
                f"Failed to estimate model {model.name or model.readable_source}: {e}"
            )
        
        supported = model_dict.get("supported", False)
        if not supported:
            raise ValueError(
                "Unsupported model. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
            )
        
        should_update = False
        task_type = model_dict.get("task_type")
        if task_type == "tts" and not model.categories:
            model.categories = [CategoryEnum.TEXT_TO_SPEECH]
            should_update = True
        elif task_type == "stt" and not model.categories:
            model.categories = [CategoryEnum.SPEECH_TO_TEXT]
            should_update = True
        
        return should_update
    
    async def _evaluate_pretrained_config(
        self,
        model: Model,
        session: Optional[AsyncSession] = None,
        raise_raw: bool = False,
    ) -> bool:
        """评估预训练模型配置"""
        # 尝试评估扩散模型
        try:
            is_image_category = await self._evaluate_diffusion_model(model)
            if is_image_category:
                return True
        except Exception:
            pass
        
        # 检查vLLM覆盖的架构
        architectures = self._get_vllm_override_architectures(model)
        if not architectures:
            try:
                pretrained_config = await get_pretrained_config_with_fallback(
                    model,
                    session=session,
                )
            except ValueError as e:
                if self._should_skip_architecture_check(model):
                    model.categories = model.categories or [CategoryEnum.LLM]
                    return True
                
                if raise_raw:
                    raise
                
                logger.debug(
                    f"Failed to get config for model {model.name or model.readable_source}, ValueError: {e}"
                )
                raise self._simplify_auto_config_value_error(e)
            except TimeoutError:
                raise Exception(
                    f"Timeout while getting config for model {model.name or model.readable_source}."
                )
            except Exception as e:
                raise Exception(
                    f"Failed to get config for model {model.name or model.readable_source}: {e}"
                )
            
            architectures = getattr(pretrained_config, "architectures", []) or []
            if not architectures and not model.backend_version:
                raise ValueError(
                    "Unrecognized architecture. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
                )
        
        model_type = detect_model_type(architectures)
        
        if (
            model.backend == BackendEnum.VLLM
            and model_type == CategoryEnum.UNKNOWN
            and not model.backend_version
        ):
            raise ValueError(
                f"Unsupported architecture: {architectures}. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
            )
        
        categories_modified = self._set_model_categories(model, model_type)
        gpus_per_replica_modified = self._set_model_gpus_per_replica(model)
        return categories_modified or gpus_per_replica_modified
    
    async def _evaluate_diffusion_model(self, model: Model) -> bool:
        """评估扩散模型"""
        # SGLang now supports Diffusers (image) models.
        if model.backend != BackendEnum.SGLANG or (
            model.categories and CategoryEnum.IMAGE not in model.categories
        ):
            return False
        
        hf_token = get_global_config().huggingface_token
        is_diffusers = await asyncio.wait_for(
            has_diffusers_model_index(model, token=hf_token), timeout=10
        )
        if is_diffusers:
            model.categories = [CategoryEnum.IMAGE]
            return True
        return False
    
    def _get_vllm_override_architectures(self, model: Model) -> List[str]:
        """获取vLLM覆盖的架构"""
        if get_backend(model) != BackendEnum.VLLM:
            return []
        
        hf_overrides = find_parameter(model.backend_parameters, ["hf-overrides"])
        if hf_overrides:
            overrides_dict = json.loads(hf_overrides)
            return overrides_dict.get("architectures", [])
        return []
    
    def _should_skip_architecture_check(self, model: Model) -> bool:
        """检查是否应该跳过架构检查"""
        if (
            model.backend == BackendEnum.CUSTOM
            or not is_built_in_backend(model.backend)
            or model.backend_version
        ):
            return True
        
        if model.backend_parameters and find_parameter(
            model.backend_parameters, ["tokenizer-mode"]
        ):
            return True
        
        return False
    
    def _simplify_auto_config_value_error(self, e: ValueError) -> ValueError:
        """简化自动配置ValueError"""
        message = str(e)
        if "trust_remote_code=True" in message:
            return ValueError(
                "The model contains custom code that must be executed to load correctly. If you trust the source, please pass the backend parameter `--trust-remote-code` to allow custom code to be run."
            )
        
        if "pip install --upgrade transformers" in message:
            return ValueError(
                "Unsupported model. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
            )
        
        return ValueError(f"Not a supported model.\n\n{message}")
    
    def _set_model_categories(self, model: Model, model_type: CategoryEnum) -> bool:
        """设置模型类别"""
        if model.categories:
            return False
        
        if model_type == CategoryEnum.EMBEDDING:
            model.categories = [CategoryEnum.EMBEDDING]
            return True
        elif model_type == CategoryEnum.RERANKER:
            model.categories = [CategoryEnum.RERANKER]
            return True
        elif model_type == CategoryEnum.LLM:
            model.categories = [CategoryEnum.LLM]
            return True
        elif model_type == CategoryEnum.UNKNOWN:
            model.categories = [CategoryEnum.LLM]
            return True
        
        return False


# 从原有调度器导入辅助函数
from gpustack.scheduler.scheduler import (
    find_candidate as _find_candidate,
    pick_highest_score_candidate,
)


async def find_candidate(
    config: Config,
    model: Model,
    workers: List[Worker],
    model_instances: List[ModelInstance],
) -> Tuple[Optional[ModelInstanceScheduleCandidate], List[str]]:
    """查找调度候选"""
    return await _find_candidate(config, model, workers, model_instances)
