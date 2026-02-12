import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

from gpustack_runtime.detector import ManufacturerEnum
from sqlmodel.ext.asyncio.session import AsyncSession
from cachetools import TTLCache
from aiolimiter import AsyncLimiter

from gpustack.api.exceptions import HTTPException
from gpustack.client.worker_filesystem_client import WorkerFilesystemClient
from gpustack.config.config import Config
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack import envs
from gpustack.routes.models import validate_model_in
from gpustack.scheduler import scheduler
from gpustack.server.catalog import model_set_specs_by_key
from gpustack.schemas.model_evaluations import (
    ModelEvaluationResult,
    ModelSpec,
    ResourceClaim,
)
from gpustack.schemas.models import (
    ModelInstance,
    BackendEnum,
    SourceEnum,
    get_backend,
    is_gguf_model,
    is_audio_model,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.worker_selector import WorkerSelector

from gpustack.utils.gpu import (
    all_gpu_match,
    any_gpu_match,
    find_one_gpu,
    compare_compute_capability,
)
from gpustack.utils.hub import (
    auth_check,
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
    is_repo_cached,
)
from gpustack.utils.task import run_in_thread
from gpustack.utils.profiling import time_decorator

logger = logging.getLogger(__name__)

evaluate_cache = TTLCache(
    maxsize=envs.MODEL_EVALUATION_CACHE_MAX_SIZE, ttl=envs.MODEL_EVALUATION_CACHE_TTL
)

# To reduce the likelihood of hitting the Hugging Face API rate limit (600 RPM)
# Limit the number of concurrent evaluations to 50 per 10 seconds
evaluate_model_limiter = AsyncLimiter(50, 10)


@time_decorator
async def evaluate_models(
    config: Config,
    session: AsyncSession,
    model_specs: List[ModelSpec],
    cluster_id: Optional[int] = None,
) -> List[ModelEvaluationResult]:
    """
    Evaluate the compatibility of a list of model specs with the available workers.
    """
    fields = {
        "deleted_at": None,
    }
    if cluster_id is not None:
        fields["cluster_id"] = cluster_id
    extra_conditions = [
        ~(
            Worker.state.in_(
                [
                    WorkerStateEnum.PROVISIONING,
                    WorkerStateEnum.DELETING,
                    WorkerStateEnum.ERROR,
                ]
            )
        )
    ]
    workers = await Worker.all_by_fields(
        session, fields=fields, extra_conditions=extra_conditions
    )

    model_instances = await ModelInstance.all_by_fields(session, fields=fields)

    if len(model_specs) == 1:
        # Sort worker for single-model evaluation only. No need for batch evaluation.
        workers = await scheduler.prioritize_workers_with_model_files(
            session, model_specs[0], workers
        )

    async def evaluate(model: ModelSpec):
        return await evaluate_model_with_cache(
            config, session, model, workers, model_instances
        )

    tasks = [evaluate(model) for model in model_specs]
    results = await asyncio.gather(*tasks)
    return results


def make_hashable_key(model: ModelSpec, workers: List[Worker]) -> str:
    key_data = json.dumps(
        {
            "model": model.model_dump(mode="json"),
            "workers": [
                w.model_dump(
                    mode="json",
                    exclude={
                        "status": {
                            "cpu": True,
                            "swap": True,
                            "filesystem": True,
                            "os": True,
                            "kernel": True,
                            "uptime": True,
                            "memory": {"utilization_rate", "used"},
                            "gpu_devices": {
                                "__all__": {
                                    "temperature": True,
                                    "core": {"utilization_rate"},
                                    "memory": {"utilization_rate", "used"},
                                },
                            },
                        },
                        "heartbeat_time": True,
                        "created_at": True,
                        "updated_at": True,
                    },
                )
                for w in workers
            ],
        },
        sort_keys=True,
    )
    return hashlib.md5(key_data.encode()).hexdigest()


async def evaluate_model_with_cache(
    config: Config,
    session: AsyncSession,
    model: ModelSpec,
    workers: List[Worker],
    model_instances: List[ModelInstance],
) -> ModelEvaluationResult:
    cache_key = make_hashable_key(model, workers)
    if cache_key in evaluate_cache:
        logger.trace(
            f"Evaluation cache hit for model: {model.name or model.readable_source}"
        )
        return evaluate_cache[cache_key]

    try:
        async with evaluate_model_limiter:
            result = await evaluate_model(
                config, session, model, workers, model_instances
            )
            evaluate_cache[cache_key] = result
    except Exception as e:
        logger.exception(
            f"Error evaluating model {model.name or model.readable_source}: {e}"
        )
        result = ModelEvaluationResult(
            compatible=False, error=True, error_message=str(e)
        )

    return result


@time_decorator
async def evaluate_model(
    config: Config,
    session: AsyncSession,
    model: ModelSpec,
    workers: List[Worker],
    model_instances: List[ModelInstance],
) -> ModelEvaluationResult:
    result = ModelEvaluationResult()

    if set_default_spec(model):
        result.default_spec = model.model_copy()

    await set_gguf_model_file_path(config, model)

    evaluations = [
        (evaluate_model_input, (session, model)),
        (evaluate_model_metadata, (config, model, workers)),
        (evaluate_environment, (model, workers)),
    ]
    for evaluation, args in evaluations:
        compatible, messages = await evaluation(*args)
        if not compatible:
            result.compatible = False
            result.compatibility_messages = messages
            return result

    workers_by_cluster: Dict[int, List[Worker]] = defaultdict(list)
    for worker in workers:
        workers_by_cluster[worker.cluster_id].append(worker)

    overcommit_clusters = []
    result.resource_claim_by_cluster_id = {}

    for cluster_id, cluster_workers in workers_by_cluster.items():
        cluster_model_instances = [
            inst for inst in model_instances if inst.cluster_id == cluster_id
        ]
        candidate, schedule_messages = await scheduler.find_candidate(
            config, model, cluster_workers, cluster_model_instances
        )
        if not candidate:
            result.scheduling_messages.extend(schedule_messages)
            continue
        if candidate.overcommit:
            overcommit_clusters.append(cluster_id)
            result.scheduling_messages.extend(schedule_messages)
            continue
        result.resource_claim_by_cluster_id[cluster_id] = (
            summarize_candidate_resource_claim(candidate)
        )

    if result.resource_claim_by_cluster_id:
        result.resource_claim = next(iter(result.resource_claim_by_cluster_id.values()))
    else:
        result.resource_claim = None
        result.compatible = False
        result.compatibility_messages.append(
            "Unable to find a schedulable worker for the model."
        )
    return result


def summarize_candidate_resource_claim(
    candidate: ModelInstanceScheduleCandidate,
) -> ResourceClaim:
    """
    Summarize the computed resource claim for a schedule candidate.
    """
    computed_resource_claims = [candidate.computed_resource_claim]

    if candidate.subordinate_workers:
        computed_resource_claims.extend(
            sw.computed_resource_claim
            for sw in candidate.subordinate_workers
            if sw.computed_resource_claim is not None
        )

    ram, vram = 0, 0
    for computed_resource_claim in computed_resource_claims:
        ram += computed_resource_claim.ram or 0
        if computed_resource_claim.vram:
            vram += sum(
                v for v in computed_resource_claim.vram.values() if v is not None
            )

    return ResourceClaim(ram=ram, vram=vram)


async def set_gguf_model_file_path(config: Config, model: ModelSpec):
    if (
        model.source == SourceEnum.HUGGING_FACE
        and "gguf" in model.huggingface_repo_id.lower()
        and not model.huggingface_filename
    ):
        model.huggingface_filename = await run_in_thread(
            get_hugging_face_model_min_gguf_path,
            timeout=15,
            model_id=model.huggingface_repo_id,
            token=config.huggingface_token,
        )
    elif (
        model.source == SourceEnum.MODEL_SCOPE
        and "gguf" in model.model_scope_model_id.lower()
        and not model.model_scope_file_path
    ):
        model.model_scope_file_path = await run_in_thread(
            get_model_scope_model_min_gguf_path,
            timeout=15,
            model_id=model.model_scope_model_id,
        )


async def evaluate_environment(
    model: ModelSpec,
    workers: List[Worker],
) -> Tuple[bool, List[str]]:
    backend = get_backend(model)

    if backend == BackendEnum.ASCEND_MINDIE and not any_gpu_match(
        workers, lambda gpu: gpu.vendor == ManufacturerEnum.ASCEND.value
    ):
        return False, [
            "The Ascend MindIE backend requires Ascend NPUs but none are available."
        ]

    if (
        backend == BackendEnum.SGLANG
        and all_gpu_match(
            workers, lambda gpu: gpu.vendor == ManufacturerEnum.NVIDIA.value
        )
        and not any_gpu_match(
            workers,
            lambda gpu: compare_compute_capability(gpu.compute_capability, "8.0") >= 0,
        )
    ):
        # Ref: https://github.com/sgl-project/sglang/issues/6006
        gpu = find_one_gpu(workers)
        return False, [
            "The SGLang backend requires NVIDIA GPUs with compute capability 8.0 or higher "
            "(e.g., A100/SM80, H100/SM90, RTX 3090/SM86). "
            + (
                f"Available GPU: {gpu.name} (compute capability: {gpu.compute_capability})"
                if gpu
                else ""
            )
        ]

    return True, []


async def evaluate_model_metadata(
    config: Config,
    model: ModelSpec,
    workers: List[Worker],
) -> Tuple[bool, List[str]]:
    try:
        if model.source == SourceEnum.LOCAL_PATH:
            # Check if local path exists on server
            path_exists_on_server = os.path.exists(model.local_path)

            if not path_exists_on_server:
                # Try to check if path exists on any worker
                try:
                    async with WorkerFilesystemClient() as filesystem_client:
                        selector = WorkerSelector(filesystem_client)

                        found_worker = await selector.find_worker_with_path(
                            workers, path=model.local_path
                        )

                        if found_worker:
                            logger.info(
                                f"Found path {model.local_path} on worker {found_worker.id}"
                            )
                        else:
                            # Path not found on any worker
                            return False, [
                                "The model file path you specified does not exist."
                                "Please ensure the model file is accessible from at least one node."
                            ]
                except Exception as e:
                    logger.warning(
                        f"Failed to check path on workers: {e}, falling back to local check"
                    )
                    # Fallback to original warning
                    return False, [
                        "Failed to get model metadata. The model file path you specified does not exist."
                    ]

        if model.source in [
            SourceEnum.HUGGING_FACE,
            SourceEnum.MODEL_SCOPE,
        ]:
            repo_id = model.huggingface_repo_id
            if model.source == SourceEnum.MODEL_SCOPE:
                repo_id = model.model_scope_model_id
            if not is_repo_cached(repo_id, model.source):
                await run_in_thread(
                    auth_check,
                    timeout=15,
                    model=model,
                    huggingface_token=config.huggingface_token,
                )

        if is_gguf_model(model):
            await scheduler.evaluate_gguf_model(model, workers=workers)
        elif model.backend == BackendEnum.VOX_BOX:
            await scheduler.evaluate_vox_box_model(config, model)
        elif not is_audio_model(model):
            await scheduler.evaluate_pretrained_config(model, workers=workers)
    except Exception as e:
        if model.env and model.env.get("GPUSTACK_SKIP_MODEL_EVALUATION"):
            logger.warning(f"Ignore model evaluation error for model {model.name}: {e}")
            return True, []

        return False, [str(e)]

    return True, []


async def evaluate_model_input(
    session: AsyncSession,
    model: ModelSpec,
) -> Tuple[bool, List[str]]:
    try:
        await validate_model_in(session, model)
    except HTTPException as e:
        return False, [e.message]
    except Exception as e:
        return False, [str(e)]

    return True, []


def set_default_spec(model: ModelSpec) -> bool:
    """
    Set the default spec for the model if it matches the catalog spec.
    """
    model_spec_in_catalog = model_set_specs_by_key.get(model.model_source_key)

    modified = False
    if model_spec_in_catalog:
        if (
            model_spec_in_catalog.backend_parameters
            and model.backend_parameters is None
        ):
            model.backend_parameters = model_spec_in_catalog.backend_parameters
            modified = True

        if model_spec_in_catalog.env and model.env is None:
            model.env = model_spec_in_catalog.env
            modified = True

        if model_spec_in_catalog.categories and not model.categories:
            model.categories = model_spec_in_catalog.categories
            modified = True

    gpus_per_replica_modified = scheduler.set_model_gpus_per_replica(model)
    return modified or gpus_per_replica_modified
