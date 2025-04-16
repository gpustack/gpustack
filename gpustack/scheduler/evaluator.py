import asyncio
import hashlib
import json
import logging
import os
from typing import List, Tuple
from sqlmodel.ext.asyncio.session import AsyncSession
from cachetools import TTLCache

from gpustack.api.exceptions import HTTPException
from gpustack.config.config import Config, VendorEnum
from gpustack.policies.base import ModelInstanceScheduleCandidate, Worker
from gpustack.routes.models import validate_model_in
from gpustack.scheduler import scheduler
from gpustack.server.catalog import model_set_specs_by_key
from gpustack.schemas.model_evaluations import (
    ModelEvaluationResult,
    ModelSpec,
    ResourceClaim,
)

from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    SourceEnum,
    get_backend,
    is_audio_model,
    is_gguf_model,
)
from gpustack.utils.hub import (
    auth_check,
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
)
from gpustack.utils.task import run_in_thread
from gpustack.utils.profiling import time_decorator

logger = logging.getLogger(__name__)

EVALUATION_CACHE_MAX_SIZE = int(
    os.environ.get("GPUSTACK_MODEL_EVALUATION_CACHE_MAX_SIZE", 1000)
)
EVALUATION_CACHE_TTL = int(os.environ.get("GPUSTACK_MODEL_EVALUATION_CACHE_TTL", 3600))
evaluate_cache = TTLCache(maxsize=EVALUATION_CACHE_MAX_SIZE, ttl=EVALUATION_CACHE_TTL)


@time_decorator
async def evaluate_models(
    config: Config, session: AsyncSession, model_specs: List[ModelSpec]
) -> List[ModelEvaluationResult]:
    """
    Evaluate the compatibility of a list of model specs with the available workers.
    """
    workers = await Worker.all(session)

    async def evaluate(model: ModelSpec):
        return await evaluate_model_with_cache(config, session, model, workers)

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
) -> ModelEvaluationResult:
    cache_key = make_hashable_key(model, workers)
    if cache_key in evaluate_cache:
        logger.trace(
            f"Evaluation cache hit for model: {model.name or model.readable_source}"
        )
        return evaluate_cache[cache_key]

    try:
        result = await evaluate_model(config, session, model, workers)
        evaluate_cache[cache_key] = result
    except Exception as e:
        logger.error(
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
) -> ModelEvaluationResult:
    result = ModelEvaluationResult()

    default_backend_parameters = get_catalog_model_spec_backend_parameter(model)
    if default_backend_parameters and not model.backend_parameters:
        model.backend_parameters = default_backend_parameters
        result.default_spec = model.model_copy()

    await set_gguf_model_file_path(config, model)

    evaluations = [
        (evaluate_model_input, (session, model)),
        (evaluate_model_metadata, (config, model)),
        (evaluate_environment, (model, workers)),
    ]
    for evaluation, args in evaluations:
        compatible, messages = await evaluation(*args)
        if not compatible:
            result.compatible = False
            result.compatibility_messages = messages
            return result

    candidate, schedule_messages = await scheduler.find_candidate(
        config, model, workers
    )
    if not candidate:
        result.compatible = False
        result.compatibility_messages.append(
            "Unable to find a schedulable worker for the model."
        )
        result.scheduling_messages = schedule_messages
    elif candidate.overcommit:
        result.compatible = False
        result.compatibility_messages.append(
            "Selected GPUs may not have enough resources to run the model."
        )
        result.scheduling_messages = schedule_messages
    else:
        result.resource_claim = summarize_candidate_resource_claim(candidate)

    return result


def summarize_candidate_resource_claim(
    candidate: ModelInstanceScheduleCandidate,
) -> ResourceClaim:
    """
    Summarize the computed resource claim for a schedule candidate.
    """
    computed_resource_claims = [candidate.computed_resource_claim]

    if candidate.rpc_servers:
        computed_resource_claims.extend(
            rpc.computed_resource_claim
            for rpc in candidate.rpc_servers
            if rpc.computed_resource_claim is not None
        )

    if candidate.ray_actors:
        computed_resource_claims.extend(
            actor.computed_resource_claim
            for actor in candidate.ray_actors
            if actor.computed_resource_claim is not None
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
        and model.backend == BackendEnum.LLAMA_BOX
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
        and model.backend == BackendEnum.LLAMA_BOX
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
    has_linux_workers = any(worker.labels.get("os") == "linux" for worker in workers)
    if get_backend(model) == BackendEnum.VLLM and not has_linux_workers:
        return False, [
            "The model requires Linux workers but none are available. Use GGUF models instead."
        ]

    only_windows_workers = all(
        worker.labels.get("os") == "windows" for worker in workers
    )
    if (
        only_windows_workers
        and model.backend == BackendEnum.VOX_BOX
        and CategoryEnum.TEXT_TO_SPEECH.value in model.categories
    ):
        return False, ["The model is not supported on Windows workers."]

    if model.backend == BackendEnum.ASCEND_MINDIE and not has_ascend_npu(workers):
        return False, [
            "The MindIE backend requires Ascend NPUs but none are available."
        ]

    return True, []


def has_ascend_npu(workers: List[Worker]) -> bool:
    for worker in workers:
        if worker.status and worker.status.gpu_devices:
            for gpu in worker.status.gpu_devices:
                if gpu.vendor == VendorEnum.Huawei.value:
                    return True
    return False


async def evaluate_model_metadata(
    config: Config,
    model: ModelSpec,
) -> Tuple[bool, List[str]]:
    try:
        await run_in_thread(
            auth_check,
            timeout=15,
            model=model,
            huggingface_token=config.huggingface_token,
        )
        if is_gguf_model(model):
            await scheduler.evaluate_gguf_model(config, model)
        elif is_audio_model(model):
            await scheduler.evaluate_audio_model(config, model)
        else:
            await scheduler.evaluate_pretrained_config(model)
    except ValueError as e:
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


def get_catalog_model_spec_backend_parameter(model: ModelSpec) -> List[str]:
    model_spec_in_catalog = model_set_specs_by_key.get(model.model_source_key)
    if model_spec_in_catalog:
        return model_spec_in_catalog.backend_parameters
    return []
