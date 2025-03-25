import asyncio
import logging
from typing import List, Tuple
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.config.config import Config
from gpustack.policies.base import Worker
from gpustack.scheduler import scheduler
from gpustack.server.catalog import model_set_specs_by_key
from gpustack.schemas.model_evaluations import ModelEvaluationResult, ModelSpec

from gpustack.schemas.models import (
    BackendEnum,
    SourceEnum,
    get_backend,
    is_audio_model,
    is_gguf_model,
)
from gpustack.utils.hub import (
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
)
from gpustack.utils.task import run_in_thread
from gpustack.utils.profiling import time_decorator

logger = logging.getLogger(__name__)


@time_decorator
async def evaluate_models(
    config: Config, session: AsyncSession, model_specs: List[ModelSpec]
) -> List[ModelEvaluationResult]:
    """
    Evaluate the compatibility of a list of model specs with the available workers.
    """
    workers = await Worker.all(session)
    async with asyncio.TaskGroup() as tg:
        tasks = {
            tg.create_task(evaluate_model(config, model, workers)): model
            for model in model_specs
        }
    return [task.result() for task in tasks]


@time_decorator
async def evaluate_model(
    config: Config,
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
        (evaluate_environment, (model, workers)),
        (evaluate_model_metadata, (config, model)),
    ]
    for evaluation, args in evaluations:
        compatible, messages = await evaluation(*args)
        if not compatible:
            result.compatible = False
            result.compatibility_messages = make_compatibility_messages_user_friendly(
                messages
            )
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

    return result


def make_compatibility_messages_user_friendly(messages: List[str]) -> List[str]:
    """
    Convert a compatibility message to a user-friendly format.
    """
    for i, message in enumerate(messages):
        if "option `trust_remote_code=True`" in message:
            messages[i] = message.replace(
                "option `trust_remote_code=True`",
                "backend parameter `--trust-remote-code`",
            )
    return messages


async def set_gguf_model_file_path(config: Config, model: ModelSpec):
    if (
        model.source == SourceEnum.HUGGING_FACE
        and "GGUF" in model.huggingface_repo_id.upper()
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
        and "GGUF" in model.model_scope_model_id.upper()
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
    has_linux_workers = any(worker.labels["os"] == "linux" for worker in workers)
    if get_backend(model) == BackendEnum.VLLM and not has_linux_workers:
        return False, [
            "The model requires Linux workers but none are available. Use GGUF models instead."
        ]

    return True, []


async def evaluate_model_metadata(
    config: Config,
    model: ModelSpec,
) -> Tuple[bool, List[str]]:
    try:
        if is_gguf_model(model):
            await scheduler.evaluate_gguf_model(config, model)
        elif is_audio_model(model):
            await scheduler.evaluate_audio_model(config, model)
        else:
            await scheduler.evaluate_pretrained_config(model)
    except Exception as e:
        return False, [str(e)]

    return True, []


def get_catalog_model_spec_backend_parameter(model: ModelSpec) -> List[str]:
    model_spec_in_catalog = model_set_specs_by_key.get(model.model_source_key)
    if model_spec_in_catalog:
        return model_spec_in_catalog.backend_parameters
    return []
