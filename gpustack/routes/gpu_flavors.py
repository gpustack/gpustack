from fastapi import APIRouter

from gpustack.schemas.model_evaluations import ModelEvaluationRequest
from gpustack.server.deps import SessionDep
from gpustack.schemas.gpu_flavors import GPUFlavorsResponse
from gpustack.schemas.workers import Worker
from gpustack.utils.hub import get_pretrained_config
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    get_model_num_attention_heads,
)

router = APIRouter()


@router.get("", response_model=GPUFlavorsResponse)
async def get_gpu_flavors(  # noqa: C901
    session: SessionDep,
    model_evaluation_in: ModelEvaluationRequest,
):
    """
    Get GPU flavors by scanning all available workers.
    Returns a type_map where key is GPU type + VRAM size and value is count.
    """
    # Get all workers
    workers = await Worker.all(session)

    type_map = {}

    for worker in workers:
        # Skip workers that are not ready or don't have GPU devices
        if not worker.status or not worker.status.gpu_devices:
            continue

        for gpu in worker.status.gpu_devices:
            # Count the GPU
            if gpu.flavor_name in type_map:
                type_map[gpu.flavor_name] += 1
            else:
                type_map[gpu.flavor_name] = 1

    model_specs = model_evaluation_in.model_specs
    if len(model_specs) == 0:
        return GPUFlavorsResponse(type_map=type_map)

    model = model_specs[0]
    # Get model's attention head count
    attention_head_num = None
    try:
        pretrained_config = get_pretrained_config(model, trust_remote_code=True)
        attention_head_num = get_model_num_attention_heads(pretrained_config)
    except Exception:
        # If we can't get attention heads, default to allowing any GPU count
        attention_head_num = None

    # Calculate allow_gpu_count based on attention heads
    allow_gpu_count = []
    if attention_head_num and attention_head_num > 0:
        # Find divisors of attention_head_num up to a reasonable limit (e.g., 32 GPUs)
        for gpu_count in range(1, attention_head_num + 1):
            if attention_head_num % gpu_count == 0:
                allow_gpu_count.append(gpu_count)
    return GPUFlavorsResponse(type_map=type_map, allow_gpu_count=allow_gpu_count)
