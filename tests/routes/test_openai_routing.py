import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.models import validate_model_in
from gpustack.schemas.models import ModelCreate
from tests.utils.model import new_model


# ---------------------------------------------------------------------------
# validate_model_in — manual-distributed gate (routes/models.py)
# ---------------------------------------------------------------------------


def _manual_distributed_model_create(enable_model_route):
    base = new_model(
        1,
        "manual-dp",
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        env={"GPUSTACK_MANUAL_DISTRIBUTED": "1"},
    )
    model_in = ModelCreate.model_validate(base.model_dump())
    model_in.enable_model_route = enable_model_route
    return model_in


@pytest.mark.asyncio
async def test_validate_manual_distributed_requires_route_disabled():
    # enable_model_route defaults to None (not False): manual-distributed must be
    # rejected, since it registers no upstream and the route would be empty.
    # No gpu_selector, so validate_gpu_ids is skipped and session is unused.
    model_in = _manual_distributed_model_create(enable_model_route=None)
    with pytest.raises(BadRequestException) as exc_info:
        await validate_model_in(session=None, model_in=model_in)
    assert "enable_model_route" in exc_info.value.message


@pytest.mark.asyncio
async def test_validate_manual_distributed_route_disabled_ok():
    model_in = _manual_distributed_model_create(enable_model_route=False)
    # Passes the manual-distributed gate; remaining validation needs no session.
    await validate_model_in(session=None, model_in=model_in)


@pytest.mark.asyncio
async def test_validate_non_manual_distributed_route_enabled_ok():
    # A normal model with the route enabled is unaffected by the manual gate.
    base = new_model(1, "normal", huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model_in = ModelCreate.model_validate(base.model_dump())
    model_in.enable_model_route = True
    await validate_model_in(session=None, model_in=model_in)


@pytest.mark.asyncio
async def test_validate_manual_dp_params_without_opt_in_rejected():
    # Hand-wired vLLM DP params but no GPUSTACK_MANUAL_DISTRIBUTED: reject, since
    # GPUStack would auto-orchestrate and the user's params would break it.
    base = new_model(
        1,
        "manual-dp-params",
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend_parameters=["--data-parallel-address=1.2.3.4"],
    )
    model_in = ModelCreate.model_validate(base.model_dump())
    with pytest.raises(BadRequestException) as exc_info:
        await validate_model_in(session=None, model_in=model_in)
    assert "GPUSTACK_MANUAL_DISTRIBUTED" in exc_info.value.message


@pytest.mark.asyncio
async def test_validate_manual_dp_params_with_opt_in_ok():
    # The opt-in env (plus enable_model_route=false) clears the gate.
    base = new_model(
        1,
        "manual-dp-params",
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend_parameters=["--data-parallel-address=1.2.3.4"],
        env={"GPUSTACK_MANUAL_DISTRIBUTED": "1"},
    )
    model_in = ModelCreate.model_validate(base.model_dump())
    model_in.enable_model_route = False
    await validate_model_in(session=None, model_in=model_in)
