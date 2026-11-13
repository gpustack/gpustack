from unittest.mock import MagicMock

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.models import validate_model_in
from gpustack.schemas.models import (
    DisaggregationSpec,
    GPUSelector,
    GPUTypeSelector,
    Model,
    ModelCreate,
    ModelUpdate,
    RoleSpec,
)
from gpustack.validation.non_model_workload import validate_non_model_workload


def deployment(**kwargs):
    return ModelCreate(
        name="frontend",
        source="local_path",
        local_path="/models/tokenizer",
        **{
            "backend": "custom-frontend",
            "env": {"GPUSTACK_NON_MODEL_WORKLOAD": "true"},
            **kwargs,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["custom-frontend", "Custom"])
async def test_custom_and_direct_image_deployments_are_valid(backend):
    await validate_model_in(
        MagicMock(),
        deployment(
            backend=backend, image_name="test:frontend", run_command="python -m service"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields,message",
    [
        ({"gpu_selector": GPUSelector(gpu_ids=["missing-worker:cuda:0"])}, "gpu_ids"),
        (
            {"gpu_type_selector": GPUTypeSelector(type="test-slice")},
            "gpu_type_selector",
        ),
        ({"backend": "vLLM"}, "custom backend"),
        ({"roles": [RoleSpec(name="router")]}, "standalone"),
        ({"disaggregation": DisaggregationSpec(mode="vllm-nixl")}, "standalone"),
    ],
)
async def test_conflicts_are_rejected_before_worker_or_backend_lookups(fields, message):
    with pytest.raises(BadRequestException) as exc:
        await validate_model_in(MagicMock(), deployment(**fields))
    assert message in exc.value.message


def test_role_local_flag_cannot_enter_group_resource_policy():
    with pytest.raises(ValueError, match="standalone"):
        validate_non_model_workload(
            deployment(
                env={},
                roles=[
                    RoleSpec(name="decode", env={"GPUSTACK_NON_MODEL_WORKLOAD": "true"})
                ],
            )
        )


@pytest.mark.parametrize("enabled", [None, "false"])
def test_absent_or_disabled_flag_does_not_restrict_existing_gpu_or_pd_config(enabled):
    env = {} if enabled is None else {"GPUSTACK_NON_MODEL_WORKLOAD": enabled}
    validate_non_model_workload(
        deployment(
            backend="vLLM",
            env=env,
            roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
            gpu_selector=GPUSelector(gpu_ids=["worker:cuda:0"]),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["gpu_selector", "gpu_type_selector"])
@pytest.mark.parametrize("update_flag", [False, True])
async def test_sparse_update_checks_saved_settings(field, update_flag):
    selector = (
        GPUSelector(gpu_ids=["worker:cuda:0"])
        if field == "gpu_selector"
        else GPUTypeSelector(type="test-slice")
    )
    base = deployment(
        env={} if update_flag else {"GPUSTACK_NON_MODEL_WORKLOAD": "true"}
    )
    stored = Model(**base.model_dump())
    if update_flag:
        setattr(stored, field, selector)
        update = {"env": {"GPUSTACK_NON_MODEL_WORKLOAD": "true"}}
    else:
        update = {field: selector}
    sparse = ModelUpdate.model_validate(
        {
            "name": "frontend",
            "source": "local_path",
            "local_path": "/models/tokenizer",
            **update,
        }
    )
    with pytest.raises(BadRequestException) as exc:
        await validate_model_in(MagicMock(), sparse, stored=stored)
    assert "not supported" in exc.value.message


@pytest.mark.asyncio
async def test_sparse_update_can_clear_a_conflicting_selector():
    stored = Model(**deployment().model_dump())
    stored.gpu_type_selector = GPUTypeSelector(type="test-slice")
    update = ModelUpdate.model_validate(
        {
            "name": "frontend",
            "source": "local_path",
            "local_path": "/models/tokenizer",
            "backend": "custom-frontend",
            "gpu_type_selector": None,
        }
    )
    await validate_model_in(MagicMock(), update, stored=stored)
