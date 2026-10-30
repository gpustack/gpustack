"""Request-boundary validations for model create/update."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from gpustack.api.exceptions import BadRequestException, register_handlers
from gpustack.routes import models
from gpustack.routes import models as models_routes
from gpustack.routes.models import _validate_categories_update
from gpustack.schemas.models import Model, ModelUpdate, ModelCreate, SourceEnum
from gpustack.server.deps import get_session, get_tenant_context


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["31231213", "3-model", "1", "org/model", "-model", "model_", "model\n"]
)
async def test_invalid_auto_route_name_returns_422_before_persistence(
    monkeypatch, name
):
    session = MagicMock(commit=AsyncMock(), rollback=AsyncMock())
    ctx = SimpleNamespace(current_principal_id=1)
    persist = AsyncMock(side_effect=AssertionError("must not persist invalid input"))
    monkeypatch.setattr(models_routes, "_persist_model_create", persist)
    monkeypatch.setattr(models_routes, "_check_model_create", AsyncMock())
    monkeypatch.setattr(
        models_routes, "_resolve_target_org", AsyncMock(return_value=(1, None))
    )
    app = FastAPI()
    app.include_router(models_routes.router, prefix="/v2/models")
    register_handlers(app)
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_tenant_context] = lambda: ctx

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v2/models",
            json={
                "name": name,
                "source": "local_path",
                "local_path": "/models/qwen",
                "enable_model_route": True,
            },
        )

    assert response.status_code == 422
    assert "name must start with a letter," in response.json()["message"]
    persist.assert_not_awaited()
    session.commit.assert_not_awaited()


def _update(categories) -> ModelUpdate:
    # model_validate (not plain construction) so ``model_fields_set`` says
    # the field was explicitly supplied, the way a request body does.
    return ModelUpdate.model_validate(_payload(categories))


def _payload(categories) -> dict:
    return {
        "name": "m",
        "source": "huggingface",
        "huggingface_repo_id": "x/y",
        "categories": categories,
    }


def _stored(categories) -> Model:
    return Model(id=1, name="m", categories=categories)


@pytest.mark.parametrize(
    "update_categories,stored_categories",
    [
        # An unchanged echo passes — including the legacy free-form values
        # internal whole-object PUTs round-trip on every write.
        (["vqa"], ["vqa"]),
        # A change to known values passes.
        (["decision"], ["llm"]),
        ([], ["llm"]),
    ],
)
def test_unchanged_or_known_categories_pass(update_categories, stored_categories):
    _validate_categories_update(_update(update_categories), _stored(stored_categories))


def test_change_to_unknown_category_rejected():
    with pytest.raises(BadRequestException):
        _validate_categories_update(_update(["Decision"]), _stored(["llm"]))


def test_legacy_echo_passes_while_a_change_is_rejected():
    # The stored row holds a pre-whitelist value: keeping it must not block
    # the row's own updates, fixing it must go through known values only.
    stored = _stored(["vqa"])
    _validate_categories_update(_update(["vqa"]), stored)
    with pytest.raises(BadRequestException):
        _validate_categories_update(_update(["vqa", "decisions"]), stored)
    _validate_categories_update(_update(["llm"]), stored)


def test_absent_categories_field_passes():
    # A sparse update that does not touch categories says nothing about
    # them — nothing to validate.
    sparse = ModelUpdate.model_validate(
        {k: v for k, v in _payload(["llm"]).items() if k != "categories"}
        | {"replicas": 2}
    )
    _validate_categories_update(sparse, _stored(["vqa"]))


def test_create_schema_still_rejects_unknown_category():
    # The create-side half of the boundary stays in the schema: fresh
    # values with no stored row to compare against.
    from gpustack.schemas.models import ModelCreate

    with pytest.raises(ValidationError, match="Invalid category"):
        ModelCreate.model_validate(_payload(["Decision"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other", ["--data-parallel-multi-port-external-lb", "--data-parallel-hybrid-lb"]
)
async def test_vllm_conflicting_load_balancing_is_rejected_before_scheduling(
    monkeypatch, other
):
    monkeypatch.setattr(models, "validate_pd_config", AsyncMock())
    model = ModelCreate(
        name="moe",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/moe",
        backend="vLLM",
        backend_parameters=["--data-parallel-external-lb", other],
    )
    with pytest.raises(BadRequestException) as error:
        await models.validate_model_in(MagicMock(), model)
    assert "load balancing" in error.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'parameter', ['--headless', '--api-server-count=0', '--pcp=2', '--dp=1', '-dpn=0']
)
async def test_native_dps_invalid_parameters_are_rejected_before_scheduling(
    monkeypatch, parameter
):
    monkeypatch.setattr(models, 'validate_pd_config', AsyncMock())
    model = ModelCreate(
        name='moe',
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id='org/moe',
        backend='vLLM',
        backend_parameters=['-dpm', parameter],
    )
    with pytest.raises(BadRequestException):
        await models.validate_model_in(MagicMock(), model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'parameters,categories',
    [
        (['--policy=random'], None),
        (['--dp=4', '--policy=consistent_hashing'], None),
        (['--dp=4', '--policy'], None),
        (['--dp=4', '--policy=random'], ['image']),
    ],
)
async def test_sglang_policy_requires_supported_paired_smg(
    monkeypatch, parameters, categories
):
    monkeypatch.setattr(models, 'validate_pd_config', AsyncMock())
    model = ModelCreate(
        name='sg',
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id='org/sg',
        backend='SGLang',
        backend_parameters=parameters,
        categories=categories or [],
    )
    with pytest.raises(BadRequestException):
        await models.validate_model_in(MagicMock(), model)


@pytest.mark.parametrize(
    'parameters',
    [
        ['--dp=4', '--policy=random'],
        ['--enable-dp-attention', '--dp-size=4', '--policy=manual'],
        ['--ep-size=8', '--policy=cache_aware'],
    ],
)
def test_sglang_valid_policy_preserves_saved_argv(parameters):
    model = ModelCreate(
        name='sg',
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id='org/sg',
        backend='SGLang',
        backend_parameters=list(parameters),
    )
    models._validate_sglang_smg_policy(model)
    assert model.backend_parameters == parameters


def test_sglang_pd_policy_is_outside_ordinary_component_validation():
    from types import SimpleNamespace
    from gpustack.utils.model_instance_workers import validate_sglang_smg_policy

    model = SimpleNamespace(
        backend='SGLang',
        disaggregation=object(),
        backend_parameters=['--policy=', '--dp=4'],
    )
    validate_sglang_smg_policy(model)
    assert model.backend_parameters == ['--policy=', '--dp=4']
