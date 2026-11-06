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
async def test_multi_port_external_invalid_parameters_are_rejected_before_scheduling(
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
@pytest.mark.parametrize('mode', ['external', 'hybrid', 'multi-port-external'])
@pytest.mark.parametrize(
    'uds', [['--uds', '/tmp/model.sock'], ['--uds=/tmp/model.sock']]
)
async def test_managed_http_modes_reject_uds_before_scheduling(monkeypatch, mode, uds):
    monkeypatch.setattr(models, 'validate_pd_config', AsyncMock())
    model = ModelCreate(
        name='moe',
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id='org/moe',
        backend='vLLM',
        backend_parameters=[f'--data-parallel-{mode}-lb', *uds],
    )
    with pytest.raises(BadRequestException) as error:
        await models.validate_model_in(MagicMock(), model)
    assert '--uds' in error.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['external', 'hybrid', 'multi-port-external'])
async def test_manual_dp_preserves_native_parameters_without_managed_validation(
    monkeypatch, mode
):
    monkeypatch.setattr(models, 'validate_pd_config', AsyncMock())
    model = ModelCreate(
        name='manual',
        source='local_path',
        local_path='/models/moe',
        backend='vLLM',
        backend_parameters=[f'--data-parallel-{mode}-lb', '--uds=/tmp/model.sock'],
        env={'GPUSTACK_MANUAL_DP': '1'},
    )
    await models.validate_model_in(MagicMock(), model)


@pytest.mark.parametrize('backend', ['SGLang', 'custom'])
def test_manual_dp_rejects_other_backends(backend):
    model = ModelCreate(
        name='manual',
        source='local_path',
        local_path='/models/moe',
        backend=backend,
        env={'GPUSTACK_MANUAL_DP': '1'},
    )
    with pytest.raises(BadRequestException) as exc:
        models._validate_manual_dp_config(model)
    assert 'ordinary vLLM' in exc.value.message


@pytest.mark.parametrize('role_env', [False, True])
def test_manual_dp_rejects_pd_and_role_environment(role_env):
    from gpustack.schemas.models import RoleSpec

    model = ModelCreate(
        name='manual',
        source='local_path',
        local_path='/models/moe',
        backend='vLLM',
        env={} if role_env else {'GPUSTACK_MANUAL_DP': '1'},
        roles=[
            RoleSpec(
                name='prefill', env={'GPUSTACK_MANUAL_DP': '1'} if role_env else {}
            )
        ],
    )
    with pytest.raises(BadRequestException) as exc:
        models._validate_manual_dp_config(model)
    assert 'ordinary vLLM' in exc.value.message


def test_manual_dp_uses_merged_environment_for_sparse_updates():
    stored = Model(id=1, name='manual', backend='vLLM', env={'GPUSTACK_MANUAL_DP': '1'})
    assert models._validate_manual_dp_config(
        ModelUpdate(
            name='manual', source='local_path', local_path='/models/moe', replicas=2
        ),
        stored,
    )
    assert not models._validate_manual_dp_config(
        ModelUpdate(
            name='manual', source='local_path', local_path='/models/moe', env={}
        ),
        stored,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('env', [{}, None, {'GPUSTACK_MANUAL_DP': '0'}])
@pytest.mark.parametrize(
    'parameters,message',
    [
        (['--data-parallel-external-lb', '--uds=/tmp/model.sock'], '--uds'),
        (
            ['--data-parallel-external-lb', '--data-parallel-hybrid-lb'],
            'load balancing',
        ),
        (['--data-parallel-multi-port-external-lb', '--dp=1'], 'data-parallel-size'),
    ],
)
async def test_disabling_manual_dp_validates_stored_parameters(
    monkeypatch, env, parameters, message
):
    monkeypatch.setattr(models, 'validate_pd_config', AsyncMock())
    stored = Model(
        id=1,
        name='manual',
        backend='vLLM',
        backend_parameters=parameters,
        env={'GPUSTACK_MANUAL_DP': '1'},
    )
    update = ModelUpdate(
        name='manual', source='local_path', local_path='/models/moe', env=env
    )
    fields = update.model_fields_set.copy()

    with pytest.raises(BadRequestException) as exc:
        await models.validate_model_in(MagicMock(), update, stored=stored)

    assert message in exc.value.message
    assert update.model_fields_set == fields
    assert update.backend_parameters is None
    assert stored.env == {'GPUSTACK_MANUAL_DP': '1'}
    assert stored.backend_parameters == parameters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'patch', [{'replicas': 2}, {'env': {}, 'backend_parameters': []}]
)
async def test_manual_dp_updates_validate_the_effective_parameter_override(
    monkeypatch, patch
):
    monkeypatch.setattr(models, 'validate_pd_config', AsyncMock())
    stored = Model(
        id=1,
        name='manual',
        backend='vLLM',
        backend_parameters=['--data-parallel-external-lb', '--uds=/tmp/model.sock'],
        env={'GPUSTACK_MANUAL_DP': '1'},
    )
    update = ModelUpdate(
        name='manual', source='local_path', local_path='/models/moe', **patch
    )
    fields = update.model_fields_set.copy()

    await models.validate_model_in(MagicMock(), update, stored=stored)

    assert update.model_fields_set == fields


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'parameters,categories',
    [
        (['--policy=random'], None),
        (['--ep-size=8', '--policy=cache_aware'], None),
        (['--enable-dp-attention', '--dp-size=1', '--policy=random'], None),
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
        ['--ep-size=8', '--dp-size=2', '--policy=cache_aware'],
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


@pytest.mark.parametrize(
    'version,image,rejected',
    [
        ('0.5.21', None, True),
        ('0.5.22', None, False),
        ('0.5.23', None, False),
        (None, None, False),
        ('custom', None, False),
        ('0.5.21', 'custom/sglang:latest', False),
    ],
)
@pytest.mark.parametrize('alias', ['--attn-dp-size', '--attention-data-parallel-size'])
def test_sglang_dpa_version_is_checked_for_known_runner_images(
    version, image, rejected, alias
):
    from gpustack.routes import models
    from tests.utils.model import new_model

    model = new_model(
        1,
        'sglang',
        huggingface_repo_id='org/model',
        backend='SGLang',
        backend_version=version,
        image_name=image,
        backend_parameters=[alias, '8', '--tp=8'],
    )
    if rejected:
        with pytest.raises(BadRequestException) as error:
            models._validate_sglang_smg_policy(model)
        assert '0.5.22' in error.value.message
    else:
        models._validate_sglang_smg_policy(model)


@pytest.mark.parametrize('alias', ['--attn-dp-size', '--attention-data-parallel-size'])
@pytest.mark.parametrize('reported', [1, 8])
def test_sglang_new_dpa_component_report_matches_effective_rank_count(alias, reported):
    from gpustack.routes.model_instances import _validate_sglang_component_layout
    from gpustack.schemas.models import ModelInstanceWorkload
    from tests.utils.model import new_model, new_model_instance

    model = new_model(
        1,
        'dpa',
        huggingface_repo_id='org/model',
        backend='SGLang',
        backend_parameters=['--tp=8', f'{alias}=8'],
    )
    instance = new_model_instance(1, 'engine', 1, worker_id=1)
    instance.port, instance.ports = 40000, [40000]
    workload = ModelInstanceWorkload(
        name='engine',
        worker_id=1,
        state='pending',
        component={'kind': 'sglang_engine', 'api_port': 40000, 'dp_size': reported},
    )
    if reported == 8:
        _validate_sglang_component_layout(instance, model, [workload])
    else:
        with pytest.raises(BadRequestException):
            _validate_sglang_component_layout(instance, model, [workload])
