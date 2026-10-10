"""Schedule execution state is scoped to its plan and updated through PUT."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes import models as routes
from gpustack.routes.models import validate_model_in
from gpustack.schemas.deployment_document import deployment_entry
from gpustack.schemas.models import (
    Model,
    ModelCreate,
    ModelPublic,
    ModelUpdate,
    GPUSelector,
    GPUTypeSelector,
    LoraListEntry,
    ScalingSchedule,
    SourceEnum,
)
from gpustack.server.model_revisions import deployment_spec
from gpustack.server.scaling_scheduler import compute_desired_replicas


def deployment(**values):
    if values.get("scaling_schedule"):
        values["scaling_schedule"] = ScalingSchedule.model_validate(
            values["scaling_schedule"]
        )
    return Model(
        **{
            'id': 1,
            'name': 'scheduled',
            'source': SourceEnum.HUGGING_FACE,
            'created_at': datetime(2026, 10, 9, tzinfo=timezone.utc),
            'updated_at': datetime(2026, 10, 9, tzinfo=timezone.utc),
            'huggingface_repo_id': 'org/model',
            'owner_principal_id': 5,
            'backend': 'vllm',
            'distributed_inference_across_workers': False,
            **values,
        }
    )


def schedule(enabled=True, replicas=4, baseline=0, paused=None):
    return {
        **({"paused": paused} if paused is not None else {}),
        'enabled': enabled,
        'baseline_replicas': baseline,
        'rules': [
            {
                'start_cron': '0 9 * * *',
                'duration_seconds': 3600,
                'replicas': replicas,
            }
        ],
    }


@pytest.fixture
def boundary(monkeypatch):
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    updates = []

    async def update(model, source, **kwargs):
        patch = (
            source
            if isinstance(source, dict)
            else {key: getattr(source, key) for key in source.model_fields_set}
        )
        updates.append(patch)
        for key, value in patch.items():
            setattr(model, key, value)
        return model

    service = SimpleNamespace(update=AsyncMock(side_effect=update))
    monkeypatch.setattr(routes, 'ModelService', lambda session: service)
    monkeypatch.setattr(routes, 'assert_cluster_belongs_to_org', AsyncMock())
    monkeypatch.setattr(routes, 'validate_model_in', AsyncMock())
    monkeypatch.setattr(routes, 'validate_shared_kv_cache', AsyncMock())
    monkeypatch.setattr(
        routes.ModelRoute, 'one_by_fields', AsyncMock(return_value=None)
    )
    monkeypatch.setattr(routes, 'revoke_model_access_cache', AsyncMock())
    monkeypatch.setattr('gpustack.envs.TIMEZONE', 'UTC')
    # Operational actions must not enter the configuration-history path.
    history = AsyncMock(
        side_effect=AssertionError('lifecycle must not create a revision')
    )
    monkeypatch.setattr(routes, 'ensure_baseline', history)
    ctx = TenantContext(
        user=SimpleNamespace(id=99, kind='user'),
        is_platform_admin=False,
        current_principal_id=5,
        org_role=None,
        accessible_cluster_ids=set(),
    )
    return session, ctx, updates


def set_time(monkeypatch, hour):
    now = datetime(2026, 10, 9, hour, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(
        routes,
        'compute_desired_replicas',
        lambda plan: compute_desired_replicas(plan, now),
    )


def edit(model, **values):
    return ModelUpdate.model_validate({**model.model_dump(), **values})


@pytest.fixture
def unavailable_placement(boundary, monkeypatch):
    monkeypatch.setattr(routes, 'validate_model_in', validate_model_in)
    worker = AsyncMock(return_value=None)
    monkeypatch.setattr(
        routes,
        'WorkerService',
        lambda session: SimpleNamespace(get_by_cluster_id_name=worker),
    )
    pool = AsyncMock(return_value=[])
    monkeypatch.setattr(routes.GPUInstanceType, 'all_by_fields', pool)
    return (*boundary, worker, pool)


@pytest.mark.asyncio
@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('placement', ['manual', 'pool'])
@pytest.mark.parametrize('lora', [False, True])
@pytest.mark.parametrize('runtime', ['catalog', 'image', 'image_reported_version'])
async def test_pause_preserves_unavailable_placement(
    unavailable_placement, full, placement, lora, runtime
):
    session, ctx, _, worker, pool = unavailable_placement
    selector = (
        {'gpu_selector': GPUSelector(gpu_ids=['removed:cuda:0'], gpus_per_replica=1)}
        if placement == 'manual'
        else {'gpu_type_selector': GPUTypeSelector(type='removed')}
    )
    model = deployment(
        cluster_id=1,
        replicas=4,
        distributed_inference_across_workers=True,
        scaling_schedule=schedule(),
        image_name='vllm/vllm-openai:nightly' if runtime != 'catalog' else None,
        backend_version=(
            'reported-version' if runtime == 'image_reported_version' else None
        ),
        lora_list=(
            [LoraListEntry(lora_name='scheduled:adapter', lora_repo_name='org/adapter')]
            if lora
            else None
        ),
        **selector,
    )
    values = {'scaling_schedule': schedule(paused=True), 'replicas': 0}
    incoming = (
        ModelUpdate.model_validate(
            {**ModelPublic.model_validate(model).model_dump(), **values}
        )
        if full
        else ModelUpdate(
            name=model.name,
            source=model.source,
            huggingface_repo_id=model.huggingface_repo_id,
            **values,
        )
    )
    if full and lora:
        assert incoming.lora_list[0].lora_name == 'adapter'
        assert model.lora_list[0].lora_name == 'scheduled:adapter'
    await routes._apply_model_update(session, ctx, model, incoming)
    assert model.replicas == 0 and model.scaling_schedule.paused is True
    if runtime == 'image_reported_version':
        assert model.backend_version is None
    if lora:
        assert model.lora_list[0].lora_name == 'scheduled:adapter'
    for field, value in selector.items():
        assert getattr(model, field) == value
    worker.assert_not_awaited()
    pool.assert_not_awaited()
    routes.assert_cluster_belongs_to_org.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'change',
    [
        'resume',
        'baseline',
        'description',
        'selector',
        'cluster',
        'name',
        'lora_name',
        'lora_repo',
        'image',
        'backend_version',
    ],
)
async def test_pause_with_configuration_changes_still_validates_placement(
    unavailable_placement, change
):
    session, ctx, updates, worker, _ = unavailable_placement
    model = deployment(
        cluster_id=1,
        replicas=4,
        gpu_selector=GPUSelector(gpu_ids=['removed:cuda:0'], gpus_per_replica=1),
        scaling_schedule=schedule(paused=change == 'resume'),
        image_name=(
            'vllm/vllm-openai:nightly'
            if change in ('image', 'backend_version')
            else None
        ),
        backend_version=(
            'reported-version' if change in ('image', 'backend_version') else None
        ),
        lora_list=(
            [LoraListEntry(lora_name='scheduled:adapter', lora_repo_name='org/adapter')]
            if change.startswith('lora_')
            else None
        ),
    )
    values = {'scaling_schedule': schedule(paused=True), 'replicas': 0}
    if change == 'resume':
        values['scaling_schedule'] = schedule(paused=False)
    elif change == 'baseline':
        values['scaling_schedule'] = schedule(paused=True, baseline=2)
    elif change == 'description':
        values['description'] = 'changed'
    elif change == 'selector':
        values['gpu_selector'] = GPUSelector(gpu_ids=['removed:cuda:1'])
    elif change == 'cluster':
        values['cluster_id'] = 2
    elif change == 'name':
        values['name'] = 'renamed'
    elif change == 'image':
        values['image_name'] = 'vllm/vllm-openai:changed'
    elif change == 'backend_version':
        values['backend_version'] = 'changed-version'
    elif change.startswith('lora_'):
        values['lora_list'] = [
            LoraListEntry(
                lora_name='changed' if change == 'lora_name' else 'adapter',
                lora_repo_name=(
                    'org/changed' if change == 'lora_repo' else 'org/adapter'
                ),
            )
        ]
    incoming = ModelUpdate.model_validate(
        {**ModelPublic.model_validate(model).model_dump(), **values}
    )
    with pytest.raises(BadRequestException) as exc_info:
        await routes._apply_model_update(session, ctx, model, incoming)
    assert exc_info.value.message == 'Worker removed not found'
    worker.assert_awaited_once()
    assert not updates


@pytest.mark.asyncio
async def test_paused_creation_still_validates_placement(unavailable_placement):
    session, _, updates, worker, _ = unavailable_placement
    incoming = ModelCreate.model_validate(
        deployment(
            cluster_id=1,
            gpu_selector=GPUSelector(gpu_ids=['removed:cuda:0'], gpus_per_replica=1),
            scaling_schedule=schedule(paused=True),
        ).model_dump()
    )
    with pytest.raises(BadRequestException) as exc_info:
        await routes._validate_model_spec(session, incoming, 5)
    assert exc_info.value.message == 'Worker removed not found'
    worker.assert_awaited_once()
    assert not updates


@pytest.mark.asyncio
async def test_pause_and_resume_reuse_put_without_changing_plan_configuration(
    boundary, monkeypatch
):
    session, ctx, updates = boundary
    set_time(monkeypatch, 9)
    model = deployment(replicas=4, scaling_schedule=schedule())
    before = deployment_spec(model)
    latest = object()
    monkeypatch.setattr(routes, 'lock_model', AsyncMock(return_value=model))
    monkeypatch.setattr(routes, 'ensure_baseline', AsyncMock(return_value=latest))
    record = AsyncMock()
    monkeypatch.setattr(routes, 'record_update', record)
    stopped = await routes.update_model(
        session,
        ctx,
        model.id,
        edit(model, scaling_schedule=schedule(paused=True), replicas=0),
    )
    assert stopped.scaling_schedule.enabled is True
    assert stopped.scaling_schedule.paused is True and stopped.replicas == 0
    assert deployment_spec(stopped) == before
    resumed = await routes.update_model(
        session, ctx, model.id, edit(model, scaling_schedule=schedule(paused=False))
    )
    assert resumed.scaling_schedule.paused is False and resumed.replicas == 4
    assert deployment_spec(resumed) == before
    assert all(call.args[2] == before for call in record.await_args_list)
    assert all('paused' not in patch for patch in updates)


@pytest.mark.asyncio
@pytest.mark.parametrize('hour', [9, 10, 23])
@pytest.mark.parametrize('baseline', [0, 2])
async def test_resume_uses_current_window_and_configured_baseline(
    boundary, monkeypatch, hour, baseline
):
    session, ctx, _ = boundary
    set_time(monkeypatch, hour)
    model = deployment(
        replicas=0, scaling_schedule=schedule(paused=True, baseline=baseline)
    )
    await routes._apply_model_update(
        session,
        ctx,
        model,
        edit(model, scaling_schedule=schedule(paused=False, baseline=baseline)),
    )
    assert model.scaling_schedule.paused is False
    assert model.replicas == (4 if hour == 9 else baseline)


@pytest.mark.asyncio
@pytest.mark.parametrize('plan', [None, schedule(enabled=False, paused=True)])
@pytest.mark.parametrize('replicas', [0, 1, 5])
async def test_manual_scaling_is_not_subject_to_schedule_pause(
    boundary, plan, replicas
):
    session, ctx, _ = boundary
    model = deployment(replicas=2, scaling_schedule=plan)
    await routes._apply_model_update(
        session, ctx, model, edit(model, replicas=replicas)
    )
    assert model.replicas == replicas
    if model.scaling_schedule:
        assert model.scaling_schedule.enabled is False
        assert model.scaling_schedule.paused is False


@pytest.mark.asyncio
@pytest.mark.parametrize('sparse', [True, False])
async def test_configuration_edits_preserve_pause_when_execution_flag_is_omitted(
    boundary, monkeypatch, sparse
):
    session, ctx, _ = boundary
    set_time(monkeypatch, 9)
    model = deployment(replicas=0, scaling_schedule=schedule(paused=True))
    values = {
        'name': model.name,
        'source': model.source,
        'huggingface_repo_id': model.huggingface_repo_id,
        'description': 'edited',
    }
    if not sparse:
        values['scaling_schedule'] = schedule(baseline=2, replicas=6)
    await routes._apply_model_update(
        session, ctx, model, ModelUpdate.model_validate(values)
    )
    assert model.replicas == 0 and model.scaling_schedule.paused is True
    assert model.description == 'edited'
    exported = deployment_entry(model, True, None)
    snapshot = deployment_spec(model)
    assert snapshot['replicas'] == exported['replicas'] == (0 if sparse else 2)
    assert 'paused' not in snapshot and 'paused' not in exported
    assert 'paused' not in snapshot['scaling_schedule']
    assert 'paused' not in exported['scaling_schedule']


@pytest.mark.asyncio
async def test_disabling_schedule_returns_to_manual_scaling(boundary):
    session, ctx, _ = boundary
    model = deployment(replicas=0, scaling_schedule=schedule(paused=True))
    await routes._apply_model_update(
        session,
        ctx,
        model,
        edit(model, replicas=3, scaling_schedule=schedule(enabled=False)),
    )
    assert model.replicas == 3
    assert (
        model.scaling_schedule.enabled is False
        and model.scaling_schedule.paused is False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('paused', [False, True])
async def test_create_honors_explicit_schedule_pause(boundary, monkeypatch, paused):
    session, _, _ = boundary
    set_time(monkeypatch, 9)
    incoming = ModelCreate.model_validate(
        deployment(scaling_schedule=schedule(paused=paused)).model_dump()
    )
    await routes._validate_model_spec(session, incoming, 5)
    assert incoming.scaling_schedule.paused is paused
    assert incoming.replicas == (0 if paused else 4)


@pytest.mark.asyncio
@pytest.mark.parametrize('hour, expected', [(9, 4), (10, 2)])
async def test_exported_deployment_does_not_inherit_paused_execution_state(
    boundary, monkeypatch, hour, expected
):
    session, _, _ = boundary
    set_time(monkeypatch, hour)
    model = deployment(replicas=0, scaling_schedule=schedule(paused=True, baseline=2))
    exported = deployment_entry(model, True, None)
    assert (
        exported['replicas'] == exported['scaling_schedule']['baseline_replicas'] == 2
    )
    assert 'paused' not in exported['scaling_schedule']
    assert model.replicas == 0 and model.scaling_schedule.paused is True
    incoming = ModelCreate.model_validate(exported)
    await routes._validate_model_spec(session, incoming, 5)
    assert incoming.scaling_schedule.paused is False
    assert incoming.replicas == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('paused, expected', [(True, 0), (False, 4)])
async def test_sparse_pause_update_persists_derived_replicas(
    boundary, monkeypatch, paused, expected
):
    session, ctx, updates = boundary
    set_time(monkeypatch, 9)
    model = deployment(
        replicas=4 if paused else 0, scaling_schedule=schedule(paused=not paused)
    )
    incoming = ModelUpdate(
        name=model.name,
        source=model.source,
        huggingface_repo_id=model.huggingface_repo_id,
        scaling_schedule=schedule(paused=paused),
    )
    assert 'replicas' not in incoming.model_fields_set
    await routes._apply_model_update(session, ctx, model, incoming)
    assert 'replicas' in incoming.model_fields_set
    assert updates[-1]['replicas'] == model.replicas == expected


@pytest.mark.asyncio
async def test_put_pause_update_authorizes_owner_and_locks_latest_row(
    boundary, monkeypatch
):
    session, ctx, _ = boundary
    model = deployment(replicas=4, scaling_schedule=schedule())
    result = MagicMock()
    result.one_or_none.return_value = model
    session.exec = AsyncMock(return_value=result)
    monkeypatch.setattr(routes, 'ensure_baseline', AsyncMock(return_value=None))
    await routes.update_model(
        session,
        ctx,
        model.id,
        edit(model, scaling_schedule=schedule(paused=True), replicas=0),
    )
    statement = session.exec.call_args.args[0]
    assert statement._for_update_arg is not None
    assert statement.get_execution_options()['populate_existing'] is True
    model.owner_principal_id = 7
    with pytest.raises(NotFoundException):
        await routes.update_model(
            session, ctx, model.id, edit(model, scaling_schedule=schedule(paused=False))
        )


def test_pause_only_exists_under_scaling_schedule():
    for schema in (Model, ModelPublic, ModelCreate, ModelUpdate):
        assert 'paused' not in schema.model_fields
        assert 'resume_replicas' not in schema.model_fields
    assert ScalingSchedule().paused is False
    assert not any(
        route.path.endswith(('/pause', '/resume')) for route in routes.router.routes
    )
    public = ModelPublic.model_validate(
        deployment(scaling_schedule=schedule(paused=True))
    )
    assert public.scaling_schedule.paused is True


@pytest.mark.asyncio
async def test_yaml_overwrite_preserves_execution_pause_on_an_enabled_plan(
    boundary, monkeypatch
):
    session, _, _ = boundary
    monkeypatch.setattr(routes, 'ensure_baseline', AsyncMock(return_value=object()))
    monkeypatch.setattr(routes, 'record_update', AsyncMock())
    monkeypatch.setattr(routes, '_own_model_routes', AsyncMock(return_value=[]))
    model = deployment(replicas=0, scaling_schedule=schedule(paused=True))
    entry = ModelCreate.model_validate(
        {
            **model.model_dump(),
            'replicas': 7,
            'scaling_schedule': schedule(baseline=2),
            'enable_model_route': False,
        }
    )
    await routes._persist_model_update(session, model, entry)
    assert model.scaling_schedule.paused is True and model.replicas == 0
    assert deployment_spec(model)['replicas'] == 2


@pytest.mark.asyncio
async def test_rollback_updates_plan_without_resuming_execution(boundary, monkeypatch):
    from gpustack.routes import model_revisions as revisions
    from gpustack.schemas.model_revisions import ModelRevision, ModelRollbackRequest

    session, ctx, _ = boundary
    model = deployment(replicas=0, scaling_schedule=schedule(paused=True))
    before = deployment_spec(model)
    target = ModelRevision(
        id=1,
        model_id=model.id,
        revision=1,
        spec={**before, 'replicas': 2, 'scaling_schedule': schedule(baseline=2)},
    )
    latest = ModelRevision(id=2, model_id=model.id, revision=2, spec=before)
    monkeypatch.setattr(revisions, 'lock_model', AsyncMock(return_value=model))
    monkeypatch.setattr(revisions, '_find_revision', AsyncMock(return_value=target))
    monkeypatch.setattr(revisions, 'ensure_baseline', AsyncMock(return_value=latest))
    monkeypatch.setattr(routes, 'ensure_baseline', AsyncMock(return_value=latest))
    record = AsyncMock()
    monkeypatch.setattr(routes, 'record_update', record)
    restored = await revisions.rollback_model(
        session, ctx, model.id, ModelRollbackRequest(target_revision=1)
    )
    assert restored.scaling_schedule.paused is True and restored.replicas == 0
    assert deployment_spec(restored)['replicas'] == 2
    assert record.await_args.args[2] == before


@pytest.mark.asyncio
async def test_worker_version_report_cannot_resume_a_paused_schedule(boundary):
    from gpustack.schemas.principals import PrincipalType

    session, ctx, _ = boundary
    ctx.user.kind = PrincipalType.SYSTEM
    model = deployment(replicas=0, scaling_schedule=schedule(paused=True))
    stale_report = edit(
        model,
        scaling_schedule=schedule(paused=False),
        backend_version='reported-version',
    )
    await routes._apply_model_update(session, ctx, model, stale_report)
    assert model.scaling_schedule.paused is True and model.replicas == 0
    assert model.backend_version == 'reported-version'
