"""Recovery decisions must see configuration edits before status reconciliation."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes.models import get_model
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    RoleSpec,
    SourceEnum,
)
from gpustack.schemas.principals import PrincipalType
from gpustack.server.controllers import model_spec_digest


@pytest.fixture
def deployment(monkeypatch):
    session = MagicMock()
    session.info = {}
    model = Model(
        id=1,
        name="pd",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        owner_principal_id=7,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
        stale=False,
    )
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER, id=7),
        is_platform_admin=False,
        current_principal_id=7,
        org_role=None,
    )
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_id", AsyncMock(return_value=model)
    )
    types = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "gpustack.server.controllers.GPUInstanceType.all_by_fields", types
    )
    return session, model, ctx, types


async def _stamp(session, model):
    digest = await model_spec_digest(session, model)
    model.instances = [
        ModelInstance(
            id=1,
            name="prefill",
            model_id=model.id,
            model_name=model.name,
            role="prefill",
            group_id="1-group",
            spec_digest=digest,
        )
    ]
    session.info.clear()


@pytest.mark.asyncio
async def test_model_detail_reports_edit_before_controller_runs(deployment):
    session, model, ctx, _ = deployment
    await _stamp(session, model)
    model.roles[0].backend_parameters = ["--max-model-len=8192"]

    public = await get_model(session, ctx, model.id)

    assert public.stale is True
    assert model.stale is False, "a read must not persist derived status"


@pytest.mark.asyncio
async def test_model_detail_accepts_rebuilt_group_before_controller_runs(deployment):
    session, model, ctx, _ = deployment
    await _stamp(session, model)
    model.stale = True

    public = await get_model(session, ctx, model.id)

    assert public.stale is False
    assert model.stale is True


@pytest.mark.asyncio
async def test_model_detail_tracks_saved_configuration_through_stop_and_start(
    deployment,
):
    session, model, ctx, _ = deployment
    await _stamp(session, model)
    model.roles[0].backend_parameters = ["--max-model-len=8192"]
    model.stale = True

    model.replicas = 0
    for member in model.instances:
        member.deleted_at = datetime.now(timezone.utc)
    assert (await get_model(session, ctx, model.id)).stale is None

    model.replicas = 1
    await _stamp(session, model)
    assert (await get_model(session, ctx, model.id)).stale is False
    assert model.stale is True, "startup must not wait for controller status"


@pytest.mark.asyncio
async def test_model_detail_ignores_deleted_generation(deployment):
    session, model, ctx, _ = deployment
    await _stamp(session, model)
    old = model.instances[0].model_copy(
        update={"id": 2, "spec_digest": "old", "deleted_at": datetime.now(timezone.utc)}
    )
    model.instances.append(old)
    model.stale = True

    assert (await get_model(session, ctx, model.id)).stale is False


@pytest.mark.asyncio
async def test_model_detail_exempts_detected_backend_version(deployment):
    session, model, ctx, _ = deployment
    await _stamp(session, model)
    model.backend_version = "0.31.0"
    model.instances[0].backend_version = "0.31.0"
    model.stale = True

    assert (await get_model(session, ctx, model.id)).stale is False


@pytest.mark.asyncio
async def test_model_detail_keeps_legacy_staleness_unknown(deployment):
    session, model, ctx, _ = deployment
    await _stamp(session, model)
    model.instances[0].spec_digest = None

    assert (await get_model(session, ctx, model.id)).stale is None


@pytest.mark.asyncio
async def test_model_detail_checks_visibility_before_staleness(deployment):
    session, model, ctx, types = deployment
    ctx.current_principal_id = 8

    with pytest.raises(NotFoundException):
        await get_model(session, ctx, model.id)

    types.assert_not_awaited()


@pytest.mark.asyncio
async def test_version_writeback_does_not_block_a_waiting_sibling(deployment):
    session, model, ctx, _ = deployment
    await _stamp(session, model)
    first = model.instances[0]
    first.backend_version = "0.31.0"
    first.state = ModelInstanceStateEnum.RUNNING
    waiting = first.model_copy(
        update={
            "id": 2,
            "role": "decode",
            "state": ModelInstanceStateEnum.SCHEDULED,
            "backend_version": None,
        }
    )
    model.instances.append(waiting)
    model.backend_version = "0.31.0"

    assert (await get_model(session, ctx, model.id)).stale is False
    assert waiting.backend_version is None
