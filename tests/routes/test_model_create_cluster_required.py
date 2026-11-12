"""POST /v2/models requires a cluster.

The column is NOT NULL, so an omitted cluster_id used to surface only after
the insert failed, as a bare 500. The requirement lives on the create route
rather than on the ModelCreate schema: import entries share that schema and
resolve their cluster by name during import planning, which can legitimately
parse without one.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gpustack.routes.models as models_routes
from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import ModelCreate, SourceEnum


def _model_in(cluster_id):
    return ModelCreate(
        name="qwen",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="repo/qwen",
        cluster_id=cluster_id,
    )


@pytest.mark.asyncio
async def test_create_without_cluster_id_is_rejected_before_any_write(monkeypatch):
    resolve = AsyncMock()
    monkeypatch.setattr(models_routes, "_resolve_target_org", resolve)
    # An untouched session proves the refusal happens at the request
    # boundary, not after planning has already touched the database.
    session = MagicMock()

    with pytest.raises(BadRequestException) as excinfo:
        await models_routes.create_model(
            session=session, ctx=SimpleNamespace(), model_in=_model_in(None)
        )

    assert excinfo.value.status_code == 400
    assert "cluster_id is required" in excinfo.value.message
    resolve.assert_not_awaited()
    assert session.mock_calls == []


@pytest.mark.asyncio
async def test_create_with_cluster_id_passes_the_guard(monkeypatch):
    resolve = AsyncMock(return_value=(1, SimpleNamespace(id=1)))
    monkeypatch.setattr(models_routes, "_resolve_target_org", resolve)
    monkeypatch.setattr(models_routes, "_check_model_create", AsyncMock())

    async def fake_persist(session, model_in, target_org_id, created_by=None):
        return SimpleNamespace(id=7)

    monkeypatch.setattr(models_routes, "_persist_model_create", fake_persist)
    session = MagicMock(commit=AsyncMock(), rollback=AsyncMock())

    model = await models_routes.create_model(
        session=session,
        ctx=SimpleNamespace(current_principal_id=1, user=None),
        model_in=_model_in(1),
    )

    assert model.id == 7
    session.commit.assert_awaited_once()
