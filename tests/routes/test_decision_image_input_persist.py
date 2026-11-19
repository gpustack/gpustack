"""The decision-API image contract is published on the model write paths.

The pre-create evaluation runs on a ModelSpec copy: its meta (the
``decision_image_input`` key included) never reaches the persisted row,
and no server-side pass re-evaluates a stored model. A vLLM decision
deployment therefore stayed keyless, ``/v1/models?with_meta=true``
published no contract, and the Playground kept sending ``assets`` to an
engine whose systemone schema forbids it. The create and update paths
must publish (or clear) the key themselves — including healing rows
created before the contract existed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.routes import models as models_module
from gpustack.routes.models import (
    _apply_model_update,
    _persist_model_create,
    _persist_model_update,
)
from gpustack.schemas.models import Model, ModelCreate, ModelUpdate, SourceEnum


def _decision_model(meta=None):
    return Model(
        id=1,
        name="d1-3b",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="LiquidAI/d1-3B",
        categories=["decision"],
        backend="vLLM",
        backend_version="0.33.0",
        meta=meta if meta is not None else {"max_model_len": 8192},
        owner_principal_id=1,
        cluster_id=1,
    )


@pytest.mark.asyncio
async def test_create_publishes_the_contract(monkeypatch):
    # The stored row carries what the create payload carried; the write
    # path itself must add the key the evaluation copy computed and the
    # UI never merges into the POST body.
    model = _decision_model()
    monkeypatch.setattr(models_module.Model, "create", AsyncMock(return_value=model))
    monkeypatch.setattr(models_module, "append_revision", AsyncMock())
    model_in = ModelCreate(
        name="d1-3b",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="LiquidAI/d1-3B",
        cluster_id=1,
        categories=["decision"],
        backend="vLLM",
        backend_version="0.33.0",
        meta={"max_model_len": 8192},
        enable_model_route=False,
    )

    created = await _persist_model_create(
        MagicMock(), model_in, target_org_id=1, created_by=1
    )

    assert created.meta["decision_image_input"] == "images"
    assert created.meta["max_model_len"] == 8192


@pytest.mark.asyncio
async def test_update_publishes_and_heals_the_contract(monkeypatch):
    # A row created before the contract existed carries no key: any PUT
    # republishes it, so editing (or a worker's whole-object PUT of
    # backend_version) heals the deployment without a redeploy.
    model = _decision_model()
    session = MagicMock()
    monkeypatch.setattr(models_module, "assert_cluster_belongs_to_org", AsyncMock())
    monkeypatch.setattr(
        models_module, "_merge_scaling_schedule_update", lambda *a, **k: None
    )
    monkeypatch.setattr(models_module, "validate_model_in", AsyncMock())
    monkeypatch.setattr(
        models_module, "apply_scaling_schedule_baseline", lambda *a, **k: None
    )
    monkeypatch.setattr(models_module, "validate_shared_kv_cache", AsyncMock())
    monkeypatch.setattr(
        models_module.ModelRoute, "one_by_fields", AsyncMock(return_value=None)
    )
    update = MagicMock()
    update.update = AsyncMock()
    with patch.object(models_module, "ModelService", return_value=update):
        await _apply_model_update(
            session,
            MagicMock(),
            model,
            ModelUpdate(
                name="d1-3b",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="LiquidAI/d1-3B",
                replicas=1,
            ),
        )

    assert model.meta["decision_image_input"] == "images"


@pytest.mark.asyncio
async def test_import_overwrite_publishes_the_contract(monkeypatch):
    # A deployment import replaces the row's whole desired state; an entry
    # that turns a stopped model into a decision vLLM deployment must
    # publish the contract like the create and PUT paths do.
    model = _decision_model()
    monkeypatch.setattr(
        models_module, "ensure_baseline", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(models_module, "record_update", AsyncMock())
    monkeypatch.setattr(models_module, "_own_model_routes", AsyncMock([]))
    update = MagicMock()
    update.update = AsyncMock()
    with patch.object(models_module, "ModelService", return_value=update):
        imported = await _persist_model_update(
            MagicMock(),
            model,
            ModelCreate(
                name="d1-3b",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="LiquidAI/d1-3B",
                cluster_id=1,
                categories=["decision"],
                backend="vLLM",
                enable_model_route=False,
            ),
            created_by=1,
        )

    assert imported.meta["decision_image_input"] == "images"


@pytest.mark.asyncio
async def test_update_clears_a_stale_contract(monkeypatch):
    # The category or backend changed after the key was written: the key
    # must not outlive the state that justified it.
    model = _decision_model(meta={"decision_image_input": "images"})
    model.categories = ["llm"]
    session = MagicMock()
    monkeypatch.setattr(models_module, "assert_cluster_belongs_to_org", AsyncMock())
    monkeypatch.setattr(
        models_module, "_merge_scaling_schedule_update", lambda *a, **k: None
    )
    monkeypatch.setattr(models_module, "validate_model_in", AsyncMock())
    monkeypatch.setattr(
        models_module, "apply_scaling_schedule_baseline", lambda *a, **k: None
    )
    monkeypatch.setattr(models_module, "validate_shared_kv_cache", AsyncMock())
    monkeypatch.setattr(
        models_module.ModelRoute, "one_by_fields", AsyncMock(return_value=None)
    )
    update = MagicMock()
    update.update = AsyncMock()
    with patch.object(models_module, "ModelService", return_value=update):
        await _apply_model_update(
            session,
            MagicMock(),
            model,
            ModelUpdate(
                name="d1-3b",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="LiquidAI/d1-3B",
                replicas=1,
            ),
        )

    assert "decision_image_input" not in model.meta
