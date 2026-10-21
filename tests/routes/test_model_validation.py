"""Request-boundary validations for model create/update."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from gpustack.api.exceptions import BadRequestException, register_handlers
from gpustack.routes import models as models_routes
from gpustack.routes.models import _validate_categories_update
from gpustack.schemas.models import Model, ModelUpdate
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
