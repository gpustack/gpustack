"""Automatic route names are validated before deployment persistence."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gpustack.api.exceptions import register_handlers
from gpustack.routes import models as models_routes
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
