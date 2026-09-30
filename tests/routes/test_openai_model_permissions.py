from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from httpx import ASGITransport, AsyncClient

from gpustack.api.auth import get_current_user, inference_scope
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import openai as openai_route
from gpustack.schemas.api_keys import PermissionScope
from gpustack.server.db import get_session
from gpustack.server.services import UserService


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/v1", "/v1-openai"])
@pytest.mark.parametrize("is_admin", [False, True])
@pytest.mark.parametrize(
    "use_key,allowed_names,expected",
    [
        (False, None, ["qwen3", "team-a/qwen3", "team-b/qwen3"]),
        (True, None, ["qwen3", "team-a/qwen3", "team-b/qwen3"]),
        (True, [], ["qwen3", "team-a/qwen3", "team-b/qwen3"]),
        (True, ["qwen3"], ["qwen3"]),
        (True, ["team-a/qwen3"], ["team-a/qwen3"]),
        (True, ["team-a/qwen3", "inaccessible"], ["team-a/qwen3"]),
        (True, ["inaccessible"], []),
    ],
)
async def test_model_list_respects_key_permissions(
    monkeypatch, prefix, is_admin, use_key, allowed_names, expected
):
    user = SimpleNamespace(id=7, is_admin=is_admin)
    key = SimpleNamespace(
        allowed_model_names=allowed_names, scope=[PermissionScope.INFERENCE]
    )
    ctx = TenantContext(
        user=user,
        is_platform_admin=is_admin,
        current_principal_id=42,
        org_role=None,
    )
    routes = [
        SimpleNamespace(
            name="qwen3",
            owner_principal_id=owner_id,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            meta={"description": str(owner_id)},
        )
        for owner_id in [None, 42, 43]
    ]
    owners = [
        SimpleNamespace(id=42, name="team-a"),
        SimpleNamespace(id=43, name="team-b"),
    ]
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            MagicMock(all=MagicMock(return_value=routes)),
            MagicMock(all=MagicMock(return_value=owners)),
        ]
    )

    async def current_user(request: Request):
        if use_key:
            request.state.api_key = key
        return user

    app = FastAPI()
    router = APIRouter(dependencies=[Depends(inference_scope)])
    api_router = (
        openai_route.get_api_router()
        if prefix == "/v1"
        else openai_route.get_legacy_api_router()
    )
    router.include_router(api_router, prefix=prefix)
    app.include_router(router)
    app.dependency_overrides[get_current_user] = current_user
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_tenant_context] = lambda: ctx

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"{prefix}/models?with_meta=true")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == expected
    assert all(model["meta"]["description"] for model in body["data"])

    # Listing and invocation must agree on the published (Org-prefixed) names.
    names = ["qwen3", "team-a/qwen3", "team-b/qwen3"]
    monkeypatch.setattr(
        UserService,
        "get_user_accessible_model_names",
        AsyncMock(return_value=set(names)),
    )
    service = UserService(session)
    assert [
        name
        for name in names
        if await service.model_allowed_for_user(name, user.id, key if use_key else None)
    ] == expected
