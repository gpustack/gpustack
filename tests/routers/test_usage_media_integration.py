"""Media (image generation) usage through the real usage API on SQLite.

A mix of chat and image rows is seeded so every assertion checks that image
units are attributed to the right rows, that the ``operations`` filter
narrows the token and request totals with them, and that the export carries
the same numbers as the breakdown table.
"""

import csv
import io
from contextlib import asynccontextmanager
from datetime import date, datetime
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import register_handlers
from gpustack.routes import usage as usage_routes
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_routes import ModelRoute
from gpustack.schemas.model_usage import ModelUsage, OperationEnum
from gpustack.schemas.principals import Principal
from gpustack.schemas.users import User
from gpustack.server.deps import get_current_user, get_session, get_tenant_context

NOW = datetime(2026, 4, 1, 0, 0, 0)
RANGE = {"start_date": "2026-04-01", "end_date": "2026-04-02", "scope": "all"}
IMAGE_ONLY = {"operations": [OperationEnum.IMAGE_GENERATION.value]}
MP_1024 = 1024 * 1024


class _Ctx:
    user = None
    is_platform_admin = True
    current_principal_id = None
    org_role = None
    current_is_personal_scope = False


def _row(*, user_id, route_id, day, operation, tokens=0, requests=1, images=0, px=0):
    return ModelUsage(
        user_id=user_id,
        user_name=f"user{user_id}",
        model_id=route_id,
        model_name=f"route{route_id}",
        model_route_id=route_id,
        model_route_name=f"route{route_id}",
        date=day,
        prompt_token_count=tokens,
        completion_token_count=0,
        prompt_cached_token_count=0,
        request_count=requests,
        image_count=images,
        image_pixel_count=px,
        operation=operation,
        created_at=NOW,
        updated_at=NOW,
    )


# user1: chats on route1 and generates 3 + 1 images on route2 (1024x1024).
# user2: generates 2 images on route2 at 512x512 and chats once.
def _seed_rows():
    # Fresh instances per database: an ORM object committed once is
    # persistent, and re-adding it to another session inserts nothing.
    return [
        _row(
            user_id=1,
            route_id=1,
            day=date(2026, 4, 1),
            operation=OperationEnum.CHAT_COMPLETION,
            tokens=500,
            requests=4,
        ),
        _row(
            user_id=1,
            route_id=2,
            day=date(2026, 4, 1),
            operation=OperationEnum.IMAGE_GENERATION,
            requests=2,
            images=3,
            px=3 * MP_1024,
        ),
        _row(
            user_id=1,
            route_id=2,
            day=date(2026, 4, 2),
            operation=OperationEnum.IMAGE_GENERATION,
            requests=1,
            images=1,
            px=MP_1024,
        ),
        _row(
            user_id=2,
            route_id=2,
            day=date(2026, 4, 2),
            operation=OperationEnum.IMAGE_GENERATION,
            requests=1,
            images=2,
            px=2 * 512 * 512,
        ),
        _row(
            user_id=2,
            route_id=1,
            day=date(2026, 4, 2),
            operation=OperationEnum.CHAT_COMPLETION,
            tokens=100,
            requests=1,
        ),
    ]


@pytest_asyncio.fixture
async def make_client():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (ModelUsage, Principal, ModelRoute, ApiKey):
            await conn.run_sync(model.__table__.create)
    async with AsyncSession(engine) as seed:
        seed.add_all(_seed_rows())
        await seed.commit()

    async def _session_dep():
        async with AsyncSession(engine) as session:
            yield session

    @asynccontextmanager
    async def _enrichment_session():
        async with AsyncSession(engine) as session:
            yield session

    clients = []

    async def _make(user: User) -> AsyncClient:
        app = FastAPI()
        register_handlers(app)
        app.include_router(usage_routes.router, prefix="/usage")
        ctx = _Ctx()
        ctx.user = user
        ctx.is_platform_admin = user.is_admin
        app.dependency_overrides[get_session] = _session_dep
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_tenant_context] = lambda: ctx
        ac = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(ac)
        return ac

    with patch.object(usage_routes, "async_session", _enrichment_session):
        yield _make
    for ac in clients:
        await ac.aclose()
    await engine.dispose()


@pytest_asyncio.fixture
async def admin_client(make_client):
    return await make_client(User(id=99, name="admin", is_admin=True))


@pytest.mark.asyncio
async def test_summary_reports_image_units_across_all_operations(admin_client):
    response = await admin_client.post(
        "/usage/breakdown", json={**RANGE, "group_by": ["user"], "page": -1}
    )

    assert response.status_code == 200, response.text
    summary = response.json()["summary"]
    assert summary["image_count"] == 6
    assert summary["image_pixels"] == 4 * MP_1024 + 2 * 512 * 512
    # Unfiltered, chat traffic still counts toward requests and tokens.
    assert summary["api_requests"] == 9
    assert summary["total_tokens"] == 600


@pytest.mark.asyncio
async def test_operations_filter_scopes_every_metric_to_image_generation(
    admin_client,
):
    response = await admin_client.post(
        "/usage/breakdown",
        json={**RANGE, "filters": IMAGE_ONLY, "group_by": ["user"], "page": -1},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["summary"]["api_requests"] == 4
    assert body["summary"]["total_tokens"] == 0
    by_user = {
        item["user"]["label"]: (item["image_count"], item["image_pixels"])
        for item in body["items"]
    }
    assert by_user == {
        "user1": (4, 4 * MP_1024),
        "user2": (2, 2 * 512 * 512),
    }


@pytest.mark.asyncio
async def test_breakdown_sorts_by_image_count(admin_client):
    response = await admin_client.post(
        "/usage/breakdown",
        json={
            **RANGE,
            "filters": IMAGE_ONLY,
            "group_by": ["date"],
            "granularity": "day",
            "sort_by": "-image_count",
            "page": -1,
        },
    )

    assert response.status_code == 200, response.text
    assert [i["image_count"] for i in response.json()["items"]] == [3, 3]


@pytest.mark.asyncio
async def test_self_scope_sees_only_own_image_usage(make_client):
    client = await make_client(User(id=2, name="user2", is_admin=False))

    response = await client.post(
        "/usage/breakdown",
        json={
            **RANGE,
            "scope": "self",
            "filters": IMAGE_ONLY,
            "group_by": ["route"],
            "page": -1,
        },
    )

    assert response.status_code == 200, response.text
    summary = response.json()["summary"]
    assert summary["image_count"] == 2
    assert summary["api_requests"] == 1


@pytest.mark.asyncio
async def test_unknown_operation_is_rejected(admin_client):
    response = await admin_client.post(
        "/usage/breakdown",
        json={
            **RANGE,
            "filters": {"operations": ["video_generation"]},
            "group_by": ["user"],
        },
    )

    assert response.status_code in (400, 422), response.text


@pytest.mark.asyncio
async def test_export_carries_image_columns_matching_breakdown(admin_client):
    response = await admin_client.post(
        "/usage/breakdown/export",
        json={
            **RANGE,
            "format": "csv",
            "filters": IMAGE_ONLY,
            "group_by": ["route"],
        },
    )

    assert response.status_code == 200, response.text
    text = response.content.decode("utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    assert len(rows) == 1
    assert rows[0]["Model"] == "route2"
    assert int(rows[0]["Images Generated"]) == 6
    assert int(rows[0]["Image Pixels"]) == 4 * MP_1024 + 2 * 512 * 512
