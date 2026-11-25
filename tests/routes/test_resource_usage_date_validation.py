from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import register_handlers
from gpustack.routes.resource_usage import (
    ResourceBreakdownRequest,
    ResourceExportRequest,
    router,
)
from gpustack.schemas.users import User
from gpustack.server.deps import get_current_user, get_session, get_tenant_context


@pytest.fixture
def usage_app_and_session():
    result = MagicMock()
    result.first.return_value = None
    result.all.return_value = []
    stream = MagicMock(close=AsyncMock())
    stream.partitions.return_value.__aiter__.return_value = []
    session = MagicMock(spec=AsyncSession)
    session.exec.return_value = result
    session.stream.return_value = stream

    app = FastAPI()
    app.include_router(router, prefix="/v2/usage")
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_current_user] = lambda: User(id=1, name="test-user")
    app.dependency_overrides[get_tenant_context] = lambda: None
    return app, session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/resource/breakdown",
        "/gpu-instances/breakdown",
        "/storage/breakdown",
        "/gpu-instances/breakdown/export",
        "/storage/breakdown/export",
        "/gpu-instances/breakdown/export/estimate",
        "/storage/breakdown/export/estimate",
    ],
)
async def test_resource_usage_rejects_reversed_dates_before_querying(
    path, usage_app_and_session
):
    app, session = usage_app_and_session
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/v2/usage{path}",
            json={
                "scope": "self",
                "start_date": "2026-01-02",
                "end_date": "2026-01-01",
                "group_by": ["date"],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "end_date"]
    session.exec.assert_not_called()
    session.execute.assert_not_called()
    session.stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/summary", "/resource-events"])
@pytest.mark.parametrize(
    "filters,expected_status,expected_message",
    [
        pytest.param(
            {}, 422, "end_date must be on or after start_date", id="date-range"
        ),
        pytest.param(
            {"organization_ids": "1"},
            403,
            "No permission to filter by organization",
            id="forbidden-filter",
        ),
    ],
)
async def test_get_resource_usage_rejects_reversed_dates_after_filter_permission_check(
    path, filters, expected_status, expected_message, usage_app_and_session
):
    app, session = usage_app_and_session
    register_handlers(app)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/v2/usage{path}",
            params={
                "scope": "self",
                "start_date": "2026-01-02",
                "end_date": "2026-01-01",
                **filters,
            },
        )

    assert response.status_code == expected_status
    assert response.json()["message"] == expected_message
    session.exec.assert_not_called()
    session.execute.assert_not_called()
    session.stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,start_date,end_date",
    [
        ("/summary", "2026-01-01", "2026-01-01"),
        ("/summary", "2026-01-01", "2026-01-02"),
        ("/resource-events", "2026-01-01", "2026-01-01"),
        ("/resource-events", "2026-01-01", "2026-01-02"),
        ("/resource-events", "2026-01-01", None),
        ("/resource-events", None, "2026-01-02"),
        ("/resource-events", None, None),
    ],
)
async def test_get_resource_usage_accepts_valid_date_filters(
    path, start_date, end_date, usage_app_and_session
):
    app, session = usage_app_and_session
    params = {"scope": "self"}
    if start_date is not None:
        params["start_date"] = start_date
    if end_date is not None:
        params["end_date"] = end_date

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/v2/usage{path}", params=params)

    assert response.status_code == 200
    session.exec.assert_awaited()


@pytest.mark.parametrize(
    "request_model", [ResourceBreakdownRequest, ResourceExportRequest]
)
@pytest.mark.parametrize("end_date", [date(2026, 1, 1), date(2026, 1, 2)])
def test_resource_usage_accepts_forward_and_same_day_ranges(request_model, end_date):
    request = request_model(
        start_date=date(2026, 1, 1), end_date=end_date, group_by=["date"]
    )
    assert request.end_date == end_date
