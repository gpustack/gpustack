from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.routes.resource_usage import (
    ResourceBreakdownRequest,
    ResourceExportRequest,
    router,
)
from gpustack.schemas.users import User
from gpustack.server.deps import get_current_user, get_session, get_tenant_context


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
async def test_resource_usage_rejects_reversed_dates_before_querying(path):
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


@pytest.mark.parametrize(
    "request_model", [ResourceBreakdownRequest, ResourceExportRequest]
)
@pytest.mark.parametrize("end_date", [date(2026, 1, 1), date(2026, 1, 2)])
def test_resource_usage_accepts_forward_and_same_day_ranges(request_model, end_date):
    request = request_model(
        start_date=date(2026, 1, 1), end_date=end_date, group_by=["date"]
    )
    assert request.end_date == end_date
