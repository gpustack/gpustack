"""External cache-service health aggregation.

An external service flips between RUNNING and UNREACHABLE on a single
probe of its registered endpoint.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.schemas.cache_providers import CacheProvider, CacheProviderHealthCheck
from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceEndpoint,
    CacheServiceModeEnum,
    CacheServiceStateEnum,
)
from gpustack.server.cache_services import CacheServiceHealthChecker


def _provider() -> CacheProvider:
    return CacheProvider(
        name="LMCache",
        health_check=CacheProviderHealthCheck(scheme="tcp"),
    )


def _external_service(**overrides) -> CacheService:
    fields = dict(
        id=7,
        name="fleet-cache",
        provider_name="LMCache",
        mode=CacheServiceModeEnum.EXTERNAL,
        cluster_id=1,
        state=CacheServiceStateEnum.RUNNING,
        healthy=True,
        endpoint=CacheServiceEndpoint(host="cache.example.com", port=8100),
    )
    fields.update(overrides)
    return CacheService(**fields)


class _FakeSessionCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


def _persistence_row(**overrides):
    fields = dict(
        state=CacheServiceStateEnum.RUNNING,
        state_message=None,
        healthy=True,
        last_check_at=None,
        deleted_at=None,
        update=AsyncMock(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.asyncio
async def test_check_service_single_endpoint_flips_to_unreachable(monkeypatch):
    checker = CacheServiceHealthChecker()
    service = _external_service()
    row = _persistence_row()
    monkeypatch.setattr(
        "gpustack.server.cache_services.async_session", lambda: _FakeSessionCtx()
    )
    monkeypatch.setattr(
        "gpustack.server.cache_services.get_cache_provider", lambda name: _provider()
    )
    monkeypatch.setattr(
        "gpustack.server.cache_services.CacheService.one_by_id",
        AsyncMock(return_value=row),
    )
    probe = AsyncMock(return_value=(False, "connection refused"))
    monkeypatch.setattr("gpustack.server.cache_services.probe_cache_service", probe)

    await checker._check_service(service)

    probe.assert_awaited_once()
    assert probe.await_args.args[1].host == "cache.example.com"
    row.update.assert_awaited_once()
    assert row.state == CacheServiceStateEnum.UNREACHABLE
    assert row.healthy is False
    assert row.state_message == "connection refused"
