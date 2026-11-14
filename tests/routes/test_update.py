import pytest

from gpustack import __version__
from gpustack.routes import update as update_route
from gpustack.server import update_check


@pytest.fixture
def update_requests(monkeypatch):
    """Record the update service URLs a released build would request."""
    requested = []

    async def fake_do_get_update(update_check_url=None, timeout=3):
        requested.append(update_check_url)
        return update_check.UpdateResponse(latest_version="99.0.0")

    monkeypatch.setattr(update_check, "is_dev_version", lambda: False)
    monkeypatch.setattr(update_check, "cached_update", None)
    monkeypatch.setattr(update_check, "cache_timestamp", None)
    monkeypatch.setattr(update_check, "do_get_update", fake_do_get_update)
    return requested


@pytest.mark.asyncio
async def test_update_route_skips_the_update_service_when_disabled(
    config, monkeypatch, update_requests
):
    monkeypatch.setattr(config, "disable_update_check", True)

    response = await update_route.update()

    assert update_requests == []
    assert response.latest_version == __version__


@pytest.mark.asyncio
async def test_update_route_queries_the_configured_update_service(
    config, monkeypatch, update_requests
):
    monkeypatch.setattr(config, "disable_update_check", False)
    monkeypatch.setattr(config, "update_check_url", "https://updates.example.com")
    monkeypatch.setattr(update_route, "__version__", "1.0.0")

    response = await update_route.update()

    assert update_requests == ["https://updates.example.com"]
    assert response.latest_version == "99.0.0"
