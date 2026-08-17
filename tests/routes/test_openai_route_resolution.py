from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import NotFoundException, ServiceUnavailableException
from gpustack.routes import openai as openai_route
from tests.utils.mock import mock_async_session


def _stub_resolution(monkeypatch, *, route):
    """Wire proxy_request_by_model up to the point of route resolution.

    ``resolve_route_targets`` always comes back empty here (the branch under
    test), and ``get_by_name`` returns ``route``, which is what decides whether
    the caller sees 404 or 503.
    """
    monkeypatch.setattr(openai_route, "async_session", lambda: mock_async_session())
    monkeypatch.setattr(
        openai_route,
        "parse_request_body",
        AsyncMock(return_value=("qwen3", False, {"model": "qwen3"}, None)),
    )
    monkeypatch.setattr(
        openai_route.UserService,
        "model_allowed_for_user",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        "resolve_route_targets",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        "get_by_name",
        AsyncMock(return_value=route),
    )


def _request():
    request = MagicMock()
    request.url.path = "/v1-openai/chat/completions"
    return request


@pytest.mark.asyncio
async def test_unknown_model_name_is_not_found(monkeypatch):
    """No route by that name really is a 404."""
    _stub_resolution(monkeypatch, route=None)

    with pytest.raises(NotFoundException) as exc:
        await openai_route.proxy_request_by_model(
            request=_request(), user=SimpleNamespace(id=1)
        )

    assert exc.value.status_code == 404
    assert exc.value.message == "Model not found"


@pytest.mark.asyncio
async def test_deployed_model_with_no_active_target_is_unavailable(monkeypatch):
    """A route whose targets have all gone UNAVAILABLE must not read as a
    missing model.

    ``ready_replicas`` drops to 0 for every model on a worker as soon as that
    worker misses /healthz, which flips each target to UNAVAILABLE and empties
    ``resolve_route_targets``. Returning 404 there tells the client a deployed
    model no longer exists and that retrying is pointless. It has to be 503.
    """
    _stub_resolution(monkeypatch, route=SimpleNamespace(id=7, name="qwen3"))

    with pytest.raises(ServiceUnavailableException) as exc:
        await openai_route.proxy_request_by_model(
            request=_request(), user=SimpleNamespace(id=1)
        )

    assert exc.value.status_code == 503
    assert exc.value.message == "No running instances available"
