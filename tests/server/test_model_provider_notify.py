"""Which config edits make ``ModelProviderController`` re-announce its routes.

A provider's McpBridge registry is derived from its endpoint, so an edit that
moves the endpoint has to reach the route layer: without the re-announce the
gateway keeps the old registry destination and traffic goes on to the host the
provider used to point at. The controller decides that from a hardcoded field
list -- exactly the kind of list a newly added endpoint field gets left out of.
"""

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.schemas.model_provider import (
    ClaudeConfig,
    ModelProvider,
    ModelProviderTypeEnum,
    OpenAIConfig,
)
from gpustack.server import controllers as controllers_module
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import ModelProviderController


def _provider(config) -> ModelProvider:
    return ModelProvider(id=7, name="claude-1", api_tokens=["sk-test"], config=config)


def _claude(url: Optional[str] = None, version: Optional[str] = None) -> ClaudeConfig:
    return ClaudeConfig(
        type=ModelProviderTypeEnum.CLAUDE, claudeCustomUrl=url, claudeVersion=version
    )


def _config_change(old, new) -> Event:
    # The shape the change detector emits: one entry per changed column, whose
    # value is an (old, new) tuple of single-element lists.
    return Event(
        type=EventType.UPDATED,
        data=_provider(new),
        changed_fields={"config": ([old], [new])},
    )


async def _looked_up_routes(monkeypatch, event: Event) -> bool:
    """Whether the notify step got as far as querying the provider's routes.

    That query is the observable half of the decision -- what follows is the
    fan-out itself, which needs real ModelRoute rows to say anything more.
    """
    lookup = AsyncMock(return_value=[])
    monkeypatch.setattr(controllers_module.ModelRouteTarget, "all_by_fields", lookup)
    monkeypatch.setattr(controllers_module.event_bus, "publish", AsyncMock())

    controller = ModelProviderController.__new__(ModelProviderController)
    await controller._notify_provider_model_routes(MagicMock(), event.data, event)

    return lookup.await_count > 0


class TestProviderRouteReannounce:
    @pytest.mark.asyncio
    async def test_a_new_claude_endpoint_reannounces(self, monkeypatch):
        # claudeCustomUrl decides the registry's domain, port, protocol and
        # static-vs-dns type, the same way openaiCustomUrl does.
        event = _config_change(_claude(), _claude(url="http://192.168.50.14:8080"))

        assert await _looked_up_routes(monkeypatch, event) is True

    @pytest.mark.asyncio
    async def test_a_moved_claude_endpoint_reannounces(self, monkeypatch):
        event = _config_change(
            _claude(url="http://192.168.50.14:8080"),
            _claude(url="https://gw.example.com/anthropic"),
        )

        assert await _looked_up_routes(monkeypatch, event) is True

    @pytest.mark.asyncio
    async def test_an_unrelated_config_edit_stays_quiet(self, monkeypatch):
        # claudeVersion reaches the plugin through the ai-proxy config, which is
        # reconciled regardless -- re-announcing on it would fan a header change
        # out across every route the provider serves.
        event = _config_change(
            _claude(url="http://192.168.50.14:8080"),
            _claude(url="http://192.168.50.14:8080", version="2025-01-01"),
        )

        assert await _looked_up_routes(monkeypatch, event) is False

    @pytest.mark.asyncio
    async def test_the_openai_endpoint_still_reannounces(self, monkeypatch):
        # The field the list was written for, kept honest next to the new one.
        event = _config_change(
            OpenAIConfig(type=ModelProviderTypeEnum.OPENAI),
            OpenAIConfig(
                type=ModelProviderTypeEnum.OPENAI,
                openaiCustomUrl="http://192.168.50.14:8080/v1",
            ),
        )

        assert await _looked_up_routes(monkeypatch, event) is True

    @pytest.mark.asyncio
    async def test_an_edit_that_is_not_the_config_stays_quiet(self, monkeypatch):
        event = Event(
            type=EventType.UPDATED,
            data=_provider(_claude(url="http://192.168.50.14:8080")),
            changed_fields={"name": (["claude-1"], ["claude-2"])},
        )

        assert await _looked_up_routes(monkeypatch, event) is False
