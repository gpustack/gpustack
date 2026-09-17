"""The cache-provider catalog as a source of its own: what a write to it may
not take away, and the binding that says how it may be configured.

Beside the module under test (``routes/cache_providers.py``) rather than with
the document parsing it drives, which lives in ``schemas``.
"""

from typing import List, Optional

import pytest
import yaml

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.cache_providers import (
    CACHE_PROVIDER_SOURCE_SPEC,
    _reject_taking_away_a_provider_in_use,
)
from gpustack.schemas.source import SourceContent, SourceTypeEnum


def _provider(name: str, version: str = "v1", **overrides) -> dict:
    """The smallest declaration that validates, which every test varies from."""
    declaration = {
        "name": name,
        "display_name": name,
        "description": f"{name} for tests.",
        "topology": "per_node",
        "default_version": version,
        "default_image": f"{name.lower()}:{version}",
        "versions": {version: {}},
        "default_run_command": f"{name.lower()} --port {{{{port}}}}",
    }
    declaration.update(overrides)
    return declaration


def _document(*providers: dict) -> str:
    return yaml.safe_dump(list(providers), sort_keys=False)


class _FakeService:
    def __init__(self, name: str, provider: str, version: Optional[str]):
        self.name = name
        self.provider_name = provider
        self.provider_version = version


class _FakeSession:
    """Stands in for the session the check queries services through."""

    def __init__(self, services: List[_FakeService]):
        self.services = services


@pytest.fixture
def services(monkeypatch):
    """Let a test declare the cache services that exist."""

    def install(*rows: _FakeService):
        async def all_services(session):
            return list(rows)

        monkeypatch.setattr(
            "gpustack.routes.cache_providers.CacheService.all",
            staticmethod(all_services),
        )

    return install


def _contents(*documents: str) -> List[SourceContent]:
    return [
        SourceContent(f"source-{index}", SourceTypeEnum.FILE, document)
        for index, document in enumerate(documents)
    ]


@pytest.mark.asyncio
async def test_a_document_dropping_a_provider_in_use_is_refused(services):
    services(_FakeService("shared-cache", "LMCache", "v0.5.3"))
    with pytest.raises(BadRequestException) as excinfo:
        await _reject_taking_away_a_provider_in_use(
            _FakeSession([]), _contents(_document(_provider("Demo")))
        )
    assert "lmcache" in str(excinfo.value.message).lower()
    assert "shared-cache" in str(excinfo.value.message)


@pytest.mark.asyncio
async def test_a_document_dropping_the_pinned_version_is_refused(services):
    services(_FakeService("shared-cache", "Demo", "v1"))
    with pytest.raises(BadRequestException) as excinfo:
        await _reject_taking_away_a_provider_in_use(
            _FakeSession([]), _contents(_document(_provider("Demo", "v2")))
        )
    message = str(excinfo.value.message)
    assert "version 'v1'" in message and "shared-cache" in message


@pytest.mark.asyncio
async def test_every_offending_pin_is_named_in_one_message(services):
    """An admin whose document is missing three providers should not have to
    submit three times to learn all three."""
    services(
        _FakeService("one", "Demo", "v1"),
        _FakeService("two", "Other", None),
        _FakeService("three", "Third", "v1"),
    )
    with pytest.raises(BadRequestException) as excinfo:
        await _reject_taking_away_a_provider_in_use(
            _FakeSession([]), _contents(_document(_provider("Demo", "v1")))
        )
    message = str(excinfo.value.message)
    assert "'other'" in message and "'third'" in message
    assert "two" in message and "three" in message
    # The one the document still carries is not reported.
    assert "'demo'" not in message


@pytest.mark.asyncio
async def test_a_service_on_a_custom_image_pins_the_opt_in_not_a_version(services):
    """The reserved "custom" version names an image of the service's own, so
    whichever versions the document declares satisfies it — but only a provider
    that still opts into custom images can launch one."""
    services(_FakeService("shared-cache", "Demo", "custom"))
    await _reject_taking_away_a_provider_in_use(
        _FakeSession([]),
        _contents(_document(_provider("Demo", "v9", custom_version=True))),
    )

    with pytest.raises(BadRequestException) as excinfo:
        await _reject_taking_away_a_provider_in_use(
            _FakeSession([]),
            _contents(_document(_provider("Demo", "v9", custom_version=False))),
        )
    assert "custom version" in str(excinfo.value.message)
    assert "shared-cache" in str(excinfo.value.message)


@pytest.mark.asyncio
async def test_a_service_that_never_named_a_version_pins_only_the_default(services):
    """An omitted version is not the reserved custom identifier: the service
    runs whatever the provider defaults to, so a document declaring any version
    satisfies it — including on a provider that does not offer custom images."""
    services(_FakeService("shared-cache", "Demo", None))
    await _reject_taking_away_a_provider_in_use(
        _FakeSession([]),
        _contents(_document(_provider("Demo", "v9", custom_version=False))),
    )

    # A document leaving the provider with no versions at all does not: there
    # is no default left to resolve.
    with pytest.raises(BadRequestException) as excinfo:
        await _reject_taking_away_a_provider_in_use(
            _FakeSession([]),
            _contents(
                _document(
                    {
                        "name": "Demo",
                        "display_name": "Demo",
                        "description": "d",
                        "topology": "per_node",
                        "custom_version": True,
                    }
                )
            ),
        )
    assert "default version" in str(excinfo.value.message)


@pytest.mark.asyncio
async def test_a_document_keeping_what_is_in_use_passes(services):
    services(_FakeService("shared-cache", "Demo", "v1"))
    await _reject_taking_away_a_provider_in_use(
        _FakeSession([]),
        _contents(_document(_provider("Demo", "v1"), _provider("Added"))),
    )


@pytest.mark.asyncio
async def test_no_cache_service_means_nothing_to_protect(services):
    services()
    await _reject_taking_away_a_provider_in_use(_FakeSession([]), _contents("[]"))


# --- the source binding ----------------------------------------------------


def test_the_spec_takes_a_document_either_way_an_admin_keeps_one():
    """Pasted in or fetched from an address of their own — the same two source
    types every other kind offers."""
    assert CACHE_PROVIDER_SOURCE_SPEC.allowed_types == (
        SourceTypeEnum.FILE,
        SourceTypeEnum.URL,
    )
    assert CACHE_PROVIDER_SOURCE_SPEC.builtin_name == "builtin"
    assert CACHE_PROVIDER_SOURCE_SPEC.pre_write_check is not None
