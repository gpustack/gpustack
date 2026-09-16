"""The cache-provider catalog as configurable content: what a document has to
satisfy to be stored, what it replaces once it is, and what a write may not take
away."""

import logging
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import List, Optional

import pytest
import pytest_asyncio
import yaml
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.cache_provider_source import (
    CacheProviderEntry,
    CacheProviderSource,
    dump_cache_providers,
    load_cache_providers_document,
    normalize_cache_provider_yaml,
    reconcile_cache_providers,
)
from gpustack.schemas.source import SourceContent, SourceTypeEnum
from gpustack.routes.cache_providers import (
    CACHE_PROVIDER_SOURCE_SPEC,
    _reject_taking_away_a_provider_in_use,
)
from gpustack.server.cache_provider_catalog import (
    asset_providers,
    builtin_catalog_text,
    get_cache_provider,
    get_cache_providers,
    providers_from_documents,
)

PACKAGED_CATALOG = (
    files("gpustack.assets").joinpath("cache-providers.yaml").read_text("utf-8")
)


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


@asynccontextmanager
async def _entries_session():
    """In-memory session over the entries table alone, matching the app's
    ``expire_on_commit=False`` (required under async SQLAlchemy)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all, tables=[CacheProviderEntry.__table__]
        )
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def session():
    async with _entries_session() as opened:
        yield opened


def _sources(*documents: str) -> List[SourceContent]:
    return [
        SourceContent(f"source-{index}", SourceTypeEnum.FILE, document)
        for index, document in enumerate(documents)
    ]


# --- storing a document ----------------------------------------------------


def test_a_source_stores_a_document_larger_than_mysql_text():
    """``TEXT`` caps at 64 KiB on MySQL, and a catalog carrying every provider's
    declaration is past it."""
    content = CacheProviderSource.__table__.c.content.type
    assert content.compile(dialect=mysql.dialect()) == "LONGTEXT"
    assert content.compile(dialect=postgresql.dialect()) == "TEXT"


def test_the_packaged_catalog_round_trips(caplog):
    """The baseline an admin downloads has to be a document they can save back
    unchanged — it is the starting point for every custom one."""
    with caplog.at_level(logging.WARNING):
        normalized = normalize_cache_provider_yaml(PACKAGED_CATALOG, strict=True)
    assert not caplog.records

    first = load_cache_providers_document(normalized, strict=True)
    second = load_cache_providers_document(
        normalize_cache_provider_yaml(normalized, strict=True), strict=True
    )
    assert [provider.model_dump() for provider in first] == [
        provider.model_dump() for provider in second
    ]
    # Declaration order is the order the cards appear in.
    assert [provider.name for provider in first] == [
        provider.name for provider in load_cache_providers_document(PACKAGED_CATALOG)
    ]


@pytest.mark.parametrize(
    "document, expected",
    [
        ("name: LMCache", "must be a YAML list"),
        ("- [1, 2]", "unreadable"),
        ("- {name: Demo}", "unreadable"),
        (": :\n", "not valid YAML"),
    ],
)
def test_a_document_that_cannot_serve_is_refused(document, expected):
    with pytest.raises(ValueError, match=expected):
        normalize_cache_provider_yaml(document, strict=True)


def test_an_empty_document_is_a_catalog_with_no_providers():
    """Distinct from an unreadable one: clearing the catalog is a thing an admin
    may legitimately do."""
    assert normalize_cache_provider_yaml("", strict=True) == "[]\n"
    assert load_cache_providers_document("") == []


@pytest.mark.parametrize(
    "overrides, reported",
    [
        ({"managed_fields": []}, "Demo.managed_fields"),
        ({"versions": {"v1": {"image_tag": "x"}}}, "Demo.versions.v1.image_tag"),
        (
            {"fields": [{"name": "size", "labl": {"default": "Size"}}]},
            "Demo.fields[0].labl",
        ),
    ],
)
def test_a_misspelled_field_is_named_rather_than_ignored(overrides, reported):
    """An unknown key changes nothing at runtime: a component whose
    ``run_command`` was spelled ``run_cmd`` launches with no command. The
    admin's own document is where that has to be caught."""
    document = _document(_provider("Demo", **overrides))
    with pytest.raises(ValueError, match="unknown cache provider field"):
        normalize_cache_provider_yaml(document, strict=True)
    with pytest.raises(ValueError, match=reported.replace("[", r"\[")):
        normalize_cache_provider_yaml(document, strict=True)


def test_an_unattended_read_keeps_what_it_can(caplog):
    """The lenient half of the same rule: a document stored by a newer version
    still serves the declarations this one understands."""
    document = _document(
        _provider("Demo", managed_fields=[]),
        {"name": "Broken", "topology": "nonsense"},
    )
    with caplog.at_level(logging.WARNING):
        providers = load_cache_providers_document(document)
    assert [provider.name for provider in providers] == ["Demo"]
    assert "Broken" in caplog.text
    assert "managed_fields" in caplog.text


def test_a_document_whose_declarations_are_all_unreadable_is_refused():
    """Serving none of them would read as "an empty catalog" and take every
    provider out of service."""
    with pytest.raises(ValueError, match="none of the declarations"):
        load_cache_providers_document(_document({"name": "Broken", "topology": "nope"}))


# --- what the sources materialize into --------------------------------------


@pytest.mark.asyncio
async def test_the_sources_materialize_into_the_catalog_readers_query(session):
    await reconcile_cache_providers(
        session, _sources(_document(_provider("Demo"), _provider("Other")))
    )

    providers = await get_cache_providers(session)
    assert [provider.name for provider in providers] == ["Demo", "Other"]
    # Card order is the document's order, not whatever the rows come back in.
    assert [entry.position for entry in await CacheProviderEntry.all(session)] == [0, 1]
    assert (await get_cache_provider(session, "demo")) is not None


@pytest.mark.asyncio
async def test_a_document_replaces_the_catalog_rather_than_adding_to_it(session):
    await reconcile_cache_providers(session, _sources(builtin_catalog_text()))
    assert "LMCache" in [
        provider.name for provider in await get_cache_providers(session)
    ]

    await reconcile_cache_providers(session, _sources(_document(_provider("Demo"))))

    assert [provider.name for provider in await get_cache_providers(session)] == [
        "Demo"
    ]
    # What the assets carry is gone, including the providers a plugin
    # contributed — which is why the baseline is the thing to edit from.
    assert (await get_cache_provider(session, "LMCache")) is None


@pytest.mark.asyncio
async def test_a_row_keeps_its_id_across_a_rewrite(session):
    await reconcile_cache_providers(session, _sources(_document(_provider("Demo"))))
    first = (await CacheProviderEntry.all(session))[0]

    await reconcile_cache_providers(
        session, _sources(_document(_provider("Demo", "v2"), _provider("Added")))
    )

    rows = {entry.name: entry for entry in await CacheProviderEntry.all(session)}
    assert rows["Demo"].id == first.id
    assert rows["Demo"].payload["default_version"] == "v2"
    assert set(rows) == {"Demo", "Added"}


@pytest.mark.asyncio
async def test_a_row_is_stamped_with_the_source_that_produced_it(session):
    """Origin display: a declaration of the admin's is told apart from one this
    release carries."""
    await reconcile_cache_providers(
        session,
        [
            SourceContent("builtin", SourceTypeEnum.BUILTIN, builtin_catalog_text()),
            SourceContent(
                "custom", SourceTypeEnum.FILE, _document(_provider("LMCache"))
            ),
        ],
    )

    rows = {entry.name: entry for entry in await CacheProviderEntry.all(session)}
    assert rows["LMCache"].source_type == SourceTypeEnum.FILE
    assert rows["XSKY MeshFusion"].source_type == SourceTypeEnum.BUILTIN


@pytest.mark.asyncio
async def test_no_source_clears_the_catalog(session):
    """Every source dropped is a real state — the packaged baseline is itself a
    source row, so nothing serving means nothing declared."""
    await reconcile_cache_providers(session, _sources(_document(_provider("Demo"))))
    await reconcile_cache_providers(session, [])

    assert await get_cache_providers(session) == []


def test_the_baseline_reads_in_declaration_order():
    """It is a document meant to be edited by hand: sorted keys open every
    declaration on whatever comes first alphabetically and bury the name in the
    middle of it."""
    first_keys = [
        line.split(":")[0].lstrip("- ")
        for line in builtin_catalog_text().splitlines()
        if line.startswith("- ")
    ]
    assert set(first_keys) == {"name"}


def test_the_builtin_baseline_carries_every_asset_declaration():
    """What an admin downloads to edit from: the packaged asset merged with what
    installed plugins contribute, in the form they save back."""
    baseline = builtin_catalog_text()
    names = [provider.name for provider in load_cache_providers_document(baseline)]
    assert names == [provider.name for provider in asset_providers()]
    # Saving it back unchanged is a no-op rather than a validation error.
    assert normalize_cache_provider_yaml(baseline, strict=True) == baseline


def test_documents_merge_in_order_with_later_ones_replacing_by_name():
    providers = providers_from_documents(
        [
            _document(_provider("Demo", "v1"), _provider("Other", "v1")),
            _document(_provider("Demo", "v2")),
        ]
    )
    assert [provider.name for provider in providers] == ["Demo", "Other"]
    # Replaced in place, so the order a user sees the cards in is stable.
    assert providers[0].default_version == "v2"


def test_dump_is_what_normalize_stores():
    providers = load_cache_providers_document(_document(_provider("Demo")))
    assert dump_cache_providers(providers) == normalize_cache_provider_yaml(
        _document(_provider("Demo"))
    )


# --- what a write may not take away ----------------------------------------


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
async def test_a_service_on_a_custom_image_pins_only_its_provider(services):
    """The reserved "custom" version names an image of the service's own, so a
    document that carries the provider satisfies it whatever versions it
    declares."""
    services(_FakeService("shared-cache", "Demo", "custom"))
    await _reject_taking_away_a_provider_in_use(
        _FakeSession([]), _contents(_document(_provider("Demo", "v9")))
    )


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
