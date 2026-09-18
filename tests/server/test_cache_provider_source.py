"""The cache-provider catalog as configurable content: what a document has to
satisfy to be stored, what it replaces once it is, and what a write may not take
away."""

import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import List

import pytest
import pytest_asyncio
import yaml
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.cache_provider_source import (
    CacheProviderEntry,
    CacheProviderSource,
    dump_cache_providers,
    load_cache_providers_document,
    normalize_cache_provider_yaml,
    reconcile_cache_providers,
)
from gpustack.schemas.source import SourceContent, SourceTypeEnum
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


def test_a_misspelled_template_filter_is_named_rather_than_left_to_launch():
    """A filter nothing defines renders through every other check and then
    raises while a launch command is being built — far from the document that
    named it, and only for the configurations reaching that template."""
    document = _document(
        _provider("Demo", default_run_command="demo --size {{ram_size|gib_to_byte}}")
    )
    with pytest.raises(ValueError, match="gib_to_byte"):
        normalize_cache_provider_yaml(document, strict=True)


def test_a_filter_is_checked_wherever_a_declaration_carries_one():
    """Not only in injections: a filter reads the same in a resource claim, and
    a check that knows which fields hold templates goes stale."""
    document = _document(
        _provider("Demo", resource_profile={"ram_gib": "{{ram_size|to_gib}}"})
    )
    with pytest.raises(ValueError, match="to_gib"):
        normalize_cache_provider_yaml(document, strict=True)


def test_the_filter_that_exists_passes():
    document = _document(
        _provider("Demo", default_run_command="demo --size {{ram_size|gib_to_bytes}}")
    )
    assert load_cache_providers_document(document, strict=True)


@pytest.mark.parametrize(
    "injection",
    [
        # Every field a mapping: what the shape of the value alone cannot tell
        # apart from a mapping of declarations keyed by name.
        {
            "env": {"MOONCAKE_CONFIG_PATH": "/tmp/x.json", "PYTHONHASHSEED": "0"},
            "files": {"/tmp/x.json": "{}"},
            "kv_transfer_config": {"kv_connector": "Demo", "kv_role": "kv_both"},
        },
        # A mapping beside a list, which is the packaged catalog's shape.
        {
            "env": {"PYTHONHASHSEED": "0"},
            "args": ["--flag"],
        },
    ],
)
def test_keys_a_document_invents_are_not_read_as_fields(injection):
    """An injection's env vars and the files it writes are keyed by names the
    document chooses. Reading them as field names rejected a declaration that
    was entirely valid — the enterprise Mooncake one, whose injection carries
    no list to give its shape away."""
    document = _document(
        _provider(
            "Demo",
            inference_backend_integrations=[
                {"backend": "vLLM", "injection": injection}
            ],
        )
    )

    assert load_cache_providers_document(document, strict=True)


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
async def test_one_unreadable_source_does_not_stop_the_others(session, caplog):
    """A document is validated when it is written, so one that will not parse
    here was corrupted after the fact. Raising would leave the table serving
    whatever it last held, with nothing to say why."""
    with caplog.at_level(logging.ERROR):
        await reconcile_cache_providers(
            session,
            [
                SourceContent("builtin", SourceTypeEnum.BUILTIN, ": : not yaml\n"),
                SourceContent(
                    "custom", SourceTypeEnum.FILE, _document(_provider("Demo"))
                ),
            ],
        )

    assert [provider.name for provider in await get_cache_providers(session)] == [
        "Demo"
    ]
    assert "builtin" in caplog.text


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


def test_an_unreadable_stored_document_does_not_refuse_a_write(caplog):
    """A pre-write check reads every stored document to judge the catalog a
    write would produce. One of them may have been corrupted after it was
    written — every document is validated as it is stored — and that is not a
    reason to refuse an admin's own write, least of all with the unexplained
    server error an exception out of a check becomes. Skipped here as the
    materialization skips it, so the check sees the catalog that will serve."""
    with caplog.at_level(logging.ERROR):
        providers = providers_from_documents(
            [
                _document(_provider("Demo", "v1")),
                "name: not a list\n",
                _document(_provider("Other", "v1")),
            ]
        )

    assert [provider.name for provider in providers] == ["Demo", "Other"]
    # Silently dropping a declaration is how a catalog goes wrong unnoticed.
    assert any("Skipping unreadable" in record.message for record in caplog.records)


def test_both_source_tables_are_registered_for_migrations():
    """``migrations/env.py`` fills ``target_metadata`` by importing this package
    alone, so a table whose module the package does not import is absent from it
    — and the next ``alembic revision --autogenerate`` writes a drop_table for it
    against a database that has one. A fresh install is no safer than an
    upgraded one.

    Asked in a new interpreter: metadata is process-global, so importing the
    module anywhere — including from this test file — registers the tables and
    would answer for an import the package does not make."""
    probe = (
        "import gpustack.schemas; from sqlmodel import SQLModel; "
        "print(sorted(t for t in SQLModel.metadata.tables "
        "if t.startswith('cache_provider_')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(
        ["cache_provider_entries", "cache_provider_sources"]
    )
