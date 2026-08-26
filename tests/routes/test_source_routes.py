"""The source APIs: the shared singleton configuration router
(``sources/routes.py``) driven through all three of its consumers — the model
catalog, the community backends and the built-in backend versions — plus the
refresh status and trigger (``routes/source_probe.py``) they are reported by.

One file, because the three consumers share one engine and one set of fakes: a
helper that diverges between them is a bug in the engine, not a difference worth
keeping two copies of.
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
import yaml
from gpustack_runner.runner import Runner
from pydantic import ValidationError
from sqlalchemy import event as sa_event, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.auth import get_admin_user
from gpustack.api.exceptions import BadRequestException, ServiceUnavailableException
from gpustack.routes import inference_backend, model_sets, ota_sources
from gpustack.routes.source_probe import run_source_probe, source_probe_status
from gpustack.schemas.catalog_source import (
    BUILTIN_CATALOG_SOURCE_NAME,
    CUSTOM_CATALOG_SOURCE_NAME,
    CatalogModelEntry,
    CatalogSource,
    normalize_catalog_yaml,
    reconcile_catalog,
)
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.inference_backend_source import (
    BUILTIN_BACKEND_SOURCE_NAME,
    CUSTOM_BACKEND_SOURCE_NAME,
    InferenceBackendSource,
    normalize_backend_yaml,
    reconcile_backend,
)
from gpustack.schemas.models import BackendEnum, BackendSourceEnum, Model, SourceEnum
from gpustack.schemas import runner_source
from gpustack.schemas.runner_source import (
    CUSTOM_RUNNER_SOURCE_NAME,
    InferenceRunnerSource,
    RunnerOverrideEntry,
    reconcile_runner_overrides,
)
from gpustack.server import controllers
from gpustack.server.sources import core as core_module
from gpustack.server.sources import probe as probe_module
from gpustack.server.sources import routes as routes_module
from gpustack.schemas.source import SourceTypeEnum
from gpustack.server.sources.core import (
    OFFICIAL_SOURCE_NAME,
    gather_and_merge,
    sha256_of,
)
from gpustack.server.sources.probe import (
    OFFICIAL_DEFAULT_HOURS,
    OTA_SERVER_URL,
    OfficialRef,
    RefreshRound,
    SourceRefresher,
)
from gpustack.server.sources.routes import (
    CustomSourceUpsert,
    OfficialSourceUpsert,
    SourceConfigUpsert,
    delete_source_config,
    get_source_config,
    reload_source_config,
    update_source_config,
)

_REAL_ASYNC_CLIENT = httpx.AsyncClient
_COMMUNITY_BACKEND_SPEC = inference_backend.COMMUNITY_BACKEND_SPEC
_CATALOG_SPEC = model_sets.CATALOG_SOURCE_SPEC
_BUILTIN_BACKEND_SPEC = inference_backend.BUILTIN_BACKEND_SPEC

# The document the faked URL fetch returns; a URL-based helper sets it per write.
_REMOTE = {"doc": ""}


def _upsert(
    official_hours: int = OFFICIAL_DEFAULT_HOURS,
    remote_enabled: bool = True,
    **custom,
) -> SourceConfigUpsert:
    """A write body: the custom half (absent when no kwargs are given, which is
    the switch back to the platform layers), whether remote content serves at all,
    and the official cadence."""
    return SourceConfigUpsert(
        remote_enabled=remote_enabled,
        custom=CustomSourceUpsert(**custom) if custom else None,
        official=OfficialSourceUpsert(auto_update_hours=official_hours),
    )


def _install_fake_url_fetch(monkeypatch):
    """Make ``fetch_source_text`` return ``_REMOTE['doc']`` without a real request,
    so a URL-only source can be configured in-memory."""

    def make_client(**kwargs):
        handler = httpx.MockTransport(
            lambda request: httpx.Response(200, text=_REMOTE["doc"])
        )
        return _REAL_ASYNC_CLIENT(transport=handler, **kwargs)

    async def resolve_nothing(request: httpx.Request) -> None:
        """The fake transport never connects, so nothing needs to resolve."""

    monkeypatch.setattr(core_module.httpx, "AsyncClient", make_client)
    monkeypatch.setattr(core_module, "reject_a_forbidden_address", resolve_nothing)


@asynccontextmanager
async def _make_source_session(*tables):
    """In-memory session over just the given tables, matching the app's
    ``expire_on_commit=False`` (required under async SQLAlchemy)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all, tables=[table.__table__ for table in tables]
        )
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
    await engine.dispose()


async def _deploy(session, name: str, backend: str, backend_version: str) -> Model:
    """A model pinning one backend version — what the in-use checks read."""
    return await Model.create(
        session,
        Model(
            name=name,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
            backend=backend,
            backend_version=backend_version,
            replicas=1,
        ),
    )


def test_the_ota_endpoints_are_one_admin_only_family_outside_any_id_namespace():
    """Configuration and probe are one family under ``/v2/ota-sources`` and
    ``/v2/source-probe``, admin-only, and mounted once.

    Asserted on the assembled tree, for three reasons: neither router declares a
    dependency of its own (the mount gates them), ``inference_backend.router`` is
    mounted twice (so a source endpoint hanging off it was exposed on the worker
    client router too), and the point of moving them out is that no fixed segment
    sits in a ``/{id}`` namespace where it resolves only by mount order.
    """
    from gpustack.routes.routes import api_router

    # Filtered before reading ``methods``: the assembled tree also carries
    # websocket routes, which have no such attribute.
    ota_routes = [
        route
        for route in api_router.routes
        if "/ota-sources" in route.path or route.path.endswith("/source-probe")
    ]
    ota = {(route.path, method) for route in ota_routes for method in route.methods}
    assert ota == {
        ("/v2/ota-sources/{kind}", "GET"),
        ("/v2/ota-sources/{kind}", "PUT"),
        ("/v2/ota-sources/{kind}", "DELETE"),
        ("/v2/ota-sources/{kind}/reload", "POST"),
        ("/v2/source-probe", "GET"),
        ("/v2/source-probe", "POST"),
    }
    for route in ota_routes:
        assert any(
            dependency.dependency is get_admin_user for dependency in route.dependencies
        )

    # Nothing source-shaped is left under a consumer, where the retired
    # ``/source`` / ``/runner-source`` mounts sat beside ``/{id}``.
    assert not any(
        route.path.endswith(("/source", "/runner-source", "/sources"))
        for route in api_router.routes
    )


def test_every_source_kind_is_addressable_and_named_as_the_probe_names_it():
    """One identifier across the two endpoints the same screen calls: the path
    ``kind`` is exactly the key ``GET /source-probe`` reports under, so a client
    needs no translation table, and every published kind is configurable."""
    assert {kind.value for kind in ota_sources.SourceKind} == {
        kind.name for kind in probe_module.OFFICIAL_KINDS
    }
    assert set(ota_sources._SPECS) == set(ota_sources.SourceKind)


# --- catalog ---------------------------------------------------------------


def _catalog(*names) -> str:
    """A catalog document with one minimal model set per name."""
    return yaml.safe_dump(
        {
            "model_sets": [
                {
                    "name": name,
                    "specs": [
                        {
                            "source": "huggingface",
                            "huggingface_repo_id": f"org/{name}",
                            "mode": "standard",
                        }
                    ],
                }
                for name in names
            ]
        }
    )


async def _seed_catalog_builtin(session, *names) -> CatalogSource:
    """The BUILTIN row as the leader's seed leaves it."""
    return await CatalogSource.create(
        session,
        CatalogSource(
            name=BUILTIN_CATALOG_SOURCE_NAME,
            source_type=SourceTypeEnum.BUILTIN,
            enabled=True,
            content=normalize_catalog_yaml(_catalog(*names)),
        ),
    )


async def _catalog_materialized(session) -> dict:
    """Reconcile as the leader would, returning name → producing source."""
    await gather_and_merge(session, CatalogSource, reconcile_catalog)
    return {
        entry.name: entry.source_name for entry in await CatalogModelEntry.all(session)
    }


class TestCatalogSourceConfig:
    @pytest_asyncio.fixture
    async def session(self):
        async with _make_source_session(CatalogSource, CatalogModelEntry) as session:
            yield session

    @pytest.mark.asyncio
    async def test_a_record_this_version_cannot_read_is_named_on_the_way_in(
        self, session, monkeypatch
    ):
        """Strictness follows who owns the document. FILE is the admin's own text,
        so a model set that will not exist is named in the response. The same
        content behind a URL he cannot edit is taken, minus that record —
        refusing it would hand him a 400 the background refresh does not.
        """
        document = yaml.safe_load(_catalog("Good"))
        document["model_sets"].append(
            {
                "name": "Typo",
                "specs": [{"source": "huggingfacce", "huggingface_repo_id": "org/x"}],
            }
        )
        document = yaml.safe_dump(document)

        with pytest.raises(BadRequestException) as rejected:
            await update_source_config(
                session,
                _CATALOG_SPEC,
                _upsert(source_type=SourceTypeEnum.FILE, content=document),
            )
        assert "Typo" in rejected.value.message
        # Refused outright: nothing stored, so the admin edits and PUTs again.
        assert (await get_source_config(session, _CATALOG_SPEC)).custom is None

        # Same document behind a URL: accepted, minus the unreadable record.
        _REMOTE["doc"] = document
        _install_fake_url_fetch(monkeypatch)
        await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(source_type=SourceTypeEnum.URL, url="https://example.com/c.yaml"),
        )
        assert await _catalog_materialized(session) == {
            "Good": CUSTOM_CATALOG_SOURCE_NAME
        }

    @pytest.mark.asyncio
    async def test_singleton_lifecycle(self, session):
        """One configuration point: PUT replaces the baseline outright, DELETE
        restores the factory state. There is no partial mode to choose."""
        await _seed_catalog_builtin(session, "Baseline", "Shared")

        # Nothing configured yet: the baseline is what the catalog serves.
        config = await get_source_config(session, _CATALOG_SPEC)
        assert config.custom is None
        assert config.official.enabled is True

        # Configure an inline document: it becomes the whole catalog, so Baseline
        # is gone rather than merged under the admin's entries.
        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(
                source_type=SourceTypeEnum.FILE,
                content=_catalog("Shared", "Mine"),
            ),
        )
        assert config.changed is True
        assert config.custom.source_type == SourceTypeEnum.FILE
        assert config.custom.updated_at is not None
        # The official slot is masked by the same write, in the same response.
        assert config.official.enabled is False
        assert await _catalog_materialized(session) == {
            "Shared": CUSTOM_CATALOG_SOURCE_NAME,
            "Mine": CUSTOM_CATALOG_SOURCE_NAME,
        }
        baseline = await CatalogSource.one_by_field(
            session, "name", BUILTIN_CATALOG_SOURCE_NAME
        )
        assert baseline.enabled is False  # masked, not deleted

        # Reopening the configuration reads the stored text back (a FILE source
        # has no URL to re-fetch, so this is the only way to edit what was saved).
        reopened = await get_source_config(session, _CATALOG_SPEC)
        assert reopened.custom.content == normalize_catalog_yaml(
            _catalog("Shared", "Mine")
        )

        # Re-PUTting the same document changes nothing and is reported as such.
        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(
                source_type=SourceTypeEnum.FILE,
                content=_catalog("Shared", "Mine"),
            ),
        )
        assert config.changed is False
        assert set(await _catalog_materialized(session)) == {"Shared", "Mine"}

        # One write carries both halves: the official cadence is set while a
        # custom source is configured, so the screen saves in a single request.
        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(
                official_hours=0,
                source_type=SourceTypeEnum.FILE,
                content=_catalog("Shared", "Mine"),
            ),
        )
        assert config.official.auto_update_hours == 0
        assert config.official.enabled is False  # still masked by the custom source
        assert config.custom is not None

        # ``custom: null`` is the switch back, and it carries a cadence too — so
        # returning to the official source is also one request.
        config = await update_source_config(
            session, _CATALOG_SPEC, _upsert(official_hours=6)
        )
        assert config.custom is None
        assert config.changed is True  # a source really was dropped
        assert config.official.auto_update_hours == 6
        assert config.official.enabled is True
        assert await _catalog_materialized(session) == {
            "Baseline": BUILTIN_CATALOG_SOURCE_NAME,
            "Shared": BUILTIN_CATALOG_SOURCE_NAME,
        }

        # Switching back again is a no-op, and says so.
        config = await update_source_config(
            session, _CATALOG_SPEC, _upsert(official_hours=6)
        )
        assert config.changed is False

        # Restore factory state: the custom source is dropped and the baseline is
        # forced back on, so the catalog can never end up empty.
        config = await delete_source_config(session, _CATALOG_SPEC)
        assert config.custom is None
        assert config.official.enabled is True
        assert await _catalog_materialized(session) == {
            "Baseline": BUILTIN_CATALOG_SOURCE_NAME,
            "Shared": BUILTIN_CATALOG_SOURCE_NAME,
        }

    @pytest.mark.asyncio
    async def test_a_custom_source_masks_both_baselines_and_reset_unmasks_them(
        self, session
    ):
        """A custom source masks (disables) the BUILTIN and OFFICIAL rows alike so
        nothing stacks under it; reset re-enables both with their content intact.
        The config API reaches those rows only through the ``enabled`` flag —
        never their name (none is accepted from the client) or type (neither is
        a writable source type).
        """
        await _seed_catalog_builtin(session, "Baseline")
        official = await CatalogSource.create(
            session,
            CatalogSource(
                name=OFFICIAL_SOURCE_NAME,
                source_type=SourceTypeEnum.OFFICIAL,
                enabled=True,
                content=normalize_catalog_yaml(_catalog("FromOfficial")),
            ),
        )

        # Default: the official slot serves. Its document is the kind's whole
        # content, so the packaged baseline is out of the merge even though its row
        # is still enabled — that replacement is ``order_source_contents``' job,
        # separate from the ``enabled`` masking this test is about.
        assert await _catalog_materialized(session) == {
            "FromOfficial": OFFICIAL_SOURCE_NAME
        }

        # No name is accepted from the client, and OFFICIAL is not a writable type.
        assert "name" not in CustomSourceUpsert.model_fields
        # The official half takes the cadence and nothing else: the document, its
        # URL and its hashes belong to the refresh task, and a request reaching
        # for them is named and refused rather than silently dropped. Whether it
        # serves is not part of this half either — that is the whole
        # configuration's ``remote_enabled``.
        for field in ("url", "content", "content_hash", "enabled"):
            with pytest.raises(ValidationError):
                OfficialSourceUpsert(**{"auto_update_hours": 12, field: "x"})
        # Neither half absorbs an unknown key, and neither takes a negative
        # cadence — both refused by the model, so a bad body never reaches a write.
        with pytest.raises(ValidationError):
            CustomSourceUpsert(source_type=SourceTypeEnum.URL, enabled=True)
        with pytest.raises(ValidationError):
            _upsert(official_hours=-1)
        with pytest.raises(BadRequestException) as rejected:
            await update_source_config(
                session,
                _CATALOG_SPEC,
                _upsert(source_type=SourceTypeEnum.OFFICIAL, content=_catalog("Mine")),
            )
        assert "'file' or 'url'" in rejected.value.message

        # PUT a custom source: OFFICIAL is masked (disabled), out of the merge,
        # but its content is left untouched for a later reset.
        await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(source_type=SourceTypeEnum.FILE, content=_catalog("Mine")),
        )
        masked = await CatalogSource.one_by_field(session, "name", OFFICIAL_SOURCE_NAME)
        assert masked.enabled is False
        assert masked.content == normalize_catalog_yaml(_catalog("FromOfficial"))
        masked_baseline = await CatalogSource.one_by_field(
            session, "name", BUILTIN_CATALOG_SOURCE_NAME
        )
        assert masked_baseline.enabled is False
        assert await _catalog_materialized(session) == {
            "Mine": CUSTOM_CATALOG_SOURCE_NAME
        }

        # Reset: the custom row is dropped and both are unmasked, back in the
        # merge with the same rows and content.
        await delete_source_config(session, _CATALOG_SPEC)
        unmasked = await CatalogSource.one_by_field(
            session, "name", OFFICIAL_SOURCE_NAME
        )
        assert unmasked.id == official.id
        assert unmasked.source_type == SourceTypeEnum.OFFICIAL
        assert unmasked.enabled is True
        assert await _catalog_materialized(session) == {
            "FromOfficial": OFFICIAL_SOURCE_NAME
        }

    @pytest.mark.asyncio
    async def test_falling_back_to_the_packaged_baseline_and_lifting_it_again(
        self, session, monkeypatch
    ):
        """The escape hatch for remote content that turns out to be wrong: the
        packaged baseline serves alone, and nothing is discarded on the way —
        both the official document and the admin's own are parked, so coming back
        needs no fetch and no re-typing.
        """
        await _seed_catalog_builtin(session, "Baseline")
        await CatalogSource.create(
            session,
            CatalogSource(
                name=OFFICIAL_SOURCE_NAME,
                source_type=SourceTypeEnum.OFFICIAL,
                enabled=True,
                content=normalize_catalog_yaml(_catalog("FromOfficial")),
            ),
        )

        # Fall back while following the official source.
        config = await update_source_config(
            session, _CATALOG_SPEC, _upsert(remote_enabled=False)
        )
        assert config.changed is True  # what is served moved, though no document did
        assert config.remote_enabled is False
        assert config.official.enabled is False
        assert await _catalog_materialized(session) == {
            "Baseline": BUILTIN_CATALOG_SOURCE_NAME
        }
        parked = await CatalogSource.one_by_field(session, "name", OFFICIAL_SOURCE_NAME)
        assert parked.content == normalize_catalog_yaml(_catalog("FromOfficial"))

        # Lift it again: the stored document is back in the merge at once, and it
        # is the whole content again — the packaged baseline steps back out.
        config = await update_source_config(session, _CATALOG_SPEC, _upsert())
        assert config.changed is True
        assert config.remote_enabled is True
        assert await _catalog_materialized(session) == {
            "FromOfficial": OFFICIAL_SOURCE_NAME
        }

        # Saving the same state again moves nothing, and says so.
        assert (
            await update_source_config(session, _CATALOG_SPEC, _upsert())
        ).changed is False

        # Now with a document of the admin's own: falling back parks it rather
        # than deleting it, so the way back is the same one field.
        await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(source_type=SourceTypeEnum.FILE, content=_catalog("Mine")),
        )
        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(
                remote_enabled=False,
                source_type=SourceTypeEnum.FILE,
                content=_catalog("Mine"),
            ),
        )
        assert config.remote_enabled is False
        # Still configured, and its text is intact — an inline document has no
        # other copy, so the fall-back must not be the thing that loses it.
        assert config.custom is not None
        assert config.custom.content == normalize_catalog_yaml(_catalog("Mine"))
        assert await _catalog_materialized(session) == {
            "Baseline": BUILTIN_CATALOG_SOURCE_NAME
        }

        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(source_type=SourceTypeEnum.FILE, content=_catalog("Mine")),
        )
        assert config.remote_enabled is True
        assert await _catalog_materialized(session) == {
            "Mine": CUSTOM_CATALOG_SOURCE_NAME
        }

        # A parked kind can still be reconfigured, and that write reaches no
        # further than the database: pointing the source somewhere else while
        # fallen back must work when the network is exactly what is broken. The
        # new URL is stored unread, so the baseline goes on serving alone.
        _install_fake_url_fetch(monkeypatch)
        _REMOTE["doc"] = _catalog("FromMyMirror")
        fetched = []
        real_fetch = routes_module.fetch_source_text

        async def counting_fetch(url):
            fetched.append(url)
            return await real_fetch(url)

        monkeypatch.setattr(routes_module, "fetch_source_text", counting_fetch)
        await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(
                remote_enabled=False,
                source_type=SourceTypeEnum.FILE,
                content=_catalog("Mine"),
            ),
        )
        url = "https://mirror.example/catalog.yaml"
        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(remote_enabled=False, source_type=SourceTypeEnum.URL, url=url),
        )
        assert fetched == []
        assert config.remote_enabled is False
        assert config.custom.url == url
        # Nothing behind it yet — the document is read by the write that puts it
        # back in service, not by the one that stores it.
        assert config.custom.content is None
        assert await _catalog_materialized(session) == {
            "Baseline": BUILTIN_CATALOG_SOURCE_NAME
        }

        config = await update_source_config(
            session, _CATALOG_SPEC, _upsert(source_type=SourceTypeEnum.URL, url=url)
        )
        assert fetched == [url]
        assert config.remote_enabled is True
        assert await _catalog_materialized(session) == {
            "FromMyMirror": CUSTOM_CATALOG_SOURCE_NAME
        }

    @pytest.mark.asyncio
    async def test_url_source_reload_only_writes_when_the_document_moved(
        self, session, monkeypatch
    ):
        """Reload re-fetches the stored URL. An unchanged document writes nothing
        (no reconcile); a moved one is applied. FILE sources are rejected, and
        with nothing configured the call is about the official slot instead."""
        _REMOTE["doc"] = _catalog("Remote")
        _install_fake_url_fetch(monkeypatch)

        # Nothing configured: the reload is about the official slot, which only
        # the leader refreshes.
        with pytest.raises(ServiceUnavailableException):
            await reload_source_config(session, _CATALOG_SPEC, None)

        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(
                source_type=SourceTypeEnum.URL,
                url="https://example.com/model-catalog.yaml",
            ),
        )
        assert config.custom.url == "https://example.com/model-catalog.yaml"
        assert config.changed is True
        assert set(await _catalog_materialized(session)) == {"Remote"}

        stored = await CatalogSource.one_by_field(
            session, "name", CUSTOM_CATALOG_SOURCE_NAME
        )
        first_updated_at = stored.updated_at

        # Same document: unchanged, and the row is left untouched.
        result = await reload_source_config(session, _CATALOG_SPEC, None)
        assert result.changed is False
        stored = await CatalogSource.one_by_field(
            session, "name", CUSTOM_CATALOG_SOURCE_NAME
        )
        assert stored.updated_at == first_updated_at

        # The remote document moves: the reload applies it, lenient like the
        # background refresh of this same URL.
        moved = yaml.safe_load(_catalog("Moved"))
        moved["model_sets"].append(
            {"name": "Newer", "specs": [{"source": "future-hub"}]}
        )
        _REMOTE["doc"] = yaml.safe_dump(moved)
        result = await reload_source_config(session, _CATALOG_SPEC, None)
        assert result.changed is True
        assert set(await _catalog_materialized(session)) == {"Moved"}

        # Switching to a FILE source carrying the same document: the content did
        # not move, but the configuration did and still has to be persisted.
        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(source_type=SourceTypeEnum.FILE, content=_catalog("Moved")),
        )
        assert config.changed is False
        assert config.custom.source_type == SourceTypeEnum.FILE
        assert config.custom.url is None
        stored = await CatalogSource.one_by_field(
            session, "name", CUSTOM_CATALOG_SOURCE_NAME
        )
        assert stored.source_type == SourceTypeEnum.FILE
        assert stored.url is None

        # A FILE source has no remote document to reload from.
        with pytest.raises(BadRequestException):
            await reload_source_config(session, _CATALOG_SPEC, None)

    @pytest.mark.asyncio
    async def test_the_masked_baseline_survives_the_first_leader_seed(
        self, session, monkeypatch, tmp_path
    ):
        """A PUT before the leader has ever seeded must not be clobbered: the
        seed's create branch enables the baseline unconditionally, so the PUT
        leaves a placeholder row for the seed to fill in."""
        await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(
                source_type=SourceTypeEnum.FILE,
                content=_catalog("Mine"),
            ),
        )

        placeholder = await CatalogSource.one_by_field(
            session, "name", BUILTIN_CATALOG_SOURCE_NAME
        )
        assert placeholder.source_type == SourceTypeEnum.BUILTIN
        assert placeholder.enabled is False
        assert placeholder.content is None

        # Now the leader starts and seeds the packaged catalog.
        catalog_file = tmp_path / "model-catalog.yaml"
        catalog_file.write_text(_catalog("Baseline"))

        @asynccontextmanager
        async def fake_session():
            yield session

        monkeypatch.setattr(controllers, "async_session", fake_session)
        controller = controllers.CatalogSourceController(
            SimpleNamespace(model_catalog_file=str(catalog_file))
        )
        await controller._seed_builtin_source()

        seeded = await CatalogSource.one_by_field(
            session, "name", BUILTIN_CATALOG_SOURCE_NAME
        )
        assert seeded.content is not None  # the seed filled the placeholder in
        assert seeded.enabled is False  # ... and kept it masked
        assert set(await _catalog_materialized(session)) == {"Mine"}

    @pytest.mark.asyncio
    async def test_the_official_address_configures_the_official_source(
        self, session, monkeypatch
    ):
        """Naming the official document's own address is how an admin says
        "follow the official source", so the write takes that path instead of
        storing a URL source that masks it — which would serve the same file the
        weaker way, by plain GET with no index checksum and no release to track.

        The match is against this cluster's own catalog variant: an admin naming
        the other one means that file, not whichever one this deployment
        resolved to.
        """
        monkeypatch.setattr(
            probe_module,
            "_packaged_catalog_filename",
            lambda: "model-catalog-modelscope.yaml",
        )
        _install_fake_url_fetch(monkeypatch)
        _REMOTE["doc"] = _catalog("Remote")
        await _seed_catalog_builtin(session, "Baseline")

        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(
                source_type=SourceTypeEnum.URL,
                url=f"{OTA_SERVER_URL}/model-catalog-modelscope.yaml",
            ),
        )
        assert config.custom is None
        assert config.official.enabled is True
        assert (
            await CatalogSource.one_by_field(
                session, "name", CUSTOM_CATALOG_SOURCE_NAME
            )
            is None
        )

        # The variant this cluster did not resolve to is a document of the
        # admin's own, so it masks the official slot like any other URL.
        config = await update_source_config(
            session,
            _CATALOG_SPEC,
            _upsert(
                source_type=SourceTypeEnum.URL,
                url=f"{OTA_SERVER_URL}/model-catalog.yaml",
            ),
        )
        assert config.custom is not None
        assert config.official.enabled is False


# --- community backends ----------------------------------------------------


def _backends(name: str, *versions: str) -> str:
    """A community-backend document carrying one backend and its versions."""
    return yaml.safe_dump(
        [
            {
                "backend_name": name,
                "version_configs": {
                    version: {"image_name": f"img:{version}"} for version in versions
                },
            }
        ]
    )


async def _seed_backend_builtin(session, content: str) -> None:
    """The BUILTIN row as the leader's seed leaves it."""
    await InferenceBackendSource.create(
        session,
        InferenceBackendSource(
            name=BUILTIN_BACKEND_SOURCE_NAME,
            source_type=SourceTypeEnum.BUILTIN,
            enabled=True,
            content=normalize_backend_yaml(content),
        ),
    )


async def _configure_community(session, content: str):
    """A community-backend source is URL-only, so drive the fake fetch with the
    given document and PUT a URL."""
    _REMOTE["doc"] = content
    return await update_source_config(
        session,
        _COMMUNITY_BACKEND_SPEC,
        _upsert(
            source_type=SourceTypeEnum.URL,
            url="https://example.com/community-backends.yaml",
        ),
    )


async def _materialize_backends(session) -> None:
    """Reconcile as the leader would."""
    await gather_and_merge(session, InferenceBackendSource, reconcile_backend)


async def _committed_source_states(session, operation):
    """The enabled source names each commit inside one operation leaves. Recorded
    on ``after_flush_postexec`` — the last point a listener may still emit SQL."""
    states = []

    def record(sync_session, flush_context):
        # Content-bearing rows only: ``order_source_contents`` drops the empty
        # ones, so a freshly created OFFICIAL placeholder is a name that feeds
        # nothing into the merge.
        names = sync_session.execute(
            select(InferenceBackendSource.name).where(
                InferenceBackendSource.enabled.is_(True),
                InferenceBackendSource.content.is_not(None),
            )
        ).scalars()
        states.append(set(names))

    sa_event.listen(session.sync_session, "after_flush_postexec", record)
    try:
        await operation()
    finally:
        sa_event.remove(session.sync_session, "after_flush_postexec", record)
    return states


async def _commit_count(session, operation) -> int:
    """How many transactions one operation commits. An intermediate commit is what
    makes a half-applied state visible to the leader and to the refresh round."""
    commits = 0

    def record(sync_session):
        nonlocal commits
        commits += 1

    sa_event.listen(session.sync_session, "after_commit", record)
    try:
        await operation()
    finally:
        sa_event.remove(session.sync_session, "after_commit", record)
    return commits


def _assert_never_shrinks_below_the_final_state(states):
    # More than one, so the multi-commit path really is under test; the exact
    # count is not the point and moves with how many rows a write touches.
    assert len(states) > 1, f"expected a multi-commit write, got {states}"
    final = states[-1]
    # Without this the superset check below passes vacuously on an empty final.
    assert final, "the operation left no enabled source at all"
    for state in states:
        assert state >= final, f"committed {state}, smaller than the final {final}"


async def _configured_content(session):
    source = await InferenceBackendSource.one_by_field(
        session, "name", CUSTOM_BACKEND_SOURCE_NAME
    )
    return source.content if source else None


class TestBackendSourceConfig:
    @pytest_asyncio.fixture
    async def session(self):
        async with _make_source_session(
            InferenceBackendSource, InferenceBackend, Model
        ) as session:
            yield session

    @pytest.fixture(autouse=True)
    def _fake_fetch(self, monkeypatch):
        _install_fake_url_fetch(monkeypatch)

    @pytest.mark.asyncio
    async def test_a_source_write_that_would_take_away_a_version_in_use_is_rejected(
        self, session
    ):
        """One check, injected once, covers both manual ways to shrink the merged
        set: a document that drops a version, and a delete that falls back to a
        baseline missing one."""
        await _seed_backend_builtin(session, _backends("my-community", "v2"))
        # The custom document replaces the baseline, so it has to carry both
        # versions for both to stay available.
        await _configure_community(session, _backends("my-community", "v1", "v2"))
        await _materialize_backends(session)

        backend = await InferenceBackend.one_by_field(
            session, "backend_name", "my-community"
        )
        assert backend.backend_source == BackendSourceEnum.COMMUNITY
        assert set(backend.version_configs.root) == {"v1", "v2"}

        model_a = await _deploy(session, "model-a", "my-community", "v1")
        await _deploy(session, "model-b", "my-community", "v2")

        # A document that no longer carries v1.
        with pytest.raises(BadRequestException) as rejected:
            await _configure_community(session, _backends("my-community", "v2", "v3"))
        assert "'v1'" in rejected.value.message and "model-a" in rejected.value.message

        # A document that no longer carries v2 — the baseline still publishes it,
        # but the baseline is masked while a custom source is configured.
        with pytest.raises(BadRequestException) as rejected:
            await _configure_community(session, _backends("my-community", "v1"))
        assert "'v2'" in rejected.value.message and "model-b" in rejected.value.message

        # Dropping the configured document falls back to the baseline, which
        # never carried v1.
        with pytest.raises(BadRequestException) as rejected:
            await delete_source_config(session, _COMMUNITY_BACKEND_SPEC)
        assert "model-a" in rejected.value.message

        # Nothing was written by any of the three: the merge still holds v1 and v2.
        assert await _configured_content(session) is not None
        await _materialize_backends(session)
        backend = await InferenceBackend.one_by_field(
            session, "backend_name", "my-community"
        )
        assert set(backend.version_configs.root) == {"v1", "v2"}

        # Scaled to zero, the very same delete goes through.
        await model_a.update(session, {"replicas": 0})
        config = await delete_source_config(session, _COMMUNITY_BACKEND_SPEC)
        assert config.custom is None

    @pytest.mark.asyncio
    async def test_no_state_committed_mid_write_holds_fewer_sources_than_the_final(
        self, session
    ):
        """Configuring a source flips a baseline over two commits (the custom row
        and the masked baseline are separate rows) and the leader reconciles
        whatever each commit leaves, so the in-between state must hold a *superset*
        of the final source set — a smaller merge orphans community backends a user
        had enabled, and the next reconcile can't restore that. Compared by name.
        """
        await _seed_backend_builtin(session, _backends("my-community", "v1"))

        states = await _committed_source_states(
            session,
            lambda: _configure_community(
                session, _backends("my-community", "v1", "v2")
            ),
        )
        _assert_never_shrinks_below_the_final_state(states)
        assert states[-1] == {CUSTOM_BACKEND_SOURCE_NAME}

    @pytest.mark.asyncio
    async def test_dropping_the_configured_source_is_one_transaction(self, session):
        """Ordering cannot save the way back: restoring the platform layers does
        not disable the custom row (the drop is what removes it), so any commit in
        between holds the custom source and OFFICIAL both enabled — which a refresh
        round landing there reads as "OFFICIAL is masked" and disables, for good.
        """
        await _seed_backend_builtin(session, _backends("my-community", "v1"))
        await _configure_community(session, _backends("my-community", "v1", "v2"))

        commits = await _commit_count(
            session, lambda: delete_source_config(session, _COMMUNITY_BACKEND_SPEC)
        )
        assert commits == 1
        assert await _configured_content(session) is None
        official = await InferenceBackendSource.one_by_field(
            session, "name", OFFICIAL_SOURCE_NAME
        )
        assert official is not None and official.enabled
        builtin = await InferenceBackendSource.one_by_field(
            session, "name", BUILTIN_BACKEND_SOURCE_NAME
        )
        assert builtin.enabled

    @pytest.mark.asyncio
    async def test_the_automatic_path_is_never_gated_by_a_model_in_use(self, session):
        """The automatic path applies official content without a version-in-use
        check (one deployment must not freeze every cluster's updates); the
        reconcile's fallback downgrades an orphaned enabled backend to custom
        instead of deleting.
        """
        await _seed_backend_builtin(session, _backends("my-community", "v1"))
        await _materialize_backends(session)
        backend = await InferenceBackend.one_by_field(
            session, "backend_name", "my-community"
        )
        await backend.update(session, {"enabled": True})
        await _deploy(session, "model-a", "my-community", "v1")

        # The source stops publishing the backend altogether.
        builtin = await InferenceBackendSource.one_by_field(
            session, "name", BUILTIN_BACKEND_SOURCE_NAME
        )
        await builtin.update(session, {"content": normalize_backend_yaml("[]")})
        await _materialize_backends(session)

        backend = await InferenceBackend.one_by_field(
            session, "backend_name", "my-community"
        )
        assert backend.backend_source == BackendSourceEnum.CUSTOM
        assert set(backend.version_configs.root) == {"v1"}

    @pytest.mark.asyncio
    async def test_a_file_source_is_rejected_url_only(self, session):
        """A community-backend source is URL-only; an inline FILE document is a
        400, not a hand-writable surface."""
        with pytest.raises(BadRequestException) as rejected:
            await update_source_config(
                session,
                _COMMUNITY_BACKEND_SPEC,
                _upsert(
                    source_type=SourceTypeEnum.FILE,
                    content=_backends("my-community", "v1"),
                ),
            )
        assert "'url'" in rejected.value.message


# --- built-in backend (runner) versions ------------------------------------


def _packaged_runners(*service_versions) -> list:
    """Stands in for the packaged gpustack-runner catalog."""
    return [
        Runner(
            backend="cuda",
            backend_version="12.4",
            original_backend_version="12.4",
            backend_variant="",
            service="vllm",
            service_version=version,
            platform="linux/amd64",
            docker_image=f"pkg:{version}",
            deprecated=False,
        )
        for version in service_versions
    ]


def _runner_document(*service_versions) -> str:
    return json.dumps(
        [
            {
                "backend": "cuda",
                "service": "vllm",
                "service_version": version,
                "platform": "linux/amd64",
                "docker_image": f"custom:{version}",
            }
            for version in service_versions
        ]
    )


async def _configure_runner(session, document: str):
    """The runner source is URL-only too."""
    _REMOTE["doc"] = document
    return await update_source_config(
        session,
        _BUILTIN_BACKEND_SPEC,
        _upsert(
            source_type=SourceTypeEnum.URL,
            url="https://example.com/runner.json",
        ),
    )


async def _materialize_runners(session) -> None:
    """Reconcile as the leader would: the in-use check reads the materialized
    table, the same way the community-backend check reads ``InferenceBackend``."""
    await gather_and_merge(session, InferenceRunnerSource, reconcile_runner_overrides)


async def _stored_runner_source(session):
    return await InferenceRunnerSource.one_by_field(
        session, "name", CUSTOM_RUNNER_SOURCE_NAME
    )


class TestRunnerSourceConfig:
    """A custom runner document replaces the packaged catalog outright, so a
    forgotten coordinate makes every model pinned to it unschedulable — and the
    failure would only show up at the next placement, far from the write.
    """

    @pytest_asyncio.fixture
    async def session(self):
        # ``Model`` too: the runner source's pre-write check reads deployments.
        async with _make_source_session(
            InferenceRunnerSource, RunnerOverrideEntry, Model
        ) as session:
            yield session

    @pytest.fixture(autouse=True)
    def _fakes(self, monkeypatch):
        """Pin the packaged catalog so assertions don't ride on the installed
        gpustack-runner release, and serve URL fetches from ``_REMOTE``."""
        monkeypatch.setattr(
            runner_source,
            "list_runners",
            lambda **filters: _packaged_runners("0.10.0", "0.11.0"),
        )
        _install_fake_url_fetch(monkeypatch)

    @pytest.mark.asyncio
    async def test_runner_source_is_url_only_with_auto_update_opt_in(self, session):
        """Reopened narrowly for offline / version-chasing: URL only (a FILE
        document is a 400, keeping the version-record footgun shut), with
        auto-refresh off until the admin opts in by the hour. Runner has no
        BUILTIN row, so the config reads cleanly without one.
        """
        assert (await get_source_config(session, _BUILTIN_BACKEND_SPEC)).custom is None

        # A FILE document is not accepted.
        with pytest.raises(BadRequestException) as rejected:
            await update_source_config(
                session,
                _BUILTIN_BACKEND_SPEC,
                _upsert(
                    source_type=SourceTypeEnum.FILE,
                    content=_runner_document("0.11.0"),
                ),
            )
        assert "'url'" in rejected.value.message

        # A URL source stores; auto-refresh defaults off.
        config = await _configure_runner(session, _runner_document("0.11.0"))
        assert config.custom is not None
        assert config.custom.auto_update_hours == 0

        # Opt in to auto-refresh every 6 hours.
        _REMOTE["doc"] = _runner_document("0.11.0")
        config = await update_source_config(
            session,
            _BUILTIN_BACKEND_SPEC,
            _upsert(
                source_type=SourceTypeEnum.URL,
                url="https://example.com/runner.json",
                auto_update_hours=6,
            ),
        )
        assert config.custom.auto_update_hours == 6
        assert (await _stored_runner_source(session)).auto_update_hours == 6

        # A negative cadence is rejected on the custom half too.
        with pytest.raises(ValidationError):
            _upsert(
                source_type=SourceTypeEnum.URL,
                url="https://example.com/runner.json",
                auto_update_hours=-1,
            )

    @pytest.mark.asyncio
    async def test_a_document_that_drops_a_deployed_version_is_rejected(self, session):
        """The two manual ways to shrink the coordinate set are a document that
        leaves one out and a delete that falls back to a packaged catalog missing
        it. Both name the models rather than failing later at placement.
        """
        await _deploy(session, "model-a", BackendEnum.VLLM.value, "0.11.0")
        await _deploy(session, "model-b", BackendEnum.VLLM.value, "0.12.0")

        # A document carrying the deployed versions goes through.
        await _configure_runner(session, _runner_document("0.11.0", "0.12.0"))
        await _materialize_runners(session)
        assert await _stored_runner_source(session) is not None

        # One that drops them does not — and every offending version is named in
        # the one refusal, so a document missing several is fixed in one pass.
        with pytest.raises(BadRequestException) as rejected:
            await _configure_runner(session, _runner_document("0.13.0"))
        for named in ("'0.11.0'", "model-a", "'0.12.0'", "model-b"):
            assert named in rejected.value.message
        # Nothing was written.
        stored = await _stored_runner_source(session)
        assert json.loads(stored.content)[0]["service_version"] in ("0.11.0", "0.12.0")

        # Deleting falls back to the packaged catalog, which carries 0.11.0 but
        # not 0.12.0 — so it goes through only once nothing pins the latter.
        model_b = await Model.one_by_field(session, "name", "model-b")
        await model_b.update(session, {"replicas": 0})
        config = await delete_source_config(session, _BUILTIN_BACKEND_SPEC)
        assert config.custom is None

    @pytest.mark.asyncio
    async def test_a_delete_that_falls_back_to_a_catalog_without_the_version(
        self, session
    ):
        """The packaged catalog never carried the version the custom document
        introduced, so dropping the document takes it away."""
        await _configure_runner(session, _runner_document("0.11.0", "9.9.9"))
        await _materialize_runners(session)
        await _deploy(session, "model-b", BackendEnum.VLLM.value, "9.9.9")

        with pytest.raises(BadRequestException) as rejected:
            await delete_source_config(session, _BUILTIN_BACKEND_SPEC)
        assert "'9.9.9'" in rejected.value.message
        assert "model-b" in rejected.value.message

    @pytest.mark.asyncio
    async def test_falling_back_to_the_packaged_catalog_answers_to_the_same_check(
        self, session
    ):
        """Falling the built-in backend versions back to the packaged catalog is
        the escape hatch for a bad official document, and it shrinks the
        coordinate set like any other manual write — so it is refused while a
        deployment pins a version only the official document carries, and allowed
        once nothing does.
        """
        await InferenceRunnerSource.create(
            session,
            InferenceRunnerSource(
                name=OFFICIAL_SOURCE_NAME,
                source_type=SourceTypeEnum.OFFICIAL,
                enabled=True,
                content=runner_source.normalize_runner_json(
                    _runner_document("0.11.0", "9.9.9")
                ),
            ),
        )
        await _materialize_runners(session)
        await _deploy(session, "model-pinned", BackendEnum.VLLM.value, "9.9.9")

        with pytest.raises(BadRequestException) as rejected:
            await update_source_config(
                session, _BUILTIN_BACKEND_SPEC, _upsert(remote_enabled=False)
            )
        assert "'9.9.9'" in rejected.value.message
        assert "model-pinned" in rejected.value.message
        # Refused before anything landed: the slot is still serving.
        still_serving = await InferenceRunnerSource.one_by_field(
            session, "name", OFFICIAL_SOURCE_NAME
        )
        assert still_serving.enabled is True

        pinned = await Model.one_by_field(session, "name", "model-pinned")
        await pinned.update(session, {"replicas": 0})
        config = await update_source_config(
            session, _BUILTIN_BACKEND_SPEC, _upsert(remote_enabled=False)
        )
        assert config.remote_enabled is False

        # Nothing overrides the packaged catalog any more, so that is what the
        # cluster resolves images from.
        await _materialize_runners(session)
        overrides = await RunnerOverrideEntry.all(session)
        assert overrides == []
        assert {
            runner.service_version for runner in runner_source.merged_runners(overrides)
        } == {"0.10.0", "0.11.0"}

        # Re-asserting the fall-back has nothing left to take away.
        assert (
            await update_source_config(
                session, _BUILTIN_BACKEND_SPEC, _upsert(remote_enabled=False)
            )
        ).remote_enabled is False

    @pytest.mark.asyncio
    async def test_what_the_check_deliberately_ignores(self, session):
        """Four deployments the check must not fire on: one on a version that was
        already unavailable before this write (a pre-existing state must not lock
        the admin out of fixing the catalog), one scaled to zero, one on a
        community backend (its images come from ``version_configs``, not this
        catalog) and one pinning no version at all (Auto resolves at placement).
        """
        await _deploy(session, "model-stale", BackendEnum.VLLM.value, "0.0.1")
        scaled_to_zero = await _deploy(
            session, "model-off", BackendEnum.VLLM.value, "0.10.0"
        )
        await scaled_to_zero.update(session, {"replicas": 0})
        await _deploy(session, "model-community", "my-community", "0.10.0")
        await _deploy(session, "model-auto", BackendEnum.VLLM.value, "")

        # None of the four pins a version this write takes away.
        await _configure_runner(session, _runner_document("0.12.0"))
        stored = await _stored_runner_source(session)
        assert json.loads(stored.content)[0]["service_version"] == "0.12.0"


# --- refresh status and manual trigger --------------------------------------


def _fake_ota_server(monkeypatch, documents: dict) -> list:
    """An OTA server serving ``documents`` plus an index over them, returning the
    list of filenames it is asked for — which is how a scoped refresh is told
    apart from a full round."""
    index = yaml.safe_dump(
        {
            "files": {
                filename: {"ref": "v1.2.3", "sha256": sha256_of(text)}
                for filename, text in documents.items()
            }
        }
    )
    requested = []

    def handle(request: httpx.Request) -> httpx.Response:
        filename = request.url.path.rsplit("/", 1)[-1]
        requested.append(filename)
        text = index if filename == "index.yaml" else documents[filename]
        return httpx.Response(200, text=text)

    def make_client(**kwargs):
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handle), **kwargs)

    async def resolve_nothing(request: httpx.Request) -> None:
        """The fake transport never connects, so nothing needs to resolve."""

    monkeypatch.setattr(probe_module.httpx, "AsyncClient", make_client)
    monkeypatch.setattr(probe_module, "reject_a_forbidden_address", resolve_nothing)
    monkeypatch.setattr(probe_module, "_applied_ref", {})
    monkeypatch.setattr(probe_module, "_last_refresh_attempt", {})
    return requested


class TestSourceProbe:
    @pytest_asyncio.fixture
    async def session(self):
        async with _make_source_session(
            CatalogSource, InferenceBackendSource, InferenceRunnerSource
        ) as session:
            yield session

    @pytest.fixture(autouse=True)
    def _packaged_catalog_variant(self, monkeypatch):
        """Pin which catalog variant this cluster resolved. Unpatched it would
        probe the network, and the ModelScope variant is the interesting one: it
        is the answer a masked kind used to get wrong."""
        monkeypatch.setattr(
            probe_module,
            "_packaged_catalog_filename",
            lambda: "model-catalog-modelscope.yaml",
        )

    @pytest.mark.asyncio
    async def test_status_reads_the_slots_from_the_rows_and_the_round_from_memory(
        self, session, monkeypatch
    ):
        """Per-kind slot fields come from the source rows (any server answers
        them); the applied tag and last error only exist where the refresher
        runs, and the response says which server that is.
        """
        await CatalogSource.create(
            session,
            CatalogSource(
                name=OFFICIAL_SOURCE_NAME,
                source_type=SourceTypeEnum.OFFICIAL,
                content="model_sets: []",
                content_hash="content-hash",
                remote_hash="remote-hash",
                auto_update_hours=12,
            ),
        )
        # A slot only reports a ref while it is the one serving, so the sibling
        # asserted on below needs its row too.
        await InferenceBackendSource.create(
            session,
            InferenceBackendSource(
                name=OFFICIAL_SOURCE_NAME,
                source_type=SourceTypeEnum.OFFICIAL,
                content="inference_backends: []",
            ),
        )

        # A standby: it reports what is stored and admits it isn't refreshing.
        status = await source_probe_status(session, None)
        assert status.refreshing_on_this_server is False
        # Enough for a client to link straight at a published document: the mirror
        # this cluster reads, joined with the file each kind is published as. It
        # is resolved, not read off the last round, so it is right before any
        # round has run at all.
        assert status.ota_server_url == "https://ota.gpustack.ai/latest"
        assert status.kinds["catalog"].filename == "model-catalog-modelscope.yaml"
        assert status.kinds["built-in-backend"].filename == "runner.py.json"
        assert status.kinds["catalog"].source_type == SourceTypeEnum.OFFICIAL
        assert status.kinds["catalog"].official_masked is False
        assert status.kinds["catalog"].auto_update_hours == 12
        assert status.kinds["catalog"].remote_hash == "remote-hash"
        assert status.kinds["catalog"].content_hash == "content-hash"
        assert status.kinds["catalog"].updated_at is not None
        assert status.kinds["catalog"].effective_tag is None
        # A kind with no row at all: its slot is empty, not an error.
        assert status.kinds["built-in-backend"].source_type is None
        assert status.kinds["built-in-backend"].official_masked is False
        assert status.refreshed_at is None

        # The leader adds the round's errors, and the tag each kind's stored
        # content came from (process state, so it outlives the round that
        # fetched it).
        monkeypatch.setattr(
            probe_module,
            "_applied_ref",
            {
                ("catalog_sources", OFFICIAL_SOURCE_NAME): OfficialRef(
                    ref="v2.2.3", filename="model-catalog-modelscope.yaml"
                ),
                ("inference_backend_sources", OFFICIAL_SOURCE_NAME): OfficialRef(
                    ref="9f8e7d6", ref_kind="main"
                ),
            },
        )
        refresher = SourceRefresher()
        refresher.last_round = RefreshRound(
            refreshed_at=datetime.now(timezone.utc),
            errors={"built-in-backend": "no latest release"},
        )
        status = await source_probe_status(session, refresher)
        assert status.refreshing_on_this_server is True
        assert status.refreshed_at is not None
        assert status.kinds["catalog"].effective_tag == "v2.2.3"
        assert status.kinds["catalog"].effective_ref_kind == "release"
        # A document pinned to main reports that commit and says which line it
        # is on.
        assert status.kinds["community-backend"].effective_tag == "9f8e7d6"
        assert status.kinds["community-backend"].effective_ref_kind == "main"
        # built-in-backend never applied any content, so it is at no ref.
        assert status.kinds["built-in-backend"].effective_tag is None
        assert status.kinds["built-in-backend"].effective_ref_kind is None
        assert status.kinds["built-in-backend"].error == "no latest release"
        assert status.kinds["catalog"].error is None

        # A custom source masks OFFICIAL: the status reports the *active* custom
        # source (its URL/hashes), flags official_masked, and shows no OFFICIAL
        # tag.
        await CatalogSource.create(
            session,
            CatalogSource(
                name="custom",
                source_type=SourceTypeEnum.URL,
                url="https://intranet.example.com/catalog.yaml",
                content="model_sets: []",
                content_hash="custom-content-hash",
                remote_hash="custom-remote-hash",
                enabled=True,
                auto_update_hours=6,
            ),
        )
        catalog = (await source_probe_status(session, refresher)).kinds["catalog"]
        assert catalog.official_masked is True
        assert catalog.source_type == SourceTypeEnum.URL
        assert catalog.url == "https://intranet.example.com/catalog.yaml"
        assert catalog.content_hash == "custom-content-hash"
        assert catalog.auto_update_hours == 6
        # OFFICIAL is not the active source, so its tag is not shown — but the
        # file it publishes still is: a custom source *replaces* that document,
        # so it is exactly what an admin downloads to edit while their own one is
        # configured. A masked kind never enters a refresh round, so this can
        # only be right if the name is resolved rather than remembered from one.
        assert catalog.effective_tag is None
        assert catalog.filename == "model-catalog-modelscope.yaml"

        # Unmasking restores the tag: the rounds after a delete resolve nothing
        # for this kind (its cadence isn't due), but the stored content is still
        # at the tag it was fetched at — reading it off the round would report
        # null here.
        custom = await CatalogSource.one_by_field(session, "name", "custom")
        await custom.delete(session)
        refresher.last_round = RefreshRound(refreshed_at=datetime.now(timezone.utc))
        catalog = (await source_probe_status(session, refresher)).kinds["catalog"]
        assert catalog.official_masked is False
        assert catalog.effective_tag == "v2.2.3"

    @pytest.mark.asyncio
    async def test_a_slot_fallen_back_to_the_baseline_reports_that_it_is_not_serving(
        self, session, monkeypatch
    ):
        """A slot the admin took out of service still holds the document it last
        fetched, so the status has to say it is not the one serving — reporting
        its ref would read as "official content is live" while the packaged
        baseline is what runs."""
        await CatalogSource.create(
            session,
            CatalogSource(
                name=OFFICIAL_SOURCE_NAME,
                source_type=SourceTypeEnum.OFFICIAL,
                enabled=False,
                content="model_sets: []",
                content_hash="content-hash",
            ),
        )
        monkeypatch.setattr(
            probe_module,
            "_applied_ref",
            {("catalog_sources", OFFICIAL_SOURCE_NAME): OfficialRef(ref="v2.2.3")},
        )

        catalog = (await source_probe_status(session, SourceRefresher())).kinds[
            "catalog"
        ]
        # Not masked by a custom source — this one was switched off deliberately.
        assert catalog.official_masked is False
        assert catalog.remote_enabled is False
        assert catalog.effective_tag is None
        # The stored document is still reported, so an admin can see what they
        # would be lifting the fall-back back onto.
        assert catalog.content_hash == "content-hash"

    @pytest.mark.asyncio
    async def test_the_manual_trigger_runs_the_scheduled_round_or_says_why_it_cannot(
        self, session, monkeypatch
    ):
        """One implementation for both paths: the manual trigger calls the very
        round the schedule calls — forced, so it checks instead of no-op'ing
        inside the cadence — and refuses with the reason when this server isn't
        it.
        """
        calls = []

        async def fake_refresh_sources(open_session, force=False, ota_server_url=None):
            calls.append((force, ota_server_url))
            return RefreshRound(changed={"catalog": True})

        monkeypatch.setattr(probe_module, "refresh_sources", fake_refresh_sources)

        @asynccontextmanager
        async def fake_session():
            yield session

        monkeypatch.setattr(probe_module, "async_session", fake_session)

        refresher = SourceRefresher(ota_server_url="https://mirror.example.com")
        result = await run_source_probe(refresher)
        assert result.changed == {"catalog": True}
        # The refresher's configuration reaches the round, and the trigger forces it.
        assert calls == [(True, "https://mirror.example.com")]
        # The round is remembered, so the status API reflects the manual run too.
        assert refresher.last_round is result

        # This server isn't the one refreshing.
        with pytest.raises(ServiceUnavailableException):
            await run_source_probe(None)

    @pytest.mark.asyncio
    async def test_reloading_a_kind_that_follows_the_ota_server_stays_on_that_kind(
        self, session, monkeypatch
    ):
        """A reload with no document of the admin's own configured refreshes that
        kind's official slot — and only that one, which is the whole point of it
        not being the global round: the siblings are neither fetched nor even
        given a row. Its outcome joins the reported round instead of replacing it.
        """
        requested = _fake_ota_server(
            monkeypatch,
            {
                "model-catalog-modelscope.yaml": _catalog("Published"),
                "community-inference-backends.yaml": "inference_backends: []",
                "runner.py.json": "[]",
            },
        )
        refresher = SourceRefresher(ota_server_url="https://ota.example.com/latest")
        # What a full round left behind, which a per-kind refresh has no business
        # discarding.
        round_at = datetime.now(timezone.utc)
        refresher.last_round = RefreshRound(
            refreshed_at=round_at,
            errors={"community-backend": "the OTA server could not be read"},
        )

        result = await reload_source_config(session, _CATALOG_SPEC, refresher)
        assert result.changed is True
        assert result.custom is None
        assert result.official.content_hash is not None
        official = await CatalogSource.one_by_field(
            session, "name", OFFICIAL_SOURCE_NAME
        )
        assert "Published" in official.content
        # The index serves every kind, but only this one's document was taken —
        # and the siblings have no row at all.
        assert requested == ["index.yaml", "model-catalog-modelscope.yaml"]
        for source_cls in (InferenceBackendSource, InferenceRunnerSource):
            assert (
                await source_cls.one_by_field(session, "name", OFFICIAL_SOURCE_NAME)
                is None
            )
        # Folded in, not replaced: the sibling's error survives and the full
        # round's timestamp still dates that round.
        assert refresher.last_round.changed == {"catalog": True}
        assert refresher.last_round.errors == {
            "community-backend": "the OTA server could not be read"
        }
        assert refresher.last_round.refreshed_at == round_at

        # Unmoved document: the checksum in the index answers it, so nothing is
        # downloaded and nothing is written. With the cadence turned off as well,
        # the index fetch is what proves the button still checks — that setting
        # withholds consent from the schedule, not from this press.
        requested.clear()
        await official.update(session, {"auto_update_hours": 0})
        assert (
            await reload_source_config(session, _CATALOG_SPEC, refresher)
        ).changed is False
        assert requested == ["index.yaml"]

        # Fallen back to the packaged baseline: there is nothing to refresh into
        # service, and saying "unchanged" would read as up to date.
        await official.update(session, {"enabled": False})
        with pytest.raises(BadRequestException):
            await reload_source_config(session, _CATALOG_SPEC, refresher)
        assert "catalog" in refresher.last_round.errors
        assert "catalog" not in refresher.last_round.changed

        # The official slot lives on the leader, so a standby says so rather than
        # refreshing it from the wrong process.
        with pytest.raises(ServiceUnavailableException):
            await reload_source_config(session, _CATALOG_SPEC, None)
