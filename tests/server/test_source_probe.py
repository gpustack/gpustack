import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio
import yaml
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.catalog_source import CatalogSource, normalize_catalog_yaml
from gpustack.schemas.inference_backend_source import InferenceBackendSource
from gpustack.schemas.runner_source import InferenceRunnerSource
from gpustack.server.sources import probe
from gpustack.server.sources import routes as sources_routes
from gpustack.schemas.source import SourceTypeEnum
from gpustack.server.sources.core import OFFICIAL_SOURCE_NAME, sha256_of
from gpustack.server.sources.probe import OFFICIAL_DEFAULT_HOURS, refresh_sources

_REAL_ASYNC_CLIENT = httpx.AsyncClient

# Files the OFFICIAL kinds read from the mirror (catalog is monkeypatched to this
# variant, the others are fixed on the descriptor).
_CATALOG_FILE = "model-catalog.yaml"
_BACKEND_FILE = "community-inference-backends.yaml"
_RUNNER_FILE = "runner.py.json"

# Stands in for the real mirror, so the tests don't ride on its URL. The path
# segment is deliberately not the real one (a mirror of your own may live at any
# path), and the trailing slash is deliberate too (a configured URL often has one).
_MIRROR_URL = "https://mirror.example.com/mirrored/"
_INDEX_URL = "https://mirror.example.com/mirrored/index.yaml"

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_HOUR = timedelta(hours=1)


def _catalog(*names) -> str:
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


def _backends(version="v1") -> str:
    # Not a built-in engine name: those are reserved, so a community document
    # naming one is rejected outright.
    return yaml.safe_dump(
        [
            {
                "backend_name": "llama.cpp",
                "version_configs": {version: {"image_name": "img"}},
            }
        ]
    )


def _runners(*versions) -> str:
    return json.dumps(
        [
            {
                "backend": "cuda",
                "service": "vllm",
                "service_version": version,
                "platform": "linux/amd64",
                "docker_image": f"img:{version}",
            }
            for version in versions
        ]
    )


class FakeMirror:
    """A stand-in for the official content mirror: ``index.yaml`` publishes a ref
    and a sha256 per file, and every other path serves a file by name. The
    published sha256 drives the download decision; the file bytes drive whether a
    download is then written. Requests are counted so a no-op round can be
    asserted to download nothing.

    ``refs`` is keyed by source repo purely for convenience (one assignment moves
    every file published from it); ``pinned`` holds the files published off
    ``main`` instead, which is what the index reports per file.
    """

    def __init__(self):
        self.refs = {"gpustack/gpustack": "v1", "gpustack/runner": "v1"}
        self.repos = {
            _CATALOG_FILE: "gpustack/gpustack",
            _BACKEND_FILE: "gpustack/gpustack",
            _RUNNER_FILE: "gpustack/runner",
        }
        self.files = {
            _CATALOG_FILE: _catalog("Official"),
            _BACKEND_FILE: _backends(),
            _RUNNER_FILE: _runners("0.12.0"),
        }
        # Files whose published sha256 is made to disagree with what is served,
        # standing in for a mirror caught mid-sync.
        self.mid_sync = set()
        # file name → the commit it is pinned to on main.
        self.pinned = {}
        self.index_requests = []
        self.file_requests = []

    def _index(self) -> str:
        return yaml.safe_dump(
            {
                "version": 1,
                "files": {
                    name: {
                        "ref": self.pinned.get(name) or self.refs[repo],
                        "ref_kind": "main" if name in self.pinned else "release",
                        "sha256": (
                            "0" * 64
                            if name in self.mid_sync
                            else sha256_of(self.files[name])
                        ),
                    }
                    for name, repo in self.repos.items()
                },
            }
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        name = url.rsplit("/", 1)[-1]
        if name == "index.yaml":
            # The whole URL, so a mis-assembled one (a doubled slash) is visible.
            self.index_requests.append(url)
            return httpx.Response(200, text=self._index())
        self.file_requests.append(name)
        text = self.files.get(name)
        return (
            httpx.Response(200, text=text) if text is not None else httpx.Response(404)
        )


@pytest.fixture(autouse=True)
def forget_refresh_state():
    """The applied-ref memo, the due tracker and the "revalidated yet" flag stand
    in for a process lifetime; no test may inherit another's."""
    for state in (probe._applied_ref, probe._last_refresh_attempt):
        state.clear()
    probe._revalidated_since_start = False
    yield
    for state in (probe._applied_ref, probe._last_refresh_attempt):
        state.clear()
    probe._revalidated_since_start = False


@pytest.fixture
def mirror(monkeypatch):
    """Route every OFFICIAL request to a fake mirror, and pin the catalog variant
    so it never does a real network probe."""
    fake = FakeMirror()

    def make_client(**kwargs):
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(fake.handle), **kwargs)

    async def resolve_nothing(request: httpx.Request) -> None:
        pass

    monkeypatch.setattr(probe, "OTA_SERVER_URL", _MIRROR_URL)
    monkeypatch.setattr(probe.httpx, "AsyncClient", make_client)
    monkeypatch.setattr(probe, "reject_a_forbidden_address", resolve_nothing)
    monkeypatch.setattr(
        probe, "get_builtin_model_catalog_file", lambda: "model-catalog.yaml"
    )
    return fake


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[
                CatalogSource.__table__,
                InferenceBackendSource.__table__,
                InferenceRunnerSource.__table__,
            ],
        )
    async with AsyncSession(engine, expire_on_commit=False) as open_session:
        yield open_session
    await engine.dispose()


async def _official(session, source_cls):
    return await source_cls.one_by_field(session, "name", OFFICIAL_SOURCE_NAME)


def _kind(name):
    return next(kind for kind in probe.OFFICIAL_KINDS if kind.name == name)


@pytest.mark.asyncio
async def test_a_round_applies_each_slot_then_returns_early_at_two_levels(
    session, mirror
):
    """One index fetch serves every kind. The two fast returns: a published
    checksum matching the stored one skips a kind with zero downloads — even when
    the release ref moved, since the decision is per file — and a document that
    normalizes to the stored text is downloaded and then not written.
    """
    result = await refresh_sources(session, now=_T0)
    assert result.changed == {
        "catalog": True,
        "community-backend": True,
        "built-in-backend": True,
    }
    assert result.errors == {}
    # One index serves all three kinds, at the configured URL
    # (whose trailing slash must not double up).
    assert mirror.index_requests == [_INDEX_URL]
    assert len(mirror.file_requests) == 3

    stored = await _official(session, CatalogSource)
    assert stored.source_type == SourceTypeEnum.OFFICIAL
    assert stored.enabled is True
    assert stored.auto_update_hours == probe.OFFICIAL_DEFAULT_HOURS
    assert stored.content == normalize_catalog_yaml(_catalog("Official"))
    assert stored.remote_hash == sha256_of(_catalog("Official"))
    first_updated_at = stored.updated_at

    # Level 1: nothing moved — the index is read, nothing is downloaded.
    mirror.index_requests.clear()
    mirror.file_requests.clear()
    result = await refresh_sources(session, now=_T0 + 13 * _HOUR)
    assert result.changed == {
        "catalog": False,
        "community-backend": False,
        "built-in-backend": False,
    }
    assert mirror.file_requests == []
    assert len(mirror.index_requests) == 1

    # A release that left these files alone still costs no download, and the
    # reported ref follows it.
    mirror.refs["gpustack/gpustack"] = "v2"
    result = await refresh_sources(session, now=_T0 + 26 * _HOUR)
    assert result.changed["catalog"] is False
    assert mirror.file_requests == []
    assert probe.applied_official_ref(_kind("catalog")).ref == "v2"

    # Level 2: the catalog moved (a comment) but normalizes to the same text — it
    # is downloaded and then not written.
    mirror.files[_CATALOG_FILE] = _catalog("Official") + "\n# regenerated\n"
    result = await refresh_sources(session, now=_T0 + 39 * _HOUR)
    assert result.changed["catalog"] is False
    assert mirror.file_requests == [_CATALOG_FILE]
    stored = await _official(session, CatalogSource)
    assert stored.updated_at == first_updated_at
    assert stored.remote_hash == sha256_of(_catalog("Official"))


@pytest.mark.asyncio
async def test_a_rejected_document_costs_its_own_kind_and_is_retried(session, mirror):
    """A kind is rejected either because its content is invalid, or because the
    document doesn't match the sha256 the index publishes — the index and
    the document are two fetches, so a mirror caught mid-sync can serve one the
    other doesn't describe, and storing it would make the reported ref lie about
    what is stored. Either way the stored content stays, the other kinds land, and
    the applied ref is not advanced, so the next due round re-reads and retries.
    """
    await refresh_sources(session, now=_T0)
    good_catalog = (await _official(session, CatalogSource)).content

    # A release whose catalog is invalid; its same-repo sibling still lands.
    mirror.refs["gpustack/gpustack"] = "v2"
    mirror.files[_CATALOG_FILE] = "model_sets: not-a-list"
    mirror.files[_BACKEND_FILE] = _backends("v2")
    result = await refresh_sources(session, now=_T0 + 13 * _HOUR)
    assert "catalog" in result.errors
    assert result.changed["community-backend"] is True
    assert (await _official(session, CatalogSource)).content == good_catalog
    assert probe.applied_official_ref(_kind("catalog")).ref == "v1"

    # The content is fixed, but the mirror is mid-sync and its index doesn't
    # describe what it serves — refused too.
    mirror.files[_CATALOG_FILE] = _catalog("Fixed")
    mirror.mid_sync.add(_CATALOG_FILE)
    result = await refresh_sources(session, now=_T0 + 26 * _HOUR)
    assert "sha256" in result.errors["catalog"]
    assert (await _official(session, CatalogSource)).content == good_catalog
    assert probe.applied_official_ref(_kind("catalog")).ref == "v1"

    # The mirror finishes syncing and the fixed document lands.
    mirror.mid_sync.clear()
    result = await refresh_sources(session, now=_T0 + 39 * _HOUR)
    assert result.errors == {}
    assert result.changed["catalog"] is True
    assert (await _official(session, CatalogSource)).content == normalize_catalog_yaml(
        _catalog("Fixed")
    )


@pytest.mark.asyncio
async def test_the_index_reports_which_line_each_document_came_from(session, mirror):
    """``ref`` is any ref string, so a document pinned to main reports that
    commit and a ``ref_kind`` of "main" while its same-repo sibling stays on the
    release line. An index that omits ``ref_kind`` still applies — it is display
    metadata, not what the round is trusted on.
    """
    await refresh_sources(session, now=_T0)
    assert probe.applied_official_ref(_kind("catalog")).ref_kind == "release"

    mirror.pinned[_CATALOG_FILE] = "9f8e7d6"
    mirror.files[_CATALOG_FILE] = _catalog("FromMain")
    result = await refresh_sources(session, now=_T0 + 13 * _HOUR)

    assert result.refs["catalog"].ref == "9f8e7d6"
    assert result.refs["catalog"].ref_kind == "main"
    assert result.refs["community-backend"].ref == "v1"
    assert result.refs["community-backend"].ref_kind == "release"
    assert probe.applied_official_ref(_kind("catalog")).ref == "9f8e7d6"

    # An index without ``ref_kind`` reads as the release line.
    mirror.pinned.clear()
    mirror.refs["gpustack/gpustack"] = "v2"
    mirror.files[_CATALOG_FILE] = _catalog("BackOnRelease")
    monkeypatched = mirror._index

    def index_without_ref_kind() -> str:
        parsed = yaml.safe_load(monkeypatched())
        for entry in parsed["files"].values():
            entry.pop("ref_kind")
        return yaml.safe_dump(parsed)

    mirror._index = index_without_ref_kind
    result = await refresh_sources(session, now=_T0 + 26 * _HOUR)
    assert result.refs["catalog"].ref == "v2"
    assert result.refs["catalog"].ref_kind == "release"


@pytest.mark.asyncio
async def test_the_first_round_after_a_start_re_reads_stored_content(
    session, mirror, monkeypatch
):
    """A restart may carry a new normalizer. The hashes say the document hasn't
    moved — true but beside the point, since the stored text came from the old
    normalizer — so the first round after a start re-reads it.
    """
    await refresh_sources(session, now=_T0)
    first_updated_at = (await _official(session, CatalogSource)).updated_at

    # A restart whose normalizer now emits something else, the document unchanged.
    def new_normalizer(raw: str) -> str:
        return normalize_catalog_yaml(raw) + "\n# emitted by the new normalizer\n"

    monkeypatch.setattr(
        probe,
        "OFFICIAL_KINDS",
        (
            probe.OfficialKind(
                "catalog",
                CatalogSource,
                new_normalizer,
                "gpustack/gpustack",
                _CATALOG_FILE,
            ),
        ),
    )
    probe._revalidated_since_start = False

    mirror.file_requests.clear()
    result = await refresh_sources(session, now=_T0 + _HOUR)
    assert mirror.file_requests == [_CATALOG_FILE]
    assert result.changed["catalog"] is True
    stored = await _official(session, CatalogSource)
    assert stored.content.endswith("# emitted by the new normalizer\n")
    assert stored.updated_at != first_updated_at


@pytest.mark.asyncio
async def test_a_source_refreshes_only_once_its_cadence_is_due(
    session, mirror, monkeypatch
):
    """``auto_update_hours`` gates a round: after the first (forced) round, a
    source is skipped until its cadence has elapsed. An explicit trigger passes
    ``force``, which skips the cadence — but not the opt-in, and not the checksum.
    """
    await refresh_sources(session, now=_T0)

    # Well within 12h: nothing is due, so not even the index is read.
    mirror.index_requests.clear()
    mirror.file_requests.clear()
    result = await refresh_sources(session, now=_T0 + 5 * _HOUR)
    assert result.changed == {}
    assert mirror.index_requests == []

    # Past 12h: due again (checksums unchanged, so still nothing downloaded).
    result = await refresh_sources(session, now=_T0 + 13 * _HOUR)
    assert mirror.index_requests
    assert mirror.file_requests == []

    # A URL source that never opted in, plus a fetch that fails the test if the
    # forced round reaches for it anyway (``_refresh_user_urls`` records the
    # failure rather than raising, so it surfaces in ``errors``).
    await InferenceRunnerSource.create(
        session,
        InferenceRunnerSource(
            name="custom",
            source_type=SourceTypeEnum.URL,
            url="https://intranet.example.com/runner.json",
            enabled=True,
            auto_update_hours=0,
        ),
    )

    async def must_not_fetch(url: str) -> str:
        raise AssertionError(f"an opted-out source must not be fetched: {url}")

    monkeypatch.setattr(probe, "fetch_source_text", must_not_fetch)

    # Forced inside the cadence: it really checks, downloads nothing (no checksum
    # moved) and leaves the opted-out source alone.
    mirror.index_requests.clear()
    result = await refresh_sources(session, now=_T0 + 14 * _HOUR, force=True)
    assert result.changed["catalog"] is False
    assert mirror.index_requests
    assert mirror.file_requests == []
    assert "inference_runner_sources:custom" not in result.errors


@pytest.mark.asyncio
async def test_a_masked_official_is_not_fetched_and_a_user_url_refreshes(
    session, mirror, monkeypatch
):
    """A user URL source masks its OFFICIAL slot (created disabled, never fetched)
    and refreshes itself when opted in, while the unmasked kinds land as usual.
    """
    stale = normalize_catalog_yaml(_catalog("Stale"))
    await CatalogSource.create(
        session,
        CatalogSource(
            name="custom",
            source_type=SourceTypeEnum.URL,
            url="https://intranet.example.com/catalog.yaml",
            content=stale,
            content_hash=sha256_of(stale),
            remote_hash=sha256_of(_catalog("Stale")),
            enabled=True,
            auto_update_hours=2,
        ),
    )

    async def fake_fetch(url: str) -> str:
        return _catalog("Intranet")

    monkeypatch.setattr(probe, "fetch_source_text", fake_fetch)

    # The user URL refreshes; the masked kind's slot is created disabled and
    # never fetched, while the unmasked kinds' slots are created enabled and
    # fetched.
    result = await refresh_sources(session, now=_T0)
    assert result.changed["catalog_sources:custom"] is True
    custom = await CatalogSource.one_by_field(session, "name", "custom")
    assert custom.content == normalize_catalog_yaml(_catalog("Intranet"))

    masked = await _official(session, CatalogSource)
    assert masked.enabled is False
    assert masked.content is None
    assert "catalog" not in result.changed
    assert result.changed["built-in-backend"] is True
    assert (await _official(session, InferenceRunnerSource)).enabled is True
    assert _CATALOG_FILE not in mirror.file_requests


@pytest.mark.asyncio
async def test_a_kind_that_opted_out_is_skipped_while_the_others_refresh(
    session, mirror
):
    """There is no global switch left: a kind stops refreshing by setting its own
    OFFICIAL row's ``auto_update_hours`` to 0. That is an opt-out, so even a
    forced round leaves it alone — and the other kinds are unaffected.
    """
    await refresh_sources(session, now=_T0)
    catalog_official = await _official(session, CatalogSource)
    await catalog_official.update(session, {"auto_update_hours": 0})

    mirror.files[_CATALOG_FILE] = _catalog("Moved")
    mirror.files[_BACKEND_FILE] = _backends("v9")
    mirror.file_requests.clear()
    result = await refresh_sources(session, now=_T0 + 13 * _HOUR, force=True)

    assert "catalog" not in result.changed
    assert result.changed["community-backend"] is True
    assert _CATALOG_FILE not in mirror.file_requests
    assert (await _official(session, CatalogSource)).content == normalize_catalog_yaml(
        _catalog("Official")
    )


@pytest.mark.asyncio
async def test_configuring_a_url_source_starts_its_cadence(
    session, mirror, monkeypatch
):
    """The config API fetches the document itself (PUT and reload), so the
    schedule must count ``auto_update_hours`` from that fetch. Otherwise the very
    next tick sees no attempt on record, treats the source as due and refetches
    at once — the stored content moves hours before the cadence says it may.

    Real ``now`` here on purpose: the config API stamps its own fetch, so the
    round it races with has to be on the same clock.
    """
    served = {"text": _catalog("Intranet")}

    async def fake_fetch(url: str) -> str:
        return served["text"]

    monkeypatch.setattr(sources_routes, "fetch_source_text", fake_fetch)
    monkeypatch.setattr(probe, "fetch_source_text", fake_fetch)

    spec = sources_routes.SourceConfigSpec(
        source_cls=CatalogSource,
        normalize=normalize_catalog_yaml,
        custom_name="custom",
        builtin_name="builtin",
    )

    # Get the forced first round out of the way: this process is now past it.
    await refresh_sources(session)

    await sources_routes.update_source_config(
        session,
        spec,
        sources_routes.SourceConfigUpsert(
            custom=sources_routes.CustomSourceUpsert(
                source_type=SourceTypeEnum.URL,
                url="https://intranet.example.com/catalog.yaml",
                auto_update_hours=1,
            ),
            official=sources_routes.OfficialSourceUpsert(
                auto_update_hours=OFFICIAL_DEFAULT_HOURS
            ),
        ),
    )
    intranet = normalize_catalog_yaml(_catalog("Intranet"))
    assert (
        await CatalogSource.one_by_field(session, "name", "custom")
    ).content == intranet

    # The remote moves right after the write. A round now is well inside the 1h
    # cadence, so it must leave the source alone.
    served["text"] = _catalog("Moved")
    result = await refresh_sources(session)
    assert "catalog_sources:custom" not in result.changed
    assert (
        await CatalogSource.one_by_field(session, "name", "custom")
    ).content == intranet

    # A manual reload takes the moved document and re-stamps the cadence, so the
    # following round leaves it alone too.
    reload_result = await sources_routes.reload_source_config(session, spec, None)
    assert reload_result.changed is True
    moved = normalize_catalog_yaml(_catalog("Moved"))
    served["text"] = _catalog("MovedAgain")
    result = await refresh_sources(session)
    assert "catalog_sources:custom" not in result.changed
    assert (
        await CatalogSource.one_by_field(session, "name", "custom")
    ).content == moved


@pytest.mark.asyncio
async def test_a_slot_the_admin_took_out_of_service_is_left_alone(session, mirror):
    """An OFFICIAL row's ``enabled`` is also the admin's escape hatch back to the
    packaged baseline, so a round must neither fetch for it nor turn it back on:
    the re-assertion above only ever masks. With every kind out of service a round
    makes no request at all, not even for the index.
    """
    await refresh_sources(session, now=_T0)
    for source_cls in (CatalogSource, InferenceBackendSource, InferenceRunnerSource):
        await (await _official(session, source_cls)).update(session, {"enabled": False})

    mirror.index_requests.clear()
    mirror.file_requests.clear()
    mirror.files[_CATALOG_FILE] = _catalog("Moved")
    mirror.refs["gpustack/gpustack"] = "v2"

    result = await refresh_sources(session, now=_T0 + 13 * _HOUR)
    assert mirror.index_requests == []
    assert mirror.file_requests == []
    assert result.changed == {}
    parked = await _official(session, CatalogSource)
    assert parked.enabled is False
    assert parked.content == normalize_catalog_yaml(_catalog("Official"))

    # Back in service: that kind is picked up again from where the mirror now is,
    # and the two still out of service stay out.
    await parked.update(session, {"enabled": True})
    result = await refresh_sources(session, now=_T0 + 26 * _HOUR)
    assert result.changed == {"catalog": True}
    assert (await _official(session, CatalogSource)).content == normalize_catalog_yaml(
        _catalog("Moved")
    )


@pytest.mark.asyncio
async def test_unmasking_a_kind_refetches_it_even_when_its_sibling_stayed_current(
    session, mirror
):
    """The download decision is per kind (its own row's checksum), not per repo:
    masking catalog while its same-repo sibling (community-backend) refreshes must
    not starve catalog once it is unmasked and the release has not moved (S1
    regression).
    """
    custom_text = normalize_catalog_yaml(_catalog("Custom"))
    custom = await CatalogSource.create(
        session,
        CatalogSource(
            name="custom",
            source_type=SourceTypeEnum.URL,
            url="https://intranet.example.com/catalog.yaml",
            content=custom_text,
            content_hash=sha256_of(custom_text),
            remote_hash=sha256_of(_catalog("Custom")),
            enabled=True,
        ),
    )

    # Round 1: catalog masked (never fetched); community-backend (same repo) is
    # fetched, marking the release applied — but only for community-backend.
    result = await refresh_sources(session, now=_T0)
    assert "catalog" not in result.changed
    assert result.changed["community-backend"] is True
    assert (await _official(session, CatalogSource)).content is None

    # Unmask by dropping the custom source through the config API, which is what
    # unmasks: a round only ever masks, so that it can never overrule an admin
    # who took the official slot out of service.
    await sources_routes.delete_source_config(
        session,
        sources_routes.SourceConfigSpec(
            source_cls=CatalogSource,
            normalize=normalize_catalog_yaml,
            custom_name=custom.name,
            builtin_name="builtin",
        ),
    )
    # The OFFICIAL catalog is still empty and the release has not moved.
    result = await refresh_sources(session, now=_T0 + 13 * _HOUR)

    catalog_official = await _official(session, CatalogSource)
    assert catalog_official.enabled is True
    assert result.changed["catalog"] is True  # not starved by the sibling
    assert catalog_official.content == normalize_catalog_yaml(_catalog("Official"))
