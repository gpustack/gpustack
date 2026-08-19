import asyncio

import httpx
import pytest
import yaml
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.catalog_source import (
    KIND_MODEL_SET,
    CatalogModelEntry,
    CatalogSource,
    normalize_catalog_yaml,
    reconcile_catalog,
)
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.inference_backend_source import (
    InferenceBackendSource,
    normalize_backend_yaml,
    reconcile_backend,
)
from gpustack.schemas.source import SourceTypeEnum
from gpustack.server.sources import core as core_module
from gpustack.server.sources.core import fetch_source_text, gather_and_merge

# --- fetching a source document by URL ---------------------------------------

# Captured before any monkeypatching so the fake client can still build a real one.
_REAL_ASYNC_CLIENT = httpx.AsyncClient

# What a host resolves to unless a test says otherwise (an allowed address).
_A_PUBLIC_ADDRESS = "93.184.216.34"


@pytest.fixture
def mock_http(monkeypatch):
    """Route ``fetch_source_text`` through a MockTransport, recording requests.
    Returns an installer taking a handler (and optional ``addresses`` map);
    resolution is stubbed since the transport never connects.
    """

    def install(handler, addresses=None):
        requests = []
        answers = addresses or {}

        def make_client(**kwargs):
            async def record(request):
                requests.append(request)
                response = handler(request)
                if isinstance(response, httpx.Response):
                    return response
                return await response

            return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(record), **kwargs)

        async def resolve(host, port):
            return [(0, 0, 0, "", (answers.get(host, _A_PUBLIC_ADDRESS), 0))]

        monkeypatch.setattr(core_module.httpx, "AsyncClient", make_client)
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
        return requests

    return install


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        None,  # a URL source without a url
        "",
        "file:///etc/passwd",  # non-http(s): local-file read
        "ssh://host/repo.git",  # non-http(s): SSRF
        "ext::sh -c touch",  # non-http(s): transport abuse
        "https://user:pat@host/catalog.yaml",  # credential in the URL
        "https:///catalog.yaml",  # no host
    ],
)
async def test_fetch_source_text_rejects_unusable_urls(url, mock_http):
    """Validation runs before any request, so nothing reaches the network."""
    requests = mock_http(lambda request: httpx.Response(200, text="model_sets: []"))
    with pytest.raises(ValueError):
        await fetch_source_text(url)
    assert requests == []


@pytest.mark.asyncio
async def test_fetch_source_text_follows_redirects(mock_http):
    """A repo raw link redirects to the CDN host; the fetch must follow it."""

    def handler(request):
        if request.url.host == "github.com":
            return httpx.Response(
                302,
                headers={
                    "Location": "https://raw.githubusercontent.com/o/r/main/c.yaml"
                },
            )
        return httpx.Response(200, text="model_sets: []\n")

    requests = mock_http(handler)
    text = await fetch_source_text("https://github.com/o/r/raw/main/c.yaml")

    assert text == "model_sets: []\n"
    assert [request.url.host for request in requests] == [
        "github.com",
        "raw.githubusercontent.com",
    ]
    # A source fetch never authenticates: no credential header is ever sent.
    assert not any("Authorization" in request.headers for request in requests)


@pytest.mark.asyncio
async def test_fetch_source_text_rejects_bad_responses(mock_http):
    # An error status surfaces as a 400 carrying the status code.
    mock_http(lambda request: httpx.Response(404))
    with pytest.raises(ValueError, match="404"):
        await fetch_source_text("https://host/c.yaml")

    # A repo *page* pasted instead of the raw file URL.
    mock_http(lambda request: httpx.Response(200, text="<!DOCTYPE html>\n<html>\n"))
    with pytest.raises(ValueError, match="raw file URL"):
        await fetch_source_text("https://host/blob/main/c.yaml")

    # Oversized body with no Content-Length: the cap must hold while streaming.
    async def body():
        for _ in range(5):
            yield b"x" * (1024 * 1024)

    mock_http(lambda request: httpx.Response(200, content=body()))
    with pytest.raises(ValueError, match="MB limit"):
        await fetch_source_text("https://host/big.yaml")

    # Not UTF-8 at all, rather than stored/hashed with U+FFFD substitutions.
    mock_http(lambda request: httpx.Response(200, content=b"model_sets: \xff\xfe[]"))
    with pytest.raises(ValueError, match="not valid UTF-8"):
        await fetch_source_text("https://host/latin1.yaml")


@pytest.mark.asyncio
async def test_a_source_never_fetches_from_a_forbidden_address(mock_http):
    """Only link-local is refused (the cloud metadata service lives there);
    private and loopback stay reachable. The check runs per redirect hop, so a
    public host cannot 302 its way in.
    """
    addresses = {
        "artifacts.internal": "10.1.2.3",
        "myself.example.com": "127.0.0.1",
        "metadata.example.com": "169.254.169.254",
    }

    def serve(request):
        return httpx.Response(200, text="model_sets: []\n")

    for host in ("artifacts.internal", "myself.example.com"):
        mock_http(serve, addresses)
        assert await fetch_source_text(f"https://{host}/c.yaml")

    mock_http(serve, addresses)
    with pytest.raises(ValueError, match="must not fetch from"):
        await fetch_source_text("https://metadata.example.com/c.yaml")

    def redirect_into_the_metadata_service(request):
        if request.url.host == "public.example.com":
            return httpx.Response(
                302, headers={"Location": "https://metadata.example.com/latest/"}
            )
        return serve(request)

    requests = mock_http(redirect_into_the_metadata_service, addresses)
    with pytest.raises(ValueError, match="must not fetch from"):
        await fetch_source_text("https://public.example.com/c.yaml")
    # The public host was reached; the hop into the metadata service was not.
    assert [request.url.host for request in requests] == ["public.example.com"]


# --- the ordered merge -------------------------------------------------------


def _catalog_content(*model_sets) -> str:
    """A normalized catalog document carrying the given (name, order) sets."""
    return normalize_catalog_yaml(
        yaml.safe_dump(
            {
                "model_sets": [
                    {
                        "name": name,
                        "order": order,
                        "specs": [
                            {
                                "source": "huggingface",
                                "huggingface_repo_id": f"org/{name}-{order}",
                                "mode": "standard",
                            }
                        ],
                    }
                    for name, order in model_sets
                ]
            }
        )
    )


def _backend_content(image_name: str, *extra_names) -> str:
    """A normalized community-backend document with a single shared version, plus
    a card per ``extra_names`` only that document carries."""
    return normalize_backend_yaml(
        yaml.safe_dump(
            [
                {
                    "backend_name": name,
                    "version_configs": {"v1": {"image_name": image_name}},
                }
                for name in ("foo", *extra_names)
            ]
        )
    )


@pytest.mark.asyncio
async def test_gather_and_merge_orders_sources_and_lets_remote_content_replace_the_baseline():
    """Two rules, one merge input (``order_source_contents``):

    - Order: the official slot first, then custom sources by ``(name, id)``
      regardless of name. Ranking by ``source_type`` not name is what stops
      ``Acme`` (which sorts before ``official``) from being overridden by a
      baseline.
    - Whole-content replacement: a remote document carries the kind's entire
      content, so the packaged baseline leaves the merge rather than layering
      under it — which is what lets a remote document withdraw an entry.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[
                CatalogSource.__table__,
                CatalogModelEntry.__table__,
                InferenceBackendSource.__table__,
                InferenceBackend.__table__,
            ],
        )

    async with AsyncSession(engine) as session:
        # "Shared" is contested by all three layers; "Baseline" only between the
        # two baselines; "Both" only between two custom sources; "Packaged" only
        # the packaged baseline carries, so it says whether that layer serves.
        await CatalogSource.create(
            session,
            CatalogSource(
                name="builtin",
                source_type=SourceTypeEnum.BUILTIN,
                content=_catalog_content(
                    ("Shared", 1), ("Baseline", 1), ("Packaged", 1)
                ),
            ),
        )
        await CatalogSource.create(
            session,
            CatalogSource(
                name="official",
                source_type=SourceTypeEnum.OFFICIAL,
                content=_catalog_content(("Shared", 2), ("Baseline", 2)),
            ),
        )
        await CatalogSource.create(
            session,
            CatalogSource(
                name="Acme",
                source_type=SourceTypeEnum.FILE,
                content=_catalog_content(("Shared", 3), ("Both", 3)),
            ),
        )
        await CatalogSource.create(
            session,
            CatalogSource(
                name="Zeta",
                source_type=SourceTypeEnum.FILE,
                content=_catalog_content(("Both", 4)),
            ),
        )

        await gather_and_merge(session, CatalogSource, reconcile_catalog)

        entries = {
            entry.name: entry
            for entry in await CatalogModelEntry.all(session)
            if entry.kind == KIND_MODEL_SET
        }
        # Remote content serves, so the packaged baseline is out of the merge
        # entirely — not merely outranked.
        assert "Packaged" not in entries
        # Custom wins over the official slot.
        assert entries["Shared"].source_name == "Acme"
        assert entries["Shared"].payload["order"] == 3
        # Same-named sets still union their specs across the serving sources.
        assert len(entries["Shared"].payload["specs"]) == 2
        assert entries["Baseline"].source_name == "official"
        assert entries["Baseline"].payload["order"] == 2
        # Between custom sources the later name still wins.
        assert entries["Both"].source_name == "Zeta"
        assert entries["Both"].payload["order"] == 4

        # The same order and the same replacement drive the community-backend
        # reconcile: "packaged-only" is a card only the baseline carries.
        await InferenceBackendSource.create(
            session,
            InferenceBackendSource(
                name="builtin",
                source_type=SourceTypeEnum.BUILTIN,
                content=_backend_content("builtin:v1", "packaged-only"),
            ),
        )
        await InferenceBackendSource.create(
            session,
            InferenceBackendSource(
                name="Acme",
                source_type=SourceTypeEnum.FILE,
                content=_backend_content("acme:v1"),
            ),
        )

        await gather_and_merge(session, InferenceBackendSource, reconcile_backend)

        backend = await InferenceBackend.one_by_fields(
            session, {"backend_name": "foo", "owner_principal_id": None}
        )
        assert backend.source_name == "Acme"
        assert backend.version_configs.root["v1"].image_name == "acme:v1"
        assert backend.version_configs.root["v1"].source_name == "Acme"
        # The custom document is the whole content, so a card only the packaged
        # baseline carried is not materialized at all.
        assert (
            await InferenceBackend.one_by_fields(
                session, {"backend_name": "packaged-only", "owner_principal_id": None}
            )
            is None
        )

    await engine.dispose()
