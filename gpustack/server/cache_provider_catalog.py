"""The cache-provider catalog: the declarations a cache service can run.

The catalog is materialized into ``CacheProviderEntry`` by the leader, from the
source rows an admin configures — the same shape the model catalog and the
community backends already have. Readers share a snapshot of that table,
invalidated across servers after the leader commits a catalog rewrite.

What stays here is the packaged side of it: reading the bundled asset and the
ones installed plugins contribute, which is what the leader seeds the baseline
row from and what an admin downloads to edit against.
"""

import asyncio
import logging
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.cache_provider_source import (
    CacheProviderEntry,
    dump_cache_providers,
    load_cache_providers_document,
    merge_cache_providers,
)
from gpustack.schemas.cache_providers import (
    CacheProvider,
    render_injection as _render_injection,
    resolved_field_values,
    with_runner_versions,
)
from gpustack.server.cache import delete_cache_by_key, locked_cached
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

BUNDLED_CATALOG_ASSET = ("gpustack.assets", "cache-providers.yaml")
CATALOG_CACHE_KEY = "CacheProviderEntry.catalog"
_catalog_lock = asyncio.Lock()
_catalog_revision: Optional[str] = None
_catalog: List[CacheProvider] = []


def _catalog_assets() -> List[Tuple[str, str]]:
    """(package, resource) of every catalog asset to read, in precedence
    order: the bundled one, then what each installed plugin ships."""
    from gpustack.extension import iter_plugin_classes

    assets = [BUNDLED_CATALOG_ASSET]
    for name, plugin_class in iter_plugin_classes():
        try:
            assets.extend(plugin_class.cache_provider_assets() or [])
        except Exception:
            logger.warning(
                f"Failed to read cache provider assets from plugin '{name}'",
                exc_info=True,
            )
    return assets


def _load_asset(package: str, resource: str) -> List[CacheProvider]:
    """One asset's declarations. A malformed declaration costs its own provider
    and not the catalog — the others still serve — which is why the lenient
    parse is the one an asset gets."""
    try:
        yaml_file = files(package).joinpath(resource)
        if not yaml_file.is_file():
            logger.warning(f"Cache provider asset {package}/{resource} not found")
            return []
        return load_cache_providers_document(yaml_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load cache providers from {package}/{resource}: {e}")
        return []


def asset_providers() -> List[CacheProvider]:
    """The declarations this installation carries: the bundled asset merged
    with every asset an installed plugin ships.

    Not what serves — that is the materialized table — but what the baseline is
    seeded from, and so what an installation falls back to with no document
    configured.
    """
    providers: List[CacheProvider] = []
    for package, resource in _catalog_assets():
        providers = merge_cache_providers(providers, _load_asset(package, resource))
    return providers


def builtin_catalog_text() -> str:
    """The packaged baseline as a document, in the form an admin edits.

    Seeded onto the BUILTIN source row and offered in the UI as the starting
    point for a document of one's own — which, under replace semantics, is the
    only way an installation carrying plugin-contributed providers can write one
    without losing them.
    """
    return dump_cache_providers(asset_providers())


def providers_from_documents(
    documents: List[Optional[str]], runners: Optional[List[Any]] = None
) -> List[CacheProvider]:
    """The catalog a sequence of documents produces, later documents replacing
    same-named declarations of earlier ones.

    What the source layer hands its checks is a list of documents rather than a
    catalog, so this is how a check judges the catalog a write would produce.

    A document that will not parse is skipped, as the materialization skips it:
    every document is validated when it is written, so one that fails here was
    corrupted after the fact, and it must not be the reason an admin's own write
    is refused — least of all with the unexplained server error that an
    exception out of a pre-write check becomes. The document being written is
    validated before it reaches any check, so nothing is waved through here.
    """
    providers: List[CacheProvider] = []
    for document in documents:
        try:
            loaded = load_cache_providers_document(document)
        except ValueError as e:
            logger.error(f"Skipping unreadable cache provider document: {e}")
            continue
        providers = merge_cache_providers(providers, loaded)
    return with_runner_versions(providers, runners)


@locked_cached(key=CATALOG_CACHE_KEY)
async def _get_catalog_revision() -> str:
    """Apply the shared cache's TTL and distributed invalidation to the snapshot."""
    return uuid4().hex


async def invalidate_cache_provider_catalog() -> None:
    """Invalidate the catalog on all servers after materialization commits."""
    await delete_cache_by_key(_key=CATALOG_CACHE_KEY)


async def _load_catalog(session: AsyncSession) -> List[CacheProvider]:
    entries = await CacheProviderEntry.all(session)
    return [
        CacheProvider(**entry.payload)
        for entry in sorted(entries, key=lambda entry: entry.position)
    ]


async def _get_catalog(session: AsyncSession) -> List[CacheProvider]:
    global _catalog, _catalog_revision
    if session.in_transaction():
        # A caller holding a connection must not wait for a loader that needs
        # another pool slot. Read its transaction's view even when the shared
        # cache is warm, and keep that view out of the shared snapshot.
        return await _load_catalog(session)
    revision = await _get_catalog_revision()
    if revision == _catalog_revision:
        return _catalog
    async with _catalog_lock:
        revision = await _get_catalog_revision()
        if revision != _catalog_revision:
            # A fresh session keeps callers' transaction snapshots and pending
            # writes out of the shared catalog.
            async with async_session() as fresh_session:
                providers = await _load_catalog(fresh_session)
            # Keep the revision captured before the query: an invalidation while
            # it is in flight must force the next reader to reload.
            _catalog, _catalog_revision = providers, revision
        return _catalog


async def get_cache_providers(session: AsyncSession) -> List[CacheProvider]:
    """The catalog in card order, with independent objects per caller."""
    return [provider.model_copy(deep=True) for provider in await _get_catalog(session)]


async def get_cache_provider(
    session: AsyncSession, name: Optional[str]
) -> Optional[CacheProvider]:
    """One provider by name, case-insensitively — the name a cache service
    stores is whatever case the document declared it in."""
    wanted = (name or "").lower()
    for provider in await _get_catalog(session):
        if provider.name.lower() == wanted:
            return provider.model_copy(deep=True)
    return None


def render_injection(
    provider: CacheProvider,
    backend_name: str,
    params: Dict[str, Any],
    framework: Optional[str] = None,
) -> Optional[Tuple[Dict[str, str], List[str], Dict[str, str]]]:
    """
    Render the connector (env, args, files) a given inference backend
    needs to attach to a service of this provider. ``framework`` is the
    engine worker's accelerator framework ("cuda", "cann", ...); it
    selects a framework-scoped integration entry when the provider
    declares one. When params carry the resolver-derived "locality"
    fact, the declaration's matching locality_params bucket fills
    placeholder defaults (explicit params win). Returns None when
    incompatible.
    """
    integration = provider.integration_for(backend_name, framework)
    if integration is None:
        return None
    # Work on a copy: the caller's dict must not accumulate one
    # provider's locality defaults across calls.
    params = dict(params)
    locality = params.get("locality")
    if locality:
        for key, value in (
            integration.injection.locality_params.get(locality) or {}
        ).items():
            params.setdefault(key, value)
    # Every declared field backstops its placeholder — an unresolved
    # {{name}} would render literally and corrupt file contents. A field
    # without a default backfills as "" (an optional field the user left
    # empty, e.g. a device name a TCP transport does not read), matching
    # the managed run-command path where None renders empty and drops
    # with its flag.
    #
    # Resolved rather than declared: a field behind a closed gate carries its
    # gated default, which is the value the launch renders — an engine's
    # segment contribution has to read 0 while a standalone store owns the
    # pool, and the plain default would say otherwise.
    resolved = resolved_field_values(provider.fields, params)
    for field in provider.fields:
        fallback = resolved.get(field.name)
        params.setdefault(field.name, fallback if fallback is not None else "")
    return _render_injection(integration, params)
