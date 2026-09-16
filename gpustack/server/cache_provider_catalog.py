"""The cache-provider catalog: the declarations a cache service can run.

The catalog is materialized into ``CacheProviderEntry`` by the leader, from the
source rows an admin configures — the same shape the model catalog and the
community backends already have. Readers query that table, so every server
serves what was last written without any process state to keep in step.

What stays here is the packaged side of it: reading the bundled asset and the
ones installed plugins contribute, which is what the leader seeds the baseline
row from and what an admin downloads to edit against.
"""

import logging
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

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
)

logger = logging.getLogger(__name__)

BUNDLED_CATALOG_ASSET = ("gpustack.assets", "cache-providers.yaml")


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


def providers_from_documents(documents: List[Optional[str]]) -> List[CacheProvider]:
    """The catalog a sequence of documents produces, later documents replacing
    same-named declarations of earlier ones.

    What the source layer hands its checks is a list of documents rather than a
    catalog, so this is how a check judges the catalog a write would produce.
    """
    providers: List[CacheProvider] = []
    for document in documents:
        providers = merge_cache_providers(
            providers, load_cache_providers_document(document)
        )
    return providers


async def get_cache_providers(session: AsyncSession) -> List[CacheProvider]:
    """The catalog as it serves, in card order."""
    entries = await CacheProviderEntry.all(session)
    return [
        CacheProvider(**entry.payload)
        for entry in sorted(entries, key=lambda entry: entry.position)
    ]


async def get_cache_provider(
    session: AsyncSession, name: Optional[str]
) -> Optional[CacheProvider]:
    """One provider by name, case-insensitively — the name a cache service
    stores is whatever case the document declared it in."""
    wanted = (name or "").lower()
    for entry in await CacheProviderEntry.all(session):
        if entry.name.lower() == wanted:
            return CacheProvider(**entry.payload)
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
    for field in provider.fields:
        params.setdefault(
            field.name, field.default if field.default is not None else ""
        )
    return _render_injection(integration, params)
