import logging
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

import yaml

from gpustack.schemas.cache_providers import (
    CacheProvider,
    render_injection as _render_injection,
    validate_injection_templates,
    validate_localized_text,
)

logger = logging.getLogger(__name__)

_cache_providers: Optional[List[CacheProvider]] = None

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


def load_cache_providers(reload: bool = False) -> List[CacheProvider]:
    """
    Load the declarative cache-provider catalog from the bundled asset and
    from every asset an installed plugin ships. The catalog is read-only
    and cached for the process lifetime.
    """
    global _cache_providers
    if _cache_providers is not None and not reload:
        return _cache_providers

    providers: List[CacheProvider] = []
    for package, resource in _catalog_assets():
        providers = _merge(providers, _load_asset(package, resource))

    _cache_providers = providers
    return _cache_providers


def _merge(
    providers: List[CacheProvider], loaded: List[CacheProvider]
) -> List[CacheProvider]:
    """Fold a newly read asset into the catalog, a same-named declaration
    replacing the one already there — in its place, so the catalog's order
    is the order a user sees the cards in."""
    # An asset naming one provider twice keeps its last declaration, the
    # way a later asset replaces an earlier one's: one name, one card.
    by_name = {provider.name.lower(): provider for provider in loaded}
    merged = [by_name.pop(p.name.lower(), p) for p in providers]
    merged.extend(
        by_name.pop(p.name.lower()) for p in loaded if p.name.lower() in by_name
    )
    return merged


def _load_asset(package: str, resource: str) -> List[CacheProvider]:
    providers: List[CacheProvider] = []
    try:
        yaml_file = files(package).joinpath(resource)
        if yaml_file.is_file():
            raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            for index, entry in enumerate(raw or []):
                # An entry that is not a mapping has no name to report;
                # its position is what identifies it in the asset.
                name = entry.get("name") if isinstance(entry, dict) else f"#{index}"
                try:
                    provider = CacheProvider(**entry)
                except Exception as e:
                    # One malformed declaration costs its own provider,
                    # not the catalog: the others still serve.
                    logger.error(f"Skipping malformed cache provider {name}: {e}")
                    continue
                # A provider violating the injection placeholder or the
                # localized-text contract is excluded outright: both fail
                # silently at runtime (literal placeholders corrupting
                # connector config, secrets riding into instance
                # snapshots, text that renders for no locale).
                violations = validate_injection_templates(provider)
                violations += validate_localized_text(provider)
                if violations:
                    logger.error(
                        f"Skipping cache provider {provider.name}: "
                        + "; ".join(violations)
                    )
                    continue
                providers.append(provider)
        else:
            logger.warning(f"Cache provider asset {package}/{resource} not found")
    except Exception as e:
        logger.error(f"Failed to load cache providers from {package}/{resource}: {e}")

    return providers


def get_cache_providers() -> List[CacheProvider]:
    return load_cache_providers()


def get_cache_provider(name: str) -> Optional[CacheProvider]:
    for provider in load_cache_providers():
        if provider.name.lower() == (name or "").lower():
            return provider
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
