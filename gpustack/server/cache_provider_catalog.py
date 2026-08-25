import logging
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

import yaml

from gpustack.schemas.cache_providers import (
    CacheProvider,
    render_injection as _render_injection,
    validate_injection_templates,
)

logger = logging.getLogger(__name__)

_cache_providers: Optional[List[CacheProvider]] = None


def load_cache_providers(reload: bool = False) -> List[CacheProvider]:
    """
    Load the declarative cache-provider catalog from the bundled asset.
    The catalog is read-only and cached for the process lifetime.
    """
    global _cache_providers
    if _cache_providers is not None and not reload:
        return _cache_providers

    providers: List[CacheProvider] = []
    try:
        yaml_file = files("gpustack.assets").joinpath("cache-providers.yaml")
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
                # A provider violating the injection placeholder contract
                # is excluded outright: its failure modes are silent at
                # runtime (literal placeholders corrupting connector
                # config, secrets riding into instance snapshots).
                violations = validate_injection_templates(provider)
                if violations:
                    logger.error(
                        f"Skipping cache provider {provider.name}: "
                        + "; ".join(violations)
                    )
                    continue
                providers.append(provider)
        else:
            logger.warning("cache-providers.yaml not found, catalog is empty")
    except Exception as e:
        logger.error(f"Failed to load cache providers: {e}")

    _cache_providers = providers
    return _cache_providers


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
    # empty, e.g. Mooncake's device_name on TCP), matching the managed
    # run-command path where None renders empty and drops with its flag.
    # metrics_target values are scrape addresses, not connector config:
    # by contract they never enter the injection namespace (they would
    # otherwise ride into the rendered snapshot on the model instance).
    scrape_only = {
        field.name for field in provider.external_fields if field.metrics_target
    }
    for name in scrape_only:
        params.pop(name, None)
    for field in provider.external_fields:
        if field.metrics_target:
            continue
        params.setdefault(
            field.name, field.default if field.default is not None else ""
        )
    for field in provider.managed_fields:
        params.setdefault(
            field.name, field.default if field.default is not None else ""
        )
    return _render_injection(integration, params)
