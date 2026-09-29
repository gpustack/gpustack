"""Resolve provider capacity declarations into cache instance reservations."""

import re
import shlex
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Dict, Optional

from gpustack.schemas.cache_providers import (
    CUSTOM_VERSION,
    CacheProvider,
    render_template,
    resolved_field_values,
)
from gpustack.schemas.cache_services import CacheServiceBase


def cache_service_resource_claim(
    service: CacheServiceBase, provider: CacheProvider, component: str
) -> Optional[Dict[str, int]]:
    """Resolve one component's declared RAM, rejecting ambiguous overrides."""
    spec = provider.get_component(component)
    profile = spec.resource_profile if spec else provider.resource_profile
    if profile is None or profile.ram_gib is None:
        return None
    config = service.config
    values = resolved_field_values(
        provider.fields, (config.fields if config else {}) or {}
    )
    rendered = render_template(profile.ram_gib, values)
    try:
        gib = Decimal(rendered)
    except (InvalidOperation, ValueError):
        raise ValueError(f"Invalid RAM resource profile for component '{component}'")
    if not gib.is_finite() or gib < 0:
        raise ValueError("RAM reservation must be finite and non-negative")
    _validate_capacity_overrides(service, provider, component, profile.ram_gib)
    return {"ram": int((gib * 1024**3).to_integral_value(rounding=ROUND_CEILING))}


def _validate_capacity_overrides(service, provider, component, template):
    """Capacity must be configured through the fields used by the reservation."""
    if service.config is None:
        return
    fields = set(re.findall(r"{{\s*(\w+)", template))
    if not fields:
        return
    version = (
        provider.custom_version_config()
        if service.provider_version == CUSTOM_VERSION
        else provider.get_version_config(service.provider_version)[0]
    )
    component_spec = provider.get_component(component)
    launch = component_spec or version
    if launch is None:
        return
    try:
        tokens = shlex.split(launch.run_command or launch.run_args or "")
    except ValueError as e:
        raise ValueError(
            f"Invalid launch template for cache provider '{provider.name}' "
            f"component '{component}': {e}"
        ) from e
    capacity_flags = set()
    for index, token in enumerate(tokens):
        if fields & set(re.findall(r"{{\s*(\w+)", token)):
            flag = token
            if not flag.startswith("-") and index:
                flag = tokens[index - 1]
            if flag.startswith("-"):
                capacity_flags.add(flag.split("=", 1)[0])
    overrides = (service.config.parameters or {}).get(component, [])
    if any(token.split("=", 1)[0] in capacity_flags for token in overrides):
        raise ValueError(
            "Configure cache RAM through provider fields, not extra parameters"
        )
    env = {
        **((version.env if version else None) or {}),
        **((component_spec.env if component_spec else None) or {}),
    }
    for key, value in env.items():
        if key in (service.config.env or {}) and fields & set(
            re.findall(r"{{\s*(\w+)", value)
        ):
            raise ValueError(
                "Configure cache RAM through provider fields, not environment overrides"
            )
