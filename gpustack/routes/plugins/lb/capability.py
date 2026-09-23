"""Shared CR plumbing for the LB capability-band plugins.

A capability plugin (session-affinity, least-load, the enterprise
prefix-affinity) runs inside the LB band — after the context role
(AUTHN/340) and before the finisher (AUTHN/325), which itself sits
after every rejection point (ext-auth 360, ip-acl 350). Within the
band the WasmPlugin priority carries no meaning for the outcome —
opinions combine by weighted sum — but the conventional positions are
kept (session-affinity 336, prefix-affinity 333, least-load 330).

Conventions every capability plugin follows, enforced only by this
module's shape (not by the framework):

* the CR name and the wasm module name are both ``gpustack-lb-<name>``;
* ``defaultConfigDisable: false`` with an inert ``defaultConfig`` —
  the plugin is instantiated on EVERY route so the filter chain
  membership never changes when a route's rule is added or removed
  (a chain rebuild drops live connections); the inert default makes
  it a no-op everywhere a matchRule does not explicitly enable it;
* the CR is published at gateway initialization by a registered plugin
  — installed means present — and per-route reconcile only adds or
  strips matchRules on it, never creates or deletes the CR;
* a rule for an ingress is replaced wholesale or stripped — never
  partially merged with the plugin's other routes' rules.
"""

import logging
from typing import Any, Callable, Dict, Optional

from gpustack.config.config import Config
from gpustack.gateway.client.extensions_higress_io_v1_api import (
    WasmPluginMatchRule,
    WasmPluginSpec,
)
from gpustack.gateway.plugins import plugin_spec_overrides

logger = logging.getLogger(__name__)

CAPABILITY_PHASE = "AUTHN"


def capability_cr_name(name: str) -> str:
    return f"gpustack-lb-{name}"


def full_ingress_name(cfg: Config, ingress_name: str) -> str:
    """The matchRule-facing ingress name: namespaced unless the routes
    live in the gateway's own namespace."""
    prefix = f"{cfg.get_namespace()}/"
    if cfg.get_namespace() == cfg.gateway_namespace:
        prefix = ""
    return f"{prefix}{ingress_name}"


def declare_rule(
    collector: Any,
    cfg: Config,
    name: str,
    full_ingress_name: str,
    rule_config: Optional[Dict[str, Any]],
    priority: int,
    inert_default_config: Dict[str, Any],
) -> None:
    """Declare the capability's per-route rule on the collector: an
    upsert when ``rule_config`` is set, a strip when it is None.

    ``create_base`` is the capability's static spec, so a CR that was
    manually deleted is recreated carrying this route's rule — the
    defensive recreate the direct-write path kept. None (module not in
    the shipped plugins manifest) means the declaration still strips
    stale rules but never creates the CR; the misconfiguration is
    logged by the init pass when its own entry cannot deploy."""
    create_base = capability_cr_spec(name, priority, cfg, inert_default_config)
    rules = (
        [
            WasmPluginMatchRule(
                config=rule_config, ingress=[full_ingress_name], configDisable=False
            )
        ]
        if rule_config is not None
        else []
    )
    collector.set_rules(
        cr_name=capability_cr_name(name),
        owner=name,
        ingresses=[full_ingress_name],
        rules=rules,
        create_base=create_base,
    )


def capability_cr_spec(
    name: str, priority: int, cfg: Config, inert_default_config: Dict[str, Any]
) -> Optional[WasmPluginSpec]:
    """The static half of a capability CR: phase, band position, module
    URL, and an inert defaultConfig that keeps the plugin instantiated
    (chain-stable) but behaviourless on routes without a matchRule.
    None when the module is not in the shipped plugins manifest — the
    plugin is registered but this dependency cannot deploy it, which
    init logs loudly."""
    cr_name = capability_cr_name(name)
    try:
        url_overrides = plugin_spec_overrides(cr_name, cfg=cfg)
    except ValueError:
        return None
    return WasmPluginSpec(
        phase=CAPABILITY_PHASE,
        priority=priority,
        **url_overrides,
        defaultConfigDisable=False,
        defaultConfig=inert_default_config,
        matchRules=[],
        failStrategy="FAIL_OPEN",
    )


def static_spec_diff(
    expected_spec: WasmPluginSpec,
) -> Callable[[Optional[WasmPluginSpec]], Optional[WasmPluginSpec]]:
    """Init-pass diff for a capability CR: rewrite the static half
    (url, phase, priority), carry the live matchRules over — the
    route-driven half belongs to the plugin's reconcile, and the two
    passes must not fight over one field."""

    def _diff(current: Optional[WasmPluginSpec]) -> Optional[WasmPluginSpec]:
        if current is None:
            return expected_spec
        current_match_rules = current.model_dump(exclude_none=True).get("matchRules")
        merged = expected_spec.model_dump(exclude_none=True)
        if current_match_rules:
            merged["matchRules"] = current_match_rules
        return WasmPluginSpec.model_validate(merged)

    return _diff
