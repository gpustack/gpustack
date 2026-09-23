"""Declarative gateway-artifact collection for the per-route reconcile.

Plugins (and the framework's own mapper sync) declare what their
matchRules should look like for the reconciled route, and the collector
turns the declarations into one read-modify-write per WasmPlugin CR at
the end of the pass — instead of each writer doing its own RMW on the
shared CRs. A plugin is free to keep writing its artifacts directly
(single-owner, per-route resources such as EnvoyFilters gain nothing
from batching); the collector is the fast path for the shared ones.

Rule ownership is declared, not inferred: every rule written through
the collector carries ``RULE_OWNER_KEY`` in its config, and a
``set_rules`` call replaces exactly the rules that key claims on the
given ingresses — every other rule on the CR survives, whoever wrote
it. That is what lets two writers (the mapper sync and the LB plugin)
share one CR without an ingress-based predicate: their rules attach to
the same ingresses by design, so the ingress list can never be the
ownership boundary.

Rules written before the owner key existed (or by hand) carry no owner.
They are recycled only through a conservative shape inference per
legacy writer (see :func:`rule_belongs_to`) so an upgrade converges;
rules whose shape matches no legacy writer are never touched.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from gpustack.config.config import Config

if TYPE_CHECKING:  # pragma: no cover
    from gpustack.gateway.client.extensions_higress_io_v1_api import (
        WasmPluginMatchRule,
        WasmPluginSpec,
    )

# gpustack.gateway's package init imports this package, so anything
# under it is imported lazily at call time rather than at module
# import — including for the dataclass annotations below, which only
# need the names at class-creation time as strings.
logger = logging.getLogger(__name__)

RULE_OWNER_KEY = "x-gpustack-owner"


@dataclass
class RouteRuleUpdate:
    """One writer's desired rules for one CR: every rule it currently
    owns on ``ingresses`` is replaced by ``rules`` (an empty list means
    strip). ``create_base`` is the full spec used verbatim when the CR
    does not exist — None means a missing CR is left alone (the init
    pass, not a route event, recreates it)."""

    owner: str
    ingresses: List[str]
    rules: List["WasmPluginMatchRule"] = field(default_factory=list)
    create_base: Optional["WasmPluginSpec"] = None


def rule_belongs_to(
    rule: "WasmPluginMatchRule", ingresses: List[str], owner: str
) -> bool:
    """Whether ``rule`` is ``owner``'s to recycle on these ingresses.

    Ownership is recorded in the rule config when the collector wrote
    it. A rule without the marker is legacy: only the two legacy
    writers' shapes are inferred (LB candidates rules, mapper
    modelMapping rules), and anything else — hand-written or foreign —
    is left strictly alone.
    """
    if not set(ingresses) & set(rule.ingress or []):
        return False
    config = getattr(rule, "config", None)
    if not isinstance(config, dict):
        return False
    recorded = config.get(RULE_OWNER_KEY)
    if isinstance(recorded, str):
        return recorded == owner
    from gpustack.gateway.utils import is_lb_match_rule

    if owner == "lb":
        return is_lb_match_rule(rule)
    if owner == "mapper":
        return "modelMapping" in config
    return False


def apply_rule_updates(
    spec: "WasmPluginSpec", updates: List[RouteRuleUpdate]
) -> "WasmPluginSpec":
    """Fold ``updates`` into ``spec``'s matchRules: drop what the
    updates' owners claim on their ingresses, append the fresh rules
    (stamped with their owner), keep a canonical order so the result
    diffs deterministically regardless of declaration order."""
    to_keep = [
        rule
        for rule in spec.matchRules or []
        if not any(
            rule_belongs_to(rule, update.ingresses, update.owner) for update in updates
        )
    ]
    for update in updates:
        for rule in update.rules:
            stamped = rule.model_copy(deep=True)
            stamped.config = {**(stamped.config or {}), RULE_OWNER_KEY: update.owner}
            to_keep.append(stamped)
    to_keep.sort(key=lambda r: (r.ingress or [""])[0])
    spec.matchRules = to_keep
    return spec


class RouteArtifactCollector:
    """Gathers :class:`RouteRuleUpdate` declarations across one
    per-route reconcile pass and flushes them grouped by CR: one
    ``ensure_wasm_plugin`` call (one GET, at most one PUT) per touched
    CR, however many writers declared rules on it."""

    def __init__(self) -> None:
        self._updates: Dict[str, List[RouteRuleUpdate]] = {}

    def set_rules(
        self,
        cr_name: str,
        owner: str,
        ingresses: List[str],
        rules: Optional[List[WasmPluginMatchRule]] = None,
        create_base: Optional[WasmPluginSpec] = None,
    ) -> None:
        self._updates.setdefault(cr_name, []).append(
            RouteRuleUpdate(
                owner=owner,
                ingresses=list(ingresses),
                rules=list(rules or []),
                create_base=create_base,
            )
        )

    async def flush(
        self, cfg: Config, extensions_api: Any, only_cr: Optional[str] = None
    ) -> None:
        """Write the collected declarations, one ensure per touched CR.
        ``only_cr`` restricts the write to a single CR — used by a
        plugin that needs its own rules on the gateway before it
        applies an ordering-sensitive artifact, without touching other
        plugins' pending declarations (which keeps the flush
        registration-order-independent)."""
        from gpustack.gateway import utils as gateway_utils

        for cr_name in [c for c in self._updates if only_cr is None or c == only_cr]:
            updates = self._updates[cr_name]
            create_base = next(
                (u.create_base for u in updates if u.create_base is not None), None
            )

            def spec_diff(
                current_spec: Optional["WasmPluginSpec"],
                _updates: List[RouteRuleUpdate] = updates,
                _create_base: Optional["WasmPluginSpec"] = create_base,
            ) -> Optional["WasmPluginSpec"]:
                if current_spec is None:
                    if _create_base is None:
                        # A manually deleted CR is not recreated from a
                        # route event, only from a deliberate publish.
                        return None
                    return apply_rule_updates(copy.deepcopy(_create_base), _updates)
                return apply_rule_updates(current_spec, _updates)

            await gateway_utils.ensure_wasm_plugin(
                api=extensions_api,
                name=cr_name,
                namespace=cfg.gateway_namespace,
                spec_diff=spec_diff,
            )
            # clear only what this pass wrote: a restricted flush leaves
            # the other CRs' declarations pending for the next one
            self._updates.pop(cr_name, None)
