"""The route-plugin contract.

A route plugin is a self-contained extension of ModelRoute: it owns its
storage (its own SQLModel tables, registered into the global metadata
at import — built-in plugin tables are part of the central Alembic
chain; an externally shipped plugin creates its own schema before
registering, and its code assumes the tables already exist), its API
surface (extra sections on the route and route-target CRUD payloads),
and its gateway artifacts (WasmPlugin CRs and friends, written by its
own reconciler).

The framework owns none of that. It provides exactly three integration
points, and a plugin that fits them needs no changes in this package:

1. :func:`gpustack.routes.plugins.registry.register_route_plugin` —
   the registration entry. Built-in plugins call it in-process;
   separately shipped (enterprise) packages register through the
   ``gpustack.route_plugins`` entry-point group and are loaded
   automatically.
2. ``initialize_gateway`` appends each plugin's
   :meth:`RoutePlugin.gateway_entries` to the WasmPlugin publication
   pass — the static half (defaultConfig, module URL, diff policy) of
   the plugin's gateway presence.
3. The ModelRoute / ModelRouteTarget CRUD handlers call
   :func:`~gpustack.routes.plugins.registry.dispatch_route_hooks` /
   ``dispatch_target_hooks`` and the enrich helpers, which fan out to
   the per-plugin hooks below.

Two rules keep plugins from stepping on each other, and the framework
does not enforce them — it cannot:

* A built-in capability plugin stores its policy in the shared
  ``model_route_capability_policies`` table (one row per capability +
  route); a plugin shipping its own table must name it
  ``model_route_plugin_<name>_…`` so two plugins cannot collide.
* A plugin patches only the Envoy route fields it owns. Two plugins
  wanting the same field is a design error in one of them; there is no
  arbitration here.
"""

import logging
from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Type,
)

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from gpustack.config.config import Config
from gpustack.routes.plugins.artifacts import RouteArtifactCollector
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget

logger = logging.getLogger(__name__)


@dataclass
class RouteReconcileContext:
    """What the per-route gateway reconcile hands each plugin: the
    entity the event was about, and the gateway handles + naming the
    plugin needs to write its artifacts. Built by the caller
    (``sync_gateway``); plugins only read it.

    ``collector`` is the declarative fast path for shared WasmPlugin
    CRs: a plugin declares its matchRules for this route and the
    framework flushes every writer's declarations with one
    read-modify-write per CR after all plugins ran. A plugin may still
    write its own artifacts directly (``extensions_api`` /
    ``istio_networking_api``) — single-owner per-route resources gain
    nothing from batching.
    """

    cfg: Config
    session: AsyncSession
    model_route: ModelRoute
    ingress_name: str
    """Bare mcp-handler style name (``ai-route-route-<id>.internal``);
    also the Istio route name an EnvoyFilter matches exactly."""

    event_is_delete: bool
    extensions_api: Any
    istio_networking_api: Any
    collector: Optional[RouteArtifactCollector] = None
    networking_api: Any = None
    """NetworkingV1Api — ingress (V1) resources; the framework's core-path
    ingresses and any plugin that manages its own ingress (the fallback
    plugin) both need it."""

    effective_name: Optional[str] = None
    """The route's gateway-facing model name (Org-prefixed for non-platform
    Orgs) — what a mapper rule or AI-proxy route name keys on."""

    fallback_destinations: Optional[List[Any]] = None
    """The route's fallback destination tuples from the shared
    ``calculate_destinations`` pass, so a plugin (the fallback one) does not
    have to recompute them."""


@dataclass
class RouteGatewayEntry:
    """One WasmPlugin CR a route plugin needs published at gateway
    initialization — the static half of its gateway presence.

    ``spec_diff`` carries the plugin's diff policy for the static pass:
    how ``ensure_wasm_plugin`` decides whether the live CR matches
    ``spec`` (full replace, defaultConfig-only compare, create-only,
    …). The route-driven half (matchRules and per-route config) is the
    plugin's own reconciler's business; an init-pass diff that leaves
    those sections of the live CR alone is what keeps the two passes
    from fighting.
    """

    name: str
    spec: Any
    spec_diff: Optional[Callable[..., Any]] = None
    create_only: bool = False


class RoutePlugin(ABC):
    """Base class for route plugins. All hooks have safe defaults; a
    plugin implements only what it needs. Subclasses must set ``name``
    and are instantiated once, at registration."""

    name: ClassVar[str]
    """Stable identifier. Names the payload section on the route /
    route-target CRUD bodies (``"plugins": {"<name>": {…}}``) and is
    the expected prefix of the plugin's table names."""

    RouteExtension: ClassVar[Optional[Type[BaseModel]]] = None
    """Schema of the plugin's section on the route CRUD payload. None
    (default) means the plugin extends nothing on the route object
    itself."""

    TargetExtension: ClassVar[Optional[Type[BaseModel]]] = None
    """Schema of the plugin's section on each route-target item. None
    (default) means no per-target extension."""

    def gateway_entries(self, cfg: Config) -> List[RouteGatewayEntry]:
        """WasmPlugin CRs to publish during gateway initialization.
        Empty by default — a plugin whose gateway presence is purely
        route-driven may still want to return its CR here so the static
        base (defaultConfig) is laid down before any route config lands
        on it."""
        return []

    def watches(self) -> Set[type]:
        """Entity classes whose changes the plugin wants to reconcile
        on. The framework subscribes to the union over registered
        plugins and calls :meth:`reconcile_route` for the affected
        routes."""
        return set()

    # The empty defaults below are intentional: a plugin implements
    # only the hooks it needs.
    async def reconcile_route(self, ctx: RouteReconcileContext) -> None:  # noqa: B027
        """Bring the gateway artifacts for one route in line with the
        plugin's own stored state. Called from the route gateway
        reconcile when an entity in :meth:`watches` changed; the plugin
        decides internally what counts as a real change (the LB
        plugin's "candidates only follow topology" rule lives in its
        implementation, not here)."""

    # ---- CRUD hooks ----
    # ``section`` is the plugin's own slice of the request body — the
    # value under its name in the ``plugins`` mapping — or None when
    # the request did not mention the plugin at all. Distinguishing
    # the two is the plugin's: None means "don't touch", an explicit
    # null inside the section means "clear".

    async def on_route_write(  # noqa: B027
        self,
        action: str,
        route: ModelRoute,
        section: Optional[Dict[str, Any]],
        session: AsyncSession,
        removed: bool = False,
    ) -> None:
        """``action`` is one of ``create`` / ``update`` / ``delete``,
        invoked inside the route CRUD transaction after the main row
        is written (delete: before the route goes away, with the row
        still in hand). Section lifecycle: absent = don't touch, a
        value = write it, ``removed=True`` (the request set the
        section to an explicit null) = delete the plugin's stored
        state; ``section`` is None then."""

    async def on_target_write(  # noqa: B027
        self,
        action: str,
        target: ModelRouteTarget,
        section: Optional[Dict[str, Any]],
        session: AsyncSession,
        removed: bool = False,
    ) -> None:
        """Per-target counterpart of :meth:`on_route_write`, invoked
        from the target CRUD paths (including the batch add/update
        path under route create/update). Route deletion does NOT fan
        out into per-target deletes — the plugin cleans up its
        per-route state, rows included, from ``on_route_write``."""

    # ---- response enrichment ----

    async def is_effective_on(  # noqa: B027
        self, route: ModelRoute, session: AsyncSession
    ) -> bool:
        """Whether this plugin has an ENABLED configuration on the route.

        The one cross-plugin probe the framework defines: the LB mode
        derivation asks every capability plugin this question, and the
        answer must reflect what the gateway is actually serving (the
        wasm rule on the CR), not what an API response could assemble.
        Unlike :meth:`enrich_routes` — which degrades to an absent
        section when it fails — a failure here is a wrong answer in the
        making, so implementations let it propagate and the reconcile
        retries. The default (no stored configuration) is False: a
        plugin that is merely registered changes nothing.
        """
        return False

    async def enrich_routes(
        self, routes: List[ModelRoute], session: AsyncSession
    ) -> Dict[int, Dict[str, Any]]:
        """Sections for a batch of route responses, keyed by route id.
        Batched (not per-row) so a list page costs one query per plugin,
        not one per row. Only called for plugins that declare a
        ``RouteExtension``; routes absent from the result get no
        section."""
        return {}

    async def enrich_targets(
        self, targets: List[ModelRouteTarget], session: AsyncSession
    ) -> Dict[int, Dict[str, Any]]:
        """Per-target counterpart of :meth:`enrich_routes`."""
        return {}
