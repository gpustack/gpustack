"""Route-plugin registry and the CRUD/gateway fan-out helpers.

Registration is the plugin's only required contact with the framework.
Two ways in:

* in-process — a built-in plugin calls :func:`register_route_plugin`
  (usually via the decorator) from a module the server imports;
* entry points — a separately shipped package declares a
  ``gpustack.route_plugins`` entry point whose value resolves to a
  :class:`~gpustack.routes.plugins.base.RoutePlugin` subclass; it is
  instantiated and registered on first use here.

The registry is process-global and write-once: names are unique and
re-registering raises, so a plugin cannot be silently swapped under a
running server.
"""

import logging
from importlib.metadata import entry_points
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio.session import AsyncSession

from gpustack.routes.plugins.base import (
    RouteGatewayEntry,
    RoutePlugin,
    RouteReconcileContext,
)
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "gpustack.route_plugins"

_REGISTRY: Dict[str, RoutePlugin] = {}
_entry_points_loaded = False


def register_route_plugin(plugin: RoutePlugin) -> RoutePlugin:
    """Register an instantiated plugin. Returns it so it works as a
    decorator on the class list form (``register = lambda cls:
    register_route_plugin(cls())``) and for direct calls."""
    name = getattr(plugin, "name", None)
    if not name:
        raise ValueError(f"{type(plugin).__name__} has no 'name'")
    if name in _REGISTRY:
        raise RuntimeError(f"route plugin '{name}' already registered")
    _REGISTRY[name] = plugin
    logger.debug("Registered route plugin '%s' (%s)", name, type(plugin).__name__)
    return plugin


def _load_entry_point_plugins() -> None:
    """Register plugins from separately installed packages. Failures
    are loud but not fatal: a broken enterprise package must not take
    the community server down, while running on without it silently is
    exactly the "silent mismatch" the plugin framework exists to
    avoid."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
            plugin = obj() if isinstance(obj, type) else obj
            if not isinstance(plugin, RoutePlugin):
                logger.error(
                    "Route plugin entry point '%s' did not resolve to a "
                    "RoutePlugin: %s",
                    ep.name,
                    type(plugin).__name__,
                )
                continue
        except Exception as e:  # noqa: BLE001
            # Isolation covers construction too, not just ep.load(): a
            # constructor exception must not take the community server
            # down with the broken package.
            logger.error("Failed to load route plugin entry point '%s': %s", ep.name, e)
            continue
        if plugin.name in _REGISTRY:
            logger.warning(
                "Route plugin '%s' from entry point '%s' is shadowed by an "
                "already-registered plugin of the same name; skipping the "
                "entry-point one.",
                plugin.name,
                ep.name,
            )
            continue
        register_route_plugin(plugin)


def route_plugins() -> List[RoutePlugin]:
    """All registered plugins in registration order (built-ins first —
    the package imports them — then entry-point plugins). No priority
    ordering: no consumer's behaviour depends on cross-plugin order
    (payload sections, CR names and Envoy fields are disjoint by the
    framework's rules), so the knob would be decorative. Reintroduce
    one only when a real ordering requirement appears."""
    global _entry_points_loaded
    if not _entry_points_loaded:
        _entry_points_loaded = True
        _load_entry_point_plugins()
        logger.debug(
            "Route plugins loaded: %s",
            ", ".join(f"{p.name} ({type(p).__name__})" for p in _REGISTRY.values())
            or "none",
        )
    return list(_REGISTRY.values())


def get_route_plugin(name: str) -> Optional[RoutePlugin]:
    route_plugins()  # ensure entry points are loaded
    return _REGISTRY.get(name)


def route_plugin_sections(payload: Any) -> Dict[str, Any]:
    """The plugins mapping the dispatcher fans out over — callers hand
    in the payload's ``plugins`` value directly (``input.plugins``),
    never the whole CRUD payload, so a plugin named ``plugins`` stays
    addressable. Tolerates any non-mapping value the same way: a plugin
    section the dispatcher cannot find reads as "not mentioned", which
    is the no-op for every hook."""
    if not isinstance(payload, dict):
        return {}
    return payload


# ---- CRUD fan-out ----
# Called from the ModelRoute / ModelRouteTarget handlers inside their
# transactions. A plugin that declares no extension and finds no
# section of its own is skipped without a call — hooks fire only for
# plugins the request actually touched (plus delete actions, which a
# plugin must see even when the request mentions nothing, to clean up).


# Marks a plugin section the request set to an explicit null — the
# removal instruction, distinct from "not mentioned" (absent key),
# which every hook reads as "don't touch".
SECTION_REMOVED = object()


def _section_for(plugin: RoutePlugin, sections: Dict[str, Any]) -> Any:
    if plugin.name not in sections:
        return None
    value = sections[plugin.name]
    if value is None:
        return SECTION_REMOVED
    if not isinstance(value, dict):
        raise ValueError(
            f"plugins.{plugin.name} must be an object or null, got {type(value).__name__}"
        )
    return value


async def dispatch_route_hooks(
    action: str,
    route: ModelRoute,
    sections: Optional[Dict[str, Any]],
    session: AsyncSession,
) -> None:
    sections = route_plugin_sections(sections)
    for plugin in route_plugins():
        section = _section_for(plugin, sections)
        removed = section is SECTION_REMOVED
        if action != "delete" and section is None and plugin.RouteExtension is None:
            continue
        try:
            await plugin.on_route_write(
                action,
                route,
                None if removed else section,
                session,
                removed=removed,
            )
        except ValueError:
            raise
        except Exception:
            logger.exception(
                "Route plugin '%s' failed on route %s %s",
                plugin.name,
                route.id,
                action,
            )
            raise


async def dispatch_target_hooks(
    action: str,
    target: ModelRouteTarget,
    sections: Optional[Dict[str, Any]],
    session: AsyncSession,
) -> None:
    sections = route_plugin_sections(sections)
    for plugin in route_plugins():
        section = _section_for(plugin, sections)
        removed = section is SECTION_REMOVED
        if action != "delete" and section is None and plugin.TargetExtension is None:
            continue
        try:
            await plugin.on_target_write(
                action,
                target,
                None if removed else section,
                session,
                removed=removed,
            )
        except ValueError:
            raise
        except Exception:
            logger.exception(
                "Route plugin '%s' failed on target %s %s",
                plugin.name,
                target.id,
                action,
            )
            raise


# ---- per-route gateway reconcile ----


async def dispatch_route_reconcile(ctx: "RouteReconcileContext") -> None:
    """Fan the per-route gateway reconcile out to every registered
    plugin. A plugin that is not registered does not
    run — reconcile wiring, like every other hook here, follows the
    registry rather than the import graph. A failing plugin aborts the
    route's reconcile (the caller reports it); half-applied artifacts
    are no worse than what a failed hardcoded call left behind.

    After every plugin ran, the collector's declarations are flushed —
    the single read-modify-write per touched CR that the declarative
    writers asked for. A plugin that raised never reaches it, so a
    failed pass leaves the shared CRs untouched as a unit.
    """
    for plugin in route_plugins():
        await plugin.reconcile_route(ctx)
    collector = getattr(ctx, "collector", None)
    if collector is not None:
        await collector.flush(ctx.cfg, ctx.extensions_api)


# ---- response enrichment ----


async def enrich_routes(
    routes: List[Any], session: AsyncSession
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Per-route ``plugins`` sections: each plugin that declares a
    RouteExtension contributes its own subsection for the routes it has
    something to say about. Routes absent from the result get no
    section — a plugin-free server yields {} and responses stay
    clean."""
    result: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for plugin in route_plugins():
        if plugin.RouteExtension is None:
            continue
        try:
            per_route = await plugin.enrich_routes(routes, session)
        except Exception:
            logger.exception(
                "Route plugin '%s' failed enriching %d routes", plugin.name, len(routes)
            )
            continue
        for route_id, section in per_route.items():
            result.setdefault(route_id, {})[plugin.name] = section
    return result


async def enrich_targets(
    targets: List[Any], session: AsyncSession
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    result: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for plugin in route_plugins():
        if plugin.TargetExtension is None:
            continue
        try:
            per_target = await plugin.enrich_targets(targets, session)
        except Exception:
            logger.exception(
                "Route plugin '%s' failed enriching %d targets",
                plugin.name,
                len(targets),
            )
            continue
        for target_id, section in per_target.items():
            result.setdefault(target_id, {})[plugin.name] = section
    return result


__all__ = [
    "ENTRY_POINT_GROUP",
    "RouteGatewayEntry",
    "RoutePlugin",
    "RouteReconcileContext",
    "dispatch_route_reconcile",
    "dispatch_route_hooks",
    "dispatch_target_hooks",
    "enrich_routes",
    "enrich_targets",
    "get_route_plugin",
    "register_route_plugin",
    "route_plugin_sections",
    "route_plugins",
]
