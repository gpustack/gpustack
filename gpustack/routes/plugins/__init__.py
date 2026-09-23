"""Route-plugin framework: a registration seam plus CRUD / gateway
fan-out for self-contained ModelRoute extensions.

A plugin owns its tables, its payload sections and its gateway
artifacts; this package owns only the integration points. See
``base.py`` for the contract and the two non-enforced rules (table
prefix, Envoy field ownership).
"""

from gpustack.routes.plugins.artifacts import RouteArtifactCollector
from gpustack.routes.plugins.base import (
    RouteGatewayEntry,
    RoutePlugin,
    RouteReconcileContext,
)
from gpustack.routes.plugins.registry import (
    ENTRY_POINT_GROUP,
    dispatch_route_hooks,
    dispatch_route_reconcile,
    dispatch_target_hooks,
    enrich_routes,
    enrich_targets,
    get_route_plugin,
    register_route_plugin,
    route_plugins,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "RouteArtifactCollector",
    "RouteGatewayEntry",
    "RoutePlugin",
    "RouteReconcileContext",
    "dispatch_route_hooks",
    "dispatch_route_reconcile",
    "dispatch_target_hooks",
    "enrich_routes",
    "enrich_targets",
    "get_route_plugin",
    "register_route_plugin",
    "route_plugins",
]

# Built-in plugins register on import. This must stay the LAST import in
# the file: the plugin packages import ``from gpustack.routes.plugins
# import RoutePlugin, ...`` back, which is only resolvable once the
# names above exist. A new built-in is a new package import here — the
# registry itself stays free of plugin names. Externally shipped
# (enterprise) plugins register via the ``gpustack.route_plugins``
# entry-point group instead; see registry._load_entry_point_plugins.
from gpustack.routes.plugins import lb  # noqa: E402,F401
from gpustack.routes.plugins import least_load  # noqa: E402,F401
from gpustack.routes.plugins import session_affinity  # noqa: E402,F401
from gpustack.routes.plugins import fallback  # noqa: E402,F401
