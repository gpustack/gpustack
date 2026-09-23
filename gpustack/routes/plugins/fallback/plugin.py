"""The fallback route plugin: everything that puts failed traffic onto the
fallback path.

Three artifacts, all owned here:

1. the mapper CR's fallback ``modelMapping`` rules (declared on the shared
   rule collector under the ``mapper`` owner — the legacy inference in the
   collector recycles the pre-plugin rules of the same shape);
2. the fallback ingress (``<name>.fallback``), selected by the
   ``x-higress-fallback-from`` header matcher;
3. the fallback EnvoyFilter that rewrites a failed response onto the
   fallback path.

The destination computation itself is the framework's (``sync_gateway``
runs one ``calculate_destinations`` pass for everyone) and arrives via
``ctx.fallback_destinations``; what to do with it is this plugin's.
"""

from typing import Any, Dict, List

from gpustack.routes.plugins import (
    RoutePlugin,
    RouteReconcileContext,
    register_route_plugin,
)

# gpustack.gateway's package init imports this package, so anything under
# it (and gpustack.server, which reaches back here through controllers)
# is imported lazily at call time rather than at module import.
from gpustack.schemas.model_routes import ModelRouteTarget

MAPPER_RULE_OWNER = "mapper"


async def _has_fallback_target(ctx: RouteReconcileContext) -> bool:
    targets = await ModelRouteTarget.all_by_field(
        ctx.session, "route_id", ctx.model_route.id
    )
    return any(
        target.deleted_at is None and bool(target.fallback_status_codes)
        for target in targets
    )


class FallbackPlugin(RoutePlugin):
    name = "fallback"

    async def reconcile_route(self, ctx: RouteReconcileContext) -> None:
        from gpustack.routes.plugins.artifacts import RouteArtifactCollector

        collector = ctx.collector or RouteArtifactCollector()
        await self._declare_mapper_rules(ctx, collector)
        await self._ensure_fallback_path(ctx)
        if ctx.collector is None:
            await collector.flush(ctx.cfg, ctx.extensions_api)

    async def _declare_mapper_rules(
        self, ctx: RouteReconcileContext, collector: Any
    ) -> None:
        """The mapper CR's fallback rules: on a fallback pass the model name
        must still resolve via modelMapping, and the rules dual-attach the
        main ingress (the main ingress may not exist when only a fallback
        model is set). Declared under the ``mapper`` owner so the collector
        recycles exactly the previous mapper rules. A delete event declares
        the strip instead — the main ingress is being removed, and nothing
        else would retire its rules before the startup cleanup pass."""
        from gpustack.gateway import utils as mcp_handler

        prefix = f"{ctx.cfg.get_namespace()}/"
        if ctx.cfg.get_namespace() == ctx.cfg.gateway_namespace:
            prefix = ""
        full_ingress_name = f"{prefix}{ctx.ingress_name}"
        if ctx.event_is_delete:
            collector.set_rules(
                cr_name=mcp_handler.gpustack_model_mapper_name,
                owner=MAPPER_RULE_OWNER,
                ingresses=[full_ingress_name],
                rules=[],
            )
            return
        fallback_destinations = ctx.fallback_destinations or []
        fallback_model_name_to_registries: Dict[str, List[str]] = {}
        for _, model_name, registry in fallback_destinations:
            registries = fallback_model_name_to_registries.setdefault(model_name, [])
            registries.append(registry.get_service_name())

        main_model_name_to_registries: Dict[str, List[str]] = {}
        from gpustack.routes.plugins.lb.gateway import lb_module_available

        if not lb_module_available(ctx.cfg):
            # Degraded mode: no LB rule lands, so the legacy main-path
            # modelMapping rule is the only rewrite — without it the
            # upstream sees the route name instead of the model name.
            for _, model_name, registry in ctx.destinations or []:
                registries = main_model_name_to_registries.setdefault(model_name, [])
                registries.append(registry.get_service_name())

        expected_rules = mcp_handler.get_expected_match_list(
            route_name=ctx.effective_name or ctx.model_route.name,
            ingress_prefix=prefix,
            ingress_name=ctx.ingress_name,
            fallback_model_name_to_registries=fallback_model_name_to_registries,
            model_name_to_registries=main_model_name_to_registries,
        )
        collector.set_rules(
            cr_name=mcp_handler.gpustack_model_mapper_name,
            owner=MAPPER_RULE_OWNER,
            # Everything the mapper attaches to this route hangs off the main
            # ingress (the fallback rules dual-attach it).
            ingresses=[full_ingress_name],
            rules=expected_rules,
        )

    async def _ensure_fallback_path(self, ctx: RouteReconcileContext) -> None:
        from gpustack.gateway import utils as mcp_handler
        from gpustack.server.bus import EventType

        has_fallback = not ctx.event_is_delete and await _has_fallback_target(ctx)
        fallback_event_type = EventType.UPDATED if has_fallback else EventType.DELETED
        fallback_name = mcp_handler.fallback_ingress_name(ctx.ingress_name)
        await mcp_handler.ensure_model_ingress(
            ingress_class_name=ctx.cfg.gateway_ingress_class,
            event_type=fallback_event_type,
            ingress_name=fallback_name,
            route_name=ctx.effective_name or ctx.model_route.name,
            namespace=ctx.cfg.get_namespace(),
            destinations=ctx.fallback_destinations or [],
            networking_api=ctx.networking_api,
            included_generic_route=False,
            included_proxy_route=ctx.model_route.generic_proxy,
            extra_annotations=mcp_handler.higress_http_header_matcher(
                "exact", "x-higress-fallback-from", ctx.ingress_name
            ),
        )
        await mcp_handler.ensure_fallback_filter(
            event_type=fallback_event_type,
            ingress_name=ctx.ingress_name,
            namespace=ctx.cfg.get_namespace(),
            networking_istio_api=ctx.istio_networking_api,
        )


fallback_plugin = register_route_plugin(FallbackPlugin())
