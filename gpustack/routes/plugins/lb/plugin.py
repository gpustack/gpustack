"""The LB route plugin: hooks, enrichment and gateway entries.

LB is a base capability riding on the route's own storage: the policy
section lands in ``ModelRoute.meta["lb"]`` and the per-target knobs are
the ``model_route_targets`` columns. The plugin module keeps what makes
it a plugin — payload validation, enrichment and the gateway presence
(two CRs from one wasm image; see gateway.py) — while the external
capability plugins (session-affinity, least-load) follow the full
self-owned-storage pattern instead.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from gpustack.config.config import Config
from gpustack.routes.plugins import (
    RouteGatewayEntry,
    RoutePlugin,
    RouteReconcileContext,
    register_route_plugin,
)
from gpustack.routes.plugins.lb.config import LBPolicyConfig
from gpustack.schemas.model_routes import ModelRoute

logger = logging.getLogger(__name__)

META_KEY = "lb"
# The derived lb mode, persisted on the route row so read paths never
# have to re-query targets and plugin storage. Written only by
# refresh_lb_mode from the reconcile path — client input carrying this
# key is stripped on write (routes/model_routes.py).
LB_MODE_META_KEY = "lb_mode"


class LBPlugin(RoutePlugin):
    name = "lb"

    RouteExtension = LBPolicyConfig

    # ---- CRUD hooks ----

    async def on_route_write(
        self,
        action: str,
        route: ModelRoute,
        section: Optional[Dict[str, Any]],
        session: AsyncSession,
        removed: bool = False,
    ) -> None:
        if removed:
            # explicit null drops the policy; LB itself stays on (it is
            # the default routing engine), losing only the extras
            route.meta = {k: v for k, v in (route.meta or {}).items() if k != META_KEY}
            return
        if action == "delete" or section is None:
            # the section rides the route row itself — nothing to clean
            # up on delete, and an unmentioned section is untouched
            return
        config = LBPolicyConfig.model_validate(section)
        # Reassign the whole dict: meta is a plain JSON column, and
        # in-place mutation of the loaded value is invisible to
        # SQLAlchemy's change tracking.
        route.meta = {**(route.meta or {}), META_KEY: config.model_dump()}

    # ---- response enrichment ----

    async def enrich_routes(
        self, routes: List[ModelRoute], session: AsyncSession
    ) -> Dict[int, Dict[str, Any]]:
        """The lb section on a route detail response. Both parts come
        straight from ``meta`` — the derived mode was persisted by
        ``refresh_lb_mode`` at the last reconcile, and the policy config
        is the stored section. Routes with no mode recorded get no
        section: plain round-robin (or a not-yet-reconciled route) has
        nothing to report."""
        sections: Dict[int, Dict[str, Any]] = {}
        for route in routes:
            meta = route.meta or {}
            mode = meta.get(LB_MODE_META_KEY)
            if mode is None:
                continue
            section: Dict[str, Any] = {"mode": mode}
            if meta.get(META_KEY) is not None:
                section["config"] = meta[META_KEY]
            sections[route.id] = section
        return sections

    # ---- derived-mode maintenance ----

    async def _derive_lb_mode(
        self, route: ModelRoute, session: AsyncSession
    ) -> Optional[str]:
        """What the route's targets add up to: weighted
        (business split) / scoring (capability policies) / invalid
        (mixed — LB refuses to render). None when the route has neither
        weights nor capability policies: plain round-robin has nothing
        to report.

        Deliberately state-independent: the mode describes the
        configuration's shape, not the live render. A target's
        ACTIVE/UNAVAILABLE transition never changes the classification
        (an all-weighted route stays weighted while its instances are
        down — it renders nothing, but it is still configured as
        weighted), which also keeps instance-health flaps from
        rewriting route.meta."""
        from sqlmodel import col

        from gpustack.schemas.model_routes import (
            ModelRouteTarget,
        )

        targets = await ModelRouteTarget.all_by_fields(
            session,
            {"deleted_at": None},
            extra_conditions=[col(ModelRouteTarget.route_id).in_([route.id])],
        )
        total = 0
        weighted = 0
        for target in targets:
            if target.fallback_status_codes:
                # fallback targets are not candidates — their weight
                # column means nothing for the split
                continue
            total += 1
            if target.weight and target.weight > 0:
                weighted += 1
        if total == 0:
            return None  # no candidate targets configured — no mode to describe

        # Which capability plugins are effective on a route: ask the
        # plugins themselves (each knows its own storage). Registry-driven
        # — lb never learns another plugin's storage location. Probe
        # failures PROPAGATE: a mode derived from "the plugin happened to
        # fail" would misdescribe live gateway behaviour (the wasm rules
        # stay on the CR regardless), so the reconcile retries instead.
        from gpustack.routes.plugins import route_plugins

        capabilities_on = False
        for plugin in route_plugins():
            if plugin.name == "lb" or plugin.RouteExtension is None:
                continue
            if await plugin.is_effective_on(route, session):
                capabilities_on = True
                break

        if weighted == 0:
            if not capabilities_on:
                # plain rr — no weight and no policy to describe. LB
                # still routes the request (round-robin is the gateway
                # plugin's built-in), but there is no mode to record.
                return None
            return "scoring"  # capability plugins' weighted sum decides
        if weighted == total:
            return "weighted"  # hash(x-request-id) % totalWeight
        return "invalid"  # mixed — LB refuses to render this route

    async def refresh_lb_mode(self, route: ModelRoute, session: AsyncSession) -> None:
        """Re-derive the lb mode and persist it into ``route.meta``.
        Called from the per-route reconcile — the mode is a passive
        record of what the targets add up to, never client input.

        ``route`` is only used for its id: the event-carried instance is
        detached (mutating it raises ObjectDereferencedError), so the
        row is re-loaded in the caller's session and written through
        that. The write goes through ``update()`` so the UPDATED event
        reaches the watch stream — the extra reconcile round it costs
        terminates here: the re-derived mode equals the stored one and
        no further write happens."""
        mode = await self._derive_lb_mode(route, session)
        current = await ModelRoute.one_by_id(session, route.id)
        if current is None or current.deleted_at is not None:
            return
        meta = dict(current.meta or {})
        if mode is None:
            changed = meta.pop(LB_MODE_META_KEY, None) is not None
        else:
            changed = meta.get(LB_MODE_META_KEY) != mode
            meta[LB_MODE_META_KEY] = mode
        if not changed:
            return
        await current.update(session, {"meta": meta})

    # ---- per-route gateway reconcile (dynamic half) ----

    async def reconcile_route(self, ctx: RouteReconcileContext) -> None:
        from gpustack.routes.plugins.artifacts import RouteArtifactCollector
        from gpustack.routes.plugins.lb.reconciler import sync_model_route_lb

        # A bare context (no collector wired) still reconciles: a local
        # collector is flushed here, trading the batched write for
        # self-containment.
        collector = ctx.collector or RouteArtifactCollector()

        # Keep the passively-derived mode fresh alongside the gateway
        # artifacts, so reads never recompute it.
        await self.refresh_lb_mode(ctx.model_route, ctx.session)
        await sync_model_route_lb(
            cfg=ctx.cfg,
            session=ctx.session,
            collector=collector,
            istio_networking_api=ctx.istio_networking_api,
            model_route=ctx.model_route,
            ingress_name=ctx.ingress_name,
            event_is_delete=ctx.event_is_delete,
            extensions_api=ctx.extensions_api,
        )
        if ctx.collector is None:
            # The LB rule itself was already flushed (restricted to its
            # own CR) inside sync_model_route_lb; this flush covers the
            # remaining declarations of a bare context, if any.
            await collector.flush(ctx.cfg, ctx.extensions_api)

    # ---- gateway presence (static half; see gateway.py for specs) ----

    def gateway_entries(self, cfg: Config) -> List[RouteGatewayEntry]:
        from gpustack.routes.plugins.lb.gateway import lb_gateway_entries

        return lb_gateway_entries(cfg)


lb_plugin = register_route_plugin(LBPlugin())
