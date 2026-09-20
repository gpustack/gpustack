"""Session-affinity capability plugin: storage in its own table
(the external-plugin pattern), gateway presence via the shared
capability-band helpers."""

import logging
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gpustack.routes.plugins import (
    RoutePlugin,
    RouteReconcileContext,
    register_route_plugin,
)
from gpustack.routes.plugins.lb.capability import (
    declare_rule,
    full_ingress_name,
)
from gpustack.routes.plugins.session_affinity.config import SessionAffinityConfig
from gpustack.routes.plugins.session_affinity.schemas import SessionAffinityPolicy
from gpustack.schemas.model_routes import ModelRoute

logger = logging.getLogger(__name__)

# Conventional position inside the capability band (795 context -> 700
# finisher); the exact value carries no meaning for the outcome --
# opinions combine by weighted sum -- but the convention is kept.
CR_PRIORITY = 780

# A config that parses but does nothing: an empty key chain yields no
# session key, so the plugin publishes no opinion. Keeps the filter
# chain membership stable on routes without a matchRule.
INERT_DEFAULT = {"sessionKeys": []}


async def _policy_for_route(
    session: AsyncSession, route_id: int
) -> Optional[SessionAffinityPolicy]:
    return await SessionAffinityPolicy.one_by_fields(
        session, {"route_id": route_id, "deleted_at": None}
    )


class SessionAffinityPlugin(RoutePlugin):
    name = "session-affinity"

    RouteExtension = SessionAffinityConfig

    def watches(self) -> Set[type]:
        return {ModelRoute}

    async def is_effective_on(self, route: ModelRoute, session: AsyncSession) -> bool:
        policy = await _policy_for_route(session, route.id)
        if policy is None:
            return False
        return SessionAffinityConfig.model_validate(policy.config).enabled

    async def on_route_write(
        self,
        action: str,
        route: ModelRoute,
        section: Optional[Dict[str, Any]],
        session: AsyncSession,
        removed: bool = False,
    ) -> None:
        existing = await _policy_for_route(session, route.id)
        if action == "delete" or removed:
            if existing is not None:
                await existing.delete(session=session)
            return
        if section is None:
            return  # not mentioned -- leave the stored policy alone
        config = SessionAffinityConfig.model_validate(section)
        source = {"route_id": route.id, "config": config.model_dump()}
        if existing is None:
            await SessionAffinityPolicy.create(session=session, source=source)
        else:
            await existing.update(session=session, source=source)

    async def enrich_routes(
        self, routes: List[ModelRoute], session: AsyncSession
    ) -> Dict[int, Dict[str, Any]]:
        route_ids = [route.id for route in routes]
        if not route_ids:
            return {}
        policies = await SessionAffinityPolicy.all_by_fields(
            session,
            {"deleted_at": None},
            extra_conditions=[col(SessionAffinityPolicy.route_id).in_(route_ids)],
        )
        return {p.route_id: p.config for p in policies}

    # ---- gateway presence (static half; published at init because the
    # plugin is registered — installed means present) ----

    def gateway_entries(self, cfg):
        from gpustack.routes.plugins import RouteGatewayEntry
        from gpustack.routes.plugins.lb.capability import (
            capability_cr_name,
            capability_cr_spec,
            static_spec_diff,
        )

        spec = capability_cr_spec(self.name, CR_PRIORITY, cfg, INERT_DEFAULT)
        if spec is None:
            return []
        return [
            RouteGatewayEntry(
                name=capability_cr_name(self.name),
                spec=spec,
                spec_diff=static_spec_diff(spec),
            )
        ]

    async def reconcile_route(self, ctx: RouteReconcileContext) -> None:
        from gpustack.routes.plugins.artifacts import RouteArtifactCollector

        policy = await _policy_for_route(ctx.session, ctx.model_route.id)
        config: Optional[SessionAffinityConfig] = None
        if policy is not None:
            config = SessionAffinityConfig.model_validate(policy.config)

        full_ingress = full_ingress_name(ctx.cfg, ctx.ingress_name)
        enabled = not ctx.event_is_delete and config is not None and config.enabled

        # A bare context (no collector wired) still reconciles: a local
        # collector is flushed here, trading the batched write for
        # self-containment.
        collector = ctx.collector or RouteArtifactCollector()
        declare_rule(
            collector=collector,
            cfg=ctx.cfg,
            name=self.name,
            full_ingress_name=full_ingress,
            rule_config=config.to_gateway_rule() if enabled else None,
            priority=CR_PRIORITY,
            inert_default_config=INERT_DEFAULT,
        )
        if ctx.collector is None:
            await collector.flush(ctx.cfg, ctx.extensions_api)


session_affinity_plugin = register_route_plugin(SessionAffinityPlugin())
