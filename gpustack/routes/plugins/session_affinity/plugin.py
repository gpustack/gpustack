"""Session-affinity capability plugin: storage in the shared capability
policy table (keyed by capability name), gateway presence via the
shared capability-band helpers."""

import logging
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.ext.asyncio import AsyncSession

from gpustack.routes.plugins import (
    RoutePlugin,
    RouteReconcileContext,
    register_route_plugin,
)
from gpustack.routes.plugins.capability_policy import (
    capability_sections,
    delete_capability_policy,
    policy_for_route,
    section_from_policy,
    store_capability_policy,
)
from gpustack.routes.plugins.lb.capability import (
    declare_rule,
    full_ingress_name,
)
from gpustack.routes.plugins.session_affinity.config import SessionAffinityConfig
from gpustack.schemas.model_routes import ModelRoute

logger = logging.getLogger(__name__)

# Conventional position inside the LB band (340 context -> 325
# finisher), after every rejection point; the exact value carries no
# meaning for the outcome -- opinions combine by weighted sum -- but
# the convention is kept.
CR_PRIORITY = 336

# A config that parses but does nothing: an empty key chain yields no
# session key, so the plugin publishes no opinion. Keeps the filter
# chain membership stable on routes without a matchRule.
INERT_DEFAULT = {"sessionKeys": []}


async def _config_for_route(
    session: AsyncSession, route_id: int
) -> Optional[SessionAffinityConfig]:
    policy = await policy_for_route(session, "session-affinity", route_id)
    if policy is None:
        return None
    return SessionAffinityConfig.model_validate(section_from_policy(policy))


class SessionAffinityPlugin(RoutePlugin):
    name = "session-affinity"

    RouteExtension = SessionAffinityConfig

    def watches(self) -> Set[type]:
        return {ModelRoute}

    async def is_effective_on(self, route: ModelRoute, session: AsyncSession) -> bool:
        config = await _config_for_route(session, route.id)
        if config is None:
            return False
        return config.enabled

    async def on_route_write(
        self,
        action: str,
        route: ModelRoute,
        section: Optional[Dict[str, Any]],
        session: AsyncSession,
        removed: bool = False,
    ) -> None:
        if action == "delete" or removed:
            # inside the caller's transaction, hard: a soft-deleted row
            # would still count against the (capability, route_id)
            # unique constraint and block adding the policy back
            await delete_capability_policy(session, "session-affinity", route.id)
            return
        if section is None:
            return  # not mentioned -- leave the stored policy alone
        config = SessionAffinityConfig.model_validate(section)
        await store_capability_policy(
            session,
            "session-affinity",
            route.id,
            config=config.model_dump(exclude={"weight"}),
            weight=config.weight,
        )

    async def enrich_routes(
        self, routes: List[ModelRoute], session: AsyncSession
    ) -> Dict[int, Dict[str, Any]]:
        return await capability_sections(
            session, "session-affinity", [route.id for route in routes]
        )

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

        config = await _config_for_route(ctx.session, ctx.model_route.id)

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
