"""Decision-service capability plugin: the ``gpustack-lb-decision-service``
wasm plugin (AUTHN/328) — per-request Jev model-selection over the candidate
set the LB context role publishes. Storage in the shared capability policy
table (capability = ``decision-service``), gateway presence via the shared
capability-band helpers; the decision-service catalogue
(defaultConfig.providers) is maintained per entry by
``providers.sync_decision_model_entry`` / ``sync_decision_provider_entry``
on Model and ModelProvider events."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError

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
from gpustack.routes.plugins.decision_service.config import DecisionServiceRouteConfig
from gpustack.routes.plugins.decision_service.providers import (
    decision_default_config,
    decision_provider_entries,
    is_decision_config,
    render_decision_model_entry,
)
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.model_routes import ModelRoute

logger = logging.getLogger(__name__)

CR_PRIORITY = 328


async def _decision_target_resolves(
    session: Optional[AsyncSession],
    config: "DecisionServiceRouteConfig",
    route_owner_principal_id: Optional[int] = None,
) -> bool:
    """Whether a section still names a decision service that can render —
    and, for the external kind, a decision model the service still offers.

    Unresolvable (deleted provider/model, a provider whose type moved off
    the decision types, a local decision model with no running servable
    instance — the catalogue entry would be absent — a local decision
    model another Org now owns (ownership can move after the write-time
    check), or a route whose ``decisionModel`` dropped out of the
    provider's pulled ``models`` cache) must not render: the rule would
    point the callout at a target that no longer exists. The provider
    cache is only populated when someone pulls ``/v1/models``, so an empty
    cache proves nothing and resolves. None (no session to query,
    framework tests) resolves too, the skip ``_validate_provider_id``
    takes."""
    if session is None:
        return True
    if config.providerId is not None:
        provider = await ModelProvider.one_by_id(session, config.providerId)
        if (
            provider is None
            or provider.deleted_at is not None
            or not is_decision_config(provider.config)
        ):
            return False
        # The same ownership re-check the modelId target takes: a later
        # owner move invalidates the write-time alignment, and the rule
        # must not keep another Org's service (its endpoint and token ride
        # the shared CR) selected.
        from gpustack.api.exceptions import InvalidException
        from gpustack.routes.model_routes import _assert_target_tenant_aligned

        try:
            _assert_target_tenant_aligned(
                route_owner_principal_id,
                getattr(provider, "owner_principal_id", None),
                "ModelProvider",
                provider.id,
            )
        except InvalidException as e:
            # The helper is a pure comparison that raises only
            # InvalidException; anything else is a bug that must surface
            # rather than strip the rule silently. The strip is a
            # security-relevant state change, so it logs at warning — at
            # debug an ownership move that invalidates routes across Orgs
            # would disable the decision capability with no
            # operator-visible symptom.
            logger.warning(
                "decision-service rule for provider %s stripped: %s",
                provider.id,
                e,
            )
            return False
        alias = config.decisionModel
        models = provider.models or []
        if not models:
            # cache never pulled: absence here is not evidence the service
            # dropped the alias
            return True
        return any(model.name == alias for model in models)
    if config.modelId is not None:
        from gpustack.schemas.models import Model as ModelRow

        model = await ModelRow.one_by_id(session, config.modelId)
        if model is None or model.deleted_at is not None:
            return False
        if "decision" not in (model.categories or []):
            return False
        # The write-time check pinned tenant alignment; ownership can move
        # after the fact, so the render gate re-checks it — a route must
        # not keep pointing its callout at a deployment another Org now
        # owns.
        from gpustack.api.exceptions import InvalidException
        from gpustack.routes.model_routes import _assert_target_tenant_aligned

        try:
            _assert_target_tenant_aligned(
                route_owner_principal_id,
                getattr(model, "owner_principal_id", None),
                "Model",
                model.id,
            )
        except InvalidException as e:
            # The narrow catch keeps a genuine programming error in the
            # assertion helper from folding into the "another Org owns it"
            # verdict; the strip is a security-relevant state change, so it
            # logs at warning — invisible at debug, an ownership move that
            # invalidates routes across Orgs would disable the decision
            # capability with no operator-visible symptom.
            logger.warning(
                "decision-service rule for model %s stripped: %s",
                model.id,
                e,
            )
            return False
        # The catalogue entry needs a renderable target (locally a servable
        # RUNNING member with a registry, in a remote cluster the gateway
        # entrypoint). The render itself is the single source of truth: a
        # looser local check would render an activeProviderId the catalogue
        # has no entry for, and a worker-blind one would diverge from a
        # remote deployment's gateway-routed entry.
        return await render_decision_model_entry(session, model.id) is not None
    return True


def decision_referencing_route_ids(policies: List[Any], provider_id: int) -> "set[int]":
    """Route ids whose stored decision-service policy pins ``provider_id``
    — the routes a provider-removal event has to re-reconcile. Sections
    always carry an explicit providerId (implicit catalogue selection
    bypassed tenant isolation), so there is no implicit set."""
    return {
        policy.route_id
        for policy in policies or []
        if (policy.config or {}).get("providerId") == provider_id
    }


async def _config_for_route(
    session: AsyncSession, route_id: int
) -> Optional[DecisionServiceRouteConfig]:
    policy = await policy_for_route(session, "decision-service", route_id)
    if policy is None:
        return None
    section = section_from_policy(policy)
    if isinstance(section, dict) and not section.get("decisionModel"):
        # Rows written before decisionModel became required carry none
        # (exclude_none dropped it at write time); the old precedence chain
        # fell back to the provider entry's own model, so reads keep doing
        # that instead of failing reconciliation for pre-existing routes.
        section = {
            **section,
            "decisionModel": await _provider_default_decision_model(
                session, section.get("providerId")
            ),
        }
    try:
        return DecisionServiceRouteConfig.model_validate(section)
    except ValidationError as e:
        # A stored section that no longer parses must degrade to inert, not
        # break the reconcile loop that reads it.
        logger.warning(
            "stored decision-service policy for route %s fails validation "
            "(%s); treating the route as having no decision-service config",
            route_id,
            e,
        )
        return None


async def _provider_default_decision_model(
    session: Optional[AsyncSession], provider_id: Optional[int]
) -> Optional[str]:
    """The provider config's own decision-engine model, when provider_id
    still names a live decision-service provider."""
    if provider_id is None or session is None:
        return None
    provider = await ModelProvider.one_by_id(session, provider_id)
    if (
        provider is None
        or provider.deleted_at is not None
        or not is_decision_config(provider.config)
    ):
        return None
    return provider.config.model


async def _validate_provider_id(
    session: Optional[AsyncSession],
    provider_id: Optional[int],
    route_owner_principal_id: Optional[int] = None,
) -> None:
    """A providerId that is not a live decision-service ModelProvider the
    route's Org owns would render an activeProviderId the wasm plugin
    cannot resolve — or worse, one that resolves to another Org's decision
    service (its endpoint and token ride the shared gateway CR). Rejected
    at write time, with the same tenant alignment the route-target path
    enforces. Skipped where there is no session to query (framework
    tests)."""
    if provider_id is None or session is None:
        return
    # The row lock pairs with the delete path in
    # routes/model_provider.py: both hold it while reading/writing the
    # references, so a provider delete cannot race a route write that is
    # mid-validation of this providerId.
    provider = await ModelProvider.one_by_id(session, provider_id, for_update=True)
    if (
        provider is None
        or provider.deleted_at is not None
        or not is_decision_config(provider.config)
    ):
        raise ValueError(
            f"providerId {provider_id} does not reference a decision-service "
            "ModelProvider (gpustack-lb-typesafe type)"
        )
    from gpustack.routes.model_routes import _assert_target_tenant_aligned

    _assert_target_tenant_aligned(
        route_owner_principal_id,
        getattr(provider, "owner_principal_id", None),
        "ModelProvider",
        provider_id,
    )


async def _validate_model_id(
    session: Optional[AsyncSession],
    model_id: Optional[int],
    route_owner_principal_id: Optional[int] = None,
) -> None:
    """A modelId that is not a live decision-category Model the route's Org
    owns would render an activeProviderId the wasm plugin cannot resolve —
    or one that resolves to another Org's decision service (its instance
    endpoint rides the shared gateway CR). Rejected at write time, with
    the same tenant alignment the route-target path enforces.

    Deliberately NOT requiring a running instance: a route may be wired up
    before the deployment is up, and the render gate strips the rule
    fail-open until an instance exists. Skipped where there is no session
    to query (framework tests)."""
    if model_id is None or session is None:
        return
    from gpustack.routes.model_routes import _assert_target_tenant_aligned
    from gpustack.schemas.models import Model as ModelRow

    model = await ModelRow.one_by_id(session, model_id, for_update=True)
    if (
        model is None
        or model.deleted_at is not None
        or "decision" not in (model.categories or [])
    ):
        raise ValueError(
            f"modelId {model_id} does not reference a locally deployed "
            "decision model (decision category)"
        )
    _assert_target_tenant_aligned(
        route_owner_principal_id,
        getattr(model, "owner_principal_id", None),
        "Model",
        model_id,
    )


async def _validate_decision_target(
    session: Optional[AsyncSession],
    config: DecisionServiceRouteConfig,
    route_owner_principal_id: Optional[int] = None,
) -> None:
    """Write-time check of whichever target the section pins; exactly one
    is guaranteed by the config's own validator."""
    if config.providerId is not None:
        await _validate_provider_id(
            session, config.providerId, route_owner_principal_id
        )
    else:
        await _validate_model_id(session, config.modelId, route_owner_principal_id)


class DecisionServicePlugin(RoutePlugin):
    # The capability key doubles as the gateway plugin name suffix: the CR
    # is gpustack-lb-decision-service (capability_cr_name).
    name = "decision-service"

    RouteExtension = DecisionServiceRouteConfig

    async def is_effective_on(self, route: ModelRoute, session: AsyncSession) -> bool:
        config = await _config_for_route(session, route.id)
        if config is None:
            return False
        if not (config.enabled and config.modelSelection is not None):
            return False
        # A providerId/modelId that no longer resolves is inert, not broken:
        # the feature is off for this route until the reference is fixed.
        return await _decision_target_resolves(
            session, config, getattr(route, "owner_principal_id", None)
        )

    async def on_route_write(
        self,
        action: str,
        route: ModelRoute,
        section: Optional[Dict[str, Any]],
        session: AsyncSession,
        removed: bool = False,
    ) -> None:
        if action == "delete" or removed:
            # hard delete, inside the caller's transaction: a soft-deleted
            # row would still count against the (capability, route_id)
            # unique constraint and block adding the policy back
            await delete_capability_policy(session, "decision-service", route.id)
            return
        if section is None:
            return
        config = DecisionServiceRouteConfig.model_validate(section)
        await _validate_decision_target(
            session,
            config,
            getattr(route, "owner_principal_id", None),
        )
        await store_capability_policy(
            session,
            "decision-service",
            route.id,
            config=config.model_dump(exclude={"weight"}, exclude_none=True),
            weight=config.weight,
        )

    async def enrich_routes(
        self, routes: List[ModelRoute], session: AsyncSession
    ) -> Dict[int, Dict[str, Any]]:
        return await capability_sections(
            session, "decision-service", [route.id for route in routes]
        )

    # ---- gateway presence (static half; published at init because the
    # plugin is registered — installed means present) ----

    def gateway_entries(self, cfg):
        from gpustack.routes.plugins import RouteGatewayEntry
        from gpustack.routes.plugins.lb.capability import (
            capability_cr_name,
            capability_cr_spec,
        )
        from gpustack.routes.plugins.decision_service.providers import (
            decision_default_config,
        )
        from gpustack.routes.plugins.decision_service.spec_diff import (
            decision_static_spec_diff,
        )

        spec = capability_cr_spec(
            self.name, CR_PRIORITY, cfg, decision_default_config(cfg)
        )
        if spec is None:
            return []
        return [
            RouteGatewayEntry(
                name=capability_cr_name(self.name),
                spec=spec,
                spec_diff=decision_static_spec_diff(spec),
            )
        ]

    async def reconcile_route(self, ctx: RouteReconcileContext) -> None:
        from gpustack.routes.plugins.artifacts import RouteArtifactCollector

        config = await _config_for_route(ctx.session, ctx.model_route.id)

        full_ingress = full_ingress_name(ctx.cfg, ctx.ingress_name)
        enabled = (
            not ctx.event_is_delete
            and config is not None
            and config.enabled
            and config.modelSelection is not None
            # an unresolvable providerId/modelId/decisionModel strips the
            # rule rather than rendering an activeProviderId or model alias
            # the plugin cannot match
            and await _decision_target_resolves(
                ctx.session,
                config,
                getattr(ctx.model_route, "owner_principal_id", None),
            )
        )

        # A bare context (no collector wired) still reconciles: a local
        # collector is flushed here, trading the batched write for
        # self-containment.
        collector = ctx.collector or RouteArtifactCollector()
        # The defensive recreate (create_base, used when the CR was deleted
        # out-of-band) must carry the providers catalogue too: a rebuilt CR
        # with only the inert switch would deploy rules whose provider
        # entries nobody fills until the next provider event.
        inert = decision_default_config(ctx.cfg)
        try:
            providers = await ModelProvider.all_by_field(
                ctx.session, "deleted_at", None
            )
            from gpustack.routes.plugins.decision_service.providers import (
                render_all_decision_model_entries,
            )

            # Bounded queries (one batch per data source, however many
            # decision models exist), each entry rendered through the same
            # helpers the scoped syncs use — a recreated catalogue cannot
            # drift from the per-entry maintenance, remote or local.
            inert["providers"] = decision_provider_entries(providers) + (
                await render_all_decision_model_entries(ctx.session)
            )
        except Exception as e:
            logger.warning(
                "could not attach the providers catalogue to the "
                "decision-service recreate base: %s",
                e,
            )
        declare_rule(
            collector=collector,
            cfg=ctx.cfg,
            name=self.name,
            full_ingress_name=full_ingress,
            rule_config=config.to_gateway_rule() if enabled else None,
            priority=CR_PRIORITY,
            inert_default_config=inert,
        )
        if ctx.collector is None:
            await collector.flush(ctx.cfg, ctx.extensions_api)


decision_service_plugin = register_route_plugin(DecisionServicePlugin())
