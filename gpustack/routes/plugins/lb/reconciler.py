"""Route-driven half of the LB plugin's gateway presence.

Called from ``sync_gateway`` (the per-route reconcile on ModelRoute
events) right after ``sync_model_route_mapper``: when the route has an
enabled LB policy, this writes the route's matchRule — candidates plus
modelMappers, the single owner of the main-path model rewrite — into the
context CR (the one named gpustack-model-mapper) and the per-route
cluster_header EnvoyFilter. When it does not, both are withdrawn, and the
mapper's dual-attached fallback rules stay untouched; the main path has no
separate rewrite to fall back to, so a refused render means the route does
not serve.

Candidates follow topology only: they are rebuilt from the route's
targets and their instances, never from instance state — health and
concurrency live in the gateway plugin. That is what keeps a state
flap from rewriting the shared CR for every route.

The Istio route name an EnvoyFilter matches is exactly the ingress
name (``ai-route-route-<id>.internal``) — the fallback filter matches
it the same way, and Istio route-name matching has no prefix
semantics, so an exact string is the whole story.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from kubernetes_asyncio.client import ApiException
from sqlalchemy.ext.asyncio import AsyncSession

from gpustack.config.config import Config
from gpustack.gateway.client.extensions_higress_io_v1_api import (
    WasmPluginMatchRule,
)
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import EnvoyFilter
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import (
    NetworkingIstioIoV1Alpha3Api,
)
from gpustack.gateway.labels_annotations import managed_labels
from gpustack.gateway.utils import DestinationTupleList
from gpustack.routes.plugins.artifacts import RouteArtifactCollector
from gpustack.routes.plugins.lb.config import LBPolicyConfig, lb_policy_from_meta
from gpustack.routes.plugins.lb.gateway import lb_module_available
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    TargetStateEnum,
)
from gpustack.schemas.models import Model

logger = logging.getLogger(__name__)

# A `.static` Higress registry name maps to an Envoy cluster whose
# listener port is 80 whatever the backend speaks; the cluster name the
# plugin wants is the full Envoy form of that service name.
TARGET_CLUSTERS_HEADER = "x-higress-target-cluster"

# The context role keeps the model-mapper CR name (in-place upgrade);
# mirror of LB_CONTEXT_CR_NAME in gateway.py, aliased to avoid an
# import cycle through the plugin package.
CONTEXT_CR_NAME = "gpustack-model-mapper"


def envoy_filter_name(ingress_name: str) -> str:
    return f"gpustack-lb-{ingress_name}"


def candidate_cluster_name(registry: Any) -> str:
    """The Envoy cluster a candidate points at, built from the
    registry's real port — DNS-typed instance registries and provider
    registries do not listen on 80, and the ingress path
    (get_service_name_with_port) never assumes they do."""
    return f"outbound|{registry.port or 80}||{registry.get_service_name()}"


def _build_candidate(
    cluster: str,
    target_id: int,
    kind: str,
    model_name: str,
    weight: Optional[int],
    max_running_requests: Optional[int],
) -> Dict[str, Any]:
    candidate: Dict[str, Any] = {
        "cluster": cluster,
        "targetId": str(target_id),
        "kind": kind,
        "modelName": model_name,
    }
    # Presence of weight is the mode switch (weighted dice roll vs
    # capability scoring); it is omitted, not zeroed, when unset.
    if weight is not None and weight > 0:
        candidate["weight"] = weight
    if max_running_requests is not None:
        candidate["maxRunningRequests"] = max_running_requests
    return candidate


async def _destinations_for_target(
    session: AsyncSession, target: ModelRouteTarget
) -> DestinationTupleList:
    """The same destination list ``calculate_destinations`` builds for
    the mapper path — cluster-registry routing, LoRA-aliased services
    and provider DNS registries included — so LB candidates and mapper
    rules always address the identical clusters. Imported lazily:
    controllers imports this module, and the dependency only runs one
    way at call time."""
    from gpustack.server.controllers import (
        calculate_model_destinations,
        provider_destinations,
    )

    if target.model_id is not None:
        model = await Model.one_by_id(session, target.model_id)
        if model is None or model.deleted_at is not None:
            return []
        return await calculate_model_destinations(
            session, model, target.overridden_model_name
        )
    return await provider_destinations(
        session=session,
        provider_id=target.provider_id,
        provider_model_name=target.overridden_model_name,
    )


def _is_fallback_target(target: ModelRouteTarget) -> bool:
    return bool(target.fallback_status_codes)


async def render_route(
    session: AsyncSession, route: ModelRoute
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]]:
    """Candidates and modelMappers for one route, or None when the
    configuration is unusable: no usable target, or mixed weights —
    some targets weighted and some not leave the latter's semantics
    dangling, so LB is refused for the route rather than guessed.

    Fallback targets (``fallback_status_codes`` set) are not candidates:
    their traffic arrives via Envoy's fallback ingress, never through
    candidate selection, and their weight column means nothing for the
    split — including them would drag every weighted route into the
    mixed-weights refusal."""
    targets: List[ModelRouteTarget] = await ModelRouteTarget.all_by_field(
        session, "route_id", route.id
    )
    active_targets = [
        t
        for t in targets
        if t.deleted_at is None
        and t.state == TargetStateEnum.ACTIVE
        and not _is_fallback_target(t)
    ]
    weighted = [t for t in active_targets if t.weight and t.weight > 0]
    if weighted and len(weighted) != len(active_targets):
        logger.error(
            "Route %s: refusing LB — %d of %d active targets carry a weight; "
            "either all targets are weighted or none may be",
            route.id,
            len(weighted),
            len(active_targets),
        )
        return None

    candidates: List[Dict[str, Any]] = []
    model_mappers: Dict[str, Dict[str, str]] = {}

    for target in active_targets:
        # Provider targets have no instances and no engine metrics;
        # the plugin skips instance selection for kind: provider.
        kind = "provider" if target.provider_id is not None else "instance"
        registries = await _destinations_for_target(session, target)
        for _, model_name, registry in registries:
            candidates.append(
                _build_candidate(
                    candidate_cluster_name(registry),
                    target.id,
                    kind,
                    model_name,
                    target.weight,
                    target.max_running_requests,
                )
            )
        if registries:
            # The main-path rewrite rides the LB selection: each candidate
            # already carries its own upstream name, and the wildcard keeps
            # the mapping total when the caller's model name is unknown to
            # this target.
            model_mappers[str(target.id)] = {"*": registries[0][1]}

    if not candidates:
        return None
    return candidates, model_mappers


def cluster_header_envoy_filter(
    name: str, namespace: str, route_name: str
) -> EnvoyFilter:
    return EnvoyFilter(
        metadata={
            "name": name,
            "namespace": namespace,
            "labels": {**managed_labels},
        },
        spec={
            "configPatches": [
                {
                    "applyTo": "HTTP_ROUTE",
                    "match": {
                        "context": "GATEWAY",
                        "routeConfiguration": {
                            "vhost": {"route": {"name": route_name}}
                        },
                    },
                    "patch": {
                        "operation": "MERGE",
                        "value": {"route": {"cluster_header": TARGET_CLUSTERS_HEADER}},
                    },
                }
            ]
        },
    )


async def _ensure_envoy_filter(
    namespace: str,
    ingress_name: str,
    istio_networking_api: NetworkingIstioIoV1Alpha3Api,
    present: bool,
) -> None:
    name = envoy_filter_name(ingress_name)
    if not present:
        try:
            await istio_networking_api.delete_envoyfilter(
                name=name, namespace=namespace
            )
            logger.info("Deleted LB EnvoyFilter %s in namespace %s.", name, namespace)
        except ApiException as e:
            if e.status != 404:
                raise
        return

    body = cluster_header_envoy_filter(
        name=name, namespace=namespace, route_name=ingress_name
    )
    try:
        await istio_networking_api.create_envoyfilter(namespace=namespace, body=body)
        logger.info("Created LB EnvoyFilter %s in namespace %s.", name, namespace)
    except ApiException as e:
        if e.status != 409:
            raise
        existing = EnvoyFilter.model_validate(
            await istio_networking_api.get_envoyfilter(namespace=namespace, name=name)
        )
        if existing.spec != body.spec:
            existing.spec = body.spec
            await istio_networking_api.edit_envoyfilter(
                name=name, namespace=namespace, body=existing
            )
            logger.info("Updated LB EnvoyFilter %s in namespace %s.", name, namespace)


async def sync_model_route_lb(
    cfg: Config,
    session: AsyncSession,
    collector: RouteArtifactCollector,
    istio_networking_api: NetworkingIstioIoV1Alpha3Api,
    model_route: ModelRoute,
    ingress_name: str,
    event_is_delete: bool,
) -> None:
    """Reconcile one route's LB gateway artifacts. ``ingress_name`` is
    the bare mcp-handler style name (``ai-route-route-<id>.internal``);
    its prefixed form addresses the matchRule, the bare form is the
    Istio route name the EnvoyFilter matches.

    The matchRule is declared on the collector (one flush per CR across
    the mapper sync and every plugin) and the EnvoyFilter — a
    single-owner, per-route resource — is written directly.
    """
    ingress_prefix = f"{cfg.get_namespace()}/"
    if cfg.get_namespace() == cfg.gateway_namespace:
        ingress_prefix = ""
    full_ingress_name = f"{ingress_prefix}{ingress_name}"

    rule_config: Optional[Dict[str, Any]] = None
    policy_config: Optional[LBPolicyConfig] = None
    if not event_is_delete and lb_module_available(cfg):
        # fresh read: the event's model_route may predate the latest
        # meta write, and the policy rides route.meta now
        fresh_route = await ModelRoute.one_by_id(session, model_route.id)
        if fresh_route is not None and fresh_route.deleted_at is None:
            policy_config = lb_policy_from_meta(fresh_route.meta)
        # LB is the default routing engine: candidates are rendered for
        # every route with usable targets — weight splitting, rr and the
        # cluster choice are all delegated to the gateway plugin, with
        # the cluster_header EnvoyFilter displacing Envoy's own
        # weighted_clusters. ``meta["lb"]`` carries only the optional
        # extras (capabilities, health overrides), and
        # ``enabled: false`` is the single opt-out.
        if policy_config is None or policy_config.enabled:
            rendered = await render_route(session, model_route)
            if rendered is not None:
                candidates, model_mappers = rendered
                rule_config = {
                    "candidates": candidates,
                    "modelMappers": model_mappers,
                }
                if policy_config is not None:
                    # the per-route policy knobs ride the same rule — a
                    # matchRule config overrides the CR defaultConfig,
                    # so health/reject/maxBodyBytes take effect per
                    # route without a deployment-level precedence rule
                    rule_config.update(policy_config.to_gateway_rule())

    collector.set_rules(
        cr_name=CONTEXT_CR_NAME,
        owner="lb",
        ingresses=[full_ingress_name],
        rules=(
            [
                WasmPluginMatchRule(
                    config=rule_config,
                    ingress=[full_ingress_name],
                    configDisable=False,
                )
            ]
            if rule_config is not None
            else []
        ),
    )
    await _ensure_envoy_filter(
        namespace=cfg.gateway_namespace,
        ingress_name=ingress_name,
        istio_networking_api=istio_networking_api,
        present=rule_config is not None,
    )
