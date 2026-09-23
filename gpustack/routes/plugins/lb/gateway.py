"""Static gateway presence of the LB plugin: two WasmPlugin CRs from
one wasm image (``gpustack-lb`` module).

- The **context role** (AUTHN/340) keeps the CR name
  ``gpustack-model-mapper``: same config location, in-place upgrade, and
  on routes without candidates it behaves exactly as model-mapper did.
- The **finisher role** (AUTHN/325) is the new ``gpustack-lb`` CR; its
  config is entirely deployment-level ("does this route do LB" lives in
  exactly one place — the context role's candidates).

Both CRs are declared here, so ``_append_route_plugin_entries`` must
drop the built-in mapper entry that ``initialize_gateway`` also lists —
ownership of a CR name passes to the route plugin that declares it.

The init-pass diff rewrites the static half (url, phase, priority,
defaultConfig) and carries the live matchRules over: the route-driven
half is written by ``reconciler.sync_model_route_lb``, and the two
passes must not fight over one field.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from gpustack.config.config import Config
from gpustack.gateway.client.extensions_higress_io_v1_api import WasmPluginSpec
from gpustack.gateway.client.networking_higress_io_v1_api import McpBridgeRegistry
from gpustack.gateway.plugins import plugin_spec_overrides
from gpustack.routes.plugins import RouteGatewayEntry
from gpustack.routes.plugins.lb.config import LB_CONTEXT_CR_NAME
from gpustack.utils.network import is_ipaddress

logger = logging.getLogger(__name__)

LB_MODULE_NAME = "gpustack-lb"
LB_FINISHER_CR_NAME = "gpustack-lb"

# The redis service registered in the default McpBridge; the plugin's
# redis.service_name points at it via get_service_name() (``.static`` /
# ``.dns`` suffix). Namespaced apart from the enterprise edition's own
# registry (``gpustack-enterprise-redis``) so both editions can coexist
# on one McpBridge.
REDIS_REGISTRY_NAME = "gpustack-redis"

# Filter-chain positions from the plugins README's intended chain: the
# LB band runs after every rejection point (ext-auth 360, ip-acl 350)
# and before the UNSPECIFIED-phase traffic work, with the capability
# plugins between 340 and 325.
LB_CONTEXT_PHASE = "AUTHN"
LB_CONTEXT_PRIORITY = 340
LB_FINISHER_PHASE = "AUTHN"
LB_FINISHER_PRIORITY = 325


def _static_fields(spec: WasmPluginSpec) -> Dict[str, Any]:
    """The fields the init pass owns. matchRules is deliberately absent:
    the reconciler owns it."""
    return spec.model_dump(exclude={"matchRules"}, exclude_none=True)


def _carry_over_match_rules(
    current_spec: Optional[WasmPluginSpec], expected_spec: WasmPluginSpec
) -> WasmPluginSpec:
    """Init-pass diff: rewrite the static half, keep the live
    matchRules (the mapper rules for non-LB routes and the LB rules for
    LB ones). A missing CR returns None so ``ensure_wasm_plugin`` treats
    it as create (with the expected matchRules — none at init)."""
    if current_spec is None:
        return expected_spec
    current_spec_dict = current_spec.model_dump(exclude_none=True)
    current_match_rules = current_spec_dict.get("matchRules")
    merged = _static_fields(expected_spec)
    if current_match_rules:
        merged["matchRules"] = current_match_rules
    return WasmPluginSpec.model_validate(merged)


def _lb_cr(
    name: str, phase: str, priority: int, default_config: Dict[str, Any], cfg: Config
) -> WasmPluginSpec:
    return WasmPluginSpec(
        phase=phase,
        priority=priority,
        **plugin_spec_overrides(LB_MODULE_NAME, cfg=cfg),
        defaultConfigDisable=False,
        defaultConfig=default_config,
        matchRules=[],
        # FAIL_CLOSE, unlike the capability plugins' FAIL_OPEN: the
        # cluster_header EnvoyFilter replaces the route's static
        # destination, so a skipped LB role means no cluster header and a
        # 503 either way — fail closed makes that failure attributable to
        # the filter instead of masquerading as a routing problem.
        failStrategy="FAIL_CLOSE",
    )


def lb_module_available(cfg: Config) -> bool:
    """Whether the shipped plugins manifest can deploy gpustack-lb. When
    it cannot, the degraded gateway entries publish a plain
    gpustack-model-mapper CR instead — and candidates/modelMappers must
    not be rendered onto it: the mapper module cannot perform the LB
    rewrite. In that state the fallback plugin re-emits the legacy
    per-route main-path modelMapping rule, so the model-name rewrite
    keeps working (round-robin over the ingress destinations) until the
    manifest is upgraded (only reachable via an operator URL override
    or a dependency skew — the pinned gpustack-higress-plugins version
    always resolves the module). The degrade is loud-logged at init."""
    try:
        plugin_spec_overrides(LB_MODULE_NAME, cfg=cfg)
        return True
    except ValueError:
        return False


def lb_gateway_entries(cfg: Config) -> List[RouteGatewayEntry]:
    try:
        return _lb_gateway_entries(cfg)
    except ValueError as e:
        # resolve_plugin raises when the shipped plugins manifest does
        # not know the gpustack-lb module (dependency not yet at the
        # version that packages it). Degrade to the plain model-mapper
        # CR rather than failing server startup — LB routes do not
        # exist until the policy table is written anyway. The plain
        # entry must actually be published: without it no component
        # owns the gpustack-model-mapper CR name and model-name
        # rewrite for fallback routes stops working on a fresh
        # cluster. create_only mirrors the pre-plugin built-in: the
        # route-driven matchRules stay with the reconciler.
        logger.error(
            "Not deploying the LB gateway roles (%s); routes keep the "
            "plain model-mapper behaviour. Upgrade gpustack-higress-plugins.",
            e,
        )
        try:
            mapper_spec = WasmPluginSpec(
                phase="AUTHN",
                priority=800,
                **plugin_spec_overrides("gpustack-model-mapper", cfg=cfg),
                defaultConfigDisable=False,
                defaultConfig={"modelMapping": {}},
                matchRules=[],
                failStrategy="FAIL_OPEN",
            )
        except ValueError:
            # Not even the plain model-mapper module resolves from this
            # manifest — nothing left to degrade to.
            return []
        return [
            RouteGatewayEntry(
                name=LB_CONTEXT_CR_NAME,
                spec=mapper_spec,
                create_only=True,
            )
        ]


def _qualify_dns_host(host: Optional[str], namespace: Optional[str]) -> Optional[str]:
    """Expand a bare Service name into a namespaced name for Envoy
    STRICT_DNS: a single-label host (e.g. ``gpustack-redis``) does not
    resolve there even though an in-pod client connects fine. Stops at
    ``.svc`` — the pod's resolv.conf search list completes the cluster
    domain, which stays correct on non-default-domain clusters. FQDNs
    and IP literals pass through untouched."""
    if not host or "." in host or not namespace:
        return host
    return f"{host}.{namespace}.svc"


def _parse_redis_url(redis_url: str) -> Optional[Tuple[Any, int]]:
    """One parse and one validation pass over ``--redis-url``, shared by
    the registry and the plugin-config builders so the two can never
    disagree about whether a URL is usable. Returns ``(parsed,
    url_port)``, or None with the refusal logged: the deployment stays
    on per-process shared data rather than publishing config nothing
    can reach."""
    from urllib.parse import urlparse

    parsed = urlparse(redis_url)
    if parsed.scheme == "rediss":
        logger.error(
            "redis_url uses rediss:// (TLS), which the gpustack-lb "
            "plugin's redis block does not support; staying on "
            "per-process shared data."
        )
        return None
    if parsed.scheme != "redis" or not parsed.hostname:
        logger.error(
            "Invalid redis_url (want redis://host[:port][/db], no "
            "credentials, no TLS); staying on per-process shared data."
        )
        return None
    if parsed.username or parsed.password:
        # The plugin's redis block would carry the credentials verbatim
        # in the WasmPlugin CR spec, readable by any principal with
        # gateway-extension access. Until a secret-backed mechanism
        # exists, a credentialed URL is refused rather than leaked.
        logger.error(
            "redis_url carries credentials; the LB plugin would "
            "materialize them in the WasmPlugin CR. Refusing; staying "
            "on per-process shared data."
        )
        return None
    try:
        url_port = parsed.port
    except ValueError:
        # urlparse raises on a malformed port (non-numeric or out of
        # range); letting it escape would surface as a misleading
        # "upgrade gpustack-higress-plugins" degrade.
        logger.error("Invalid redis_url port; staying on per-process shared data.")
        return None
    return parsed, url_port


def _redis_registry_from_parsed(
    parsed: Any, url_port: int, namespace: Optional[str]
) -> McpBridgeRegistry:
    """The McpBridge registry entry for the redis service behind a
    validated URL: ``static`` for IP hosts (Higress resolves them
    without DNS, address carried in the domain as ``host:port``),
    ``dns`` for hostnames. The registry name is fixed
    (:data:`REDIS_REGISTRY_NAME`) so ``get_service_name()`` — what the
    plugin's ``redis.service_name`` points at — is stable across URL
    edits."""
    registry_type = "static" if is_ipaddress(parsed.hostname) else "dns"
    if registry_type == "static":
        # The static cluster Higress builds from this registry always
        # listens on 80 (outbound|80||<name>.static) whatever the
        # backend speaks — the real address is the domain's host:port.
        # Verified against a live gateway's cluster dump; writing 6379
        # here points the plugin at a cluster that does not exist.
        port = 80
        host = parsed.hostname
        if ":" in host:
            # An IPv6 literal: urlparse strips the brackets, but the
            # host:port domain needs them back to tell the literal
            # apart from the port.
            host = f"[{host}]"
        # host:port only — netloc would drag any userinfo along into the
        # domain, and credentials belong in the plugin config, not in the
        # Envoy cluster address. A static registry encodes the real
        # backend port in the domain, so an omitted port means Redis's
        # own default, never 80.
        domain = f"{host}:{url_port or 6379}"
    else:
        # The standard port when the URL omits it — None would propagate
        # into the registry and the plugin's service_port.
        port = url_port or 6379
        domain = _qualify_dns_host(parsed.hostname, namespace)
    return McpBridgeRegistry(
        name=REDIS_REGISTRY_NAME,
        type=registry_type,
        domain=domain,
        port=port,
        protocol="tcp",
    )


def redis_registry_from_url(
    redis_url: str, namespace: Optional[str] = None
) -> Optional[McpBridgeRegistry]:
    """The McpBridge registry for ``--redis-url``, or None on an
    unusable URL (see _parse_redis_url for every refusal reason)."""
    parsed = _parse_redis_url(redis_url)
    if parsed is None:
        return None
    return _redis_registry_from_parsed(parsed[0], parsed[1], namespace)


def _redis_config_from_parsed(
    parsed: Any, registry: McpBridgeRegistry
) -> Dict[str, Any]:
    """The plugin's ``redis`` config block for a validated URL, keyed on
    the McpBridge registry rather than a raw host: ``service_name`` is
    the registry's service name (``gpustack-redis.static`` /
    ``.dns``), ``service_port`` the registry port."""
    config: Dict[str, Any] = {
        "service_name": registry.get_service_name(),
        "service_port": registry.port,
    }
    db = (parsed.path or "").lstrip("/")
    if db.isdigit():
        config["database"] = int(db)
    return config


def _redis_config_from_url(
    redis_url: Optional[str], namespace: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    if not redis_url:
        return None
    parsed = _parse_redis_url(redis_url)
    if parsed is None:
        return None
    registry = _redis_registry_from_parsed(parsed[0], parsed[1], namespace)
    return _redis_config_from_parsed(parsed[0], registry)


def _lb_gateway_entries(cfg: Config) -> List[RouteGatewayEntry]:
    # Deployment-level redis from --redis-url: the one knob that must
    # land on BOTH roles' defaultConfig (the plugin README's structural
    # rule — one side only fails silently). The service the config
    # points at is the McpBridge registry registered alongside at
    # gateway init (see ensure_mcp_resources).
    namespace = getattr(cfg, "gateway_namespace", None)
    redis_url = getattr(cfg, "redis_url", None)
    redis_block = _redis_config_from_url(redis_url, namespace)

    context_default: Dict[str, Any] = {
        "mode": "context",
        # keeps the model-mapper behaviour on routes without LB
        "modelMapping": {},
    }
    finisher_default: Dict[str, Any] = {"mode": "finisher"}
    if redis_block is not None:
        context_default["redis"] = dict(redis_block)
        finisher_default["redis"] = dict(redis_block)

    context_spec = _lb_cr(
        LB_CONTEXT_CR_NAME,
        LB_CONTEXT_PHASE,
        LB_CONTEXT_PRIORITY,
        context_default,
        cfg,
    )
    finisher_spec = _lb_cr(
        LB_FINISHER_CR_NAME,
        LB_FINISHER_PHASE,
        LB_FINISHER_PRIORITY,
        finisher_default,
        cfg,
    )
    return [
        RouteGatewayEntry(
            name=LB_CONTEXT_CR_NAME,
            spec=context_spec,
            spec_diff=lambda current: _carry_over_match_rules(current, context_spec),
        ),
        RouteGatewayEntry(
            name=LB_FINISHER_CR_NAME,
            spec=finisher_spec,
            spec_diff=lambda current: _carry_over_match_rules(current, finisher_spec),
        ),
    ]
