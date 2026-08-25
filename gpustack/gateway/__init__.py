import time
import asyncio
import base64
import os
import logging
import yaml
import copy
from functools import partial
from typing import Any, Dict, Tuple, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from kubernetes_asyncio import client as k8s_client
from kubernetes_asyncio.client import Configuration
from kubernetes_asyncio.config.kube_config import KubeConfigLoader, KubeConfigMerger
from kubernetes_asyncio.config.incluster_config import (
    InClusterConfigLoader,
    SERVICE_TOKEN_FILENAME,
    SERVICE_CERT_FILENAME,
)
from kubernetes_asyncio.client.rest import ApiException
from gpustack.api.auth import (
    GATEWAY_ASSERTED_ACCESS_KEY_HEADER,
    GATEWAY_ASSERTED_KEY_REF_HEADER,
    GATEWAY_AUTH_TOKEN_HEADER,
)
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack import envs
from gpustack.gateway import client as gw_client
from gpustack.gateway.client import (
    McpBridge,
    McpBridgeSpec,
    McpBridgeRegistry,
    WasmPluginSpec,
)
from gpustack.gateway.labels_annotations import managed_labels, match_labels
from gpustack.gateway.utils import (
    default_mcp_bridge_name,
    openai_model_prefixes,
    anthropic_model_exact,
    gpustack_ai_proxy_name,
    gpustack_model_mapper_name,
    gpustack_generic_proxy_router_name,
    mcp_ingress_equal,
    get_default_mcpbridge_ref,
    ensure_wasm_plugin,
    router_header_key,
    gpustack_original_path_header,
    gpustack_fallback_path_header,
)
from gpustack.gateway.ext_auth import (
    ext_auth_init_spec_diff,
    ext_auth_resource_name,
    ext_auth_spec,
)
from gpustack.gateway.plugins import plugin_entry, plugin_spec_overrides

logger = logging.getLogger(__name__)

mcp_registry_port = 80

# The caller identity every Higress plugin agrees on by constant, ext-auth
# included. Named here because two plugins below have to spell it identically:
# one strips whatever the client sent, the other reads whatever survives.
consumer_header = "x-mse-consumer"

supported_openai_routes = [
    route for v in openai_model_prefixes for route in v.flattened_prefixes()
]

supported_anthropic_routes = [
    route for v in anthropic_model_exact for route in v.flattened_prefixes()
]

async_gateway_config: Configuration = None


def init_async_k8s_config(cfg: Config):
    if cfg.gateway_mode == GatewayModeEnum.disabled:
        return
    global async_gateway_config
    if async_gateway_config is not None:
        return
    configuration = Configuration()
    if cfg.gateway_mode == GatewayModeEnum.incluster:
        cfg_loader = InClusterConfigLoader(
            token_filename=SERVICE_TOKEN_FILENAME,
            cert_filename=SERVICE_CERT_FILENAME,
        )
        cfg_loader.load_and_set(configuration)
    else:
        cfg_loader = KubeConfigLoader(
            config_dict=KubeConfigMerger(cfg.gateway_kubeconfig).config
        )
        if not cfg_loader._load_user_token():
            cfg_loader._load_user_pass_token()
        cfg_loader._load_cluster_info()
        cfg_loader._set_config(configuration)
    async_gateway_config = configuration


def get_async_k8s_config(cfg: Config) -> Optional[Configuration]:
    if cfg.gateway_mode == GatewayModeEnum.disabled:
        return None
    global async_gateway_config
    if async_gateway_config is None:
        init_async_k8s_config(cfg=cfg)
    return async_gateway_config


def wait_for_apiserver_ready(cfg: Config, timeout: int = 60, interval: int = 5):
    async def get_api_resources():
        config = get_async_k8s_config(cfg)
        start = time.time()
        v1 = k8s_client.CoreV1Api(k8s_client.ApiClient(configuration=config))
        while True:
            try:
                await v1.get_api_resources()
                break
            except Exception:
                if time.time() - start > timeout:
                    raise
                await asyncio.sleep(interval)

    try:
        asyncio.run(get_api_resources())
    except asyncio.CancelledError:
        raise


def get_gpustack_higress_registry(cfg: Config) -> McpBridgeRegistry:
    registry_type = "dns"
    domain = f"{cfg.service_discovery_name}.{cfg.get_namespace()}.svc"
    mcp_port = cfg.get_api_port()
    if cfg.gateway_mode != GatewayModeEnum.incluster:
        registry_type = "static"
        mcp_port = mcp_registry_port
        port = cfg.get_api_port()
        if cfg.gateway_mode == GatewayModeEnum.external:
            address = cfg.get_advertise_address()
        elif cfg.gateway_mode == GatewayModeEnum.embedded:
            address = "127.0.0.1"
        domain = f"{address}:{port}"

    mcp_registry_name = (
        "gpustack"
        if cfg.server_role() != Config.ServerRole.WORKER
        else "gpustack-worker"
    )
    registry = McpBridgeRegistry(
        type=registry_type,
        name=mcp_registry_name,
        port=mcp_port,
        protocol="http",
        domain=domain,
    )
    return registry


async def ensure_mcp_resources(cfg: Config, api_client: k8s_client.ApiClient):
    api = gw_client.NetworkingHigressIoV1Api(api_client)
    # use default name for embedded mode
    gateway_namespace = cfg.gateway_namespace
    try:
        data: Dict[str, Any] = await api.get_mcpbridge(
            namespace=gateway_namespace, name=default_mcp_bridge_name
        )
        default_bridge = McpBridge.model_validate(data)
    except ApiException as e:
        if e.status == 404:
            default_bridge = None
        else:
            raise
    target_registry = get_gpustack_higress_registry(cfg=cfg)
    try:
        if not default_bridge:
            bridge = McpBridge(
                metadata={"name": "default", "namespace": gateway_namespace},
                spec=McpBridgeSpec(registries=[target_registry]),
            )
            await api.create_mcpbridge(namespace=gateway_namespace, body=bridge)
        else:
            should_update = False
            registries = (
                default_bridge.spec.registries
                if default_bridge.spec and default_bridge.spec.registries
                else []
            )
            if not any(r.name == target_registry.name for r in registries):
                if default_bridge.spec is None:
                    default_bridge.spec = McpBridgeSpec()
                registries.append(target_registry)
                default_bridge.spec.registries = registries
                should_update = True
            else:
                registry = next(r for r in registries if r.name == target_registry.name)
                if (
                    registry.type != target_registry.type
                    or registry.domain != target_registry.domain
                    or registry.port != target_registry.port
                    or registry.protocol != target_registry.protocol
                ):
                    registry.type = target_registry.type
                    registry.domain = target_registry.domain
                    registry.port = target_registry.port
                    registry.protocol = target_registry.protocol
                    should_update = True
            if should_update:
                await api.edit_mcpbridge(
                    namespace=gateway_namespace, name='default', body=default_bridge
                )
    except ApiException as e:
        raise RuntimeError("Failed to ensure ingress resources") from e


async def ensure_ingress_resources(cfg: Config, api_client: k8s_client.ApiClient):
    """
    Ensure the ingress resources to route traffic to mcpbridge are created.
    """
    gateway_namespace = cfg.gateway_namespace
    hostname = cfg.get_external_hostname()
    tls_secret_name = cfg.get_tls_secret_name()
    network_v1_client = k8s_client.NetworkingV1Api(api_client=api_client)
    ingress_name = envs.GATEWAY_MIRROR_INGRESS_NAME
    try:
        ingress: k8s_client.V1Ingress = await network_v1_client.read_namespaced_ingress(
            name=ingress_name, namespace=gateway_namespace
        )
    except ApiException as e:
        if e.status == 404:
            ingress = None
        else:
            raise
    registry = get_gpustack_higress_registry(cfg=cfg)
    expected_rule = k8s_client.V1IngressRule(
        http=k8s_client.V1HTTPIngressRuleValue(
            paths=[
                k8s_client.V1HTTPIngressPath(
                    path="/",
                    path_type="Prefix",
                    backend=k8s_client.V1IngressBackend(
                        resource=get_default_mcpbridge_ref()
                    ),
                )
            ]
        ),
    )

    expected_ingress = k8s_client.V1Ingress(
        metadata=k8s_client.V1ObjectMeta(
            name=ingress_name,
            namespace=gateway_namespace,
            annotations={
                "higress.io/destination": f"{registry.get_service_name_with_port()}",
                "higress.io/ignore-path-case": "false",
            },
            labels=managed_labels,
        ),
        spec=k8s_client.V1IngressSpec(
            ingress_class_name=cfg.gateway_ingress_class,
            rules=[expected_rule],
        ),
    )
    if tls_secret_name is not None:
        expected_ingress.spec.tls = [
            k8s_client.V1IngressTLS(
                hosts=[hostname] if hostname is not None else None,
                secret_name=tls_secret_name,
            )
        ]
    if hostname is not None:
        host_rule = copy.deepcopy(expected_rule)
        host_rule.host = hostname
        expected_ingress.spec.rules.append(host_rule)

    if not ingress:
        await network_v1_client.create_namespaced_ingress(
            namespace=gateway_namespace, body=expected_ingress
        )
    elif match_labels(getattr(ingress.metadata, 'labels', {}), managed_labels):
        # only update ingress managed by gpustack
        if not mcp_ingress_equal(ingress, expected_ingress):
            await network_v1_client.replace_namespaced_ingress(
                name=ingress_name, namespace=gateway_namespace, body=expected_ingress
            )


def ext_auth_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    """The static base of the ext-auth CR.

    Scoped to gpustack's own inference routes by route-name prefix, so
    unrelated traffic on a shared Higress gateway is never authenticated:
    model-route ingresses are named ``ai-route-route-<id>.internal`` (plus
    ``.fallback.``), nothing else matches, and non-matching routes pass
    through untouched.

    The key tables and the PUBLIC route rules are absent here and filled in by
    :class:`~gpustack.server.gateway_auth_reconciler.GatewayAuthReconciler`;
    see :func:`gpustack.gateway.ext_auth.ext_auth_init_spec_diff` for how they
    survive this rewrite on restart.
    """
    return ext_auth_resource_name, ext_auth_spec(
        cfg=cfg, registry=get_gpustack_higress_registry(cfg=cfg)
    )


# Manifest name, which is also the ``gateway_plugin`` key -- distinct from the
# ``gpustack-ai-statistics`` resource it is deployed as. Kept beside the model
# that consumes it, as ext_auth.py does with its own.
ai_statistics_plugin_name = "ai-statistics"


class AiStatisticsOverride(BaseModel):
    """``gateway_plugin["ai-statistics"].config``.

    ``attributes`` is not settable: the ``consumer`` attribute is what carries
    caller identity into the access log, and this plugin is the only thing that
    writes it there.
    """

    model_config = ConfigDict(extra="forbid")

    # Bodies of these content types are parsed for token usage. ``audio/pcm`` is
    # rejected below rather than declared away, so the error names the value.
    #
    # Defaults to the deprecated GPUSTACK_GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES
    # when that is set: dropping it silently would stop metering a content type
    # somebody had added, which surfaces as a wrong bill rather than an error.
    enable_content_types: List[str] = Field(
        default_factory=envs.deprecated_ai_statistics_content_types
    )


def ai_statistics_override(cfg: Config) -> AiStatisticsOverride:
    entry = plugin_entry(ai_statistics_plugin_name, cfg)
    if entry is None or not entry.config:
        return AiStatisticsOverride()
    try:
        return AiStatisticsOverride.model_validate(entry.config)
    except ValidationError as e:
        raise ValueError(
            f"Invalid gateway_plugin.{ai_statistics_plugin_name}.config: {e}"
        ) from e


def ai_statistics_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    resource_name = "gpustack-ai-statistics"
    expected_spec = WasmPluginSpec(
        defaultConfig={
            "enable_content_types": ai_statistics_override(cfg).enable_content_types,
            "attributes": [
                {
                    "apply_to_log": True,
                    "apply_to_span": False,
                    "key": "consumer",
                    "value": consumer_header,
                    "value_source": "request_header",
                }
            ],
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="UNSPECIFIED_PHASE",
        priority=900,
        **plugin_spec_overrides(ai_statistics_plugin_name, cfg=cfg),
    )
    return resource_name, expected_spec


def model_pre_route_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    resource_name = "gpustack-set-model-pre-route"
    enabled_path_suffixes = supported_openai_routes + supported_anthropic_routes
    enabled_path_prefixes = ["/model/proxy"]
    expected_spec = WasmPluginSpec(
        defaultConfig={
            'clusterNameHeader': router_header_key,
            'routeNameHeader': 'X-GPUStack-Route-Name',
            'enableOnPathSuffix': enabled_path_suffixes,
            'enableOnPathPrefix': enabled_path_prefixes,
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="AUTHN",
        priority=90,
        **plugin_spec_overrides("gpustack-set-header-pre-route", cfg=cfg),
    )
    return resource_name, expected_spec


def model_mapper_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    return gpustack_model_mapper_name, WasmPluginSpec(
        phase="AUTHN",
        priority=800,
        **plugin_spec_overrides("gpustack-model-mapper", cfg=cfg),
        defaultConfigDisable=False,
        defaultConfig={"modelMapping": {}},
        matchRules=[],
        failStrategy="FAIL_OPEN",
    )


class HeaderRule(BaseModel):
    key: Optional[str] = None
    newKey: Optional[str] = None
    oldKey: Optional[str] = None
    fromKey: Optional[str] = None
    toKey: Optional[str] = None
    value: Optional[str] = None
    newValue: Optional[str] = None
    appendValue: Optional[str] = None
    value_type: Optional[Literal["object", "bool", "number", "string"]] = None
    strategy: Optional[Literal["RETAIN_FIRST", "RETAIN_LAST", "RETAIN_UNIQUE"]] = None
    host_pattern: Optional[str] = None
    path_pattern: Optional[str] = None


def transform_header(
    operate: Literal["remove", "rename", "replace", "add", "append", "map", "dedupe"],
    *rules: HeaderRule,
) -> Dict[str, Any]:
    # TODO: add validation in the future
    return {
        "headers": [rule.model_dump(exclude_none=True) for rule in rules],
        "operate": operate,
    }


def transformer_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    resource_name = "gpustack-header-transformer"
    expected_spec = WasmPluginSpec(
        defaultConfig={
            "reqRules": [
                # Strip everything a client must not be able to say about
                # itself. Runs at priority 810, ahead of ext-auth at 360 in the
                # same AUTHN phase, so removal always precedes injection of the
                # trusted values.
                #
                # The two identity headers are as load-bearing as the gateway
                # token: the server skips authentication entirely when it sees
                # them behind a valid gateway token, so a client that could set
                # one would be able to name any access_key -- and with
                # authentication moved to the edge there is no second gate
                # behind this one.
                #
                # ``x-mse-consumer`` is stripped here rather than left to
                # ext-auth because ext-auth only reaches its own removal on
                # routes it owns -- its route gate returns first for everything
                # else. This plugin has no such gate, so it covers the routes
                # ext-auth declines: the control-plane mirror ingress, and any
                # other tenant on a shared gateway. Without it a client could
                # name itself on those routes and be believed, because
                # ai-statistics and token-usage read this header globally and
                # write it into the access log and the usage records. Stripping
                # it globally is safe precisely because ext-auth runs later and
                # replaces it with an authoritative value where it applies.
                transform_header(
                    "remove",
                    HeaderRule(
                        key=GATEWAY_AUTH_TOKEN_HEADER,
                    ),
                    HeaderRule(
                        key=router_header_key,
                    ),
                    HeaderRule(
                        key=GATEWAY_ASSERTED_ACCESS_KEY_HEADER,
                    ),
                    HeaderRule(
                        key=GATEWAY_ASSERTED_KEY_REF_HEADER,
                    ),
                    HeaderRule(
                        key=consumer_header,
                    ),
                ),
                transform_header(
                    "rename",
                    HeaderRule(
                        oldKey="x-gpustack-model",
                        newKey="x-higress-llm-model",
                    ),
                    HeaderRule(
                        oldKey=gpustack_fallback_path_header,
                        newKey=":path",
                    ),
                ),
                transform_header(
                    "dedupe",
                    HeaderRule(
                        key="x-gpustack-model",
                        strategy="RETAIN_FIRST",
                    ),
                    HeaderRule(
                        key="x-higress-llm-model",
                        strategy="RETAIN_FIRST",
                    ),
                    HeaderRule(
                        key=":path",
                        strategy="RETAIN_LAST",
                    ),
                ),
                transform_header(
                    "map",
                    HeaderRule(
                        fromKey=':path',
                        toKey=gpustack_original_path_header,
                    ),
                ),
                transform_header(
                    "remove",
                    HeaderRule(
                        key=gpustack_fallback_path_header,
                    ),
                ),
            ],
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="AUTHN",
        priority=810,
        **plugin_spec_overrides("transformer", cfg=cfg),
    )
    return resource_name, expected_spec


def generic_proxy_router_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    """
    Generic-proxy router. Operates in two complementary modes per request and
    fully replaces Higress's built-in ``model-router`` — that plugin must be
    disabled wherever this one is active, otherwise both will write
    ``x-higress-llm-model`` and the winner becomes priority-dependent.

    1. **Path-driven**: when ``:path`` matches ``prefix`` (``/model/proxy/``),
       the first segment after the prefix is the alias id; it is looked up in
       ``aliasNameMapping`` and the resolved model name is written into
       ``targetHeader`` and back into the body's ``model`` field (JSON or
       multipart). Used by gpustack's generic-proxy routes.
    2. **Body-driven** (model-router parity): when the path doesn't match
       ``prefix`` but does match ``enableOnPathSuffix``, the ``model`` field
       is read from the JSON / multipart body and projected into
       ``targetHeader``. Used by the standard OpenAI / Anthropic AI routes.

    Priority 900 matches model-router's original slot, so downstream plugins
    (ext-auth at 360, rate-limit ~600) keep observing the resolved model name
    without any other ordering change. Phase ``AUTHN`` ensures the header is
    set before ext-auth runs.

    ``defaultConfig.aliasNameMapping`` is maintained per-route by
    [[generic_proxy_router_diff_spec]] — each entry maps ``str(route_id)`` to
    the route's effective model name (with owner-name prefix for
    non-platform Orgs). The reconciler only mutates ``aliasNameMapping``;
    it never touches
    ``defaultConfigDisable``, since flipping that rewrites Envoy's filter
    chain and tears down every live connection.

    Runtime shape after two generic routes (id=1 "route-one", id=2 "route-two")
    have been reconciled:

        apiVersion: extensions.higress.io/v1alpha1
        kind: WasmPlugin
        metadata:
          name: gpustack-model-router
        spec:
          phase: AUTHN
          priority: 900
          defaultConfigDisable: false
          defaultConfig:
            prefix: "/model/proxy/"
            targetHeader: "x-higress-llm-model"
            enableOnPathSuffix:
              - /v1/chat/completions
              - /v1/messages
              - ...
            aliasNameMapping:
              "1": "route-one"
              "2": "route-two"
    """
    # K8s WasmPlugin resource name — intentionally reuses the legacy
    # ``gpustack-model-router`` slot previously occupied by Higress's built-in
    # model-router so the upgrade is an in-place swap. The plugin image name
    # passed to ``get_plugin_url_with_name_and_version`` below stays
    # ``gpustack-generic-proxy-router`` (its identity in the plugin manifest).
    resource_name = gpustack_generic_proxy_router_name
    enabled_path_suffixes = supported_openai_routes + supported_anthropic_routes
    expected_spec = WasmPluginSpec(
        defaultConfig={
            "prefix": "/model/proxy/",
            "targetHeader": "x-higress-llm-model",
            "enableOnPathSuffix": enabled_path_suffixes,
            "aliasNameMapping": {},
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="AUTHN",
        priority=900,
        **plugin_spec_overrides("gpustack-generic-proxy-router", cfg=cfg),
    )
    return resource_name, expected_spec


def token_usage_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    registry = get_gpustack_higress_registry(cfg=cfg)
    resource_name = "gpustack-token-usage"
    expected_spec = WasmPluginSpec(
        defaultConfig={
            'endpoint': {
                "path": "/v2/usage/gateway-metrics",
                "service_name": registry.get_service_name(),
                "service_port": registry.port,
            },
            'header_add': {
                GATEWAY_AUTH_TOKEN_HEADER: cfg.get_derived_gateway_token(),
            },
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="UNSPECIFIED_PHASE",
        priority=400,
        **plugin_spec_overrides("gpustack-token-usage", cfg=cfg),
    )
    return resource_name, expected_spec


def ai_proxy_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    resource_name = gpustack_ai_proxy_name
    expected_spec = WasmPluginSpec(
        defaultConfig={},
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        priority=100,
        phase="UNSPECIFIED_PHASE",
        **plugin_spec_overrides("gpustack-ai-proxy", cfg=cfg),
    )
    return resource_name, expected_spec


async def ensure_tls_secret(cfg: Config, api_client: k8s_client.ApiClient):
    """
    Ensure the TLS secret if ssl key pair is provided.
    """
    ssl_keyfile = cfg.ssl_keyfile
    ssl_certfile = cfg.ssl_certfile
    if not ssl_keyfile or not ssl_certfile:
        return
    if not (os.path.isfile(ssl_keyfile) and os.path.isfile(ssl_certfile)):
        raise RuntimeError(
            f"SSL keyfile {ssl_keyfile} or certfile {ssl_certfile} does not exist"
        )

    # read key and cert files and encode into base64
    with open(ssl_keyfile, 'rb') as f:
        ssl_key_bytes = f.read()
    with open(ssl_certfile, 'rb') as f:
        ssl_cert_bytes = f.read()
    ssl_key_data = base64.b64encode(ssl_key_bytes).decode()
    ssl_cert_data = base64.b64encode(ssl_cert_bytes).decode()

    gateway_namespace = cfg.gateway_namespace
    core_v1_client = k8s_client.CoreV1Api(api_client=api_client)
    secret_name = cfg.get_tls_secret_name()
    to_create_tls_secret = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(
            name=secret_name,
            namespace=gateway_namespace,
            labels=managed_labels,
        ),
        type="kubernetes.io/tls",
        data={
            "tls.key": ssl_key_data,
            "tls.crt": ssl_cert_data,
        },
    )
    try:
        existing_secret: k8s_client.V1Secret = (
            await core_v1_client.read_namespaced_secret(
                name=secret_name, namespace=gateway_namespace
            )
        )
    except ApiException as e:
        if e.status == 404:
            existing_secret = None
        else:
            raise
    if not existing_secret:
        await core_v1_client.create_namespaced_secret(
            namespace=gateway_namespace, body=to_create_tls_secret
        )
    elif match_labels(getattr(existing_secret.metadata, 'labels', {}), managed_labels):
        if existing_secret.data != to_create_tls_secret.data:
            await core_v1_client.replace_namespaced_secret(
                name=secret_name, namespace=gateway_namespace, body=to_create_tls_secret
            )


async def ensure_gateway_timeout(cfg: Config, api_client: k8s_client.ApiClient):
    namespace = cfg.gateway_namespace
    higress_config_name = "higress-config"
    core_v1_client = k8s_client.CoreV1Api(api_client=api_client)
    try:
        higress_config: k8s_client.V1ConfigMap = (
            await core_v1_client.read_namespaced_config_map(
                name=higress_config_name, namespace=namespace
            )
        )
        should_update = False
        config_data: str = higress_config.data["higress"]
        config = yaml.safe_load(config_data)
        idle_timeout = (
            config.get("downstream", {}).get("idleTimeout")
            if isinstance(config, dict)
            else None
        )
        if idle_timeout is None or str(idle_timeout) != f"{envs.PROXY_TIMEOUT}":
            config.setdefault("downstream", {})["idleTimeout"] = envs.PROXY_TIMEOUT
            higress_config.data["higress"] = yaml.safe_dump(config)
            should_update = True
        upstream_idle_timeout = (
            config.get("upstream", {}).get("idleTimeout")
            if isinstance(config, dict)
            else None
        )
        if (
            upstream_idle_timeout is None
            or str(upstream_idle_timeout) != f"{envs.PROXY_UPSTREAM_IDLE_TIMEOUT}"
        ):
            config.setdefault("upstream", {})[
                "idleTimeout"
            ] = envs.PROXY_UPSTREAM_IDLE_TIMEOUT
            higress_config.data["higress"] = yaml.safe_dump(config)
            should_update = True
        if should_update:
            await core_v1_client.replace_namespaced_config_map(
                name=higress_config_name,
                namespace=namespace,
                body=higress_config,
            )
    except Exception as e:
        logger.error(f"Failed to read or parse Higress config map: {e}")
        raise


def spec_replace(
    current_spec: Optional[WasmPluginSpec],
    expected_spec: WasmPluginSpec,
    create_only: bool = False,
) -> WasmPluginSpec:
    if current_spec is None:
        return expected_spec
    if create_only:
        if current_spec.url != expected_spec.url:
            current_spec.url = expected_spec.url
        if current_spec.sha256 != expected_spec.sha256:
            current_spec.sha256 = expected_spec.sha256
        return current_spec
    return expected_spec


# Keys in the generic-proxy router's defaultConfig that survive init-time
# diffs: when a key is non-None on the live spec, the diff carries it over
# instead of letting the factory's defaults overwrite it. Add a key here when
# you introduce a new field that should be either controller-managed (like
# ``aliasNameMapping``) or operator-tunable (like ``maxBodyBytes``).
_GENERIC_PROXY_ROUTER_PRESERVED_KEYS = [
    "aliasNameMapping",
    "maxBodyBytes",
]


def generic_proxy_router_spec_diff(
    current_spec: Optional[WasmPluginSpec],
    expected_spec: WasmPluginSpec,
) -> WasmPluginSpec:
    """
    Init-time diff for the generic-proxy router. Returns a fresh spec based on
    ``expected_spec`` so every ``defaultConfig`` key (prefix, targetHeader,
    enableOnPathSuffix, ...) is refreshed on each startup, except for keys
    listed in ``_GENERIC_PROXY_ROUTER_PRESERVED_KEYS`` — those are carried
    over from the live spec whenever the live value is non-None.

    The input ``expected_spec`` is not mutated, so the caller can reuse it
    across invocations.
    """
    if current_spec is None:
        return expected_spec
    merged_default_config = dict(expected_spec.defaultConfig or {})
    current_default_config = current_spec.defaultConfig or {}
    for key in _GENERIC_PROXY_ROUTER_PRESERVED_KEYS:
        value = current_default_config.get(key)
        if value is not None:
            merged_default_config[key] = value
    return expected_spec.model_copy(update={"defaultConfig": merged_default_config})


def validate_ai_statistics_plugin_content_types(cfg: Config):
    for content_type in ai_statistics_override(cfg).enable_content_types:
        if content_type == "audio/pcm":
            raise ValueError(
                "audio/pcm content type is not supported in ai statistics plugin"
            )


def initialize_gateway(cfg: Config, timeout: int = 60, interval: int = 5):
    if cfg.gateway_mode == GatewayModeEnum.disabled:
        return
    init_async_k8s_config(cfg=cfg)
    wait_for_apiserver_ready(cfg=cfg, timeout=timeout, interval=interval)
    if cfg.gateway_mode in [
        GatewayModeEnum.embedded,
        GatewayModeEnum.external,
        GatewayModeEnum.incluster,
    ]:
        validate_ai_statistics_plugin_content_types(cfg=cfg)
        plugin_list: List[Tuple[str, WasmPluginSpec]] = [
            ext_auth_plugin(cfg=cfg),
            ai_statistics_plugin(cfg=cfg),
            generic_proxy_router_plugin(cfg=cfg),
            ai_proxy_plugin(cfg=cfg),
            model_pre_route_plugin(cfg=cfg),
            model_mapper_plugin(cfg=cfg),
        ]
        if cfg.server_role() != Config.ServerRole.WORKER:
            plugin_list.append(transformer_plugin(cfg=cfg))
            plugin_list.append(token_usage_plugin(cfg=cfg))

        async def prepare():
            api_client = k8s_client.ApiClient(
                configuration=get_async_k8s_config(cfg=cfg)
            )
            await ensure_tls_secret(cfg=cfg, api_client=api_client)
            await ensure_mcp_resources(cfg=cfg, api_client=api_client)
            if cfg.gateway_mode != GatewayModeEnum.incluster:
                await ensure_gateway_timeout(cfg=cfg, api_client=api_client)
                await ensure_ingress_resources(cfg=cfg, api_client=api_client)
            for plugin_name, plugin_spec in plugin_list:
                if plugin_name == gpustack_generic_proxy_router_name:
                    spec_diff_func = partial(
                        generic_proxy_router_spec_diff, expected_spec=plugin_spec
                    )
                elif plugin_name == ext_auth_resource_name:
                    # Hybrid resource: the static base is rewritten from cfg on
                    # every start, the key tables and PUBLIC rules are carried
                    # over from the live CR because the database owns them.
                    spec_diff_func = partial(
                        ext_auth_init_spec_diff, expected_spec=plugin_spec
                    )
                else:
                    create_only = plugin_name in [
                        gpustack_ai_proxy_name,
                        gpustack_model_mapper_name,
                    ]
                    spec_diff_func = partial(
                        spec_replace,
                        expected_spec=plugin_spec,
                        create_only=create_only,
                    )
                await ensure_wasm_plugin(
                    api=gw_client.ExtensionsHigressIoV1Api(api_client),
                    name=plugin_name,
                    namespace=cfg.gateway_namespace,
                    spec_diff=spec_diff_func,
                )
            # Envoy starts loading these the moment Higress pushes the ECDS
            # resource, which is now -- and where a module is fetched over
            # HTTP, that is while the server serving it is still several
            # startup steps from binding its port. Logging the URLs is what
            # makes that visible at all; verify_published_plugin_modules
            # checks them once the API is up.
            logger.info(
                "Published %d WasmPlugin resources in namespace %s: %s",
                len(plugin_list),
                cfg.gateway_namespace,
                ", ".join(
                    f"{name}={spec.url}" for name, spec in plugin_list if spec.url
                ),
            )

        try:
            asyncio.run(prepare())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError("Failed to initialize gateway resources") from e
