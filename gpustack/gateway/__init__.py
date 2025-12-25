import time
import asyncio
import base64
import os
from typing import Any, Dict, Tuple, List, Optional
from kubernetes_asyncio import client as k8s_client
from kubernetes_asyncio.client import Configuration
from kubernetes_asyncio.config.kube_config import KubeConfigLoader, KubeConfigMerger
from kubernetes_asyncio.config.incluster_config import (
    InClusterConfigLoader,
    SERVICE_TOKEN_FILENAME,
    SERVICE_CERT_FILENAME,
)
from kubernetes_asyncio.client.rest import ApiException
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
    mcp_ingress_equal,
    get_default_mcpbridge_ref,
    ensure_wasm_plugin,
)
from gpustack.gateway.plugins import (
    get_plugin_url_with_name_and_version,
)

mcp_registry_port = 80

supported_openai_routes = [
    route for v in openai_model_prefixes for route in v.flattened_prefixes()
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
    # default to incluster mode
    registry_type = "dns"
    domain = f"{cfg.service_discovery_name}.{cfg.get_gateway_namespace()}.svc"
    if cfg.gateway_mode != GatewayModeEnum.incluster:
        registry_type = "static"
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
        port=mcp_registry_port,
        protocol="http",
        domain=domain,
    )
    return registry


async def ensure_mcp_resources(cfg: Config, api_client: k8s_client.ApiClient):
    api = gw_client.NetworkingHigressIoV1Api(api_client)
    # use default name for embedded mode
    gateway_namespace = cfg.get_gateway_namespace()
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
    gateway_namespace = cfg.get_gateway_namespace()
    hostname = cfg.get_external_hostname()
    tls_secret_name = cfg.get_tls_secret_name()
    network_v1_client = k8s_client.NetworkingV1Api(api_client=api_client)
    ingress_name = "gpustack"
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

    expected_ingress = k8s_client.V1Ingress(
        metadata=k8s_client.V1ObjectMeta(
            name=ingress_name,
            namespace=gateway_namespace,
            annotations={
                "higress.io/destination": f"{registry.name}.{registry.type}:{registry.port}",
                "higress.io/ignore-path-case": "false",
            },
            labels=managed_labels,
        ),
        spec=k8s_client.V1IngressSpec(
            ingress_class_name='higress',
            rules=[
                k8s_client.V1IngressRule(
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
            ],
        ),
    )
    if tls_secret_name is not None:
        expected_ingress.spec.tls = [
            k8s_client.V1IngressTLS(
                hosts=[hostname] if hostname is not None else None,
                secret_name=tls_secret_name,
            )
        ]
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
    resource_name = "gpustack-llm-ext-auth"
    registry = get_gpustack_higress_registry(cfg=cfg)
    match_list = [
        {"match_rule_path": route, "match_rule_type": "exact"}
        for route in supported_openai_routes
    ]
    match_list.append({"match_rule_path": "/model/proxy", "match_rule_type": "prefix"})
    expected_spec = WasmPluginSpec(
        defaultConfig={
            "http_service": {
                "authorization_request": {
                    "allowed_headers": [
                        {"exact": "X-GPUStack-Real-IP"},
                        {"exact": "x-higress-llm-model"},
                        {"exact": "cookie"},
                    ]
                },
                "authorization_response": {
                    "allowed_upstream_headers": [
                        {"exact": "X-Mse-Consumer"},
                        {"exact": "Authorization"},
                    ]
                },
                "endpoint": {
                    "path": "/token-auth",
                    "request_method": "GET",
                    "service_name": f"{registry.name}.{registry.type}",
                    "service_port": registry.port,
                },
                "endpoint_mode": "forward_auth",
                "timeout": envs.HIGRESS_EXT_AUTH_TIMEOUT_MS,
            },
            "match_list": match_list,
            "match_type": "blacklist",
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        phase="AUTHN",
        priority=360,
        url=get_plugin_url_with_name_and_version(
            name="ext-auth", version="2.0.0", cfg=cfg
        ),
    )
    return resource_name, expected_spec


def ai_statistics_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    resource_name = "gpustack-ai-statistics"
    expected_spec = WasmPluginSpec(
        defaultConfig={
            "attributes": [
                {
                    "apply_to_log": True,
                    "apply_to_span": False,
                    "key": "consumer",
                    "value": "x-mse-consumer",
                    "value_source": "request_header",
                }
            ]
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="UNSPECIFIED_PHASE",
        priority=900,
        url=get_plugin_url_with_name_and_version(
            name="ai-statistics", version="2.0.0", cfg=cfg
        ),
    )
    return resource_name, expected_spec


def model_router_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    resource_name = "gpustack-model-router"
    enabled_paths = supported_openai_routes.copy()
    enabled_paths.append("/model/proxy")
    expected_spec = WasmPluginSpec(
        defaultConfig={
            'modelToHeader': 'x-higress-llm-model',
            'enableOnPathSuffix': enabled_paths,
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="AUTHN",
        priority=900,
        url=get_plugin_url_with_name_and_version(
            name="model-router", version="2.0.0", cfg=cfg
        ),
    )
    return resource_name, expected_spec


def transformer_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    resource_name = "gpustack-header-transformer"
    expected_spec = WasmPluginSpec(
        defaultConfig={
            "reqRules": [
                {
                    "headers": [
                        {
                            "newKey": "x-higress-llm-model",
                            "oldKey": "x-gpustack-model",
                        }
                    ],
                    "operate": "rename",
                }
            ],
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="AUTHN",
        priority=410,
        url=get_plugin_url_with_name_and_version(
            name="transformer", version="2.0.0", cfg=cfg
        ),
    )
    return resource_name, expected_spec


def token_usage_plugin(cfg: Config) -> Tuple[str, WasmPluginSpec]:
    resource_name = "gpustack-token-usage"
    expected_spec = WasmPluginSpec(
        defaultConfig={
            'realIPToHeader': "X-GPUStack-Real-IP",
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="AUTHN",
        priority=900,
        url=get_plugin_url_with_name_and_version(
            name="gpustack-token-usage", version="1.0.0", cfg=cfg
        ),
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

    gateway_namespace = cfg.get_gateway_namespace()
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


def initialize_gateway(cfg: Config, timeout: int = 60, interval: int = 5):
    if cfg.gateway_mode == GatewayModeEnum.disabled:
        return
    init_async_k8s_config(cfg=cfg)
    wait_for_apiserver_ready(cfg=cfg, timeout=timeout, interval=interval)
    if cfg.gateway_mode in [
        GatewayModeEnum.embedded,
        GatewayModeEnum.external,
    ]:
        plugin_list: List[Tuple[str, WasmPluginSpec]] = [
            ext_auth_plugin(cfg=cfg),
            ai_statistics_plugin(cfg=cfg),
            model_router_plugin(cfg=cfg),
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
            await ensure_ingress_resources(cfg=cfg, api_client=api_client)
            for plugin_name, plugin_spec in plugin_list:
                await ensure_wasm_plugin(
                    api=gw_client.ExtensionsHigressIoV1Api(api_client),
                    name=plugin_name,
                    namespace=cfg.get_gateway_namespace(),
                    expected=plugin_spec,
                )

        try:
            asyncio.run(prepare())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError("Failed to initialize gateway resources") from e
