import time
import asyncio
from typing import Any, Dict
from kubernetes_asyncio import client as k8s_client
from kubernetes_asyncio.client import Configuration
from kubernetes_asyncio.config.kube_config import KubeConfigLoader, KubeConfigMerger
from kubernetes_asyncio.config.incluster_config import (
    InClusterConfigLoader,
    SERVICE_TOKEN_FILENAME,
    SERVICE_CERT_FILENAME,
)
from kubernetes_asyncio.client.rest import ApiException
from gpustack.config.config import Config, GatewayModeEnum
from gpustack.gateway import client as gw_client
from gpustack.gateway.client import (
    McpBridge,
    McpBridgeSpec,
    McpBridgeRegistry,
    WasmPlugin,
    WasmPluginSpec,
)
from gpustack.gateway.labels_annotations import managed_labels, match_labels
from gpustack.gateway.utils import (
    default_mcp_bridge_name,
    openai_model_prefixes,
    mcp_ingress_equal,
    get_default_mcpbridge_ref,
)
from gpustack.gateway.plugins import (
    get_plugin_url_with_name_and_version,
    get_plugin_url_prefix,
)

# plugin_prefix is updated by get_plugin_url_prefix in initialize_gateway
plugin_prefix = ""
mcp_registry_name = "gpustack"
mcp_registry_port = 80

supported_openai_routes = [route for v in openai_model_prefixes.values() for route in v]


def wait_for_apiserver_ready(cfg: Config, timeout: int = 60, interval: int = 5):
    async def get_api_resources():
        config = cfg.get_async_k8s_config()
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
    registry_type = "static"
    if cfg.gateway_mode == GatewayModeEnum.external and cfg.gateway_kubeconfig is None:
        registry_type = "dns"
    address = (
        cfg.get_advertise_address()
        if cfg.gateway_mode != GatewayModeEnum.embedded
        else "127.0.0.1"
    )
    port = (
        cfg.worker_port
        if cfg.server_role() == Config.ServerRole.WORKER
        else cfg.api_port
    )
    domain = (
        f"{address}:{port}"
        if registry_type == "static"
        else f"{cfg.service_discovery_name}.{cfg.get_gateway_namespace()}.svc"
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
                    )
                )
            ],
        ),
    )
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


async def ensure_ext_auth(cfg: Config, api_client: k8s_client.ApiClient):
    resource_name = "gpustack-llm-ext-auth"
    api = gw_client.ExtensionsHigressIoV1Api(api_client=api_client)
    gateway_namespace = cfg.get_gateway_namespace()
    try:
        data: Dict[str, Any] = await api.get_wasmplugin(
            namespace=gateway_namespace, name=resource_name
        )
        ext_auth = WasmPlugin.model_validate(data)
    except ApiException as e:
        if e.status == 404:
            ext_auth = None
        else:
            raise
    registry = get_gpustack_higress_registry(cfg=cfg)
    expected_spec = WasmPluginSpec(
        defaultConfig={
            "http_service": {
                "authorization_request": {
                    "allowed_headers": [{"exact": "x-higress-llm-model"}]
                },
                "authorization_response": {
                    "allowed_upstream_headers": [
                        {"exact": "X-Mse-Consumer"},
                        {"exact": "Authentication"},
                    ]
                },
                "endpoint": {
                    "path": "/token-auth",
                    "request_method": "POST",
                    "service_name": f"{registry.name}.{registry.type}",
                    "service_port": registry.port,
                },
                "endpoint_mode": "forward_auth",
                "timeout": 1000,
            },
            "match_list": [
                {"match_rule_path": route, "match_rule_type": "exact"}
                for route in supported_openai_routes
            ],
            "match_type": "blacklist",
        },
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        phase="AUTHN",
        priority=360,
        url=get_plugin_url_with_name_and_version(
            name="ext-auth", version="2.0.0", prefix=plugin_prefix
        ),
    )
    if ext_auth is None:
        ext_auth = WasmPlugin(
            metadata={
                "name": resource_name,
                "namespace": gateway_namespace,
                "labels": managed_labels,
            },
            spec=expected_spec,
        )
        await api.create_wasmplugin(namespace=gateway_namespace, body=ext_auth)
    elif match_labels(getattr(ext_auth.metadata, 'labels', {}), managed_labels):
        should_update = False
        spec = ext_auth.spec
        if not spec:
            ext_auth.spec = expected_spec
            should_update = True
        elif spec.defaultConfig is None:
            ext_auth.spec.defaultConfig = expected_spec.defaultConfig
            should_update = True
        else:
            # only compare the endpoint related fields for update
            http_service: Dict[str, Any] = ext_auth.spec.defaultConfig.get(
                "http_service", {}
            )
            endpoint: Dict[str, Any] = http_service.get("endpoint", {})
            if (
                endpoint.get("path")
                != expected_spec.defaultConfig["http_service"]["endpoint"]["path"]
                or endpoint.get("request_method")
                != expected_spec.defaultConfig["http_service"]["endpoint"][
                    "request_method"
                ]
                or endpoint.get("service_name")
                != expected_spec.defaultConfig["http_service"]["endpoint"][
                    "service_name"
                ]
                or endpoint.get("service_port")
                != expected_spec.defaultConfig["http_service"]["endpoint"][
                    "service_port"
                ]
            ):
                http_service["endpoint"] = expected_spec.defaultConfig["http_service"][
                    "endpoint"
                ]
                ext_auth.spec.defaultConfig["http_service"] = http_service
                should_update = True
        if should_update:
            ext_auth.spec = spec
            await api.edit_wasmplugin(
                namespace=gateway_namespace, name=resource_name, body=ext_auth
            )


async def ensure_ai_statistics(cfg: Config, api_client: k8s_client.ApiClient):
    resource_name = "gpustack-ai-statistics"
    api = gw_client.ExtensionsHigressIoV1Api(api_client=api_client)
    gateway_namespace = cfg.get_gateway_namespace()
    try:
        data: Dict[str, Any] = await api.get_wasmplugin(
            namespace=gateway_namespace,
            name=resource_name,
        )
        ai_stats = WasmPlugin.model_validate(data)
    except ApiException as e:
        if e.status == 404:
            ai_stats = None
        else:
            raise
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
            name="ai-statistics", version="2.0.0", prefix=plugin_prefix
        ),
    )
    if ai_stats is None:
        ai_stats = WasmPlugin(
            metadata={
                "name": resource_name,
                "namespace": gateway_namespace,
                "labels": managed_labels,
            },
            spec=expected_spec,
        )
        await api.create_wasmplugin(namespace=gateway_namespace, body=ai_stats)
    # no dynamic data, skip updating for now


async def ensure_model_router(cfg: Config, api_client: k8s_client.ApiClient):
    resource_name = "gpustack-model-router"
    gateway_namespace = cfg.get_gateway_namespace()
    api = gw_client.ExtensionsHigressIoV1Api(api_client=api_client)
    try:
        data: Dict[str, Any] = await api.get_wasmplugin(
            namespace=gateway_namespace, name=resource_name
        )
        model_router = WasmPlugin.model_validate(data)
    except ApiException as e:
        if e.status == 404:
            model_router = None
        else:
            raise
    expected_spec = WasmPluginSpec(
        defaultConfig={'modelToHeader': 'x-higress-llm-model'},
        defaultConfigDisable=False,
        failStrategy="FAIL_OPEN",
        imagePullPolicy="UNSPECIFIED_POLICY",
        matchRules=[],
        phase="AUTHN",
        priority=900,
        url=get_plugin_url_with_name_and_version(
            name="model-router", version="2.0.0", prefix=plugin_prefix
        ),
    )
    if model_router is None:
        model_router = WasmPlugin(
            metadata={
                "name": resource_name,
                "namespace": gateway_namespace,
                "labels": managed_labels,
            },
            spec=expected_spec,
        )
        await api.create_wasmplugin(namespace=gateway_namespace, body=model_router)
    # no dynamic data, skip updating for now


def set_async_k8s_config(cfg: Config):
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
    cfg.set_async_k8s_config(configuration)


def initialize_gateway(cfg: Config, timeout: int = 60, interval: int = 5):
    if cfg.gateway_mode == GatewayModeEnum.disabled:
        return
    global plugin_prefix
    plugin_prefix = get_plugin_url_prefix(cfg=cfg)
    set_async_k8s_config(cfg=cfg)
    wait_for_apiserver_ready(cfg=cfg, timeout=timeout, interval=interval)
    if cfg.gateway_mode in [
        GatewayModeEnum.embedded,
        GatewayModeEnum.external,
    ]:

        async def prepare():
            api_client = k8s_client.ApiClient(configuration=cfg.get_async_k8s_config())
            await ensure_mcp_resources(cfg=cfg, api_client=api_client)
            await ensure_ingress_resources(cfg=cfg, api_client=api_client)
            await ensure_ext_auth(cfg=cfg, api_client=api_client)
            await ensure_ai_statistics(cfg=cfg, api_client=api_client)
            await ensure_model_router(cfg=cfg, api_client=api_client)

        try:
            asyncio.run(prepare())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError("Failed to initialize gateway resources") from e
