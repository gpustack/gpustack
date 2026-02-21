import logging
import copy
import math
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any, Literal, Callable
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.gateway.labels_annotations import managed_labels, match_labels
from gpustack.gateway import ai_proxy_types
from gpustack.gateway.client.networking_higress_io_v1_api import (
    NetworkingHigressIoV1Api,
    McpBridge,
    McpBridgeRegistry,
    McpBridgeSpec,
    McpBridgeProxy,
)
from gpustack.gateway.client.extensions_higress_io_v1_api import (
    WasmPlugin,
    WasmPluginSpec,
    ExtensionsHigressIoV1Api,
    WasmPluginMatchRule,
)
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import (
    NetworkingIstioIoV1Alpha3Api,
    EnvoyFilter,
    get_ingress_fallback_envoyfilter,
)
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstancePublic,
)
from gpustack.schemas.model_provider import (
    ModelProvider,
    ModelProviderTypeEnum,
)
from gpustack.schemas.model_routes import ModelRoute
from gpustack.server.bus import EventType
from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster
from gpustack.utils.network import is_ipaddress
from kubernetes_asyncio import client as k8s_client
from kubernetes_asyncio.client import ApiException, V1IngressTLS
from gpustack.envs import GATEWAY_MIRROR_INGRESS_NAME

logger = logging.getLogger(__name__)

default_mcp_bridge_name = "default"
gpustack_ai_proxy_name = "gpustack-ai-proxy"
gpustack_model_mapper_name = "gpustack-model-mapper"
model_ingress_prefix = "ai-route-model-"
model_route_ingress_prefix = "ai-route-route-"
provider_id_prefix = "provider-"
model_id_prefix = "model-"

# Type alias for destination tuples
# Each tuple contains (weight: int, model_name: str, registry: McpBridgeRegistry)
DestinationTupleList = List[Tuple[int, str, McpBridgeRegistry]]


@dataclass
class RoutePrefix:
    prefixes: List[str]
    support_legacy: bool = True

    def flattened_prefixes(self) -> List[str]:
        versioned_prefixes = ["/v1"]
        if self.support_legacy:
            versioned_prefixes.append("/v1-openai")
        flattened = []
        for versioned_prefix in versioned_prefixes:
            for prefix in self.prefixes:
                flattened.append(f"{versioned_prefix}{prefix}")
        return flattened

    def regex_prefixes(self) -> List[str]:
        """
        Returns regex patterns for the prefixes, considering versioning and legacy support.
        It supports removing -openai suffix from the versioned prefix with rewrite-target: /$1$3
        """
        versioned_prefix = f"/(v1){'(-openai)?' if self.support_legacy else '()'}"
        return [f"{versioned_prefix}({prefix})" for prefix in self.prefixes]


openai_model_prefixes: List[RoutePrefix] = [
    RoutePrefix(
        [
            "/chat/completions",
            "/completions",
            "/embeddings",
            "/audio/transcriptions",
            "/audio/speech",
            "/images/generations",
            "/images/edits",
        ]
    ),
    RoutePrefix(["/rerank"], False),
]


def get_default_mcpbridge_ref(
    mcp_bridge_name: str = default_mcp_bridge_name,
) -> k8s_client.V1TypedLocalObjectReference:
    # the name is hardcoded in Higress MCP Bridge controller
    return k8s_client.V1TypedLocalObjectReference(
        api_group='networking.higress.io',
        kind='McpBridge',
        name=mcp_bridge_name,
    )


def wrap_route(
    path: str,
    path_type: str,
    backend: Optional[k8s_client.V1IngressBackend] = None,
) -> k8s_client.V1HTTPIngressPath:
    if backend is None:
        backend = k8s_client.V1IngressBackend(
            resource=get_default_mcpbridge_ref(),
        )
    return k8s_client.V1HTTPIngressPath(
        path=path,
        path_type=path_type,
        backend=backend,
    )


def anthropic_routes() -> List[k8s_client.V1HTTPIngressPath]:
    return [
        wrap_route("/v1/messages", "Prefix"),
        wrap_route("/v1/complete", "Exact"),
    ]


def ingress_rule_for_model() -> k8s_client.V1IngressRule:
    paths: List[k8s_client.V1HTTPIngressPath] = []
    for route_prefix in openai_model_prefixes:
        for prefix in route_prefix.regex_prefixes():
            paths.append(wrap_route(path=prefix, path_type="ImplementationSpecific"))
    return k8s_client.V1IngressRule(http=k8s_client.V1HTTPIngressRuleValue(paths=paths))


def cluster_mcp_bridge_name(cluster_id: int) -> str:
    # higress_controller has hardcoded mcp bridge name to 'default'
    # the name should be based on cluster_id if higress_controller supports multiple mcp bridges
    return default_mcp_bridge_name


def model_mcp_bridge_name(cluster_id: int) -> str:
    return cluster_mcp_bridge_name(cluster_id)


def model_route_cleanup_prefix(model_route_id: int) -> str:
    return f"{model_route_ingress_prefix}{model_route_id}"


def model_route_ingress_name(model_route_id: int) -> str:
    return f"{model_route_ingress_prefix}{model_route_id}.internal"


def fallback_ingress_name(name: str) -> str:
    split_name = name.rsplit('.', 1)
    if len(split_name) == 1:
        return f"{name}.fallback"
    return f"{split_name[0]}.fallback.{split_name[1]}"


def model_ingress_name(model_id: int) -> str:
    return f"{model_ingress_prefix}{model_id}"


def cluster_worker_prefix(cluster_id: int) -> str:
    return f"cluster-{cluster_id}-worker-"


def model_prefix(model_id: int) -> str:
    return f"{model_id_prefix}{model_id}-"


def model_instance_prefix(
    model_instance: Union[ModelInstance, ModelInstancePublic]
) -> str:
    return f"{model_prefix(model_instance.model_id)}{model_instance.id}"


def model_instance_registry(
    model_instance: Union[ModelInstance, ModelInstancePublic],
    worker: Optional[Worker] = None,
) -> Optional[McpBridgeRegistry]:
    name = model_instance_prefix(model_instance)
    if worker is not None and worker.proxy_mode == ModelInstanceProxyModeEnum.WORKER:
        return worker_registry(worker, name_override=name)
    address = model_instance.worker_advertise_address or model_instance.worker_ip
    if address is None or address == "" or model_instance.port is None:
        return None
    domain = address
    port = model_instance.port
    registry_type = "dns"
    if is_ipaddress(address):
        domain = f"{address}:{model_instance.port}"
        port = 80
        registry_type = "static"
    return McpBridgeRegistry(
        domain=domain,
        port=port,
        name=name,
        protocol="http",
        type=registry_type,
    )


def worker_registry(
    worker: Worker, name_override: Optional[str] = None
) -> Optional[McpBridgeRegistry]:
    address = worker.advertise_address or worker.ip
    if (
        address is None
        or address == ""
        or worker.port is None
        or worker.proxy_mode != ModelInstanceProxyModeEnum.WORKER
    ):
        return None
    domain = address
    port = worker.port
    registry_type = "dns"
    if is_ipaddress(address):
        domain = f"{address}:{worker.port}"
        port = 80
        registry_type = "static"
    return McpBridgeRegistry(
        domain=domain,
        port=port,
        name=name_override or f"{cluster_worker_prefix(worker.cluster_id)}{worker.id}",
        protocol="http",
        type=registry_type,
    )


def cluster_registry(cluster: Cluster) -> Optional[McpBridgeRegistry]:
    if cluster.gateway_endpoint is None and cluster.reported_gateway_endpoint is None:
        return None
    return McpBridgeRegistry(
        domain=cluster.gateway_endpoint or cluster.reported_gateway_endpoint,
        port=80,
        name="cluster-gateway",
        protocol="http",
        type="static",
    )


def provider_registry_name(id: int) -> str:
    return f"{provider_id_prefix}{id}"


def provider_registry(provider: ModelProvider) -> Optional[McpBridgeRegistry]:
    domain = provider.config.get_service_registry()
    if domain is None:
        return None
    provider_url = provider.config.get_base_url()
    result = urlparse(url=provider_url)
    protocol = "https"
    port = 443
    if result.scheme == "http":
        protocol = "http"
        port = 80
    registry_type = "dns"
    parsed_domain = urlparse(f"//{domain}")
    if is_ipaddress(parsed_domain.hostname):
        registry_type = "static"
        port = 80
    elif result.port is not None:
        port = result.port
    registry_name = provider_registry_name(provider.id)
    proxyName = f"{registry_name}-proxy" if provider.proxy_url else None
    return McpBridgeRegistry(
        domain=domain,
        port=port,
        name=registry_name,
        protocol=protocol,
        type=registry_type,
        proxyName=proxyName,
    )


def provider_proxy(provider: ModelProvider) -> Optional[McpBridgeProxy]:
    if provider.proxy_url is None:
        return None
    proxy_url = urlparse(provider.proxy_url)
    scheme = proxy_url.scheme
    port = proxy_url.port
    if port is None:
        port = 443 if scheme == "https" else 80
    # timeout in seconds
    connection_timeout = provider.proxy_timeout or 5
    return McpBridgeProxy(
        name=f"{provider_registry_name(provider.id)}-proxy",
        serverAddress=proxy_url.hostname,
        serverPort=port,
        type=scheme.upper(),
        # convert to milliseconds
        connectTimeout=connection_timeout * 1000,
    )


def provider_proxy_plugin_spec(
    *providers: ModelProvider,
) -> Tuple[List[Dict[str, Any]], List[WasmPluginMatchRule]]:
    provider_list = []
    match_rules = []
    sorted_providers: List[ModelProvider] = sorted(providers, key=lambda p: p.id)
    for provider in sorted_providers:
        registry = provider_registry(provider)
        if registry is None:
            continue
        service_name = registry.get_service_name()
        default_config_data = {
            "id": provider_registry_name(provider.id),
            "apiTokens": provider.api_tokens,
            **provider.config.model_dump_with_default_override(),
            "type": provider.config.type.value,
        }
        accessible_llm_model = next(
            (model.name for model in provider.models or [] if model.category == "llm"),
            None,
        )
        # Failover has more config
        if accessible_llm_model and len(provider.api_tokens) > 1:
            default_config_data["failover"] = ai_proxy_types.FailoverConfig(
                enabled=True,
                healthCheckModel=accessible_llm_model,
            )
        default_config = ai_proxy_types.AIProxyDefaultConfig.model_validate(
            default_config_data
        )
        provider_list.append(
            default_config.model_dump(by_alias=True, exclude_none=True)
        )
        active_config = ai_proxy_types.ActiveConfig(
            activeProviderId=provider_registry_name(provider.id),
        ).model_dump(exclude_none=True)
        match_rules.append(
            WasmPluginMatchRule(
                config=active_config,
                service=[service_name],
                configDisable=False,
            )
        )
    return provider_list, match_rules


def diff_registries(
    existing: List[McpBridgeRegistry],
    desired: List[McpBridgeRegistry],
    to_delete_prefix: Optional[str] = None,
) -> Tuple[bool, List[McpBridgeRegistry]]:
    desired_map = {
        reg.name: idx for idx, reg in enumerate(desired) if reg.name is not None
    }
    total_list = []
    need_update = False
    for registry in existing:
        if registry.name not in desired_map:
            # delete registries that are not in the current list
            if to_delete_prefix is not None and registry.name.startswith(
                to_delete_prefix
            ):
                need_update = True
            else:
                # keep unrelated registries
                total_list.append(registry)
        else:
            # update existing registries
            idx = desired_map.pop(registry.name)
            if registry != desired[idx]:
                need_update = True
                registry = desired[idx]
            total_list.append(registry)
    # add new registries
    for idx in desired_map.values():
        need_update = True
        total_list.append(desired[idx])

    total_list.sort(key=lambda r: r.name or "")
    return need_update, total_list


def diff_proxies(
    existing: List[McpBridgeProxy],
    desired: List[McpBridgeProxy],
    to_delete_prefix: Optional[str] = None,
) -> Tuple[bool, List[McpBridgeProxy]]:
    desired_map = {
        reg.name: idx for idx, reg in enumerate(desired) if reg.name is not None
    }
    total_list = []
    need_update = False
    for proxy in existing:
        if proxy.name not in desired_map:
            # delete registries that are not in the current list
            if to_delete_prefix is not None and proxy.name.startswith(to_delete_prefix):
                need_update = True
            else:
                # keep unrelated proxies
                total_list.append(proxy)
        else:
            # update existing proxies
            idx = desired_map.pop(proxy.name)
            if proxy != desired[idx]:
                need_update = True
                proxy = desired[idx]
            total_list.append(proxy)
    # add new proxies
    for idx in desired_map.values():
        need_update = True
        total_list.append(desired[idx])

    total_list.sort(key=lambda r: r.name or "")
    return need_update, total_list


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def ensure_mcp_bridge(
    client: NetworkingHigressIoV1Api,
    namespace: str,
    mcp_bridge_name: str,
    desired_registries: List[McpBridgeRegistry],
    to_delete_prefix: Optional[str] = None,
    desired_proxies: List[McpBridgeProxy] = None,
    to_delete_proxies_prefix: Optional[str] = None,
):
    existing_bridge = None
    try:
        mcpbridge_dict = await client.get_mcpbridge(namespace, mcp_bridge_name)
        existing_bridge = McpBridge.model_validate(mcpbridge_dict)
    except ApiException as e:
        if e.status != 404:
            raise
    if existing_bridge is None:
        mcpbridge_body = McpBridge(
            metadata={
                "name": mcp_bridge_name,
                "namespace": namespace,
                "labels": managed_labels,
            },
            spec=McpBridgeSpec(registries=desired_registries, proxies=desired_proxies),
        )
        await client.create_mcpbridge(
            namespace=namespace,
            body=mcpbridge_body,
        )
        logger.info(f"Created MCP Bridge {mcp_bridge_name} in namespace {namespace}.")
    else:
        registry_need_update, registry_list = diff_registries(
            existing=existing_bridge.spec.registries or [],
            desired=desired_registries,
            to_delete_prefix=to_delete_prefix,
        )
        proxy_need_update = False
        proxy_list = existing_bridge.spec.proxies or []
        if desired_proxies is not None:
            proxy_need_update, proxy_list = diff_proxies(
                existing=existing_bridge.spec.proxies or [],
                desired=desired_proxies,
                to_delete_prefix=to_delete_proxies_prefix,
            )

        if registry_need_update or proxy_need_update:
            registry_list.sort(key=lambda r: r.name or "")
            proxy_list.sort(key=lambda r: r.name or "")
            existing_bridge.spec.registries = registry_list
            existing_bridge.spec.proxies = proxy_list
            await client.edit_mcpbridge(
                name=mcp_bridge_name,
                namespace=namespace,
                body=existing_bridge,
            )
            logger.info(
                f"Updated MCP Bridge {mcp_bridge_name} in namespace {namespace}."
            )


def generate_model_ingress(
    ingress_name: str,
    namespace: str,
    route_name: str,
    destinations: str,
    hostname: Optional[str] = None,
    tls: Optional[List[V1IngressTLS]] = None,
    included_generic_route: Optional[bool] = False,
    included_proxy_route: Optional[bool] = False,
    extra_annotations: Optional[Dict[str, str]] = None,
) -> k8s_client.V1Ingress:
    annotations = {
        "higress.io/rewrite-target": "/$1$3",
        "higress.io/destination": destinations,
        "higress.io/ignore-path-case": 'true',
        **higress_http_header_matcher("exact", "x-higress-llm-model", route_name),
    }
    if extra_annotations is not None:
        annotations.update(extra_annotations)
    metadata = k8s_client.V1ObjectMeta(
        name=ingress_name,
        namespace=namespace,
        annotations=annotations,
        labels=managed_labels,
    )
    expected_rule = ingress_rule_for_model()

    if included_proxy_route:
        # to compatible with rewrite-target /$1$3, the first capturing group is empty
        expected_rule.http.paths.append(
            wrap_route(
                "/()model/proxy(/|$)(.*)",
                "ImplementationSpecific",
            )
        )
    if included_generic_route:
        expected_rule.http.paths.append(wrap_route("/", "Prefix"))
    # support for Anthropic API
    expected_rule.http.paths.extend(anthropic_routes())
    spec = k8s_client.V1IngressSpec(ingress_class_name="higress", rules=[expected_rule])
    if hostname is not None:
        hostname_rule = copy.deepcopy(expected_rule)
        hostname_rule.host = hostname
        spec.rules.append(hostname_rule)
    spec.tls = tls
    ingress = k8s_client.V1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=metadata,
        spec=spec,
    )
    return ingress


def higress_metadata_equal(
    existing_metadata: Optional[k8s_client.V1ObjectMeta],
    expected_metadata: Optional[k8s_client.V1ObjectMeta],
) -> bool:
    existing_metadata = existing_metadata or k8s_client.V1ObjectMeta()
    expected_metadata = expected_metadata or k8s_client.V1ObjectMeta()
    if existing_metadata.annotations is None:
        existing_metadata.annotations = {}
    if expected_metadata.annotations is None:
        expected_metadata.annotations = {}
    for key in set(
        k for k in expected_metadata.annotations if k.startswith("higress.io")
    ):
        if existing_metadata.annotations.get(key) != expected_metadata.annotations.get(
            key
        ):
            return False
    return True


def ingress_tls_equal(
    existing: Optional[k8s_client.V1IngressTLS],
    expected: Optional[k8s_client.V1IngressTLS],
) -> bool:
    if (existing is None) != (expected is None):
        return False
    if existing and expected:
        if len(existing) != len(expected):
            return False
        for etls, xtls in zip(existing, expected):
            # only compares hosts and secret_name for tls equal
            if getattr(etls, 'hosts', None) != getattr(xtls, 'hosts', None):
                return False
            if getattr(etls, 'secret_name', None) != getattr(xtls, 'secret_name', None):
                return False
    return True


def mcp_ingress_equal(
    existing: k8s_client.V1Ingress, expected: k8s_client.V1Ingress
) -> bool:
    if not higress_metadata_equal(
        existing_metadata=existing.metadata, expected_metadata=expected.metadata
    ):
        return False
    if existing.spec is None or expected.spec is None:
        return False
    if not ingress_tls_equal(
        existing=getattr(existing.spec, 'tls', None),
        expected=getattr(expected.spec, 'tls', None),
    ):
        return False
    if len(existing.spec.rules or []) != len(expected.spec.rules or []):
        return False

    for existing_rule, expected_rule in zip(
        existing.spec.rules or [], expected.spec.rules or []
    ):
        if getattr(existing_rule, 'host', None) != getattr(expected_rule, 'host', None):
            return False
        if existing_rule.http is None or expected_rule.http is None:
            return False
        if len(existing_rule.http.paths or []) != len(expected_rule.http.paths or []):
            return False
        for existing_path, expected_path in zip(
            existing_rule.http.paths or [], expected_rule.http.paths or []
        ):
            if existing_path.path != expected_path.path:
                return False
            if existing_path.path_type != expected_path.path_type:
                return False
            if existing_path.backend.resource != expected_path.backend.resource:
                return False
    return True


def scale_weight(weight_instance_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Scale weights based on the least common multiple of counts to maintain proportionality.
    """
    counts = [count for _, count in weight_instance_pairs if count > 0]
    if not counts:
        return weight_instance_pairs
    lcm_count = math.lcm(*counts)
    scaled = [
        (weight * lcm_count // count if count > 0 else 0, count)
        for weight, count in weight_instance_pairs
    ]
    return scaled


def hamilton_calculate_weight(
    weight_instance_pairs: List[Tuple[int, int]],
    max_weight: Optional[int] = 0,
) -> List[int]:
    """
    hamilton_calculate_weight to allocate percentage based on weight and instance count.
    The total should be 100.

    :param weight_instance_pairs: weight and instance count pairs
    :type weight_instance_pairs: List[Tuple[int, int]]
    :return: list of percentage for instance
    :rtype: List[int]
    """
    weight_instance_pairs = scale_weight(weight_instance_pairs)
    instances_info = []
    for weight, instance_count in weight_instance_pairs:
        for _ in range(instance_count):
            instances_info.append({'weight': weight, 'group_weight': weight})
    total_weight = sum(max(info['weight'], max_weight) for info in instances_info)
    if total_weight == 0:
        return []
    for info in instances_info:
        weight = max(info['weight'], max_weight)
        info['exact_quota'] = weight * 100 / total_weight
        info['floor_quota'] = int(info['exact_quota'])
        info['remainder'] = info['exact_quota'] - info['floor_quota']

    total_floor = sum(info['floor_quota'] for info in instances_info)
    remaining_seats = 100 - total_floor
    sorted_instances = sorted(instances_info, key=lambda x: -x['remainder'])
    for i in range(remaining_seats):
        sorted_instances[i]['floor_quota'] += 1
    return [info['floor_quota'] for info in instances_info]


def model_instances_registry_list(
    model_instances: List[Union[ModelInstance, ModelInstancePublic]],
    workers: Optional[Dict[int, Worker]] = None,
) -> DestinationTupleList:
    registries: DestinationTupleList = []
    for model_instance in model_instances:
        worker = (
            (workers or {}).get(model_instance.worker_id)
            if model_instance.worker_id
            else None
        )
        registry = model_instance_registry(model_instance, worker=worker)
        if registry is not None:
            registries.append((1, model_instance.model_name, registry))
    return registries


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def ensure_model_ingress(
    ingress_name: str,
    route_name: str,
    namespace: str,
    destinations: DestinationTupleList,
    event_type: EventType,
    networking_api: k8s_client.NetworkingV1Api,
    included_generic_route: Optional[bool] = False,
    included_proxy_route: Optional[bool] = False,
    extra_annotations: Optional[Dict[str, str]] = None,
):
    """
    Ensure the model ingress resource in Kubernetes matches the desired state.

    Parameters:
        ingress_name (str): The name of the ingress resource.
        namespace (str): The Kubernetes namespace for the ingress resource.
        destinations (DestinationTupleList): Weighted list of MCP Bridge registries for traffic routing.
        route_name (str): The name of the model route for which ingress is managed.
        event_type (EventType): The event type (CREATED, UPDATED, DELETED) triggering reconciliation.
        networking_api (k8s_client.NetworkingV1Api): The Kubernetes networking API client.
        hostname (Optional[str]): The external hostname for ingress routing.
        tls_secret_name (Optional[str]): TLS secret name for HTTPS ingress.
        included_generic_route (bool): Whether to include a generic '/' route for fallback traffic. Used in worker gateway.
        included_proxy_route (bool): Whether to include a proxy route for model traffic (e.g., /model/proxy/{model_name}). Used in server gateway.
    """
    if event_type == EventType.DELETED:
        try:
            await networking_api.delete_namespaced_ingress(
                name=ingress_name, namespace=namespace
            )
            logger.info(
                f"Deleted model ingress {ingress_name} for model route {route_name}"
            )
        except ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete ingress {ingress_name}: {e}")
        return

    expected_destinations = '\n'.join(
        [
            f"{persentage}% {registry.get_service_name_with_port()}"
            for persentage, _, registry in destinations
        ]
    )
    try:
        existing_ingress: Optional[k8s_client.V1Ingress] = (
            await networking_api.read_namespaced_ingress(
                name=ingress_name, namespace=namespace
            )
        )
    except ApiException as e:
        if e.status != 404:
            logger.error(f"Failed to get ingress {ingress_name}: {e}")
            return
        existing_ingress = None
    hostname, tls = await mirror_hostname_tls_from_ingress(
        network_v1_client=networking_api,
        gateway_namespace=namespace,
        target_ingress_name=GATEWAY_MIRROR_INGRESS_NAME,
    )
    expected_ingress = generate_model_ingress(
        ingress_name=ingress_name,
        route_name=route_name,
        namespace=namespace,
        destinations=expected_destinations,
        hostname=hostname,
        tls=tls,
        included_generic_route=included_generic_route,
        included_proxy_route=included_proxy_route,
        extra_annotations=extra_annotations,
    )

    if existing_ingress is None:
        await networking_api.create_namespaced_ingress(
            namespace=namespace,
            body=expected_ingress,
        )
        logger.info(
            f"Created model ingress {ingress_name} for model route {route_name}"
        )
    else:
        is_equal = mcp_ingress_equal(
            existing=existing_ingress, expected=expected_ingress
        )
        if not is_equal:
            existing_ingress.spec = expected_ingress.spec
            metadata = existing_ingress.metadata or k8s_client.V1ObjectMeta()
            metadata.annotations = metadata.annotations or {}
            expected_higress_keys = set()
            for key, value in (expected_ingress.metadata.annotations or {}).items():
                if key.startswith("higress.io"):
                    metadata.annotations[key] = value
                    expected_higress_keys.add(key)
            to_delete = [
                key
                for key in metadata.annotations.keys()
                if key.startswith("higress.io") and key not in expected_higress_keys
            ]
            for key in to_delete:
                del metadata.annotations[key]

            await networking_api.replace_namespaced_ingress(
                name=ingress_name,
                namespace=namespace,
                body=existing_ingress,
            )
            logger.info(
                f"Updated model ingress {ingress_name} for model route {route_name}"
            )


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def ensure_wasm_plugin(
    api: ExtensionsHigressIoV1Api,
    name: str,
    namespace: str,
    spec_diff: Callable[[Optional[WasmPluginSpec]], WasmPluginSpec],
    extra_labels: Optional[Dict[str, str]] = None,
):
    labels = copy.deepcopy(managed_labels)
    if extra_labels:
        labels.update(extra_labels)
    current_plugin = None
    try:
        data: Dict[str, Any] = await api.get_wasmplugin(namespace=namespace, name=name)
        current_plugin = WasmPlugin.model_validate(data)
    except ApiException as e:
        if e.status == 404:
            current_plugin = None
        else:
            raise
    current_spec = getattr(current_plugin, 'spec', None)
    expected = spec_diff(copy.deepcopy(current_spec))
    if current_plugin is None:
        wasm_plugin_body = WasmPlugin(
            metadata={
                "name": name,
                "namespace": namespace,
                "labels": labels,
            },
            spec=expected,
        )
        await api.create_wasmplugin(
            namespace=namespace,
            body=wasm_plugin_body,
        )
        logger.info(f"Created WasmPlugin {name} in namespace {namespace}.")
    elif match_labels(current_plugin.metadata.get("labels", {}), labels):
        current_spec = (
            current_plugin.spec.model_dump(exclude_none=True)
            if current_plugin.spec
            else {}
        )
        expected_spec = expected.model_dump(exclude_none=True) if expected else {}
        if current_spec != expected_spec:
            current_plugin.spec = expected
            await api.edit_wasmplugin(
                namespace=namespace,
                name=name,
                body=current_plugin,
            )
            logger.info(f"Updated WasmPlugin {name} in namespace {namespace}.")


async def cleanup_model_mapper(
    namespace: str,
    expected_ingresses: List[str],
    config: k8s_client.Configuration,
    extra_labels: Optional[Dict[str, str]] = None,
):
    api = ExtensionsHigressIoV1Api(k8s_client.ApiClient(config))
    labels = copy.deepcopy(managed_labels)
    if extra_labels:
        labels.update(extra_labels)

    def spec_diff(current_spec: Optional[WasmPluginSpec]) -> WasmPluginSpec:
        if current_spec is None:
            return current_spec
        to_keep_rules: List[WasmPluginMatchRule] = []
        for rule in current_spec.matchRules or []:
            if any(ingress in expected_ingresses for ingress in rule.ingress):
                to_keep_rules.append(rule)
            else:
                logger.info(
                    f"Removing rule with ingress {rule.ingress} from model mapper plugin as it is not in expected ingresses."
                )
        to_keep_rules.sort(key=lambda r: r.ingress[0] if r.ingress else "")
        current_spec.matchRules = to_keep_rules
        return current_spec

    await ensure_wasm_plugin(
        api=api,
        name=gpustack_model_mapper_name,
        namespace=namespace,
        spec_diff=spec_diff,
        extra_labels=extra_labels,
    )


async def cleanup_ingresses(
    namespace: str,
    expected_names: List[str],
    config: k8s_client.Configuration,
    cleanup_prefix: str,
    reason: str = "orphaned",
):
    networking_api = k8s_client.NetworkingV1Api(k8s_client.ApiClient(config))
    try:
        # Use label selector to filter only managed ingresses
        label_selector = ','.join([f"{k}={v}" for k, v in managed_labels.items()])
        ingresses = await networking_api.list_namespaced_ingress(
            namespace=namespace,
            label_selector=label_selector,
        )
        for ingress in ingresses.items:
            # name must be not None due to label selector
            name: str = ingress.metadata.name
            if name in expected_names or not name.startswith(cleanup_prefix):
                continue
            await networking_api.delete_namespaced_ingress(
                name=name, namespace=namespace
            )
            logger.info(
                f"Deleted {reason} model ingress {name} in namespace {namespace}."
            )
    except Exception as e:
        logger.error(f"Error cleaning up {reason} model ingresses: {e}")


async def ensure_model_mcp_bridge(
    event_type: EventType,
    model_id: int,
    model_instances: List[Union[ModelInstance, ModelInstancePublic]],
    networking_higress_api: NetworkingHigressIoV1Api,
    namespace: str,
    cluster_id: int,
    workers: Optional[Dict[int, Worker]] = None,
) -> List[McpBridgeRegistry]:
    desired_registry: List[McpBridgeRegistry] = []
    to_delete_prefix: Optional[str] = model_prefix(model_id)
    if event_type != EventType.DELETED:
        for model_instance in model_instances:
            worker = (
                (workers or {}).get(model_instance.worker_id)
                if model_instance.worker_id
                else None
            )
            registry = model_instance_registry(model_instance, worker=worker)
            if registry is not None:
                desired_registry.append(registry)
    await ensure_mcp_bridge(
        client=networking_higress_api,
        namespace=namespace,
        mcp_bridge_name=model_mcp_bridge_name(cluster_id),
        desired_registries=desired_registry,
        to_delete_prefix=to_delete_prefix,
    )
    return desired_registry


async def mirror_hostname_tls_from_ingress(
    network_v1_client: k8s_client.NetworkingV1Api,
    gateway_namespace: str,
    target_ingress_name: str,
) -> Tuple[Optional[str], Optional[List[V1IngressTLS]]]:
    """
    Mirror TLS settings from an existing ingress to be used in the gateway.

    Parameters:
        api_client (k8s_client.ApiClient): The Kubernetes API client.
        gateway_namespace (str): The namespace where the gateway ingress resides.
        target_ingress_name (str): The name of the ingress to mirror TLS settings from.

    Returns:
        Optional[Tuple[Optional[str], Optional[str]]]: A tuple containing the hostname and TLS secret name,
        or None if the target ingress does not exist or has no TLS settings.
    """
    try:
        ingress: k8s_client.V1Ingress = await network_v1_client.read_namespaced_ingress(
            name=target_ingress_name, namespace=gateway_namespace
        )
    except ApiException as e:
        if e.status == 404:
            logger.warning(
                f"Target ingress {target_ingress_name} not found in namespace {gateway_namespace} for TLS mirroring."
            )
            return None
        else:
            raise

    tls = getattr(ingress.spec, 'tls', None)
    hostname = None
    for rule in ingress.spec.rules or []:
        if rule.host:
            hostname = rule.host
            break
    return hostname, tls


def get_expected_match_list(
    route_name: str,
    ingress_prefix: str,
    ingress_name: str,
    model_name_to_registries: Dict[str, List[str]],
    fallback_model_name_to_registries: Dict[str, List[str]],
) -> List[WasmPluginMatchRule]:
    match_list: List[WasmPluginMatchRule] = []
    ingress_name = f"{ingress_prefix}{ingress_name}"
    for model_name, service_names in model_name_to_registries.items():
        config = {"modelMapping": {route_name: model_name}}
        match_list.append(
            WasmPluginMatchRule(
                config=config,
                ingress=[ingress_name],
                configDisable=False,
                service=service_names,
            )
        )
    for model_name, service_names in fallback_model_name_to_registries.items():
        # the fallback mapping should include both normal ingress and fallback ingress
        # as the normal ingress may not exist when only fallback model is set
        fallback_name = fallback_ingress_name(ingress_name)
        config = {"modelMapping": {route_name: model_name}}
        match_list.append(
            WasmPluginMatchRule(
                config=config,
                ingress=[ingress_name, fallback_name],
                configDisable=False,
                service=service_names,
            )
        )
    return match_list


def higress_http_header_matcher(
    operator: Literal["exact", "regex", "prefix"],
    header_key: str,
    header_value: str,
) -> Dict[str, str]:
    header_matcher = "match-header"
    return {
        f"higress.io/{operator}-{header_matcher}-{header_key}": header_value,
    }


async def cleanup_fallback_filters(
    namespace: str,
    expected_names: List[str],
    cleanup_prefix: str,
    reason: str = "orphaned",
    networking_istio_api: Optional[NetworkingIstioIoV1Alpha3Api] = None,
    k8s_config: Optional[k8s_client.Configuration] = None,
):
    if networking_istio_api is None:
        if k8s_config is None:
            raise ValueError(
                "Either networking_istio_api or k8s_config must be provided."
            )
        networking_istio_api = NetworkingIstioIoV1Alpha3Api(
            k8s_client.ApiClient(k8s_config)
        )
    try:
        label_selector = ','.join([f"{k}={v}" for k, v in managed_labels.items()])
        filters = await networking_istio_api.list_envoyfilters(
            namespace=namespace,
            label_selector=label_selector,
        )
        items: List[Dict[str, Any]] = filters.get('items', [])
        for filter_item in items:
            # name must be not None due to label selector
            name = filter_item.get("metadata", {}).get("name", None)
            if (
                name is None
                or name in expected_names
                or not name.startswith(cleanup_prefix)
            ):
                continue
            await networking_istio_api.delete_envoyfilter(
                name=name, namespace=namespace
            )
            logger.info(
                f"Deleted {reason} fallback filter {name} in namespace {namespace}."
            )
    except Exception as e:
        logger.error(f"Error cleaning up {reason} fallback filters: {e}")


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def ensure_fallback_filter(
    event_type: EventType,
    ingress_name: str,
    namespace: str,
    networking_istio_api: NetworkingIstioIoV1Alpha3Api,
):
    if event_type == EventType.DELETED:
        await cleanup_fallback_filters(
            namespace=namespace,
            expected_names=[],
            networking_istio_api=networking_istio_api,
            cleanup_prefix=ingress_name,
            reason="event deleted",
        )
        return
    existing_filter = None
    try:
        filter_dict = await networking_istio_api.get_envoyfilter(
            namespace=namespace, name=ingress_name
        )
        existing_filter = EnvoyFilter.model_validate(filter_dict)
    except ApiException as e:
        if e.status != 404:
            raise
    except Exception as e:
        raise e
    expected_filter = get_ingress_fallback_envoyfilter(
        ingress_name=ingress_name,
        namespace=namespace,
        labels={**managed_labels},
    )
    if existing_filter is None:
        await networking_istio_api.create_envoyfilter(
            namespace=namespace,
            body=expected_filter,
        )
        logger.info(
            f"Created fallback EnvoyFilter {ingress_name} in namespace {namespace}."
        )
    else:
        existing_spec_dict = existing_filter.spec.model_dump(exclude_none=True)
        expected_spec_dict = expected_filter.spec.model_dump(exclude_none=True)
        if existing_spec_dict != expected_spec_dict:
            existing_filter.spec = expected_filter.spec
            await networking_istio_api.edit_envoyfilter(
                name=ingress_name,
                namespace=namespace,
                body=existing_filter,
            )
            logger.info(
                f"Updated fallback EnvoyFilter {ingress_name} in namespace {namespace}."
            )


def ai_proxy_openai_provider_config(id: str) -> Dict[str, Any]:
    return ai_proxy_types.AIProxyDefaultConfig(
        type=ModelProviderTypeEnum.OPENAI,
        id=id,
        failover=ai_proxy_types.FailoverConfig(enabled=False),
        retryOnFailure=ai_proxy_types.EnableState(enabled=False),
    ).model_dump(exclude_none=True, exclude_unset=True)


def compare_and_append_default_proxy_config(
    existing_providers: List[Dict[str, Any]],
    expected_providers: List[Dict[str, Any]],
    operating_id_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    to_keep_config = []
    for provider in existing_providers:
        provider_id: Optional[str] = provider.get('id', None)
        if (
            provider_id is None
            or operating_id_prefix is None
            or not provider_id.startswith(operating_id_prefix)
        ):
            to_keep_config.append(provider)
            continue
    return_providers = expected_providers.copy()
    return_providers.extend(to_keep_config)
    return_providers.sort(key=lambda p: p.get("id", ""))
    return return_providers


def compare_and_append_proxy_match_rules(
    existing_rules: List[WasmPluginMatchRule],
    expected_rules: List[WasmPluginMatchRule],
    operating_id_prefix: Optional[str] = None,
) -> List[WasmPluginMatchRule]:
    to_keep_config = []
    for rule in existing_rules:
        provider_id: Optional[str] = rule.config.get('activeProviderId', None)
        if (
            provider_id is None
            or operating_id_prefix is None
            or not provider_id.startswith(operating_id_prefix)
        ):
            to_keep_config.append(rule)
            continue

    return_rules = expected_rules.copy()
    return_rules.extend(to_keep_config)
    return_rules.sort(key=lambda r: (r.config.get("activeProviderId", None) or ""))
    return return_rules


async def cleanup_ai_proxy_config(
    providers: List[ModelProvider],
    routes: List[ModelRoute],
    k8s_config: k8s_client.Configuration,
    namespace: str,
):
    prefixes_to_keep = {model_route_cleanup_prefix(route.id) for route in routes}
    prefixes_to_keep.update(
        {provider_registry_name(provider.id) for provider in providers}
    )

    def should_keep(provider_id: str) -> bool:
        for prefix in prefixes_to_keep:
            if provider_id.startswith(prefix):
                return True
        return False

    try:
        extensions_api = ExtensionsHigressIoV1Api(k8s_client.ApiClient(k8s_config))
        ai_proxy_data = await extensions_api.get_wasmplugin(
            namespace=namespace,
            name=gpustack_ai_proxy_name,
        )
        existing_plugin = WasmPlugin.model_validate(ai_proxy_data)
        current_providers = existing_plugin.spec.defaultConfig.get("providers", [])
        filtered_providers = [
            p for p in current_providers if p.get("id") and should_keep(p.get("id"))
        ]
        existing_plugin.spec.defaultConfig["providers"] = filtered_providers
        filtered_provider_ids = {
            p.get("id") for p in filtered_providers if p.get("id") is not None
        }
        filtered_rules = [
            r
            for r in existing_plugin.spec.matchRules or []
            if r.config.get("activeProviderId") in filtered_provider_ids
        ]
        existing_plugin.spec.matchRules = filtered_rules
        await extensions_api.edit_wasmplugin(
            namespace=namespace,
            name=gpustack_ai_proxy_name,
            body=existing_plugin,
        )
    except k8s_client.ApiException as e:
        logger.error(
            f"Failed to cleanup gpustack AI proxy wasmplugin {gpustack_ai_proxy_name}: {e}"
        )
        raise


async def cleanup_mcpbridge_registry(
    providers: List[ModelProvider],
    model_instances: List[ModelInstance],
    workers: List[Worker],
    namespace: str,
    k8s_config: k8s_client.Configuration,
):
    worker_by_id = {worker.id: worker for worker in workers}
    networking_higress_api = NetworkingHigressIoV1Api(k8s_client.ApiClient(k8s_config))
    # cleanup providers
    desired_registries = []
    desired_proxies = []
    for provider in providers:
        registry = provider_registry(provider=provider)
        if registry is not None:
            desired_registries.append(registry)
        proxy = provider_proxy(provider=provider)
        if proxy is not None:
            desired_proxies.append(proxy)
    to_delete_prefix = provider_id_prefix
    await ensure_mcp_bridge(
        client=networking_higress_api,
        namespace=namespace,
        mcp_bridge_name=default_mcp_bridge_name,
        desired_registries=desired_registries,
        to_delete_prefix=to_delete_prefix,
        desired_proxies=desired_proxies,
        to_delete_proxies_prefix=provider_id_prefix,
    )
    # cleanup model instances
    desired_registries = []
    to_delete_prefix = model_id_prefix
    for instance in model_instances:
        worker = worker_by_id.get(instance.worker_id)
        registry = model_instance_registry(instance, worker=worker)
        if registry is not None:
            desired_registries.append(registry)
    await ensure_mcp_bridge(
        client=networking_higress_api,
        namespace=namespace,
        mcp_bridge_name=default_mcp_bridge_name,
        desired_registries=desired_registries,
        to_delete_prefix=to_delete_prefix,
    )


def ai_proxy_diff_spec(
    current_spec: Optional[WasmPluginSpec],
    expected_providers: List[Dict[str, Any]],
    expected_match_rules: List[WasmPluginMatchRule],
    operating_id_prefix: Optional[str] = None,
) -> WasmPluginSpec:
    if current_spec is None:
        return current_spec
    current_spec.defaultConfig["providers"] = compare_and_append_default_proxy_config(
        existing_providers=current_spec.defaultConfig.get("providers", []),
        expected_providers=expected_providers,
        operating_id_prefix=operating_id_prefix,
    )
    current_spec.matchRules = compare_and_append_proxy_match_rules(
        existing_rules=current_spec.matchRules or [],
        expected_rules=expected_match_rules,
        operating_id_prefix=operating_id_prefix,
    )
    return current_spec
