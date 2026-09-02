import re
import hashlib
import logging
import copy
import math
from urllib.parse import urlparse
from dataclasses import dataclass, field as dataclass_field
from functools import partial
from typing import (
    List,
    Iterable,
    Optional,
    Tuple,
    Union,
    Dict,
    Any,
    Literal,
    Callable,
    Set,
    Mapping,
)
from tenacity import retry, stop_after_attempt, wait_fixed
from fastapi import HTTPException
from starlette.datastructures import Headers
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
    Model,
    ModelInstance,
    ModelInstancePublic,
)
from gpustack.schemas.model_provider import (
    ModelProvider,
    ModelProviderTypeEnum,
)
from gpustack.schemas.model_routes import ModelRoute
from gpustack.server.bus import EventType
from gpustack.server.db import async_session
from gpustack.server.services import ModelInstanceService, WorkerService
from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster
from gpustack.utils.network import is_ipaddress
from kubernetes_asyncio import client as k8s_client
from kubernetes_asyncio.client import ApiException, V1IngressTLS
from gpustack.envs import GATEWAY_MIRROR_INGRESS_NAME
from gpustack.api.exceptions import NotFoundException
from gpustack.websocket_proxy.message import ServerInfo, RegisteredClientInfo

logger = logging.getLogger(__name__)

default_mcp_bridge_name = "default"
gpustack_ai_proxy_name = "gpustack-ai-proxy"
gpustack_model_mapper_name = "gpustack-model-mapper"
gpustack_generic_proxy_router_name = "gpustack-model-router"
model_ingress_prefix = "ai-route-model-"
model_route_ingress_prefix = "ai-route-route-"
provider_id_prefix = "provider-"
model_id_prefix = "model-"
# AI proxy provider id prefix for self-hosted deployments. One provider per
# Model (deployment), shared by every ModelRoute pointing at it. Supersedes the
# per-route ``ai-route-route-<route_id>`` ids, which a route retires itself when
# it reconciles (see ``legacy_model_route_provider_id``).
model_ai_proxy_provider_prefix = "gpustack-model-"
# Legacy per-route provider ids happen to reuse the route ingress prefix — alias
# it so call sites read as provider ids rather than as ingress names.
legacy_model_route_provider_prefix = model_route_ingress_prefix
# Provider ids the reconciler may garbage collect once no match rule references
# them: the per-deployment ids it owns, plus the legacy per-route ids those
# supersede.
_collectable_provider_prefixes = (
    model_ai_proxy_provider_prefix,
    legacy_model_route_provider_prefix,
)

router_header_key = "X-GPUStack-Model-Instance"
gpustack_original_path_header = "x-gpustack-original-path"
gpustack_fallback_path_header = "x-gpustack-fallback-path"

# Type alias for destination tuples
# Each tuple contains (weight: int, model_name: str, registry: McpBridgeRegistry)
DestinationTupleList = List[Tuple[int, str, McpBridgeRegistry]]


@dataclass
class ModelAIProxyGroup:
    """AI proxy grouping for one deployment (Model) inside a single route.

    The upstream services of a deployment are its model instances' registries
    (plus the LoRA aliases of those instances), or the remote cluster gateway
    registry when the deployment lives in another cluster. They all accept the
    same credential — the registration token of the deployment's cluster — so
    one provider per Model is both necessary and sufficient, and it is shared by
    every route pointing at that Model.

    ``service_names`` and ``fallback_service_names`` are kept apart because the
    main and the fallback ingress each get their own match rule.
    ``fallback_service_names`` is a subset: a fallback target also serves the
    main ingress (see the FIXME in ``sync_gateway``), so the fallback rule
    covers fewer services, never other ones.
    """

    model_id: int
    api_tokens: List[str] = dataclass_field(default_factory=list)
    service_names: Set[str] = dataclass_field(default_factory=set)
    fallback_service_names: Set[str] = dataclass_field(default_factory=set)
    # Declared on the deployment (``Model.native_anthropic_api``). One answer
    # per Model is not a simplification but a constraint: the provider entry is
    # per Model, while the image -- and so the API surfaces actually served --
    # is settled per instance, so a deployment spread over mismatched images
    # has to answer once for all of them.
    native_anthropic_api: bool = False

    def provider_id(self) -> str:
        return model_ai_proxy_provider_id(self.model_id)


@dataclass
class RoutePrefix:
    prefixes: List[str]
    support_legacy: bool = False
    additional_versions: Optional[List[str]] = None

    def flattened_prefixes(self) -> List[str]:
        versioned_prefixes = ["/v1"]
        if self.support_legacy:
            versioned_prefixes.append("/v1-openai")
        if self.additional_versions:
            versioned_prefixes.extend(self.additional_versions)
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
        versioned_prefixes = [f"/(v1){'(-openai)?' if self.support_legacy else '()'}"]
        if self.additional_versions:
            versioned_prefixes.extend(
                f"/({re.escape(additional_version.lstrip('/'))})()"
                for additional_version in self.additional_versions
            )
        return [
            f"{versioned_prefix}({prefix})"
            for versioned_prefix in versioned_prefixes
            for prefix in self.prefixes
        ]


# Paths routed like OpenAI (model-name resolution / same ingress); includes vLLM extras.
openai_model_prefixes: List[RoutePrefix] = [
    RoutePrefix(
        [
            "/chat/completions",
            "/completions",
            "/responses",
            "/embeddings",
            "/audio/transcriptions",
            "/audio/speech",
            "/images/generations",
            "/images/edits",
        ],
        True,
    ),
    RoutePrefix(
        [
            "/audio/translations",
            "/images/variations",
            "/moderations",
            "/score",
        ]
    ),
    RoutePrefix(["/rerank"], additional_versions=["/v2"]),
]

anthropic_model_exact: List[RoutePrefix] = [
    RoutePrefix(["/messages", "/messages/count_tokens", "/complete"]),
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
    routes = []
    for route_exact in anthropic_model_exact:
        for prefix in route_exact.regex_prefixes():
            routes.append(wrap_route(path=prefix, path_type="ImplementationSpecific"))
    return routes


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


def model_ai_proxy_provider_id(model_id: int) -> str:
    """AI proxy provider id for a self-hosted deployment (Model)."""
    return f"{model_ai_proxy_provider_prefix}{model_id}"


def legacy_model_route_provider_id(model_route_id: int) -> str:
    """AI proxy provider id written by versions that keyed providers per route.

    Superseded by ``model_ai_proxy_provider_id``. A route drops its own legacy
    entry when it reconciles — the legacy rule carries the same ingress the
    reconcile owns, and the provider is then garbage collected as unreferenced —
    so migration happens in one CR write per route, never leaving a route
    without a provider.
    """
    return f"{legacy_model_route_provider_prefix}{model_route_id}"


def model_route_ingress_name(model_route_id: int) -> str:
    return f"{model_route_ingress_prefix}{model_route_id}.internal"


def fallback_ingress_name(name: str) -> str:
    split_name = name.rsplit('.', 1)
    if len(split_name) == 1:
        return f"{name}.fallback"
    return f"{split_name[0]}.fallback.{split_name[1]}"


def route_ingress_names_for_plugins(
    model_route_id: int, resource_namespace: str, gateway_namespace: str
) -> Tuple[str, str]:
    """The (main, fallback) ingress names as WasmPlugin match rules store them.

    Higress qualifies an ingress reference with ``<namespace>/`` when the ingress
    lives outside the gateway's own namespace. The route reconciler and the
    startup pruner must spell the names identically or they will not match.
    """
    prefix = "" if resource_namespace == gateway_namespace else f"{resource_namespace}/"
    ingress_name = model_route_ingress_name(model_route_id)
    return (
        f"{prefix}{ingress_name}",
        f"{prefix}{fallback_ingress_name(ingress_name)}",
    )


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


def lora_registry_name_suffix(lora_route_name: str) -> str:
    """Stable RFC1123-safe suffix identifying a LoRA adapter's aliased registry.

    Derived from the LoRA route name (``<base>:<lora>``) by hashing, so the same
    LoRA always yields the same suffix regardless of position in ``lora_list`` or
    of characters in the LoRA name. Both the McpBridge registration side and the
    destination side compute it independently and must agree, otherwise the alias
    service would dangle (503).
    """
    digest = hashlib.sha256(lora_route_name.encode("utf-8")).hexdigest()[:8]
    return f"l{digest}"


def model_instance_registry(
    model_instance: Union[ModelInstance, ModelInstancePublic],
    worker: Optional[Worker] = None,
    name_suffix: Optional[str] = None,
) -> Optional[McpBridgeRegistry]:
    name = model_instance_prefix(model_instance)
    if name_suffix:
        name = f"{name}-{name_suffix}"
    if worker is not None:
        if worker.proxy_mode == ModelInstanceProxyModeEnum.WORKER:
            return _worker_reserve_proxy_registry(worker, name)
        elif worker.proxy_mode == ModelInstanceProxyModeEnum.TUNNEL:
            return _worker_tunnel_proxy_registry(worker, name)
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


def _worker_reserve_proxy_registry(
    worker: Worker, name_override: Optional[str] = None
) -> McpBridgeRegistry:
    """Build an McpBridgeRegistry entry for a worker in DIRECT or WORKER proxy mode.

    Uses ``worker.advertise_address`` when available, otherwise falls back to
    ``worker.ip``. For raw IP addresses the registry type is set to ``static``
    and the host:port pair is encoded in the domain field; for hostnames the
    type is ``dns`` and the port is carried separately.

    Returns ``None`` if the worker has no resolvable address or port.
    """
    address = worker.advertise_address or worker.ip
    if address is None or address == "" or worker.port is None:
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


def _worker_tunnel_proxy_registry(
    worker: Worker, name_override: Optional[str] = None
) -> Optional[McpBridgeRegistry]:
    """Build an McpBridgeRegistry entry for a worker in TUNNEL proxy mode.

    Points the registry at the server-side HTTP proxy address stored in
    ``worker.proxy_address``, which is populated by
    ``worker_websocket_connect_callback`` when the worker's WebSocket tunnel
    connects. The gateway routes inference requests to this proxy, which then
    tunnels them to the worker via the persistent WebSocket connection.

    Returns ``None`` if the worker has no proxy address (i.e. the WebSocket
    tunnel has not yet connected).
    """
    if worker.get_proxy_address() is None:
        return None
    # proxy address must be a valid URL and the netloc must be a valid IP.
    result = urlparse(worker.get_proxy_address())
    protocol = "http" if result.scheme == "http" else "https"
    port = result.port or (80 if protocol == "http" else 443)
    return McpBridgeRegistry(
        domain=f"{result.hostname}:{port}",
        port=80,
        name=name_override or f"{cluster_worker_prefix(worker.cluster_id)}{worker.id}",
        protocol=protocol,
        type="static",
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


def provider_proxy_name(id: int) -> str:
    return f"{provider_registry_name(id)}-proxy"


def provider_registry(provider: ModelProvider) -> Optional[McpBridgeRegistry]:
    provider_url = provider.config.get_base_url()
    if provider_url is None:
        return None
    result = urlparse(url=provider_url)
    protocol = "http" if result.scheme == "http" else "https"
    port = 443 if protocol == "https" else 80
    registry_type = (
        "static" if result.hostname and is_ipaddress(result.hostname) else "dns"
    )
    if registry_type == "static":
        domain = result.netloc
        if result.port is None:
            domain = f"{domain}:{port}"
    else:
        domain = result.hostname
        if result.port is not None:
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
        name=provider_proxy_name(provider.id),
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
            "type": provider.config.ai_proxy_provider_type(),
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
    to_delete_names: Optional[List[str]] = None,
) -> Tuple[bool, List[McpBridgeRegistry]]:
    desired_map = {
        reg.name: idx for idx, reg in enumerate(desired) if reg.name is not None
    }
    to_delete_name_set = set(to_delete_names or [])
    total_list = []
    need_update = False
    for registry in existing:
        name = registry.name
        if name not in desired_map:
            # delete registries that match the delete prefix or an exact name.
            # ``name`` is Optional[str]; a nameless entry matches nothing and
            # is kept as unrelated.
            if name is not None and (
                (to_delete_prefix is not None and name.startswith(to_delete_prefix))
                or name in to_delete_name_set
            ):
                need_update = True
            else:
                # keep unrelated registries
                total_list.append(registry)
        else:
            # update existing registries
            idx = desired_map.pop(name)
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
    to_delete_names: Optional[List[str]] = None,
) -> Tuple[bool, List[McpBridgeProxy]]:
    desired_map = {
        reg.name: idx for idx, reg in enumerate(desired) if reg.name is not None
    }
    to_delete_name_set = set(to_delete_names or [])
    total_list = []
    need_update = False
    for proxy in existing:
        name = proxy.name
        if name not in desired_map:
            # delete proxies that match the delete prefix or an exact name.
            # ``name`` is Optional[str]; a nameless entry matches nothing and
            # is kept as unrelated.
            if name is not None and (
                (to_delete_prefix is not None and name.startswith(to_delete_prefix))
                or name in to_delete_name_set
            ):
                need_update = True
            else:
                # keep unrelated proxies
                total_list.append(proxy)
        else:
            # update existing proxies
            idx = desired_map.pop(name)
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
    to_delete_names: Optional[List[str]] = None,
    desired_proxies: List[McpBridgeProxy] = None,
    to_delete_proxies_prefix: Optional[str] = None,
    to_delete_proxies_names: Optional[List[str]] = None,
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
            to_delete_names=to_delete_names,
        )
        proxy_need_update = False
        proxy_list = existing_bridge.spec.proxies or []
        if desired_proxies is not None:
            proxy_need_update, proxy_list = diff_proxies(
                existing=existing_bridge.spec.proxies or [],
                desired=desired_proxies,
                to_delete_prefix=to_delete_proxies_prefix,
                to_delete_names=to_delete_proxies_names,
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
    ingress_class_name: str = "higress",
) -> k8s_client.V1Ingress:
    retry_policies = "error,timeout,http_503,http_502,non_idempotent"
    matcher_op = "exact"
    annotations = {
        "higress.io/rewrite-target": "/$1$3",
        "higress.io/destination": destinations,
        "higress.io/ignore-path-case": 'true',
        "higress.io/proxy-next-upstream-tries": '2',
        "higress.io/proxy-next-upstream": retry_policies,
        **higress_http_header_matcher(matcher_op, "x-higress-llm-model", route_name),
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
        # to compatible with rewrite-target /$1$3, the first capturing group is empty.
        # The /\d+ variant strips the route-id segment from /model/proxy/<id>/<path>
        # so the upstream receives /<path>. The id-less variant preserves the legacy
        # /model/proxy/<path> + X-GPUStack-Model header form. The more specific rule
        # is listed first so Higress tries id-based matching before falling back.
        expected_rule.http.paths.append(
            wrap_route(
                r"/()model/proxy/\d+(/|$)(.*)",
                "ImplementationSpecific",
            )
        )
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
    spec = k8s_client.V1IngressSpec(
        ingress_class_name=ingress_class_name, rules=[expected_rule]
    )
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
    downstream_model_name: Optional[str] = None,
    registry_name_suffix: Optional[str] = None,
) -> DestinationTupleList:
    registries: DestinationTupleList = []
    for model_instance in model_instances:
        worker = (
            (workers or {}).get(model_instance.worker_id)
            if model_instance.worker_id
            else None
        )
        registry = model_instance_registry(
            model_instance, worker=worker, name_suffix=registry_name_suffix
        )
        if registry is not None:
            registries.append(
                (1, downstream_model_name or model_instance.model_name, registry)
            )
    return registries


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def ensure_model_ingress(
    ingress_name: str,
    ingress_class_name: str,
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
    if event_type == EventType.DELETED or not destinations:
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
        ingress_class_name=ingress_class_name,
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
    lora_route_names: Optional[List[str]] = None,
) -> List[McpBridgeRegistry]:
    desired_registry: List[McpBridgeRegistry] = []
    to_delete_prefix: Optional[str] = model_prefix(model_id)
    # Each LoRA gets its own registry aliasing the same instance address under a
    # distinct service name, so the gateway can weight traffic across LoRAs and the
    # model-mapper can rewrite per LoRA (both key on service). See calculate_model_destinations.
    name_suffixes = [None] + [
        lora_registry_name_suffix(name) for name in lora_route_names or []
    ]
    if event_type != EventType.DELETED:
        for model_instance in model_instances:
            worker = (
                (workers or {}).get(model_instance.worker_id)
                if model_instance.worker_id
                else None
            )
            for name_suffix in name_suffixes:
                registry = model_instance_registry(
                    model_instance, worker=worker, name_suffix=name_suffix
                )
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
        Tuple[Optional[str], Optional[List[V1IngressTLS]]]: A tuple containing the hostname and ingress TLS settings.
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
            return None, None
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
        extra_req_headers={
            gpustack_fallback_path_header: f'%REQ({gpustack_original_path_header.upper()})%'
        },
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


# Anthropic capabilities added on top of the openai provider's own defaults.
# Keys are the plugin's ``ApiName`` strings and their spelling is load-bearing:
# ai-proxy matches them verbatim and drops keys it does not recognize without
# complaining. Declaring them is what stops it converting an inbound
# /v1/messages; the openai provider keeps its full default set either way,
# which is why this is a top-up rather than a provider-type swap.
_ai_proxy_anthropic_capabilities: Dict[str, str] = {
    "anthropic/v1/messages": "/v1/messages",
    "anthropic/v1/messages/count_tokens": "/v1/messages/count_tokens",
}


def ai_proxy_model_provider_config(
    id: str,
    api_tokens: Optional[List[str]] = None,
    native_anthropic_api: bool = False,
) -> Dict[str, Any]:
    """Build an openai-type provider entry for a self-hosted upstream.

    ``api_tokens`` makes ai-proxy hold the upstream credential statically:
    ``apiTokens`` has the highest priority in its openai provider, so the
    credential no longer has to be injected per request by ``/token-auth``.
    Left empty, ai-proxy falls back to reading the inbound ``Authorization``
    header, which is the pre-existing behavior.

    ``native_anthropic_api`` adds the Anthropic surfaces to ``capabilities``.
    The provider type stays ``openai`` either way: it carries the widest default
    capability set of any provider and never rejects an API it has no capability
    for, where the ``vllm`` provider would trade ~20 of those defaults for two
    and turn every unlisted API into a gateway error. A top-up on ``openai`` is
    strictly additive; a type swap is not.
    """
    return ai_proxy_types.AIProxyDefaultConfig(
        type=ModelProviderTypeEnum.OPENAI,
        id=id,
        apiTokens=api_tokens or None,
        capabilities=_ai_proxy_anthropic_capabilities if native_anthropic_api else None,
        failover=ai_proxy_types.FailoverConfig(enabled=False),
        retryOnFailure=ai_proxy_types.EnableState(enabled=False),
        # ``exclude_unset`` keeps the nested configs down to the flags set here,
        # matching the entries already in the CR; ``apiTokens`` is omitted by
        # ``exclude_none`` alone.
    ).model_dump(exclude_none=True, exclude_unset=True)


def model_ai_proxy_plugin_spec(
    groups: Iterable[ModelAIProxyGroup],
    main_ingress: str,
    fallback_ingress: str,
) -> Tuple[List[Dict[str, Any]], List[WasmPluginMatchRule]]:
    """Build the providers and match rules of one route, grouped by deployment.

    ``main_ingress`` / ``fallback_ingress`` must already carry the namespace
    prefix Higress expects for cross-namespace routes.

    A rule is keyed by (ingress, services) so two routes pointing at the same
    Model each get their own rule while sharing one provider entry. Service
    lists are sorted so an unchanged deployment produces a byte-identical CR and
    the reconciler's diff stays quiet.
    """
    providers: List[Dict[str, Any]] = []
    match_rules: List[WasmPluginMatchRule] = []
    for group in sorted(groups, key=lambda g: g.model_id):
        if not group.service_names and not group.fallback_service_names:
            continue
        provider_id = group.provider_id()
        providers.append(
            ai_proxy_model_provider_config(
                provider_id,
                api_tokens=group.api_tokens,
                native_anthropic_api=group.native_anthropic_api,
            )
        )
        for ingress, service_names in (
            (main_ingress, group.service_names),
            (fallback_ingress, group.fallback_service_names),
        ):
            if not service_names:
                continue
            match_rules.append(
                WasmPluginMatchRule(
                    config={"activeProviderId": provider_id},
                    configDisable=False,
                    service=sorted(service_names),
                    ingress=[ingress],
                )
            )
    return providers, match_rules


def compare_and_append_default_proxy_config(
    existing_providers: List[Dict[str, Any]],
    expected_providers: List[Dict[str, Any]],
    operating_id_prefix: Optional[str] = None,
    referenced_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Merge ``expected_providers`` into the existing list.

    An existing entry is dropped when it is owned by this reconcile
    (``operating_id_prefix``), when it is superseded by an expected entry with
    the same id, or — for the ids under ``_collectable_provider_prefixes`` — when
    ``referenced_ids`` says no remaining match rule points at it. That last rule
    garbage collects a deployment's provider once the last route stops targeting
    it, without any one route having to know about the others, and it retires a
    legacy per-route provider in the same write that replaces its rule.
    """
    expected_ids = {
        provider.get('id') for provider in expected_providers if provider.get('id')
    }
    to_keep_config = []
    for provider in existing_providers:
        provider_id: Optional[str] = provider.get('id', None)
        if provider_id is None:
            to_keep_config.append(provider)
            continue
        if operating_id_prefix is not None and provider_id.startswith(
            operating_id_prefix
        ):
            continue
        if provider_id in expected_ids:
            continue
        if (
            referenced_ids is not None
            and provider_id.startswith(_collectable_provider_prefixes)
            and provider_id not in referenced_ids
        ):
            continue
        to_keep_config.append(provider)
    return_providers = expected_providers.copy()
    return_providers.extend(to_keep_config)
    return_providers.sort(key=lambda p: p.get("id", ""))
    return return_providers


def _match_rule_provider_id(rule: WasmPluginMatchRule) -> Optional[str]:
    # ``config`` is Optional on the CR model, so a hand-edited ``config: null``
    # would otherwise raise on attribute access.
    return (rule.config or {}).get("activeProviderId", None)


def _match_rule_sort_key(rule: WasmPluginMatchRule) -> Tuple[str, str, str]:
    return (
        _match_rule_provider_id(rule) or "",
        ",".join(rule.ingress or []),
        ",".join(rule.service or []),
    )


def compare_and_append_proxy_match_rules(
    existing_rules: List[WasmPluginMatchRule],
    expected_rules: List[WasmPluginMatchRule],
    operating_id_prefix: Optional[str] = None,
    owned_ingresses: Optional[Set[str]] = None,
) -> List[WasmPluginMatchRule]:
    """Merge ``expected_rules`` into the existing list.

    Ownership can be expressed two ways: by provider id prefix (external model
    providers, whose rules match on service only) or by ingress
    (``owned_ingresses``, one route's ingress plus its fallback ingress). The
    latter is required now that provider ids are per deployment: dropping rules
    by id prefix would delete the rules other routes hold for the same
    deployment.
    """
    to_keep_config = []
    owned = owned_ingresses or set()
    for rule in existing_rules:
        provider_id: Optional[str] = _match_rule_provider_id(rule)
        if (
            provider_id is not None
            and operating_id_prefix is not None
            and provider_id.startswith(operating_id_prefix)
        ):
            continue
        if owned and owned.intersection(rule.ingress or []):
            continue
        to_keep_config.append(rule)

    return_rules = expected_rules.copy()
    return_rules.extend(to_keep_config)
    return_rules.sort(key=_match_rule_sort_key)
    return return_rules


async def cleanup_ai_proxy_config(
    providers: List[ModelProvider],
    models: List[Model],
    routes: List[ModelRoute],
    expected_ingresses: Set[str],
    k8s_config: k8s_client.Configuration,
    namespace: str,
):
    """Prune the ai-proxy CR at startup, before the controllers replay routes.

    Kept: one entry per live external provider, one per live deployment, and the
    legacy per-route entry of every live route. Legacy entries are deliberately
    *not* pruned here: each route retires its own when it reconciles, in the same
    write that adds the deployment entry replacing it, so no route is ever left
    without a provider. Pruning them here would open a window between this pass
    and the route replay — which only starts after leader election.

    Dropped: everything else, i.e. entries whose route, deployment or provider no
    longer exists — the case this pass exists for, since nothing will reconcile
    them.

    Rules are filtered on both axes. Provider retention alone is not enough now
    that provider ids are per deployment: a route deleted while the server was
    down leaves a rule that still references a live deployment provider, and no
    reconcile will ever revisit it. ``expected_ingresses`` therefore carries the
    namespace-prefixed ingress names (main and fallback) of every live route, in
    the same form the rules store. Rules with no ingress at all belong to external
    providers, which match on service, and are judged by provider id only.
    """
    ids_to_keep = {model_ai_proxy_provider_id(model.id) for model in models}
    ids_to_keep.update({provider_registry_name(provider.id) for provider in providers})
    ids_to_keep.update({legacy_model_route_provider_id(route.id) for route in routes})

    def should_keep_rule(
        rule: WasmPluginMatchRule, kept_provider_ids: Set[str]
    ) -> bool:
        if _match_rule_provider_id(rule) not in kept_provider_ids:
            return False
        if not rule.ingress:
            return True
        return any(ingress in expected_ingresses for ingress in rule.ingress)

    try:
        extensions_api = ExtensionsHigressIoV1Api(k8s_client.ApiClient(k8s_config))
        ai_proxy_data = await extensions_api.get_wasmplugin(
            namespace=namespace,
            name=gpustack_ai_proxy_name,
        )
        existing_plugin = WasmPlugin.model_validate(ai_proxy_data)
        default_config = existing_plugin.spec.defaultConfig or {}
        current_providers = default_config.get("providers", [])
        filtered_providers = [
            p for p in current_providers if p.get("id") and p.get("id") in ids_to_keep
        ]
        default_config["providers"] = filtered_providers
        existing_plugin.spec.defaultConfig = default_config
        filtered_provider_ids = {
            p.get("id") for p in filtered_providers if p.get("id") is not None
        }
        filtered_rules = []
        for rule in existing_plugin.spec.matchRules or []:
            if should_keep_rule(rule, filtered_provider_ids):
                filtered_rules.append(rule)
            else:
                logger.info(
                    f"Removing ai proxy rule with provider "
                    f"{_match_rule_provider_id(rule)} and ingress {rule.ingress} "
                    "as its route, deployment or provider no longer exists."
                )
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


# Generic-proxy router plugin: defaultConfig.aliasNameMapping is a flat
# ``{str(route_id): effective_model_name}`` dict. The plugin extracts the alias
# id between ``prefix`` and the next ``/`` in :path, looks it up in the
# mapping, and writes the resolved name into ``targetHeader``. The reconcilers
# below mutate that mapping per route; other keys in defaultConfig (prefix,
# targetHeader, any future config) are preserved verbatim.


def generic_proxy_router_diff_spec(
    current_spec: Optional[WasmPluginSpec],
    route_id: int,
    route_name: Optional[str],
) -> Optional[WasmPluginSpec]:
    """
    Set or remove ``aliasNameMapping[str(route_id)]`` on the plugin spec.
    Other entries stay untouched. Returns None if the plugin doesn't exist yet
    — init handles that.

    Pass ``route_name=None`` to remove the alias for this route (e.g. when
    generic_proxy is toggled off or the route is deleted).

    Do NOT touch defaultConfigDisable here — flipping it rewrites Envoy's
    filter chain and tears down every live connection. Only the mapping
    changes between reconciliations; the enable flag is locked at plugin
    creation (see generic_proxy_router_plugin).
    """
    if current_spec is None:
        return current_spec
    default_config = dict(current_spec.defaultConfig or {})
    mapping = dict(default_config.get("aliasNameMapping") or {})
    key = str(route_id)
    if route_name is None:
        mapping.pop(key, None)
    else:
        mapping[key] = route_name
    default_config["aliasNameMapping"] = mapping
    current_spec.defaultConfig = default_config
    return current_spec


def cleanup_generic_proxy_router_spec_diff(
    current_spec: Optional[WasmPluginSpec],
    expected_route_ids: Set[int],
) -> Optional[WasmPluginSpec]:
    """
    Drop aliasNameMapping entries whose key is not in ``expected_route_ids``.
    Used on startup to prune entries for routes that were deleted or had
    generic_proxy toggled off while the server was down.
    """
    if current_spec is None:
        return current_spec
    default_config = dict(current_spec.defaultConfig or {})
    mapping = default_config.get("aliasNameMapping") or {}
    expected_keys = {str(rid) for rid in expected_route_ids}
    retained = {k: v for k, v in mapping.items() if k in expected_keys}
    default_config["aliasNameMapping"] = retained
    current_spec.defaultConfig = default_config
    return current_spec


async def cleanup_generic_proxy_router(
    routes: List[ModelRoute],
    k8s_config: k8s_client.Configuration,
    namespace: str,
):
    """Prune generic-proxy router entries to those for existing generic_proxy routes."""
    expected_route_ids = {
        route.id for route in routes if getattr(route, "generic_proxy", False)
    }
    api = ExtensionsHigressIoV1Api(k8s_client.ApiClient(k8s_config))
    await ensure_wasm_plugin(
        api=api,
        name=gpustack_generic_proxy_router_name,
        namespace=namespace,
        spec_diff=partial(
            cleanup_generic_proxy_router_spec_diff,
            expected_route_ids=expected_route_ids,
        ),
    )


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
    owned_ingresses: Optional[Set[str]] = None,
) -> WasmPluginSpec:
    if current_spec is None:
        return current_spec
    if current_spec.defaultConfig is None:
        current_spec.defaultConfig = {}
    match_rules = compare_and_append_proxy_match_rules(
        existing_rules=current_spec.matchRules or [],
        expected_rules=expected_match_rules,
        operating_id_prefix=operating_id_prefix,
        owned_ingresses=owned_ingresses,
    )
    # Providers are merged against the *resulting* rules so a deployment
    # provider disappears together with the last rule referencing it.
    current_spec.defaultConfig["providers"] = compare_and_append_default_proxy_config(
        existing_providers=current_spec.defaultConfig.get("providers", []),
        expected_providers=expected_providers,
        operating_id_prefix=operating_id_prefix,
        referenced_ids={
            provider_id
            for provider_id in map(_match_rule_provider_id, match_rules)
            if provider_id
        },
    )
    current_spec.matchRules = match_rules
    return current_spec


def get_instance_id_from_header(headers: Mapping[str, str]) -> int:
    """Parse the model instance ID from the ``x-gpustack-model-instance`` routing header.

    The header value follows the pattern
    ``model-<model_id>-<instance_id>[-l<lora>].<type>`` injected by the API
    gateway. The instance ID is the second numeric segment; LoRA targets append
    an extra ``-l<hash>`` alias segment (see ``lora_registry_name_suffix``) which
    must be skipped.

    Raises:
        HTTPException (400): if the header is absent.
        NotFoundException: if the header value does not match the expected pattern.
    """
    if not isinstance(headers, Headers):
        headers = Headers(headers)
    model_destination = headers.get(router_header_key)
    if model_destination is None:
        raise HTTPException(
            status_code=400, detail=f"Missing {router_header_key} header"
        )

    # Match pattern: model-<model_id>-<instance_id>[-<alias>].<type>
    # instance_id is the second numeric segment, optionally followed by a LoRA alias.
    match = re.match(r'^model-\d+-(\d+)(?:-[^.]+)?\..+', model_destination)
    if not match:
        raise NotFoundException(
            message=f"Invalid model destination format: {model_destination}"
        )

    return int(match.group(1))


async def resolve_instance_address_from_model_header(
    headers: Dict[str, str],
) -> Tuple[Optional[str], int]:
    """Resolve the target worker (IP, port) for an inference request.

    Parses the ``x-gpustack-model-instance`` routing header injected by the API gateway
    to extract the model instance ID, then queries the database for that
    instance's worker IP and inference port.

    Used as the ``header_router`` callback of ``HTTPSProxyServer`` in tunnel
    proxy mode so the proxy knows which instance address to forward each request to.

    Returns ``(None, 0)`` when the header is absent or the instance cannot be
    resolved, causing the proxy to fall back to URI-based routing.
    """
    try:
        instance_id = get_instance_id_from_header(headers)
    except HTTPException as e:
        logger.trace(f"direct proxying request as: {e}")
        return None, 0
    except Exception as e:
        logger.debug(f"Error parsing model destination header: {e}")
        return None, 0
    async with async_session() as session:
        model_instance_service = ModelInstanceService(session)
        model_instance: ModelInstance = await model_instance_service.get_by_id(
            instance_id
        )
        if model_instance is None:
            logger.error(f"Model instance with ID {instance_id} not found.")
            return None, 0
        if model_instance.worker_ip is None or len(model_instance.ports) == 0:
            logger.error(
                f"Model instance with ID {instance_id} do not get scheduled yet."
            )
            return None, 0
        return model_instance.worker_ip, model_instance.ports[0]


async def worker_websocket_connect_callback(
    _server: Optional[ServerInfo],
    client: Optional[RegisteredClientInfo],
    proxy_address: Optional[str] = None,
) -> None:
    """Update ``worker.proxy_address`` in the database when a tunnel connects or disconnects.

    Called by ``MessageServerHandler`` as the ``callback_on_connect`` /
    ``callback_on_disconnect`` hook. On connect, ``proxy_address`` is the
    server-side HTTP proxy URL the gateway should route to; on disconnect it is
    ``None``, clearing the field so the worker is no longer reachable via tunnel.

    The worker is looked up by matching ``client.client_id`` against
    ``Worker.worker_uuid``. If no matching worker is found the callback logs an
    error and returns without modifying the database.
    """
    if client is None:
        return
    async with async_session() as session:
        worker = await Worker.one_by_field(
            session=session, field="worker_uuid", value=str(client.client_id)
        )
        if worker is None:
            logger.error(f"Worker with UUID {client.client_id} not found.")
            return
        if worker.proxy_address == proxy_address:
            return
        worker.proxy_address = proxy_address
        await WorkerService(session).update(worker)
