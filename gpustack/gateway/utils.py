import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.gateway.labels_annotations import managed_labels, match_labels
from gpustack.gateway.client.networking_higress_io_v1_api import (
    NetworkingHigressIoV1Api,
    McpBridge,
    McpBridgeRegistry,
    McpBridgeSpec,
)
from gpustack.gateway.client.extensions_higress_io_v1_api import (
    WasmPlugin,
    WasmPluginSpec,
    ExtensionsHigressIoV1Api,
)
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstancePublic,
    Model,
    ModelPublic,
)
from gpustack.server.bus import EventType
from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster
from gpustack.utils.network import is_ipaddress
from kubernetes_asyncio import client as k8s_client
from kubernetes_asyncio.client import ApiException

logger = logging.getLogger(__name__)

default_mcp_bridge_name = "default"


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


def ingress_rule_for_model(
    mcp_bridge_name: str = default_mcp_bridge_name,
) -> k8s_client.V1IngressRule:
    paths: List[k8s_client.V1HTTPIngressPath] = []
    for route_prefix in openai_model_prefixes:
        for prefix in route_prefix.regex_prefixes():
            paths.append(
                k8s_client.V1HTTPIngressPath(
                    path=prefix,
                    path_type="ImplementationSpecific",
                    backend=k8s_client.V1IngressBackend(
                        resource=get_default_mcpbridge_ref(mcp_bridge_name),
                    ),
                )
            )
    return k8s_client.V1IngressRule(http=k8s_client.V1HTTPIngressRuleValue(paths=paths))


def cluster_mcp_bridge_name(cluster_id: int) -> str:
    # higress_controller has hardcoded mcp bridge name to 'default'
    # the name should be based on cluster_id if higress_controller supports multiple mcp bridges
    return default_mcp_bridge_name


def model_mcp_bridge_name(cluster_id: int) -> str:
    return cluster_mcp_bridge_name(cluster_id)


def model_ingress_name(model_id: int) -> str:
    return f"ai-route-model-{model_id}"


def cluster_worker_prefix(cluster_id: int) -> str:
    return f"cluster-{cluster_id}-worker-"


def model_prefix(model_instance: Union[ModelInstance, ModelInstancePublic]) -> str:
    return f"model-{model_instance.model_id}-"


def model_instance_prefix(
    model_instance: Union[ModelInstance, ModelInstancePublic]
) -> str:
    return f"{model_prefix(model_instance)}{model_instance.id}"


def model_instance_registry(
    model_instance: Union[ModelInstance, ModelInstancePublic]
) -> Optional[McpBridgeRegistry]:
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
        name=model_instance_prefix(model_instance),
        protocol="http",
        type=registry_type,
    )


def worker_registry(worker: Worker) -> Optional[McpBridgeRegistry]:
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
        name=f"{cluster_worker_prefix(worker.cluster_id)}{worker.id}",
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


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def ensure_mcp_bridge(
    client: NetworkingHigressIoV1Api,
    namespace: str,
    mcp_bridge_name: str,
    desired_registries: List[McpBridgeRegistry],
    to_delete_prefix: Optional[str] = None,
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
            spec=McpBridgeSpec(registries=desired_registries),
        )
        await client.create_mcpbridge(
            namespace=namespace,
            body=mcpbridge_body,
        )
        logger.info(f"Created MCP Bridge {mcp_bridge_name} in namespace {namespace}.")
    else:
        need_update, registry_list = diff_registries(
            existing=existing_bridge.spec.registries or [],
            desired=desired_registries,
            to_delete_prefix=to_delete_prefix,
        )
        if need_update:
            existing_bridge.spec.registries = registry_list
            await client.edit_mcpbridge(
                name=mcp_bridge_name,
                namespace=namespace,
                body=existing_bridge,
            )
            logger.info(
                f"Updated MCP Bridge {mcp_bridge_name} in namespace {namespace}."
            )


def generate_model_ingress(
    namespace: str,
    model: Model,
    destinations: str,
    hostname: Optional[str] = None,
    tls_secret_name: Optional[str] = None,
    included_generic_route: Optional[bool] = False,
    included_proxy_route: Optional[bool] = False,
) -> k8s_client.V1Ingress:
    ingress_name = model_ingress_name(model.id)
    namespace = namespace
    metadata = k8s_client.V1ObjectMeta(
        name=ingress_name,
        namespace=namespace,
        annotations={
            "higress.io/rewrite-target": "/$1$3",
            "higress.io/destination": destinations,
            "higress.io/exact-match-header-x-higress-llm-model": model.name,
            "higress.io/ignore-path-case": 'true',
        },
        labels=managed_labels,
    )
    bridge_name = model_mcp_bridge_name(model.cluster_id)
    expected_rule = ingress_rule_for_model(
        mcp_bridge_name=bridge_name,
    )

    if included_proxy_route:
        # to compatible with rewrite-target /$1$3, the first capturing group is empty
        expected_rule.http.paths.append(
            k8s_client.V1HTTPIngressPath(
                path="/()model/proxy(/|$)(.*)",
                path_type="ImplementationSpecific",
                backend=k8s_client.V1IngressBackend(
                    resource=get_default_mcpbridge_ref(mcp_bridge_name=bridge_name),
                ),
            )
        )
    if included_generic_route:
        expected_rule.http.paths.append(
            k8s_client.V1HTTPIngressPath(
                path="/",
                path_type="Prefix",
                backend=k8s_client.V1IngressBackend(
                    resource=get_default_mcpbridge_ref(mcp_bridge_name=bridge_name),
                ),
            )
        )

    expected_rule.host = hostname
    spec = k8s_client.V1IngressSpec(ingress_class_name="higress", rules=[expected_rule])
    if tls_secret_name is not None:
        spec.tls = [
            k8s_client.V1IngressTLS(
                hosts=[hostname] if hostname is not None else None,
                secret_name=tls_secret_name,
            )
        ]
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


def replace_registry_weight(registry_list: List[Tuple[int, McpBridgeRegistry]]):
    """
    return persentage dict for each registry.
    The total should not exceed 100.
    """
    total = sum(count for count, _ in registry_list)
    acc = 0
    for i, (count, registry) in enumerate(registry_list):
        if i < len(registry_list) - 1:
            percent = round(100 * count / total) if total > 0 else 0
            registry_list[i] = (percent, registry)
            acc += percent
        else:
            # The last one, ensure the total sum is 100
            percent = 100 - acc
            registry_list[i] = (percent, registry)


def model_instances_registry_list(
    model_instances: List[Union[ModelInstance, ModelInstancePublic]]
) -> List[Tuple[int, McpBridgeRegistry]]:
    registries: List[Tuple[int, McpBridgeRegistry]] = []
    for model_instance in model_instances:
        registry = model_instance_registry(model_instance)
        if registry is not None:
            registries.append((1, registry))
    return registries


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def ensure_model_ingress(
    namespace: str,
    destinations: List[Tuple[int, McpBridgeRegistry]],
    model: Union[Model, ModelPublic],
    event_type: EventType,
    networking_api: k8s_client.NetworkingV1Api,
    hostname: Optional[str] = None,
    tls_secret_name: Optional[str] = None,
    included_generic_route: Optional[bool] = False,
    included_proxy_route: Optional[bool] = False,
):
    """
    Ensure the model ingress resource in Kubernetes matches the desired state.

    Parameters:
        namespace (str): The Kubernetes namespace for the ingress resource.
        destinations (List[Tuple[int, McpBridgeRegistry]]): Weighted list of MCP Bridge registries for traffic routing.
        model (Union[Model, ModelPublic]): The model object for which ingress is managed.
        event_type (EventType): The event type (CREATED, UPDATED, DELETED) triggering reconciliation.
        networking_api (k8s_client.NetworkingV1Api): The Kubernetes networking API client.
        hostname (Optional[str]): The external hostname for ingress routing.
        tls_secret_name (Optional[str]): TLS secret name for HTTPS ingress.
        included_generic_route (bool): Whether to include a generic '/' route for fallback traffic. Used in worker gateway.
        included_proxy_route (bool): Whether to include a proxy route for model traffic (e.g., /model/proxy/{model_name}). Used in server gateway.
    """
    ingress_name = model_ingress_name(model.id)
    if event_type == EventType.DELETED:
        try:
            await networking_api.delete_namespaced_ingress(
                name=ingress_name, namespace=namespace
            )
            logger.info(f"Deleted model ingress {ingress_name} for model {model.name}")
        except ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete ingress {ingress_name}: {e}")
        return

    expected_destinations = '\n'.join(
        [
            f"{persentage}% {registry.name}.{registry.type}:{registry.port}"
            for persentage, registry in destinations
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
    expected_ingress = generate_model_ingress(
        namespace=namespace,
        model=model,
        destinations=expected_destinations,
        hostname=hostname,
        tls_secret_name=tls_secret_name,
        included_generic_route=included_generic_route,
        included_proxy_route=included_proxy_route,
    )
    if existing_ingress is None:
        await networking_api.create_namespaced_ingress(
            namespace=namespace,
            body=expected_ingress,
        )
        logger.info(f"Created model ingress {ingress_name} for model {model.name}")
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
            logger.info(f"Updated model ingress {ingress_name} for model {model.name}")


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def ensure_wasm_plugin(
    api: ExtensionsHigressIoV1Api, name: str, namespace: str, expected: WasmPluginSpec
):
    current_plugin = None
    try:
        data: Dict[str, Any] = await api.get_wasmplugin(namespace=namespace, name=name)
        current_plugin = WasmPlugin.model_validate(data)
    except ApiException as e:
        if e.status == 404:
            current_plugin = None
        else:
            raise
    if current_plugin is None:
        wasm_plugin_body = WasmPlugin(
            metadata={
                "name": name,
                "namespace": namespace,
                "labels": managed_labels,
            },
            spec=expected,
        )
        await api.create_wasmplugin(
            namespace=namespace,
            body=wasm_plugin_body,
        )
        logger.info(f"Created WasmPlugin {name} in namespace {namespace}.")
    elif match_labels(current_plugin.metadata.get("labels", {}), managed_labels):
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


async def cleanup_orphaned_model_ingresses(
    namespace: str,
    existing_model_ids: List[int],
    config: k8s_client.Configuration,
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
            if ingress.metadata and ingress.metadata.name:
                name = ingress.metadata.name
                if name.startswith("ai-route-model-"):
                    try:
                        model_id_str = name[len("ai-route-model-") :]
                        model_id = int(model_id_str)
                        if model_id not in existing_model_ids:
                            await networking_api.delete_namespaced_ingress(
                                name=name, namespace=namespace
                            )
                            logger.info(
                                f"Deleted orphaned model ingress {name} in namespace {namespace}."
                            )
                    except ValueError:
                        # not a valid model id, skip
                        continue
    except Exception as e:
        print(f"Error cleaning up orphaned model ingresses: {e}")


async def ensure_model_instance_mcp_bridge(
    event_type: EventType,
    model_instance: Union[ModelInstance, ModelInstancePublic],
    networking_higress_api: NetworkingHigressIoV1Api,
    namespace: str,
    cluster_id: int,
) -> List[McpBridgeRegistry]:
    desired_registry: List[McpBridgeRegistry] = []
    to_delete_prefix: Optional[str] = None
    if event_type == EventType.DELETED:
        to_delete_prefix = model_instance_prefix(model_instance)
    else:
        registry = model_instance_registry(model_instance)
        if registry is not None:
            desired_registry.append(registry)
    await ensure_mcp_bridge(
        client=networking_higress_api,
        namespace=namespace,
        mcp_bridge_name=cluster_mcp_bridge_name(cluster_id),
        desired_registries=desired_registry,
        to_delete_prefix=to_delete_prefix,
    )
    return desired_registry
