import logging
from typing import List, Optional, Tuple, Union, Dict
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.gateway.labels_annotations import managed_labels
from gpustack.gateway.client.networking_higress_io_v1_api import (
    NetworkingHigressIoV1Api,
    McpBridge,
    McpBridgeRegistry,
    McpBridgeSpec,
)
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstancePublic,
    Model,
    ModelPublic,
    CategoryEnum,
)
from gpustack.server.bus import EventType
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster
from kubernetes_asyncio import client as k8s_client
from kubernetes_asyncio.client import ApiException

logger = logging.getLogger(__name__)

default_mcp_bridge_name = "default"

openai_model_prefixes: Dict[CategoryEnum, List[str]] = {
    CategoryEnum.LLM: ["/v1/chat/completions", "/v1/completions"],
    CategoryEnum.EMBEDDING: ["/v1/embeddings"],
    CategoryEnum.SPEECH_TO_TEXT: ["/v1/audio/transcriptions"],
    CategoryEnum.TEXT_TO_SPEECH: ["/v1/audio/speech"],
    CategoryEnum.RERANKER: ["/v1/rerank"],
}


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
    model: Union[Model, ModelPublic], mcp_bridge_name: str = default_mcp_bridge_name
) -> k8s_client.V1IngressRule:
    paths: List[k8s_client.V1HTTPIngressPath] = []
    for category, prefixes in openai_model_prefixes.items():
        if category not in model.categories:
            continue
        for prefix in prefixes:
            paths.append(
                k8s_client.V1HTTPIngressPath(
                    path=prefix,
                    path_type="Exact",
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
    if (
        model_instance.worker_ip is None
        or model_instance.worker_ip == ""
        or model_instance.port is None
    ):
        return None
    return McpBridgeRegistry(
        domain=f"{model_instance.worker_ip}:{model_instance.port}",
        port=80,
        name=model_instance_prefix(model_instance),
        protocol="http",
        type="static",
    )


def worker_registry(worker: Worker) -> Optional[McpBridgeRegistry]:
    if worker.ip is None or worker.ip == "" or worker.gateway_port is None:
        return None
    return McpBridgeRegistry(
        domain=f"{worker.ip}:{worker.gateway_port}",
        port=80,
        name=f"{cluster_worker_prefix(worker.cluster_id)}{worker.id}",
        protocol="http",
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
    namespace: str, model: Model, destinations: str
) -> k8s_client.V1Ingress:
    ingress_name = model_ingress_name(model.id)
    namespace = namespace
    metadata = k8s_client.V1ObjectMeta(
        name=ingress_name,
        namespace=namespace,
        annotations={
            "higress.io/destination": destinations,
            "higress.io/ignore-path-case": 'true',
            "higress.io/exact-match-header-x-higress-llm-model": model.name,
        },
    )
    expected_rule = ingress_rule_for_model(
        mcp_bridge_name=model_mcp_bridge_name(model.cluster_id),
        model=model,
    )
    spec = k8s_client.V1IngressSpec(ingress_class_name="higress", rules=[expected_rule])
    ingress = k8s_client.V1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=metadata,
        spec=spec,
    )
    return ingress


def mcp_ingress_equal(
    existing: k8s_client.V1Ingress, expected: k8s_client.V1Ingress
) -> bool:
    existing_metadata = existing.metadata or k8s_client.V1ObjectMeta()
    expected_metadata = expected.metadata or k8s_client.V1ObjectMeta()
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
    if existing.spec is None or expected.spec is None:
        return False
    if len(existing.spec.rules or []) != len(expected.spec.rules or []):
        return False
    for existing_rule, expected_rule in zip(
        existing.spec.rules or [], expected.spec.rules or []
    ):
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
):
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
            for key in [
                "higress.io/destination",
                "higress.io/exact-match-header-x-higress-llm-model",
            ]:
                metadata.annotations[key] = expected_ingress.metadata.annotations.get(
                    key
                )
            await networking_api.patch_namespaced_ingress(
                name=ingress_name,
                namespace=namespace,
                body=existing_ingress,
            )
            logger.info(f"Updated model ingress {ingress_name} for model {model.name}")
