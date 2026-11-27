from copy import deepcopy
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit
from kubernetes_asyncio import client
from kubernetes_asyncio.client import V1ObjectMeta
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

from gpustack.utils.network import is_ipaddress

# --- McpBridge Data Structures (Pydantic) ---

GROUP = "networking.istio.io"
VERSION = "v1alpha3"
PLURAL = "envoyfilters"
HIGRESS_DEFAULT_CONFIG = Path("/etc/gpustack/data/defaultConfig")
HIGRESS_CONFIGMAP = "configmaps/higress-config.yaml"

_HCM = "envoy.filters.network.http_connection_manager"
_ROUTER = "envoy.filters.http.router"
_CLUSTER = "gpustack-otlp-traces"


class ApplyToStringEnum(str, Enum):
    INVALID = "INVALID"
    LISTENER = "LISTENER"
    FILTER_CHAIN = "FILTER_CHAIN"
    NETWORK_FILTER = "NETWORK_FILTER"
    HTTP_FILTER = "HTTP_FILTER"
    ROUTE_CONFIGURATION = "ROUTE_CONFIGURATION"
    VIRTUAL_HOST = "VIRTUAL_HOST"
    HTTP_ROUTE = "HTTP_ROUTE"
    CLUSTER = "CLUSTER"
    EXTENSION_CONFIG = "EXTENSION_CONFIG"
    BOOTSTRAP = "BOOTSTRAP"
    LISTENER_FILTER = "LISTENER_FILTER"


class MatchContextEnum(str, Enum):
    ANY = "ANY"
    SIDECAR_INBOUND = "SIDECAR_INBOUND"
    SIDECAR_OUTBOUND = "SIDECAR_OUTBOUND"
    GATEWAY = "GATEWAY"


class OperationStringEnum(str, Enum):
    INVALID = "INVALID"
    MERGE = "MERGE"
    ADD = "ADD"
    REMOVE = "REMOVE"
    INSERT_BEFORE = "INSERT_BEFORE"
    INSERT_AFTER = "INSERT_AFTER"
    INSERT_FIRST = "INSERT_FIRST"
    REPLACE = "REPLACE"


class PatchFilterClassEnum(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    AUTHN = "AUTHN"
    AUTHZ = "AUTHZ"
    STATS = "STATS"


class RouteActionEnum(str, Enum):
    ANY = "ANY"
    ROUTE = "ROUTE"
    REDIRECT = "REDIRECT"
    DIRECT_RESPONSE = "DIRECT_RESPONSE"


class WorkloadSelector(BaseModel):
    labels: Optional[Dict[str, str]] = Field(default_factory=dict)


class PolicyTargetReference(BaseModel):
    group: str
    kind: str
    name: str
    namespace: str = ""


class ProxyMatch(BaseModel):
    proxyVersion: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ListenerMatch(BaseModel):
    portNumber: Optional[int] = None
    portName: Optional[str] = None
    filterChain: Optional[Dict[str, Any]] = None
    listenerFilter: Optional[str] = None
    name: Optional[str] = None


class VirtualHostMatchRoute(BaseModel):
    name: Optional[str] = None
    action: Optional[RouteActionEnum] = None


class VirtualHostMatch(BaseModel):
    name: Optional[str] = None
    domainName: Optional[str] = None
    route: Optional[VirtualHostMatchRoute] = None


class RouteConfigurationMatch(BaseModel):
    portNumber: Optional[int] = None
    portName: Optional[str] = None
    gateway: Optional[str] = None
    vhost: Optional[VirtualHostMatch] = None
    name: Optional[str] = None


class ClusterMatch(BaseModel):
    portNumber: Optional[int] = None
    service: Optional[str] = None
    subset: Optional[str] = None
    name: Optional[str] = None


class MatchObjectType(BaseModel):
    listener: Optional[ListenerMatch] = None
    routeConfiguration: Optional[RouteConfigurationMatch] = None
    cluster: Optional[ClusterMatch] = None


class EnvoyConfigObjectMatch(MatchObjectType):
    context: MatchContextEnum
    proxy: Optional[ProxyMatch] = None


class Patch(BaseModel):
    operation: OperationStringEnum
    value: Optional[Dict[str, Any]] = None
    filterClass: Optional[PatchFilterClassEnum] = None


class EnvoyConfigPatchObject(BaseModel):
    applyTo: ApplyToStringEnum
    match: Optional[EnvoyConfigObjectMatch] = None
    patch: Optional[Patch] = None


class EnvoyFilterSpec(BaseModel):
    workloadSelector: Optional[WorkloadSelector] = None
    targetRefs: Optional[List[PolicyTargetReference]] = None
    configPatches: Optional[List[EnvoyConfigPatchObject]] = None
    priority: Optional[int] = None


class EnvoyFilter(BaseModel):
    apiVersion: str = f"{GROUP}/{VERSION}"
    kind: str = "EnvoyFilter"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    spec: Optional[EnvoyFilterSpec] = None

    def parse_metadata(self) -> V1ObjectMeta:
        """
        Parse the metadata dictionary into a V1ObjectMeta object.
        """
        return V1ObjectMeta(**self.metadata) if self.metadata else V1ObjectMeta()


class NetworkingIstioIoV1Alpha3Api:
    def __init__(self, api_client: client.ApiClient):
        self.custom_api = client.CustomObjectsApi(api_client)

    async def edit_envoyfilter(
        self, namespace: str, name: str, body: EnvoyFilter
    ) -> Dict[str, Any]:
        """Edit (replace) a EnvoyFilter resource."""
        return await self.custom_api.replace_namespaced_custom_object(
            GROUP,
            VERSION,
            namespace,
            PLURAL,
            name,
            (
                body.model_dump(by_alias=True, exclude_none=True)
                if isinstance(body, EnvoyFilter)
                else body
            ),
        )

    async def create_envoyfilter(
        self, namespace: str, body: EnvoyFilter
    ) -> Dict[str, Any]:
        """Create a EnvoyFilter resource in the given namespace."""
        return await self.custom_api.create_namespaced_custom_object(
            GROUP,
            VERSION,
            namespace,
            PLURAL,
            (
                body.model_dump(by_alias=True, exclude_none=True)
                if isinstance(body, EnvoyFilter)
                else body
            ),
        )

    async def get_envoyfilter(self, namespace: str, name: str) -> Dict[str, Any]:
        """Get a EnvoyFilter resource by name."""
        return await self.custom_api.get_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name
        )

    async def list_envoyfilters(
        self, namespace: str, label_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """List all EnvoyFilter resources in the given namespace."""
        return await self.custom_api.list_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, label_selector=label_selector
        )

    async def patch_envoyfilter(
        self, namespace: str, name: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Patch a EnvoyFilter resource."""
        return await self.custom_api.patch_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name, body
        )

    async def delete_envoyfilter(
        self, namespace: str, name: str, body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Delete a EnvoyFilter resource."""
        return await self.custom_api.delete_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name, body=body
        )


def get_4xx_5xx_fallback_value(
    ingress_name: str,
    fallback_header: str = "x-higress-fallback-from",
    extra_req_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    response_headers = [
        {
            "append": False,
            "header": {"key": fallback_header, "value": ingress_name},
        }
    ]
    request_headers = response_headers.copy()
    if extra_req_headers:
        for key, value in extra_req_headers.items():
            request_headers.append(
                {
                    "append": False,
                    "header": {"key": key, "value": value},
                }
            )

    redirect_policy = {
        "keep_original_response_code": False,
        "max_internal_redirects": 10,
        "only_redirect_upstream_code": False,
        "request_headers_to_add": request_headers,
        "response_headers_to_add": response_headers,
        "use_original_request_body": True,
        "use_original_request_uri": True,
    }

    action = {
        "name": "action",
        "typed_config": {
            "@type": "type.googleapis.com/udpa.type.v1.TypedStruct",
            "type_url": "type.googleapis.com/envoy.extensions.http.custom_response.redirect_policy.v3.RedirectPolicy",
            "value": redirect_policy,
        },
    }

    def predicate_response_code(code_class: str) -> Dict[str, Any]:
        return {
            "single_predicate": {
                "input": {
                    "name": f"{code_class}_response",
                    "typed_config": {
                        "@type": "type.googleapis.com/envoy.type.matcher.v3.HttpResponseStatusCodeClassMatchInput"
                    },
                },
                "value_match": {"exact": code_class},
            }
        }

    # matcher
    matcher = {
        "on_match": {"action": action},
        "predicate": {
            "or_matcher": {
                "predicate": [
                    predicate_response_code("4xx"),
                    predicate_response_code("5xx"),
                ]
            }
        },
    }

    # custom_response_matcher
    custom_response_matcher = {"matcher_list": {"matchers": [matcher]}}

    typed_per_filter_config = {
        "typed_per_filter_config": {
            "envoy.filters.http.custom_response": {
                "@type": "type.googleapis.com/udpa.type.v1.TypedStruct",
                "type_url": "type.googleapis.com/envoy.extensions.filters.http.custom_response.v3.CustomResponse",
                "value": {"custom_response_matcher": custom_response_matcher},
            }
        }
    }
    return typed_per_filter_config


def get_ingress_fallback_envoyfilter(
    ingress_name: str,
    namespace: str,
    fallback_header: str = "x-higress-fallback-from",
    labels: Optional[Dict[str, str]] = None,
    extra_req_headers: Optional[Dict[str, str]] = None,
) -> EnvoyFilter:
    object_match = EnvoyConfigObjectMatch(
        context=MatchContextEnum.GATEWAY,
        routeConfiguration=RouteConfigurationMatch(
            vhost=VirtualHostMatch(
                route=VirtualHostMatchRoute(
                    name=ingress_name,
                ),
            ),
        ),
    )
    envoyfilter = EnvoyFilter(
        metadata={
            "name": ingress_name,
            "namespace": namespace,
            "labels": labels or {},
        },
        spec=EnvoyFilterSpec(
            configPatches=[
                EnvoyConfigPatchObject(
                    applyTo=ApplyToStringEnum.HTTP_ROUTE,
                    match=object_match,
                    patch=Patch(
                        operation=OperationStringEnum.MERGE,
                        value=get_4xx_5xx_fallback_value(
                            ingress_name, fallback_header, extra_req_headers
                        ),
                    ),
                )
            ]
        ),
    )
    return envoyfilter


def configure_higress_access_log_context(
    mesh: dict,
    defaults: dict,
    *,
    trace_enabled: bool,
    log_enabled: bool,
) -> bool:
    """Switch the bundled access-log format while preserving user formats."""
    default_format = defaults["accessLogFormat"]
    trace_format = default_format.replace(
        '"trace_id":"%REQ(X-B3-TRACEID)%"',
        '"traceparent":"%REQ(TRACEPARENT):55%"',
    )
    original = (defaults["accessLogEncoding"], default_format.strip())
    json_default = ("JSON", default_format.strip())
    traced = ("JSON", trace_format.strip())
    if not isinstance(mesh, dict) or not isinstance(mesh.get("accessLogFormat"), str):
        return False
    current = (mesh.get("accessLogEncoding"), mesh["accessLogFormat"].strip())
    if current not in (original, json_default, traced):
        return False
    encoding = "JSON" if trace_enabled or log_enabled else original[0]
    access_format = trace_format if trace_enabled else default_format
    if current == (encoding, access_format.strip()):
        return False
    mesh.update(accessLogEncoding=encoding, accessLogFormat=access_format)
    return True


def get_ingress_tracing_envoyfilter(
    name: str, namespace: str, endpoint: Optional[str]
) -> EnvoyFilter:
    request_id_header = {
        "header": {"key": "x-gpustack-request-id", "value": "%REQ(X-REQUEST-ID)%"},
        "append_action": "ADD_IF_ABSENT",
    }
    hcm = {
        "@type": (
            "type.googleapis.com/envoy.extensions.filters.network."
            "http_connection_manager.v3.HttpConnectionManager"
        ),
        # Local replies can leave the filter chain before the gateway plugin runs.
        "local_reply_config": {
            "mappers": [
                {
                    "filter": {"not_health_check_filter": {}},
                    "headers_to_add": [deepcopy(request_id_header)],
                }
            ]
        },
    }
    patches = []
    if endpoint:
        cluster, tracing = _ingress_trace_config(endpoint)
        patches.append(
            {
                "applyTo": "CLUSTER",
                "match": {"context": "GATEWAY"},
                "patch": {"operation": "ADD", "value": cluster},
            }
        )
        hcm["tracing"] = tracing

    patches.append(
        _merge_http_filter(
            "NETWORK_FILTER", {"name": _HCM}, {"name": _HCM, "typed_config": hcm}
        )
    )
    if endpoint:
        patches.append(
            _merge_http_filter(
                "HTTP_FILTER",
                {"name": _HCM, "subFilter": {"name": _ROUTER}},
                {
                    "name": _ROUTER,
                    "typed_config": {
                        "@type": (
                            "type.googleapis.com/envoy.extensions.filters.http."
                            "router.v3.Router"
                        ),
                        "start_child_span": True,
                    },
                },
            )
        )
    patches.append(
        {
            "applyTo": "ROUTE_CONFIGURATION",
            "match": {"context": "GATEWAY"},
            "patch": {
                "operation": "MERGE",
                "value": {"response_headers_to_add": [request_id_header]},
            },
        }
    )
    return EnvoyFilter.model_validate(
        {
            "metadata": {"name": name, "namespace": namespace, "resourceVersion": "1"},
            "spec": {"configPatches": patches},
        }
    )


def _ingress_trace_config(endpoint: str) -> Tuple[dict, dict]:
    address = urlsplit(f"//{endpoint}")
    host = address.hostname
    if is_ipaddress(host):
        discovery = {"type": "STATIC"}
    else:
        discovery = {"type": "STRICT_DNS", "dns_lookup_family": "V4_PREFERRED"}

    cluster = {
        "name": _CLUSTER,
        **discovery,
        "connect_timeout": "1s",
        "load_assignment": {
            "cluster_name": _CLUSTER,
            "endpoints": [
                {
                    "lb_endpoints": [
                        {
                            "endpoint": {
                                "address": {
                                    "socket_address": {
                                        "address": host,
                                        "port_value": address.port,
                                    }
                                }
                            }
                        }
                    ]
                }
            ],
        },
        "typed_extension_protocol_options": {
            "envoy.extensions.upstreams.http.v3.HttpProtocolOptions": {
                "@type": (
                    "type.googleapis.com/envoy.extensions.upstreams.http.v3."
                    "HttpProtocolOptions"
                ),
                "explicit_http_config": {"http2_protocol_options": {}},
            }
        },
    }
    tracing = {
        "random_sampling": {"value": 100},
        # Envoy's default http.url tag includes the original path and query.
        "max_path_tag_length": 0,
        "custom_tags": [
            {"tag": "gpustack.request_id", "request_header": {"name": "x-request-id"}}
        ],
        "provider": {
            "name": "envoy.tracers.opentelemetry",
            "typed_config": {
                "@type": "type.googleapis.com/envoy.config.trace.v3.OpenTelemetryConfig",
                "service_name": "gpustack.higress",
                "grpc_service": {
                    "envoy_grpc": {"cluster_name": _CLUSTER},
                    "timeout": "3s",
                },
            },
        },
    }
    return cluster, tracing


def _merge_http_filter(apply_to: str, match: dict, value: dict) -> dict:
    return {
        "applyTo": apply_to,
        "match": {
            "context": "GATEWAY",
            "listener": {"filterChain": {"filter": match}},
        },
        "patch": {"operation": "MERGE", "value": value},
    }
