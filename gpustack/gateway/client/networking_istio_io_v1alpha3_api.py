from enum import Enum
from kubernetes_asyncio import client
from kubernetes_asyncio.client import V1ObjectMeta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- McpBridge Data Structures (Pydantic) ---

GROUP = "networking.istio.io"
VERSION = "v1alpha3"
PLURAL = "envoyfilters"


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
) -> Dict[str, Any]:
    header = {
        "append": False,
        "header": {"key": fallback_header, "value": ingress_name},
    }

    redirect_policy = {
        "keep_original_response_code": False,
        "max_internal_redirects": 10,
        "only_redirect_upstream_code": False,
        "request_headers_to_add": [header],
        "response_headers_to_add": [header],
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
    labels: Optional[Dict[str, str]] = {},
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
            "labels": labels,
        },
        spec=EnvoyFilterSpec(
            configPatches=[
                EnvoyConfigPatchObject(
                    applyTo=ApplyToStringEnum.HTTP_ROUTE,
                    match=object_match,
                    patch=Patch(
                        operation=OperationStringEnum.MERGE,
                        value=get_4xx_5xx_fallback_value(ingress_name, fallback_header),
                    ),
                )
            ]
        ),
    )
    return envoyfilter
