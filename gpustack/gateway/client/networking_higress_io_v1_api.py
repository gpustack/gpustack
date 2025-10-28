from kubernetes_asyncio import client
from kubernetes_asyncio.client import V1ObjectMeta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- McpBridge Data Structures (Pydantic) ---

GROUP = "networking.higress.io"
VERSION = "v1"
PLURAL = "mcpbridges"


class McpBridgeProxy(BaseModel):
    connectTimeout: Optional[int] = None
    listenerPort: Optional[int] = None
    name: Optional[str] = None
    serverAddress: Optional[str] = None
    serverPort: Optional[int] = None
    type: Optional[str] = None


class VPortService(BaseModel):
    name: Optional[str] = None
    value: Optional[int] = None


class VPort(BaseModel):
    default: Optional[int] = None
    services: Optional[List[VPortService]] = Field(default_factory=list)


class McpBridgeRegistry(BaseModel):
    allowMcpServers: Optional[List[str]] = Field(default_factory=list)
    authSecretName: Optional[str] = None
    consulDatacenter: Optional[str] = None
    consulNamespace: Optional[str] = None
    consulRefreshInterval: Optional[int] = None
    consulServiceTag: Optional[str] = None
    domain: Optional[str] = None
    enableMCPServer: Optional[bool] = None
    enableScopeMcpServers: Optional[bool] = None
    mcpServerBaseUrl: Optional[str] = None
    mcpServerExportDomains: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    nacosAccessKey: Optional[str] = None
    nacosAddressServer: Optional[str] = None
    nacosGroups: Optional[List[str]] = Field(default_factory=list)
    nacosNamespace: Optional[str] = None
    nacosNamespaceId: Optional[str] = None
    nacosRefreshInterval: Optional[int] = None
    nacosSecretKey: Optional[str] = None
    name: Optional[str] = None
    port: Optional[int] = None
    protocol: Optional[str] = None
    proxyName: Optional[str] = None
    sni: Optional[str] = None
    type: Optional[str] = None
    vport: Optional[VPort] = None
    zkServicesPath: Optional[List[str]] = Field(default_factory=list)


class McpBridgeSpec(BaseModel):
    proxies: Optional[List[McpBridgeProxy]] = Field(default_factory=list)
    registries: Optional[List[McpBridgeRegistry]] = Field(default_factory=list)


class McpBridgeStatus(BaseModel):
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class McpBridge(BaseModel):
    apiVersion: str = f"{GROUP}/{VERSION}"
    kind: str = "McpBridge"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    spec: Optional[McpBridgeSpec] = None
    status: Optional[McpBridgeStatus] = None

    def parse_metadata(self) -> V1ObjectMeta:
        """
        Parse the metadata dictionary into a V1ObjectMeta object.
        """
        return V1ObjectMeta(**self.metadata) if self.metadata else V1ObjectMeta()


class NetworkingHigressIoV1Api:
    def __init__(self, api_client: client.ApiClient):
        self.custom_api = client.CustomObjectsApi(api_client)

    async def edit_mcpbridge(
        self, namespace: str, name: str, body: McpBridge
    ) -> Dict[str, Any]:
        """Edit (replace) a McpBridge resource."""
        return await self.custom_api.replace_namespaced_custom_object(
            GROUP,
            VERSION,
            namespace,
            PLURAL,
            name,
            (
                body.model_dump(by_alias=True, exclude_none=True)
                if isinstance(body, McpBridge)
                else body
            ),
        )

    async def create_mcpbridge(self, namespace: str, body: McpBridge) -> Dict[str, Any]:
        """Create a McpBridge resource in the given namespace."""
        return await self.custom_api.create_namespaced_custom_object(
            GROUP,
            VERSION,
            namespace,
            PLURAL,
            (
                body.model_dump(by_alias=True, exclude_none=True)
                if isinstance(body, McpBridge)
                else body
            ),
        )

    async def get_mcpbridge(self, namespace: str, name: str) -> Dict[str, Any]:
        """Get a McpBridge resource by name."""
        return await self.custom_api.get_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name
        )

    async def list_mcpbridges(
        self, namespace: str, label_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """List all McpBridge resources in the given namespace."""
        return await self.custom_api.list_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, label_selector=label_selector
        )

    async def patch_mcpbridge(
        self, namespace: str, name: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Patch a McpBridge resource."""
        return await self.custom_api.patch_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name, body
        )

    async def delete_mcpbridge(
        self, namespace: str, name: str, body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Delete a McpBridge resource."""
        return await self.custom_api.delete_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name, body=body
        )
