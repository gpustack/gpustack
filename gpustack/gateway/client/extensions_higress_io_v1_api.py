from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from kubernetes_asyncio import client

GROUP = "extensions.higress.io"
VERSION = "v1alpha1"
PLURAL = "wasmplugins"

# --- WasmPlugin Data Structures (Pydantic) ---


class WasmPluginMatchRule(BaseModel):
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    configDisable: Optional[bool] = None
    domain: Optional[List[str]] = Field(default_factory=list)
    ingress: Optional[List[str]] = Field(default_factory=list)
    service: Optional[List[str]] = Field(default_factory=list)


class WasmPluginVMEnv(BaseModel):
    name: Optional[str] = None
    value: Optional[str] = None
    valueFrom: Optional[str] = None  # INLINE or HOST


class WasmPluginVMConfig(BaseModel):
    env: Optional[List[WasmPluginVMEnv]] = Field(default_factory=list)


class WasmPluginSpec(BaseModel):
    defaultConfig: Optional[Dict[str, Any]] = Field(default_factory=dict)
    defaultConfigDisable: Optional[bool] = None
    failStrategy: Optional[str] = None  # FAIL_CLOSE, FAIL_OPEN
    imagePullPolicy: Optional[str] = None  # UNSPECIFIED_POLICY, IfNotPresent, Always
    imagePullSecret: Optional[str] = None
    matchRules: Optional[List[WasmPluginMatchRule]] = None
    phase: Optional[str] = None  # UNSPECIFIED_PHASE, AUTHN, AUTHZ, STATS
    pluginConfig: Optional[Dict[str, Any]] = None
    pluginName: Optional[str] = None
    priority: Optional[int] = None
    sha256: Optional[str] = None
    url: Optional[str] = None
    verificationKey: Optional[str] = None
    vmConfig: Optional[WasmPluginVMConfig] = None


class WasmPluginStatus(BaseModel):
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WasmPlugin(BaseModel):
    apiVersion: str = f"{GROUP}/{VERSION}"
    kind: str = "WasmPlugin"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    spec: Optional[WasmPluginSpec] = None
    status: Optional[WasmPluginStatus] = None


class ExtensionsHigressIoV1Api:
    def __init__(self, api_client: client.ApiClient):
        self.custom_api = client.CustomObjectsApi(api_client)

    async def edit_wasmplugin(
        self, namespace: str, name: str, body: WasmPlugin
    ) -> Dict[str, Any]:
        """Edit (replace) a WasmPlugin resource."""
        return await self.custom_api.replace_namespaced_custom_object(
            GROUP,
            VERSION,
            namespace,
            PLURAL,
            name,
            (
                body.model_dump(by_alias=True, exclude_none=True)
                if isinstance(body, WasmPlugin)
                else body
            ),
        )

    async def create_wasmplugin(
        self, namespace: str, body: WasmPlugin
    ) -> Dict[str, Any]:
        """Create a WasmPlugin resource in the given namespace."""
        return await self.custom_api.create_namespaced_custom_object(
            GROUP,
            VERSION,
            namespace,
            PLURAL,
            (
                body.model_dump(by_alias=True, exclude_none=True)
                if isinstance(body, WasmPlugin)
                else body
            ),
        )

    async def get_wasmplugin(self, namespace: str, name: str) -> Dict[str, Any]:
        """Get a WasmPlugin resource by name."""
        return await self.custom_api.get_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name
        )

    async def list_wasmplugins(
        self, namespace: str, label_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """List all WasmPlugin resources in the given namespace."""
        return await self.custom_api.list_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, label_selector=label_selector
        )

    async def patch_wasmplugin(
        self, namespace: str, name: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Patch a WasmPlugin resource."""
        return await self.custom_api.patch_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name, body
        )

    async def delete_wasmplugin(
        self, namespace: str, name: str, body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Delete a WasmPlugin resource."""
        return await self.custom_api.delete_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name, body=body
        )
