import os
from dataclasses import dataclass
from typing import Optional, List
from fastapi import FastAPI
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.utils.network import get_first_non_loopback_ip
from fastapi.staticfiles import StaticFiles

oci_plugin_prefix = "oci://higress-registry.cn-hangzhou.cr.aliyuncs.com/plugins/"
http_path_prefix = "wasm-plugins"


@dataclass
class HigressPlugin:
    name: str
    version: str
    digest: Optional[str] = None
    registry_prefix: str = oci_plugin_prefix

    def get_oci_url(self) -> Optional[str]:
        if self.digest is not None:
            return f"{self.registry_prefix}{self.name}:{self.version}@{self.digest}"
        return None

    def get_local_path(self) -> str:
        return os.path.join(self.name, self.version, "plugin.wasm")

    def get_path(self, cfg: Optional[Config] = None) -> str:
        prefix = get_plugin_url_prefix(cfg)
        if prefix.startswith("oci://"):
            return self.get_oci_url()
        return f"{prefix}/{self.get_local_path()}"


# the digest value is hardcoded in 2025-10-29
supported_plugins: List[HigressPlugin] = [
    HigressPlugin(
        name="ai-statistics",
        version="2.0.0",
        digest="sha256:237a93b4fcaddedc80fc714af4f234ea95ab19ddc13101f6b5f34e7edec5527f",
    ),
    HigressPlugin(
        name="ext-auth",
        version="2.0.0",
        digest="sha256:b02da08e0dd519ee0180696f062bb15a04b37de0a88a2c1b46836c07d1e9adab",
    ),
    HigressPlugin(
        name="model-router",
        version="2.0.0",
        digest="sha256:f4a30a40f069c1ef68ec87e71a739b3246801cfac7388c98590dd98c89be02cb",
    ),
    HigressPlugin(
        name="transformer",
        version="2.0.0",
        digest="sha256:40ff2e000dcfd94da76c36593bdfd50f4d0f094309c013732da8e1c35171f321",
    ),
    HigressPlugin(
        name="model-mapper",
        version="2.0.0",
        digest="sha256:d6e99e739f83eacf5efef5d00876efcc4ac849a80f19eb19a914d0aa2cce267b",
    ),
    HigressPlugin(
        name="ai-proxy",
        version="2.0.0",
        digest="sha256:a3005bed7d70e8281d4bd26bf5520485a4d7b7994b5e67b9b7ab27b9b4b52212",
    ),
    HigressPlugin(
        name="gpustack-token-usage",
        version="1.0.0",
        digest="sha256:82928ef0b70f1e7b83e8dfafb5f9fe3f2047e6a109928c0676d8eb8701dc2e62",
        registry_prefix="oci://docker.io/gpustack/higress-plugin-",
    ),
]


def get_wasm_plugin_dir(mkdirs: bool = False) -> Optional[str]:
    plugin_dir = os.getenv("GPUSTACK_HIGRESS_PLUGIN_DIR", None)
    if plugin_dir is not None:
        full_path = os.path.join(plugin_dir, http_path_prefix)
        if not mkdirs:
            if os.path.isdir(full_path):
                return full_path
        else:
            os.makedirs(full_path, exist_ok=True)
            return full_path
    return None


def get_plugin_url_with_name_and_version(
    name: str, version: str, cfg: Optional[Config] = None
) -> str:
    target = next(
        (p for p in supported_plugins if p.name == name and p.version == version), None
    )
    if target is None:
        raise ValueError(f"Plugin {name} with version {version} is not supported.")
    return target.get_path(cfg)


def get_plugin_url_prefix(cfg: Optional[Config] = None):
    plugin_dir = get_wasm_plugin_dir()
    address: Optional[str] = None
    if cfg is not None and plugin_dir is not None and os.path.isdir(plugin_dir):
        if cfg.gateway_mode == GatewayModeEnum.embedded:
            address = "127.0.0.1"
        elif cfg.gateway_mode == GatewayModeEnum.incluster:
            address = get_first_non_loopback_ip()
        elif cfg.gateway_mode == GatewayModeEnum.external:
            address = cfg.get_advertise_address()
        return f"http://{address}:{cfg.get_api_port()}/{http_path_prefix}"
    return oci_plugin_prefix


def register(cfg: Config, app: FastAPI):
    plugin_dir = os.getenv("GPUSTACK_HIGRESS_PLUGIN_DIR", None)
    plugin_prefix = get_plugin_url_prefix(cfg)
    if plugin_dir is not None and plugin_prefix.startswith("http://"):
        app.mount(
            f"/{http_path_prefix}",
            app=StaticFiles(directory=os.path.join(plugin_dir, http_path_prefix)),
            name=http_path_prefix,
        )
