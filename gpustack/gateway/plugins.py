import json
from dataclasses import dataclass
from urllib.parse import quote
from importlib.resources import files
from typing import Optional, List
from gpustack.config.config import Config
from gpustack_higress_plugins.server import router as higress_plugins_router

# Reuse the same prefix as the plugin server router
http_path_prefix = higress_plugins_router.prefix.removeprefix("/")


@dataclass
class HigressPlugin:
    name: str
    version: str

    def get_path(self, cfg: Optional[Config] = None) -> str:
        path = "/".join(
            [quote(self.name, safe=""), quote(self.version, safe=""), "plugin.wasm"]
        )
        return f"{get_plugin_url_prefix(cfg)}/{path}"


def _load_plugins_from_manifest() -> List[HigressPlugin]:
    manifest_text = (
        files("gpustack_higress_plugins")
        .joinpath("manifest.json")
        .read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_text)
    return [
        HigressPlugin(name=name, version=info["latest"])
        for name, info in manifest["plugins"].items()
    ]


supported_plugins: List[HigressPlugin] = _load_plugins_from_manifest()


def get_plugin_url_with_name_and_version(
    name: str, version: str, cfg: Optional[Config] = None
) -> str:
    target = next(
        (p for p in supported_plugins if p.name == name and p.version == version), None
    )
    if target is None:
        raise ValueError(f"Plugin {name} with version {version} is not supported.")
    return target.get_path(cfg)


def get_plugin_url_prefix(cfg: Optional[Config] = None) -> str:
    base_url = "http://127.0.0.1"
    if cfg is not None and cfg.gateway_plugin_server_url:
        base_url = cfg.gateway_plugin_server_url
    return f"{base_url}/{http_path_prefix}"
