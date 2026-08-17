import json
from dataclasses import dataclass
from urllib.parse import quote
from importlib.resources import files
from typing import Any, Dict, Optional, List
from gpustack.config.config import Config, GatewayPluginEntry
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
    """Where Envoy fetches plugin modules from.

    This setting decides something the code cannot: whether a gateway pod can
    still load its modules while the GPUStack server is down. Unset, the server
    serves them itself, so a pod cold-starting during an outage gets nothing --
    and ``failStrategy: FAIL_OPEN`` turns a module that fails to load into
    inference routes served with no authentication at all. The chart points it
    at a separate deployment, which is what takes module distribution out of
    the server's availability domain; embedded mode does not, and relies on
    Envoy having cached the module for the life of the pod.

    Deliberately no ``sha256`` on what comes back, though the spec accepts one
    and an operator can supply it. The URL already carries the plugin version
    and a version's bytes do not change, so a digest adds no discrimination the
    path does not; and tampering with this fetch requires the same position on
    the network as tampering with the authorization calls to ``/token-auth``,
    which is the stronger attack of the two. It would harden a link that is not
    the weakest one.
    """
    base_url = "http://127.0.0.1"
    if cfg is not None and cfg.gateway_plugin_server_url:
        base_url = cfg.gateway_plugin_server_url
    return f"{base_url}/{http_path_prefix}"


def plugin_entry(name: str, cfg: Optional[Config]) -> Optional[GatewayPluginEntry]:
    """The operator's overrides for one plugin, keyed by its *manifest* name.

    That is the name in ``supported_plugins`` and in the module URL, which for
    five of the eight plugins is not the name of the WasmPlugin resource they
    end up as -- ``gpustack-ext-auth`` is deployed as ``gpustack-llm-ext-auth``,
    ``transformer`` as ``gpustack-header-transformer``, and so on. Keying by
    the resource name instead would silently match nothing.
    """
    if cfg is None:
        return None
    return (cfg.gateway_plugin or {}).get(name)


def plugin_spec_overrides(
    name: str, version: str, cfg: Optional[Config] = None
) -> Dict[str, Any]:
    """The distribution half of a ``WasmPluginSpec``: where the module comes
    from and how it is verified.

    Splat into the spec (``**plugin_spec_overrides(...)``) instead of setting
    ``url=`` directly, so every plugin honours ``gateway_plugin.<name>.url``
    the same way. Overriding the URL without the matching ``sha256`` is
    accepted -- Envoy simply does not verify the module then -- but the pair is
    what an operator pointing at a private registry normally wants.
    """
    entry = plugin_entry(name, cfg)
    if entry is not None and entry.url:
        spec: Dict[str, Any] = {"url": entry.url}
    else:
        spec = {"url": get_plugin_url_with_name_and_version(name, version, cfg)}
    if entry is not None and entry.sha256:
        spec["sha256"] = entry.sha256
    if entry is not None and entry.image_pull_policy:
        spec["imagePullPolicy"] = entry.image_pull_policy
    return spec
