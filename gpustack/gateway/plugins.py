import os
from typing import Optional
from fastapi import FastAPI
from gpustack.config.config import Config
from fastapi.staticfiles import StaticFiles

# the digest value is hardcoded in 2025-10-29
supported_plugins = {
    "ai-statistics": [
        (
            "2.0.0",
            "sha256:394b188ebc23935047752cbbfab0418a9085377fab680430738bd7ab324d2c3e",
        )
    ],
    "ext-auth": [
        (
            "2.0.0",
            "sha256:b02da08e0dd519ee0180696f062bb15a04b37de0a88a2c1b46836c07d1e9adab",
        )
    ],
    "model-router": [
        (
            "2.0.0",
            "sha256:4367cff0ce1349ea9885b7e80e495bcd443931d302b6dc804946ecda78545632",
        )
    ],
}

oci_plugin_hostname_path = "higress-registry.cn-hangzhou.cr.aliyuncs.com/plugins"
http_path_prefix = "wasm-plugins"


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
    name: str, version: str, prefix: Optional[str] = None
) -> str:
    if prefix is None:
        prefix = get_plugin_url_prefix()
    supported_versions = [v[0] for v in supported_plugins[name]]
    if name not in supported_plugins or version not in supported_versions:
        raise ValueError(f"Plugin {name} with version {version} is not supported.")
    if prefix.startswith("oci://"):
        return f"{prefix}/{name}:{version}"
    return f"{prefix}/{name}/{version}/plugin.wasm"


def get_plugin_url_prefix(cfg: Optional[Config] = None) -> str:
    plugin_dir = get_wasm_plugin_dir()
    port = (
        cfg.worker_port
        if cfg and cfg.server_role() == Config.ServerRole.WORKER
        else cfg.api_port if cfg else None
    )
    if cfg is not None and plugin_dir is not None and os.path.isdir(plugin_dir):
        return f"http://{cfg.get_advertise_address()}:{port}/{http_path_prefix}"
    return f"oci://{oci_plugin_hostname_path}"


def register(cfg: Config, app: FastAPI):
    plugin_dir = os.getenv("GPUSTACK_HIGRESS_PLUGIN_DIR", None)
    prefix = get_plugin_url_prefix(cfg)

    if plugin_dir is not None and prefix.startswith("http://"):
        app.mount(
            f"/{http_path_prefix}",
            app=StaticFiles(directory=os.path.join(plugin_dir, http_path_prefix)),
            name=http_path_prefix,
        )
