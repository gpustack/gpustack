import json
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from gpustack.config.config import Config
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import (
    HIGRESS_CONFIGMAP,
    HIGRESS_DEFAULT_CONFIG,
    configure_higress_access_log_context,
    get_ingress_tracing_envoyfilter,
)
from gpustack.utils.atomic_file import write_text_atomically

logger = logging.getLogger(__name__)

_HIGRESS_TRACING_FILTER = "gpustack-otel-tracing"


def prepare_gateway_config(cfg: Config, config_path: Path) -> None:
    """Prepare files consumed by the embedded Higress services."""
    higress_embedded_kubeconfig = Path(cfg.higress_base_dir()) / "kubeconfig"

    _prepare_higress_observability(
        higress_embedded_kubeconfig.parent / "apiserver",
        cfg.effective_trace_otlp_endpoint,
        log_enabled=bool(cfg.log_otlp_endpoint),
    )
    with open(config_path, "w") as f:
        f.write(f"DATA_DIR={cfg.data_dir}\n")
        f.write(f"LOG_DIR={cfg.log_dir}\n")
        f.write(f"GATEWAY_HTTP_PORT={cfg.get_gateway_port()}\n")
        f.write(f"GATEWAY_HTTPS_PORT={cfg.tls_port}\n")
        f.write(f"GATEWAY_CONCURRENCY={cfg.gateway_concurrency}\n")
        f.write(f"GPUSTACK_API_PORT={cfg.get_api_port()}\n")
        f.write(f"EMBEDDED_KUBECONFIG_PATH={higress_embedded_kubeconfig}\n")
    with open(higress_embedded_kubeconfig, "w") as f:
        f.write(
            f"""apiVersion: v1
kind: Config
clusters:
  - name: higress
    cluster:
      server: https://127.0.0.1:{os.getenv('APISERVER_PORT', '18443')}
      insecure-skip-tls-verify: true
users:
  - name: higress-admin
    user: {{}}
contexts:
  - name: higress
    context:
      cluster: higress
      user: higress-admin
current-context: higress
"""
        )


def _set_higress_access_log_context(
    configmap_path: Path, template: dict, *, trace_enabled: bool, log_enabled: bool
) -> None:
    defaults = yaml.safe_load(template["data"]["mesh"])
    configmap = (
        yaml.safe_load(configmap_path.read_text())
        if configmap_path.exists()
        else template
    )
    mesh = yaml.safe_load(configmap["data"]["mesh"])
    if configure_higress_access_log_context(
        mesh, defaults, trace_enabled=trace_enabled, log_enabled=log_enabled
    ):
        configmap["data"]["mesh"] = yaml.safe_dump(mesh, sort_keys=False)
        write_text_atomically(
            configmap_path, yaml.safe_dump(configmap, sort_keys=False)
        )
    if log_enabled and not _supports_higress_log_collection(mesh):
        logger.warning(
            "Could not verify the Higress access-log template. Check that emitted "
            "records are JSON with string path and request_id fields."
        )


def _supports_higress_log_collection(mesh: dict) -> bool:
    if not isinstance(mesh, dict):
        return False
    try:
        fields = json.loads(mesh.get("accessLogFormat", ""))
    except (TypeError, ValueError, RecursionError):
        return False
    return isinstance(fields, dict) and all(
        isinstance(fields.get(field), str) for field in ("path", "request_id")
    )


def _prepare_higress_observability(
    apiserver_dir: Path, endpoint: Optional[str], *, log_enabled: bool
) -> None:
    """Configure the embedded gateway before the file-backed API server starts."""
    configmap_path = apiserver_dir / HIGRESS_CONFIGMAP
    filter_path = apiserver_dir / "envoyfilters" / f"{_HIGRESS_TRACING_FILTER}.yaml"
    enabled = bool(endpoint) or log_enabled
    if not enabled:
        filter_path.unlink(missing_ok=True)
        if not configmap_path.exists():
            return

    template = yaml.safe_load((HIGRESS_DEFAULT_CONFIG / HIGRESS_CONFIGMAP).read_text())
    if enabled:
        resource = get_ingress_tracing_envoyfilter(
            _HIGRESS_TRACING_FILTER, template["metadata"]["namespace"], endpoint
        )
        write_text_atomically(
            filter_path,
            yaml.safe_dump(
                resource.model_dump(mode="json", exclude_none=True), sort_keys=False
            ),
        )
    _set_higress_access_log_context(
        configmap_path, template, trace_enabled=bool(endpoint), log_enabled=log_enabled
    )
