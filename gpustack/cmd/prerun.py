import os
import sys
import argparse
from decimal import Decimal, ROUND_DOWN
from glob import escape as escape_glob
import json
import logging
from pathlib import Path
import re
from shutil import move
from string import Template
from tempfile import NamedTemporaryFile
from typing import List, Dict, Optional
from urllib.parse import urlsplit

import yaml
from gpustack_runtime import envs as runtime_envs
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.envs import MIGRATION_DATA_DIR, DATA_MIGRATION
from gpustack.logging import setup_logging
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import (
    HIGRESS_CONFIGMAP,
    HIGRESS_DEFAULT_CONFIG,
    configure_higress_access_log_context,
    get_ingress_tracing_envoyfilter,
)
from gpustack.cmd.start import (
    start_cmd_options,
    parse_args,
)
from gpustack.utils.envs import get_gpustack_env
from gpustack.utils.ephemeral_ports import ensure_reserved_against_ephemeral
from gpustack.utils.network import (
    is_port_available,
    is_port_reusable,
    parse_port_range,
)
from gpustack.utils.s6_services import (
    gateway_services,
    postgres_services,
    migration_services,
    observability_services,
    log_collection_services,
    trace_storage_services,
    all_services,
    gpustack_service_name,
)
from gpustack.worker.log_export import ENGINE_LOG_EXPORT_DIR

logger = logging.getLogger(__name__)
ALLOY_LOG_TEMPLATE = Path("/etc/gpustack/logging/docker.alloy.tmpl")
_DOCKER_MOUNTINFO = Path("/proc/self/mountinfo")
_DOCKER_CONTAINER_MOUNT = re.compile(r"/containers/([0-9a-f]{64})/[^/]+$")
_HIGRESS_TRACING_FILTER = "gpustack-otel-tracing"
_TEMPO_DURATION_PART = re.compile(
    r"([0-9]+(?:\.[0-9]*)?|\.[0-9]+)(ns|us|µs|μs|ms|s|m|h)"
)
_TEMPO_DURATION_UNITS = {
    "ns": 1,
    "us": 1_000,
    "µs": 1_000,
    "μs": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60_000_000_000,
    "h": 3_600_000_000_000,
}


def setup_prerun_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "prerun",
        description="Perform pre-run checks and setup s6-overlay configuration for GPUStack.",
    )
    # following args are hidden and used for debugging or advanced usage
    parser_server.add_argument(
        "--s6-base-path",
        type=str,
        help=argparse.SUPPRESS,
        default=get_gpustack_env("S6_BASE_PATH"),
    )
    start_cmd_options(parser_server)

    parser_server.set_defaults(func=run)


def run(args: argparse.Namespace):
    try:
        cfg = parse_args(args)
        setup_logging(cfg.debug)
        logger.info(
            "Starting pre-run checks and setup s6-overlay configuration for GPUStack..."
        )
        s6_base_path = args.s6_base_path or "/etc/s6-overlay/s6-rc.d"
        enabled_services = determine_enabled_services(
            cfg,
            logs_active=prepare_log_collection(cfg),
            tempo_active=prepare_builtin_tempo(cfg),
        )
        # migrate hardcode postgres data dir if needed for determining dependency services
        migrate_hardcode_postgres_data_and_password(cfg, enabled_services)
        dependency_services = determine_dependency_services(cfg)
        if len(dependency_services) == 0 and len(enabled_services) == 0:
            logger.info("No extra s6 services for gpustack to enable.")
        else:
            logger.info(
                f"Enabled s6 services: {enabled_services}, dependencies for gpustack: {dependency_services}"
            )
        prepare_s6_overlay(enabled_services, dependency_services, Path(s6_base_path))

        check_ports_availability(cfg, *enabled_services)
        reserve_ports_against_ephemeral(cfg)
        if "postgres" in enabled_services:
            prepare_postgres_config(cfg)
        if cfg.gateway_mode == GatewayModeEnum.embedded:
            prepare_gateway_config(cfg)
        if use_builtin_grafana(cfg):
            prepare_observability_config(cfg)
        logger.info("Pre-run checks and setup completed successfully.")
    except Exception as e:
        logger.fatal(f"Failed to pre-check the configuration: {e}")
        sys.exit(1)


def ports_for_services(cfg: Config) -> Dict[int, str]:
    ports = {}
    is_server = cfg.server_role() in [Config.ServerRole.SERVER, Config.ServerRole.BOTH]
    is_worker = cfg.server_role() in [Config.ServerRole.WORKER, Config.ServerRole.BOTH]
    # postgres
    if not cfg.database_url and is_server:
        postgres_services.set_ports(cfg, ports)

    if cfg.gateway_mode == GatewayModeEnum.embedded:
        gateway_services.set_ports(cfg, ports)

    if use_builtin_grafana(cfg):
        observability_services.set_ports(cfg, ports)

    # gpustack server/worker
    gateway_disabled = cfg.gateway_mode == GatewayModeEnum.disabled
    enabled_tls = cfg.ssl_certfile is not None and cfg.ssl_keyfile is not None
    if is_server:
        ports[cfg.port] = gpustack_service_name
        ports[cfg.proxy_port] = gpustack_service_name
        if enabled_tls:
            ports[cfg.tls_port] = gpustack_service_name
        if not cfg.disable_metrics:
            ports[cfg.metrics_port] = gpustack_service_name
    if is_worker:
        ports[cfg.worker_port] = gpustack_service_name
        if not cfg.disable_worker_metrics:
            ports[cfg.worker_metrics_port] = gpustack_service_name
    # when gateway is disabled, api port is not required
    if not gateway_disabled:
        ports[cfg.api_port] = gpustack_service_name

    if cfg.use_builtin_tempo() or _is_helm_tempo_pod(cfg):
        trace_storage_services.set_ports(cfg, ports)

    return ports


def check_ports_availability(cfg: Config, *services: str):
    # Implement port availability checks here
    all_services = list(services) + [gpustack_service_name]
    ports = ports_for_services(cfg)
    # Tempo's longrun waits for occupied ports without blocking the Server.
    ports_to_check = {
        port: service
        for port, service in ports.items()
        if service in all_services and service not in trace_storage_services.services
    }
    should_fail = False
    for port, service in ports_to_check.items():
        if is_port_available(port):
            continue
        if is_port_reusable(port):
            # Only residual TCP state (e.g. TIME_WAIT from a process that
            # just exited) holds the port. The s6 services bind with
            # SO_REUSEADDR, the same way this probe just did, so the port
            # is effectively available for them.
            logger.info(
                f"Port {port} required for service '{service}' is held only by "
                "residual TCP state and will be rebound."
            )
            continue
        logger.error(f"Port {port} required for service '{service}' is not available.")
        should_fail = True
    if should_fail:
        raise Exception("One or more required ports are not available.")


def reserve_ports_against_ephemeral(cfg: Config):
    """
    Reserve gpustack's service and Ray port ranges against the kernel's
    ephemeral port range so outbound connections (by gpustack, higress, or
    any other sibling process sharing the netns) don't transiently squat on
    ports that Ray or inference servers will later bind().

    Runs here (in prerun) rather than inside the worker so the reservation
    is applied before any s6 service starts. Skipped on server-only hosts:
    the configured ranges are only bound by model instances and Ray, both
    of which run on worker hosts.
    """
    if cfg.server_role() not in [Config.ServerRole.WORKER, Config.ServerRole.BOTH]:
        return

    ranges = []
    for name in ("service_port_range", "ray_port_range"):
        value = getattr(cfg, name, None)
        if not value:
            continue
        try:
            ranges.append((name, parse_port_range(value)))
        except ValueError:
            logger.debug("Skipping unparseable %s=%r", name, value)
    if ranges:
        ensure_reserved_against_ephemeral(ranges)


def cleanup_s6_services(base_path: Path, *services: str):
    for service in services:
        service_path = base_path / service
        if service_path.exists():
            service_path.unlink()


def create_s6_services(base_path: Path, *services: str):
    for service in services:
        service_path = base_path / service
        service_path.parent.mkdir(parents=True, exist_ok=True)
        service_path.touch()


def migrate_hardcode_postgres_data_and_password(
    cfg: Config, enabled_services: List[str]
):
    if "postgres" not in enabled_services:
        return
    # following paths are hardcoded in the postgres s6 service scripts in v2.0.0.
    # in post 2.0.0 versions, we support custom data dir via cfg.data_dir.
    # here we migrate the data from hardcoded paths to the new paths if needed.
    pair = {
        Path("/var/lib/gpustack/postgres/data"): Path(cfg.postgres_base_dir()) / "data",
        Path("/var/lib/gpustack/postgres_root_pass"): Path(cfg.data_dir)
        / "postgres_root_pass",
        Path("/var/lib/gpustack/run/migration_done"): get_migration_done_file(cfg),
    }
    for hardcode_path, target_path in pair.items():
        if hardcode_path == target_path or not hardcode_path.exists():
            continue
        if target_path.exists():
            logger.warning(
                f"Both hardcoded postgres file/dir {hardcode_path} and postgres file/dir with data_dir {target_path} exist. Only {target_path} will be used."
            )
            continue
        logger.info(
            f"Migrating hardcoded postgres file/dir {hardcode_path} to {target_path}"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        move(hardcode_path, target_path)


def prepare_postgres_config(cfg: Config):
    # prepare postgres dirs
    # same reason as gateway_shared_config_dir
    config_path = prepare_env("GPUSTACK_POSTGRES_CONFIG", "postgresql")
    with open(config_path, "w") as f:
        f.write(f"DATA_DIR={cfg.data_dir}\n")
        f.write(f"LOG_DIR={cfg.log_dir}\n")
        f.write(f"EMBEDDED_DATABASE_PORT={cfg.database_port}\n")
        f.write(f"STATE_MIGRATION_DONE_FILE={get_migration_done_file(cfg)}\n")
        f.write(f"POSTGRES_DATA_DIR={os.path.join(cfg.postgres_base_dir(), 'data')}\n")


def get_migration_done_file(cfg: Config) -> Path:
    return Path(cfg.data_dir) / "run" / "state_migration_done"


def prepare_gateway_config(cfg: Config):
    # prepare gateway dirs
    config_path = prepare_env("GPUSTACK_GATEWAY_CONFIG", "gateway")
    higress_embedded_kubeconfig = Path(cfg.higress_base_dir()) / "kubeconfig"

    if cfg.gateway_mode == GatewayModeEnum.embedded:
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


def _write_yaml_atomically(path: Path, content: dict) -> None:
    _write_text_atomically(path, yaml.safe_dump(content, sort_keys=False))


def _write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        temporary_path = Path(output.name)
        try:
            output.write(content)
            output.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


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
        _write_yaml_atomically(configmap_path, configmap)
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
        _write_yaml_atomically(
            filter_path, resource.model_dump(mode="json", exclude_none=True)
        )
    _set_higress_access_log_context(
        configmap_path, template, trace_enabled=bool(endpoint), log_enabled=log_enabled
    )


def _log_export_options(cfg: Config) -> dict:
    endpoint = cfg.log_otlp_endpoint or ""
    try:
        url = urlsplit(endpoint)
        port = url.port
    except ValueError:
        raise ValueError("Invalid OTLP logs URL") from None
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or not url.path
        or url.username is not None
        or url.query
        or url.fragment
        or port == 0
        or any(character.isspace() for character in endpoint)
    ):
        raise ValueError("Expected a full OTLP/HTTP logs URL without credentials")

    authorization = (
        cfg.log_otlp_authorization.get_secret_value()
        if cfg.log_otlp_authorization
        else ""
    )
    if (authorization or cfg.log_otlp_ca_file) and url.scheme != "https":
        raise ValueError("OTLP log authentication and custom CAs require HTTPS")
    if any(ord(character) < 32 or ord(character) == 127 for character in authorization):
        raise ValueError("Invalid OTLP Authorization header")

    headers = (
        f'{{"Authorization" = {json.dumps(authorization)}}}' if authorization else "{}"
    )
    return {
        "endpoint": json.dumps(endpoint),
        "headers": headers,
        "ca_file": json.dumps(cfg.log_otlp_ca_file or ""),
    }


def _docker_application_log_receiver() -> str:
    # Docker bind-mounts these generated files even with a custom hostname
    # or a private cgroup namespace.
    container_ids = set()
    try:
        for line in _DOCKER_MOUNTINFO.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 5 or fields[4] not in (
                "/etc/hostname",
                "/etc/hosts",
                "/etc/resolv.conf",
            ):
                continue
            match = _DOCKER_CONTAINER_MOUNT.search(fields[3])
            if match:
                container_ids.add(match[1])
    except OSError:
        pass

    if len(container_ids) != 1:
        logger.warning("Unable to identify this Docker container for application logs")
        return ""

    (container_id,) = container_ids
    host = runtime_envs.GPUSTACK_RUNTIME_DOCKER_HOST.removeprefix("http+")
    return f'''loki.source.docker "application" {{
  host = {json.dumps(host)}
  targets = [{{"__meta_docker_container_id" = {json.dumps(container_id)}}}]
  forward_to = [otelcol.receiver.loki.application.receiver]
}}

otelcol.receiver.loki "application" {{
  output {{
    logs = [gpustack.logs.export.application]
  }}
}}'''


def _check_log_redact_patterns() -> None:
    name = "GPUSTACK_LOG_REDACT_PATTERNS"
    try:
        patterns = json.loads(os.getenv(name) or "[]")
    except json.JSONDecodeError:
        raise ValueError(f"{name} must be a JSON array of strings") from None
    if not isinstance(patterns, list):
        raise ValueError(f"{name} must be a JSON array of strings")
    for index, pattern in enumerate(patterns):
        # Alloy compiles the expressions with RE2 when it loads the pipeline.
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{name}[{index}] must be a non-empty RE2 expression")


def prepare_log_collection(cfg: Config) -> bool:
    """Generate the Docker Alloy config when log export is configured."""
    path = Path(os.getenv("GPUSTACK_RUN_DIR", "/run/gpustack")) / "logging/config.alloy"
    try:
        path.unlink(missing_ok=True)
        if not cfg.log_otlp_endpoint or cfg.log_pod_release:
            return False

        _check_log_redact_patterns()
        options = _log_export_options(cfg)
        directory = escape_glob(str(Path(cfg.log_dir)))
        state = Path(cfg.data_dir) / "log-collector"
        engine = [
            f"{directory}/{ENGINE_LOG_EXPORT_DIR}/archives/*/{name}{suffix}"
            for name in ("container.log", "container.workload-*.log")
            for suffix in ("", ".[0-9]*")
        ]
        higress_receiver = ""
        if cfg.gateway_mode == GatewayModeEnum.embedded:
            higress = [
                f"{directory}/higress/access.log{suffix}" for suffix in ("", ".[1-5]")
            ]
            higress_receiver = f'''otelcol.receiver.filelog "higress" {{
  include = {json.dumps(higress)}
  start_at = "end"
  storage = otelcol.storage.file.offsets.handler

  output {{
    logs = [gpustack.logs.export.higress]
  }}
}}'''

        config = Template(ALLOY_LOG_TEMPLATE.read_text()).substitute(
            **options,
            offsets=json.dumps(str(state)),
            application_receiver=_docker_application_log_receiver(),
            engine=json.dumps(engine),
            higress_receiver=higress_receiver,
        )

        _write_text_atomically(path.with_name("storage-path"), str(state / "alloy"))
        # The generated config can contain credentials. The temporary file has mode 0600.
        _write_text_atomically(path, config)
    except (OSError, TypeError, KeyError, ValueError) as exc:
        logger.error("Unable to enable application log collection: %s", exc)
        return False
    return True


def _is_helm_tempo_pod(cfg: Config) -> bool:
    pod = os.getenv("GPUSTACK_BUILTIN_TEMPO_POD")
    return bool(pod and pod == cfg.server_id and not cfg.disable_tracing)


def _valid_tempo_retention(retention: str) -> bool:
    # Go durations support compound units and truncate each part to nanoseconds.
    duration = retention.removeprefix("+")
    parts = list(_TEMPO_DURATION_PART.finditer(duration))
    if "".join(part[0] for part in parts) != duration:
        return False
    nanoseconds = sum(
        (Decimal(part[1]) * _TEMPO_DURATION_UNITS[part[2]]).to_integral_value(
            rounding=ROUND_DOWN
        )
        for part in parts
    )
    return 0 < nanoseconds < 2**63


def prepare_builtin_tempo(cfg: Config) -> bool:
    """Write the configuration consumed by the supervised Tempo service."""
    helm_tempo = _is_helm_tempo_pod(cfg)
    if not helm_tempo and not cfg.use_builtin_tempo():
        return False

    path = Path(os.getenv("GPUSTACK_RUN_DIR") or "/run/gpustack") / "tempo/config.yaml"
    data = Path(cfg.data_dir) / "tempo"
    retention_env = "GPUSTACK_BUILTIN_TEMPO_RETENTION"
    default_retention = "168h"
    retention = os.getenv(retention_env) or default_retention
    if not _valid_tempo_retention(retention):
        retention = default_retention
        logger.warning(
            "Invalid %s. Using %s for embedded Tempo retention.",
            retention_env,
            retention,
        )
    max_bytes_env = "GPUSTACK_BUILTIN_TEMPO_MAX_RETAINED_BYTES"
    default_max_bytes = 30_000_000_000
    try:
        max_retained_bytes = int(os.getenv(max_bytes_env, str(default_max_bytes)))
    except ValueError:
        max_retained_bytes = 0
    if not 0 < max_retained_bytes < 2**64:
        max_retained_bytes = default_max_bytes
        logger.warning(
            "Invalid %s. Using %d bytes for embedded Tempo retention.",
            max_bytes_env,
            max_retained_bytes,
        )
    loopback = "127.0.0.1"
    server = {
        "http_listen_address": loopback,
        "http_listen_port": cfg.builtin_tempo_http_port,
        "grpc_listen_address": loopback,
        "grpc_listen_port": cfg.builtin_tempo_grpc_port,
    }
    if helm_tempo and cfg.builtin_tempo_tls_dir:
        tls_dir = Path(cfg.builtin_tempo_tls_dir)
        server.update(
            http_listen_address="0.0.0.0",
            http_tls_config={
                "cert_file": str(tls_dir / "server.crt"),
                "key_file": str(tls_dir / "server.key"),
                "client_ca_file": str(tls_dir / "ca.crt"),
                "client_auth_type": "RequireAndVerifyClientCert",
            },
        )
    try:
        _write_yaml_atomically(
            path,
            {
                "target": "all",
                "server": server,
                "distributor": {
                    "receivers": {
                        "otlp": {
                            "protocols": {
                                "grpc": {
                                    "endpoint": f"0.0.0.0:{cfg.builtin_tempo_otlp_port}"
                                },
                            },
                        },
                    },
                },
                "ingester": {
                    "lifecycler": {"address": loopback},
                    "flush_all_on_shutdown": True,
                },
                "memberlist": {"bind_addr": [loopback], "bind_port": 0},
                "storage": {
                    "trace": {
                        "backend": "local",
                        "blocklist_poll": "1m",
                        "wal": {"path": str(data / "wal")},
                        "local": {"path": str(data / "blocks")},
                    },
                },
                "compactor": {
                    "compaction": {
                        "block_retention": retention,
                        "max_retained_bytes": max_retained_bytes,
                    }
                },
                "usage_report": {"reporting_enabled": False},
            },
        )
    except OSError as exc:
        logger.warning("Unable to configure embedded Tempo: %s", exc)
        return False
    return True


def prepare_observability_config(cfg: Config):
    env_config_path = prepare_env("GPUSTACK_OBSERVABILITY_CONFIG", "observability")
    with open(env_config_path, "w") as f:
        f.write(f"DATA_DIR={cfg.data_dir}\n")
        f.write(f"LOG_DIR={cfg.log_dir}\n")
        f.write(f"PROMETHEUS_PORT={cfg.builtin_prometheus_port}\n")
        f.write(f"GF_SERVER_HTTP_PORT={cfg.builtin_grafana_port}\n")
        f.write(f"PROMETHEUS_DATA_DIR={os.path.join(cfg.data_dir, 'prometheus')}\n")
        f.write(f"GF_PATHS_DATA={os.path.join(cfg.data_dir, 'grafana')}\n")
        f.write(f"GF_PATHS_LOGS={os.path.join(cfg.log_dir, 'grafana')}\n")
        f.write(
            f"GF_PATHS_PLUGINS={os.path.join(cfg.data_dir, 'grafana', 'plugins')}\n"
        )

    prometheus_config_path = Path(
        os.getenv("PROMETHEUS_CONFIG_FILE", "/etc/prometheus/prometheus.yml")
    )
    prometheus_config_path.parent.mkdir(parents=True, exist_ok=True)
    prometheus_config = f"""# Managed by GPUStack
global:
  scrape_interval: 15s
  scrape_timeout: 10s
  evaluation_interval: 15s
scrape_configs:
  - job_name: gpustack-worker-discovery
    scrape_interval: 5s
    http_sd_configs:
      - url: "http://127.0.0.1:{cfg.metrics_port}/metrics/targets"
        refresh_interval: 1m
  - job_name: gpustack-proxy-worker-discovery
    scrape_interval: 5s
    proxy_url: "http://127.0.0.1:{cfg.get_proxy_port()}"
    # The forward proxy authenticates every request, so scrapes routed
    # through it present the derived gateway token. Prometheus sends this
    # as the Authorization header on the proxied absolute-URI request.
    authorization:
      credentials: "{cfg.get_derived_gateway_token()}"
    http_sd_configs:
      - url: "http://127.0.0.1:{cfg.metrics_port}/metrics/proxy-targets"
        refresh_interval: 1m
  - job_name: gpustack-server
    scrape_interval: 5s
    static_configs:
      - targets:
          - 127.0.0.1:{cfg.metrics_port}
"""

    if cfg.gateway_mode == GatewayModeEnum.embedded:
        pilot_status_port = os.getenv('GATEWAY_PILOT_AGENT_METRICS_PORT', '15020')
        prometheus_config += f"""  - job_name: gpustack-gateway
    metrics_path: /stats/prometheus
    scrape_interval: 5s
    static_configs:
      - targets:
          - 127.0.0.1:{pilot_status_port}
        labels:
          higress: {cfg.gateway_namespace}-higress-gateway
"""

    if cfg.gateway_mode == GatewayModeEnum.incluster:
        # The in-cluster gateway pods carry the standard prometheus.io/*
        # annotations (set by the higress-core chart), so pod service discovery
        # follows replicas and pod recreation without further configuration.
        # Discovery uses the server pod's own service account, which the chart
        # grants pods get/list/watch to.
        prometheus_config += f"""  - job_name: higress-gateway-pods
    metrics_path: /stats/prometheus
    scrape_interval: 5s
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - {cfg.gateway_namespace}
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: "true"
      - source_labels: [__meta_kubernetes_pod_name]
        action: keep
        regex: .*higress-gateway.*
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\\d+)?;(\\d+)
        replacement: $1:$2
        target_label: __address__
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
      - target_label: higress
        replacement: {cfg.gateway_namespace}-higress-gateway
"""

    prometheus_config_path.write_text(prometheus_config)
    # The config carries the gateway token, so keep it to the services that
    # read it. A read-only or exotic mount can refuse chmod even on a file
    # this process just wrote; failing to tighten the mode must not take
    # the whole server down with it.
    try:
        os.chmod(prometheus_config_path, 0o600)
    except OSError as e:
        logger.warning(
            f"Failed to restrict permissions on {prometheus_config_path}: {e}"
        )

    grafana_provisioning_dir = Path(
        os.getenv("GF_PATHS_PROVISIONING", "/etc/grafana/provisioning")
    )
    datasource_path = grafana_provisioning_dir / "datasources" / "datasource.yaml"
    datasource_path.parent.mkdir(parents=True, exist_ok=True)
    datasource_path.write_text(
        f"""apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    uid: prometheus
    url: http://127.0.0.1:{cfg.builtin_prometheus_port}/prometheus
    isDefault: true
    access: proxy
    editable: true
    orgId: 1
"""
    )


def determine_enabled_services(
    cfg: Config, logs_active: bool = False, tempo_active: bool = False
) -> List[str]:
    services = []
    # embedded database
    if cfg.database_url is None and cfg.server_role() in [
        Config.ServerRole.SERVER,
        Config.ServerRole.BOTH,
    ]:
        services.extend(postgres_services.all_services())

    # gateway services
    if cfg.gateway_mode == GatewayModeEnum.embedded:
        services.extend(gateway_services.all_services())

    # embedded observability
    if use_builtin_grafana(cfg):
        services.extend(observability_services.all_services())
    if logs_active:
        services.extend(log_collection_services.all_services())
    if tempo_active:
        services.extend(trace_storage_services.all_services())

    return services


def use_builtin_grafana(cfg: Config) -> bool:
    if cfg.disable_builtin_observability:
        return False
    if cfg.grafana_url is not None:
        return False

    server_role = cfg.server_role() in [
        Config.ServerRole.SERVER,
        Config.ServerRole.BOTH,
    ]
    return server_role


def determine_dependency_services(cfg: Config) -> List[str]:
    dependencies = []

    if cfg.server_role() in [
        Config.ServerRole.SERVER,
        Config.ServerRole.BOTH,
    ]:
        # embedded database
        if cfg.database_url is None:
            dependencies.extend(postgres_services.dep_services)

        # migration
        old_db_file = Path(cfg.data_dir) / "database.db"
        should_migrate = (
            MIGRATION_DATA_DIR is not None or DATA_MIGRATION
        ) and old_db_file.exists()
        if should_migrate and MIGRATION_DATA_DIR is not None:
            logger.warning(
                f"The environment variable GPUSTACK_MIGRATION_DATA_DIR is deprecated. The migration target dir will be set to {cfg.data_dir} instead."
            )
        # This is the hardcooded migration done file path
        migration_done = get_migration_done_file(cfg).exists()
        postgres_data_exists = (Path(cfg.data_dir) / "postgres" / "data").exists()
        if (
            cfg.database_url is None
            and should_migrate
            and not migration_done
            and not postgres_data_exists
        ):
            dependencies.extend(migration_services.dep_services)

    # gateway services
    if cfg.gateway_mode == GatewayModeEnum.embedded:
        dependencies.extend(gateway_services.dep_services)

    return dependencies


def prepare_s6_overlay(
    enabled_services: List[str],
    dependency_services: List[str],
    s6_base_path: Path = Path("/etc/s6-overlay/s6-rc.d"),
):

    s6_overlay_path = s6_base_path / "user/contents.d"
    s6_overlay_path.mkdir(parents=True, exist_ok=True)
    cleanup_s6_services(s6_overlay_path, *all_services())

    create_s6_services(
        s6_overlay_path, *(set(enabled_services) | set(dependency_services))
    )


def prepare_env(env_name: str, scope: str, env_file_name: str = ".env") -> Path:
    config_path = os.getenv(env_name)
    if config_path is None:
        base_dir = Path(os.getenv("GPUSTACK_RUN_DIR", "/run/gpustack"))
        config_path = base_dir / scope / env_file_name
    else:
        config_path = Path(config_path)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    return config_path
