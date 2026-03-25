import os
import sys
import argparse
import logging
from pathlib import Path
from shutil import move
from typing import List, Dict
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.envs import MIGRATION_DATA_DIR, DATA_MIGRATION
from gpustack.logging import setup_logging
from gpustack.cmd.start import (
    start_cmd_options,
    parse_args,
)
from gpustack.utils.envs import get_gpustack_env
from gpustack.utils.network import is_port_available
from gpustack.utils.s6_services import (
    gateway_services,
    postgres_services,
    migration_services,
    observability_services,
    all_services,
    gpustack_service_name,
)

logger = logging.getLogger(__name__)


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
        enabled_services = determine_enabled_services(cfg)
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

    return ports


def check_ports_availability(cfg: Config, *services: str):
    # Implement port availability checks here
    all_services = list(services) + [gpustack_service_name]
    ports = ports_for_services(cfg)
    ports_to_check = {
        port: service
        for port, service in ports.items()
        if not all_services or service in all_services
    }
    should_fail = False
    for port, service in ports_to_check.items():
        if not is_port_available(port):
            logger.error(
                f"Port {port} required for service '{service}' is not available."
            )
            should_fail = True
    if should_fail:
        raise Exception("One or more required ports are not available.")


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
    prometheus_config_path.write_text(
        f"""# Managed by GPUStack
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
    http_sd_configs:
      - url: "http://127.0.0.1:{cfg.metrics_port}/metrics/proxy-targets"
        refresh_interval: 1m
  - job_name: gpustack-server
    scrape_interval: 5s
    static_configs:
      - targets:
          - 127.0.0.1:{cfg.metrics_port}
"""
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


def determine_enabled_services(cfg: Config) -> List[str]:
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
