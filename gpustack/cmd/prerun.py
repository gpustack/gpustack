import os
import sys
import argparse
import logging
from typing import List, Dict, Optional
from gpustack.config.config import (
    Config,
    GatewayModeEnum,
)
from gpustack.envs import MIGRATION_DATA_DIR
from gpustack.logging import setup_logging
from gpustack.cmd.start import (
    OptionalBoolAction,
    start_cmd_options,
    parse_args,
)
from gpustack.utils.envs import get_gpustack_env
from gpustack.utils.network import is_port_available
from gpustack.utils.s6_services import (
    gateway_services,
    postgres_services,
    migration_services,
    all_dependent_services,
    all_services,
    gpustack_service_name,
)

logger = logging.getLogger(__name__)


def setup_prerun_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "prerun",
        help="Perform pre-run checks and setup s6-overlay configuration for GPUStack.",
        description="Perform pre-run checks and setup s6-overlay configuration for GPUStack.",
    )
    # following args are hidden and used for debugging or advanced usage
    parser_server.add_argument(
        "--s6-base-path",
        type=str,
        help=argparse.SUPPRESS,
        default=get_gpustack_env("S6_BASE_PATH"),
    )
    parser_server.add_argument(
        "--skip-data-dir-check",
        action=OptionalBoolAction,
        help=argparse.SUPPRESS,
        default=False,
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
        if args.skip_data_dir_check is False:
            check_for_data_dir(cfg)

        enabled_services = determine_enabled_services(cfg)
        dependency_services = determine_dependency_services(cfg)
        if len(dependency_services) == 0 and len(enabled_services) == 0:
            logger.info("No extra s6 services for gpustack to enable.")
        else:
            logger.info(
                f"Enabled s6 services: {enabled_services}, dependencies for gpustack: {dependency_services}"
            )
        prepare_s6_overlay(enabled_services, dependency_services, s6_base_path)

        check_ports_availability(cfg, *enabled_services)
        if "postgres" in enabled_services:
            prepare_postgres_config(cfg)
        if cfg.gateway_mode == GatewayModeEnum.embedded:
            prepare_gateway_config(cfg)
        logger.info("Pre-run checks and setup completed successfully.")
    except Exception as e:
        logger.fatal(f"Failed to pre-check the configuration: {e}")
        sys.exit(1)


def check_for_data_dir(cfg: Config):
    if cfg.data_dir != Config.get_data_dir():
        raise Exception(
            f"Custom data directory '{cfg.data_dir}' is not supported in container environment."
        )


def ports_for_services(cfg: Config) -> Dict[int, str]:
    ports = {}
    is_server = cfg.server_role() in [Config.ServerRole.SERVER, Config.ServerRole.BOTH]
    is_worker = cfg.server_role() in [Config.ServerRole.WORKER, Config.ServerRole.BOTH]
    # postgres
    if not cfg.database_url and is_server:
        postgres_services.set_ports(cfg, ports)

    if cfg.gateway_mode == GatewayModeEnum.embedded:
        gateway_services.set_ports(cfg, ports)

    # gpustack server/worker
    gateway_disabled = cfg.gateway_mode == GatewayModeEnum.disabled
    if is_server:
        ports[cfg.port] = gpustack_service_name
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
    ports = ports_for_services(cfg)
    ports_to_check = {
        port: service
        for port, service in ports.items()
        if not services or service in services
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


def cleanup_s6_services(base_path: str, *services: str):
    for service in services:
        service_path = os.path.join(base_path, service)
        if os.path.exists(service_path):
            # it should be a normal file
            os.remove(service_path)


def create_s6_services(base_path: str, *services: str):
    for service in services:
        service_path = os.path.join(base_path, service)
        os.makedirs(os.path.dirname(service_path), exist_ok=True)
        with open(service_path, "w"):
            pass


def prepare_postgres_config(cfg: Config):
    # prepare postgres dirs
    os.makedirs(
        os.getenv("GPUSTACK_POSTGRES_DIR", cfg.postgres_base_dir()),
        exist_ok=True,
    )

    config_path = os.path.join(cfg.postgres_base_dir(), ".env")
    with open(config_path, "w") as f:
        f.write(f"DATA_DIR={cfg.data_dir}\n")
        f.write(f"LOG_DIR={cfg.log_dir}\n")
        f.write(f"EMBEDDED_DATABASE_PORT={cfg.database_port}\n")


def prepare_gateway_config(cfg: Config):
    # prepare gateway dirs
    os.makedirs(
        os.getenv("GPUSTACK_GATEWAY_DIR", cfg.higress_base_dir()),
        exist_ok=True,
    )
    # ensure higress data dir exists
    os.makedirs(cfg.higress_base_dir(), exist_ok=True)
    config_path = os.path.join(cfg.higress_base_dir(), ".env")
    higress_embedded_kubeconfig = os.path.join(cfg.higress_base_dir(), "kubeconfig")

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
      server: https://localhost:{os.getenv('APISERVER_PORT', '18443')}
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

    return services


def determine_dependency_services(cfg: Config) -> List[str]:
    dependencies = []
    # embedded database
    if cfg.database_url is None and cfg.server_role() in [
        Config.ServerRole.SERVER,
        Config.ServerRole.BOTH,
    ]:
        dependencies.extend(postgres_services.dep_services)

    # migration
    should_migrate = MIGRATION_DATA_DIR is not None
    # even if the migration_done file path is hardcoded, we still need to join with data_dir
    # in the container environment, data_dir is always /var/lib/gpustack
    migration_done = os.path.exists(
        os.path.join(cfg.data_dir, "run/state_migration_done")
    )
    # The postgres startup script use the hardcode path /var/lib/gpustack/postgres/data,
    # But for the accommodation of custom data dir, we need to check the actual data dir
    postgres_data_exists = os.path.exists(
        os.path.join(cfg.data_dir, "postgres", "data")
    )
    if should_migrate and not migration_done and not postgres_data_exists:
        dependencies.extend(migration_services.dep_services)

    # gateway services
    if cfg.gateway_mode == GatewayModeEnum.embedded:
        dependencies.extend(gateway_services.dep_services)

    return dependencies


def prepare_s6_overlay(
    enabled_services: List[str],
    dependency_services: List[str],
    s6_base_path: Optional[str],
):
    if s6_base_path is None:
        s6_base_path = "/etc/s6-overlay/s6-rc.d"

    # ensure dirs exist
    gpustack_dependencies_path = os.path.join(s6_base_path, "gpustack/dependencies")
    os.makedirs(gpustack_dependencies_path, exist_ok=True)
    cleanup_s6_services(gpustack_dependencies_path, *all_dependent_services())

    s6_overlay_path = os.path.join(s6_base_path, "user/contents.d")
    os.makedirs(s6_overlay_path, exist_ok=True)
    cleanup_s6_services(s6_overlay_path, *all_services())

    create_s6_services(gpustack_dependencies_path, *dependency_services)
    create_s6_services(s6_overlay_path, *enabled_services)
