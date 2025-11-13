import argparse
import logging
import sys
from typing import Dict, Any

from gpustack import __version__, __git_commit__
from gpustack.cmd.start import load_config_from_yaml
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.utils.envs import get_gpustack_env, get_gpustack_env_bool


logger = logging.getLogger(__name__)


class OptionalBoolAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(OptionalBoolAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs
        )
        self.default = kwargs.get("default")

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def setup_reload_config_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "reload-config",
        help="Reload GPUStack configuration from args and config file.",
        description="Reload GPUStack configuration from command line arguments and config file, equivalent to re-running cfg = parse_args(args).",
    )

    # Common settings group - same as start command
    group = parser.add_argument_group("Common settings")
    group.add_argument(
        "--advertise-address",
        type=str,
        help="The IP address to expose for external access. If not set, the system will auto-detect a suitable local IP address.",
        default=get_gpustack_env("ADVERTISE_ADDRESS"),
    )
    group.add_argument(
        "--port",
        type=int,
        help="Port to bind the server to.",
        default=get_gpustack_env("PORT"),
    )
    group.add_argument(
        "--tls-port",
        type=int,
        help="Port to bind the TLS server to.",
        default=get_gpustack_env("TLS_PORT"),
    )
    group.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML config file.",
        default=get_gpustack_env("CONFIG_FILE"),
    )
    group.add_argument(
        "-d",
        "--debug",
        action=OptionalBoolAction,
        help="Enable debug mode.",
        default=get_gpustack_env_bool("DEBUG"),
    )
    group.add_argument(
        "--data-dir",
        type=str,
        help="Directory to store data. The default is OS specific.",
        default=get_gpustack_env("DATA_DIR"),
    )
    group.add_argument(
        "--cache-dir",
        type=str,
        help="Directory to store cache (e.g., model files). The default is <data-dir>/cache.",
        default=get_gpustack_env("CACHE_DIR"),
    )
    group.add_argument(
        "--bin-dir",
        type=str,
        help="Directory to store additional binaries, e.g., versioned backend executables.",
        default=get_gpustack_env("BIN_DIR"),
    )
    group.add_argument(
        "--pipx-path",
        type=str,
        help="Path to the pipx executable, used to install versioned backends.",
        default=get_gpustack_env("PIPX_PATH"),
    )
    group.add_argument(
        "--huggingface-token",
        type=str,
        help="User Access Token to authenticate to the Hugging Face Hub.",
        default=get_gpustack_env("HF_TOKEN"),
    )
    group.add_argument(
        "--system-default-container-registry",
        type=str,
        help="Default container registry for GPUStack to pull system and inference images. The default is 'docker.io'.",
        default=get_gpustack_env("SYSTEM_DEFAULT_CONTAINER_REGISTRY"),
    )
    group.add_argument(
        "--image-name-override",
        type=str,
        help="Override the default image name for the GPUStack container.",
        default=get_gpustack_env("IMAGE_NAME_OVERRIDE"),
    )
    group.add_argument(
        "--image-repo",
        type=str,
        help="Override the default image repository gpustack/gpustack for the GPUStack container.",
        default=get_gpustack_env("IMAGE_REPO"),
    )
    group.add_argument(
        "--gateway-mode",
        type=str,
        help="Gateway running mode. Options: embedded, in-cluster, external, disabled, or auto (default).",
        default=get_gpustack_env("GATEWAY_MODE"),
    )
    group.add_argument(
        "--gateway-kubeconfig",
        type=str,
        help="Path to the kubeconfig file for gatway. Only useful for external gateway-mode.",
        default=get_gpustack_env("GATEWAY_KUBECONFIG"),
    )
    group.add_argument(
        "--gateway-concurrency",
        type=int,
        help="Number of concurrent connections for the gateway. The default is 16.",
        default=get_gpustack_env("GATEWAY_CONCURRENCY"),
    )
    group.add_argument(
        "--service-discovery-name",
        type=str,
        help="the name of the service discovery service in DNS. Only useful when deployed in Kubernetes with service discovery.",
        default=get_gpustack_env("SERVICE_DISCOVERY_NAME"),
    )
    group.add_argument(
        "--namespace",
        type=str,
        help="Kubernetes namespace for GPUStack to deploy gateway routing rules and model instances.",
        default=get_gpustack_env("NAMESPACE"),
    )

    # Server settings group
    group = parser.add_argument_group("Server settings")
    group.add_argument(
        "--api-port",
        type=int,
        help="Port to bind the GPUStack API server to.",
        default=get_gpustack_env("API_PORT"),
    )
    group.add_argument(
        "--database-port",
        type=int,
        help="Port of the database. Example: 5432 for PostgreSQL.",
        default=get_gpustack_env("DATABASE_PORT"),
    )
    group.add_argument(
        "--metrics-port",
        type=int,
        help="Port to expose server metrics.",
        default=get_gpustack_env("METRICS_PORT"),
    )
    group.add_argument(
        "--database-url",
        type=str,
        help="URL of the database. Example: postgresql://user:password@hostname:port/db_name.",
        default=get_gpustack_env("DATABASE_URL"),
    )
    group.add_argument(
        "--disable-worker",
        action=OptionalBoolAction,
        help="Disable embedded worker.",
        default=get_gpustack_env_bool("DISABLE_WORKER"),
    )
    group.add_argument(
        "--disable-metrics",
        action=OptionalBoolAction,
        help="Disable server metrics.",
        default=get_gpustack_env_bool(
            "DISABLE_METRICS",
        ),
    )
    group.add_argument(
        "--bootstrap-password",
        type=str,
        help="Initial password for the default admin user. Random by default.",
        default=get_gpustack_env("BOOTSTRAP_PASSWORD"),
    )
    group.add_argument(
        "--ssl-keyfile",
        type=str,
        help="Path to the SSL key file.",
        default=get_gpustack_env("SSL_KEYFILE"),
    )
    group.add_argument(
        "--ssl-certfile",
        type=str,
        help="Path to the SSL certificate file.",
        default=get_gpustack_env("SSL_CERTFILE"),
    )
    group.add_argument(
        "--force-auth-localhost",
        action=OptionalBoolAction,
        help="Force authentication for requests originating from localhost (127.0.0.1)."
        "When set to True, all requests from localhost will require authentication.",
        default=get_gpustack_env_bool("FORCE_AUTH_LOCALHOST"),
    )
    group.add_argument(
        "--disable-update-check",
        action=OptionalBoolAction,
        help="Disable update check.",
        default=get_gpustack_env_bool("DISABLE_UPDATE_CHECK"),
    )
    group.add_argument(
        "--disable-openapi-docs",
        action=OptionalBoolAction,
        help="Disable autogenerated OpenAPI documentation endpoints (Swagger UI at /docs, ReDoc at /redoc, and the raw spec at /openapi.json).",
        default=get_gpustack_env_bool("DISABLE_OPENAPI_DOCS"),
    )
    group.add_argument(
        "--model-catalog-file",
        type=str,
        help="Path or URL to the model catalog file.",
        default=get_gpustack_env("MODEL_CATALOG_FILE"),
    )
    group.add_argument(
        "--enable-cors",
        action=OptionalBoolAction,
        help="Enable Cross-Origin Resource Sharing (CORS) on the server.",
        default=get_gpustack_env_bool("ENABLE_CORS"),
    )
    group.add_argument(
        "--allow-credentials",
        action=OptionalBoolAction,
        help="Allow cookies and credentials in cross-origin requests.",
        default=get_gpustack_env_bool("ALLOW_CREDENTIALS"),
    )
    group.add_argument(
        "--allow-origins",
        action='append',
        help='Origins allowed for cross-origin requests. Specify the flag multiple times for multiple origins. Example: --allow-origins https://example.com --allow-origins https://api.example.com. Default: ["*"] (all origins allowed).',
    )
    group.add_argument(
        "--allow-methods",
        action='append',
        help='HTTP methods allowed in cross-origin requests. Specify the flag multiple times for multiple methods. Example: --allow-methods GET --allow-methods POST. Default: ["GET", "POST"].',
    )
    group.add_argument(
        "--allow-headers",
        action='append',
        help='HTTP request headers allowed in cross-origin requests. Specify the flag multiple times for multiple headers. Example: --allow-headers Authorization --allow-headers Content-Type. Default: ["Authorization", "Content-Type"].',
    )

    # Worker settings group
    group = parser.add_argument_group("Worker settings")
    group.add_argument(
        "-t",
        "--token",
        type=str,
        help="Shared secret used to add a worker.",
        default=get_gpustack_env("TOKEN"),
    )
    group.add_argument(
        "-s",
        "--server-url",
        type=str,
        help="Server to connect to.",
        default=get_gpustack_env("SERVER_URL"),
    )
    group.add_argument(
        "--worker-ip",
        type=str,
        help="IP address of the worker node. Auto-detected by default.",
        default=get_gpustack_env("WORKER_IP"),
    )
    group.add_argument(
        "--worker-ifname",
        type=str,
        help="Network interface name of the worker node. Auto-detected by default.",
        default=get_gpustack_env("WORKER_IFNAME"),
    )
    group.add_argument(
        "--worker-name",
        type=str,
        help="Name of the worker node. Use the hostname by default.",
        default=get_gpustack_env("WORKER_NAME"),
    )
    group.add_argument(
        "--worker-port",
        type=int,
        help="Port to bind the worker to.",
        default=get_gpustack_env("WORKER_PORT"),
    )
    group.add_argument(
        "--service-port-range",
        type=str,
        help="Port range for inference services, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive. The default is '40000-40063'.",
        default=get_gpustack_env("SERVICE_PORT_RANGE"),
    )
    group.add_argument(
        "--distributed-worker-port-range",
        type=str,
        help="Generic port range for distributed worker processes (e.g., NCCL/TCP communication), specified as 'N1-N2'. Both ends inclusive. The default is '40200-40999'.",
        default=get_gpustack_env("DISTRIBUTED_WORKER_PORT_RANGE"),
    )
    group.add_argument(
        "--disable-worker-metrics",
        action=OptionalBoolAction,
        help="Disable worker metrics.",
        default=get_gpustack_env_bool(
            "DISABLE_WORKER_METRICS",
        ),
    )
    group.add_argument(
        "--worker-metrics-port",
        type=int,
        help="Port to expose worker metrics.",
        default=get_gpustack_env("WORKER_METRICS_PORT"),
    )
    group.add_argument(
        "--log-dir",
        type=str,
        help="Directory to store logs.",
        default=get_gpustack_env("LOG_DIR"),
    )
    group.add_argument(
        "--system-reserved",
        type=dict,
        help="The system reserves resources during scheduling, measured in GiB. "
        "Where RAM is reserved per worker, and VRAM is reserved per GPU device. "
        "By default, 2 GiB of RAM and 1G of VRAM is reserved. "
        "Example: '{\"ram\": 2, \"vram\": 1}' or '{\"memory\": 2, \"gpu_memory\": 1}', "
        "Note: The 'memory' and 'gpu_memory' keys are deprecated and will be removed in future releases.",
        default=get_gpustack_env("SYSTEM_RESERVED"),
    )
    group.add_argument(
        "--tools-download-base-url",
        type=str,
        help="Base URL to download dependency tools.",
        default=get_gpustack_env("TOOLS_DOWNLOAD_BASE_URL"),
    )
    group.add_argument(
        "--enable-hf-transfer",
        action=OptionalBoolAction,
        help="Enable faster downloads from the Hugging Face Hub using hf_transfer.",
        default=get_gpustack_env("ENABLE_HF_TRANSFER"),
    )

    parser.set_defaults(func=run)


def run(args):
    """Reload configuration from args and config file."""
    try:
        logger.info("Starting configuration reload...")
        logger.info(f"GPUStack version: {__version__} ({__git_commit__})")

        # Validate and filter configuration changes against whitelist
        filtered_config_data = filter_configuration_changes(args)

        # Create a custom parse function that uses filtered config
        cfg = parse_args_with_filter(args, filtered_config_data)

        # Setup logging with new debug setting
        setup_logging(cfg.debug)

        # Display key configuration information
        display_config_summary(cfg)

    except Exception as e:
        logger.error(f"Failed to reload configuration: {e}")
        sys.exit(1)


def display_config_summary(cfg):
    """Display a summary of the reloaded configuration - only show whitelisted fields."""
    logger.info("=== Configuration Reload Summary ===")

    for field in CONFIG_WHITELIST:
        if hasattr(cfg, field):
            value = getattr(cfg, field)
            if value is not None:
                logger.info(f"- reload: {field} = {value}")
    logger.info("Configuration successfully reloaded.")

    logger.info("=====================================")


def filter_whitelisted_yaml_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter YAML configuration data to only allow whitelisted options.

    Returns filtered configuration data.
    """
    if not config_data:
        return config_data

    filtered_data = {}

    for key, value in config_data.items():
        # Convert key to match Config class attribute names (snake_case)
        config_key = key.replace('-', '_')

        if config_key in CONFIG_WHITELIST:
            filtered_data[key] = value
            logger.info(f"Allowing YAML configuration: {key} = {value}")

    return filtered_data


# Configuration whitelist - only these options can be modified via reload-config
CONFIG_WHITELIST = {
    'debug',  # Log level is safe to change
    'system_default_container_registry',  # Container registry is safe to change
}


def filter_configuration_changes(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Filter configuration changes to only allow whitelisted safe options.

    Returns a dictionary of allowed configuration changes.
    """
    # For simplicity and safety, we only allow explicitly whitelisted options
    # that are not None (i.e., were actually provided by the user)
    allowed_changes = {}

    # Check each argument
    for attr_name in vars(args):
        if attr_name.startswith('_') or attr_name == 'func':
            continue

        new_value = getattr(args, attr_name)

        # Skip if value is None (not set by user)
        if new_value is None:
            continue

        # Check if this option is whitelisted
        if attr_name in CONFIG_WHITELIST:
            allowed_changes[attr_name] = new_value

    return allowed_changes


def parse_args_with_filter(args: argparse.Namespace, filtered_changes: Dict[str, Any]):
    """
    Parse arguments with filtered configuration changes.

    This function reuses the logic from start.py but applies whitelist filtering.
    """

    config_data = {}

    # Handle config file if provided
    if args.config_file:
        yaml_data = load_config_from_yaml(args.config_file)
        # Apply whitelist filtering to YAML config
        filtered_yaml_data = filter_whitelisted_yaml_config(yaml_data)
        config_data.update(filtered_yaml_data)

    # Apply filtered command line changes (these override config file)
    for key, value in filtered_changes.items():
        config_data[key] = value

    # Create config with filtered data - only use the filtered config data
    # Don't call set_common_options/set_server_options/set_worker_options
    # as they would re-apply all command line arguments including blocked ones
    try:
        cfg = Config(**config_data)
    except Exception as e:
        raise Exception(f"Config error: {e}")

    return cfg
