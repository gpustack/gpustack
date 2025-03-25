import argparse
import asyncio
import json
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional

import yaml

from gpustack.config.config import set_global_config
from gpustack.logging import setup_logging
from gpustack.worker.worker import Worker
from gpustack.config import Config
from gpustack.server.server import Server


logger = logging.getLogger(__name__)


class OptionalBoolAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(OptionalBoolAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs
        )
        self.default = None

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def setup_start_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "start",
        help="Run GPUStack server or worker.",
        description="Run GPUStack server or worker.",
    )
    group = parser_server.add_argument_group("Common settings")
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
        "-t",
        "--token",
        type=str,
        help="Shared secret used to add a worker.",
        default=get_gpustack_env("TOKEN"),
    )
    group.add_argument(
        "--huggingface-token",
        type=str,
        help="User Access Token to authenticate to the Hugging Face Hub.",
        default=os.getenv("HF_TOKEN"),
    )
    group.add_argument(
        "--enable-ray",
        action=OptionalBoolAction,
        help="Enable Ray.",
        default=get_gpustack_env_bool("ENABLE_RAY"),
    )
    group.add_argument(
        "--ray-args",
        action='append',
        help="Arguments to pass to Ray.",
    )

    group = parser_server.add_argument_group("Server settings")
    group.add_argument(
        "--host",
        type=str,
        help="Host to bind the server to.",
        default=get_gpustack_env("HOST"),
    )
    group.add_argument(
        "--port",
        type=int,
        help="Port to bind the server to.",
        default=get_gpustack_env("PORT"),
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
        "--ollama-library-base-url",
        type=str,
        help="Base URL of the Ollama library. The default is https://registry.ollama.ai.",
        default=get_gpustack_env("OLLAMA_LIBRARY_BASE_URL"),
    )
    group.add_argument(
        "--disable-update-check",
        action=OptionalBoolAction,
        help="Disable update check.",
        default=get_gpustack_env_bool("DISABLE_UPDATE_CHECK"),
    )
    group.add_argument(
        "--update-check-url",
        type=str,
        help=argparse.SUPPRESS,
        default=get_gpustack_env("UPDATE_CHECK_URL"),
    )
    group.add_argument(
        "--model-catalog-file",
        type=str,
        help="Path or URL to the model catalog file.",
        default=get_gpustack_env("MODEL_CATALOG_FILE"),
    )
    group.add_argument(
        "--ray-port",
        type=int,
        help="Port of Ray (GCS server). Used when Ray is enabled. The default is 40096.",
        default=get_gpustack_env("RAY_PORT"),
    )
    group.add_argument(
        "--ray-client-server-port",
        type=int,
        help="Port of Ray Client Server. Used when Ray is enabled. The default is 40097.",
        default=get_gpustack_env("RAY_CLIENT_SERVER_PORT"),
    )

    group = parser_server.add_argument_group("Worker settings")
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
        "--rpc-server-port-range",
        type=str,
        help="Port range for RPC servers, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive. The default is '40064-40095'.",
        default=get_gpustack_env("RPC_SERVER_PORT_RANGE"),
    )
    group.add_argument(
        "--ray-node-manager-port",
        type=int,
        help="Port of Ray node manager. Used when Ray is enabled. The default is 40098.",
        default=get_gpustack_env("RAY_NODE_MANAGER_PORT"),
    )
    group.add_argument(
        "--ray-object-manager-port",
        type=int,
        help="Port of Ray object manager. Used when Ray is enabled. The default is 40099.",
        default=get_gpustack_env("RAY_OBJECT_MANAGER_PORT"),
    )
    group.add_argument(
        "--ray-worker-port-range",
        type=str,
        help="Port range for Ray worker processes, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive. The default is '40100-40131'.",
        default=get_gpustack_env("RAY_WORKER_PORT_RANGE"),
    )
    group.add_argument(
        "--disable-metrics",
        action=OptionalBoolAction,
        help="Disable metrics.",
        default=get_gpustack_env_bool(
            "DISABLE_METRICS",
        ),
    )
    group.add_argument(
        "--disable-rpc-servers",
        action=OptionalBoolAction,
        help="Disable RPC servers.",
        default=get_gpustack_env_bool(
            "DISABLE_METRICS",
        ),
    )
    group.add_argument(
        "--metrics-port",
        type=int,
        help="Port to expose metrics.",
        default=get_gpustack_env("METRICS_PORT"),
    )
    group.add_argument(
        "--log-dir",
        type=str,
        help="Directory to store logs.",
        default=get_gpustack_env("LOG_DIR"),
    )
    group.add_argument(
        "--rpc-server-args",
        action='append',
        help="Arguments to pass to the RPC servers.",
    )
    group.add_argument(
        "--system-reserved",
        type=json.loads,
        help="The system reserves resources during scheduling, measured in GiB. \
        Where RAM is reserved per worker, and VRAM is reserved per GPU device. \
        By default, 2 GiB of RAM and 1G of VRAM is reserved. \
        Example: '{\"ram\": 2, \"vram\": 1}' or '{\"memory\": 2, \"gpu_memory\": 1}', \
        Note: The 'memory' and 'gpu_memory' keys are deprecated and will be removed in future releases.",
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
        default=os.getenv("HF_HUB_ENABLE_HF_TRANSFER"),
    )
    group.add_argument(
        "--enable-cors",
        action=OptionalBoolAction,
        help="Enable CORS in server.",
        default=get_gpustack_env_bool("ENABLE_CORS"),
    )
    group.add_argument(
        "--allow-origins",
        action='append',
        help="A list of origins that should be permitted to make cross-origin requests.",
    )
    group.add_argument(
        "--allow-credentials",
        action=OptionalBoolAction,
        help="Indicate that cookies should be supported for cross-origin requests.",
        default=get_gpustack_env_bool("ALLOW_CREDENTIALS"),
    )
    group.add_argument(
        "--allow-methods",
        action='append',
        help="A list of HTTP methods that should be allowed for cross-origin requests.",
    )
    group.add_argument(
        "--allow-headers",
        action='append',
        help="A list of HTTP request headers that should be supported for cross-origin requests.",
    )

    parser_server.set_defaults(func=run)


def run(args: argparse.Namespace):
    try:
        cfg = parse_args(args)
        setup_logging(cfg.debug)
        debug_env_info()
        set_third_party_env(cfg=cfg)
        multiprocessing.set_start_method('spawn')

        if cfg.server_url:
            run_worker(cfg)
        else:
            run_server(cfg)
    except Exception as e:
        logger.fatal(e)


def run_server(cfg: Config):
    sub_processes = []

    if not cfg.disable_worker:
        scheme = "http://"
        if cfg.ssl_certfile:
            scheme = "https://"
        cfg.server_url = (
            f"{scheme}127.0.0.1:{cfg.port}" if cfg.port else f"{scheme}127.0.0.1"
        )
        worker = Worker(cfg, is_embedded=True)
        worker_process = multiprocessing.Process(target=worker.start)
        sub_processes = [worker_process]

    server = Server(config=cfg, sub_processes=sub_processes)

    try:
        asyncio.run(server.start())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as e:
        logger.error(f"Error running server: {e}")
    finally:
        logger.info("Server has shut down.")


def run_worker(cfg: Config):
    worker = Worker(cfg)

    worker.start()


def load_config_from_yaml(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def parse_args(args: argparse.Namespace) -> Config:
    config_data = {}
    if args.config_file:
        config_data.update(load_config_from_yaml(args.config_file))

    # CLI args have higher priority than config file
    set_common_options(args, config_data)
    set_server_options(args, config_data)
    set_worker_options(args, config_data)

    try:
        cfg = Config(**config_data)
    except Exception as e:
        raise Exception(f"Config error: {e}")

    set_global_config(cfg)
    return cfg


def set_config_option(args, config_data: dict, option_name: str):
    option_value = getattr(args, option_name, None)
    if option_value is not None:
        config_data[option_name] = option_value


def set_common_options(args, config_data: dict):
    options = [
        "debug",
        "data_dir",
        "cache_dir",
        "bin_dir",
        "pipx_path",
        "token",
        "huggingface_token",
        "enable_ray",
        "ray_args",
    ]

    for option in options:
        set_config_option(args, config_data, option)


def set_server_options(args, config_data: dict):
    options = [
        "host",
        "port",
        "database_url",
        "disable_worker",
        "bootstrap_password",
        "ssl_keyfile",
        "ssl_certfile",
        "force_auth_localhost",
        "ollama_library_base_url",
        "disable_update_check",
        "update_check_url",
        "model_catalog_file",
        "ray_port",
        "ray_client_server_port",
        "enable_cors",
        "allow_origins",
        "allow_credentials",
        "allow_methods",
        "allow_headers",
    ]

    for option in options:
        set_config_option(args, config_data, option)


def set_worker_options(args, config_data: dict):
    options = [
        "server_url",
        "worker_ip",
        "worker_name",
        "worker_port",
        "disable_metrics",
        "disable_rpc_servers",
        "metrics_port",
        "service_port_range",
        "rpc_server_port_range",
        "log_dir",
        "rpc_server_args",
        "system_reserved",
        "tools_download_base_url",
        "ray_node_manager_port",
        "ray_object_manager_port",
        "ray_worker_port_range",
        "enable_hf_transfer",
    ]

    for option in options:
        set_config_option(args, config_data, option)


def get_gpustack_env(env_var: str) -> Optional[str]:
    env_name = "GPUSTACK_" + env_var
    return os.getenv(env_name)


def get_gpustack_env_bool(env_var: str) -> Optional[bool]:
    env_name = "GPUSTACK_" + env_var
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value.lower() in ["true", "True"]
    return None


def debug_env_info():
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        logger.debug(f"Using HF_ENDPOINT: {hf_endpoint}")


def set_third_party_env(cfg: Config):
    if cfg.enable_hf_transfer:
        # https://huggingface.co/docs/huggingface_hub/guides/download#faster-downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.debug("set env HF_HUB_ENABLE_HF_TRANSFER=1")
