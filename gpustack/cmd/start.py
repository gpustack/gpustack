import argparse
import asyncio
import json
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional

import yaml

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
        help="Directory to store data. Default is OS specific.",
        default=get_gpustack_env("DATA_DIR"),
    )
    group.add_argument(
        "-t",
        "--token",
        type=str,
        help="Shared secret used to add a worker.",
        default=get_gpustack_env("TOKEN"),
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
        help=argparse.SUPPRESS,
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
        help="Base URL of the Ollama library. Default is https://registry.ollama.ai.",
        default=get_gpustack_env("OLLAMA_LIBRARY_BASE_URL"),
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
        "--disable-metrics",
        action=OptionalBoolAction,
        help="Disable metrics.",
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
        "--worker-port",
        type=int,
        help="Port to bind the worker to.",
        default=get_gpustack_env("WORKER_PORT"),
    )
    group.add_argument(
        "--log-dir",
        type=str,
        help="Directory to store logs.",
        default=get_gpustack_env("LOG_DIR"),
    )
    group.add_argument(
        "--system-reserved",
        type=json.loads,
        help="The system reserves resources during scheduling, measured in GiB. \
        Where memory is reserved per worker, and gpu_memory is reserved per GPU device. \
        1 GiB of memory is reserved per worker and 1 GiB of GPU memory is reserved \
        per GPU device. Example: '{\"memory\": 1, \"gpu_memory\": 1}'.",
        default=get_gpustack_env("SYSTEM_RESERVED"),
    )

    parser_server.set_defaults(func=run)


def run(args: argparse.Namespace):
    try:
        cfg = parse_args(args)
        setup_logging(cfg.debug)
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
        worker = Worker(cfg)
        worker_process = multiprocessing.Process(target=worker.start, args=(True,))
        sub_processes = [worker_process]

    server = Server(config=cfg, sub_processes=sub_processes)

    asyncio.run(server.start())


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

    return Config(**config_data)


def set_config_option(args, config_data: dict, option_name: str):
    option_value = getattr(args, option_name, None)
    if option_value is not None:
        config_data[option_name] = option_value


def set_common_options(args, config_data: dict):
    options = [
        "debug",
        "data_dir",
        "token",
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
        "system_reserved",
        "ssl_keyfile",
        "ssl_certfile",
        "force_auth_localhost",
        "ollama_library_base_url",
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
        "metrics_port",
        "log_dir",
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
