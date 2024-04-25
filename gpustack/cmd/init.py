import os
import uvicorn
import uvicorn.config


from ..core.app import app
from ..core.config import configs
from ..core.db import init_db
from ..core.runtime.ray import RayRuntime
from ..core.services import start_redis_server
from ..logging import logger, uvicorn_log_config
from ..utils import get_first_non_loopback_ip, is_command_available


def setup_init_cmd(subparsers):
    parser_init = subparsers.add_parser(
        "init", help="Initialize GPUStack.", description="Initialize GPUStack."
    )
    group = parser_init.add_argument_group("Basic settings")
    group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
        default=True,
    )
    group.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode, with watchfile and reload.",
        default=False,
    )
    group.add_argument(
        "--model",
        type=str,
        help="ID of a huggingface model to serve on start. Example: Qwen/Qwen1.5-1.8B-Chat",
        default="Qwen/Qwen1.5-1.8B-Chat",
    )
    group = parser_init.add_argument_group("Connection settings")
    group.add_argument(
        "--node-ip-address",
        type=str,
        help="IP address of the node. Auto-detected by default.",
    )
    group.add_argument(
        "--data-dir",
        type=str,
        help="Directory to store data. Default is OS specific.",
    )
    group.add_argument(
        "--database-url",
        type=str,
        help="URL of the database. Example: postgresql://user:password@hostname:port/db_name",
    )
    group.add_argument(
        "--redis-url",
        type=str,
        help="URL of the redis server. Example: redis://hostname:port",
    )
    parser_init.set_defaults(func=run)


def run(args):
    set_configs(args)

    preflight_check()

    bootstrap_dependencies()

    # Start FastAPI server
    if configs.dev:
        uvicorn.run(
            "gpustack.core.app:app",
            host="0.0.0.0",
            port=80,
            log_config=uvicorn_log_config,
            reload=True,
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=80, log_config=uvicorn_log_config)


def bootstrap_dependencies():
    redis_dir = f"{configs.data_dir}/redis"
    redis_log_file = f"{redis_dir}/redis-server.log"

    if not os.path.exists(redis_dir):
        try:
            os.makedirs(redis_dir)
        except Exception as e:
            logger.error(f"Failed to create Redis directory: {e}")

    if not os.path.exists(redis_log_file):
        try:
            open(redis_log_file, "a").close()
        except Exception as e:
            logger.error(f"Failed to create Redis log file: {e}")

    start_redis_server(
        executable="redis-server",
        stderr_file=redis_log_file,
        stdout_file=redis_log_file,
    )

    init_db()

    RayRuntime()


def preflight_check():
    if not is_command_available("redis-server"):
        raise Exception(
            "CLI `redis-server` is not available. Please install it first. "
            "Note: this prerequisite will be removed in the future."
        )


def set_configs(args):
    if args.model:
        configs.model = args.model

    if args.debug:
        configs.debug = args.debug

    if args.dev:
        configs.dev = args.dev

    if args.node_ip_address:
        configs.node_ip_address = args.node_ip_address
    else:
        configs.node_ip_address = get_first_non_loopback_ip()

    if args.data_dir:
        configs.data_dir = args.data_dir

    if args.database_url:
        configs.database_url = args.database_url
    else:
        configs.database_url = f"sqlite:///{configs.data_dir}/database.db"

    if args.redis_url:
        configs.redis_url = args.redis_url
