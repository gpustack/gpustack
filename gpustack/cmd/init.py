import uvicorn
import uvicorn.config


from ..core.app import app
from ..core.config import configs
from ..core.db import init_db
from ..logging import uvicorn_log_config
from ..utils import get_first_non_loopback_ip


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
    parser_init.set_defaults(func=run)


def run(args):
    set_configs(args)

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

    init_db()


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
