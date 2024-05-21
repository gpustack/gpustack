import argparse
import uvicorn
import uvicorn.config


from gpustack.server.server import Server
from gpustack.server.config import ServerConfig
from gpustack.server.db import init_db
from gpustack.logging import uvicorn_log_config
from gpustack.utils import get_first_non_loopback_ip


def setup_server_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "server", help="Run management server.", description="Run management server."
    )
    group = parser_server.add_argument_group("Basic settings")
    group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
        default=True,
    )
    group.add_argument(
        "--model",
        type=str,
        help="ID of a huggingface model to serve on start. Example: Qwen/Qwen1.5-1.8B-Chat",
        default="Qwen/Qwen1.5-1.8B-Chat",
    )

    group = parser_server.add_argument_group("Node settings")
    group.add_argument(
        "--node-ip",
        type=str,
        help="IP address of the node. Auto-detected by default.",
    )

    group = parser_server.add_argument_group("Data settings")
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

    parser_server.set_defaults(func=run_server)


def run_server(args):
    cfg = to_config(args)

    server = Server(cfg)

    server.start()


def to_config(args) -> ServerConfig:
    cfg = ServerConfig()
    if args.model:
        cfg.model = args.model

    if args.debug:
        cfg.debug = args.debug

    if args.node_ip:
        cfg.node_ip = args.node_ip
    else:
        cfg.node_ip = get_first_non_loopback_ip()

    if args.data_dir:
        cfg.data_dir = args.data_dir

    if args.database_url:
        cfg.database_url = args.database_url
    else:
        cfg.database_url = f"sqlite:///{cfg.data_dir}/database.db"

    return cfg
