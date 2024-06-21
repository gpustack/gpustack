import argparse
import asyncio
import multiprocessing

from gpustack.worker.worker import Worker
from gpustack.config import Config
from gpustack.server.server import Server


def setup_start_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "start",
        help="Run GPUStack server or worker.",
        description="Run GPUStack server or worker.",
    )
    group = parser_server.add_argument_group("Common settings")
    group.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode.",
        default=True,
    )
    group.add_argument(
        "--data-dir",
        type=str,
        help="Directory to store data. Default is OS specific.",
    )
    group.add_argument(
        "-t",
        "--token",
        type=str,
        help="Shared secret used to add a worker.",
    )

    group = parser_server.add_argument_group("Server settings")
    group.add_argument(
        "--database-url",
        type=str,
        help="URL of the database. Example: postgresql://user:password@hostname:port/db_name",
    )
    group.add_argument(
        "--disable-worker",
        action="store_true",
        help="Disable embedded worker.",
        default=False,
    )
    group.add_argument(
        "--serve-default-models",
        action="store_true",
        help="Serve default models on bootstrap.",
        default=False,
    )
    group.add_argument(
        "--bootstrap-password",
        type=str,
        help="Initial password for the default admin user. Random by default.",
    )

    group = parser_server.add_argument_group("Worker settings")
    group.add_argument(
        "-s",
        "--server-url",
        type=str,
        help="Server to connect to.",
    )
    group.add_argument(
        "--node-ip",
        type=str,
        help="IP address of the node. Auto-detected by default.",
    )
    group.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Enable metrics.",
        default=True,
    )
    group.add_argument(
        "--metrics-port",
        type=int,
        help="Port to expose metrics.",
        default=10051,
    )
    group.add_argument(
        "--log-dir",
        type=str,
        help="Directory to store logs.",
    )

    parser_server.set_defaults(func=run)


def run(args):
    cfg = parse_args(args)
    if cfg.server_url:
        run_worker(cfg)
    else:
        run_server(cfg)


def run_server(cfg: Config):
    sub_processes = []

    if not cfg.disable_worker:
        cfg.server_url = "http://127.0.0.1"
        worker = Worker(cfg)
        worker_process = multiprocessing.Process(target=worker.start, args=(True,))
        sub_processes = [worker_process]

    server = Server(config=cfg, sub_processes=sub_processes)

    asyncio.run(server.start())


def run_worker(cfg: Config):
    worker = Worker(cfg)

    worker.start()


def parse_args(args) -> Config:
    cfg = Config(
        debug=args.debug,
        data_dir=args.data_dir,
        token=args.token,
        database_url=args.database_url,
        disable_worker=args.disable_worker,
        serve_default_models=args.serve_default_models,
        bootstrap_password=args.bootstrap_password,
        server_url=args.server_url,
        node_ip=args.node_ip,
        enable_metrics=args.enable_metrics,
        metrics_port=args.metrics_port,
        log_dir=args.log_dir,
    )

    return cfg
