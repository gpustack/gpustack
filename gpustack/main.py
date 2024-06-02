import argparse
import signal
import sys

from gpustack import logging
from gpustack.cmd.server import setup_server_cmd
from gpustack.cmd.agent import setup_agent_cmd


def handle_signal(sig, frame):
    pass
    # sys.exit(0)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

logging.setup_logging()


def main():
    parser = argparse.ArgumentParser(
        description="GPUStack",
        conflict_handler="resolve",
        add_help=True,
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=55, indent_increment=2, width=200
        ),
    )
    subparsers = parser.add_subparsers(help="sub-command help")

    setup_server_cmd(subparsers)
    setup_agent_cmd(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
