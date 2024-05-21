import argparse
from ..utils import get_first_non_loopback_ip
from ..core.config import configs


def setup_agent_cmd(subparsers):
    parser_agent: argparse.ArgumentParser = subparsers.add_parser(
        "agent",
        help="Run node agent.",
        description="Run node agent.",
    )
    group = parser_agent.add_argument_group("Cluster settings")
    group.add_argument(
        "--server",
        type=str,
        help="Server to connect to.",
    )
    group = parser_agent.add_argument_group("Node settings")
    group.add_argument(
        "--node-ip",
        type=str,
        help="IP address of the node. Auto-detected by default.",
    )
    parser_agent.set_defaults(func=run_agent)


def run_agent(args):
    set_configs(args)


def set_configs(args):
    if args.address:
        configs.address = args.address
    else:
        raise ValueError("Address of the head node is required.")

    if args.node_ip_address:
        configs.node_ip_address = args.node_ip_address
    else:
        configs.node_ip_address = get_first_non_loopback_ip()
