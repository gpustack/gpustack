from ..utils import get_first_non_loopback_ip
from ..core.config import configs


def setup_join_cmd(subparsers):
    parser_join = subparsers.add_parser(
        "join",
        help="Join GPUStack as a worker node.",
        description="Join GPUStack as a worker node.",
    )
    group = parser_join.add_argument_group("Connection settings")
    group.add_argument(
        "--address",
        type=str,
        help="Address of the head node. Example: 192.168.0.1:6379",
    )
    group.add_argument(
        "--node-ip-address",
        type=str,
        help="IP address of the node. Auto-detected by default.",
    )
    parser_join.set_defaults(func=run)


def run(args):
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
