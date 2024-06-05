import argparse
from gpustack.agent.agent import Agent
from gpustack.agent.config import AgentConfig
from gpustack.utils import get_first_non_loopback_ip


def setup_agent_cmd(subparsers):
    parser_agent: argparse.ArgumentParser = subparsers.add_parser(
        "agent",
        help="Run node agent.",
        description="Run node agent.",
    )
    group = parser_agent.add_argument_group("Basic settings")
    group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
        default=True,
    )
    group.add_argument(
        "--metrics-port",
        type=int,
        help="Port to expose metrics on, -1 to disable. Default is 10051.",
        default=10051,
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
    cfg = to_agent_config(args)

    agent = Agent(cfg)

    agent.start()


def to_agent_config(args) -> AgentConfig:
    cfg = AgentConfig()

    if args.debug:
        cfg.debug = args.debug

    if args.node_ip:
        cfg.node_ip = args.node_ip
    else:
        cfg.node_ip = get_first_non_loopback_ip()

    if args.server:
        cfg.server = args.server

    if args.metrics_port != -1:
        cfg.metric_enabled = True
        cfg.metrics_port = args.metrics_port
    else:
        raise ValueError("Server address is required.")

    return cfg
