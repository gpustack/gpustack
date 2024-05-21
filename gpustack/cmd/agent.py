import argparse
from gpustack.agent.agent import Agent
from gpustack.agent.config import AgentConfig


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
    cfg = to_config(args)

    agent = Agent(cfg)

    agent.start()


def to_config(args) -> AgentConfig:
    if args.server:
        pass
    else:
        raise ValueError("Server address is required.")

    return AgentConfig()
