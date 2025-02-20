import argparse
import sys
from gpustack.cli.base import parse_arguments
from gpustack.cli.chat import ChatCLIClient


def setup_chat_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "chat",
        help="Chat with a large language model.",
        description="Chat with a large language model.",
    )

    parser.add_argument("model", type=str, help="The model to use for chat")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        type=str,
        help="The prompt to send to the model",
    )
    parser.add_argument(
        "--system-prompt",
        nargs="?",
        type=str,
        default=None,
        help="The system prompt to send to the model",
    )
    parser.add_argument(
        "--user-prompt",
        nargs="?",
        type=str,
        default=None,
        help="The user prompt to send to the model",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    parser.set_defaults(func=run)


def run(args):
    try:
        cfg = parse_arguments(args)
        ChatCLIClient(cfg).run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        sys.exit(1)
