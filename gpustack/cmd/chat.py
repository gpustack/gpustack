import argparse
from gpustack.chat.manager import ChatManager, parse_arguments


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

    parser.set_defaults(func=run)


def run(args):
    cfg = parse_arguments(args)
    ChatManager(cfg).start()
