import argparse
import logging

from gpustack.client.generated_clientset import ClientSet
from gpustack.cmd.start import get_gpustack_env
from gpustack.schemas.users import UserUpdate
from gpustack.security import generate_secure_password

logger = logging.getLogger(__name__)


def setup_reset_admin_password_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "reset-admin-password",
    )
    parser.add_argument(
        "-s",
        "--server-url",
        type=str,
        help="Server to connect to.",
        default=get_gpustack_env("SERVER_URL"),
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API Key to connect server.",
        default=get_gpustack_env("API_KEY"),
    )

    parser.set_defaults(func=run)


def run(args):
    try:
        server_url = "http://localhost"
        if args.server_url is not None:
            server_url = args.server_url

        api_key = None
        if args.api_key is not None:
            api_key = args.api_key

        client = ClientSet(
            base_url=server_url,
            api_key=api_key,
        )

        user_me = client.users.get("me")
        user_update = UserUpdate(**user_me.model_dump())
        reset_password = generate_secure_password()
        user_update.password = reset_password
        user_update.require_password_change = True
        client.users.update("me", user_update)

        print(f"Reset admin password: {reset_password}")
    except Exception as e:
        logger.fatal(f"Failed to reset admin password: {e}")
