import argparse
import logging
import requests

from gpustack.client.generated_clientset import ClientSet
from gpustack.cmd.start import get_gpustack_env
from gpustack.schemas.users import UserUpdate
from gpustack.security import generate_secure_password

logger = logging.getLogger(__name__)


def setup_reset_admin_password_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "reset-admin-password",
        help="Reset the admin user's password.",
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
        # default using api port instead of web port
        server_url = None
        server_urls = ["http://localhost", "http://localhost:30080"]
        if args.server_url is not None:
            server_url = args.server_url
        else:
            for url in server_urls:
                if test_url_accessible(url):
                    server_url = url
                    break
            if server_url is None:
                raise Exception(
                    "Cannot connect to local gpustack server. Please specify --server-url"
                )

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


def test_url_accessible(url: str) -> bool:
    try:
        resp = requests.get(f"{url}/healthz", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False
