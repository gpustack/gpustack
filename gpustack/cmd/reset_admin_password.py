import argparse
import logging
import os

import requests

from gpustack.api.auth import SESSION_COOKIE_NAME
from gpustack.client.generated_clientset import ClientSet
from gpustack.cmd.local_auth import (
    add_local_auth_arguments,
    mint_admin_jwt,
    read_local_jwt_secret,
)
from gpustack.cmd.start import get_gpustack_env
from gpustack.config.config import Config
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

    add_local_auth_arguments(parser)

    parser.set_defaults(func=run)


def run(args):
    try:
        server_url = None
        # default using api port instead of web port
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

        api_key = args.api_key

        extra_headers = None
        if api_key is None:
            # Mint a short-lived admin JWT from the
            # locally readable JWT secret so this CLI keeps working on the
            # server host without requiring an explicit API key.
            jwt_secret = _resolve_jwt_secret(args)
            jwt_token = mint_admin_jwt(jwt_secret, args.admin_username)
            # Pass the JWT as a Cookie header at construction time so every
            # sub-client (client.users, client.workers, ...) shares the same
            # authenticated HTTPClient instance. Reassigning
            # ``client.http_client`` after construction does NOT propagate to
            # the sub-clients, which keep a reference to the original.
            extra_headers = {"Cookie": f"{SESSION_COOKIE_NAME}={jwt_token}"}

        client = ClientSet(
            base_url=server_url,
            api_key=api_key,
            headers=extra_headers,
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


def _resolve_jwt_secret(args) -> str:
    # Unlike reload-config, resetting the password has no other auth path, so a
    # missing secret is fatal here — wrap the shared reader and raise a helpful
    # message pointing at the data dir.
    secret = read_local_jwt_secret(args)
    if not secret:
        data_dir = args.data_dir or Config.get_data_dir()
        jwt_secret_path = os.path.join(data_dir, "jwt_secret_key")
        raise Exception(
            "Cannot authenticate to the local gpustack server. "
            f"No JWT secret found (checked --jwt-secret-key, "
            f"GPUSTACK_JWT_SECRET_KEY, and {jwt_secret_path}). "
            "Please provide --api-key, --jwt-secret-key, or "
            "GPUSTACK_JWT_SECRET_KEY; alternatively run this command on the "
            "server host with --data-dir pointing at the server data directory."
        )
    return secret


def test_url_accessible(url: str) -> bool:
    try:
        resp = requests.get(f"{url}/healthz", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False
