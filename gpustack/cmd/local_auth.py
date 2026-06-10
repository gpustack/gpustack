"""Shared helpers for CLI commands that authenticate to a local GPUStack
server using on-disk secrets instead of an explicit API key.

Both ``reload-config`` and ``reset-admin-password`` run on the server host and
mint a short-lived admin JWT from the locally readable ``jwt_secret_key`` (or
``--jwt-secret-key`` / ``GPUSTACK_JWT_SECRET_KEY``), so the argument
definitions and the JWT-minting logic live here to avoid drift between the two
commands.
"""

import argparse
import os
from datetime import timedelta
from typing import Optional

from gpustack.config.config import Config
from gpustack.security import JWTManager
from gpustack.utils.envs import get_gpustack_env

# Admin JWTs minted locally are only used to reach the local server endpoint
# for a single request, so a short lifetime is plenty.
_ADMIN_JWT_TTL = timedelta(minutes=5)


def add_local_auth_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the data-dir / jwt-secret-key / admin-username arguments shared by
    CLI commands that authenticate to the local server via a locally minted
    admin JWT.
    """
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        help="Directory of the local gpustack data dir. When --api-key is not "
        "provided, the CLI reads locally-stored secrets from here to "
        "authenticate.",
        default=get_gpustack_env("DATA_DIR"),
    )
    parser.add_argument(
        "--jwt-secret-key",
        type=str,
        help="JWT secret key used to mint a short-lived admin token locally for "
        "the server endpoint. If unset, the CLI reads it from "
        "<data-dir>/jwt_secret_key. Prefer the GPUSTACK_JWT_SECRET_KEY "
        "environment variable over passing it on the command line, which is "
        "visible in the process list. Use this in distributed deployments where "
        "the operator passed --jwt-secret-key to the server (in which case the "
        "file is not written to disk).",
        default=get_gpustack_env("JWT_SECRET_KEY"),
    )
    parser.add_argument(
        "-u",
        "--admin-username",
        type=str,
        help="Username used as the JWT subject when authenticating to the "
        "server endpoint locally. Defaults to 'admin'. Override if the admin "
        "user was renamed; it must match an existing admin user.",
        default=get_gpustack_env("ADMIN_USERNAME") or "admin",
    )


def read_local_jwt_secret(args: argparse.Namespace) -> Optional[str]:
    """Resolve the JWT secret used to mint a local admin token.

    Returns the explicit ``--jwt-secret-key`` if given, otherwise the contents
    of ``<data-dir>/jwt_secret_key``. Returns ``None`` when no secret can be
    found so callers can decide whether that is fatal.
    """
    # Explicit secret wins — covers distributed deployments that pass
    # --jwt-secret-key to the server, in which case the secret is NOT
    # written to <data_dir>/jwt_secret_key.
    secret = getattr(args, "jwt_secret_key", None)
    if secret and secret.strip():
        return secret.strip()

    data_dir = getattr(args, "data_dir", None) or Config.get_data_dir()
    jwt_secret_path = os.path.join(data_dir, "jwt_secret_key")
    if not os.path.exists(jwt_secret_path):
        return None
    with open(jwt_secret_path, "r", encoding="utf-8") as f:
        secret = f.read().strip()
    return secret or None


def mint_admin_jwt(secret: str, admin_username: str) -> str:
    """Mint a short-lived admin JWT for authenticating to the local server."""
    return JWTManager(
        secret_key=secret,
        expires_delta=_ADMIN_JWT_TTL,
    ).create_jwt_token(admin_username)
