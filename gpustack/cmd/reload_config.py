import argparse
import logging
import sys
import requests
from typing import Dict, Any

from gpustack import __version__, __git_commit__
from gpustack.api.auth import SESSION_COOKIE_NAME
from gpustack.cmd.local_auth import (
    add_local_auth_arguments,
    mint_admin_jwt,
    read_local_jwt_secret,
)
from gpustack.cmd.start import load_config_from_yaml
from gpustack.config.config import Config
from gpustack.config.registration import read_worker_token
from gpustack.utils.envs import get_gpustack_env
from gpustack.utils.config import (
    WHITELIST_CONFIG_FIELDS,
    coerce_value_by_field,
    filter_whitelisted_yaml_config,
)
from gpustack.logging import setup_logging
from gpustack.client.generated_http_client import default_versioned_prefix

logger = logging.getLogger(__name__)


class OptionalBoolAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(OptionalBoolAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs
        )
        self.default = kwargs.get("default")

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def setup_reload_config_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "reload-config",
        help="Reload GPUStack configuration.",
        description=("Reload GPUStack configuration via --set, --file, or --list."),
    )

    parser.add_argument(
        "--set",
        action="append",
        help=(
            "Set a single configuration value: --set key=value (key in hyphen-case). "
            "Only whitelisted fields are applied. "
            "Values are coerced by target field type. "
            "Lists accept comma-separated strings "
            "(e.g., --set allow-origins=https://a.com,https://b.com). "
            "Dicts require JSON string "
            "(e.g., --set system-reserved='{\"ram\":2,\"vram\":1}'). "
            "Invalid JSON will cause an error and exit."
        ),
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Load configuration from YAML file: --file /path/to/gpustack_config.yaml",
    )
    parser.add_argument(
        "--list",
        action=OptionalBoolAction,
        help=(
            "List whitelisted fields that can be updated, can't use with --set or --file."
        ),
        default=False,
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Admin API key to authenticate to the server endpoint. Note the "
        "worker endpoint does not accept an admin API key; it uses the local "
        "worker token.",
        default=get_gpustack_env("API_KEY"),
    )
    add_local_auth_arguments(parser)

    parser.add_argument(
        "--server-port",
        type=int,
        help="Port of the GPUStack API server to target.",
        default=get_gpustack_env("API_PORT"),
    )
    parser.add_argument(
        "--worker-port",
        type=int,
        help="Port of the GPUStack worker to target.",
        default=get_gpustack_env("WORKER_PORT"),
    )

    parser.set_defaults(func=run)


def run(args):
    try:
        logger.info("Starting configuration reload...")
        logger.info(f"GPUStack version: {__version__} ({__git_commit__})")
        if handle_list_mode(args):
            return

        cfg = parse_args_with_filter(args, {})
        payload = {}
        for field in WHITELIST_CONFIG_FIELDS:
            if hasattr(cfg, field):
                value = getattr(cfg, field)
                if value is not None:
                    payload[field] = value

        setup_logging(cfg.debug)
        apply_runtime_updates(payload, args)
        display_config_summary(cfg)

    except Exception as e:
        logger.error(f"Failed to reload configuration: {e}")
        sys.exit(1)


def display_config_summary(cfg):
    """Display a summary of the reloaded configuration - only show whitelisted fields."""
    logger.info("=== Configuration Reload Summary ===")

    for field in WHITELIST_CONFIG_FIELDS:
        if hasattr(cfg, field):
            value = getattr(cfg, field)
            if value is not None:
                logger.info(f"- reload: {field} = {value}")
    logger.info("Configuration successfully reloaded.")

    logger.info("=====================================")


def parse_args_with_filter(args: argparse.Namespace, filtered_changes: Dict[str, Any]):
    """
    Parse arguments with filtered configuration changes.

    This function reuses the logic from start.py but applies whitelist filtering.
    """

    config_data = {}

    # Handle config file if provided
    if getattr(args, "file", None):
        yaml_data = load_config_from_yaml(args.file)
        filtered_yaml_data = filter_whitelisted_yaml_config(yaml_data or {})
        config_data.update(filtered_yaml_data)

    if getattr(args, "set", None):
        for item in args.set:
            if "=" not in item:
                raise Exception(f"Invalid --set value: {item}. Use key=value")
            k, v = item.split("=", 1)
            key = k.replace("-", "_")
            if key in WHITELIST_CONFIG_FIELDS:
                config_data[key] = coerce_value_by_field(key, v)

    # Apply filtered command line changes (these override config file)
    for key, value in filtered_changes.items():
        config_data[key] = value

    # Create config with filtered data - only use the filtered config data
    # Don't call set_common_options/set_server_options/set_worker_options
    # as they would re-apply all command line arguments including blocked ones
    return Config(**config_data)


def resolve_scope_headers(args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    """Build per-endpoint auth headers from credentials available on this host.

    The command targets two endpoints that enforce different auth mechanisms,
    so each gets its own credential. We do NOT gate endpoints on a role-marker
    file: markers such as ``bootstrap_version`` are absent on legacy (<= v2.0.0)
    installs, so role-guessing would silently skip the server reload on an
    upgraded combined node. Instead we attach whatever credential each endpoint
    accepts and let a ConnectionError (endpoint not running here) decide what to
    skip — see ``apply_runtime_updates``.

      - server (get_admin_user): an explicit --api-key, or a short-lived admin
        JWT minted from the locally readable jwt_secret_key, sent as a session
        cookie.
      - worker (worker_auth): the local worker token sent as a bearer token.
        An admin --api-key is NOT accepted here, so it is only used as a
        fallback when no local worker token exists.
    """
    data_dir = getattr(args, "data_dir", None) or Config.get_data_dir()
    api_key = getattr(args, "api_key", None)

    headers: Dict[str, Dict[str, str]] = {}

    # Server endpoint: an admin API key works via the bearer path, otherwise
    # mint a short-lived admin JWT from the local jwt_secret_key.
    if api_key:
        headers["server"] = {"Authorization": f"Bearer {api_key}"}
    else:
        jwt_secret = read_local_jwt_secret(args)
        if jwt_secret:
            jwt_token = mint_admin_jwt(
                jwt_secret, getattr(args, "admin_username", "admin")
            )
            headers["server"] = {"Cookie": f"{SESSION_COOKIE_NAME}={jwt_token}"}

    # Worker endpoint: worker_auth only accepts the worker/registration token,
    # not an admin API key — prefer the local worker token and fall back to
    # --api-key only when none is present.
    worker_token = read_worker_token(data_dir)
    if worker_token:
        headers["worker"] = {"Authorization": f"Bearer {worker_token}"}
    elif api_key:
        headers["worker"] = {"Authorization": f"Bearer {api_key}"}

    return headers


def record_runtime_update_response(
    url: str,
    status_code: int,
    failures: list[str],
    auth_failures: list[str],
) -> bool:
    if status_code == 200:
        logger.info(f"Applied runtime config via {url}")
        return True
    logger.warning(f"Failed to apply config via {url}: {status_code}")
    if status_code in (401, 403):
        auth_failures.append(f"{url}: HTTP {status_code}")
    else:
        failures.append(f"{url}: HTTP {status_code}")
    return False


def apply_runtime_updates(
    payload: Dict[str, Any],
    args: argparse.Namespace,
):
    server_port = getattr(args, "server_port", None) or 30080
    worker_port = getattr(args, "worker_port", None) or 10150
    scope_headers = resolve_scope_headers(args)
    if not scope_headers:
        raise Exception(
            "No credential available to authenticate to the local config "
            "endpoints. Provide --api-key, or run this command on the server "
            "host (reads <data-dir>/jwt_secret_key) or a worker host (reads "
            "<data-dir>/worker_token); use --data-dir to point at the data "
            "directory."
        )
    endpoints = {
        "server": f"http://127.0.0.1:{server_port}{default_versioned_prefix}/config",
        "worker": f"http://127.0.0.1:{worker_port}{default_versioned_prefix}/config",
    }
    applied = False
    failures: list[str] = []
    auth_failures: list[str] = []
    for scope, url in endpoints.items():
        if scope not in scope_headers:
            # No credential for this endpoint's auth scheme. Skip rather than
            # emit a misleading 401.
            logger.debug(f"Skipping {scope} config endpoint: no local credential")
            continue
        try:
            resp = requests.put(
                url, json=payload, headers=scope_headers[scope], timeout=5
            )
            applied = (
                record_runtime_update_response(
                    url, resp.status_code, failures, auth_failures
                )
                or applied
            )
        except requests.exceptions.ConnectionError:
            # This endpoint's role (server/worker) isn't running on this host —
            # e.g. on a worker host where the server port is not bound. Expected;
            # not a failure.
            logger.debug(f"Skipping {scope} config endpoint at {url}: not reachable")
        except Exception as e:
            logger.warning(f"Failed to apply config via {url}: {e}")
            failures.append(f"{url}: {e}")

    if failures:
        raise Exception("Failed to apply runtime config to: " + "; ".join(failures))
    if not applied:
        if auth_failures:
            raise Exception(
                "Failed to apply runtime config to: " + "; ".join(auth_failures)
            )
        raise Exception(
            "No reachable config endpoint accepted the update. Ensure the "
            "gpustack server or worker is running on this host, or target it "
            "with --server-port/--worker-port."
        )


def list_runtime_values(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    s_port = getattr(args, "server_port", None) or 30080
    w_port = getattr(args, "worker_port", None) or 10150
    scope_headers = resolve_scope_headers(args)
    endpoints = {
        "server": f"http://127.0.0.1:{s_port}{default_versioned_prefix}/config",
        "worker": f"http://127.0.0.1:{w_port}{default_versioned_prefix}/config",
    }
    for scope, url in endpoints.items():
        if scope not in scope_headers:
            continue
        try:
            resp = requests.get(url, timeout=2, headers=scope_headers[scope])
            if resp.status_code == 200:
                results[scope] = resp.json()
        except Exception:
            continue
    return results


def handle_list_mode(args) -> bool:
    if not getattr(args, "list", False):
        return False
    print("Whitelisted fields:")
    for field in sorted(WHITELIST_CONFIG_FIELDS):
        print(f"- {field.replace('_', '-')}")
    runtime_values = list_runtime_values(args)
    if runtime_values:
        print("Current config values:")
        for scope, conf in runtime_values.items():
            for field in sorted(WHITELIST_CONFIG_FIELDS):
                if field in conf and conf[field] is not None:
                    print(f"- {scope}: {field.replace('_', '-')} = {conf[field]}")
    return True
