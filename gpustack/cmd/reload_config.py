import argparse
import logging
import sys
import requests
from typing import Dict, Any

from gpustack import __version__, __git_commit__
from gpustack.cmd.start import load_config_from_yaml
from gpustack.config.config import Config
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
        help="API Key to authenticate as admin.",
        default=get_gpustack_env("API_KEY"),
    )

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


def apply_runtime_updates(
    payload: Dict[str, Any],
    args: argparse.Namespace,
):
    api_key = getattr(args, "api_key", None)
    server_port = getattr(args, "server_port") or 30080
    worker_port = getattr(args, "worker_port") or 10150
    urls = [
        f"http://127.0.0.1:{server_port}{default_versioned_prefix}/config",
        f"http://127.0.0.1:{worker_port}{default_versioned_prefix}/config",
    ]
    for url in urls:
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            resp = requests.put(url, json=payload, headers=headers)
            if resp.status_code == 200:
                logger.info(f"Applied runtime config via {url}")
            else:
                logger.warning(f"Failed to apply config via {url}: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Failed to apply config via {url}: {e}")


def list_runtime_values(
    api_key: str | None = None,
    server_port: int | None = None,
    worker_port: int | None = None,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    s_port = server_port or 30080
    w_port = worker_port or 10150
    endpoints = {
        "server": f"http://127.0.0.1:{s_port}{default_versioned_prefix}/config",
        "worker": f"http://127.0.0.1:{w_port}{default_versioned_prefix}/config",
    }
    for scope, url in endpoints.items():
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            resp = requests.get(url, timeout=2, headers=headers)
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
    runtime_values = list_runtime_values(
        api_key=getattr(args, "api_key", None),
        server_port=getattr(args, "server_port", None),
        worker_port=getattr(args, "worker_port", None),
    )
    if runtime_values:
        print("Current config values:")
        for scope, conf in runtime_values.items():
            for field in sorted(WHITELIST_CONFIG_FIELDS):
                if field in conf and conf[field] is not None:
                    print(f"- {scope}: {field.replace('_', '-')} = {conf[field]}")
    return True
