import argparse
import logging
import sys
import json
from typing import Dict, Any, get_type_hints, get_origin, get_args

from gpustack import __version__, __git_commit__
from gpustack.cmd.start import load_config_from_yaml
from gpustack.config.config import Config
from gpustack.logging import setup_logging


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

    parser.set_defaults(func=run)


def run(args):
    """Reload configuration via --set/--file/--list."""
    try:
        logger.info("Starting configuration reload...")
        logger.info(f"GPUStack version: {__version__} ({__git_commit__})")

        if getattr(args, "list", False):
            print("Whitelisted fields:")
            for field in sorted(CONFIG_WHITELIST):
                print(f"- {field.replace('_', '-')}")
            return

        cfg = parse_args_with_filter(args, {})

        setup_logging(cfg.debug)
        display_config_summary(cfg)

    except Exception as e:
        logger.error(f"Failed to reload configuration: {e}")
        sys.exit(1)


def display_config_summary(cfg):
    """Display a summary of the reloaded configuration - only show whitelisted fields."""
    logger.info("=== Configuration Reload Summary ===")

    for field in CONFIG_WHITELIST:
        if hasattr(cfg, field):
            value = getattr(cfg, field)
            if value is not None:
                logger.info(f"- reload: {field} = {value}")
    logger.info("Configuration successfully reloaded.")

    logger.info("=====================================")


def filter_whitelisted_yaml_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter YAML configuration data to only allow whitelisted options.

    Returns filtered configuration data.
    """
    if not config_data:
        return config_data

    filtered_data = {}

    for key, value in config_data.items():
        config_key = key.replace('-', '_')
        if config_key in CONFIG_WHITELIST:
            filtered_data[config_key] = value
            logger.info(f"Allowing YAML configuration: {config_key} = {value}")

    return filtered_data


# Configuration whitelist - only these options can be modified via reload-config
CONFIG_WHITELIST = {
    'debug',  # Log level is safe to change
    'system_default_container_registry',  # Container registry is safe to change
}


def filter_configuration_changes(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Filter configuration changes to only allow whitelisted safe options.

    Returns a dictionary of allowed configuration changes.
    """
    # For simplicity and safety, we only allow explicitly whitelisted options
    # that are not None (i.e., were actually provided by the user)
    allowed_changes = {}

    # Check each argument
    for attr_name in vars(args):
        if attr_name.startswith('_') or attr_name == 'func':
            continue

        new_value = getattr(args, attr_name)

        # Skip if value is None (not set by user)
        if new_value is None:
            continue

        # Check if this option is whitelisted
        if attr_name in CONFIG_WHITELIST:
            allowed_changes[attr_name] = new_value

    return allowed_changes


def _unwrap_optional(tp):
    origin = get_origin(tp)
    if origin is None:
        return tp
    args = get_args(tp)
    non_none = [a for a in args if a is not type(None)]
    return non_none[0] if non_none else tp


def _parse_bool(v):
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def _coerce_value_by_field(field: str, v):
    hints = get_type_hints(Config)
    tp = hints.get(field)
    if tp is None:
        return v
    tp = _unwrap_optional(tp)
    origin = get_origin(tp)
    if tp is bool:
        return _parse_bool(v)
    if tp is int:
        return int(v)
    if tp is float:
        return float(v)
    if tp is str:
        return str(v)
    if origin is list:
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return list(v)
    if tp is dict or origin is dict:
        if isinstance(v, str):
            return json.loads(v)
        return dict(v)
    return v


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
            if key in CONFIG_WHITELIST:
                config_data[key] = _coerce_value_by_field(key, v)

    # Apply filtered command line changes (these override config file)
    for key, value in filtered_changes.items():
        config_data[key] = value

    # Create config with filtered data - only use the filtered config data
    # Don't call set_common_options/set_server_options/set_worker_options
    # as they would re-apply all command line arguments including blocked ones
    return Config(**config_data)
