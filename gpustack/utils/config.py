from typing import Any, Dict, get_type_hints, get_args, get_origin

from fastapi import Request
from transformers.hf_argparser import string_to_bool

from gpustack.config.config import Config


WHITELIST_CONFIG_FIELDS = {
    "debug",
    "system_default_container_registry",
}

READ_ONLY_CONFIG_FIELDS = WHITELIST_CONFIG_FIELDS.union(
    {
        "server_external_url",
    }
)


def _unwrap_optional(tp):
    origin = get_origin(tp)
    if origin is None:
        return tp
    args = get_args(tp)
    non_none = [a for a in args if a is not type(None)]
    return non_none[0] if non_none else tp


def coerce_value_by_field(field: str, v):
    hints = get_type_hints(Config)
    tp = hints.get(field)
    if tp is None:
        return v
    tp = _unwrap_optional(tp)
    origin = get_origin(tp)
    if tp is bool:
        return string_to_bool(v)
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
            import json

            return json.loads(v)
        return dict(v)
    return v


def filter_whitelisted_yaml_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    if not config_data:
        return config_data
    filtered_data = {}
    for key, value in config_data.items():
        config_key = key.replace('-', '_')
        if config_key in WHITELIST_CONFIG_FIELDS:
            filtered_data[config_key] = value
    return filtered_data


def is_local_request(request: Request) -> bool:
    host = request.client.host
    if host in ("127.0.0.1", "::1"):
        return True
    return False
