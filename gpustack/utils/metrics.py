from typing import Optional
from gpustack.utils.compat_importlib import pkg_resources
import yaml


def get_builtin_metrics_config() -> dict:
    metrics_config_file_path = get_builtin_metrics_config_file_path()
    with open(metrics_config_file_path, "r") as f:
        raw_data = yaml.safe_load(f)
        return raw_data


def get_builtin_metrics_config_file_path() -> str:
    metrics_config_file_name = "metrics_config.yaml"
    metrics_config_file_path = str(
        pkg_resources.files("gpustack.assets.metrics_config").joinpath(
            metrics_config_file_name
        )
    )
    return metrics_config_file_path


def get_runtime_metrics_config(metrics_config: dict, runtime: str) -> Optional[dict]:
    return metrics_config.get("runtime_mapping", {}).get(runtime)
