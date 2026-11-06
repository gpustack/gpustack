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


def runtime_metric_labels(native_labels: dict, platform_labels: dict) -> dict:
    """Keep platform identity authoritative without discarding native fields."""
    labels = dict(native_labels)
    for key, value in platform_labels.items():
        if key in native_labels and native_labels[key] != value:
            native_key = f"native_{key}"
            if native_key in native_labels or native_key in platform_labels:
                raise ValueError("Native metric label conflicts with platform identity")
            labels[native_key] = native_labels[key]
        labels[key] = value
    return labels
