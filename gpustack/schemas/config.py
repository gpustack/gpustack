from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from gpustack import __benchmark_runner_version__


class GatewayModeEnum(str, Enum):
    """
    For both server and worker gateway mode
    auto - automatically detect the gateway mode
    embedded - use the embedded gateway(server)
    incluster - connect to an in-cluster gateway(server)
    external - connect to an external gateway(server)
    disabled - disable the gateway(server/worker)

    The support of worker gateway mode is reserved for future use.
    Only incluster and external modes are supported for worker.
    """

    auto = "auto"
    embedded = "embedded"
    incluster = "incluster"
    external = "external"
    disabled = "disabled"


class ModelInstanceProxyModeEnum(str, Enum):
    """
    Enum for Model Instance Proxy Mode
    WORKER - Proxy through the worker
    DIRECT - Direct access to the model instance
    DELEGATED - Preserved for proxying through cluster gateway (not implemented yet)
    """

    WORKER = "worker"
    DIRECT = "direct"
    DELEGATED = "delegated"


class SensitivePredefinedConfig(BaseModel):
    # Common options
    huggingface_token: Optional[str] = Field(
        default=None, json_schema_extra={"env_var": "HF_TOKEN"}
    )


class PredefinedConfig(SensitivePredefinedConfig):
    # Common options
    debug: bool = False
    cache_dir: Optional[str] = None
    log_dir: Optional[str] = None
    bin_dir: Optional[str] = None
    benchmark_dir: Optional[str] = None
    system_default_container_registry: Optional[str] = None
    image_name_override: Optional[str] = None
    image_repo: str = "gpustack/gpustack"
    benchmark_image_repo: str = (
        f"gpustack/benchmark-runner:{__benchmark_runner_version__}"
    )
    gateway_mode: GatewayModeEnum = GatewayModeEnum.auto
    gateway_kubeconfig: Optional[str] = None
    gateway_namespace: str = "higress-system"
    service_discovery_name: Optional[str] = None
    namespace: str = "gpustack-system"

    # Worker options
    disable_worker_metrics: bool = False
    worker_port: int = 10150
    worker_metrics_port: int = 10151
    service_port_range: Optional[str] = "40000-40063"
    ray_port_range: Optional[str] = "41000-41999"
    benchmark_max_duration_seconds: Optional[int] = None
    system_reserved: Optional[dict] = None
    pipx_path: Optional[str] = None
    tools_download_base_url: Optional[str] = None
    enable_hf_transfer: bool = False
    enable_hf_xet: bool = False
    proxy_mode: Optional[ModelInstanceProxyModeEnum] = None


class PredefinedConfigNoDefaults(PredefinedConfig):
    debug: Optional[bool] = None
    disable_worker_metrics: Optional[bool] = None
    enable_hf_transfer: Optional[bool] = None
    enable_hf_xet: Optional[bool] = None
    worker_port: Optional[int] = None
    worker_metrics_port: Optional[int] = None
    service_port_range: Optional[str] = None
    ray_port_range: Optional[str] = None
    benchmark_max_duration_seconds: Optional[int] = None
    image_repo: Optional[str] = None
    benchmark_image_repo: Optional[str] = None
    gateway_mode: Optional[str] = None
    gateway_namespace: Optional[str] = None
    namespace: Optional[str] = None


def parse_base_model_to_env_vars(
    config: BaseModel,
) -> dict[str, str]:
    env_vars = {}
    for field_name, field in config.__class__.model_fields.items():
        extra = getattr(field, 'json_schema_extra', None) or {}
        env_var = extra.get("env_var")
        if env_var is None:
            # assuming the field name is in snake_case
            env_var = f"GPUSTACK_{field_name.upper()}"
        value = getattr(config, field_name)
        if value is not None:
            if isinstance(value, bool):
                env_vars[env_var] = "true" if value else "false"
            else:
                env_vars[env_var] = str(value)
    return env_vars
