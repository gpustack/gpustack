from enum import Enum
from typing import Optional
from pydantic import BaseModel


class GatewayModeEnum(str, Enum):
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


class PredefinedConfig(BaseModel):
    # Common options
    debug: bool = False
    cache_dir: Optional[str] = None
    log_dir: Optional[str] = None
    bin_dir: Optional[str] = None
    system_default_container_registry: Optional[str] = None
    image_name_override: Optional[str] = None
    image_repo: str = "gpustack/gpustack"
    gateway_mode: GatewayModeEnum = GatewayModeEnum.auto
    gateway_kubeconfig: Optional[str] = None
    gateway_concurrency: int = 16
    service_discovery_name: Optional[str] = None
    namespace: Optional[str] = None
    huggingface_token: Optional[str] = None

    # Worker options
    disable_worker_metrics: bool = False
    worker_port: int = 10150
    worker_metrics_port: int = 10151
    service_port_range: Optional[str] = "40000-40063"
    ray_port_range: Optional[str] = "41000-41999"
    resources: Optional[dict] = None
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
    image_repo: Optional[str] = None
    gateway_mode: Optional[str] = None
    gateway_concurrency: Optional[int] = None
