from typing import Optional, List


class CommonConfigMixin:
    # Common options
    debug: bool = False
    data_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    bin_dir: Optional[str] = None
    pipx_path: Optional[str] = None
    token: Optional[str] = None
    huggingface_token: Optional[str] = None
    enable_ray: bool = False
    ray_args: Optional[List[str]] = None
    ray_node_manager_port: int = 40098
    ray_object_manager_port: int = 40099


class ServerConfigCLIMixin:
    # Server options
    host: Optional[str] = "0.0.0.0"
    port: Optional[int] = None
    database_url: Optional[str] = None
    disable_worker: bool = False
    bootstrap_password: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    force_auth_localhost: bool = False
    ollama_library_base_url: Optional[str] = "https://registry.ollama.ai"
    disable_update_check: bool = False
    disable_openapi_docs: bool = False
    update_check_url: Optional[str] = None
    model_catalog_file: Optional[str] = None
    ray_port: int = 40096
    ray_client_server_port: int = 40097
    enable_cors: bool = False
    allow_origins: Optional[List[str]] = ['*']
    allow_credentials: bool = False
    allow_methods: Optional[List[str]] = ['GET', 'POST']
    allow_headers: Optional[List[str]] = ['Authorization', 'Content-Type']


class ServerConfigMixin(ServerConfigCLIMixin):
    # not supported in args
    jwt_secret_key: Optional[str] = None


class WorkerConfigCLIMixin:
    # Worker options
    server_url: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_name: Optional[str] = None
    worker_port: int = 10150
    disable_metrics: bool = False
    disable_rpc_servers: bool = False
    metrics_port: int = 10151
    service_port_range: Optional[str] = "40000-40063"
    rpc_server_port_range: Optional[str] = "40064-40095"
    log_dir: Optional[str] = None
    rpc_server_args: Optional[List[str]] = None
    system_reserved: Optional[dict] = None
    tools_download_base_url: Optional[str] = None
    ray_worker_port_range: Optional[str] = "40100-40131"
    enable_hf_transfer: bool = False
    enable_hf_xet: bool = False


class WorkerConfigMixin(WorkerConfigCLIMixin):
    # not supported in args
    resources: Optional[dict] = None


def list_config_attributes(config_class: object) -> List[str]:
    return [
        name
        for name, value in config_class.__dict__.items()
        if not name.startswith('__')
        and not callable(value)
        and not isinstance(value, (classmethod, staticmethod))
    ]


def set_config_option_from_class(config_mixin, args, config_data: dict):
    for option in list_config_attributes(config_mixin):
        option_value = getattr(args, option, None)
        if option_value is not None:
            config_data[option] = option_value


def set_common_options(args, config_data: dict):
    set_config_option_from_class(CommonConfigMixin, args, config_data)


def set_server_options(args, config_data: dict):
    set_config_option_from_class(ServerConfigCLIMixin, args, config_data)


def set_worker_options(args, config_data: dict):
    set_config_option_from_class(WorkerConfigCLIMixin, args, config_data)
