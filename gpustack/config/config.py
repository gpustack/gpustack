import os
import secrets
from typing import TYPE_CHECKING, List, Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings
from gpustack.utils import validators
from gpustack.schemas.workers import (
    CPUInfo,
    FileSystemInfo,
    GPUDeviceInfo,
    KernelInfo,
    MemoryInfo,
    MountPoint,
    OperatingSystemInfo,
    SwapInfo,
    SystemInfo,
    UptimeInfo,
    VendorEnum,
    GPUDevicesInfo,
)
from gpustack.utils import platform
from gpustack.utils.platform import DeviceTypeEnum, device_type_from_vendor

_config = None


class Config(BaseSettings):
    """A class used to define GPUStack configuration.

    Attributes:
        debug: Enable debug mode.
        data_dir: Directory to store data. Default is OS specific.
        token: Shared secret used to add a worker.
        huggingface_token: User Access Token to authenticate to the Hugging Face Hub.
        enable_ray: Enable Ray.
        ray_args: Additional arguments to pass to Ray.

        host: Host to bind the server to.
        port: Port to bind the server to.
        ssl_keyfile: Path to the SSL key file.
        ssl_certfile: Path to the SSL certificate file.
        database_url: URL of the database.
        disable_worker: Disable embedded worker.
        bootstrap_password: Password for the bootstrap admin user.
        jwt_secret_key: Secret key for JWT. Auto-generated by default.
        force_auth_localhost: Force authentication for requests originating from
                              localhost (127.0.0.1). When set to True, all requests
                              from localhost will require authentication.
        ollama_library_base_url: Base URL of the Ollama library. Default is https://registry.ollama.ai.
        disable_update_check: Disable update check.
        update_check_url: URL to check for updates.
        model_catalog_file: Path or URL to the model catalog file.
        ray_port: Port of Ray (GCS server). Used when Ray is enabled. Default is 40096.
        ray_client_server_port: Port of Ray Client Server. Used when Ray is enabled. Default is 40097.

        server_url: URL of the server.
        worker_ip: IP address of the worker node. Auto-detected by default.
        worker_name: Name of the worker node. Use the hostname by default.
        disable_metrics: Disable metrics.
        disable_rpc_servers: Disable RPC servers.
        metrics_port: Port to expose metrics on.
        worker_port: Port to bind the worker to.
        service_port_range: Port range for inference services, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive. Default is '40000-40063'.
        rpc_server_port_range: Port range for RPC servers, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive. Default is '40064-40095'.
        ray_node_manager_port: Raylet port for node manager. Used when Ray is enabled. Default is 40098.
        ray_object_manager_port: Raylet port for object manager. Used when Ray is enabled. Default is 40099.
        ray_worker_port_range: Port range for Ray worker processes, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive. Default is '40100-40131'.
        log_dir: Directory to store logs.
        bin_dir: Directory to store additional binaries, e.g., versioned backend executables.
        pipx_path: Path to the pipx executable, used to install versioned backends.
        system_reserved: Reserved system resources.
        tools_download_base_url: Base URL to download dependency tools.
        enable_hf_transfer: Speed up file transfers with the huggingface Hub.
        enable_cors: Enable CORS in server.
        allow_origins: A list of origins that should be permitted to make cross-origin requests.
        allow_credentials: Indicate that cookies should be supported for cross-origin requests.
        allow_methods: A list of HTTP methods that should be allowed for cross-origin requests.
        allow_headers: A list of HTTP request headers that should be supported for cross-origin requests.
    """

    # Common options
    debug: bool = False
    data_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    token: Optional[str] = None
    huggingface_token: Optional[str] = None
    enable_ray: bool = False
    ray_args: Optional[List[str]] = None

    # Server options
    host: Optional[str] = "0.0.0.0"
    port: Optional[int] = None
    database_url: Optional[str] = None
    disable_worker: bool = False
    bootstrap_password: Optional[str] = None
    jwt_secret_key: Optional[str] = None
    system_reserved: Optional[dict] = None
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    force_auth_localhost: bool = False
    ollama_library_base_url: Optional[str] = "https://registry.ollama.ai"
    disable_update_check: bool = False
    update_check_url: Optional[str] = None
    model_catalog_file: Optional[str] = None
    ray_port: int = 40096
    ray_client_server_port: int = 40097
    enable_cors: bool = False
    allow_origins: Optional[List[str]] = ['*']
    allow_credentials: bool = False
    allow_methods: Optional[List[str]] = ['GET', 'POST']
    allow_headers: Optional[List[str]] = ['Authorization', 'Content-Type']

    # Worker options
    server_url: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_name: Optional[str] = None
    disable_metrics: bool = False
    disable_rpc_servers: bool = False
    worker_port: int = 10150
    metrics_port: int = 10151
    service_port_range: Optional[str] = "40000-40063"
    rpc_server_port_range: Optional[str] = "40064-40095"
    ray_node_manager_port: int = 40098
    ray_object_manager_port: int = 40099
    ray_worker_port_range: Optional[str] = "40100-40131"
    log_dir: Optional[str] = None
    resources: Optional[dict] = None
    bin_dir: Optional[str] = None
    pipx_path: Optional[str] = None
    tools_download_base_url: Optional[str] = None
    rpc_server_args: Optional[List[str]] = None
    enable_hf_transfer: bool = False

    def __init__(self, **values):
        super().__init__(**values)

        # common options
        if self.data_dir is None:
            self.data_dir = self.get_data_dir()

        if self.cache_dir is None:
            self.cache_dir = os.path.join(self.data_dir, "cache")

        if self.bin_dir is None:
            self.bin_dir = os.path.join(self.data_dir, "bin")

        if self.log_dir is None:
            self.log_dir = os.path.join(self.data_dir, "log")

        if not self._is_server() and not self.token:
            raise Exception("Token is required when running as a worker")

        self.prepare_token()
        self.prepare_jwt_secret_key()

        # server options
        self.init_database_url()

        if self.system_reserved is None:
            self.system_reserved = {"ram": 2, "vram": 1}

    @model_validator(mode="after")
    def check_all(self):  # noqa: C901
        if 'PYTEST_CURRENT_TEST' in os.environ:
            # Skip validation during tests
            return self

        if (self.ssl_keyfile and not self.ssl_certfile) or (
            self.ssl_certfile and not self.ssl_keyfile
        ):
            raise Exception(
                'Both "ssl_keyfile" and "ssl_certfile" must be provided, or neither.'
            )

        if self.server_url:
            self.server_url = self.server_url.rstrip("/")
            if validators.url(self.server_url) is not True:
                raise Exception("Invalid server URL.")

        if self.ollama_library_base_url:
            self.ollama_library_base_url = self.ollama_library_base_url.rstrip("/")
            if validators.url(self.ollama_library_base_url) is not True:
                raise Exception("Invalid Ollama library base URL.")

        if self.resources:
            self.get_gpu_devices()
            self.get_system_info()

        if self.enable_ray:
            self.check_ray()

        if self.service_port_range:
            self.check_port_range(self.service_port_range)

        if self.rpc_server_port_range:
            self.check_port_range(self.rpc_server_port_range)

        if self.ray_worker_port_range:
            self.check_port_range(self.ray_worker_port_range)

        return self

    def check_ray(self):
        system = platform.system()
        if system != "linux":
            raise Exception("Ray is only supported on Linux.")

        if not TYPE_CHECKING:
            try:
                from vllm.platforms import current_platform
            except ImportError:
                raise Exception(
                    "vLLM is not installed. Please install vLLM to work with Ray."
                )

            device_str = current_platform.ray_device_key
            if not device_str:
                raise Exception(
                    f"current platform {current_platform.device_name} does not support Ray."
                )

    def check_port_range(self, port_range: str):
        ports = port_range.split("-")
        if len(ports) != 2:
            raise Exception(f"Invalid port range: {port_range}")
        if not ports[0].isdigit() or not ports[1].isdigit():
            raise Exception("Port range must be numeric")
        if int(ports[0]) > int(ports[1]):
            raise Exception(f"Invalid port range: {ports[0]} > {ports[1]}")

    def get_system_info(self) -> SystemInfo:  # noqa: C901
        """get system info from resources
        resource example:
        ```yaml
        resources:
            cpu:
              total: 10
            memory:
              total: 34359738368
              is_unified_memory: true
            swap:
              total: 3221225472
            filesystem:
              - name: Macintosh HD
                mount_point: /
                mount_from: /dev/disk3s1s1
                total: 994662584320
            os:
              name: macOS
              version: "14.5"
            kernel:
              name: Darwin
              release: 23.5.0
              version: "Darwin Kernel Version 23.5.0: Wed May  1 20:12:58 PDT 2024;"
              architecture: ""
            uptime:
              uptime: 355250885
              boot_time: 2025-02-24T09:17:51.337+0800
        ```
        """
        system_info: SystemInfo = SystemInfo()
        if not self.resources:
            return None

        cpu_dict = self.resources.get("cpu")
        if cpu_dict and cpu_dict.get("total"):
            system_info.cpu = CPUInfo(total=cpu_dict.get("total"))

        memory_dict = self.resources.get("memory")
        if memory_dict and memory_dict.get("total"):
            system_info.memory = MemoryInfo(total=memory_dict.get("total"))

        swap_dict = self.resources.get("swap")
        if swap_dict and swap_dict.get("total"):
            system_info.swap = SwapInfo(total=swap_dict.get("total"))

        filesystem_dict = self.resources.get("filesystem")
        if filesystem_dict:
            filesystem: FileSystemInfo = []
            for fs in filesystem_dict:
                name = fs.get("name")
                mount_point = fs.get("mount_point")
                mount_from = fs.get("mount_from")
                total = fs.get("total")
                if not name:
                    raise Exception("Filesystem name is required")
                if not mount_point:
                    raise Exception("Filesystem mount_point is required")
                if not mount_from:
                    raise Exception("Filesystem mount_from is required")
                if total is None:
                    raise Exception("Filesystem total is required")
                filesystem.append(
                    MountPoint(
                        name=name,
                        mount_point=mount_point,
                        mount_from=mount_from,
                        total=total,
                    )
                )
            system_info.filesystem = filesystem

        os_dict = self.resources.get("os")
        if os_dict:
            name = os_dict.get("name")
            version = os_dict.get("version")
            if not name:
                raise Exception("OS name is required")
            if not version:
                raise Exception("OS version is required")
            system_info.os = OperatingSystemInfo(name=name, version=version)

        kernel_dict = self.resources.get("kernel")
        if kernel_dict:
            name = kernel_dict.get("name")
            release = kernel_dict.get("release")
            version = kernel_dict.get("version")
            architecture = kernel_dict.get("architecture")
            if not name:
                raise Exception("Kernel name is required")
            if not release:
                raise Exception("Kernel release is required")
            if not version:
                raise Exception("Kernel version is required")
            system_info.kernel = KernelInfo(
                name=name, release=release, version=version, architecture=architecture
            )

        uptime_dict = self.resources.get("uptime")
        if uptime_dict:
            uptime = uptime_dict.get("uptime")
            boot_time = uptime_dict.get("boot_time")
            if uptime is None:
                raise Exception("Uptime is required")
            if not boot_time:
                raise Exception("Boot time is required")
            system_info.uptime = UptimeInfo(uptime=uptime, boot_time=boot_time)

        if not any(
            [
                system_info.cpu,
                system_info.memory,
                system_info.swap,
                system_info.filesystem,
                system_info.os,
                system_info.kernel,
                system_info.uptime,
            ]
        ):
            return None

        return system_info

    def get_gpu_devices(self) -> GPUDevicesInfo:
        """get gpu devices from resources
        resource example:
        ```yaml
        resources:
            gpu_devices:
            - name: Apple M1 Pro
              vendor: Apple
              index: 0
              memory:
                  total: 22906503168
                  is_unified_memory: true
        ```
        """
        gpu_devices: GPUDevicesInfo = []
        if not self.resources:
            return None

        gpu_device_dict = self.resources.get("gpu_devices")
        if not gpu_device_dict:
            return None

        for gd in gpu_device_dict:
            name = gd.get("name")
            index = gd.get("index")
            vendor = gd.get("vendor")
            memory = gd.get("memory")
            type = gd.get("type") or device_type_from_vendor(vendor)

            if not name:
                raise Exception("GPU device name is required")

            if index is None:
                raise Exception("GPU device index is required")

            if vendor not in VendorEnum.__members__.values():
                raise Exception(
                    "Unsupported GPU device vendor, supported vendors are: Apple, NVIDIA, 'Moore Threads', Huawei, AMD, Hygon"
                )

            if not memory:
                raise Exception("GPU device memory is required")

            if type not in DeviceTypeEnum.__members__.values():
                raise Exception(
                    "Unsupported GPU type, supported type are: cuda, musa, npu, mps, rocm, dcu"
                )

            memory_total = memory.get("total")
            memory_is_unified_memory = memory.get("is_unified_memory", False)
            if memory_total is None:
                raise Exception("GPU device memory total is required")

            gpu_devices.append(
                GPUDeviceInfo(
                    name=name,
                    index=index,
                    vendor=vendor,
                    memory=MemoryInfo(
                        total=memory_total, is_unified_memory=memory_is_unified_memory
                    ),
                    type=type,
                )
            )

        return gpu_devices

    def init_database_url(self):
        if self.database_url is None:
            self.database_url = f"sqlite:///{self.data_dir}/database.db"
            return

        if (
            not self.database_url.startswith("sqlite://")
            and not self.database_url.startswith("postgresql://")
            and not self.database_url.startswith("mysql+pymysql://")
        ):
            raise Exception(
                "Unsupported database scheme. Supported databases are sqlite, postgresql, and mysql."
            )

    @staticmethod
    def get_data_dir():
        app_name = "gpustack"
        if os.name == "nt":  # Windows
            data_dir = os.path.join(os.environ["APPDATA"], app_name)
        elif os.name == "posix":
            data_dir = f"/var/lib/{app_name}"
        else:
            raise Exception("Unsupported OS")

        return os.path.abspath(data_dir)

    class Config:
        env_prefix = "GPU_STACK_"
        protected_namespaces = ('settings_',)

    def prepare_token(self):
        if self.token is not None:
            return

        token_path = os.path.join(self.data_dir, "token")
        if os.path.exists(token_path):
            with open(token_path, "r") as file:
                token = file.read().strip()
        else:
            token = secrets.token_hex(16)
            os.makedirs(self.data_dir, exist_ok=True)
            with open(token_path, "w") as file:
                file.write(token + "\n")

        self.token = token

    def prepare_jwt_secret_key(self):
        if self.jwt_secret_key is not None:
            return

        key_path = os.path.join(self.data_dir, "jwt_secret_key")
        if os.path.exists(key_path):
            with open(key_path, "r") as file:
                key = file.read().strip()
        else:
            key = secrets.token_hex(32)
            os.makedirs(self.data_dir, exist_ok=True)
            with open(key_path, "w") as file:
                file.write(key)

        self.jwt_secret_key = key

    def _is_server(self):
        return self.server_url is None


def get_global_config() -> Config:
    return _config


def set_global_config(cfg: Config):
    global _config
    _config = cfg
    return cfg
