import os
import secrets
from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings
from gpustack.utils import validators

_config = None


class Config(BaseSettings):
    """A class used to define GPUStack configuration.

    Attributes:
        debug: Enable debug mode.
        data_dir: Directory to store data. Default is OS specific.
        token: Shared secret used to add a worker.

        host: Host to bind the server to.
        port: Port to bind the server to.
        ssl_keyfile: Path to the SSL key file.
        ssl_certfile: Path to the SSL certificate file.
        database_url: URL of the database.
        disable_worker: Disable embedded worker.
        bootstrap_password: Password for the bootstrap admin user.
        jwt_secret_key: Secret key for JWT. Auto-generated by default.
        system_reserved: Reserved system resources.
        force_auth_localhost: Force authentication for requests originating from
                              localhost (127.0.0.1). When set to True, all requests
                              from localhost will require authentication.
        ollama_library_base_url: Base URL of the Ollama library. Default is https://registry.ollama.ai.

        server_url: URL of the server.
        worker_ip: IP address of the worker node. Auto-detected by default.
        worker_name: Name of the worker node. Use the hostname by default.
        disable_metrics: Disable metrics.
        metrics_port: Port to expose metrics on.
        worker_port: Port to bind the worker to.
        log_dir: Directory to store logs.
    """

    # Common options
    debug: bool = False
    data_dir: Optional[str] = None
    token: Optional[str] = None
    huggingface_token: Optional[str] = None

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

    # Worker options
    server_url: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_name: Optional[str] = None
    disable_metrics: bool = False
    worker_port: int = 10150
    metrics_port: int = 10151
    log_dir: Optional[str] = None

    def __init__(self, **values):
        super().__init__(**values)

        # common options
        if self.data_dir is None:
            self.data_dir = self.get_data_dir()

        if self.log_dir is None:
            self.log_dir = os.path.join(self.data_dir, "log")

        if not self._is_server() and not self.token:
            raise Exception("Token is required when running as a worker")

        self.prepare_token()
        self.prepare_jwt_secret_key()

        # server options
        self.init_database_url()

        if self.system_reserved is None:
            self.system_reserved = {"ram": 2, "vram": 0}

    @model_validator(mode="after")
    def check_all(self):
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

        return self

    def init_database_url(self):
        if self.database_url is None:
            self.database_url = f"sqlite:///{self.data_dir}/database.db"
            return

        if not self.database_url.startswith(
            "sqlite://"
        ) and not self.database_url.startswith("postgresql://"):
            raise Exception(
                "Unsupported database scheme. Supported databases are sqlite and postgresql."
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
                file.write(token)

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
