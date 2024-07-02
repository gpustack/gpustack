import os
import secrets
from pydantic import model_validator
from pydantic_settings import BaseSettings
from gpustack.utils import get_first_non_loopback_ip


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
        secret_key: Secret key for the application.
        system_reserved: Reserved system resources.
        force_auth_localhost: Force authentication for requests originating from
                              localhost (127.0.0.1). When set to True, all requests
                              from localhost will require authentication.

        server_url: URL of the server.
        worker_ip: IP address of the worker node. Auto-detected by default.
        enable_metrics: Enable metrics.
        metrics_port: Port to expose metrics on.
        worker_port: Port to bind the worker to.
        log_dir: Directory to store logs.
    """

    # Common options
    debug: bool = False
    data_dir: str | None = None
    token: str | None = None

    # Server options
    host: str | None = None
    port: int | None = None
    database_url: str | None = None
    disable_worker: bool = False
    bootstrap_password: str | None = None
    secret_key: str = secrets.token_hex(16)
    system_reserved: dict | None = None
    ssl_keyfile: str | None = None
    ssl_certfile: str | None = None
    force_auth_localhost: bool = False

    # Worker options
    server_url: str | None = None
    worker_ip: str | None = None
    enable_metrics: bool = True
    worker_port: int = 10150
    metrics_port: int = 10151
    log_dir: str | None = None

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

        # server options
        self.init_database_url()

        # worker options
        if self.worker_ip is None:
            self.worker_ip = get_first_non_loopback_ip()

    @model_validator(mode="after")
    def check_ssl_files(self):
        if (self.ssl_keyfile and not self.ssl_certfile) or (
            self.ssl_certfile and not self.ssl_keyfile
        ):
            raise Exception(
                'Both "ssl_keyfile" and "ssl_certfile" must be provided, or neither.'
            )
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
            data_dir = os.path.expanduser(f"~/.local/share/{app_name}")
        else:
            raise Exception("Unsupported OS")

        return os.path.abspath(data_dir)

    class Config:
        env_prefix = "GPU_STACK_"

    def prepare_token(self):
        token_path = os.path.join(self.data_dir, "token")
        if os.path.exists(token_path):
            with open(token_path, "r") as file:
                token = file.read().strip()
        else:
            token = secrets.token_hex(16)
            os.makedirs(self.data_dir, exist_ok=True)
            with open(token_path, "w") as file:
                file.write(token)

        if self.token is None:
            self.token = token
        elif self.token != token:
            with open(token_path, "w") as file:
                file.write(self.token)

        return token

    def _is_server(self):
        return self.server_url is None
