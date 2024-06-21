import os
import secrets
from pydantic_settings import BaseSettings
from gpustack.utils import get_first_non_loopback_ip


class Config(BaseSettings):
    """A class used to define GPUStack configuration.

    Attributes:
        debug: Enable debug mode.
        data_dir: Directory to store data. Default is OS specific.
        token: Shared secret used to add a worker.

        database_url: URL of the database.
        disable_worker: Disable embedded worker.
        serve_default_models: Serve default models on bootstrap.
        bootstrap_password: Password for the bootstrap admin user.
        secret_key: Secret key for the application.

        server_url: URL of the server.
        node_ip: IP address of the node. Auto-detected by default.
        enable_metrics: Enable metrics.
        metrics_port: Port to expose metrics on.
        log_dir: Directory to store logs.
    """

    # Common options
    debug: bool = False
    data_dir: str | None = None
    token: str | None = None

    # Server options
    database_url: str | None = None
    disable_worker: bool = False
    serve_default_models: bool | None = None
    bootstrap_password: str | None = None
    secret_key: str = secrets.token_urlsafe(32)

    # Worker options
    server_url: str | None = None
    node_ip: str | None = None
    enable_metrics: bool = True
    metrics_port: int = 10051
    log_dir: str | None = None

    def __init__(self, **values):
        super().__init__(**values)

        # common options
        if self.data_dir is None:
            self.data_dir = self.get_data_dir()

        if self.log_dir is None:
            self.log_dir = os.path.join(self.data_dir, "log")

        if not self._is_server() and not self.token:
            raise ValueError("Token is required when running as a worker")
        self.prepare_token()

        # server options
        if self.database_url is None:
            self.database_url = f"sqlite+aiosqlite:///{self.data_dir}/database.db"

        # worker options
        if self.node_ip is None:
            self.node_ip = get_first_non_loopback_ip()

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
            token = secrets.token_urlsafe(16)
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
