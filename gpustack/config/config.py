import os
import secrets
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """A class used to define GPUStack configuration.

    Attributes:
        debug: Enable debug mode.
        data_dir: Directory to store data. Default is OS specific.

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
    data_dir: str = Field(default_factory=lambda: Config.get_data_dir())

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
    log_dir: str = os.path.expanduser("~/.local/share/gpustack/log")

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
