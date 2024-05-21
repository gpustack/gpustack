import os
import secrets
from pydantic import Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """A class used to define server configuration.

    Attributes:
        debug: Enable debug mode.
        model: ID of a huggingface model to be served on bootstrap.
        data_dir: Directory to store data. Default is OS specific.
        database_url: URL of the database.
        bootstrap_password: Password for the bootstrap admin user.
        secret_key: Secret key for the application.
    """

    debug: bool = False
    node_ip: str | None = None
    address: str | None = None
    model: str | None = None
    data_dir: str = Field(default_factory=lambda: ServerConfig.get_data_dir())
    database_url: str | None = None
    bootstrap_password: str | None = None
    secret_key: str = secrets.token_urlsafe(32)

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
