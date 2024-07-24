import base64
from urllib.parse import urlparse
from .generated_http_client import HTTPClient
from typing import Optional

from .generated_worker_client import WorkerClient
from .generated_model_client import ModelClient
from .generated_model_instance_client import ModelInstanceClient
from .generated_user_client import UserClient


class ClientSet:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        if headers is None:
            headers = {}

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif username and password:
            base64_credentials = base64.b64encode(
                f"{username}:{password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {base64_credentials}"

        self.base_url = base_url
        self.headers = headers

        verify = None
        parsed_url = urlparse(base_url)
        if parsed_url.hostname == "127.0.0.1" and parsed_url.scheme == "https":
            verify = False

        http_client = HTTPClient(base_url=base_url, verify_ssl=verify).with_headers(
            headers
        )

        self.workers = WorkerClient(http_client)
        self.models = ModelClient(http_client)
        self.model_instances = ModelInstanceClient(http_client)
        self.users = UserClient(http_client)
