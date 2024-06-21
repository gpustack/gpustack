import base64
from .generated_http_client import HTTPClient
from typing import Optional

from .generated_node_client import NodeClient
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
    ):
        headers = {}

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif username and password:
            base64_credentials = base64.b64encode(
                f"{username}:{password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {base64_credentials}"

        http_client = HTTPClient(base_url=base_url).with_headers(headers)

        self.nodes = NodeClient(http_client)
        self.models = ModelClient(http_client)
        self.model_instances = ModelInstanceClient(http_client)
        self.users = UserClient(http_client)
