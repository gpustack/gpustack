from .generated_http_client import HTTPClient


from .generated_node_client import NodeClient

from .generated_model_client import ModelClient

from .generated_model_instance_client import ModelInstanceClient

from .generated_user_client import UserClient


class ClientSet:

    def __init__(self, base_url: str):
        http_client = HTTPClient(base_url=base_url)

        self.nodes = NodeClient(http_client)

        self.models = ModelClient(http_client)

        self.model_instances = ModelInstanceClient(http_client)

        self.users = UserClient(http_client)
