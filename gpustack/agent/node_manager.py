import os
import platform
import socket
from gpustack.api.exceptions import is_already_exists, is_error_response
from gpustack.generated_client.api.nodes import create_node_v1_nodes_post
from gpustack.generated_client.client import Client
from gpustack.logging import logger
from gpustack.schemas.nodes import Node, ResourceSummary


class NodeManager:
    def __init__(self, node_ip: str, client: Client):
        self._registration_completed = False
        self._hostname = socket.gethostname()
        self._node_ip = node_ip
        self._client = client

    def sync_node_status(self):
        """
        Should be called periodically to sync the node status with the server.
        It registers the node with the server if necessary.
        """

        logger.info("Syncing node status.")
        self._register_with_server()
        self._update_node_status()

    def _update_node_status(self):
        # TODO update node status if there is any change or enough time passed since last update

        pass

    def _register_with_server(self):
        if self._registration_completed:
            return

        node = self._initialize_node()
        self._register_node(node)
        self._registration_completed = True

    def _register_node(self, node: Node):
        logger.info(
            f"Registering node: {node.name}",
        )

        result = create_node_v1_nodes_post.sync(client=self._client, body=node)

        if not is_error_response(result):
            logger.info(
                f"Node {node.name} registered.",
            )
            return

        if is_already_exists(result):
            logger.info(f"Node {node.name} already exists, skip registration.")
        else:
            logger.error(f"Failed to register node: {result.message}")

    def _initialize_node(self):
        node = Node(
            name=self._hostname,
            hostname=self._hostname,
            address=self._node_ip,
            resources=ResourceSummary(allocatable={}, capacity={}),
        )

        os_info = os.uname()
        arch_info = platform.machine()

        node.labels = {
            "os": os_info.sysname,
            "arch": arch_info,
        }

        return node

    def _register_shutdown_hooks(self):
        pass
