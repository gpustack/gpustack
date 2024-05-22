import os
import platform
import socket
import time
from gpustack.agent.config import AgentConfig
from gpustack.api.exceptions import is_error_response, is_already_exists
from gpustack.generated_client.api.nodes import (
    create_node_v1_nodes_post,
)
from gpustack.utils import run_periodically_async
from gpustack.logging import logger
from gpustack.generated_client.client import Client
from gpustack.schemas.nodes import Node, ResourceSummary


class Agent:
    def __init__(self, cfg: AgentConfig):
        self._cfg = cfg
        self._registration_completed = False
        self._client = Client(base_url=cfg.server)
        self._hostname = socket.gethostname()

    def start(self):
        """
        Start the agent.
        """

        logger.info("Starting GPUStack agent.")

        # Report the node status to the server periodically.
        run_periodically_async(self.sync_node_status, 5 * 60)

        self.sync_loop()

    def sync_loop(self):
        """
        Main loop for processing changes.
        It watches task changes from server and processes them.
        """

        logger.info("Starting sync loop.")
        while True:
            time.sleep(1)

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
            address=self._cfg.node_ip,
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
