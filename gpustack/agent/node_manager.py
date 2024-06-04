import os
import platform
import socket
import logging

from gpustack.api.exceptions import (
    AlreadyExistsException,
)
from gpustack.client import ClientSet
from gpustack.schemas.nodes import Node, ResourceSummary


logger = logging.getLogger(__name__)


class NodeManager:
    def __init__(self, node_ip: str, clientset: ClientSet):
        self._registration_completed = False
        self._hostname = socket.gethostname()
        self._node_ip = node_ip
        self._clientset = clientset

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

        try:
            self._clientset.nodes.create(node)
        except AlreadyExistsException:
            logger.info(f"Node {node.name} already exists, skip registration.")
            return
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return

        logger.info(f"Node {node.name} registered.")

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
