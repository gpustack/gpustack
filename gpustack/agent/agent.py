import socket
import time
from gpustack.agent.config import AgentConfig
from gpustack.utils import run_periodically_async
from gpustack.logging import logger
from gpustack.generated_client.client import Client


class Agent:
    def __init__(self, cfg: AgentConfig):
        self._cfg = cfg
        self._registration_completed = False
        self._client = Client(base_url=cfg.server)
        self._localhost = socket.gethostbyname("localhost")

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
        # 1. get node from server
        # 2. update node status if there is any change or enough time passed since last update

        pass

    def _register_with_server(self):
        if self._registration_completed:
            return

        node = self._initialize_node()
        self._register_node(node)
        self._registration_completed = True

    def _register_node(self, node):
        # 1. create a node using the client
        # 2. if the node is already registered, update the node
        pass

    def _initialize_node(self):
        # initialize a node with the current system information
        pass

    def _register_shutdown_hooks(self):
        pass
