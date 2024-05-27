import time
import multiprocessing
import logging

from gpustack.agent.config import AgentConfig
from gpustack.agent.node_manager import NodeManager
from gpustack.agent.task_manager import TaskManager
from gpustack.utils import run_periodically_async
from gpustack.generated_client.client import Client


logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, cfg: AgentConfig):
        client = Client(base_url=cfg.server)
        self._node_manager = NodeManager(node_ip=cfg.node_ip, client=client)
        self._task_manager = TaskManager(cfg.server)

    def start(self):
        """
        Start the agent.
        """

        logger.info("Starting GPUStack agent.")

        # Report the node status to the server periodically.
        run_periodically_async(self._node_manager.sync_node_status, 5 * 60)

        self.sync_loop()

    def sync_loop(self):
        """
        Main loop for processing changes.
        It watches task changes from server and processes them.
        """

        logger.info("Starting sync loop.")

        while True:
            try:
                pool = multiprocessing.Pool()
                self._task_manager.watch_tasks(pool)
            except Exception as e:
                logger.error(f"Error watching tasks: {e}")
            finally:
                pool.terminate()

            time.sleep(5)  # rewatch tasks if it fails
