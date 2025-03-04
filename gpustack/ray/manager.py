import asyncio
import logging
import os
import subprocess
import sysconfig
from urllib.parse import urlsplit
from gpustack.config import Config


logger = logging.getLogger(__name__)


class RayManager:
    """
    RayManager manages Ray nodes.
    """

    def __init__(self, cfg: Config, head: bool = False, pure_head: bool = False):
        self._cfg = cfg
        self._head = head
        self._pure_head = pure_head
        self._role = "head" if head else "worker"
        self._ray_address = get_ray_address(cfg.server_url, 6379)

        self._ray_args = cfg.ray_args
        self._ray_process = None
        self._log_file_path = f"{cfg.log_dir}/ray-{self._role}.log"
        self._check_interval = 15

    async def start(self):
        while True:
            await asyncio.sleep(self._check_interval)
            await self._start()

    async def _start(self):
        current = self._ray_process
        if self._ray_process:
            returncode = current.poll()
            if returncode is None:
                # Ray is running
                return

            raise Exception(
                f"Ray exited with code {returncode}, check logs in {self._log_file_path} to diagnose."
            )

        await self._start_ray()

    async def _start_ray(self):
        logger.info(f"Starting Ray {self._role}.")

        command_path = os.path.join(sysconfig.get_path("scripts"), "ray")
        arguments = ["start", "--block"]
        if self._head:
            arguments.extend(["--head"])
        else:
            arguments.extend(["--address", self._ray_address])

        if self._pure_head:
            arguments.extend(["--num-cpus=0", "--num-gpus=0"])

        if self._cfg.worker_ip:
            arguments.extend(["--node-ip-address", self._cfg.worker_ip])

        if self._ray_args:
            arguments.extend(self._ray_args)

        logger.debug(f"Run Ray with arguments: {' '.join([command_path] + arguments)}")

        proc = subprocess.Popen(
            [command_path] + arguments,
            stdout=open(self._log_file_path, "w"),
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

        self._ray_process = proc

        await asyncio.sleep(5)
        if proc.poll() is not None:
            raise Exception(
                f"Failed to start Ray {self._role}. Check logs in {self._log_file_path} to diagnose."
            )

        logger.info(f"Started Ray {self._role}.")


def get_ray_address(server_url: str, ray_port: int) -> str:
    """
    Get the Ray address from the server URL and ray port.
    """
    parsed = urlsplit(server_url)
    hostport = parsed.netloc

    parts = hostport.rsplit(':', 1)
    host = parts[0] if len(parts) == 2 else hostport

    return f"{host}:{ray_port}"
