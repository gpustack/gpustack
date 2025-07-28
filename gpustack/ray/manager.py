import asyncio
import logging
import os
import subprocess
from urllib.parse import urlsplit
from gpustack.config import Config
from gpustack.utils.network import parse_port_range
from gpustack.utils.command import get_command_path


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
        self._ray_head_host = cfg.worker_ip if head else extract_host(cfg.server_url)
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

            logger.error(
                f"Ray exited with code {returncode}, check logs in {self._log_file_path} to diagnose. Restarting..."
            )

        await self._start_ray()

    async def _start_ray(self):
        logger.info(f"Starting Ray {self._role}.")

        min_worker_port, max_worker_port = parse_port_range(
            self._cfg.ray_worker_port_range
        )

        command_path = get_command_path('ray')
        arguments = [
            "start",
            "--block",
            "--node-manager-port",
            str(self._cfg.ray_node_manager_port),
            "--object-manager-port",
            str(self._cfg.ray_object_manager_port),
            "--dashboard-agent-grpc-port",
            str(self._cfg.ray_dashboard_agent_grpc_port),
            "--dashboard-agent-listen-port",
            str(self._cfg.ray_dashboard_agent_listen_port),
            "--metrics-export-port",
            str(self._cfg.ray_metrics_export_port),
            "--min-worker-port",
            str(min_worker_port),
            "--max-worker-port",
            str(max_worker_port),
        ]
        if self._head:
            arguments.extend(
                [
                    "--head",
                    "--port",
                    str(self._cfg.ray_port),
                    "--ray-client-server-port",
                    str(self._cfg.ray_client_server_port),
                    "--dashboard-port",
                    str(self._cfg.ray_dashboard_port),
                ]
            )
            # If the worker IP is provided, use it as the dashboard host.
            # Therefore, the dashboard will be accessible from the worker IP,
            # and allow all non-head nodes to connect to it.
            if self._ray_head_host:
                arguments.extend(
                    [
                        "--dashboard-host",
                        str(self._ray_head_host),
                    ]
                )
        else:
            arguments.extend(
                [
                    "--address",
                    f"{self._ray_head_host}:{self._cfg.ray_port}",
                ]
            )

        if self._pure_head:
            arguments.extend(["--num-cpus=0", "--num-gpus=0"])

        if self._cfg.worker_ip:
            arguments.extend(["--node-ip-address", self._cfg.worker_ip])

        if self._cfg.ray_args:
            arguments.extend(self._cfg.ray_args)

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
            logger.error(
                f"Failed to start Ray {self._role}. Check logs in {self._log_file_path} to diagnose."
            )

        logger.info(f"Started Ray {self._role}.")


def extract_host(url: str) -> str:
    """
    Extract the host from the given URL.
    """
    us = urlsplit(url)
    ps = us.netloc.rsplit(':', 1)

    return ps[0] if len(ps) == 2 else us.netloc
