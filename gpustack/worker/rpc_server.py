from contextlib import redirect_stderr, redirect_stdout
import logging
import multiprocessing
import os
import signal
import subprocess
import sys

import setproctitle

from gpustack.worker.inference_server import get_llama_box_command
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.signal import signal_handler

logger = logging.getLogger(__name__)


class RPCServerProcessInfo:
    def __init__(self, process: multiprocessing.Process, port: int, gpu_index: int):
        self.process = process
        self.port = port
        self.gpu_index = gpu_index


class RPCServer:
    def __init__(
        self,
    ):
        pass

    @staticmethod
    def start(port: int, gpu_index: int, log_file_path: str):
        setproctitle.setproctitle(f"gpustack_rpc_server_process: gpu_{gpu_index}")
        signal.signal(signal.SIGTERM, signal_handler)

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                RPCServer._start(port, gpu_index)

    def _start(port: int, gpu_index: int):
        command_path = pkg_resources.files(
            "gpustack.third_party.bin.llama-box"
        ).joinpath(get_llama_box_command("gpu"))

        arguments = [
            "--rpc-server-host",
            "0.0.0.0",
            "--rpc-server-port",
            str(port),
            "--rpc-server-main-gpu",
            str(gpu_index),
        ]
        env = os.environ.copy()

        try:
            logger.info("Starting llama-box rpc server")
            logger.debug(
                f"Run llama-box: {command_path} rpc server with arguments: {' '.join(arguments)}"
            )
            subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )
        except Exception as e:
            error_message = f"Failed to run the llama-box rpc server: {e}"
            logger.error(error_message)
            raise error_message
