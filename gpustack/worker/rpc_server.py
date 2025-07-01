from contextlib import redirect_stderr, redirect_stdout
import logging
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

import setproctitle

from gpustack.utils import platform
from gpustack.utils.command import normalize_parameters
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.process import add_signal_handlers
from gpustack.worker.backends.base import get_env_name_by_vendor
from gpustack.worker.tools_manager import (
    is_disabled_dynamic_link,
    BUILTIN_LLAMA_BOX_VERSION,
)


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
    def start(
        port: int,
        gpu_index: int,
        vendor: str,
        log_file_path: str,
        cache_dir: str,
        bin_dir: Optional[str] = None,
        args: Optional[List[str]] = None,
    ):
        setproctitle.setproctitle(f"gpustack_rpc_server_process: gpu_{gpu_index}")
        add_signal_handlers()

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                RPCServer._start(port, gpu_index, vendor, cache_dir, bin_dir, args)

    def _start(
        port: int,
        gpu_index: int,
        vendor: str,
        cache_dir: str,
        bin_dir: Optional[str] = None,
        args: Optional[List[str]] = None,
    ):
        base_path = (
            pkg_resources.files("gpustack.third_party.bin").joinpath(
                'llama-box/llama-box-default'
            )
            if bin_dir is None
            or not is_disabled_dynamic_link(
                BUILTIN_LLAMA_BOX_VERSION, platform.device()
            )
            else (
                (
                    Path(bin_dir)
                    / 'llama-box'
                    / 'static'
                    / f'llama-box-{BUILTIN_LLAMA_BOX_VERSION}'
                )
            )
        )
        command_path = base_path / RPCServer.get_llama_box_rpc_server_command()

        arguments = [
            "--rpc-server-host",
            "0.0.0.0",
            "--rpc-server-port",
            str(port),
            "--rpc-server-main-gpu",
            str(0),
            # llama-box allows users to define custom flags, even if those flags are not currently supported by llama-box.
            # We use origin-rpc-server-main-gpu to specify the GPU that the RPC server is actually running on.
            "--origin-rpc-server-main-gpu",
            str(gpu_index),
            "--rpc-server-cache",
            "--rpc-server-cache-dir",
            cache_dir,
        ]

        if args:
            remove_arguments = [
                "rpc-server-host",
                "rpc-server-port",
                "rpc-server-main-gpu",
                "origin-rpc-server-main-gpu",
            ]
            args = normalize_parameters(args, removes=remove_arguments)
            arguments.extend(args)

        env_name = get_env_name_by_vendor(vendor)
        env = os.environ.copy()
        env[env_name] = str(gpu_index)
        cwd = str(command_path.parent)
        if platform.system() == "linux":
            ld_library_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                ":".join([cwd, ld_library_path]) if ld_library_path else cwd
            )
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
                cwd=cwd,
            )
        except Exception as e:
            error_message = f"Failed to run the llama-box rpc server: {e}"
            logger.error(error_message)
            raise Exception(error_message) from e

    @staticmethod
    def get_llama_box_rpc_server_command() -> str:
        command = "llama-box-rpc-server"
        if platform.system() == "windows":
            command += ".exe"
        return command
