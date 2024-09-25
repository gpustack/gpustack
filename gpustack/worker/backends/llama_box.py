import logging
import platform
import subprocess
import sys

from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceStateEnum,
)
from gpustack.utils.command import get_platform_command
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class LlamaBoxServer(InferenceServer):
    def start(self):  # noqa: C901
        cmd_extra_key = (
            "gpu"
            if self._model_instance.gpu_indexes
            and len(self._model_instance.gpu_indexes) > 0
            else "cpu"
        )

        command_path = pkg_resources.files(
            "gpustack.third_party.bin.llama-box"
        ).joinpath(get_llama_box_command(cmd_extra_key))

        layers = -1
        claim = self._model_instance.computed_resource_claim
        if claim is not None and claim.get("offload_layers") is not None:
            layers = claim.get("offload_layers")

        main_worker_tensor_split = []
        if (
            self._model_instance.gpu_indexes
            and len(self._model_instance.gpu_indexes) > 0
        ):
            vram_claims = claim.get("vram").values()
            main_worker_tensor_split = vram_claims

        workers = self._clientset.workers.list()
        worker_map = {worker.id: worker for worker in workers.items}
        rpc_servers, rpc_server_tensor_split = get_rpc_servers(
            self._model_instance, worker_map
        )

        arguments = [
            "--host",
            "0.0.0.0",
            "--embeddings",
            "--gpu-layers",
            str(layers),
            "--parallel",
            "4",
            "--ctx-size",
            "8192",
            "--port",
            str(self._model_instance.port),
            "--model",
            self._model_path,
            "--no-mmap",
        ]

        if rpc_servers:
            rpc_servers_argument = ",".join(rpc_servers)
            arguments.extend(["--rpc", rpc_servers_argument])

        final_tensor_split = []
        if rpc_server_tensor_split:
            final_tensor_split.extend(rpc_server_tensor_split)
            final_tensor_split.extend(main_worker_tensor_split)
        elif len(main_worker_tensor_split) > 1:
            final_tensor_split.extend(main_worker_tensor_split)

        if final_tensor_split:
            tensor_split_argument = ",".join(
                [str(int(tensor / (1024 * 1024))) for tensor in final_tensor_split]
            )  # convert to MiB to prevent overflow

            arguments.extend(["--tensor-split", tensor_split_argument])

        if self._model.backend_parameters:
            # A make-it-work solution for now.
            # TODO Fine-grained control of llama-box parameters.
            arguments.extend(self._model.backend_parameters)

        env = self.get_inference_running_env(self._model_instance.gpu_indexes)
        try:
            logger.info("Starting llama-box server")
            logger.debug(
                f"Run llama-box: {command_path} with arguments: {' '.join(arguments)}"
            )
            subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )
        except Exception as e:
            error_message = f"Failed to run the llama-box server: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(self._model_instance.id, **patch_dict)
            except Exception as ue:
                logger.error(f"Failed to update model instance: {ue}")


def get_llama_box_command(extra_key):
    command_map = {
        ("Windows", "amd64", "gpu"): "llama-box-windows-amd64-cuda-12.6.exe",
        ("Darwin", "amd64", "gpu"): "llama-box-darwin-amd64-metal",
        ("Darwin", "arm64", "gpu"): "llama-box-darwin-arm64-metal",
        ("Linux", "amd64", "gpu"): "llama-box-linux-amd64-cuda-12.6",
        ("Linux", "amd64", "cpu"): "llama-box-linux-amd64-avx2",
        ("Linux", "arm64", "cpu"): "llama-box-linux-arm64-neon",
        ("Windows", "amd64", "cpu"): "llama-box-windows-amd64-avx2.exe",
        ("Windows", "arm64", "cpu"): "llama-box-windows-arm64-neon.exe",
    }

    command = get_platform_command(command_map, extra_key)
    if command == "":
        raise Exception(
            f"No supported llama-box command found "
            f"for {platform.system()} {platform.machine()}."
        )
    return command


def get_rpc_servers(model_instance: ModelInstance, worker_map):
    rpc_servers = []
    rpc_tensor_split = []
    if model_instance.distributed_servers and model_instance.distributed_servers.get(
        "rpc_servers"
    ):
        for rpc_server in model_instance.distributed_servers.get("rpc_servers"):
            r_worker = worker_map.get(rpc_server.get("worker_id"))
            r_ip = r_worker.ip
            r_port = r_worker.status.rpc_servers.get(rpc_server.get("gpu_index")).port
            r_ts = list(
                rpc_server.get("computed_resource_claim", {}).get("vram").values()
            )

            rpc_tensor_split.extend(r_ts)
            rpc_servers.append(f"{r_ip}:{r_port}")
    return rpc_servers, rpc_tensor_split
