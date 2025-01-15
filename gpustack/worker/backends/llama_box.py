import glob
import logging
import os
from pathlib import Path
import subprocess
import sys
import psutil

from gpustack.utils import platform
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceStateEnum,
    is_embedding_model,
    is_image_model,
    is_renaker_model,
)
from gpustack.utils.command import find_parameter, get_versioned_command
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class LlamaBoxServer(InferenceServer):
    def start(self):  # noqa: C901
        command_path = pkg_resources.files(
            "gpustack.third_party.bin.llama-box"
        ).joinpath(get_llama_box_command())

        if self._model.backend_version:
            command_path = os.path.join(
                self._config.bin_dir,
                get_versioned_command(
                    get_llama_box_command(), self._model.backend_version
                ),
            )

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

        default_parallel = "4"
        if is_renaker_model(self._model) or is_embedding_model(self._model):
            default_parallel = "1"

        arguments = [
            "--host",
            "0.0.0.0",
            "--embeddings",
            "--gpu-layers",
            str(layers),
            "--parallel",
            default_parallel,
            "--ctx-size",
            "8192",
            "--port",
            str(self._model_instance.port),
            "--model",
            self._model_path,
            "--alias",
            self._model.name,
            "--no-mmap",
            "--no-warmup",
        ]

        if is_renaker_model(self._model):
            arguments.append("--rerank")

        if is_image_model(self._model):
            # TODO support multi-GPU for image models
            arguments.extend(["--images", "--image-vae-tiling", "--tensor-split", "1"])

        mmproj = find_parameter(self._model.backend_parameters, ["mmproj"])
        default_mmproj = get_mmproj_file(self._model_path)
        if mmproj is None and default_mmproj:
            arguments.extend(["--mmproj", default_mmproj])

        if rpc_servers:
            rpc_servers_argument = ",".join(rpc_servers)
            arguments.extend(["--rpc", rpc_servers_argument])

        final_tensor_split = []
        if rpc_server_tensor_split:
            final_tensor_split.extend(rpc_server_tensor_split)
            final_tensor_split.extend(main_worker_tensor_split)
        elif len(main_worker_tensor_split) > 1:
            final_tensor_split.extend(main_worker_tensor_split)

        user_tensor_split = find_parameter(
            self._model.backend_parameters, ["ts", "tensor-split"]
        )
        if user_tensor_split is None and final_tensor_split:
            tensor_split_argument = ",".join(
                [str(int(tensor / (1024 * 1024))) for tensor in final_tensor_split]
            )  # convert to MiB to prevent overflow

            arguments.extend(["--tensor-split", tensor_split_argument])

        if self._model.backend_parameters:
            self.normalize_mmproj_path()
            # append user-provided parameters
            for param in self._model.backend_parameters:
                if "=" not in param:
                    arguments.append(param)
                else:
                    key, value = param.split('=', 1)
                    arguments.extend([key, value])

        worker_gpu_devices = None
        worker = worker_map.get(self._model_instance.worker_id)
        if worker and worker.status.gpu_devices:
            worker_gpu_devices = worker.status.gpu_devices

        env = self.get_inference_running_env(
            self._model_instance.gpu_indexes, worker_gpu_devices
        )
        try:
            logger.info("Starting llama-box server")
            logger.debug(
                f"Run llama-box: {command_path} with arguments: {' '.join(arguments)}"
            )

            proc = subprocess.Popen(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )

            set_priority(proc.pid)
            exit_code = proc.wait()
            self.exit_with_code(exit_code)

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
            sys.exit(1)

    def normalize_mmproj_path(self):
        """
        We provide a syntax sugar for the user to specify the mmproj file relative to the model path.
        So, users can specify --mmproj=mmproj.gguf instead of --mmproj=/path/to/mmproj.gguf.
        This function normalizes the file path to the same directory of the model path and set it back to the backend parameters.
        """

        model_dir = Path(self._model_path).parent
        mmproj_param = "--mmproj"
        for i, param in enumerate(self._model.backend_parameters):
            if '=' in param:
                key, value = param.split('=', 1)
                if key == mmproj_param and not Path(value).is_absolute():
                    self._model.backend_parameters[i] = (
                        f"{mmproj_param}={model_dir / value}"
                    )
            else:
                if param == mmproj_param and i + 1 < len(
                    self._model.backend_parameters
                ):
                    value = self._model.backend_parameters[i + 1]
                    if not Path(value).is_absolute():
                        self._model.backend_parameters[i + 1] = str(model_dir / value)


def get_mmproj_file(model_path: str) -> str:
    directory = os.path.dirname(model_path)
    pattern = os.path.join(directory, '*mmproj*.gguf')
    files = glob.glob(pattern)

    if files:
        return files[0]


def set_priority(pid: int):
    if platform.system() != "windows":
        return

    try:
        priority_class = psutil.ABOVE_NORMAL_PRIORITY_CLASS
        proc = psutil.Process(pid)
        proc.nice(priority_class)
        logger.debug(f"Set process {proc.pid} priority to {priority_class}")
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.error(f"Failed to set priority for process {pid}: {e}")


def get_llama_box_command():
    command = "llama-box"
    if platform.system() == "windows":
        command += ".exe"
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
