import asyncio
import logging
import os
import platform
import subprocess
import sys
import time
from typing import List, Optional

from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceUpdate,
    SourceEnum,
    ModelInstanceStateEnum,
)
from gpustack.utils.command import get_platform_command
from gpustack.worker.downloaders import HfDownloader, OllamaLibraryDownloader
from gpustack.utils.compat_importlib import pkg_resources

logger = logging.getLogger(__name__)


def time_decorator(func):
    """
    A decorator that logs the execution time of a function.
    """

    if asyncio.iscoroutinefunction(func):

        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            logger.info(
                f"{func.__name__} execution time: {end_time - start_time:.2f} seconds"
            )
            return result

        return async_wrapper
    else:

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(
                f"{func.__name__} execution time: {end_time - start_time} seconds"
            )
            return result

        return sync_wrapper


def download_model(
    mi: ModelInstance,
    cache_dir: Optional[str] = None,
    ollama_library_base_url: Optional[str] = None,
) -> str:
    if mi.source == SourceEnum.HUGGING_FACE:
        return HfDownloader.download(
            repo_id=mi.huggingface_repo_id,
            filename=mi.huggingface_filename,
            cache_dir=os.path.join(cache_dir, "huggingface"),
        )
    elif mi.source == SourceEnum.OLLAMA_LIBRARY:
        ollama_downloader = OllamaLibraryDownloader(
            registry_url=ollama_library_base_url
        )
        return ollama_downloader.download(
            model_name=mi.ollama_library_model_name,
            cache_dir=os.path.join(cache_dir, "ollama"),
        )


class InferenceServer:
    @time_decorator
    def __init__(
        self,
        clientset: ClientSet,
        mi: ModelInstance,
        cfg: Config,
    ):
        setup_logging(cfg.debug)
        self.hijack_tqdm_progress()

        self._clientset = clientset
        self._model_instance = mi
        try:
            self._until_model_instance_initializing()
            patch_dict = {
                "download_progress": 0,
                "state": ModelInstanceStateEnum.DOWNLOADING,
            }
            self._update_model_instance(mi.id, **patch_dict)

            cache_dir = os.path.join(cfg.data_dir, "cache")
            self._model_path = download_model(
                mi, cache_dir, ollama_library_base_url=cfg.ollama_library_base_url
            )

            patch_dict = {"state": ModelInstanceStateEnum.RUNNING, "state_message": ""}
            self._update_model_instance(mi.id, **patch_dict)
        except Exception as e:
            error_message = f"Failed to download model: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(mi.id, **patch_dict)
            except Exception as ue:
                logger.error(f"Failed to update model instance: {ue}")
            sys.exit(1)

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

        env = get_inference_running_env(self._model_instance.gpu_indexes)
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

    def hijack_tqdm_progress(server_self):
        """
        Monkey patch the tqdm progress bar to update the model instance download progress.
        tqdm is used by hf_hub_download under the hood.
        """
        from tqdm import tqdm

        _original_init = tqdm.__init__
        _original_update = tqdm.update

        # Record summary to work with multiple tracking progress.
        # For cases:
        # - Download split GGUF model files.
        # - Download the entire Huggingface repo.
        global model_total, model_n
        model_total, model_n = 0, 0

        def _new_init(self: tqdm, *args, **kwargs):
            global model_total, model_n
            kwargs["disable"] = False  # enable the progress bar anyway
            _original_init(self, *args, **kwargs)

            model_n += self.n
            model_total += self.total

        def _new_update(self: tqdm, n=1):
            global model_total, model_n
            _original_update(self, n)
            model_n += n

            try:
                patch_dict = {
                    "download_progress": round(
                        (float(model_n) / float(model_total)) * 100, 2
                    )
                }
                server_self._update_model_instance_set(
                    server_self._model_instance, **patch_dict
                )
            except Exception as e:
                logger.error(f"Failed to update model instance: {e}")

        tqdm.__init__ = _new_init
        tqdm.update = _new_update

    def _update_model_instance_set(self, mi: ModelInstance, **kwargs):
        """
        Update model instances of the same model and worker.
        So that they can share the same download progress.
        """

        instances = self._clientset.model_instances.list(
            {
                "model_id": mi.model_id,
                "worker_id": mi.worker_id,
                "state": ModelInstanceStateEnum.DOWNLOADING.value,
            }
        )

        for instance in instances.items:
            mi = ModelInstanceUpdate(**instance.model_dump())
            for key, value in kwargs.items():
                setattr(mi, key, value)
            self._clientset.model_instances.update(id=instance.id, model_update=mi)

    def _until_model_instance_initializing(self):
        for _ in range(5):
            mi = self._clientset.model_instances.get(id=self._model_instance.id)
            if mi.state == ModelInstanceStateEnum.INITIALIZING:
                return
            time.sleep(1)

        raise Exception("Timeout waiting for model instance to be initializing.")

    def _update_model_instance(self, id: str, **kwargs):
        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            setattr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)


def cuda_driver_installed() -> bool:
    try:
        result = subprocess.run(
            ['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def get_inference_running_env(gpu_indexes: List[int] = None):
    env = os.environ.copy()
    system = platform.system()

    if system == "Darwin":
        return None
    elif (system == "Linux" or system == "Windows") and gpu_indexes:
        env["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_indexes])
        return env
    else:
        # TODO: support more.
        return None


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
