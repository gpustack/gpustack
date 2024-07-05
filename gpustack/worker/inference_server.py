import asyncio
import logging
import platform
import subprocess
import sys
import time
from typing import Optional

from gpustack.client.generated_clientset import ClientSet
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
            logger.info(f"{func.__name__} execution time: {end_time - start_time} s")
            return result

        return async_wrapper
    else:

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} execution time: {end_time - start_time} s")
            return result

        return sync_wrapper


def download_model(mi: ModelInstance) -> str:
    if mi.source == SourceEnum.HUGGING_FACE:
        return HfDownloader.download(
            repo_id=mi.huggingface_repo_id,
            filename=mi.huggingface_filename,
        )
    elif mi.source == SourceEnum.OLLAMA_LIBRARY:
        return OllamaLibraryDownloader.download(model_name=mi.ollama_library_model_name)


class InferenceServer:
    @time_decorator
    def __init__(self, clientset: ClientSet, mi: ModelInstance):
        self.hijack_tqdm_progress()

        self._clientset = clientset
        self._model_instance = mi
        try:
            patch_dict = {
                "download_progress": 0,
                "state": ModelInstanceStateEnum.downloading,
            }
            self._update_model_instance(mi.id, **patch_dict)

            self._model_path = download_model(mi)

            patch_dict = {"state": ModelInstanceStateEnum.running}
            self._update_model_instance(mi.id, **patch_dict)
        except Exception as e:
            try:
                patch_dict = {
                    "state_message": str(e),
                    "state": ModelInstanceStateEnum.error,
                }
                self._update_model_instance(mi.id, **patch_dict)
            except Exception as e:
                logger.error(f"Failed to update model instance: {e}")

            raise e

    def start(self):
        command_path = pkg_resources.files(
            "gpustack.third_party.bin.llama-box"
        ).joinpath(self._get_command())

        layers = -1
        claim = self._model_instance.computed_resource_claim
        if claim is not None and claim.get("offload_layers") is not None:
            layers = claim.get("offload_layers")

        env = self._get_env(self._model_instance.gpu_index)

        arguments = [
            "--host",
            "0.0.0.0",
            "--gpu-layers",
            str(layers),
            "--parallel",
            "5",
            "--port",
            str(self._model_instance.port),
            "--model",
            self._model_path,
        ]

        try:
            subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )
        except Exception as e:
            logger.error(f"Failed to run the llama.cpp server: {e}")

    def _get_env(self, gpu_index: Optional[int] = None):
        index = gpu_index or 0
        system = platform.system()

        if system == "Darwin":
            return None
        elif system == "Linux":
            return {"CUDA_VISIBLE_DEVICES": str(index)}
        else:
            # TODO: support more.
            return None

    def _get_command(self):
        command_map = {
            ("Windows", "amd64"): "llama-box-windows-amd64-cuda-12.5.exe",
            ("Darwin", "amd64"): "llama-box-darwin-amd64-metal",
            ("Darwin", "arm64"): "llama-box-darwin-arm64-metal",
            ("Linux", "amd64"): "llama-box-linux-amd64-cuda-12.5",
        }

        command = get_platform_command(command_map)
        if command == "":
            raise Exception(
                f"No supported llama-box command found "
                f"for {platform.system()} {platform.machine()}."
            )
        return command

    def hijack_tqdm_progress(server_self):
        """
        Monkey patch the tqdm progress bar to update the model instance download progress.
        tqdm is used by hf_hub_download under the hood.
        """
        from tqdm import tqdm

        _original_init = tqdm.__init__
        _original_update = tqdm.update

        def _new_init(self: tqdm, *args, **kwargs):
            kwargs["disable"] = False  # enable the progress bar anyway
            _original_init(self, *args, **kwargs)

        def _new_update(self: tqdm, n=1):
            _original_update(self, n)

            try:
                patch_dict = {
                    "download_progress": round(
                        (float(self.n) / float(self.total)) * 100, 2
                    )
                }
                server_self._update_model_instance(
                    server_self._model_instance.id, **patch_dict
                )
            except Exception as e:
                logger.error(f"Failed to update model instance: {e}")

        tqdm.__init__ = _new_init
        tqdm.update = _new_update

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
