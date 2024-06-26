import asyncio
import logging
import platform
import subprocess
import sys
import time
import importlib.resources as pkg_resources

from gpustack.client.generated_clientset import ClientSet
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceUpdate,
    SourceEnum,
    ModelInstanceStateEnum,
)
from gpustack.worker.downloaders import HfDownloader, OllamaLibraryDownloader


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
        command_path = pkg_resources.files("gpustack.third_party.llama_cpp").joinpath(
            self._get_command()
        )

        layers = -1
        claim = self._model_instance.computed_resource_claim
        if claim is not None and claim.get("offload_layers") is not None:
            layers = claim.get("offload_layers")

        env = self._get_env(self._model_instance.gpu_index)

        arguments = [
            "--host",
            "0.0.0.0",
            "--n-gpu-layers",
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

    def _get_env(self, gpu_index: int | None = None):
        index = gpu_index or 0
        match platform.system():
            case "Darwin":
                return None
            case "Linux":
                # TODO: support more.
                return {"CUDA_VISIBLE_DEVICES": str(index)}

    def _get_command(self):
        command = ""

        match platform.system():
            case "Darwin":
                if "amd64" in platform.machine() or "x86_64" in platform.machine():
                    command = "server-macos-x64"
                elif "arm" in platform.machine() or "aarch64" in platform.machine():
                    command = "server-macos-arm64"
            case "Linux":
                if "amd64" in platform.machine() or "x86_64" in platform.machine():
                    command = "server-ubuntu-x64"

        if command == "":
            raise ValueError(
                "Unsupported platform: %s %s" % (platform.system(), platform.machine())
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
