import asyncio
import logging
import os
import sys
import time
from typing import List, Optional
from abc import ABC, abstractmethod

from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceUpdate,
    SourceEnum,
    ModelInstanceStateEnum,
)
from gpustack.utils import platform
from gpustack.worker.downloaders import (
    HfDownloader,
    ModelScopeDownloader,
    OllamaLibraryDownloader,
)

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
    elif mi.source == SourceEnum.MODEL_SCOPE:
        return ModelScopeDownloader.download(
            model_id=mi.model_scope_model_id,
            file_path=mi.model_scope_file_path,
            cache_dir=os.path.join(cache_dir, "model_scope"),
        )


class InferenceServer(ABC):
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
            self._model = self._clientset.models.get(id=mi.model_id)
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

    @abstractmethod
    def start(self):
        pass

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

    @staticmethod
    def get_inference_running_env(gpu_indexes: List[int] = None):
        env = os.environ.copy()
        system = platform.system()

        if system == "darwin":
            return None
        elif (system == "linux" or system == "windows") and gpu_indexes:
            env["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_indexes])
            return env
        else:
            # TODO: support more.
            return None
