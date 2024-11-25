import asyncio
import logging
import os
import sys
import threading
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
    get_backend,
)
from gpustack.schemas.workers import VendorEnum, GPUDevicesInfo
from gpustack.utils import platform
from gpustack.worker.downloaders import (
    HfDownloader,
    ModelScopeDownloader,
    OllamaLibraryDownloader,
)
from gpustack.worker.tools_manager import ToolsManager

logger = logging.getLogger(__name__)
lock = threading.Lock()

ACCELERATOR_VENDOR_TO_ENV_NAME = {
    VendorEnum.NVIDIA: "CUDA_VISIBLE_DEVICES",
    VendorEnum.Huawei: "ASCEND_RT_VISIBLE_DEVICES",
}


def time_decorator(func):
    """
    A decorator that logs the execution time of a function.
    """

    if asyncio.iscoroutinefunction(func):

        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            logger.debug(
                f"{func.__name__} execution time: {end_time - start_time:.2f} seconds"
            )
            return result

        return async_wrapper
    else:

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(
                f"{func.__name__} execution time: {end_time - start_time} seconds"
            )
            return result

        return sync_wrapper


def download_model(
    mi: ModelInstance,
    cache_dir: Optional[str] = None,
    ollama_library_base_url: Optional[str] = None,
    huggingface_token: Optional[str] = None,
) -> str:
    if mi.source == SourceEnum.HUGGING_FACE:
        return HfDownloader.download(
            repo_id=mi.huggingface_repo_id,
            filename=mi.huggingface_filename,
            token=huggingface_token,
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
    elif mi.source == SourceEnum.LOCAL_PATH:
        return mi.local_path


def get_model_file_size(mi: ModelInstance, cfg: Config) -> Optional[int]:
    if mi.source == SourceEnum.HUGGING_FACE:
        return HfDownloader.get_model_file_size(
            model_instance=mi,
            token=cfg.huggingface_token,
        )
    elif mi.source == SourceEnum.MODEL_SCOPE:
        return ModelScopeDownloader.get_model_file_size(
            model_instance=mi,
        )

    return None


class InferenceServer(ABC):
    @time_decorator
    def __init__(
        self,
        clientset: ClientSet,
        mi: ModelInstance,
        cfg: Config,
    ):
        setup_logging(debug=cfg.debug)

        model_file_size = get_model_file_size(mi, cfg)
        if model_file_size:
            logger.debug(f"Model file size: {model_file_size}")
            self._model_file_size = model_file_size
            self._model_downloaded_size = 0
        # for download progress update frequency control
        self._last_download_update_time = time.time()
        self.hijack_tqdm_progress()

        self._clientset = clientset
        self._model_instance = mi
        self._config = cfg
        try:
            self._model = self._clientset.models.get(id=mi.model_id)
            self._until_model_instance_initializing()

            if self._model.backend_version:
                tools_manager = ToolsManager(
                    tools_download_base_url=cfg.tools_download_base_url,
                    bin_dir=cfg.bin_dir,
                    pipx_path=cfg.pipx_path,
                )
                backend = get_backend(self._model)
                tools_manager.prepare_versioned_backend(
                    backend, self._model.backend_version
                )

            patch_dict = {
                "download_progress": 0,
                "state": ModelInstanceStateEnum.DOWNLOADING,
            }
            self._update_model_instance(mi.id, **patch_dict)

            self._model_path = download_model(
                mi,
                cfg.cache_dir,
                ollama_library_base_url=cfg.ollama_library_base_url,
                huggingface_token=cfg.huggingface_token,
            )

            patch_dict = {
                "state": ModelInstanceStateEnum.RUNNING,
                "state_message": "",
            }
            self._update_model_instance(mi.id, **patch_dict)
        except Exception as e:
            error_message = f"Failed to initilialze: {e}"
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

        def _new_init(self: tqdm, *args, **kwargs):
            kwargs["disable"] = False  # enable the progress bar anyway
            _original_init(self, *args, **kwargs)

            if hasattr(server_self, '_model_file_size'):
                # Resume downloading
                server_self._model_downloaded_size += self.n

        def _new_update(self: tqdm, n=1):
            _original_update(self, n)

            # This is the default for single tqdm downloader like ollama
            # TODO we may want to unify to always get the size before downloading.
            total_size = self.total
            downloaded_size = self.n
            if hasattr(server_self, '_model_file_size'):
                # This is summary for group downloading
                total_size = server_self._model_file_size
                server_self._model_downloaded_size += n
                downloaded_size = server_self._model_downloaded_size

            with lock:
                try:
                    if (
                        time.time() - server_self._last_download_update_time < 2
                        and downloaded_size != total_size
                    ):
                        # Only update after 2-second interval or download is completed.
                        return

                    patch_dict = {
                        "download_progress": round(
                            (float(downloaded_size) / float(total_size)) * 100, 2
                        )
                    }
                    server_self._update_model_instance_set(
                        server_self._model_instance, **patch_dict
                    )
                    server_self._last_download_update_time = time.time()
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
    def get_inference_running_env(
        gpu_indexes: List[int] = None, gpu_devices: GPUDevicesInfo = None
    ):
        env = os.environ.copy()
        system = platform.system()

        if system == "darwin":
            return None
        elif (system == "linux" or system == "windows") and gpu_indexes:
            vendor = None
            if gpu_devices:
                # Now use the first GPU index to get the vendor
                first_index = gpu_indexes[0]
                gpu_device = next(
                    (d for d in gpu_devices if d.index == first_index), None
                )
                vendor = gpu_device.vendor if gpu_device else None

            env_name = get_env_name_by_vendor(vendor)
            env[env_name] = ",".join([str(i) for i in gpu_indexes])
            return env
        else:
            # TODO: support more.
            return None


def get_env_name_by_vendor(vendor: str) -> str:
    env_name = next(
        (
            v
            for k, v in ACCELERATOR_VENDOR_TO_ENV_NAME.items()
            if vendor is not None and k.value.lower() in vendor.lower()
        ),
        "CUDA_VISIBLE_DEVICES",
    )

    return env_name
