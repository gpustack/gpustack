import asyncio
import logging
import os
import sys
import threading
import time
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceUpdate,
    SourceEnum,
    ModelInstanceStateEnum,
    get_backend,
    get_mmproj_filename,
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
    VendorEnum.AMD: "ROCR_VISIBLE_DEVICES",
    VendorEnum.Hygon: "HIP_VISIBLE_DEVICES",
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
    model: Model,
    cache_dir: Optional[str] = None,
    ollama_library_base_url: Optional[str] = None,
    huggingface_token: Optional[str] = None,
) -> str:
    if model.source == SourceEnum.HUGGING_FACE:
        return HfDownloader.download(
            repo_id=model.huggingface_repo_id,
            filename=model.huggingface_filename,
            extra_filename=get_mmproj_filename(model),
            token=huggingface_token,
            cache_dir=os.path.join(cache_dir, "huggingface"),
        )
    elif model.source == SourceEnum.OLLAMA_LIBRARY:
        ollama_downloader = OllamaLibraryDownloader(
            registry_url=ollama_library_base_url
        )
        return ollama_downloader.download(
            model_name=model.ollama_library_model_name,
            cache_dir=os.path.join(cache_dir, "ollama"),
        )
    elif model.source == SourceEnum.MODEL_SCOPE:
        return ModelScopeDownloader.download(
            model_id=model.model_scope_model_id,
            file_path=model.model_scope_file_path,
            extra_file_path=get_mmproj_filename(model),
            cache_dir=os.path.join(cache_dir, "model_scope"),
        )
    elif model.source == SourceEnum.LOCAL_PATH:
        return model.local_path


def get_model_file_size(model: Model, cfg: Config) -> Optional[int]:
    if model.source == SourceEnum.HUGGING_FACE:
        return HfDownloader.get_model_file_size(
            model=model,
            token=cfg.huggingface_token,
        )
    elif model.source == SourceEnum.MODEL_SCOPE:
        return ModelScopeDownloader.get_model_file_size(
            model=model,
        )

    return None


def get_file_size(
    huggingface_repo_id: Optional[str] = None,
    huggingface_filename: Optional[str] = None,
    model_scope_model_id: Optional[str] = None,
    model_scope_file_path: Optional[str] = None,
    huggingface_token: Optional[str] = None,
) -> str:
    if huggingface_repo_id is not None:
        return HfDownloader.get_file_size(
            repo_id=huggingface_repo_id,
            filename=huggingface_filename,
            token=huggingface_token,
        )
    elif model_scope_model_id is not None:
        return ModelScopeDownloader.get_file_size(
            model_id=model_scope_model_id,
            file_path=model_scope_file_path,
        )


class InferenceServer(ABC):
    @time_decorator
    def __init__(
        self,
        clientset: ClientSet,
        mi: ModelInstance,
        cfg: Config,
    ):
        setup_logging(debug=cfg.debug)

        try:
            self._clientset = clientset
            self._model_instance = mi
            self._config = cfg
            self.get_model()

            model_file_size = get_model_file_size(self._model, cfg)
            if model_file_size:
                logger.debug(f"Model file size: {model_file_size}")
                self._model_file_size = model_file_size
                self._model_downloaded_size = 0

            # for download progress update frequency control
            self._last_download_update_time = time.time()
            self.hijack_tqdm_progress()

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
                self._model,
                cfg.cache_dir,
                ollama_library_base_url=cfg.ollama_library_base_url,
                huggingface_token=cfg.huggingface_token,
            )

            patch_dict = {
                "state": ModelInstanceStateEnum.STARTING,
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

    def get_model(self):
        model = self._clientset.models.get(id=self._model_instance.model_id)
        data_dir = self._config.data_dir
        for i, param in enumerate(model.backend_parameters):
            model.backend_parameters[i] = param.replace("{data_dir}", data_dir)

        self._model = model

    def exit_with_code(self, exit_code: int):
        if exit_code < 0:
            signal_number = -exit_code
            exit_code = 128 + signal_number
        sys.exit(exit_code)

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
        gpu_indexes: List[int] = None, gpu_devices: GPUDevicesInfo = None, backend=None
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
            set_vllm_env(env, vendor, backend, gpu_indexes, gpu_devices)

            return env
        else:
            # TODO: support more.
            return None


def set_vllm_env(
    env: Dict[str, str],
    vendor: VendorEnum,
    backend: str,
    gpu_indexes: List[int] = None,
    gpu_devices: GPUDevicesInfo = None,
):

    system = platform.system()
    if not gpu_indexes or not gpu_devices:
        return

    if system != "linux" or vendor != VendorEnum.AMD or backend != "vllm":
        return

    llvm = None
    for g in gpu_devices:
        if (
            g.index in gpu_indexes
            and g.labels.get("llvm")
            and g.vendor == VendorEnum.AMD
        ):
            llvm = g.labels.get("llvm")
            break

    if llvm:
        # vllm supports llvm target: gfx908;gfx90a;gfx942;gfx1100,
        # try to use the similar LLVM target that is supported.
        # https://docs.vllm.ai/en/v0.6.2/getting_started/amd-installation.html#build-from-source-rocm
        # https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
        emulate_gfx_version = {
            "gfx1101": "11.0.0",
            "gfx1102": "11.0.0",
        }

        if emulate_gfx_version.get(llvm):
            env["HSA_OVERRIDE_GFX_VERSION"] = emulate_gfx_version[llvm]


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
