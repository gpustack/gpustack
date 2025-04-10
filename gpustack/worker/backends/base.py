import glob
import logging
import os
import sys
import threading
from typing import Dict, List
from abc import ABC, abstractmethod

from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config, set_global_config
from gpustack.logging import setup_logging
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    get_backend,
)
from gpustack.schemas.workers import VendorEnum, GPUDevicesInfo
from gpustack.server.bus import Event
from gpustack.utils.profiling import time_decorator
from gpustack.utils import platform, envs
from gpustack.worker.tools_manager import ToolsManager

logger = logging.getLogger(__name__)
lock = threading.Lock()

ACCELERATOR_VENDOR_TO_ENV_NAME = {
    VendorEnum.NVIDIA: "CUDA_VISIBLE_DEVICES",
    VendorEnum.Huawei: "ASCEND_RT_VISIBLE_DEVICES",
    VendorEnum.AMD: "ROCR_VISIBLE_DEVICES",
    VendorEnum.Hygon: "HIP_VISIBLE_DEVICES",
}


class ModelInstanceStateError(Exception):
    pass


class InferenceServer(ABC):
    @time_decorator
    def __init__(
        self,
        clientset: ClientSet,
        mi: ModelInstance,
        cfg: Config,
    ):
        setup_logging(debug=cfg.debug)
        set_global_config(cfg)

        try:
            self._clientset = clientset
            self._model_instance = mi
            self._config = cfg
            self._worker = self._clientset.workers.get(self._model_instance.worker_id)

            self.get_model()

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

            logger.info("Preparing model files...")

            self._until_model_instance_starting()

            logger.info("Model files are ready.")
        except ModelInstanceStateError:
            sys.exit(1)
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

    def _stop_when_starting(self, event: Event) -> bool:
        if event.data["state"] == ModelInstanceStateEnum.ERROR:
            raise ModelInstanceStateError()
        elif event.data["state"] == ModelInstanceStateEnum.STARTING:
            self._model_path = event.data["resolved_path"]
            return True

        return False

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

    def _until_model_instance_starting(self):
        self._clientset.model_instances.watch(
            callback=None,
            stop_condition=self._stop_when_starting,
            params={"id": self._model_instance.id},
        )

    def _update_model_instance(self, id: str, **kwargs):
        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            setattr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)

    def get_inference_running_env(
        self, env: Dict[str, str] = None, version: str = None
    ) -> Dict[str, str]:
        if env is None:
            env = os.environ.copy()

        system = platform.system()
        gpu_indexes = self._model_instance.gpu_indexes
        if (system == "linux" or system == "windows") and gpu_indexes:
            vendor = None
            gpu_devices = None
            if self._worker and self._worker.status.gpu_devices:
                gpu_devices = self._worker.status.gpu_devices

            if gpu_devices:
                # Now use the first GPU index to get the vendor
                first_index = gpu_indexes[0]
                gpu_device = next(
                    (d for d in gpu_devices if d.index == first_index), None
                )
                vendor = gpu_device.vendor if gpu_device else None

            env_name = get_env_name_by_vendor(vendor)
            env[env_name] = ",".join([str(i) for i in gpu_indexes])

            if get_backend(self._model) == BackendEnum.VLLM:
                set_vllm_env(env, vendor, gpu_indexes, gpu_devices)
            elif get_backend(self._model) == BackendEnum.ASCEND_MINDIE:
                set_ascend_mindie_env(env, vendor, gpu_indexes, gpu_devices, version)

        env.update(self._model.env or {})

        return env


def real_model_path(model_paths: List[str]) -> str:
    """
    Get the real model path from the resolved paths.
    """
    if len(model_paths) == 0:
        raise ValueError("Model paths are empty.")

    model_path = model_paths[0]
    # resolve glob pattern
    if "*" in model_path:
        match_paths = glob.glob(model_path)
        if len(match_paths) == 0:
            raise ValueError(f"No match file found for {model_path}")
        return sorted(match_paths)[0]


def set_vllm_env(
    env: Dict[str, str],
    vendor: VendorEnum,
    gpu_indexes: List[int] = None,
    gpu_devices: GPUDevicesInfo = None,
):
    if 'VLLM_USE_V1' not in env:
        # Revisit this when vllm v1 engine is stable
        # Ref: https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html
        # Known issues:
        # - https://github.com/vllm-project/vllm/issues/16141
        env['VLLM_USE_V1'] = "0"

    system = platform.system()
    if not gpu_indexes or not gpu_devices:
        return

    if system != "linux" or vendor != VendorEnum.AMD:
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


def set_ascend_mindie_env(
    env: Dict[str, str],
    vendor: VendorEnum,
    gpu_indexes: List[int] = None,
    gpu_devices: GPUDevicesInfo = None,
    version: str = None,
):
    system = platform.system()
    if not gpu_indexes or not gpu_devices:
        return

    if system != "linux" or vendor != VendorEnum.Huawei:
        return

    if not version:
        version = "1.0.0"

    # Select root path
    root_path = next(
        (
            rp
            for rp in envs.get_unix_available_root_paths_of_ascend()
            if rp.joinpath("mindie", version).is_dir()
        ),
        None,
    )

    # Get the paths of mindie set_env.sh
    mindie_rt_env_script = root_path.joinpath(
        "mindie", version, "mindie-rt", "set_env.sh"
    )
    mindie_torch_env_script = root_path.joinpath(
        "mindie", version, "mindie-torch", "set_env.sh"
    )
    mindie_service_env_script = root_path.joinpath(
        "mindie", version, "mindie-service", "set_env.sh"
    )
    mindie_llm_env_script = root_path.joinpath(
        "mindie", version, "mindie-llm", "set_env.sh"
    )

    # Extract the environment variables from the script
    env_diff = envs.extract_unix_vars_of_source(
        [
            mindie_rt_env_script,
            mindie_torch_env_script,
            mindie_service_env_script,
            mindie_llm_env_script,
        ]
    )

    # Update the environment variables
    env.update(env_diff)


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
