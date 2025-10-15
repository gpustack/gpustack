import glob
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from gpustack_runner import list_service_runners
from gpustack_runtime.deployer.__utils__ import compare_versions, correct_runner_image
from gpustack_runtime.detector import (
    manufacturer_to_backend,
    detect_devices,
    ManufacturerEnum,
)
from gpustack_runtime.detector.ascend import get_ascend_cann_variant

from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config, set_global_config, get_global_config
from gpustack.logging import setup_logging
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    get_backend,
)
from gpustack.schemas.workers import VendorEnum, GPUDevicesInfo, WorkerBase
from gpustack.server.bus import Event
from gpustack.utils.gpu import all_gpu_match
from gpustack.utils.profiling import time_decorator
from gpustack.utils import platform, envs
from gpustack_runtime.logging import setup_logging as setup_runtime_logging

logger = logging.getLogger(__name__)
lock = threading.Lock()

_VISIBLE_DEVICES_ENV_NAME_MAPPER = {
    VendorEnum.NVIDIA: "CUDA_VISIBLE_DEVICES",
    VendorEnum.Huawei: "ASCEND_RT_VISIBLE_DEVICES",
    VendorEnum.AMD: "ROCR_VISIBLE_DEVICES",
    VendorEnum.Hygon: "HIP_VISIBLE_DEVICES",
    VendorEnum.Iluvatar: "CUDA_VISIBLE_DEVICES",
    VendorEnum.Cambricon: "MLU_VISIBLE_DEVICES",
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
        worker_id: int,
        inference_backend: InferenceBackend,
    ):
        setup_logging(debug=cfg.debug)
        setup_runtime_logging()
        set_global_config(cfg)

        try:
            self._clientset = clientset
            self._model_instance = mi
            self._config = cfg
            self._worker = self._clientset.workers.get(worker_id)

            self.get_model()
            self.inference_backend = inference_backend
            if (
                not inference_backend
                and self._model.image_name
                and self._model.run_command
            ):
                # Any deployment that directly specifies an image and command is treated as a Custom backend.
                # A basic InferenceBackend object is created to prevent exceptions in subsequent workflows.
                self.inference_backend = InferenceBackend(
                    backend_name=BackendEnum.CUSTOM.value,
                    run_command=self._model.run_command,
                )

            logger.info("Preparing model files...")

            self._until_model_instance_starting()

            logger.info("Model files are ready.")
        except ModelInstanceStateError:
            sys.exit(1)
        except Exception as e:
            error_message = f"Failed to initialize: {e}"
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
            self._model_path = str(Path(event.data["resolved_path"]).absolute())
            return True

        return False

    @abstractmethod
    def start(self):
        pass

    def get_model(self):
        model = self._clientset.models.get(id=self._model_instance.model_id)
        data_dir = self._config.data_dir
        for i, param in enumerate(model.backend_parameters or []):
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

    def _update_model_instance(self, id: int, **kwargs):
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

            # Keep GPU indexes in ascending order,
            # this is important for some vendors like Ascend CANN,
            # see https://github.com/gpustack/gpustack/issues/2527.
            env_name = get_visible_devices_env_name(vendor)
            env[env_name] = ",".join(map(str, sorted(gpu_indexes)))

            if get_backend(self._model) == BackendEnum.VLLM:
                set_vllm_env(env, vendor, gpu_indexes, gpu_devices)
            elif get_backend(self._model) == BackendEnum.ASCEND_MINDIE:
                # Enable service monitor mode for Ascend MindIE
                # https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0316.html
                env["MIES_SERVICE_MONITOR_MODE"] = "1"

                set_ascend_mindie_env(env, vendor, gpu_indexes, gpu_devices, version)

        env.update(self._model.env or {})

        return env

    def build_versioned_command_args(
        self,
        default_args: List[str],
        model_path: Optional[str] = None,
        port: Optional[int] = None,
    ) -> List[str]:
        """
        Override default startup arguments based on version configuration
        when the version uses non-built-in version and defines a custom run_command

        Args:
        - default_args: The default command argument list (e.g., ["vllm", "serve", "/path/to/model"]).
        - model_path: Path used to replace {{model_path}}; if None, fall back to self._model_path.
        - port: Port used to replace {{port}}; if None, fall back to self._model_instance.port.

        Returns:
            The final command argument list used for container execution.
        """

        # if no version or inference backend is available, return default_args
        version = getattr(self._model, "backend_version", None)
        if not version or not getattr(self, "inference_backend", None):
            return default_args

        # Load version configuration
        version_config = None
        try:
            version_config = self.inference_backend.get_version_config(version)
        except Exception:
            version_config = self.inference_backend.version_configs.root.get(version)

        # Only perform replacement when the version uses non-built-in version and defines run_command
        if (
            version_config
            and getattr(version_config, "built_in_frameworks", None) is None
            and getattr(version_config, "run_command", None)
        ):
            resolved_model_path = (
                model_path
                if model_path is not None
                else getattr(self, "_model_path", None)
            )
            resolved_port = (
                port
                if port is not None
                else getattr(self._model_instance, "port", None)
            )

            command = self.inference_backend.replace_command_param(
                version,
                resolved_model_path,
                resolved_port,
                version_config.run_command,
            )
            if command:
                return command.split(" ")

        # Return original default_args by default
        return default_args

    def _get_backend_image_name(self, backend_type: Optional[str] = None) -> str:
        """
        Get supported backend images from gpustack-runner.

        Args:
            backend_type: Optional backend type override (e.g., "cann" for MindIE)
                         If not provided, will be derived from GPU vendor

        Returns:
            Docker image name for the backend
        """
        # Get GPU vendor from the first GPU assigned to this model instance
        vendor = None
        if (
            self._model_instance.gpu_indexes
            and self._worker
            and self._worker.status.gpu_devices
        ):
            gpu_devices = self._worker.status.gpu_devices
            first_index = self._model_instance.gpu_indexes[0]
            gpu_device = next((d for d in gpu_devices if d.index == first_index), None)
            if gpu_device:
                vendor = gpu_device.vendor.lower()

        if vendor == "Huawei":
            vendor = ManufacturerEnum.ASCEND
        # Determine backend_type if not provided
        if backend_type is None:
            backend_type = manufacturer_to_backend(vendor) if vendor else None

        # Get supported images from gpustack-runner
        runtime_version = ""
        try:
            devices = detect_devices()
            runtime_version = next(
                (
                    device.runtime_version
                    for device in devices
                    if device.manufacturer == vendor
                ),
                "",
            )
            logger.debug(f"runtime_version: {runtime_version}")
        except Exception as e:
            logger.error(f"Failed to detect devices: {e}")
        runner_param = {
            "backend": backend_type.lower() if backend_type else None,
            "service": self._model.backend.lower() if self._model.backend else None,
            "platform": platform.system_arch(),
        }
        if self._model.backend_version:
            runner_param["service_version"] = self._model.backend_version

        service_list = list_service_runners(**runner_param)

        docker_image = ""
        if self._model.image_name:
            docker_image = self._model.image_name
        elif service_list and len(service_list) > 0:
            service = service_list[0]
            logger.debug(f"Get {len(service.versions)} service runners")
            runner_info = next(
                (
                    (p.docker_image, v)
                    for v in service.versions
                    for b in v.backends
                    for b_ver in b.versions
                    if compare_versions(b_ver.version, runtime_version) <= 0
                    for b_var in b_ver.variants
                    for p in b_var.platforms
                    if p.docker_image
                ),
                ("", None),
            )
            if runner_info[0]:
                docker_image = runner_info[0]
            if runner_info[1] and not self._model.backend_version:
                self._model.backend_version = runner_info[1].version
                self._clientset.models.update(self._model.id, self._model)
        if not docker_image:
            docker_image = self.inference_backend.get_image_name(
                self._model.backend_version
            )

        if not docker_image and self._model.backend_version:
            docker_image = f"gpustack/runner:{backend_type}{runtime_version}-{self._model.backend}{self._model.backend_version}"

        # Finally, invoke the runtime to make a correction to the docker_image.
        if docker_image:
            docker_image = correct_runner_image(docker_image)

        logger.info(f"{self._model.backend} image name: {docker_image}")

        return docker_image


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
    vendor: ManufacturerEnum,
    gpu_indexes: List[int] = None,
    gpu_devices: GPUDevicesInfo = None,
):
    system = platform.system()
    if not gpu_indexes or not gpu_devices:
        return

    if system != "linux" or vendor != ManufacturerEnum.AMD:
        return

    cc = None
    for g in gpu_devices:
        if g.index in gpu_indexes and g.compute_compatibility:
            cc = g.compute_compatibility
            break

    if cc:
        # vllm supports llvm target: gfx908;gfx90a;gfx942;gfx1100,
        # try to use the similar LLVM target that is supported.
        # https://docs.vllm.ai/en/v0.6.2/getting_started/amd-installation.html#build-from-source-rocm
        # https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
        emulate_gfx_version = {
            "gfx1101": "11.0.0",
            "gfx1102": "11.0.0",
        }

        if emulate_gfx_version.get(cc):
            env["HSA_OVERRIDE_GFX_VERSION"] = emulate_gfx_version[cc]


def set_ascend_mindie_env(
    env: Dict[str, str],
    vendor: ManufacturerEnum,
    gpu_indexes: List[int] = None,
    gpu_devices: GPUDevicesInfo = None,
    version: str = "latest",
):
    system = platform.system()
    if not gpu_indexes or not gpu_devices:
        return

    if system != "linux" or vendor != ManufacturerEnum.ASCEND:
        return

    # Select root path
    root_path = next(
        (
            rp
            for rp in envs.get_unix_available_root_paths_of_ascend()
            if rp.joinpath("mindie", version).is_dir()
        ),
        None,
    )
    if not root_path:
        logger.error(
            "Ascend MindIE root path not found. " "Please check the installation."
        )
        return

    # Extract the environment variables from the script
    # - Get the paths of Ascend MindIE set_env.sh
    script_paths = [
        root_path.joinpath("mindie", version, "mindie-rt", "set_env.sh"),
        root_path.joinpath("mindie", version, "mindie-torch", "set_env.sh"),
        root_path.joinpath("mindie", version, "mindie-service", "set_env.sh"),
        root_path.joinpath("mindie", version, "mindie-llm", "set_env.sh"),
    ]
    # - Get the paths of Ascend MindIE virtual environment if needed
    cfg = get_global_config()
    venv_dir = Path(cfg.data_dir).joinpath("venvs", "mindie", version)
    venv_path = venv_dir.joinpath("bin", "activate")
    if venv_dir.is_dir() and venv_path.is_file():
        script_paths.append(venv_path)

    # Update the environment variables with diff
    env_diff = envs.extract_unix_vars_of_source(script_paths)
    env.update(env_diff)


def get_visible_devices_env_name(vendor: str) -> str:
    """
    Get the environment variable name for visible devices based on the vendor.
    For example, if the vendor is "NVIDIA", it returns "CUDA_VISIBLE_DEVICES",
    if the vendor is "Huawei", it returns "ASCEND_RT_VISIBLE_DEVICES".
    Return "CUDA_VISIBLE_DEVICES" if the vendor is not recognized.
    """

    return next(
        (
            v
            for k, v in _VISIBLE_DEVICES_ENV_NAME_MAPPER.items()
            if vendor is not None and k.value.lower() in vendor.lower()
        ),
        "CUDA_VISIBLE_DEVICES",
    )


def is_ascend_310p(worker: WorkerBase) -> bool:
    """
    Check if the model instance is running on VLLM Ascend 310P.
    """

    return all_gpu_match(
        worker,
        lambda gpu: (
            gpu.vendor == ManufacturerEnum.ASCEND.value
            and get_ascend_cann_variant(gpu.arch_family) == "310p"
        ),
    )
