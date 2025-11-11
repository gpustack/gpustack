import logging
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, List
from abc import ABC, abstractmethod

from gpustack_runner import list_backend_runners
from gpustack_runner.runner import BackendVersionedRunner
from gpustack_runtime.deployer import ContainerResources, ContainerMount, ContainerPort
from gpustack_runtime.deployer.__utils__ import compare_versions
from gpustack_runtime.detector import (
    manufacturer_to_backend,
    ManufacturerEnum,
    backend_to_manufacturer,
)
from gpustack_runtime.detector.ascend import get_ascend_cann_variant
from gpustack_runtime import envs as runtime_envs
from gpustack_runtime.logging import setup_logging as setup_runtime_logging

from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config, set_global_config
from gpustack.logging import setup_logging
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    ModelUpdate,
)
from gpustack.schemas.workers import GPUDevicesInfo
from gpustack.server.bus import Event
from gpustack.utils.gpu import parse_gpu_id
from gpustack.utils.hub import (
    get_pretrained_config,
    get_hf_text_config,
    get_max_model_len,
)
from gpustack.utils.profiling import time_decorator
from gpustack.utils import platform

logger = logging.getLogger(__name__)
lock = threading.Lock()


class ModelInstanceStateError(Exception):
    pass


class InferenceServer(ABC):
    _model_path: Optional[str] = None
    _draft_model_path: Optional[str] = None
    """
    The absolute path to the model files.
    This is set when the model instance state changes to STARTING.
    """

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
            if not self._worker:
                raise KeyError(f"Worker {worker_id} not found")

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
            if not self.inference_backend:
                raise KeyError(
                    f"Inference backend {self._model.backend} not specified or not found"
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
            if event.data["draft_model_resolved_path"]:
                self._draft_model_path = str(
                    Path(event.data["draft_model_resolved_path"]).absolute()
                )
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

    def _handle_error(self, error: Exception):
        """
        Handle errors during backend server startup in a unified way.
        Updates model instance state and re-raises the original error.
        """
        cause = getattr(error, "__cause__", None)
        cause_text = f": {cause}" if cause else ""
        error_message = f"Failed to run {self._model.backend}: {error}{cause_text}"

        try:
            patch_dict = {
                "state_message": error_message,
                "state": ModelInstanceStateEnum.ERROR,
            }
            self._update_model_instance(self._model_instance.id, **patch_dict)
        except Exception as ue:
            logger.error(f"Failed to update model instance: {ue}")

        raise error

    def _get_distributed_metadata(self) -> tuple[bool, bool, bool]:
        """
        Return distributed deployment status and role flags.

        Returns:
            A tuple indicating (
                is_distributed,
                is_distributed_leader,
                is_distributed_follower
            ).
        """
        model_instance = self._model_instance
        dservers = model_instance.distributed_servers
        subworkers = (
            dservers.subordinate_workers
            if dservers and dservers.subordinate_workers
            else []
        )
        is_distributed = bool(subworkers)
        is_distributed_leader = (
            is_distributed and model_instance.worker_id == self._worker.id
        )
        is_distributed_follower = is_distributed and not is_distributed_leader
        return is_distributed, is_distributed_leader, is_distributed_follower

    def _derive_max_model_len(self) -> Optional[int]:
        """
        Derive max model length from model config.
        Returns None if unavailable.
        """
        try:
            pretrained_config = get_pretrained_config(self._model)
            pretrained_or_hf_text_config = get_hf_text_config(pretrained_config)
            return get_max_model_len(pretrained_or_hf_text_config)
        except Exception as e:
            logger.error(f"Failed to derive max model length: {e}")

        return None

    def _get_configured_env(self, **kwargs) -> Dict[str, str]:
        """
        Get the environment variables for the model instance.
        Merge the model's env with the system env.
        If there are conflicts, the model's env takes precedence.

        Returns:
            A dictionary of environment variables for the model instance.
        """

        env = {}
        if not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT:
            env = {
                # Exclude the following env vars,
                # which are reserved for gpustack internal use.
                # - start with GPUSTACK_, PIP_, PIPX_, UV_, POETRY_, S6_, PGCONFIG_, POSTGRES_.
                # - end with _VISIBLE_DEVICES, _DISABLE_REQUIRE, _DRIVER_CAPABILITIES, _PATH.
                # - miscellaneous item.
                #
                # FIXME(thxCode): Make this configurable.
                k: v
                for k, v in os.environ.items()
                if not (
                    k.startswith(
                        (
                            "GPUSTACK_",
                            "PIP_",
                            "PIPX_",
                            "POETRY_",
                            "UV_",
                            "S6_",
                            "PGCONFIG_",
                            "POSTGRES_",
                        )
                    )
                    or k.endswith(
                        (
                            "_VISIBLE_DEVICES",
                            "_DISABLE_REQUIRE",
                            "_DRIVER_CAPABILITIES",
                            "_PATH",
                        )
                    )
                    or (
                        k
                        in (
                            "DEBIAN_FRONTEND",
                            "LANG",
                            "LANGUAGE",
                            "LC_ALL",
                            "PYTHON_VERSION",
                            "HOME",
                            "HOSTNAME",
                            "PWD",
                            "_",
                            "TERM",
                            "SHLVL",
                            "LS_COLORS",
                            "PATH",
                        )
                    )
                )
            }

        if self._model.env:
            env.update(self._model.env)

        return env

    @lru_cache
    def _get_selected_gpu_devices(self) -> GPUDevicesInfo:
        """
        Get the GPU devices assigned to the model instance.

        Returns:
            A list of GPU device information assigned to the model instance.
        """
        minstance = self._model_instance
        dservers = minstance.distributed_servers
        if (
            dservers
            and dservers.subordinate_workers
            and minstance.worker_id != self._worker.id
        ):
            subworker = next(
                (
                    w
                    for w in dservers.subordinate_workers
                    if w.worker_id == self._worker.id
                ),
                None,
            )
            gpu_indexes = sorted(subworker.gpu_indexes or [])
        else:
            gpu_indexes = sorted(self._model_instance.gpu_indexes or [])

        # When doing manual selection, the device type is further confirmed in the selection information.
        # This helps to find the correct item when there are multiple devices mixed in one node.
        # For example, a node includes both NVIDIA device and AMD device.
        #
        # FIXME(thxCode): Currently, there is not field to indicate the device vendor corresponding to a certain device index.
        #                 We should extend the processing of indexes selection, preserving both index and device type,
        #                 and support automatic selection as well.
        gpu_index_types: Dict[int, set[int]] = {}  # device index -> {device type}
        if self._model.gpu_selector and self._model.gpu_selector.gpu_ids:
            for gpu_id in self._model.gpu_selector.gpu_ids:
                is_valid, matched = parse_gpu_id(gpu_id)
                if not is_valid:
                    continue
                if matched.get("worker_name") != self._worker.name:
                    continue
                gpu_device_type = matched.get("device")
                gpu_index = int(matched.get("gpu_index"))
                if gpu_index not in gpu_index_types:
                    gpu_index_types[gpu_index] = set()
                gpu_index_types[gpu_index].add(gpu_device_type)

        gpu_devices: GPUDevicesInfo = []
        if gpu_indexes and self._worker.status.gpu_devices:
            if gpu_index_types:
                for i, d in enumerate(self._worker.status.gpu_devices):
                    if d.index not in gpu_index_types:
                        continue
                    if d.type not in gpu_index_types[d.index]:
                        continue
                    # For example, with d = {"index": 0, "type": "cuda"},
                    # before discard: gpu_index_types = {0: {cuda,rocm}, 1: {cuda}}
                    # after discard:  gpu_index_types = {0: {     rocm}, 1: {cuda}}
                    gpu_index_types[d.index].discard(d.type)
                    gpu_devices.append(self._worker.status.gpu_devices[i])
            else:
                for index in gpu_indexes:
                    gpu_device = next(
                        (
                            d
                            for d in self._worker.status.gpu_devices
                            if d.index == index
                        ),
                        None,
                    )
                    if gpu_device:
                        gpu_devices.append(gpu_device)
        return gpu_devices

    def _get_configured_resources(
        self, mount_all_devices: bool = False
    ) -> ContainerResources:
        """
        Get the resource requests for the model instance.

        Args:
            mount_all_devices:
                Whether to mount all available GPU devices.
                If true, ignores the GPUs assigned to the model instance and try to mount all available GPUs.

        Returns:
            A ContainerResources object representing the resource requests for the model instance.

        Raises:
            If the GPUs assigned to the model instance are of different types.
        """
        resources = ContainerResources()
        gpu_devices = self._get_selected_gpu_devices()
        if gpu_devices:
            gpu_type = gpu_devices[0].type
            for device in gpu_devices[1:]:
                if device.type != gpu_type:
                    raise RuntimeError(
                        "All GPUs assigned to the model instance must be of the same type."
                    )
            key = runtime_envs.GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY.get(
                gpu_type
            )
            if key:
                resources[key] = (
                    ",".join(str(d.index) for d in gpu_devices)
                    if not mount_all_devices
                    else "all"
                )
        return resources

    def _get_configured_mounts(self) -> List[ContainerMount]:
        """
        Get the volume mounts for the model instance.
        If runtime mirrored deployment is enabled, no mounts will be set up.

        Returns:
            A list of ContainerMount objects for the model instance.
        """
        mounts: List[ContainerMount] = []
        if (
            self._model_path
            and not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT
        ):
            model_dir = os.path.dirname(self._model_path)
            mounts.append(
                ContainerMount(
                    path=model_dir,
                ),
            )
        return mounts

    def _get_configured_ports(self) -> List[ContainerPort]:
        """
        Get the ports for the model instance.

        Returns:
            A list of ContainerPort objects for the model instance.
        """
        return [
            ContainerPort(
                internal=port,
            )
            for port in self._model_instance.ports or []
        ]

    def _get_serving_port(self) -> int:
        """
        Get the (main) serving port for the model instance.

        Returns:
            The (main) serving port for the model instance.
        """
        return (
            self._model_instance.ports[0]
            if self._model_instance.ports
            else self._model_instance.port
        )

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
        version = self._model.backend_version
        if not version or not self.inference_backend:
            return default_args

        # Load version configuration
        version_config = None
        try:
            version_config, version = self.inference_backend.get_version_config(version)
        except Exception:
            version_config = self.inference_backend.version_configs.root.get(version)

        # Only perform replacement when the version uses non-built-in version and defines run_command
        if (
            version_config
            and version_config.built_in_frameworks is None
            and version_config.run_command
        ):
            resolved_model_path = (
                model_path if model_path is not None else self._model_path
            )
            resolved_port = port if port is not None else self._model_instance.port
            resolved_model_name = self._model_instance.model_name

            command = self.inference_backend.replace_command_param(
                version=version,
                model_path=resolved_model_path,
                port=resolved_port,
                worker_ip=self._worker.ip,
                model_name=resolved_model_name,
                command=version_config.run_command,
            )
            if command:
                return command.split()

        # Return original default_args by default
        return default_args

    def _get_configured_image(
        self,
        backend: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve the container image to use for the current backend, then apply
        registry override once if needed.

        See _resolve_image for resolution details.
        """
        image_name, target_version = self._resolve_image(backend)
        if image_name is None:
            return None
        # Update model backend service version at upper layer if we detected it
        if target_version and not self._model.backend_version:
            self._update_model_backend_service_version(target_version)
        return self._apply_registry_override(image_name)

    def _resolve_image(  # noqa: C901
        self,
        backend: Optional[str] = None,
    ) -> (Optional[str], Optional[str]):
        """
        Resolve the container image to use for the current backend.

        This method returns the raw image name without applying any registry
        override. Callers should apply overrides as needed.

        Precedence:
        1) Explicitly configured image on the model (self._model.image_name)
        2) Prefer image name from the user's config when using custom backend or built-in backend with a custom version
        3) Auto-detected image from gpustack-runner based on device vendor/arch and backend

        Return:
            image_name, backend_version

        """
        # 1) Return directly if explicitly provided.
        if self._model.image_name:
            return self._model.image_name, None

        # 2) Configuration takes priority when backend_version is set
        if self._model and self.inference_backend:
            image_name, target_version = self.inference_backend.get_image_name(
                self._model.backend_version
            )
            if image_name and target_version:
                return image_name, target_version

        """
        Prepare queries for retrieving runners.
        """

        def get_docker_image(bvr: BackendVersionedRunner) -> str:
            return bvr.variants[0].services[0].versions[0].platforms[0].docker_image

        # Get vendor from selected devices at first,
        # if no specified, retrieve from the first device of the worker.
        vendor, runtime_version, arch_family = None, None, None
        gpu_devices = self._get_selected_gpu_devices()
        if gpu_devices:
            gpu_device = gpu_devices[0]
            vendor, runtime_version, arch_family = (
                gpu_device.vendor,
                gpu_device.runtime_version,
                gpu_device.arch_family,
            )
        elif self._worker.status.gpu_devices:
            gpu_device = self._worker.status.gpu_devices[0]
            vendor, runtime_version, arch_family = (
                gpu_device.vendor,
                gpu_device.runtime_version,
                gpu_device.arch_family,
            )
        if not vendor:
            # Return directly if there is not a valid device.
            # GPUStack-Runner does not provide CPU-only platform images.
            # To use a CPU-only version, user must configure in `Inference Backend` page.
            return None

        # Determine backend if not provided.
        if not backend:
            backend = manufacturer_to_backend(ManufacturerEnum(vendor))
        elif vendor != backend_to_manufacturer(backend):
            # Return directly if selected vendor is not matched the backend.
            return None

        """
        Retrieve runners by queries.

        For example, the queries of runners is as below.

        - backend: cuda
          backend_variant: None
          service: vllm
          service_version: 0.10.0
          platform: linux/amd64
        - backend: cann
          backend_variant: 910b
          service: vllm
          service_version: 0.10.0
          platform: linux/arm64
        """

        backend_variant = None
        service = self._model.backend.lower()
        service_version = (
            self._model.backend_version if self._model.backend_version else None
        )

        # Default variant for some backends.
        if backend == "cann":
            if arch_family:
                backend_variant = get_ascend_cann_variant(arch_family)
            if not backend_variant:
                backend_variant = "910b"

        runners = list_backend_runners(
            backend=backend,
            backend_variant=backend_variant,
            service=service,
            service_version=service_version,
            platform=platform.system_arch(),
        )
        if not runners:
            # Return directly if there is not a valid runner.
            return None, None

        """
        Pick the appropriate backend version from among the multiple versions.

        For example, the content of runners is as below.

        [
            {
                "backend": "cuda",
                "versions": [
                    {
                        "version": "12.8",
                        ...
                    },
                    {
                        "version": "12.6",
                        ...
                    },
                    {
                        "version": "12.4",
                        ...
                    }
                ]
            }
        ]
        """

        backend_versioned_runners = runners[0].versions

        try:
            # Try to update backend version for server model.
            service_version = (
                backend_versioned_runners[0].variants[0].services[0].versions[0].version
            )
        except Exception as e:
            logger.error(
                f"Failed to update model service version {service_version}: {e}"
            )

        # Return directly if there is only one versioned backend.
        if len(backend_versioned_runners) == 1:
            return get_docker_image(backend_versioned_runners[0]), service_version

        backend_version = runtime_version

        # Iterate all backend versions, and get the one that less or equal to backend version.
        # Here, we assume the runners' sequence is ordered and arranged in descending order.
        if backend_version:
            for backend_versioned_runner in backend_versioned_runners:
                if (
                    compare_versions(backend_versioned_runner.version, backend_version)
                    <= 0
                ):
                    return get_docker_image(backend_versioned_runner), service_version

        # Return the last(oldest) backend version of selected runner
        # if failed to detect host backend version or no backend version matched.
        #
        # NB(thxCode): Not using the latest backend version is to keep backend version idempotence
        #              when the gpustack-runner adds new backend version.
        return get_docker_image(backend_versioned_runners[-1]), service_version

    def _update_model_backend_service_version(
        self, service_version: Optional[str]
    ) -> None:
        """
        Update model backend (service) version back to server if not already set.

        This method is extracted from image resolution flow to be called from the upper
        layer after the version is detected.
        """
        if not service_version:
            return
        try:
            self._model.backend_version = service_version
            self._clientset.models.update(
                self._model.id, ModelUpdate(**self._model.model_dump())
            )
        except Exception as e:
            logger.error(
                f"Failed to update model service version {service_version}: {e}"
            )

    def _apply_registry_override(self, image: str) -> str:
        """
        Apply registry override rules to a resolved image name.

        - If system_default_container_registry is set and the image does not
          contain an explicit registry host, prefix it with the configured registry.

        This helps enterprise/offline environments redirect image pulls to a mirror
        or local registry.
        """

        registry = (self._config.system_default_container_registry or "").strip()
        if not registry:
            return image

        # Normalize registry value (remove trailing slash if present)
        registry = registry.rstrip("/")

        def has_explicit_registry(img: str) -> bool:
            # Detect if the image has an explicit registry host in its first segment.
            image_split = img.split("/", 1)
            if len(image_split) < 2:
                return False
            first = image_split[0]
            return "." in first or ":" in first or first == "localhost"

        if has_explicit_registry(image):
            logger.debug(f"Docker pull command: docker pull {image}")
            return image
        final = f"{registry}/{image}"
        logger.debug(f"Docker pull command: docker pull {final}")
        return final


def is_ascend_310p(devices: GPUDevicesInfo) -> bool:
    """
    Check if the model instance is running on VLLM Ascend 310P.
    """

    return all(
        gpu.vendor == ManufacturerEnum.ASCEND.value
        and get_ascend_cann_variant(gpu.arch_family) == "310p"
        for gpu in devices
    )


def is_ascend(devices: GPUDevicesInfo) -> bool:
    """
    Check if all devices are Ascend.
    """

    return all(gpu.vendor == ManufacturerEnum.ASCEND.value for gpu in devices)


def cal_distributed_parallelism_arguments(
    model_instance: ModelInstance,
) -> tuple[int, int]:
    pp = len(model_instance.distributed_servers.subordinate_workers) + 1
    tp = len(model_instance.gpu_indexes) if model_instance.gpu_indexes else 1
    uneven_pp = tp
    uneven = False
    for subordinate_worker in model_instance.distributed_servers.subordinate_workers:
        num_gpus = len(subordinate_worker.gpu_indexes)
        uneven_pp += num_gpus
        if num_gpus != tp:
            uneven = True

    if uneven:
        tp = 1
        pp = uneven_pp
        logger.warning(
            f"The number of GPUs selected for each worker is not equal: {num_gpus} != {tp}, fallback to using pipeline parallelism."
        )
    return tp, pp
