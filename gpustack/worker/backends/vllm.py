import json
import logging
import os
import sys
from typing import Dict, List, Optional, Iterator

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    ContainerMount,
    WorkloadPlan,
    WorkloadStatus,
    create_workload,
    delete_workload,
    get_workload,
    logs_workload,
)

from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.workers import VendorEnum
from gpustack.utils.command import find_parameter
from gpustack.utils.envs import sanitize_env
from gpustack.utils.gpu import all_gpu_match
from gpustack.utils.hub import (
    get_hf_text_config,
    get_max_model_len,
    get_pretrained_config,
)
from gpustack.worker.backends.base import InferenceServer, is_ascend_310p

logger = logging.getLogger(__name__)


class VLLMServer(InferenceServer):
    """
    Containerized vLLM inference server backend using gpustack-runtime.

    This backend runs vLLM in a Docker container managed by gpustack-runtime,
    providing better isolation, resource management, and deployment consistency.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._workload_name: Optional[str] = None

    def start(self):  # noqa: C901
        try:
            # Setup container mounts
            mounts = []
            if hasattr(self, "_model_path") and self._model_path:
                model_dir = os.path.dirname(self._model_path)
                mounts.append(ContainerMount(path=model_dir))

            # Setup environment variables
            envs = self._setup_environment()

            # Build vLLM command arguments
            arguments = self._build_vllm_arguments()

            # Get vLLM image name
            image_name = self._get_backend_image_name()

            if not image_name:
                raise ValueError("Can't find compatible vLLM image")

            # Create container configuration
            run_container = Container(
                image=image_name,
                name=self._model_instance.name,
                profile=ContainerProfileEnum.RUN,
                execution=ContainerExecution(
                    privileged=True,
                    args=arguments,
                ),
                envs=[
                    ContainerEnv(name=name, value=value) for name, value in envs.items()
                ],
                mounts=mounts,
            )

            # Store workload name for management operations
            self._workload_name = self._model_instance.name

            workload_plan = WorkloadPlan(
                name=self._workload_name,
                host_network=True,
                containers=[run_container],
            )

            logger.info(f"Creating vLLM container workload: {self._workload_name}")
            logger.info(f"Container image name: {image_name} arguments: {arguments}")
            create_workload(workload_plan)

            logger.info(
                f"vLLM container workload {self._workload_name} created successfully"
            )

        except Exception as e:
            self._handle_error(e)

    def _setup_environment(self) -> Dict[str, str]:
        """
        Setup environment variables for the vLLM container server.
        """
        env = os.environ.copy()

        # Apply vLLM distributed environment setup
        self.set_vllm_distributed_env(env)

        # Apply GPUStack's inference environment setup
        env = self.get_inference_running_env(env)

        # Add model-specific environment variables
        if self._model.env:
            env.update(self._model.env)

        # Log environment variables
        env_view = None
        if logger.isEnabledFor(logging.DEBUG):
            env_view = sanitize_env(env)
        elif self._model.env:
            # If the model instance has its own environment variables,
            # display the mutated environment variables.
            env_view = self._model.env
            for k, v in self._model.env.items():
                env_view[k] = env.get(k, v)
        if env_view:
            logger.info(
                f"With environment variables(inconsistent input items mean unchangeable):{os.linesep}"
                f"{os.linesep.join(f'{k}={v}' for k, v in sorted(env_view.items()))}"
            )

        return env

    def _build_vllm_arguments(self) -> List[str]:
        """
        Build vLLM command arguments for container execution.
        """
        arguments = [
            "vllm",
            "serve",
            self._model_path,
        ]

        arguments = self.build_versioned_command_args(arguments)

        derived_max_model_len = self._derive_max_model_len()
        if derived_max_model_len and derived_max_model_len > 8192:
            arguments.extend(["--max-model-len", "8192"])

        auto_parallelism_arguments = get_auto_parallelism_arguments(
            self._model.backend_parameters, self._model_instance
        )
        arguments.extend(auto_parallelism_arguments)

        if is_distributed_vllm(self._model_instance):
            arguments.extend(["--distributed-executor-backend", "ray"])

        # For Ascend 310P, we need to enforce eager execution and default dtype to float16
        if is_ascend_310p(self._worker):
            arguments.extend(
                [
                    "--enforce-eager",
                    "--dtype",
                    "float16",
                ]
            )

        if self._model.backend_parameters:
            arguments.extend(self._model.backend_parameters)

        built_in_arguments = [
            "--host",
            "0.0.0.0",
        ]

        has_port = any(arg == "--port" for arg in arguments)
        if not has_port:
            built_in_arguments.extend(["--port", str(self._model_instance.port)])

        built_in_arguments.extend(
            ["--served-model-name", self._model_instance.model_name]
        )

        # Extend the built-in arguments at the end so that
        # they cannot be overridden by the user-defined arguments
        arguments.extend(built_in_arguments)

        return arguments

    def _handle_error(self, error: Exception):
        """
        Handle errors during container server startup.
        """
        error_message = f"Failed to run the vLLM container server: {error}"
        logger.exception(error_message)

        try:
            patch_dict = {
                "state_message": error_message,
                "state": ModelInstanceStateEnum.ERROR,
            }
            self._update_model_instance(self._model_instance.id, **patch_dict)
        except Exception as ue:
            logger.error(f"Failed to update model instance: {ue}")

        sys.exit(1)

    def set_vllm_distributed_env(self, env: Dict[str, str]):
        """
        Set up vLLM distributed environment variables.
        This method is reused from the original VLLMServer implementation.
        """
        model_instance = self._model_instance
        worker = self._worker
        subordinate_workers = model_instance.distributed_servers.subordinate_workers
        if not subordinate_workers:
            return

        device_str = "GPU"
        if all_gpu_match(worker, lambda gpu: gpu.vendor == VendorEnum.Huawei.value):
            device_str = "NPU"

        ray_placement_group_bundles: List[Dict[str, float]] = []
        bundle_indexes = []
        bundle_index_offset = 0
        # we add all gpus of involved workers to the bundle, then pick gpus by index
        for i in range(len(worker.status.gpu_devices)):
            bundle = {
                device_str: 1,
                f"node:{worker.ip}": 0.001,
            }
            ray_placement_group_bundles.append(bundle)
        for gpu_index in model_instance.gpu_indexes:
            bundle_indexes.append(gpu_index)
        bundle_index_offset += len(worker.status.gpu_devices)

        for subordinate_worker in subordinate_workers:
            for i in range(subordinate_worker.total_gpus):
                bundle = {
                    device_str: 1,
                    f"node:{subordinate_worker.worker_ip}": 0.001,
                }
                ray_placement_group_bundles.append(bundle)
            for gpu_index in subordinate_worker.gpu_indexes:
                bundle_indexes.append(bundle_index_offset + gpu_index)
            bundle_index_offset += subordinate_worker.total_gpus

        # encoded to json and set in GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES env
        env["GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES"] = json.dumps(
            ray_placement_group_bundles
        )
        # helps to pick specific gpus
        env["VLLM_RAY_BUNDLE_INDICES"] = ", ".join([str(x) for x in bundle_indexes])

        logger.debug(
            f"Set GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES: {env['GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES']}. "
            f"Set VLLM_RAY_BUNDLE_INDICES: {env['VLLM_RAY_BUNDLE_INDICES']}"
        )

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

    def get_container_logs(
        self,
        tail: Optional[int] = 100,
        follow: bool = False,
        timestamps: bool = True,
        since: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Get container logs.

        Args:
            tail: Number of lines to show from the end of the logs
            follow: Whether to follow log output (stream new logs)
            timestamps: Whether to include timestamps in log output
            since: Show logs since timestamp (Unix timestamp)

        Returns:
            Iterator of log lines
        """
        if not self._workload_name:
            logger.warning(
                "No workload name available. Container may not be started yet."
            )
            return iter([])

        try:
            logger.info(f"Getting logs for vLLM container: {self._workload_name}")
            return logs_workload(
                name=self._workload_name,
                tail=tail,
                follow=follow,
                timestamps=timestamps,
                since=since,
            )
        except Exception as e:
            logger.error(f"Failed to get container logs: {e}")
            return iter([])

    def get_container_status(self) -> Optional[WorkloadStatus]:
        """
        Get the current status of the container.

        Returns:
            WorkloadStatus object containing container information, or None if not found.
        """
        if not self._workload_name:
            logger.warning(
                "No workload name available. Container may not be started yet."
            )
            return None

        try:
            status = get_workload(self._workload_name)
            if status:
                logger.info(
                    f"vLLM container {self._workload_name} status: {status.state}"
                )
            else:
                logger.warning(f"vLLM container {self._workload_name} not found")
            return status
        except Exception as e:
            logger.error(f"Failed to get container status: {e}")
            return None

    def stop_container(self) -> bool:
        """
        Stop the container.

        Returns:
            True if container was stopped successfully, False otherwise.
        """
        if not self._workload_name:
            logger.warning(
                "No workload name available. Container may not be started yet."
            )
            return False

        try:
            logger.info(f"Stopping vLLM container: {self._workload_name}")
            result = delete_workload(self._workload_name)

            if result:
                logger.info(
                    f"vLLM container {self._workload_name} stopped successfully"
                )
                # Update model instance state
                try:
                    patch_dict = {
                        "state_message": "Container stopped by user request",
                        "state": ModelInstanceStateEnum.ERROR,
                    }
                    self._update_model_instance(self._model_instance.id, **patch_dict)
                except Exception as ue:
                    logger.error(f"Failed to update model instance state: {ue}")
                return True
            else:
                logger.warning(
                    f"vLLM container {self._workload_name} was not found or already stopped"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    def restart_container(self) -> bool:
        """
        Restart the container by stopping and starting it again.

        Returns:
            True if container was restarted successfully, False otherwise.
        """
        logger.info(f"Restarting vLLM container: {self._workload_name}")

        # Stop the container first
        if not self.stop_container():
            logger.error("Failed to stop container for restart")
            return False

        # Wait a moment for cleanup
        import time

        time.sleep(2)

        # Start the container again
        try:
            self.start()
            logger.info(f"vLLM container {self._workload_name} restarted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            return False


def get_auto_parallelism_arguments(
    backend_parameters: List[str], model_instance: ModelInstance
) -> List[str]:
    parallelism = find_parameter(
        backend_parameters,
        ["tensor-parallel-size", "tp", "pipeline-parallel-size", "pp"],
    )

    if parallelism is not None:
        return []

    if is_distributed_vllm(model_instance):
        # distributed across multiple workers
        pp = len(model_instance.distributed_servers.subordinate_workers) + 1
        tp = len(model_instance.gpu_indexes) if model_instance.gpu_indexes else 1
        uneven_pp = tp
        uneven = False
        for (
            subordinate_worker
        ) in model_instance.distributed_servers.subordinate_workers:
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
        return [
            "--tensor-parallel-size",
            str(tp),
            "--pipeline-parallel-size",
            str(pp),
        ]

    if model_instance.gpu_indexes is not None and len(model_instance.gpu_indexes) > 1:
        # single worker with multiple GPUs
        return [
            "--tensor-parallel-size",
            str(len(model_instance.gpu_indexes)),
        ]

    return []


def is_distributed_vllm(model_instance: ModelInstance) -> bool:
    return (
        model_instance.distributed_servers
        and model_instance.distributed_servers.subordinate_workers
    )
