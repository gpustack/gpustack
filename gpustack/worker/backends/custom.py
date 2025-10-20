import logging
from typing import Dict, Optional

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.worker.backends.base import InferenceServer

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    WorkloadPlan,
    create_workload,
    ContainerPort,
)

logger = logging.getLogger(__name__)


class CustomServer(InferenceServer):
    """
    Generic pluggable inference server backend with container management capabilities.

    This backend allows users to specify any command and automatically handles:
    - Command path resolution
    - Version management
    - Environment variable setup
    - Model path and port configuration
    - Backend parameters passing
    - Error handling and logging
    - Container management operations (logs, stop, status, etc.)

    Usage:
        Set model.backend_command to specify the command name (e.g., "vllm", "custom-server")
        The backend will automatically call get_command_path(command_name) to resolve the path.
    """

    _workload_name: Optional[str] = None

    def start(self):
        try:
            mounts = self._get_configured_mounts()

            envs = self._setup_environment()

            # Get resources configuration
            resources = self._get_configured_resources()

            # Get serving port
            serving_port = self._get_serving_port()

            image_cmd = []
            command = self.inference_backend.replace_command_param(
                self._model.backend_version,
                self._model_path,
                serving_port,
                self._model_instance.model_name,
                self._model.run_command,
            )
            if command:
                image_cmd.extend(command.split())
            if self._model.backend_parameters:
                image_cmd.extend(self._model.backend_parameters)

            image_name = (
                self._model.image_name
                or self.inference_backend.get_image_name(self._model.backend_version)
            )

            run_container = Container(
                image=image_name,
                name=self._model_instance.name,
                profile=ContainerProfileEnum.RUN,
                execution=ContainerExecution(
                    privileged=True,
                    args=image_cmd,
                ),
                envs=[
                    ContainerEnv(
                        name=name,
                        value=value,
                    )
                    for name, value in envs.items()
                ],
                mounts=mounts,
                resources=resources,
                ports=[
                    ContainerPort(
                        internal=serving_port,
                    )
                ],
            )

            # Store workload name for management operations
            self._workload_name = self._model_instance.name

            workload_plan = WorkloadPlan(
                name=self._workload_name,
                host_network=True,
                containers=[run_container],
            )

            logger.info(f"Creating workload: {self._workload_name}")
            logger.info(f"Container image name: {image_name} arguments: {image_cmd}")
            create_workload(workload_plan)

            logger.info(f"Workload {self._workload_name} created successfully")

        except Exception as e:
            self._handle_error(e)

    def _setup_environment(self) -> Dict[str, str]:
        """
        Setup environment variables for the inference server.
        """

        # Apply GPUStack's inference environment setup
        env = self._get_configured_env()

        return env

    def _handle_error(self, error: Exception):
        """
        Handle errors during server startup.
        """
        command_name = getattr(self._model, 'run_command', 'unknown')
        cause = getattr(error, "__cause__", None)
        cause_text = f": {cause}" if cause else ""
        error_message = f"Failed to run {command_name}: {error}{cause_text}"

        try:
            patch_dict = {
                "state_message": error_message,
                "state": ModelInstanceStateEnum.ERROR,
            }
            self._update_model_instance(self._model_instance.id, **patch_dict)
        except Exception as ue:
            logger.error(f"Failed to update model instance: {ue}")

        raise error
