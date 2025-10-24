import logging
import os
from typing import Dict, Optional, List

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.base import InferenceServer

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    WorkloadPlan,
    create_workload,
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
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(f"Starting Custom model instance: {self._model_instance.name}")

        env = self._get_configured_env()

        command_args = []
        command = self.inference_backend.replace_command_param(
            version=self._model.backend_version,
            model_path=self._model_path,
            port=self._get_serving_port(),
            model_name=self._model.name,
            command=self._model.run_command,
        )
        if command:
            command_args.extend(command.split())
        if self._model.backend_parameters:
            command_args.extend(self._model.backend_parameters)

        self._create_workload(
            command_args,
            env,
        )

    def _create_workload(
        self,
        command_args: List[str],
        env: Dict[str, str],
    ):
        # Store workload name for management operations
        self._workload_name = self._model_instance.name

        image_name = self._model.image_name or self.inference_backend.get_image_name(
            self._model.backend_version
        )
        if not image_name:
            raise ValueError("Failed to get Custom backend image name")

        resources = self._get_configured_resources()

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        run_container = Container(
            image=image_name,
            name=self._model_instance.name,
            profile=ContainerProfileEnum.RUN,
            execution=ContainerExecution(
                privileged=True,
                args=command_args,
            ),
            envs=[
                ContainerEnv(
                    name=name,
                    value=value,
                )
                for name, value in env.items()
            ],
            mounts=mounts,
            resources=resources,
            ports=ports,
        )

        logger.info(f"Creating container workload: {self._workload_name}")
        logger.info(
            f"With image: {image_name}, "
            f"arguments: [{' '.join(command_args)}], "
            f"ports: [{','.join([str(port.internal) for port in ports])}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=self._workload_name,
            host_network=True,
            shm_size=10 * 1 << 30,  # 10 GiB
            containers=[run_container],
        )
        create_workload(workload_plan)

        logger.info(f"Created container workload {self._workload_name}")

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
