import logging
import os
import shlex
from typing import Dict, List

from gpustack.schemas.models import ModelInstanceDeploymentMetadata
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.base import InferenceServer

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    WorkloadPlan,
    create_workload,
    ContainerRestartPolicyEnum,
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

    def start(self):
        try:
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(
            f"Starting custom backend model instance: {self._model_instance.name}"
        )

        deployment_metadata = self._get_deployment_metadata()

        env = self._get_configured_env()

        command_args = []
        command = self.inference_backend.replace_command_param(
            version=self._model.backend_version,
            model_path=self._model_path,
            port=self._get_serving_port(),
            worker_ip=self._worker.ip,
            model_name=self._model.name,
            command=self._model.run_command,
        )
        if command:
            command_args.extend(shlex.split(command))

        command_args.extend(self._flatten_backend_param())

        self._create_workload(
            deployment_metadata=deployment_metadata,
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
        command_args: List[str],
        env: Dict[str, str],
    ):
        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get Custom backend image")

        resources = self._get_configured_resources()

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        # Get container entrypoint from inference backend configuration
        container_entrypoint = None
        if self.inference_backend:
            container_entrypoint = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=True,
                command=container_entrypoint,
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

        logger.info(
            f"Creating custom backend container workload: {deployment_metadata.name}"
        )
        logger.info(
            f"With image: {image}, "
            f"{('entrypoint: ' + str(container_entrypoint) + ', ') if container_entrypoint else ''}"
            f"arguments: [{' '.join(command_args)}], "
            f"ports: [{','.join([str(port.internal) for port in ports])}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=deployment_metadata.name,
            host_network=True,
            shm_size=10 * 1 << 30,  # 10 GiB
            containers=[run_container],
        )
        create_workload(self._transform_workload_plan(workload_plan))

        logger.info(
            f"Created custom backend container workload: {deployment_metadata.name}"
        )
