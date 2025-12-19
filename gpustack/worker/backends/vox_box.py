import logging
import os
from typing import Optional, List, Dict

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


class VoxBoxServer(InferenceServer):
    def start(self):
        try:
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(f"Starting VoxBox model instance: {self._model_instance.name}")

        deployment_metadata = self._get_deployment_metadata()

        env = self._get_configured_env()

        command_script = self._get_serving_command_script(env)

        command_args = self._build_command_args(
            port=self._get_serving_port(),
        )

        self._create_workload(
            deployment_metadata=deployment_metadata,
            command_script=command_script,
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
        command_script: Optional[str],
        command_args: List[str],
        env: Dict[str, str],
    ):
        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get VoxBox backend image")

        resources = self._get_configured_resources(
            # Pass-through all devices as vox-box handles device itself.
            mount_all_devices=True,
        )

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
                command_script=command_script,
                args=command_args,
            ),
            envs=[
                ContainerEnv(
                    name=name,
                    value=value,
                )
                for name, value in env.items()
            ],
            resources=resources,
            mounts=mounts,
            ports=ports,
        )

        logger.info(f"Creating VoxBox container workload: {deployment_metadata.name}")
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

        logger.info(f"Created VoxBox container workload: {deployment_metadata.name}")

    def _build_command_args(self, port: int) -> List[str]:
        arguments = [
            "vox-box",
            "start",
            "--model",
            self._model_path,
            "--data-dir",
            self._config.data_dir,
        ]
        # Allow version-specific command override if configured (before appending extra args)
        arguments = self.build_versioned_command_args(
            arguments,
            model_path=self._model_path,
            port=port,
        )
        arguments.extend(self._flatten_backend_param())
        # Append immutable arguments to ensure proper operation for accessing
        immutable_arguments = [
            "--host",
            self._worker.ip,
            "--port",
            str(port),
        ]
        if self._model_instance.gpu_indexes is not None:
            immutable_arguments.extend(
                [
                    "--device",
                    f"cuda:{self._model_instance.gpu_indexes[0]}",
                ]
            )
        arguments.extend(immutable_arguments)

        return arguments
