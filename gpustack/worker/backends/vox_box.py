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

        command = None
        if self.inference_backend:
            command = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )

        command_script = self._get_serving_command_script(env)

        command_args = self._build_command_args(
            port=self._get_serving_port(),
        )

        self._create_workload(
            deployment_metadata=deployment_metadata,
            command=command,
            command_script=command_script,
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
        command: Optional[List[str]],
        command_script: Optional[str],
        command_args: List[str],
        env: Dict[str, str],
    ):
        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get VoxBox backend image")

        # Command script will override the given command,
        # so we need to prepend command to command args.
        if command_script and command:
            command_args = command + command_args
            command = None

        resources = self._get_configured_resources(
            # Pass-through all devices as vox-box handles device itself.
            mount_all_devices=True,
        )

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        # Read container config from environment variables
        container_config = self._get_container_env_config(env)

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=True,
                command=command,
                command_script=command_script,
                args=command_args,
                run_as_user=container_config.user,
                run_as_group=container_config.group,
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
            f"command: [{' '.join(command) if command else ''}], "
            f"arguments: [{' '.join(command_args)}], "
            f"ports: [{','.join([str(port.internal) for port in ports])}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=deployment_metadata.name,
            host_network=True,
            shm_size=int(container_config.shm_size_gib * (1 << 30)),
            containers=[run_container],
            run_as_user=container_config.user,
            run_as_group=container_config.group,
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
