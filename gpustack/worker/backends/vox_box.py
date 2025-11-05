import logging
import os
from typing import Optional, List, Dict

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
    _workload_name: Optional[str] = None

    def start(self):
        try:
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(f"Starting Vox-Box model instance: {self._model_instance.name}")

        env = self._get_configured_env()

        command_args = self._build_command_args(
            port=self._get_serving_port(),
        )

        self._create_workload(
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        command_args: List[str],
        env: Dict[str, str],
    ):
        # Store workload name for management operations
        self._workload_name = self._model_instance.name

        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get VoxBox backend image")

        resources = self._get_configured_resources(
            # Pass-through all devices as vox-box handles device itself.
            mount_all_devices=True,
        )

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
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
            resources=resources,
            mounts=mounts,
            ports=ports,
        )

        logger.info(f"Creating Vox-Box container workload: {self._workload_name}")
        logger.info(
            f"With image: {image}, "
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

        logger.info(f"Created Vox-Box container workload {self._workload_name}")

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
        if self._model.backend_parameters:
            arguments.extend(self._model.backend_parameters)
        # Append immutable arguments to ensure proper operation for accessing
        immutable_arguments = [
            "--host",
            "0.0.0.0",
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
