import logging
import os
import sys
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.worker.backends.base import InferenceServer

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    ContainerMount,
    WorkloadPlan,
    create_workload,
    ContainerPort,
)

logger = logging.getLogger(__name__)


class VoxBoxServer(InferenceServer):
    def start(self):
        try:
            # Get vox-box image name via runtime
            image_name = self._get_backend_image_name()
            if not image_name:
                raise ValueError("Can't find compatible vox-box image")

            # Get resources configuration
            resources = self._get_configured_resources(
                # Pass-through all devices as vox-box handles device itself.
                mount_all_devices=True,
            )

            # Setup container mounts
            mounts = []
            if hasattr(self, "_model_path") and self._model_path:
                model_dir = os.path.dirname(self._model_path)
                mounts.append(ContainerMount(path=model_dir))

            # Get configured environment variables
            envs = self._get_configured_env()

            # Get serving port
            serving_port = self._get_serving_port()

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
                port=serving_port,
            )

            if self._model.backend_parameters:
                arguments.extend(self._model.backend_parameters)

            # Append immutable arguments to ensure proper operation for accessing
            immutable_arguments = [
                "--host",
                "0.0.0.0",
                "--port",
                str(serving_port),
            ]
            if self._model_instance.gpu_indexes is not None:
                immutable_arguments.extend(
                    [
                        "--device",
                        f"cuda:{self._model_instance.gpu_indexes[0]}",
                    ]
                )
            arguments.extend(immutable_arguments)

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
                    ContainerEnv(
                        name=name,
                        value=value,
                    )
                    for name, value in envs.items()
                ],
                resources=resources,
                mounts=mounts,
                ports=[
                    ContainerPort(
                        internal=serving_port,
                    ),
                ],
            )

            # Store workload name for management operations
            self._workload_name = self._model_instance.name

            workload_plan = WorkloadPlan(
                name=self._workload_name,
                host_network=True,
                containers=[run_container],
            )

            logger.info(f"Creating vox-box container workload: {self._workload_name}")
            logger.debug(f"Run vox-box with arguments: {' '.join(arguments)}")
            if self._model.env:
                logger.debug(
                    f"Model environment variables: {', '.join(f'{key}={value}' for key, value in self._model.env.items())}"
                )
            logger.info(f"Container image name: {image_name} arguments: {arguments}")
            create_workload(workload_plan)

            logger.info(
                f"vox-box container workload {self._workload_name} created successfully"
            )
        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception):
        """
        Handle errors during vox-box container server startup.
        """
        cause = getattr(error, "__cause__", None)
        cause_text = f": {cause}" if cause else ""
        error_message = (
            f"Failed to run the vox-box container server: {error}{cause_text}"
        )
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
