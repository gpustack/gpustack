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
)

logger = logging.getLogger(__name__)


class VoxBoxServer(InferenceServer):
    def start(self):
        try:
            # Setup container mounts
            mounts = []
            if hasattr(self, "_model_path") and self._model_path:
                model_dir = os.path.dirname(self._model_path)
                mounts.append(ContainerMount(path=model_dir))

            # Build vox-box command arguments
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
                port=self._model_instance.port,
            )

            # Build built-in args, avoiding duplicates if version-specific command already sets them
            built_in_arguments = [
                "--host",
                "0.0.0.0",
            ]

            has_port = any(arg == "--port" for arg in arguments)
            if not has_port:
                built_in_arguments.extend(["--port", str(self._model_instance.port)])

            if self._model.backend_parameters:
                arguments.extend(self._model.backend_parameters)

            if self._model_instance.gpu_indexes is not None:
                arguments.extend(
                    [
                        "--device",
                        f"cuda:{self._model_instance.gpu_indexes[0]}",
                    ]
                )

            # Extend built-in arguments at the end
            arguments.extend(built_in_arguments)

            # Setup environment variables
            env = os.environ.copy()
            env.update(self._model.env or {})

            # Get vox-box image name via runtime
            image_name = self._get_backend_image_name()
            if not image_name:
                raise ValueError("Can't find compatible vox-box image")

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
                    ContainerEnv(name=name, value=value) for name, value in env.items()
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
