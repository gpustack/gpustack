import logging
import os
import sys
from typing import Dict, List, Optional, Iterator

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.worker.backends.base import InferenceServer

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
    list_workloads,
    logs_workload,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._workload_name: Optional[str] = None

    def start(self, **kwargs):
        try:
            mounts = []
            if hasattr(self, "_model_path") and self._model_path:
                model_dir = os.path.dirname(self._model_path)
                mounts.append(ContainerMount(path=model_dir))

            envs = self._setup_environment()

            # Get resources configuration
            resources = self._get_configured_resources()

            image_cmd = []
            command = self.inference_backend.replace_command_param(
                self._model.backend_version,
                self._model_path,
                self._model_instance.port,
                self._model.run_command,
            )
            if command:
                image_cmd.extend(command.split(" "))
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
            )

            # Store workload name for management operations
            self._workload_name = self._model_instance.name

            workload_plan = WorkloadPlan(
                name=self._workload_name,
                host_network=True,
                containers=[run_container],
            )

            logger.info(f"Creating workload: {self._workload_name}")
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
        error_message = f"Failed to run the {command_name} error: {error}{cause_text}"
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
            logger.info(f"Getting logs for container: {self._workload_name}")
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
                logger.info(f"Container {self._workload_name} status: {status.state}")
            else:
                logger.warning(f"Container {self._workload_name} not found")
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
            logger.info(f"Stopping container: {self._workload_name}")
            result = delete_workload(self._workload_name)

            if result:
                logger.info(f"Container {self._workload_name} stopped successfully")
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
                    f"Container {self._workload_name} was not found or already stopped"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    def list_all_workloads(
        self, labels: Optional[Dict[str, str]] = None
    ) -> List[WorkloadStatus]:
        """
        List all workloads, optionally filtered by labels.

        Args:
            labels: Optional dictionary of labels to filter workloads

        Returns:
            List of WorkloadStatus objects
        """
        try:
            workloads = list_workloads(labels=labels)
            logger.info(f"Found {len(workloads)} workloads")

            for workload in workloads:
                logger.info(f"  - {workload.name}: {workload.state}")

            return workloads
        except Exception as e:
            logger.error(f"Failed to list workloads: {e}")
            return []

    def restart_container(self) -> bool:
        """
        Restart the container by stopping and starting it again.

        Returns:
            True if container was restarted successfully, False otherwise.
        """
        logger.info(f"Restarting container: {self._workload_name}")

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
            logger.info(f"Container {self._workload_name} restarted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            return False

    def get_container_resource_usage(self) -> Optional[Dict]:
        """
        Get container resource usage information.

        Returns:
            Dictionary containing resource usage information, or None if not available.
        """
        status = self.get_container_status()
        if not status:
            return None

        # Extract resource information from status
        # Note: The actual fields depend on the WorkloadStatus structure
        resource_info = {
            "name": status.name,
            "state": status.state,
            "created_at": getattr(status, 'created_at', None),
            "started_at": getattr(status, 'started_at', None),
        }

        # Add container-specific information if available
        if hasattr(status, 'containers') and status.containers:
            container = status.containers[0]  # Assuming single container
            resource_info.update(
                {
                    "container_id": getattr(container, 'id', None),
                    "image": getattr(container, 'image', None),
                    "status": getattr(container, 'status', None),
                }
            )

        return resource_info
