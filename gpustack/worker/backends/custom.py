import logging
import os
import shlex
from typing import Dict, List, Optional, Tuple

from gpustack.schemas.models import ModelInstanceDeploymentMetadata, find_role
from gpustack.utils.command import format_backend_parameters
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.base import InferenceServer
from gpustack.worker.pd_router import is_managed_router

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

# A router needs descriptors for both incoming and backend connections.
_MANAGED_ROUTER_COMMAND_SCRIPT = r"""#!/bin/sh
# Linux runner shells support soft/hard nofile flags.
# shellcheck disable=SC3045
set -e

target=65536
soft=$(ulimit -Sn)
hard=$(ulimit -Hn)
if [ "$hard" != unlimited ] && [ "$hard" -lt "$target" ]; then
    echo "Warning: PD router nofile hard limit $hard is below $target;" \
        "using the hard limit." >&2
    target=$hard
fi
if [ "$soft" != unlimited ] && [ "$soft" -lt "$target" ]; then
    if ! ulimit -Sn "$target"; then
        echo "Failed to raise PD router nofile soft limit from $soft to $target" \
            "(hard=$hard)." >&2
        exit 1
    fi
fi
echo "PD router nofile limits: soft=$(ulimit -Sn), hard=$(ulimit -Hn)" >&2
exec "$@"
"""


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

        command = None
        if self.inference_backend:
            command = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )

        command_args, injected = self._build_command_args(entrypoint=command)

        try:
            self._update_model_instance(
                self._model_instance.id,
                injected_backend_parameters=format_backend_parameters(injected) or None,
            )
        except Exception as e:
            logger.warning(
                "Failed to persist injected backend parameters for "
                f"{self._model_instance.name}: {e}"
            )

        self._create_workload(
            deployment_metadata=deployment_metadata,
            command=command,
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
        command: Optional[List[str]],
        command_args: List[str],
        env: Dict[str, str],
    ):
        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get Custom backend image")

        resources = self._get_configured_resources()

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        # Read container config from environment variables
        container_config = self._get_container_env_config(env)

        role_name = getattr(self._model_instance, "role", None)
        source_model = self._model_spec or self._model
        role = find_role(source_model, role_name)
        # Read image overrides before rendering fills in the default runner image.
        # Deployment and role images may not contain a shell.
        if (
            is_managed_router(self._model, role_name)
            and not source_model.image_name
            and not (role and role.image_name)
        ):
            if not command and not command_args:
                raise ValueError("Managed PD router requires an explicit command")
            router_command = [
                "/bin/sh",
                "-c",
                _MANAGED_ROUTER_COMMAND_SCRIPT,
                "gpustack-pd-router",
            ]
            if command:
                command_args = command + command_args
                command = router_command
            else:
                # Keep the image's init entrypoint in front of the router.
                command_args = router_command + command_args

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=True,
                command=command,
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
            mounts=mounts,
            resources=resources,
            ports=ports,
        )

        logger.info(
            f"Creating custom backend container workload: {deployment_metadata.name}"
        )
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
            host_ipc=self._host_ipc_enabled(),
            shm_size=int(container_config.shm_size_gib * (1 << 30)),
            containers=[run_container],
            run_as_user=container_config.user,
            run_as_group=container_config.group,
        )
        create_workload(self._transform_workload_plan(workload_plan))

        logger.info(
            f"Created custom backend container workload: {deployment_metadata.name}"
        )

    def _build_command_args(
        self, entrypoint: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        command_args = []

        selected_gpu_indexes = sorted(d.index for d in self._get_selected_gpu_devices())

        command_args_inline = self.inference_backend.replace_command_param(
            version=self._model.backend_version,
            model_path=self._model_path,
            port=self._get_serving_port(),
            worker_ip=self._worker.ip,
            model_name=self._model.name,
            gpu_count=len(selected_gpu_indexes),
            gpu_ids=selected_gpu_indexes,
            command=self._model.run_command,
            env=self._model.env,
        )
        if command_args_inline:
            command_args = shlex.split(command_args_inline)

        # Add user-defined backend parameters
        user_backend_parameters = self._flatten_backend_param()
        command_args.extend(user_backend_parameters)

        injected = self._get_injected_backend_parameters(
            command_args, user_backend_parameters, entrypoint
        )

        return command_args, injected
