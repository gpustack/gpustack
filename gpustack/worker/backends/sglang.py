import logging
import os
from typing import Dict, List, Optional

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    WorkloadPlan,
    create_workload,
)

from gpustack.schemas.models import ModelInstance
from gpustack.utils.command import find_parameter
from gpustack.utils.envs import sanitize_env
from gpustack.utils.network import get_free_port
from gpustack.worker.backends.base import (
    InferenceServer,
    cal_distributed_parallelism_arguments,
)

logger = logging.getLogger(__name__)


class SGLangServer(InferenceServer):
    """
    Containerized SGLang inference server backend using gpustack-runtime.

    This backend runs SGLang in a Docker container managed by gpustack-runtime,
    providing better isolation, resource management, and deployment consistency.
    """

    _workload_name: Optional[str] = None

    def start(self):  # noqa: C901
        try:
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(f"Starting SGLang model instance: {self._model_instance.name}")

        is_distributed, _, _ = self.is_distributed()

        # Setup environment variables
        envs = self._get_configured_env(is_distributed)

        # Build SGLang command arguments
        arguments = self._build_sglang_arguments(
            port=self._get_serving_port(),
            is_distributed=is_distributed,
        )

        self._create_workload(
            command_args=arguments,
            env=envs,
        )

    def _create_workload(
        self,
        command_args: List[str],
        env: Dict[str, str],
    ):
        # Store workload name for management operations
        self._workload_name = self._model_instance.name

        # Get resources configuration
        resources = self._get_configured_resources()

        # Setup container mounts
        mounts = self._get_configured_mounts()

        # Get SGLang image name
        image_name = self._get_configured_image()
        if not image_name:
            raise ValueError("Can't find compatible SGLang image")

        ports = self._get_configured_ports()

        # Create container configuration
        run_container = Container(
            image=image_name,
            name=self._model_instance.name,
            profile=ContainerProfileEnum.RUN,
            execution=ContainerExecution(
                args=command_args,
            ),
            envs=[ContainerEnv(name=name, value=value) for name, value in env.items()],
            resources=resources,
            mounts=mounts,
            ports=ports,
        )

        # Store workload name for management operations
        self._workload_name = self._model_instance.name

        workload_plan = WorkloadPlan(
            name=self._workload_name,
            host_network=True,
            shm_size=10 * 1 << 30,  # 10 GiB
            containers=[run_container],
        )

        logger.info(f"Creating SGLang container workload: {self._workload_name}")
        logger.info(
            f"With image: {image_name}, "
            f"arguments: [{' '.join(command_args)}], "
            f"ports: [{','.join([str(port.internal) for port in ports])}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )
        create_workload(workload_plan)

        logger.info(
            f"SGLang container workload {self._workload_name} created successfully"
        )

    def _get_configured_env(self, is_distributed: bool) -> Dict[str, str]:
        """
        Setup environment variables for the SGLang container server.
        """

        # Apply GPUStack's inference environment setup
        env = super()._get_configured_env()

        # Apply SGLang distributed environment setup
        if is_distributed:
            self.set_sglang_distributed_env(env)

        return env

    def _build_sglang_arguments(self, port: int, is_distributed: bool) -> List[str]:
        """
        Build SGLang command arguments for container execution.
        """
        arguments = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self._model_path,
        ]

        # Allow version-specific command override if configured (before appending extra args)
        arguments = self.build_versioned_command_args(arguments)

        derived_max_model_len = self._derive_max_model_len()
        if derived_max_model_len and derived_max_model_len > 8192:
            arguments.extend(["--context-length", "8192"])

        # Add auto parallelism arguments if needed
        auto_parallelism_arguments = get_auto_parallelism_arguments(
            self._model.backend_parameters, self._model_instance, is_distributed
        )
        arguments.extend(auto_parallelism_arguments)

        # Add multi-node deployment parameters if needed
        if is_distributed:
            multinode_arguments = self._get_multinode_arguments()
            arguments.extend(multinode_arguments)

        # Add user-defined backend parameters
        if self._model.backend_parameters:
            arguments.extend(self._model.backend_parameters)

        # Set host and port
        arguments.extend(
            [
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ]
        )

        return arguments

    def _get_multinode_arguments(self) -> List[str]:
        """
        Get multi-node deployment arguments for SGLang.
        """
        arguments = []

        # Check if this is a multi-node deployment
        if not (
            hasattr(self._model_instance, 'distributed_servers')
            and self._model_instance.distributed_servers
            and self._model_instance.distributed_servers.subordinate_workers
        ):
            return []

        subordinate_workers = (
            self._model_instance.distributed_servers.subordinate_workers
        )
        total_nodes = len(subordinate_workers) + 1  # +1 for the current node

        # Find the current node's rank
        current_worker_ip = self._worker.ip
        node_rank = 0  # Default to 0 (master node)
        is_main_worker = current_worker_ip == self._model_instance.worker_ip

        # Determine node rank based on worker IP
        if not is_main_worker:
            for idx, worker in enumerate(subordinate_workers):
                if worker.worker_ip == current_worker_ip:
                    node_rank = idx + 1  # Subordinate workers start from rank 1
                    break

        # Get or allocate distributed communication port
        dist_init_port = get_free_port(port_range=self._config.ray_worker_port_range)

        # Add multi-node parameters
        arguments.extend(
            [
                "--nnodes",
                str(total_nodes),
                "--node-rank",
                str(node_rank),
                "--dist-init-addr",
                f"{self._model_instance.worker_ip}:{dist_init_port}",
            ]
        )

        return arguments

    def set_sglang_distributed_env(self, env: Dict[str, str]):
        """
        Set up distributed environment variables for SGLang.
        """
        # Set up distributed training environment
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_SOCKET_IFNAME"] = "^lo,docker0"

        # Set master address and port for distributed training
        if (
            hasattr(self._model_instance, 'distributed_servers')
            and self._model_instance.distributed_servers
        ):
            subordinate_workers = (
                self._model_instance.distributed_servers.subordinate_workers
            )
            if subordinate_workers:
                master_worker = subordinate_workers[0]
                env["MASTER_ADDR"] = master_worker.worker_ip or "localhost"
                env["MASTER_PORT"] = str(
                    get_free_port(port_range=self._config.ray_worker_port_range)
                )


def get_auto_parallelism_arguments(
    backend_parameters: List[str],
    model_instance: ModelInstance,
    is_distributed: bool,
) -> List[str]:
    """
    Get auto parallelism arguments for SGLang based on GPU configuration.
    """
    arguments = []

    parallelism = find_parameter(
        backend_parameters,
        ["tp-size", "tp", "pp-size", "pp"],
    )

    if parallelism is not None:
        return []

    if is_distributed:
        # distributed across multiple workers
        (tp, pp) = cal_distributed_parallelism_arguments(model_instance)
        return [
            "--tp-size",
            str(tp),
            "--pp-size",
            str(pp),
        ]

    # Check if tensor parallelism is already specified
    if model_instance.gpu_indexes and len(model_instance.gpu_indexes) > 1:
        gpu_count = len(model_instance.gpu_indexes)
        if gpu_count > 1:
            arguments.extend(["--tp-size", str(gpu_count)])

    return arguments
