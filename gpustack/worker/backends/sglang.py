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
    ContainerRestartPolicyEnum,
)

from gpustack.schemas.models import (
    ModelInstance,
    SpeculativeAlgorithmEnum,
    CategoryEnum,
)
from gpustack.utils.command import find_parameter
from gpustack.utils.envs import sanitize_env
from gpustack.utils.network import get_free_port
from gpustack.worker.backends.base import (
    InferenceServer,
    cal_distributed_parallelism_arguments,
    is_ascend,
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
            if CategoryEnum.IMAGE in self._model.categories:
                self._start_diffusion()
            else:
                self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(f"Starting SGLang model instance: {self._model_instance.name}")

        is_distributed, _, _ = self._get_distributed_metadata()

        # Setup environment variables
        env = self._get_configured_env(is_distributed)

        command_script = self._get_serving_command_script(env)

        # Build SGLang command arguments
        command_args = self._build_command_args(
            port=self._get_serving_port(), is_distributed=is_distributed
        )

        self._create_workload(
            command_script=command_script,
            command_args=command_args,
            env=env,
        )

    def _start_diffusion(self):
        logger.info(
            f"Starting SGLang Diffusion model instance: {self._model_instance.name}"
        )

        # Setup environment variables
        env = self._get_configured_env(False)

        command_script = self._get_serving_command_script(env)

        # Build SGLang command arguments
        command_args = self._build_diffusion_args(port=self._get_serving_port())

        self._create_workload(
            command_script=command_script,
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        command_script: Optional[str],
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
        image = self._get_configured_image()
        if not image:
            raise ValueError("Can't find compatible SGLang image")

        ports = self._get_configured_ports()

        # Create container configuration
        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=True,
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

        workload_plan = WorkloadPlan(
            name=self._workload_name,
            host_network=True,
            shm_size=10 * 1 << 30,  # 10 GiB
            containers=[run_container],
        )

        logger.info(f"Creating SGLang container workload: {self._workload_name}")
        logger.info(
            f"With image: {image}, "
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

        # Apply distributed environment variables
        if is_distributed:
            self._set_distributed_env(env)

        return env

    def _set_distributed_env(self, env: Dict[str, str]):
        """
        Set up environment variables for distributed execution.
        """

        if is_ascend(self._get_selected_gpu_devices()):
            # See https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi-node_dsv3.2.html.
            if "HCCL_SOCKET_IFNAME" not in env:
                env["HCCL_IF_IP"] = self._worker.ip
                env["HCCL_SOCKET_IFNAME"] = f"={self._worker.ifname}"
                env["GLOO_SOCKET_IFNAME"] = self._worker.ifname
                env["TP_SOCKET_IFNAME"] = self._worker.ifname
            return

        if "NCCL_SOCKET_IFNAME" not in env:
            env["NCCL_SOCKET_IFNAME"] = f"={self._worker.ifname}"
            env["GLOO_SOCKET_IFNAME"] = self._worker.ifname

    def _build_command_args(self, port: int, is_distributed: bool) -> List[str]:
        """
        Build SGLang command arguments for container execution.
        """
        arguments = [
            "sglang",
            "serve",
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

        # Add metrics arguments if needed
        metrics_arguments = get_metrics_arguments(
            self._model.backend_parameters, self._model.env
        )
        arguments.extend(metrics_arguments)

        # Add speculative config arguments if needed
        speculative_config_arguments = self._get_speculative_arguments()
        arguments.extend(speculative_config_arguments)

        # Add multi-node deployment parameters if needed
        if is_distributed:
            multinode_arguments = self._get_multinode_arguments()
            arguments.extend(multinode_arguments)

        # Add hierarchical cache arguments if needed
        hicache_arguments = self._get_hicache_arguments()
        arguments.extend(hicache_arguments)

        # Add user-defined backend parameters
        if self._model.backend_parameters:
            arguments.extend(self._model.backend_parameters)

        # Add platform-specific parameters
        if is_ascend(self._get_selected_gpu_devices()):
            # See https://github.com/sgl-project/sglang/pull/7722.
            arguments.extend(
                [
                    "--attention-backend",
                    "ascend",
                ]
            )

        # Set host and port
        arguments.extend(
            [
                "--host",
                self._worker.ip,
                "--port",
                str(port),
            ]
        )

        return arguments

    def _build_diffusion_args(self, port: int):
        arguments = [
            "sglang",
            "serve",
            "--model-path",
            self._model_path,
        ]

        # Allow version-specific command override if configured (before appending extra args)
        arguments = self.build_versioned_command_args(arguments)

        # Add auto parallelism arguments if needed
        auto_parallelism_arguments = get_auto_parallelism_arguments(
            self._model.backend_parameters, self._model_instance, False
        )
        arguments.extend(auto_parallelism_arguments)

        # Set host and port
        arguments.extend(
            [
                "--host",
                self._worker.ip,
                "--port",
                str(port),
            ]
        )

        return arguments

    def _get_hicache_arguments(self) -> List[str]:
        """
        Get hierarchical KV cache arguments for SGLang.
        """
        extended_kv_cache = self._model.extended_kv_cache
        if not (extended_kv_cache and extended_kv_cache.enabled):
            return []

        arguments = ["--enable-hierarchical-cache"]
        if extended_kv_cache.chunk_size and extended_kv_cache.chunk_size > 0:
            arguments.extend(
                [
                    "--page-size",
                    str(extended_kv_cache.chunk_size),
                ]
            )

        if extended_kv_cache.ram_size and extended_kv_cache.ram_size > 0:
            arguments.extend(
                [
                    "--hicache-size",
                    str(extended_kv_cache.ram_size),
                ]
            )

        if extended_kv_cache.ram_ratio and extended_kv_cache.ram_ratio > 0:
            arguments.extend(
                [
                    "--hicache-ratio",
                    str(extended_kv_cache.ram_ratio),
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
            self._model_instance.distributed_servers
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

        dist_port_range = self._config.distributed_worker_port_range
        dist_init_port = get_free_port(port_range=dist_port_range)

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

    def _get_speculative_arguments(self) -> List[str]:
        """
        Get speculative arguments for SGLang.
        """

        speculative_config = self._model.speculative_config
        if not speculative_config or not speculative_config.enabled:
            return []

        sglang_speculative_algorithm_mapping = {
            SpeculativeAlgorithmEnum.EAGLE3: "EAGLE3",
            SpeculativeAlgorithmEnum.MTP: "EAGLE",  # SGLang uses "EAGLE" for MTP
            SpeculativeAlgorithmEnum.NGRAM: "NGRAM",
        }

        arguments = []
        method = sglang_speculative_algorithm_mapping.get(
            speculative_config.algorithm, None
        )
        if method:
            arguments.extend(
                [
                    "--speculative-algorithm",
                    method,
                ]
            )

            if speculative_config.num_draft_tokens:
                arguments.extend(
                    [
                        "--speculative-num-draft-tokens",
                        str(speculative_config.num_draft_tokens),
                    ]
                )

            if speculative_config.ngram_max_match_length:
                arguments.extend(
                    [
                        "--speculative-ngram-max-match-window-size",
                        str(speculative_config.ngram_max_match_length),
                    ]
                )

            if speculative_config.ngram_min_match_length:
                arguments.extend(
                    [
                        "--speculative-ngram-min-match-window-size",
                        str(speculative_config.ngram_min_match_length),
                    ]
                )

            if speculative_config.draft_model and self._draft_model_path:
                arguments.extend(
                    [
                        "--speculative-draft-model",
                        self._draft_model_path,
                    ]
                )

        return arguments


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


def get_metrics_arguments(
    backend_parameters: List[str], env: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Get metrics flag for SGLang.
    """

    metrics_flag = find_parameter(
        backend_parameters,
        ["enable-metrics"],
    )

    if metrics_flag is not None:
        return []

    if env and env.get("GPUSTACK_DISABLE_METRICS"):
        return []

    return ["--enable-metrics"]
