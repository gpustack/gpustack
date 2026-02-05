import logging
import os
from typing import Dict, List, Optional, Tuple

from gpustack_runtime.detector import ManufacturerEnum
from packaging.version import Version
from packaging.specifiers import SpecifierSet

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    WorkloadPlan,
    create_workload,
    ContainerRestartPolicyEnum,
)
from gpustack_runtime.deployer.__utils__ import compare_versions

from gpustack.scheduler.model_registry import is_multimodal_model
from gpustack.schemas.models import (
    ModelInstance,
    SpeculativeAlgorithmEnum,
    CategoryEnum,
    ModelInstanceDeploymentMetadata,
)
from gpustack.utils.command import find_parameter
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.base import (
    InferenceServer,
    cal_distributed_parallelism_arguments,
    is_ascend,
    is_ascend_310p,
)

logger = logging.getLogger(__name__)


class SGLangServer(InferenceServer):
    """
    Containerized SGLang inference server backend using gpustack-runtime.

    This backend runs SGLang in a Docker container managed by gpustack-runtime,
    providing better isolation, resource management, and deployment consistency.
    """

    is_diffusion = False

    def start(self):  # noqa: C901
        try:
            if CategoryEnum.IMAGE in self._model.categories:
                self.is_diffusion = True
                self._start_diffusion()
            else:
                self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(f"Starting SGLang model instance: {self._model_instance.name}")

        deployment_metadata = self._get_deployment_metadata()

        # Setup environment variables
        env = self._get_configured_env(
            is_distributed=deployment_metadata.distributed,
        )

        command = None
        if self.inference_backend:
            command = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )

        command_script = self._get_serving_command_script(env)

        # Build SGLang command arguments
        command_args = self._build_command_args(
            port=self._get_serving_port(),
            is_distributed=deployment_metadata.distributed,
            is_distributed_leader=deployment_metadata.distributed_leader,
        )

        self._create_workload(
            deployment_metadata=deployment_metadata,
            command=command,
            command_script=command_script,
            command_args=command_args,
            env=env,
        )

    def _start_diffusion(self):
        logger.info(
            f"Starting SGLang Diffusion model instance: {self._model_instance.name}"
        )

        deployment_metadata = self._get_deployment_metadata()

        # Setup environment variables
        env = self._get_configured_env(
            is_distributed=False,
        )

        command = None
        if self.inference_backend:
            command = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )

        command_script = self._get_serving_command_script(env)

        command_args = self._build_command_args_for_diffusion(
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
            raise ValueError("Can't find compatible SGLang image")

        if (
            self.is_diffusion
            and compare_versions(self._model.backend_version, "0.5.5") < 0
        ):
            raise ValueError(
                "SGLang versions <= 0.5.5 do not support Diffusion models."
            )

        # Command script will override the given command,
        # so we need to prepend command to command args.
        if command_script and command:
            command_args = command + command_args
            command = None

        resources = self._get_configured_resources()

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

        logger.info(f"Creating SGLang container workload: {deployment_metadata.name}")
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

        logger.info(f"Created SGLang container workload: {deployment_metadata.name}")

    def _get_configured_env(self, is_distributed: bool) -> Dict[str, str]:
        """
        Get environment variables for SGLang service.
        """

        # Apply GPUStack's inference environment setup
        env = super()._get_configured_env()

        # Optimize environment variables
        # -- Disable OpenMP parallelism to avoid resource contention, increases model loading.
        env["OMP_NUM_THREADS"] = env.pop("OMP_NUM_THREADS", "1")
        # -- Enable safetensors GPU loading pass-through for faster model loading.
        env["SAFETENSORS_FAST_GPU"] = env.pop("SAFETENSORS_FAST_GPU", "1")
        # -- Observe RUN:AI streamer model loading.
        env["RUNAI_STREAMER_MEMORY_LIMIT"] = env.pop("RUNAI_STREAMER_MEMORY_LIMIT", "0")
        env["RUNAI_STREAMER_LOG_TO_STDERR"] = env.pop(
            "RUNAI_STREAMER_LOG_TO_STDERR", "1"
        )
        env["RUNAI_STREAMER_LOG_LEVEL"] = env.pop("RUNAI_STREAMER_LOG_LEVEL", "INFO")

        # Apply distributed environment variables
        if is_distributed:
            self._set_distributed_env(env)

        # Apply Ascend-specific environment variables
        if is_ascend(self._get_selected_gpu_devices()):
            self._set_ascend_env(env)

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

    def _set_ascend_env(self, env: Dict[str, str]):
        """
        Set up environment variables for Ascend devices.
        """

        # -- Optimize Pytorch NPU operations delivery performance.
        env["TASK_QUEUE_ENABLE"] = env.pop("TASK_QUEUE_ENABLE", "1")
        # -- Enable NUMA coarse-grained binding.
        env["CPU_AFFINITY_CONF"] = env.pop("CPU_AFFINITY_CONF", "1")
        # -- Reuse memory in multi-streams.
        env["PYTORCH_NPU_ALLOC_CONF"] = env.pop(
            "PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"
        )
        # -- Increase HCCL connection timeout to avoid issues in large clusters.
        env["HCCL_CONNECT_TIMEOUT"] = env.pop("HCCL_CONNECT_TIMEOUT", "7200")
        # -- Enable RDMA PCIe direct post with no strict mode for better performance.
        env["HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT"] = env.pop(
            "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT", "TRUE"
        )
        if not is_ascend_310p(self._get_selected_gpu_devices()):
            # -- Disable HCCL execution timeout for better stability.
            env["HCCL_EXEC_TIMEOUT"] = env.pop("HCCL_EXEC_TIMEOUT", "0")
            # -- Enable the communication is scheduled by AI Vector directly with ROCE, instead of AI CPU.
            env["HCCL_OP_EXPANSION_MODE"] = env.pop("HCCL_OP_EXPANSION_MODE", "AIV")

    def _build_command_args(
        self,
        port: int,
        is_distributed: bool,
        is_distributed_leader: bool,
    ) -> List[str]:
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
        specified_max_model_len = find_parameter(
            self._model.backend_parameters,
            ["context-length"],
        )
        if (
            specified_max_model_len is None
            and derived_max_model_len
            and derived_max_model_len > 8192
        ):
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

        # Add multimodal argument if needed
        if is_multimodal_model(self._get_model_architecture()):
            arguments.append("--enable-multimodal")

        # Add speculative config arguments if needed
        speculative_config_arguments = self._get_speculative_arguments()
        arguments.extend(speculative_config_arguments)

        # Add multi-node deployment parameters if needed
        if is_distributed:
            multinode_arguments = self._get_multinode_arguments(
                is_distributed_leader=is_distributed_leader
            )
            arguments.extend(multinode_arguments)

        # Add hierarchical cache arguments if needed
        hicache_arguments = self._get_hicache_arguments()
        arguments.extend(hicache_arguments)

        if (
            self._model_instance.computed_resource_claim
            and self._model_instance.computed_resource_claim.vram_utilization
        ):
            input_utilization = find_parameter(
                self._model.backend_parameters, ["mem-fraction-static"]
            )
            if not input_utilization:
                arguments.extend(
                    [
                        "--mem-fraction-static",
                        str(
                            self._model_instance.computed_resource_claim.vram_utilization
                        ),
                    ]
                )

        # Add user-defined backend parameters
        arguments.extend(self._flatten_backend_param())

        # Add platform-specific parameters
        if is_ascend(self._get_selected_gpu_devices()):
            # See https://github.com/sgl-project/sglang/pull/7722.
            arguments.extend(
                [
                    "--attention-backend",
                    "ascend",
                ]
            )
            if is_multimodal_model(self._get_model_architecture()):
                arguments.extend(
                    [
                        "--mm-attention-backend",
                        "ascend_attn",
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

    def _build_command_args_for_diffusion(self, port: int):
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

        attention_arguments = self._get_attention_backend_for_diffusion()
        arguments.extend(attention_arguments)

        # Add user-defined backend parameters
        arguments.extend(self._flatten_backend_param())

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

    def _get_attention_backend_for_diffusion(self) -> List[str]:
        if (
            find_parameter(self._model.backend_parameters, ["attention-backend"])
            is not None
        ):
            return []

        devices = self._get_selected_gpu_devices()
        if devices and all(
            (d.vendor or "").lower() == ManufacturerEnum.NVIDIA for d in devices
        ):
            spec = SpecifierSet(">=8.0,<9.0")
            in_range = True
            for d in devices:
                cap = d.compute_capability
                try:
                    v = Version(cap) if cap is not None else None
                except Exception:
                    v = None
                if v is None or v not in spec:
                    in_range = False
                    break
            if not in_range:
                return ["--attention-backend", "torch_sdpa"]

        return []

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

    def _get_multinode_arguments(self, is_distributed_leader: bool) -> List[str]:
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

        # Determine node rank based on worker IP
        if not is_distributed_leader:
            for idx, worker in enumerate(subordinate_workers):
                if worker.worker_ip == current_worker_ip:
                    node_rank = idx + 1  # Subordinate workers start from rank 1
                    break

        # Add multi-node parameters
        arguments.extend(
            [
                "--nnodes",
                str(total_nodes),
                "--node-rank",
                str(node_rank),
                "--dist-init-addr",
                # During distributed setup,
                # we must get more than one port here,
                # so we use ports[1] for distributed initialization.
                f"{self._model_instance.worker_ip}:{self._model_instance.ports[1]}",
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

            num_steps = find_parameter(
                self._model.backend_parameters, ["speculative-num-steps"]
            )
            topk = find_parameter(
                self._model.backend_parameters, ["speculative-eagle-topk"]
            )
            if num_steps is None and topk is None:
                default_steps, default_topk = self._get_default_speculative_steps_topk()
                arguments.extend(
                    [
                        "--speculative-num-steps",
                        str(default_steps),
                        "--speculative-eagle-topk",
                        str(default_topk),
                    ]
                )

        return arguments

    def _get_default_speculative_steps_topk(self) -> Tuple[int, int]:
        """
        Get the default speculative steps and topk for SGLang.
        Ref: https://github.com/sgl-project/sglang/blob/67fca6b297bf0202941bde7b608c6da14f6a8776/python/sglang/srt/server_args.py#L4363
        """
        architectures = getattr(self._pretrained_config, "architectures", []) or []
        arch = architectures[0] if architectures else ""
        if arch in [
            "DeepseekV32ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV2ForCausalLM",
            "GptOssForCausalLM",
            "BailingMoeForCausalLM",
            "BailingMoeV2ForCausalLM",
        ]:
            return (3, 1)
        else:
            # The default value for all other models
            return (5, 4)


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
        [
            "tensor-parallel-size",
            "tp-size",
            "pipeline-parallel-size",
            "pp-size",
            "data-parallel-size",
            "dp-size",
        ],
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
