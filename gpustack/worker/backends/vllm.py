import json
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
    ContainerMount,
    ContainerPort,
    ContainerRestartPolicyEnum,
)
from gpustack_runtime.detector import ManufacturerEnum

from gpustack.schemas.models import (
    ModelInstance,
    SpeculativeAlgorithmEnum,
    SpeculativeConfig,
    ModelInstanceDeploymentMetadata,
    is_omni_model,
)
from gpustack.utils import network
from gpustack.utils.command import (
    find_parameter,
    find_bool_parameter,
    find_int_parameter,
    extend_args_no_exist,
)
from gpustack.utils.envs import sanitize_env
from gpustack.utils.unit import byte_to_gib
from gpustack.worker.backends.base import (
    InferenceServer,
    is_ascend_310p,
    is_ascend,
    cal_distributed_parallelism_arguments,
)

logger = logging.getLogger(__name__)


class VLLMServer(InferenceServer):
    """
    Containerized vLLM inference server backend using gpustack-runtime.

    This backend runs vLLM in a Docker container managed by gpustack-runtime,
    providing better isolation, resource management, and deployment consistency.
    """

    def start(self):  # noqa: C901
        try:
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(f"Starting vLLM model instance: {self._model_instance.name}")

        # Prepare distributed information.
        deployment_metadata = self._get_deployment_metadata()

        env = self._get_configured_env(
            is_distributed=deployment_metadata.distributed,
        )

        command = None
        if self.inference_backend:
            command = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )

        command_script = self._get_serving_command_script(env)

        command_args = self._build_command_args(
            port=self._get_serving_port(),
            is_distributed=deployment_metadata.distributed,
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
            raise ValueError("Failed to get vLLM backend image")

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

        # Adjust run container for distributed follower.
        if deployment_metadata.distributed_follower:
            ray_command, ray_command_args, ray_ports = self._build_ray_configuration(
                is_leader=False,
            )

            # Command script will override the given command,
            # so we need to prepend command to command args.
            if command_script:
                ray_command_args = ray_command + ray_command_args
                ray_command = None

            run_container.execution.command = ray_command
            # run_container.execution.command_script = command_script # already set
            run_container.execution.args = ray_command_args
            run_container.ports = ray_ports

        # Create sidecar container for distributed leader.
        sidecar_container = None
        if deployment_metadata.distributed_leader:
            run_container.mounts.append(
                ContainerMount(
                    path="/tmp",
                    volume="tmp-volume",
                ),
            )

            ray_command, ray_command_args, ray_ports = self._build_ray_configuration(
                is_leader=True,
            )

            # Command script will override the given command,
            # so we need to prepend command to command args.
            if command_script:
                ray_command_args = ray_command + ray_command_args
                ray_command = None

            sidecar_container = Container(
                image=image,
                name="ray-head",
                profile=ContainerProfileEnum.RUN,
                restart_policy=ContainerRestartPolicyEnum.NEVER,
                execution=ContainerExecution(
                    privileged=True,
                    command=ray_command,
                    command_script=command_script,
                    args=ray_command_args,
                    run_as_user=container_config.user,
                    run_as_group=container_config.group,
                ),
                envs=run_container.envs,
                resources=run_container.resources,
                mounts=run_container.mounts,
                ports=ray_ports,
            )

        logger.info(f"Creating vLLM container workload: {deployment_metadata.name}")
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
            containers=(
                [run_container]
                if not sidecar_container
                else [run_container, sidecar_container]
            ),
            run_as_user=container_config.user,
            run_as_group=container_config.group,
        )
        create_workload(self._transform_workload_plan(workload_plan))

        logger.info(f"Created vLLM container workload: {deployment_metadata.name}")

    def _get_configured_env(self, is_distributed: bool) -> Dict[str, str]:
        """
        Get environment variables for vLLM service
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

        # Apply LMCache environment variables if extended KV cache is enabled
        self._set_lmcache_env(env)

        # Apply distributed environment variables
        if is_distributed:
            self._set_distributed_env(env)

        # Apply Ascend-specific environment variables
        if is_ascend(self._get_selected_gpu_devices()):
            self._set_ascend_env(env)

        return env

    def _set_lmcache_env(self, env: Dict[str, str]):
        """
        Set up LMCache environment variables if extended KV cache is enabled.
        """
        extended_kv_cache = self._model.extended_kv_cache
        if not (extended_kv_cache and extended_kv_cache.enabled):
            return

        if extended_kv_cache.chunk_size and extended_kv_cache.chunk_size > 0:
            env["LMCACHE_CHUNK_SIZE"] = str(extended_kv_cache.chunk_size)

        if extended_kv_cache.ram_size and extended_kv_cache.ram_size > 0:
            # Explicitly specified RAM size for KV cache
            env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(extended_kv_cache.ram_size)
        elif extended_kv_cache.ram_ratio and extended_kv_cache.ram_ratio > 0:
            # Calculate RAM size based on ratio of total VRAM claim
            vram_claim = self._get_total_vram_claim()
            ram_size = int(vram_claim * extended_kv_cache.ram_ratio)
            env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(byte_to_gib(ram_size))

    def _set_distributed_env(self, env: Dict[str, str]):
        """
        Set up environment variables for distributed execution.
        """
        # Configure Internal communication IP and port.
        # see https://docs.vllm.ai/en/stable/configuration/env_vars.html.
        env["VLLM_HOST_IP"] = self._worker.ip
        # During distributed setup,
        # we must get more than one port here,
        # so we use ports[-1] for distributed initialization.
        env["VLLM_PORT"] = str(self._model_instance.ports[-1])

        # Disable Ray logging to stderr by default,
        # see https://github.com/gpustack/gpustack/issues/4158#issuecomment-3809213348.
        env["RAY_LOG_TO_STDERR"] = env.pop("RAY_LOG_TO_STDERR", "0")
        # To reduce verbosity, set Ray backend log level to warning by default.
        env["RAY_BACKEND_LOG_LEVEL"] = env.pop("RAY_BACKEND_LOG_LEVEL", "warning")

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

    def _get_speculative_arguments(self) -> List[str]:
        """
        Get speculative arguments for vLLM.
        """

        speculative_config: SpeculativeConfig = self._model.speculative_config
        if not speculative_config or not speculative_config.enabled:
            return []

        vllm_speculative_algorithm_mapping = {
            SpeculativeAlgorithmEnum.EAGLE3: "eagle3",
            SpeculativeAlgorithmEnum.MTP: "deepseek_mtp",
            SpeculativeAlgorithmEnum.NGRAM: "ngram",
        }

        method = vllm_speculative_algorithm_mapping.get(
            speculative_config.algorithm, None
        )
        if method:
            sp_dict = {
                "method": method,
            }
            if speculative_config.num_draft_tokens:
                sp_dict["num_speculative_tokens"] = speculative_config.num_draft_tokens
            if speculative_config.ngram_max_match_length:
                sp_dict["prompt_lookup_max"] = speculative_config.ngram_max_match_length
            if speculative_config.ngram_min_match_length:
                sp_dict["prompt_lookup_min"] = speculative_config.ngram_min_match_length
            if speculative_config.draft_model and self._draft_model_path:
                sp_dict["model"] = self._draft_model_path
            return [
                "--speculative-config",
                json.dumps(sp_dict),
            ]
        return []

    def _get_total_vram_claim(self) -> int:
        """
        Calculate total VRAM claim for the model instance on current worker.
        """
        vram = 0
        computed_resource_claim = self._model_instance.computed_resource_claim
        if self._worker.id != self._model_instance.worker_id:
            dservers = self._model_instance.distributed_servers
            subworkers = (
                dservers.subordinate_workers
                if dservers and dservers.subordinate_workers
                else []
            )
            for subworker in subworkers:
                if subworker.worker_id == self._worker.id:
                    computed_resource_claim = subworker.computed_resource_claim
                    break

        if not computed_resource_claim:
            return vram

        for _, vram_claim in computed_resource_claim.vram.items():
            vram += vram_claim

        return vram

    def _build_command_args(
        self,
        port: int,
        is_distributed: bool,
    ) -> List[str]:
        """
        Build vLLM command arguments for container execution.
        """
        arguments = [
            "vllm",
            "serve",
            self._model_path,
        ]

        # Allow version-specific command override if configured (before appending extra args)
        arguments = self.build_versioned_command_args(arguments)

        # Omni modalities
        omni_enabled = find_bool_parameter(
            self._model.backend_parameters,
            ["omni"],
        )
        is_omni = is_omni_model(self._model)
        if is_omni and not omni_enabled:
            arguments.extend(
                [
                    "--omni",
                ]
            )

        if not is_omni:
            derived_max_model_len = self._derive_max_model_len()
            specified_max_model_len = find_parameter(
                self._model.backend_parameters,
                ["max-model-len"],
            )
            if (
                specified_max_model_len is None
                and derived_max_model_len
                and derived_max_model_len > 8192
            ):
                arguments.extend(["--max-model-len", "8192"])

        auto_parallelism_arguments = get_auto_parallelism_arguments(
            self._model.backend_parameters,
            self._model_instance,
            is_distributed,
        )
        arguments.extend(auto_parallelism_arguments)

        # Add speculative config arguments if needed
        speculative_config_arguments = self._get_speculative_arguments()
        arguments.extend(speculative_config_arguments)

        if is_distributed:
            arguments.extend(["--distributed-executor-backend", "ray"])
            dps = find_int_parameter(
                self._model.backend_parameters, ["data-parallel-size", "dp"]
            )
            if dps and dps > 1:
                # Prefer to use Ray backend for data parallelism if DP size is specified.
                dpb = find_parameter(
                    self._model.backend_parameters, ["data-parallel-backend", "dpb"]
                )
                if dpb is None:
                    arguments.extend(["--data-parallel-backend", "ray"])
                # Specify a port for DP RPC communication,
                # we must get more than one port here, see gpustack/worker/serve_manager.py,
                # so we use ports[1] for DP RPC communication.
                arguments.extend(
                    ["--data-parallel-rpc-port", str(self._model_instance.ports[1])]
                )

        if self._model.extended_kv_cache and self._model.extended_kv_cache.enabled:
            vendor, _, _ = self._get_device_info()
            if vendor in [ManufacturerEnum.NVIDIA, ManufacturerEnum.AMD]:
                arguments.extend(
                    [
                        "--kv-transfer-config",
                        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
                    ]
                )
            else:
                logger.warning(
                    "Extended KV cache for vLLM is only supported on NVIDIA and AMD GPUs. Skipping LMCache configuration."
                )

        # For Ascend 310P, we need to enforce eager execution and default dtype to float16
        if is_ascend_310p(self._get_selected_gpu_devices()):
            arguments.extend(
                [
                    "--enforce-eager",
                    "--dtype",
                    "float16",
                ]
            )

        # Inject user-defined backend parameters
        arguments.extend(self._flatten_backend_param())

        # Append immutable arguments to ensure proper operation for accessing
        # Only add if not already present in arguments
        extend_args_no_exist(
            arguments,
            ("--host", self._worker.ip),
            ("--port", str(port)),
            ("--served-model-name", self._model_instance.model_name),
        )

        return arguments

    def _build_ray_configuration(
        self,
        is_leader: bool,
    ) -> (List[str], List[str], Optional[List[ContainerPort]]):
        # Parse the Ray port range from configuration,
        # assign ports in order as below:
        # 1.  GCS server port (the first port of the range)
        # 2.  Client port (reserved for compatibility, not used anymore, see https://github.com/gpustack/gpustack/issues/4171)
        # 3.  Dashboard port
        # 4.  Dashboard gRPC port (no longer used, since Ray 2.45.0 kept for backward compatibility)
        # 5.  Dashboard agent gRPC port
        # 6.  Dashboard agent listen port
        # 7.  Metrics export port
        # 8.  Node Manager port
        # 9.  Object Manager port
        # 10. Raylet runtime env agent port
        # 11. Minimum port number for the worker
        # 12. Maximum port number for the worker (the last port of the range)

        start, end = network.parse_port_range(self._config.ray_port_range)
        gcs_server_port = start
        # client_port = start + 1
        dashboard_port = start + 2
        dashboard_grpc_port = start + 3
        dashboard_agent_grpc_port = start + 4
        dashboard_agent_listen_port = start + 5
        metrics_export_port = start + 6
        node_manager_port = start + 7
        object_manager_port = start + 8
        raylet_runtime_env_agent_port = start + 9
        worker_port_min = start + 10
        worker_port_max = end

        command = [
            "ray",
            "start",
        ]
        arguments = [
            "--block",
            "--disable-usage-stats",
            "--verbose",
            f"--node-manager-port={node_manager_port}",
            f"--object-manager-port={object_manager_port}",
            f"--runtime-env-agent-port={raylet_runtime_env_agent_port}",
            f"--dashboard-agent-grpc-port={dashboard_agent_grpc_port}",
            f"--dashboard-agent-listen-port={dashboard_agent_listen_port}",
            f"--metrics-export-port={metrics_export_port}",
            f"--min-worker-port={worker_port_min}",
            f"--max-worker-port={worker_port_max}",
            f"--node-ip-address={self._worker.ip}",
        ]
        ports = [
            ContainerPort(
                internal=port,
            )
            for port in [
                dashboard_grpc_port,
                dashboard_agent_grpc_port,
                dashboard_agent_listen_port,
                metrics_export_port,
                node_manager_port,
                object_manager_port,
                raylet_runtime_env_agent_port,
            ]
        ]

        if is_leader:
            arguments.extend(
                [
                    "--head",
                    f"--port={gcs_server_port}",
                    f"--dashboard-host={self._worker.ip}",
                    f"--dashboard-port={dashboard_port}",
                ]
            )
            ports.extend(
                [
                    ContainerPort(
                        internal=port,
                    )
                    for port in [gcs_server_port, dashboard_port]
                ]
            )
        else:
            arguments.extend(
                [
                    f"--address={self._model_instance.worker_ip}:{gcs_server_port}",
                ]
            )

        return command, arguments, ports


def get_auto_parallelism_arguments(
    backend_parameters: List[str],
    model_instance: ModelInstance,
    is_distributed: bool,
) -> List[str]:
    parallelism = find_parameter(
        backend_parameters,
        [
            "tensor-parallel-size",
            "tp",
            "pipeline-parallel-size",
            "pp",
            "data-parallel-size",
            "dp",
        ],
    )

    if parallelism is not None:
        return []

    if is_distributed:
        # distributed across multiple workers
        (tp, pp) = cal_distributed_parallelism_arguments(model_instance)
        return [
            "--tensor-parallel-size",
            str(tp),
            "--pipeline-parallel-size",
            str(pp),
        ]

    if model_instance.gpu_indexes is not None and len(model_instance.gpu_indexes) > 1:
        # single worker with multiple GPUs
        return [
            "--tensor-parallel-size",
            str(len(model_instance.gpu_indexes)),
        ]

    return []
