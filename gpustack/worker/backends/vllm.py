import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
from gpustack_runtime.deployer.__utils__ import compare_versions
from gpustack_runtime.detector import ManufacturerEnum, manufacturer_to_backend
from gpustack.schemas.models import (
    ModelInstance,
    SpeculativeAlgorithmEnum,
    SpeculativeConfig,
    ModelInstanceDeploymentMetadata,
    is_audio_model,
    is_omni_model,
)
from gpustack.utils import network
from gpustack.utils.command import (
    ExecutorBackend,
    find_parameter,
    find_bool_parameter,
    find_int_parameter,
    extend_args_no_exist,
    format_backend_parameters,
    resolve_executor_backend,
)
from gpustack.utils.envs import sanitize_env
from gpustack.utils.unit import byte_to_gib
from gpustack.utils.vllm_topology import (
    MultinodeShape,
    MultinodeUserParallelism,
    parse_user_parallelism,
    validate_multinode_topology,
    resolve_data_parallel_load_balance_mode,
    subordinates_serve_api,
)
from gpustack.worker.backends.base import (
    InferenceServer,
    is_ascend_310p,
    is_ascend,
    cal_distributed_parallelism_arguments,
    read_lora_max_rank,
)

logger = logging.getLogger(__name__)

# vLLM only accepts a fixed set of values for --max-lora-rank; a rank in between
# must be rounded up to the next allowed value.
_VLLM_LORA_RANK_CHOICES = (8, 16, 32, 64, 128, 256, 320, 512)


def _round_up_vllm_lora_rank(rank: int) -> int:
    for choice in _VLLM_LORA_RANK_CHOICES:
        if rank <= choice:
            return choice
    return rank  # Beyond the known set: pass through and let vLLM validate it.


def extend_vllm_mounted_lora_arguments(
    arguments: List[str],
    mounted_loras: Optional[list],
    base_model_name: str,
    backend_parameters: Optional[List[str]],
) -> None:
    """
    Inject vLLM LoRA flags from model instance mounted_loras.

    Skips when the user already set --lora-modules in backend_parameters.

    vLLM expects adapters via --lora-modules {json} ... where each JSON object
    contains name, path, and base_model_name.
    """
    if not mounted_loras:
        return
    if find_parameter(backend_parameters or [], ["lora-modules", "lora_modules"]):
        return

    modules = []
    for m in mounted_loras:
        if not m.lora_name or not m.path:
            continue
        modules.append(
            {
                # m.lora_name is the fully-qualified "<base>:<lora>" id; passed
                # through so the engine registers under the same name clients send.
                "name": m.lora_name,
                "path": m.path,
                "base_model_name": base_model_name,
            }
        )
    if not modules:
        return

    extend_args_no_exist(arguments, "--enable-lora")
    extend_args_no_exist(arguments, ("--max-loras", str(len(modules))))
    if not any(
        a == "--lora-modules" or a.startswith("--lora-modules=") for a in arguments
    ):
        arguments.append("--lora-modules")
        for m in modules:
            arguments.append(json.dumps(m))

    if not find_parameter(backend_parameters or [], ["max-lora-rank", "max_lora_rank"]):
        max_rank = read_lora_max_rank([m["path"] for m in modules])
        if max_rank:
            extend_args_no_exist(
                arguments,
                ("--max-lora-rank", str(_round_up_vllm_lora_rank(max_rank))),
            )


@dataclass
class _VLLMArgsContext:
    """
    Derived inputs shared by every `_build_*_arguments` step.

    Computed once per `_build_command_args` invocation so individual builders
    stay side-effect-free and don't re-evaluate the same predicates.
    """

    port: int
    is_distributed: bool
    executor_backend: ExecutorBackend
    topology: Optional["MultinodeTopology"]
    is_omni: bool
    is_audio: bool
    entrypoint: Optional[List[str]]
    deployment_metadata: Optional[ModelInstanceDeploymentMetadata]


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
            deployment_metadata=deployment_metadata,
        )

        # Resolve image first so that backend_version is populated before
        # building command args (version-gated arguments depend on it).
        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get vLLM backend image")

        command = ["vllm", "serve"]
        if self.inference_backend:
            entrypoint = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )
            if entrypoint:
                command = entrypoint

        command_script = self._get_serving_command_script(env)

        command_args, injected = self._build_command_args(
            port=self._get_serving_port(),
            is_distributed=deployment_metadata.distributed,
            entrypoint=command,
            deployment_metadata=deployment_metadata,
        )

        try:
            self._update_model_instance(
                self._model_instance.id,
                injected_backend_parameters=format_backend_parameters(injected) or None,
            )
        except Exception as e:
            logger.warning(
                f"Failed to persist injected backend parameters for {self._model_instance.name}: {e}"
            )

        self._create_workload(
            deployment_metadata=deployment_metadata,
            command=command,
            command_script=command_script,
            command_args=command_args,
            env=env,
            image=image,
        )

    def _create_workload(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
        command: Optional[List[str]],
        command_script: Optional[str],
        command_args: List[str],
        env: Dict[str, str],
        image: str,
    ):
        command, command_args = self._override_entrypoint(
            command, command_args, command_script
        )

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

        executor_backend = resolve_executor_backend(
            self._model.backend_parameters, self._model.backend_version
        )

        # Adjust run container for distributed follower (Ray path only).
        # For MP multi-node followers, --headless and topology args injected
        # by _build_command_args, so no command swap is needed.
        if deployment_metadata.distributed_follower and executor_backend == "ray":
            ray_command, ray_command_args, ray_ports = self._build_ray_configuration(
                is_leader=False,
            )
            ray_command, ray_command_args = self._override_entrypoint(
                ray_command,
                ray_command_args,
                command_script,
            )

            run_container.execution.command = ray_command
            # run_container.execution.command_script = command_script # already set
            run_container.execution.args = ray_command_args
            run_container.ports = ray_ports

        # Create sidecar container for distributed leader (Ray path only).
        # Headless leader runs vLLM directly with DP coordination args; no
        # ray-head sidecar is required.
        sidecar_container = None
        if deployment_metadata.distributed_leader and executor_backend == "ray":
            run_container.mounts.append(
                ContainerMount(
                    path="/tmp",
                    volume="tmp-volume",
                ),
            )

            ray_command, ray_command_args, ray_ports = self._build_ray_configuration(
                is_leader=True,
            )
            ray_command, ray_command_args = self._override_entrypoint(
                ray_command,
                ray_command_args,
                command_script,
            )

            # Copy envs and override RAY_LOG_TO_STDERR for the sidecar
            # so Ray head logs go to stderr (captured by container log stream),
            # while keeping RAY_LOG_TO_STDERR=0 in the main container to avoid
            # polluting vLLM's log output with Ray worker logs.
            sidecar_envs = list(run_container.envs)
            ray_stderr_found = False
            for i, e in enumerate(sidecar_envs):
                if e.name == "RAY_LOG_TO_STDERR":
                    sidecar_envs[i] = ContainerEnv(name="RAY_LOG_TO_STDERR", value="1")
                    ray_stderr_found = True
                    break
            if not ray_stderr_found:
                sidecar_envs.append(ContainerEnv(name="RAY_LOG_TO_STDERR", value="1"))

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
                envs=sidecar_envs,
                resources=run_container.resources,
                mounts=run_container.mounts,
                ports=ray_ports,
            )

        logger.info(f"Creating vLLM container workload: {deployment_metadata.name}")
        logger.info(
            f"With image: {image}, "
            f"command: [{' '.join(command) if command else ''}], "
            f"arguments: [{' '.join(command_args) if command_args else ''}], "
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

    def _get_configured_env(
        self,
        is_distributed: bool,
        deployment_metadata: Optional[ModelInstanceDeploymentMetadata] = None,
    ) -> Dict[str, str]:
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

        # Persist the torch compile cache so repeated starts don't recompile.
        self._set_cache_env(env)

        # Apply LMCache environment variables if extended KV cache is enabled
        self._set_lmcache_env(env)

        # Apply distributed environment variables
        if is_distributed:
            self._set_distributed_env(env, deployment_metadata)

        # Apply Ascend-specific environment variables
        if is_ascend(self._get_selected_gpu_devices()):
            self._set_ascend_env(env)

        return env

    def _set_cache_env(self, env: Dict[str, str]):
        """
        Point VLLM_CACHE_ROOT at a persistent directory under gpustack's data dir
        so the torch compile cache survives container restarts. The directory is
        inherited by the inference container via gpustack-runtime's mirrored
        deployment (worker's data-dir mount is replicated to the vLLM container).
        """
        if "VLLM_CACHE_ROOT" in env:
            return
        if not self._config or not self._config.cache_dir:
            return
        cache_dir = os.path.join(self._config.cache_dir, "vllm")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
            logger.warning(
                f"Failed to create vLLM cache dir {cache_dir}: {e}. "
                "Torch compile cache will not be persisted."
            )
            return
        env["VLLM_CACHE_ROOT"] = cache_dir

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

    def _resolve_multinode_shape(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
    ) -> Optional[str]:
        """Best-effort shape lookup, swallowing inference errors. ``None`` if
        the topology cannot be derived (e.g. user-arg contradictions); the
        actual failure will surface from the command builder where it can be
        attributed to a specific instance."""
        try:
            return cal_multinode_topology(
                self._model_instance,
                deployment_metadata,
                parse_user_parallelism(self._model.backend_parameters),
            ).shape
        except ValueError:
            return None

    def _set_distributed_env(
        self,
        env: Dict[str, str],
        deployment_metadata: Optional[ModelInstanceDeploymentMetadata] = None,
    ):
        """
        Set up environment variables for distributed execution.
        """
        # Configure Internal communication IP and port.
        # see https://docs.vllm.ai/en/stable/configuration/env_vars.html.
        env["VLLM_HOST_IP"] = self._worker.ip

        ports = self._model_instance.ports or []
        executor_backend = resolve_executor_backend(
            self._model.backend_parameters, self._model.backend_version
        )
        if executor_backend == "mp":
            # VLLM_DP_MASTER_PORT belongs to the DP coordinator path
            # (vllm-project/vllm#42585). mp_only has dp=1 and no DP master,
            # so injecting it would bind a port for nobody to use.
            shape = (
                self._resolve_multinode_shape(deployment_metadata)
                if deployment_metadata is not None
                else None
            )
            if shape in ("dp_only", "nested") and ports:
                env["VLLM_DP_MASTER_PORT"] = str(ports[-1])
            # Pin vLLM's internal init port to a reserved one so get_open_port()
            # can't grab a kernel-stealable ephemeral port (#5657). Needs the
            # runner patch that makes get_open_port() honor VLLM_PORT.
            if len(ports) > 3:
                env["VLLM_PORT"] = str(ports[3])
        else:
            if ports:
                env["VLLM_PORT"] = str(ports[-1])
            env["RAY_LOG_TO_STDERR"] = env.pop("RAY_LOG_TO_STDERR", "0")
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
            SpeculativeAlgorithmEnum.MTP: "mtp",
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
        entrypoint: Optional[List[str]] = None,
        deployment_metadata: Optional[ModelInstanceDeploymentMetadata] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Build vLLM command arguments for container execution.

        Returns:
            A tuple of (full_arguments, injected_backend_parameters) where
            injected_backend_parameters contains only the arguments automatically
            added by GPUStack, excluding the entrypoint/model path and
            user-specified backend parameters.
        """
        ctx = self._make_args_context(
            port, is_distributed, entrypoint, deployment_metadata
        )

        arguments: List[str] = [self._model_path]
        arguments = self.build_versioned_command_args(arguments)

        arguments.extend(self._build_omni_arguments(ctx))
        arguments.extend(self._build_max_model_len_arguments(ctx))
        arguments.extend(
            get_auto_parallelism_arguments(
                self._model.backend_parameters,
                self._model_instance,
                ctx.is_distributed,
                ctx.deployment_metadata,
                self._model.backend_version,
            )
        )
        arguments.extend(self._get_speculative_arguments())
        arguments.extend(
            get_access_log_arguments(
                self._model.backend_parameters, self._model.backend_version
            )
        )
        arguments.extend(
            get_cache_report_arguments(
                self._model.backend_parameters, self._model.backend_version
            )
        )
        arguments.extend(self._build_ray_distributed_arguments(ctx))
        arguments.extend(self._build_mp_multinode_arguments(ctx))
        arguments.extend(self._build_extended_kv_cache_arguments(ctx))
        arguments.extend(self._build_ascend_310p_arguments(ctx))

        extend_vllm_mounted_lora_arguments(
            arguments,
            self._model_instance.mounted_loras,
            self._model.name,
            self._model.backend_parameters,
        )

        # Inject user-defined backend parameters
        user_backend_parameters = self._flatten_backend_param()
        arguments.extend(user_backend_parameters)

        extend_args_no_exist(
            arguments,
            ("--host", self._worker.ip),
            ("--port", str(ctx.port)),
            ("--served-model-name", self._model_instance.model_name),
        )

        # An API-serving subordinate (hybrid-LB / external-LB) binds ports[0] on
        # its own IP, but that port number is chosen on the leader node and never
        # re-checked here, so it may already be taken on this worker. Fail fast
        # with a clear message instead of letting vLLM crash on bind.
        if (
            subordinates_serve_api(self._model.backend_parameters)
            and ctx.topology is not None
            and ctx.topology.is_follower
            and not network.is_port_available(ctx.port, host=self._worker.ip)
        ):
            raise ValueError(
                f"vLLM subordinate serving port {ctx.port} is already in use on "
                f"worker {self._worker.ip}. The port is selected on the leader node "
                "and shared across nodes but was not verified free here; free it on "
                "this worker, then redeploy."
            )

        injected = self._get_injected_backend_parameters(
            arguments, user_backend_parameters, entrypoint
        )
        return arguments, injected

    def _make_args_context(
        self,
        port: int,
        is_distributed: bool,
        entrypoint: Optional[List[str]],
        deployment_metadata: Optional[ModelInstanceDeploymentMetadata],
    ) -> _VLLMArgsContext:
        executor_backend = resolve_executor_backend(
            self._model.backend_parameters, self._model.backend_version
        )
        topology: Optional[MultinodeTopology] = None
        if (
            is_distributed
            and executor_backend == "mp"
            and deployment_metadata is not None
        ):
            topology = cal_multinode_topology(
                self._model_instance,
                deployment_metadata,
                parse_user_parallelism(self._model.backend_parameters),
            )
        return _VLLMArgsContext(
            port=port,
            is_distributed=is_distributed,
            executor_backend=executor_backend,
            topology=topology,
            is_omni=is_omni_model(self._model),
            is_audio=is_audio_model(self._model),
            entrypoint=entrypoint,
            deployment_metadata=deployment_metadata,
        )

    def _build_omni_arguments(self, ctx: _VLLMArgsContext) -> List[str]:
        if not ctx.is_omni:
            return []
        if find_bool_parameter(self._model.backend_parameters, ["omni"]):
            return []
        return ["--omni"]

    def _build_max_model_len_arguments(self, ctx: _VLLMArgsContext) -> List[str]:
        if ctx.is_omni or ctx.is_audio:
            return []
        if (
            find_parameter(self._model.backend_parameters, ["max-model-len"])
            is not None
        ):
            return []
        derived_max_model_len = self._derive_max_model_len()
        if derived_max_model_len and derived_max_model_len > 8192:
            return ["--max-model-len", "8192"]
        return []

    def _build_ray_distributed_arguments(self, ctx: _VLLMArgsContext) -> List[str]:
        """
        Ray sidecar path only: force --distributed-executor-backend ray and
        default DP backend to ray when user-specified DP > 1.
        """
        if not ctx.is_distributed or ctx.executor_backend != "ray":
            return []

        arguments: List[str] = ["--distributed-executor-backend", "ray"]
        dps = find_int_parameter(
            self._model.backend_parameters, ["data-parallel-size", "dp"]
        )
        if dps and dps > 1:
            dpb = find_parameter(
                self._model.backend_parameters, ["data-parallel-backend", "dpb"]
            )
            if dpb is None:
                arguments.extend(["--data-parallel-backend", "ray"])
            # ports[1] is reserved for DP RPC, see gpustack/worker/serve_manager.py.
            arguments.extend(
                ["--data-parallel-rpc-port", str(self._model_instance.ports[1])]
            )
        return arguments

    def _build_mp_multinode_arguments(
        self,
        ctx: _VLLMArgsContext,
    ) -> List[str]:
        """
        MP multi-node (non-Ray) path. Emits the parameter shape matching the
        resolved topology, then suppresses the API server on headless followers.

        - ``dp_only``  → ``--data-parallel-*`` only; every node is a DP engine
          head (no PP/TP spans nodes). See :meth:`_dp_only_args`.
        - ``mp_only``  → ``--nnodes`` + ``--node-rank``; one DP rank spread over
          all nodes for cross-node TP/PP.
        - ``nested``   → both sets; vLLM derives node role internally via
          ``node_rank % nnodes_within_dp``.

        Followers carry ``--headless`` to skip the API server (the leader always
        exposes it). Exception: when subordinates serve their own API (hybrid-LB
        / external-LB) every DP engine serves, so ``--headless`` is not injected.
        """
        if (
            not ctx.is_distributed
            or ctx.executor_backend != "mp"
            or ctx.topology is None
        ):
            return []

        topology = ctx.topology
        leader_ip = self._model_instance.worker_ip
        # serve_manager._assign_ports reserves ports[1] for the DP RPC endpoint
        # (ZMQ) and ports[2] for the PyTorch master endpoint (TCPStore). The two
        # channels use different protocols and can't share one port, so each
        # shape consumes the subset it needs.
        dp_rpc_port = str(self._model_instance.ports[1])
        master_port = str(self._model_instance.ports[2])

        arguments: List[str] = []
        if (
            find_parameter(
                self._model.backend_parameters, ["distributed-executor-backend"]
            )
            is None
        ):
            arguments.extend(["--distributed-executor-backend", "mp"])

        load_balance_mode = resolve_data_parallel_load_balance_mode(
            self._model.backend_parameters,
        )

        if topology.shape == "dp_only":
            shape_args = self._dp_only_args(
                topology, leader_ip, dp_rpc_port, load_balance_mode
            )
        elif topology.shape == "mp_only":
            shape_args = self._cross_node_args(topology, leader_ip, master_port)
        else:  # nested
            shape_args = self._cross_node_args(
                topology, leader_ip, master_port, dp_rpc_port=dp_rpc_port
            )
        extend_args_no_exist(arguments, *shape_args)

        if topology.is_follower and not subordinates_serve_api(
            self._model.backend_parameters
        ):
            extend_args_no_exist(arguments, "--headless")
        return arguments

    def _dp_only_args(
        self,
        topology: "MultinodeTopology",
        leader_ip: str,
        dp_rpc_port: str,
        load_balance_mode: str,
    ) -> List[Tuple[str, str]]:
        """Per-node DP-head arguments for the ``dp_only`` shape.

        Internal-LB / hybrid-LB emit ``--data-parallel-start-rank``. external-LB
        is one rank per node, so it emits a per-node ``--data-parallel-rank``
        (start_rank == this node's DP rank when each node hosts a single rank) —
        unless the user pinned the rank, in which case GPUStack defers to it.
        """
        args = [
            ("--data-parallel-address", leader_ip),
            ("--data-parallel-rpc-port", dp_rpc_port),
        ]
        backend_parameters = self._model.backend_parameters
        if find_parameter(backend_parameters, ["data-parallel-rank"]) is not None:
            # User input wins: leave their --data-parallel-rank untouched. A
            # single shared value across workers would collapse every node to the
            # same rank, so warn for multi-worker layouts.
            logger.warning(
                "vLLM external-LB: --data-parallel-rank is user-specified; "
                "GPUStack will not auto-assign per-node ranks. Ensure each worker "
                "gets a distinct rank in a multi-worker deployment."
            )
            return args
        if load_balance_mode == "external":
            if topology.dpl != 1:
                raise ValueError(
                    "vLLM external-LB requires exactly one DP rank per node "
                    f"(this node hosts {topology.dpl}); tensor-parallel-size * "
                    "pipeline-parallel-size must equal the node's GPU count."
                )
            return [("--data-parallel-rank", str(topology.start_rank)), *args]
        return [("--data-parallel-start-rank", str(topology.start_rank)), *args]

    def _cross_node_args(
        self,
        topology: "MultinodeTopology",
        leader_ip: str,
        master_port: str,
        dp_rpc_port: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Cross-node TP/PP arguments shared by ``mp_only`` and ``nested``.

        Passing ``dp_rpc_port`` adds the DP channel (the ``nested`` case), where
        vLLM derives each node's DP rank from ``node_rank // nnodes_within_dp``.
        """
        args = [
            ("--nnodes", str(topology.nnodes)),
            ("--node-rank", str(topology.node_rank)),
            ("--master-addr", leader_ip),
            ("--master-port", master_port),
        ]
        if dp_rpc_port is not None:
            args += [
                ("--data-parallel-address", leader_ip),
                ("--data-parallel-rpc-port", dp_rpc_port),
            ]
        return args

    def _build_extended_kv_cache_arguments(self, ctx: _VLLMArgsContext) -> List[str]:
        extended = self._model.extended_kv_cache
        if not (extended and extended.enabled):
            return []

        vendor, _, _ = self._get_device_info()
        if vendor not in {
            manufacturer_to_backend(ManufacturerEnum.NVIDIA),
            manufacturer_to_backend(ManufacturerEnum.AMD),
        }:
            logger.warning(
                "Extended KV cache for vLLM is only supported on NVIDIA and AMD GPUs. Skipping LMCache configuration."
            )
            return []

        return [
            "--kv-transfer-config",
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
        ]

    def _build_ascend_310p_arguments(self, ctx: _VLLMArgsContext) -> List[str]:
        if not is_ascend_310p(self._get_selected_gpu_devices()):
            return []
        return ["--enforce-eager", "--dtype", "float16"]

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


@dataclass
class MultinodeTopology:
    """Resolved multi-node deployment topology from one node's perspective.

    Fields:
      - ``shape``: which set of vLLM arguments to emit (see
        :py:meth:`VLLMServer._build_mp_multinode_arguments`).
      - ``tp`` / ``pp``: cluster-wide, identical on every node.
      - ``dp``: total DP rank count across the whole cluster.
      - ``dpl``: ``--data-parallel-size-local`` value to emit on this node.
        In ``dp_only`` shape this may differ per node (heterogeneous clusters);
        in ``mp_only`` / ``nested`` it is always 1.
      - ``nnodes`` / ``node_rank``: physical cluster size and this node's
        position in it (0 = leader).
      - ``start_rank``: ``--data-parallel-start-rank`` to emit in ``dp_only``.
        In other shapes vLLM derives DP rank internally from
        ``node_rank % nnodes_within_dp``; we still surface the implied value
        for diagnostics.
      - ``is_follower``: whether this node should carry ``--headless``.
    """

    shape: MultinodeShape
    tp: int
    pp: int
    dp: int
    dpl: int
    nnodes: int
    node_rank: int
    start_rank: int
    is_follower: bool


def cal_multinode_topology(
    model_instance: ModelInstance,
    deployment_metadata: ModelInstanceDeploymentMetadata,
    user: Optional[MultinodeUserParallelism] = None,
) -> MultinodeTopology:
    """Translate user-supplied parallelism hints + physical topology into the
    vLLM parameter shape this node should emit.

    Delegates cluster-level validation and shape inference to
    :func:`validate_multinode_topology`, then layers on the node-perspective
    fields (``my_idx`` / ``start_rank`` / ``is_follower``).
    """
    g_main = len(model_instance.gpu_indexes) if model_instance.gpu_indexes else 1
    subordinate = (
        model_instance.distributed_servers.subordinate_workers
        if model_instance.distributed_servers
        and model_instance.distributed_servers.subordinate_workers
        else []
    )
    gpu_per_node = [g_main] + [len(s.gpu_indexes or []) for s in subordinate]
    nnodes = len(gpu_per_node)

    # This node's physical rank.
    if deployment_metadata.distributed_follower:
        my_idx = (deployment_metadata.distributed_follower_index or 0) + 1
        is_follower = True
    else:
        my_idx = 0
        is_follower = False
    if my_idx >= nnodes:
        raise ValueError(
            f"vLLM multi-node: follower index {my_idx} out of bounds "
            f"(cluster has {nnodes} workers)."
        )

    validated = validate_multinode_topology(gpu_per_node, user)

    if validated.shape == "dp_only":
        start_rank = sum(validated.dpl_per_node[:my_idx])
    else:
        # mp_only / nested: vLLM derives DP rank from node_rank // nnodes_within_dp.
        start_rank = my_idx // validated.nnodes_within_dp

    return MultinodeTopology(
        shape=validated.shape,
        tp=validated.tp,
        pp=validated.pp,
        dp=validated.dp,
        dpl=validated.dpl_per_node[my_idx],
        nnodes=nnodes,
        node_rank=my_idx,
        start_rank=start_rank,
        is_follower=is_follower,
    )


def get_auto_parallelism_arguments(
    backend_parameters: List[str],
    model_instance: ModelInstance,
    is_distributed: bool,
    deployment_metadata: Optional[ModelInstanceDeploymentMetadata] = None,
    backend_version: Optional[str] = None,
) -> List[str]:
    if (
        is_distributed
        and deployment_metadata is not None
        and resolve_executor_backend(backend_parameters, backend_version) == "mp"
    ):
        # MP multi-node: derive shape-specific defaults for tp/pp/dp/dpl
        # so the cluster works out-of-the-box; subordinate workers receive
        # the same backend_parameters, so any non-None field acts as a hard
        # cluster-wide constraint.
        user = parse_user_parallelism(backend_parameters)
        topology = cal_multinode_topology(model_instance, deployment_metadata, user)
        derived: List[str] = []
        if user.tp is None:
            derived.extend(["--tensor-parallel-size", str(topology.tp)])
        if user.pp is None and topology.pp > 1:
            derived.extend(["--pipeline-parallel-size", str(topology.pp)])
        # dp/dpl are only meaningful when the shape actually emits them
        # (dp_only and nested). mp_only operates with dp=1, dpl=1 implicit.
        if topology.shape in ("dp_only", "nested"):
            if user.dp is None:
                derived.extend(["--data-parallel-size", str(topology.dp)])
            if user.dpl is None:
                derived.extend(["--data-parallel-size-local", str(topology.dpl)])
        return derived

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
        # distributed across multiple workers (Ray sidecar path)
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


def get_access_log_arguments(
    backend_parameters: List[str], backend_version: Optional[str] = None
) -> List[str]:
    """
    Get default vLLM access log filter arguments.
    --disable-access-log-for-endpoints was introduced in vLLM 0.16.0.
    """
    if not backend_version:
        return []
    if compare_versions(backend_version, "0.16.0") < 0:
        return []

    access_log_filter = find_parameter(
        backend_parameters,
        ["disable-access-log-for-endpoints"],
    )
    if access_log_filter is not None:
        return []

    return ["--disable-access-log-for-endpoints", "/metrics"]


def get_cache_report_arguments(
    backend_parameters: List[str], backend_version: Optional[str] = None
) -> List[str]:
    """
    Auto-inject `--enable-prompt-tokens-details` so vLLM populates
    `usage.prompt_tokens_details.cached_tokens` in OpenAI responses.

    Only injected for vLLM >= v0.9.0.1 — earlier V1 builds silently dropped
    the field (https://github.com/vllm-project/vllm/pull/18149).

    Prefix caching itself is the user's responsibility (`--enable-prefix-caching`):
    V1 has it on by default, V0 does not.
    """
    if not backend_version:
        return []
    if compare_versions(backend_version, "0.9.0.1") < 0:
        return []
    if find_bool_parameter(
        backend_parameters,
        ["enable-prompt-tokens-details", "no-enable-prompt-tokens-details"],
    ):
        return []
    return ["--enable-prompt-tokens-details"]
