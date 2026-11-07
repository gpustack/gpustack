import json
import logging
import os
from pathlib import Path
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
    ContainerFile,
)
from gpustack_runtime.deployer.__utils__ import compare_versions
from gpustack_runtime.detector import ManufacturerEnum, manufacturer_to_backend
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstance,
    SpeculativeAlgorithmEnum,
    SpeculativeConfig,
    ModelInstanceDeploymentMetadata,
    get_backend,
    is_audio_model,
    is_omni_model,
    DPRankPlacement,
    ModelInstanceWorkload,
    ModelInstanceComponent,
    ModelInstanceWorkloadStateEnum,
)
from gpustack.utils import network
from gpustack.utils.command import (
    ExecutorBackend,
    find_parameter,
    find_last_parameter,
    find_bool_parameter,
    find_int_parameter,
    extend_args_no_exist,
    format_backend_parameters,
    resolve_executor_backend,
)
from gpustack.utils.envs import sanitize_env
from gpustack.utils.unit import byte_to_gib
from gpustack.utils.vllm_kv_cache import ascend_local_kv_cache_unsupported_reason
from gpustack.utils.vllm_topology import (
    MultinodeShape,
    MultinodeUserParallelism,
    parse_user_parallelism,
    validate_multinode_topology,
    external_lb_parallelism,
    managed_lb_parallelism,
    multi_port_external_lb_parallelism,
    validate_multi_port_external_lb_parameters,
    validate_managed_http_parameters,
    external_lb_fragments,
    vllm_lb_mode,
    normalize_vllm_parameters,
)
from gpustack.utils.model_config import get_hf_text_config
from gpustack.utils.template import render_values
from gpustack.worker.pd_injection import managed_pd_engine_id
from gpustack.utils.model_instance_workers import get_model_instance_workloads
from gpustack.worker.kv_transfer import (
    TOP_LEVEL_ONLY_KEYS,
    compose_kv_transfer_config,
    descriptor_in,
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

_DEVICE_TASK_EXEC = """import json, os, sys
key, count, indexes = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
visible = os.environ.get(key)
if visible is None and key != 'CUDA_VISIBLE_DEVICES':
    raise SystemExit('Missing runtime visible device mask: ' + key)
pool = visible.split(',') if visible is not None else [str(i) for i in range(count)]
if len(pool) != count or any(not item.strip() for item in pool):
    raise SystemExit('Runtime visible device count differs from the assigned device count')
os.environ[key] = ','.join(pool[i].strip() for i in indexes)
if key == 'ASCEND_RT_VISIBLE_DEVICES':
    os.environ['NPU_VISIBLE_DEVICES'] = os.environ[key]
cache_root = os.environ.get('VLLM_CACHE_ROOT')
if cache_root:
    # AOT graphs encode logical device indices, which depend on the local pool size.
    os.environ['VLLM_CACHE_ROOT'] = os.path.join(
        cache_root, 'isolated_devices', str(len(indexes))
    )
os.execvpe(sys.argv[4], sys.argv[4:], os.environ)
"""


def _top_level_only_keys(model) -> Tuple[str, ...]:
    """Settings this engine reads only off a top-level connector descriptor.

    Declared in `pd-modes.yaml` under `composed_cache.top_level_only` and read
    here rather than inside the composer, which stays a pure function over
    descriptors. A catalog that cannot be reached is not a reason to fail a
    launch: the composition still happens, just without lifting anything, which
    is the behaviour of every release before the declaration existed.
    """
    try:
        from gpustack.server.pd_mode_catalog import get_composed_cache

        declared = get_composed_cache(get_backend(model))
    except Exception as e:  # noqa: BLE001 - catalog is advisory here
        logger.debug("Could not read the composed-cache declaration: %s", e)
        return TOP_LEVEL_ONLY_KEYS
    return tuple(declared.top_level_only) if declared else TOP_LEVEL_ONLY_KEYS


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
            is_distributed=(
                deployment_metadata.distributed
                and get_model_instance_workloads(self._model_instance) is None
            ),
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
        managed = get_model_instance_workloads(self._model_instance) is not None
        original_command, original_args = command, command_args
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
            files=self._get_configured_files(),
            ports=ports,
        )

        executor_backend = resolve_executor_backend(
            self._model.backend_parameters,
            self._model.backend_version,
            self._model.image_name,
        )
        if managed:
            executor_backend = "mp"

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
                files=run_container.files,
                ports=ray_ports,
            )

        logger.info(
            "%s vLLM workload: %s",
            "Preparing" if managed else "Creating",
            deployment_metadata.name,
        )
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
            # Host IPC only when a shared cache service is attached: its
            # CUDA-IPC transfer path imports the engine's KV buffers by IPC
            # handle. Everything else keeps the private /dev/shm sized by
            # shm_size.
            host_ipc=self._host_ipc_enabled(),
            shm_size=int(container_config.shm_size_gib * (1 << 30)),
            containers=(
                [run_container]
                if not sidecar_container
                else [run_container, sidecar_container]
            ),
            run_as_user=container_config.user,
            run_as_group=container_config.group,
        )
        if managed:
            configure = {
                "multi_port_external": self._configure_multi_port_external_workload,
                "hybrid": self._configure_hybrid_workload,
                "external": self._configure_external_workload,
            }[vllm_lb_mode(self._model.backend_parameters)]
            manifest = configure(workload_plan, original_command, original_args)
            self._workload_plan_callback(
                [manifest], [self._transform_workload_plan(workload_plan)]
            )
        else:
            create_workload(self._transform_workload_plan(workload_plan))

        logger.info(
            "%s vLLM workload: %s",
            "Prepared" if managed else "Created",
            deployment_metadata.name,
        )

    def _configure_multi_port_external_workload(
        self, plan: WorkloadPlan, command: Optional[List[str]], arguments: List[str]
    ) -> ModelInstanceWorkload:
        """Register multi-port External-LB child APIs while retaining its single launcher."""
        mi = self._model_instance
        if mi.role:
            from gpustack.validation.pd.roles import validate_external_pd_membership

            validate_external_pd_membership(
                str(self._model.disaggregation.mode),
                BackendEnum.VLLM,
                self._model.backend_parameters,
            )
        parameters = normalize_vllm_parameters(
            list(command or []) + arguments, tokenized=True
        )
        validate_multi_port_external_lb_parameters(parameters, tokenized=True)
        config = get_hf_text_config(self._get_pretrained_config())
        if not any(
            getattr(config, key, 0)
            for key in (
                "n_routed_experts",
                "num_local_experts",
                "num_experts",
                "moe_num_experts",
            )
        ):
            raise ValueError(
                "Multi-port External-LB requires a verified MoE model configuration."
            )
        devices = self._get_selected_gpu_devices()
        if (
            not devices
            or devices[0].type not in ("cuda", "cann")
            or any(device.type != devices[0].type for device in devices)
        ):
            raise ValueError(
                "Multi-port External-LB requires homogeneous CUDA or Ascend devices."
            )
        if mi.role and devices[0].type != "cuda":
            raise ValueError("Multi-port External-LB PD supports only CUDA NIXL.")
        members = [mi] + list(mi.distributed_servers.subordinate_workers or [])
        index = [member.worker_id for member in members].index(self._worker.id)
        counts = [
            len(devices) if i == index else len(w.gpu_indexes or [])
            for i, w in enumerate(members)
        ]
        topology = multi_port_external_lb_parallelism(
            self._model.backend_parameters, counts
        )
        if (
            multi_port_external_lb_parallelism(parameters, counts, tokenized=True)
            != topology
        ):
            raise ValueError(
                "Multi-port External-LB command must match the allocated parallelism."
            )
        fragments = external_lb_fragments(counts, topology, index)
        start = find_last_parameter(
            parameters, ["data-parallel-start-rank", "dpr", "dps"], tokenized=True
        )
        if start is not None and int(start) != fragments[0].dp_rank:
            raise ValueError(
                "Multi-port External-LB start rank must match the allocated Worker ranks."
            )
        bands = mi.named_ports or {}
        api = bands.get("_vllm_external_api")
        rpc = bands.get("_vllm_external_rpc")
        supervisor = bands.get("_vllm_multi_port_external_supervisor")
        master = bands.get("_vllm_mp_connecting")
        if (
            not api
            or api.count != topology.dp
            or not rpc
            or not supervisor
            or not master
        ):
            raise ValueError(
                "Multi-port External-LB API, RPC and supervisor ports must be allocated first."
            )
        requested = find_last_parameter(
            parameters, ["data-parallel-supervisor-port"], tokenized=True
        )
        if requested is not None and int(requested) != supervisor.base:
            raise ValueError(
                "Multi-port External-LB command must use the reserved supervisor port."
            )
        if api.base <= supervisor.base < api.base + api.count:
            raise ValueError(
                "Multi-port External-LB supervisor port overlaps rank APIs."
            )
        base = self._external_command(command, arguments)
        base += [
            "--data-parallel-multi-port-external-lb",
            "--data-parallel-size",
            str(topology.dp),
            "--data-parallel-size-local",
            str(len(fragments)),
            "--data-parallel-start-rank",
            str(fragments[0].dp_rank),
            "--tensor-parallel-size",
            str(topology.tp),
            "--pipeline-parallel-size",
            str(topology.pp),
            "--data-parallel-address",
            mi.worker_ip or self._worker.ip,
            "--data-parallel-rpc-port",
            str(rpc.base),
            "--data-parallel-supervisor-port",
            str(supervisor.base),
            "--api-server-count",
            "1",
            "--host",
            self._worker.ip,
            "--port",
            str(api.base + fragments[0].dp_rank),
        ]
        container = plan.containers[0]
        if mi.role in ("prefill", "decode"):
            base, pd_env = self._managed_pd_endpoint(
                base, api.base + fragments[0].dp_rank, container
            )
            env = {item.name: item.value for item in container.envs or []}
            env.update(pd_env)
            container.envs = [
                ContainerEnv(name=name, value=value) for name, value in env.items()
            ]
        container.envs = [
            item
            for item in container.envs or []
            if item.name
            not in ("VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT", "VLLM_PORT")
        ] + [
            ContainerEnv(
                name="VLLM_DP_MASTER_IP", value=mi.worker_ip or self._worker.ip
            ),
            ContainerEnv(name="VLLM_DP_MASTER_PORT", value=str(master.base)),
        ]
        container.execution.command, container.execution.args = (
            self._override_entrypoint(base, [], container.execution.command_script)
        )
        self._configure_shutdown_budget(plan, [{'argv': base}])
        return ModelInstanceWorkload(
            name=plan.name,
            worker_id=self._worker.id,
            state=ModelInstanceWorkloadStateEnum.PENDING,
            dp_rank_placements=[
                DPRankPlacement(
                    dp_rank=f.dp_rank,
                    node_rank=0,
                    gpu_indexes=[devices[position].index for position in f.positions],
                    api_port=api.base + f.dp_rank,
                    ready=False,
                )
                for f in fragments
            ],
        )

    def _configure_hybrid_workload(
        self, plan: WorkloadPlan, command: Optional[List[str]], arguments: List[str]
    ) -> ModelInstanceWorkload:
        """Run one native shared API, or supervise all allocated compute fragments."""
        mi = self._model_instance
        parameters = normalize_vllm_parameters(
            list(command or []) + arguments, tokenized=True
        )
        validate_managed_http_parameters(parameters, tokenized=True)
        if vllm_lb_mode(parameters, tokenized=True) not in ("internal", "hybrid"):
            raise ValueError("The command must preserve the selected Hybrid-LB mode.")
        devices = self._get_selected_gpu_devices()
        members = [mi] + list(mi.distributed_servers.subordinate_workers or [])
        index = [member.worker_id for member in members].index(self._worker.id)
        counts = [
            len(devices) if i == index else len(w.gpu_indexes or [])
            for i, w in enumerate(members)
        ]
        topology = managed_lb_parallelism(self._model.backend_parameters, counts)
        if vllm_lb_mode(self._model.backend_parameters) != "hybrid":
            raise ValueError("Hybrid components require a Hybrid-LB model.")
        fragments = external_lb_fragments(counts, topology, index)
        api = (mi.named_ports or {}).get("_vllm_external_api")
        rpc = (mi.named_ports or {}).get("_vllm_external_rpc")
        if not api or api.count != 1 or not rpc or rpc.count != 1:
            raise ValueError("Hybrid API and RPC ports must be allocated first.")
        if (
            not devices
            or devices[0].type not in ("cuda", "cann")
            or any(d.type != devices[0].type for d in devices)
        ):
            raise ValueError("Hybrid-LB requires homogeneous CUDA or Ascend devices.")
        container = plan.containers[0]
        owns_api = fragments[0].node_rank == 0
        base = self._external_command(
            command, arguments, keep_api_server_count=owns_api
        )
        if mi.role in ("prefill", "decode"):
            base, pd_env = self._managed_pd_endpoint(base, api.base, container)
            env = {item.name: item.value for item in container.envs or []}
            env.update(pd_env)
            container.envs = [
                ContainerEnv(name=name, value=value) for name, value in env.items()
            ]
        if owns_api:
            if (
                find_last_parameter(parameters, ["api-server-count"], tokenized=True)
                == "0"
            ):
                raise ValueError("A Hybrid API task requires API server processes.")
            child = base + self._hybrid_launch_arguments(
                topology, fragments[0], len(fragments)
            )
            child += [
                "--data-parallel-hybrid-lb",
                "--host",
                self._worker.ip,
                "--port",
                str(api.base),
            ]
            container.execution.command, container.execution.args = (
                self._override_entrypoint(child, [], container.execution.command_script)
            )
            self._configure_shutdown_budget(plan, [{'argv': child}])
        else:
            visible = (
                "CUDA_VISIBLE_DEVICES"
                if devices[0].type == "cuda"
                else "ASCEND_RT_VISIBLE_DEVICES"
            )
            tasks = [
                {
                    "name": f"dp{fragment.dp_rank}-node{fragment.node_rank}",
                    "argv": [
                        "python3",
                        "-c",
                        _DEVICE_TASK_EXEC,
                        visible,
                        str(len(devices)),
                        json.dumps(fragment.positions),
                        *base,
                        *self._hybrid_launch_arguments(topology, fragment, 1),
                        "--headless",
                        "--api-server-count",
                        "0",
                    ],
                }
                for fragment in fragments
            ]
            self._configure_rank_supervisor(plan, tasks)
        master = (mi.named_ports or {}).get("_vllm_mp_connecting")
        container.envs = [
            e
            for e in container.envs or []
            if e.name not in ("VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT", "VLLM_PORT")
        ] + [
            ContainerEnv(
                name="VLLM_DP_MASTER_IP", value=mi.worker_ip or self._worker.ip
            )
        ]
        if master:
            container.envs.append(
                ContainerEnv(name="VLLM_DP_MASTER_PORT", value=str(master.base))
            )
        return ModelInstanceWorkload(
            name=plan.name,
            worker_id=self._worker.id,
            state=ModelInstanceWorkloadStateEnum.PENDING,
            component=ModelInstanceComponent(
                kind="vllm_hybrid",
                api_port=api.base if owns_api else None,
                dp_size=len(fragments) if owns_api else None,
            ),
        )

    def _hybrid_launch_arguments(self, topology, fragment, local_dp) -> List[str]:
        mi = self._model_instance
        arguments = [
            "--data-parallel-size",
            str(topology.dp),
            "--data-parallel-size-local",
            str(local_dp),
            "--data-parallel-start-rank",
            str(fragment.dp_rank),
            "--tensor-parallel-size",
            str(topology.tp),
            "--pipeline-parallel-size",
            str(topology.pp),
            "--data-parallel-address",
            mi.worker_ip or self._worker.ip,
            "--data-parallel-rpc-port",
            str(mi.named_ports["_vllm_external_rpc"].base),
        ]
        if topology.nnodes_within_dp > 1:
            store = mi.named_ports.get("_vllm_external_store")
            master = mi.named_ports.get("_vllm_mp_connecting")
            if not store or store.count != topology.dp or not master:
                raise ValueError("Cross-Worker Hybrid ranks require reserved MP ports.")
            arguments += [
                "--distributed-executor-backend",
                "mp",
                "--data-parallel-backend",
                "mp",
                "--nnodes",
                str(topology.dp * topology.nnodes_within_dp),
                "--node-rank",
                str(fragment.dp_rank * topology.nnodes_within_dp + fragment.node_rank),
                "--master-addr",
                mi.worker_ip,
                "--master-port",
                str(store.base),
            ]
        return arguments

    def _configure_external_workload(
        self, plan: WorkloadPlan, command: Optional[List[str]], arguments: List[str]
    ) -> ModelInstanceWorkload:
        """Build one container with all local rank commands, without starting it."""
        mi = self._model_instance
        parameters = normalize_vllm_parameters(
            list(command or []) + arguments, tokenized=True
        )
        validate_managed_http_parameters(parameters, tokenized=True)
        if vllm_lb_mode(parameters, tokenized=True) not in ("internal", "external"):
            raise ValueError("The command must preserve the selected External-LB mode.")
        if (
            mi.role not in (None, "prefill", "decode")
            or vllm_lb_mode(self._model.backend_parameters) != "external"
        ):
            raise ValueError(
                "Managed External-LB requires an External-LB engine instance."
            )
        devices = self._get_selected_gpu_devices()
        if (
            not devices
            or devices[0].type not in ("cuda", "cann")
            or any(d.type != devices[0].type for d in devices)
        ):
            raise ValueError(
                "Managed External-LB requires homogeneous CUDA or Ascend devices."
            )
        members = [mi] + list(mi.distributed_servers.subordinate_workers or [])
        worker_index = [w.worker_id for w in members].index(self._worker.id)
        counts = [
            len(devices) if i == worker_index else len(w.gpu_indexes or [])
            for i, w in enumerate(members)
        ]
        topology = external_lb_parallelism(self._model.backend_parameters, counts)
        if topology.dp > 1:
            config = get_hf_text_config(self._get_pretrained_config())
            if not any(
                getattr(config, key, 0)
                for key in (
                    "n_routed_experts",
                    "num_local_experts",
                    "num_experts",
                    "moe_num_experts",
                )
            ):
                raise ValueError(
                    "Managed vLLM External-LB requires a verified MoE model configuration."
                )
        fragments = external_lb_fragments(counts, topology, worker_index)
        device_type = devices[0].type
        visible_env = (
            "CUDA_VISIBLE_DEVICES"
            if device_type == "cuda"
            else "ASCEND_RT_VISIBLE_DEVICES"
        )
        base = self._external_command(command, arguments)
        ports = mi.ports or []
        rpc = (mi.named_ports or {}).get("_vllm_external_rpc")
        master = (mi.named_ports or {}).get("_vllm_mp_connecting")
        ports = [
            port
            for port in ports
            if not any(
                band.base <= port < band.base + band.count
                for name, band in (mi.named_ports or {}).items()
                if name != "_vllm_external_api"
            )
        ]
        if len(ports) != topology.dp or not rpc or not master:
            raise ValueError(
                "Managed External-LB rank and communication ports must be allocated first."
            )
        tasks, placements = self._external_rank_tasks(
            base,
            devices,
            topology,
            visible_env,
            ports,
            rpc.base,
            fragments,
            members,
            plan.containers[0],
        )
        container = plan.containers[0]
        self._configure_rank_supervisor(plan, tasks)
        container.envs = [
            item
            for item in container.envs or []
            if item.name
            not in ("VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT", "VLLM_PORT")
        ] + [
            ContainerEnv(
                name="VLLM_DP_MASTER_IP", value=mi.worker_ip or self._worker.ip
            ),
            ContainerEnv(name="VLLM_DP_MASTER_PORT", value=str(master.base)),
        ]
        return ModelInstanceWorkload(
            name=plan.name,
            worker_id=self._worker.id,
            state=ModelInstanceWorkloadStateEnum.PENDING,
            dp_rank_placements=placements,
        )

    def _configure_shutdown_budget(self, plan: WorkloadPlan, tasks: List[Dict]) -> int:
        """Let vLLM drain and release devices before parent and container cleanup."""
        container = plan.containers[0]
        env = {item.name: item.value for item in container.envs or []}
        shutdown_timeout = 0
        for task in tasks:
            argv, request_timeout = task['argv'], '0'
            for index, argument in enumerate(argv):
                key, separator, value = argument.partition('=')
                if key.replace('_', '-') == '--shutdown-timeout':
                    request_timeout = (
                        value
                        if separator
                        else (argv[index + 1] if index + 1 < len(argv) else '')
                    )
            worker_timeout = {**env, **task.get('env', {})}.get(
                'VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS', '5'
            )
            values = []
            for name, value in (
                ('--shutdown-timeout', request_timeout),
                ('VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS', worker_timeout),
            ):
                try:
                    parsed = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'{name} must be a non-negative integer') from exc
                if parsed < 0:
                    raise ValueError(f'{name} must be a non-negative integer')
                values.append(parsed)
            # vLLM's DP Supervisor reserves five seconds beyond child shutdown.
            shutdown_timeout = max(shutdown_timeout, max(values) + 5)
        # Reserve separate windows for forced process reaping and container exit.
        plan.termination_grace_period_seconds = max(
            plan.termination_grace_period_seconds, shutdown_timeout + 10
        )
        logger.info(
            'vLLM shutdown budget: graceful=%ss, cleanup=5s, container=%ss',
            shutdown_timeout,
            plan.termination_grace_period_seconds,
        )
        return shutdown_timeout

    def _configure_rank_supervisor(self, plan, tasks) -> None:
        """Keep every local compute child under the container's process lifecycle."""
        container = plan.containers[0]
        shutdown_timeout = self._configure_shutdown_budget(plan, tasks)
        container.files = list(container.files or []) + [
            ContainerFile(
                path="/gpustack/supervisor.py",
                mode=0o444,
                content=Path(__file__).with_name("supervisor.py").read_text(),
            ),
            ContainerFile(
                path="/gpustack/tasks.json",
                mode=0o444,
                content=json.dumps(
                    {
                        "tasks": tasks,
                        "shutdown_timeout": shutdown_timeout,
                        "cleanup_timeout": 5,
                    }
                ),
            ),
        ]
        container.execution.command, container.execution.args = (
            self._override_entrypoint(
                ["python3", "/gpustack/supervisor.py", "/gpustack/tasks.json"],
                [],
                container.execution.command_script,
            )
        )

    @staticmethod
    def _external_command(
        command: Optional[List[str]],
        arguments: List[str],
        *,
        keep_api_server_count=False,
    ) -> List[str]:
        """Keep non-managed options from a direct vllm serve command."""
        argv = normalize_vllm_parameters(arguments, tokenized=True)
        if not (len(argv) > 1 and Path(argv[0]).name == "vllm" and argv[1] == "serve"):
            argv = normalize_vllm_parameters(list(command or []) + argv, tokenized=True)
        if (
            len(argv) < 3
            or Path(argv[0]).name != "vllm"
            or argv[1] != "serve"
            or any(token in {"&&", "||", "|", ";", ">", "<", "&"} for token in argv)
        ):
            raise ValueError(
                "Managed External-LB requires a directly executable vllm serve command."
            )
        if find_last_parameter(
            argv, ["distributed-executor-backend"], tokenized=True
        ) not in (
            None,
            "mp",
            "uni",
        ) or find_last_parameter(
            argv, ["data-parallel-backend"], tokenized=True
        ) not in (
            None,
            "mp",
        ):
            raise ValueError(
                "Managed External-LB requires local multiprocessing executors."
            )
        managed_keys = {
            "--port",
            "--host",
            "--data-parallel-rank",
            "-dpn",
            "--data-parallel-start-rank",
            "-dpr",
            "-dps",
            "--data-parallel-supervisor-port",
            "--data-parallel-size",
            "-dp",
            "--data-parallel-size-local",
            "-dpl",
            "--tensor-parallel-size",
            "-tp",
            "--pipeline-parallel-size",
            "-pp",
            "--data-parallel-address",
            "--data-parallel-rpc-port",
            "--api-server-count",
            "--data-parallel-rank-local",
            "-dprl",
            "--node-rank",
            "--nnodes",
            "--master-addr",
            "--master-port",
        }
        if keep_api_server_count:
            managed_keys.remove("--api-server-count")
        base = []
        index = 0
        while index < len(argv):
            token = argv[index]
            key, sep, _ = token.partition("=")
            if key in ("--device-ids", "-di"):
                raise ValueError(
                    "Device selection is managed by GPUStack; use the deployment GPU selector."
                )
            if key in managed_keys:
                index += 1 if sep else 2
            elif key in (
                "--data-parallel-external-lb",
                "--no-data-parallel-external-lb",
                "-dpe",
                "--headless",
                "--data-parallel-multi-port-external-lb",
                "-dpm",
                "--data-parallel-hybrid-lb",
                "--no-data-parallel-hybrid-lb",
                "-dph",
            ):
                index += 1
                if (
                    not sep
                    and index < len(argv)
                    and argv[index].lower()
                    in (
                        "true",
                        "false",
                        "1",
                        "0",
                        "yes",
                        "no",
                        "on",
                        "off",
                        "t",
                        "f",
                        "y",
                        "n",
                    )
                ):
                    index += 1
            else:
                base.append(token)
                index += 1
        return base

    def _external_rank_tasks(
        self,
        base,
        devices,
        topology,
        visible_env,
        ports,
        rpc_port,
        fragments,
        members,
        container,
    ):
        """Describe rank commands and host placements for the allocated device pool."""
        tasks, placements = [], []
        for fragment in fragments:
            rank, node = fragment.dp_rank, fragment.node_rank
            positions = fragment.positions
            child = base + [
                "--data-parallel-external-lb",
                "--data-parallel-size",
                str(topology.dp),
                "--data-parallel-rank",
                str(rank),
                "--data-parallel-size-local",
                "1",
                "--tensor-parallel-size",
                str(topology.tp),
                "--pipeline-parallel-size",
                str(topology.pp),
                "--data-parallel-address",
                self._model_instance.worker_ip or self._worker.ip,
                "--data-parallel-rpc-port",
                str(rpc_port),
                "--api-server-count",
                "0" if node else "1",
            ]
            if topology.nnodes_within_dp > 1:
                store = (self._model_instance.named_ports or {}).get(
                    "_vllm_external_store"
                )
                if store is None or store.count != topology.dp:
                    raise ValueError(
                        "Cross-Worker ranks require a reserved MP store port band"
                    )
                # Multi-node initialization joins the complete world across DP.
                # Every fragment must rendezvous with global rank zero.
                child += [
                    "--distributed-executor-backend",
                    "mp",
                    "--data-parallel-backend",
                    "mp",
                    "--nnodes",
                    str(topology.dp * topology.nnodes_within_dp),
                    "--node-rank",
                    str(rank * topology.nnodes_within_dp + node),
                    "--master-addr",
                    members[0].worker_ip,
                    "--master-port",
                    str(store.base),
                ]
            if node:
                child.append("--headless")
            else:
                child += ["--host", self._worker.ip, "--port", str(ports[rank])]
            task_env = {}
            if self._model_instance.role in ("prefill", "decode"):
                child, task_env = self._external_pd_rank(
                    child, rank, ports[rank], container
                )
            tasks.append(
                {
                    "name": f"dp{rank}-node{node}",
                    "env": task_env,
                    "argv": [
                        "python3",
                        "-c",
                        _DEVICE_TASK_EXEC,
                        visible_env,
                        str(len(devices)),
                        json.dumps(positions),
                        *child,
                    ],
                }
            )
            placements.append(
                DPRankPlacement(
                    dp_rank=rank,
                    node_rank=node,
                    gpu_indexes=[devices[i].index for i in positions],
                    api_port=ports[rank] if node == 0 else None,
                    ready=False if node == 0 else None,
                )
            )
        return tasks, placements

    def _external_pd_rank(self, argv, rank, port, container):
        """Render one logical engine's connector state after final argv composition."""
        return self._managed_pd_endpoint(argv, port, container, dp_rank=rank)

    def _managed_pd_endpoint(self, argv, port, container, *, dp_rank=None):
        """Validate the connector endpoint shared by managed PD launch modes."""
        injection = self._pd_injection(dp_rank=dp_rank, port=port)
        arguments = list(argv)
        positions = [
            index
            for index, token in enumerate(arguments)
            if token.split("=", 1)[0].replace("_", "-") == "--kv-transfer-config"
        ]
        if not positions:
            arguments += injection.args if injection else []
            positions = [
                index
                for index, token in enumerate(arguments)
                if token.split("=", 1)[0].replace("_", "-") == "--kv-transfer-config"
            ]
        if not positions:
            raise ValueError("Managed PD requires a KV transfer descriptor")
        index = positions[-1]
        equal = "=" in arguments[index]
        raw = arguments[index].split("=", 1)[1] if equal else arguments[index + 1]
        descriptor = json.loads(raw)
        if dp_rank is not None:
            self._pd_engine_identity(
                descriptor, managed_pd_engine_id(self._model_instance, dp_rank)
            )
        mode_name = str(
            getattr(getattr(self._model, "disaggregation", None), "mode", "")
        )
        if mode_name == "vllm-ascend-mooncake":
            from gpustack.server.pd_mode_catalog import get_pd_mode

            band = (self._model_instance.named_ports or {}).get("kv_port")
            if band is None:
                raise ValueError("Managed PD requires a reserved connector KV port")
            connectors = set(get_pd_mode(mode_name).connector_variants.values())
            self._pd_reserved_kv_port(descriptor, band.base, connectors)
        encoded = json.dumps(descriptor, separators=(",", ":"))
        if equal:
            arguments[index] = "--kv-transfer-config=" + encoded
        else:
            arguments[index + 1] = encoded
        variables = self._template_variables(port=port)
        if dp_rank is not None:
            variables["dp_rank"] = dp_rank
        env = dict(injection.env) if injection else {}
        supplied = render_values(self._model.env or {}, variables) or {}
        for name in (
            "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "VLLM_NIXL_SIDE_CHANNEL_PORT",
            "VLLM_MOONCAKE_BOOTSTRAP_PORT",
            "HCCL_IF_IP",
        ):
            if name in env and name in supplied and supplied[name] != env[name]:
                raise ValueError(
                    f"{name} conflicts with the reserved connector endpoint managed by GPUStack"
                )
        env.update(supplied)
        for path, content in (injection.files if injection else {}).items():
            rank_path = (
                f"/gpustack/pd/dp{dp_rank}/{path.lstrip('/')}"
                if dp_rank is not None
                else path
            )
            arguments = [value.replace(path, rank_path) for value in arguments]
            env = {key: value.replace(path, rank_path) for key, value in env.items()}
            container.files = list(container.files or []) + [
                ContainerFile(path=rank_path, content=content, mode=0o444)
            ]
        return arguments, env

    @staticmethod
    def _pd_reserved_kv_port(descriptor, port, connectors):
        if descriptor.get("kv_connector") in connectors:
            if descriptor.get("kv_port", port) != port:
                raise ValueError(
                    "KV port conflicts with the reserved connector endpoint managed by GPUStack"
                )
            descriptor["kv_port"] = port
        for child in descriptor.get("kv_connector_extra_config", {}).get(
            "connectors", []
        ):
            if isinstance(child, dict):
                VLLMServer._pd_reserved_kv_port(child, port, connectors)

    @staticmethod
    def _pd_engine_identity(descriptor, engine_id):
        """Keep composed children on their logical engine's top-level identity."""
        if not isinstance(descriptor, dict):
            raise ValueError("Managed PD KV transfer configuration must be an object")
        descriptor["engine_id"] = engine_id
        for child in descriptor.get("kv_connector_extra_config", {}).get(
            "connectors", []
        ):
            if isinstance(child, dict):
                VLLMServer._pd_engine_identity(child, engine_id)
                child.pop("engine_id", None)

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
        self._set_cache_env(env, "VLLM_CACHE_ROOT", "vllm")

        # Apply LMCache environment variables if extended KV cache is enabled
        self._set_lmcache_env(env)

        # Apply distributed environment variables
        if is_distributed:
            self._set_distributed_env(env, deployment_metadata)

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

        if extended_kv_cache.is_shared():
            # Shared mode: connector env resolved server-side is snapshotted
            # onto the instance; local CPU cache sizing does not apply.
            cache_config = self._model_instance.cache_config
            if cache_config and cache_config.injected:
                env.update(self._render_cache_injection_env(cache_config.env))
            else:
                reason = (
                    f" Reason: {cache_config.reason}"
                    if cache_config and cache_config.reason
                    else ""
                )
                logger.warning(f"Starting without shared KV cache.{reason}")
            return

        vendor, _, _ = self._get_device_info()
        if vendor == manufacturer_to_backend(ManufacturerEnum.ASCEND):
            # Ascend runs the connector vllm-ascend ships rather than LMCache,
            # and takes its sizing from --kv-transfer-config; LMCache reads
            # none of these variables there.
            return

        if extended_kv_cache.chunk_size and extended_kv_cache.chunk_size > 0:
            env["LMCACHE_CHUNK_SIZE"] = str(extended_kv_cache.chunk_size)

        ram_bytes = self._resolve_local_kv_cache_bytes()
        if ram_bytes:
            gib = byte_to_gib(ram_bytes)
            env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(
                int(gib) if gib.is_integer() else gib
            )

    def _resolve_local_kv_cache_bytes(self) -> Optional[int]:
        """Host RAM the local KV cache may occupy on this worker, in bytes.

        The explicitly configured size wins; otherwise the ratio applies to
        the instance's VRAM claim on this worker. ``None`` when the deployment
        sets neither. Mirrors what the scheduler reserved for the instance
        here (``get_computed_ram_claim``), which is a per-worker figure.
        """
        extended_kv_cache = self._model.extended_kv_cache
        if not extended_kv_cache:
            return None

        if extended_kv_cache.ram_size and extended_kv_cache.ram_size > 0:
            return extended_kv_cache.ram_size * 1024**3

        if extended_kv_cache.ram_ratio and extended_kv_cache.ram_ratio > 0:
            return int(self._get_total_vram_claim() * extended_kv_cache.ram_ratio)

        return None

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
            self._model.backend_parameters,
            self._model.backend_version,
            self._model.image_name,
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
            # Only dp_only needs a reserved start port for vLLM's TCPStore
            # (#5657); with nnodes > 1 that port comes from --master-port, and a
            # shared start port makes cross-node WorkerProcs racing for their
            # XPUB socket collide (#6019).
            if shape == "dp_only" and len(ports) > 3:
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

        for vram_claim in (computed_resource_claim.vram or {}).values():
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
                self._model.image_name,
                legacy_launch=get_model_instance_workloads(self._model_instance)
                is None,
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
        cache_args = self._build_extended_kv_cache_arguments(ctx)
        arguments.extend(cache_args)
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

        # Compose after cache injection and user parameters have contributed
        # their descriptors, so the engine receives one --kv-transfer-config.
        # Both PD roles put the PD connector first and the cache second. Passing
        # the cache descriptor identifies its origin independently of flag order.
        arguments = compose_kv_transfer_config(
            arguments,
            getattr(self._model_instance, "role", None),
            cache_first=descriptor_in(cache_args),
            top_level_only=_top_level_only_keys(self._model),
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
            self._model.backend_parameters,
            self._model.backend_version,
            self._model.image_name,
        )
        topology: Optional[MultinodeTopology] = None
        if (
            is_distributed
            and (
                get_model_instance_workloads(self._model_instance) is None
                or vllm_lb_mode(self._model.backend_parameters) != "multi_port_external"
            )
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
        if (
            (
                get_model_instance_workloads(self._model_instance) is not None
                and vllm_lb_mode(self._model.backend_parameters)
                == "multi_port_external"
            )
            or not ctx.is_distributed
            or ctx.executor_backend != "ray"
        ):
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
        MP multi-node (non-Ray) path. Dispatches to one of three parameter
        shapes based on the resolved topology:

        - ``dp_only``  → ``--data-parallel-*`` only; vLLM treats every node as
          a DP engine head (no PP/TP spans nodes). Only followers carry
          ``--data-parallel-start-rank``; on the leader it would select hybrid
          LB instead of internal LB.
        - ``mp_only``  → ``--nnodes`` + ``--node-rank`` only; a single DP rank
          is spread across all nodes for cross-node TP/PP.
        - ``nested``   → both sets; vLLM derives node role internally via
          ``node_rank % nnodes_within_dp``.

        Followers also carry ``--headless`` to skip the API server. The leader
        always exposes the HTTP API (handled by --host/--port elsewhere).
        Exception: under ``--data-parallel-hybrid-lb`` every DP engine serves
        its own API, so ``--headless`` is not injected.
        """
        if (
            (
                get_model_instance_workloads(self._model_instance) is not None
                and vllm_lb_mode(self._model.backend_parameters)
                == "multi_port_external"
            )
            or not ctx.is_distributed
            or ctx.executor_backend != "mp"
            or ctx.topology is None
        ):
            return []

        topology = ctx.topology
        leader_ip = self._model_instance.worker_ip
        arguments: List[str] = []
        if (
            find_parameter(
                self._model.backend_parameters, ["distributed-executor-backend"]
            )
            is None
        ):
            arguments.extend(["--distributed-executor-backend", "mp"])

        # serve_manager._assign_ports reserves ports[1] for the DP RPC
        # endpoint (ZMQ) and ports[2] for the PyTorch master endpoint
        # (TCPStore). The two channels use different protocols and can't
        # share one port, so each shape consumes the subset it needs.
        dp_rpc_port = str(self._model_instance.ports[1])
        master_port = str(self._model_instance.ports[2])
        if topology.shape == "dp_only":
            # Since vLLM 0.28, --data-parallel-start-rank on the leader reads as
            # an opt-in to hybrid LB, which then rejects the headless followers.
            # The leader's DP rank is implicitly 0, so only followers send it.
            if topology.start_rank > 0:
                extend_args_no_exist(
                    arguments,
                    ("--data-parallel-start-rank", str(topology.start_rank)),
                )
            extend_args_no_exist(
                arguments,
                ("--data-parallel-address", leader_ip),
                ("--data-parallel-rpc-port", dp_rpc_port),
            )
        elif topology.shape == "mp_only":
            extend_args_no_exist(
                arguments,
                ("--nnodes", str(topology.nnodes)),
                ("--node-rank", str(topology.node_rank)),
                ("--master-addr", leader_ip),
                ("--master-port", master_port),
            )
        else:  # nested
            extend_args_no_exist(
                arguments,
                ("--nnodes", str(topology.nnodes)),
                ("--node-rank", str(topology.node_rank)),
                ("--master-addr", leader_ip),
                ("--master-port", master_port),
                ("--data-parallel-address", leader_ip),
                ("--data-parallel-rpc-port", dp_rpc_port),
            )

        # Hybrid-LB requires every DP engine (leader and followers alike) to
        # run its own API server, so skip the headless injection when the user
        # opts into it.
        hybrid_lb = find_bool_parameter(
            normalize_vllm_parameters(self._model.backend_parameters),
            ["data-parallel-hybrid-lb", "dph"],
            negative_param_names=["no-data-parallel-hybrid-lb"],
            last_wins=True,
            tokenized=True,
        )
        if topology.is_follower and not hybrid_lb:
            extend_args_no_exist(arguments, "--headless")
        return arguments

    def _build_extended_kv_cache_arguments(self, ctx: _VLLMArgsContext) -> List[str]:
        extended = self._model.extended_kv_cache
        if not (extended and extended.enabled):
            return []

        if extended.is_shared():
            # Shared mode: accelerator support is the provider catalog's
            # call — the server-side resolver picks the integration by the
            # worker's framework and snapshots the outcome, so the worker
            # applies the snapshot verbatim. Without an injected snapshot
            # the instance starts degraded (no shared cache), so no
            # connector args apply.
            cache_config = self._model_instance.cache_config
            if cache_config and cache_config.injected:
                return list(cache_config.args or [])
            return []

        # Local mode wires an in-process offload connector, and which one
        # depends on the accelerator: LMCache has no CANN build in the runner
        # images, so Ascend goes through the connector vllm-ascend ships.
        vendor, _, arch_family = self._get_device_info()
        if vendor == manufacturer_to_backend(ManufacturerEnum.ASCEND):
            return self._build_ascend_local_kv_cache_arguments(arch_family)

        if vendor not in {
            manufacturer_to_backend(ManufacturerEnum.NVIDIA),
            manufacturer_to_backend(ManufacturerEnum.AMD),
        }:
            logger.warning(
                "Local extended KV cache for vLLM is only supported on NVIDIA, AMD and Ascend devices. Skipping KV cache offload configuration."
            )
            return []

        return [
            "--kv-transfer-config",
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
        ]

    def _build_ascend_local_kv_cache_arguments(
        self, arch_family: Optional[str]
    ) -> List[str]:
        """Wire vllm-ascend's NPU-native CPU offload connector.

        ``SimpleCPUOffloadConnector`` resolves to the NPU worker that
        vllm-ascend registers over the upstream CUDA one, so local mode on
        Ascend offloads KV blocks to host memory the way it does elsewhere —
        the same deployment fields, a different engine-native mechanism, as
        on SGLang with its hierarchical cache.

        The capacity goes in as a per-rank budget. Its sibling
        ``cpu_bytes_to_use`` is deployment-wide and the connector divides that
        by ``world_size``, which counts TP x PP x PCP and leaves data
        parallelism out: under ``--data-parallel-size N`` every DP engine
        would reserve a full share and the host would hold N times the cache
        the scheduler reserved for it. Dividing the worker's own budget by the
        cards it was given keeps the sizing independent of the engine's
        parallelism, and equal to what ``get_computed_ram_claim`` set aside.
        """
        version = self._model.backend_version
        reason = ascend_local_kv_cache_unsupported_reason(arch_family, version)
        if reason:
            if not getattr(
                self._model, "disaggregation", None
            ) or ascend_local_kv_cache_unsupported_reason(arch_family, None):
                logger.warning(f"{reason} Skipping KV cache offload configuration.")
                return []
            from packaging.version import InvalidVersion, Version

            try:
                parsed = Version(version)
            except InvalidVersion:
                parsed = None
            if parsed and not (
                parsed.local or parsed.is_prerelease or self._model.image_name
            ):
                logger.warning(
                    "%s PD engine-version ranges are advisory; attempting requested KV cache offload configuration.",
                    reason,
                )
            else:
                logger.info(
                    "PD KV cache offload compatibility is unverified for the custom "
                    "image or non-release build. Built-in version check: %s "
                    "Attempting requested KV cache offload configuration.",
                    reason,
                )

        extra_config = {}
        ram_bytes = self._resolve_local_kv_cache_bytes()
        if ram_bytes:
            ranks = len(self._get_selected_gpu_devices()) or 1
            extra_config["cpu_bytes_to_use_per_rank"] = ram_bytes // ranks

        extended = self._model.extended_kv_cache
        if extended.chunk_size and extended.chunk_size > 0:
            # The connector carries no chunking of its own: it stores whole KV
            # blocks and reads its block sizes off the engine. Mapping the
            # chunk size onto --block-size would retune the engine's paging
            # rather than the cache, and vllm-ascend overrides it anyway
            # whenever prefix caching is on — the mode this connector requires.
            logger.warning(
                "Cache chunk size does not apply to extended KV cache on Ascend. Ignoring it."
            )

        return [
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "SimpleCPUOffloadConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": extra_config,
                },
                separators=(",", ":"),
            ),
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
    image_name: Optional[str] = None,
    *,
    legacy_launch: bool = False,
) -> List[str]:
    if not legacy_launch and vllm_lb_mode(backend_parameters) == "multi_port_external":
        return []
    if (
        is_distributed
        and deployment_metadata is not None
        and resolve_executor_backend(backend_parameters, backend_version, image_name)
        == "mp"
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
        tp, pp = cal_distributed_parallelism_arguments(model_instance)
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
