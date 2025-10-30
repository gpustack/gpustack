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

from gpustack.schemas.models import ModelInstance
from gpustack.utils.command import find_parameter
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

    _workload_name: Optional[str] = None

    def start(self):  # noqa: C901
        try:
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(f"Starting vLLM model instance: {self._model_instance.name}")

        # Prepare distributed information.
        is_distributed, is_distributed_leader, is_distributed_follower = (
            self._get_distributed_metadata()
        )

        env = self._get_configured_env(
            is_distributed=is_distributed,
        )

        command_args = self._build_command_args(
            port=self._get_serving_port(),
            is_distributed=is_distributed,
        )

        self._create_workload(
            command_args=command_args,
            env=env,
            is_distributed_leader=is_distributed_leader,
            is_distributed_follower=is_distributed_follower,
        )

    def _create_workload(
        self,
        command_args: List[str],
        env: Dict[str, str],
        is_distributed_leader: bool,
        is_distributed_follower: bool,
    ):
        # Store workload name for management operations
        self._workload_name = self._model_instance.name

        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get vLLM backend image")

        resources = self._get_configured_resources()

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        run_container = Container(
            image=image,
            name=self._model_instance.name,
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=True,
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

        # Adjust run container for distributed follower.
        if is_distributed_follower:
            ray_command_args, ray_ports = self._build_ray_configuration(
                is_leader=False,
            )

            run_container.execution.args = ray_command_args
            run_container.ports = ray_ports

        # Create sidecar container for distributed leader.
        sidecar_container = None
        if is_distributed_leader:
            run_container.mounts.append(
                ContainerMount(
                    path="/tmp",
                    volume="tmp-volume",
                ),
            )

            ray_command_args, ray_ports = self._build_ray_configuration(
                is_leader=True,
            )

            sidecar_container = Container(
                image=image,
                name=f"{self._model_instance.name}-ray",
                profile=ContainerProfileEnum.RUN,
                restart_policy=ContainerRestartPolicyEnum.NEVER,
                execution=ContainerExecution(
                    privileged=True,
                    args=ray_command_args,
                ),
                envs=run_container.envs,
                resources=run_container.resources,
                mounts=run_container.mounts,
                ports=ray_ports,
            )

        logger.info(f"Creating vLLM container workload: {self._workload_name}")
        logger.info(
            f"With image: {image}, "
            f"arguments: [{' '.join(command_args)}], "
            f"ports: [{','.join([str(port.internal) for port in ports])}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=self._workload_name,
            host_network=True,
            shm_size=10 * 1 << 30,  # 10 GiB
            containers=(
                [run_container]
                if not sidecar_container
                else [run_container, sidecar_container]
            ),
        )
        create_workload(workload_plan)

        logger.info(f"Created vLLM container workload {self._workload_name}")

    def _get_configured_env(self, is_distributed: bool) -> Dict[str, str]:
        """
        Setup environment variables for the vLLM container server.
        """

        # Apply GPUStack's inference environment setup
        env = super()._get_configured_env()

        # Apply LMCache environment variables if extended KV cache is enabled
        self._set_lmcache_env(env)

        # Apply distributed environment variables
        if is_distributed:
            self._set_distributed_env(env)

        return env

    def _set_lmcache_env(self, env: Dict[str, str]):
        """
        Set up LMCache environment variables if extended KV cache is enabled.
        """
        if not (
            self._model.extended_kv_cache and self._model.extended_kv_cache.enabled
        ):
            return

        if (
            self._model.extended_kv_cache.chunk_size
            and self._model.extended_kv_cache.chunk_size > 0
        ):
            env["LMCACHE_CHUNK_SIZE"] = str(self._model.extended_kv_cache.chunk_size)

        if (
            self._model.extended_kv_cache.ram_size
            and self._model.extended_kv_cache.ram_size > 0
        ):
            # Explicitly specified RAM size for KV cache
            env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(
                self._model.extended_kv_cache.ram_size
            )
        elif (
            self._model.extended_kv_cache.ram_ratio
            and self._model.extended_kv_cache.ram_ratio > 0
        ):
            # Calculate RAM size based on ratio of total VRAM claim
            vram_claim = self._get_total_vram_claim()
            env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(byte_to_gib(vram_claim))

    def _set_distributed_env(self, env: Dict[str, str]):
        """
        Set up environment variables for distributed execution.
        """
        if is_ascend(self._get_selected_gpu_devices()):
            # See https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi-node_dsv3.2.html.
            if "HCCL_SOCKET_IFNAME" not in env:
                env["HCCL_IF_IP"] = self._worker.ip
                env["HCCL_SOCKET_IFNAME"] = self._worker.ifname
                env["GLOO_SOCKET_IFNAME"] = self._worker.ifname
                env["TP_SOCKET_IFNAME"] = self._worker.ifname
            return

        if "NCCL_SOCKET_IFNAME" not in env:
            env["NCCL_SOCKET_IFNAME"] = self._worker.ifname
            env["GLOO_SOCKET_IFNAME"] = self._worker.ifname

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

        derived_max_model_len = self._derive_max_model_len()
        if derived_max_model_len and derived_max_model_len > 8192:
            arguments.extend(["--max-model-len", "8192"])

        auto_parallelism_arguments = get_auto_parallelism_arguments(
            self._model.backend_parameters,
            self._model_instance,
            is_distributed,
        )
        arguments.extend(auto_parallelism_arguments)

        if is_distributed:
            arguments.extend(["--distributed-executor-backend", "ray"])

        if self._model.extended_kv_cache and self._model.extended_kv_cache.enabled:
            arguments.extend(
                [
                    "--kv-transfer-config",
                    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
                ]
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
        if self._model.backend_parameters:
            arguments.extend(self._model.backend_parameters)

        # Append immutable arguments to ensure proper operation for accessing
        immutable_arguments = [
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--served-model-name",
            self._model_instance.model_name,
        ]
        arguments.extend(immutable_arguments)

        return arguments

    def _build_ray_configuration(
        self,
        is_leader: bool,
    ) -> (List[str], Optional[List[ContainerPort]]):
        gcs_server_port = 6379
        client_port = 10001
        dashboard_port = 8265

        if is_leader:
            arguments = [
                "ray",
                "start",
                "--block",
                "--head",
                f"--port={gcs_server_port}",
                f"--ray-client-server-port={client_port}",
                f"--dashboard-host={self._worker.ip}",
                f"--dashboard-port={dashboard_port}",
                f"--node-ip-address={self._worker.ip}",
                "--disable-usage-stats",
                "--verbose",
            ]
            ports = [
                ContainerPort(
                    internal=port,
                )
                for port in [gcs_server_port, client_port, dashboard_port]
            ]

            return arguments, ports

        arguments = [
            "ray",
            "start",
            "--block",
            f"--address={self._model_instance.worker_ip}:{gcs_server_port}",
            f"--node-ip-address={self._worker.ip}",
            "--disable-usage-stats",
            "--verbose",
        ]

        return arguments, None


def get_auto_parallelism_arguments(
    backend_parameters: List[str],
    model_instance: ModelInstance,
    is_distributed: bool,
) -> List[str]:
    parallelism = find_parameter(
        backend_parameters,
        ["tensor-parallel-size", "tp", "pipeline-parallel-size", "pp"],
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
