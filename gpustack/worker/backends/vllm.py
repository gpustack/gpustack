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

from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum
from gpustack.utils.command import find_parameter
from gpustack.utils.envs import sanitize_env
from gpustack.utils.hub import (
    get_hf_text_config,
    get_max_model_len,
    get_pretrained_config,
)
from gpustack.worker.backends.base import InferenceServer, is_ascend_310p

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

        env = self._get_configured_env()

        command_args = self._build_command_args(
            port=self._get_serving_port(),
        )

        self._create_workload(
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        command_args: List[str],
        env: Dict[str, str],
    ):
        # Store workload name for management operations
        self._workload_name = self._model_instance.name

        image_name = self._get_backend_image_name()
        if not image_name:
            raise ValueError("Failed to get vLLM backend image name")

        resources = self._get_configured_resources()

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        run_container = Container(
            image=image_name,
            name=self._model_instance.name,
            profile=ContainerProfileEnum.RUN,
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

        logger.info(f"Creating vLLM container workload: {self._workload_name}")
        logger.info(
            f"With image: {image_name}, "
            f"arguments: [{' '.join(command_args)}], "
            f"ports: [{','.join([str(port.internal) for port in ports])}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=self._workload_name,
            host_network=True,
            containers=[run_container],
        )
        create_workload(workload_plan)

        logger.info(f"Created vLLM container workload {self._workload_name}")

    def _get_configured_env(self) -> Dict[str, str]:
        """
        Setup environment variables for the vLLM container server.
        """

        # Apply GPUStack's inference environment setup
        env = super()._get_configured_env()

        # Apply LMCache environment variables if extended KV cache is enabled
        self._set_lmcache_env(env)

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
            self._model.extended_kv_cache.max_local_cpu_size
            and self._model.extended_kv_cache.max_local_cpu_size > 0
        ):
            env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(
                self._model.extended_kv_cache.max_local_cpu_size
            )
        else:
            env["LMCACHE_LOCAL_CPU"] = str(False).lower()

        if self._model.extended_kv_cache.remote_url:
            env["LMCACHE_REMOTE_URL"] = self._model.extended_kv_cache.remote_url
            # This is the claimed default value from LMCache docs
            # However, an assertion fails in LMCache if not explicitly set
            env["LMCACHE_REMOTE_SERDE"] = "naive"

    def _build_command_args(self, port: int) -> List[str]:
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
            self._model.backend_parameters, self._model_instance
        )
        arguments.extend(auto_parallelism_arguments)

        if is_distributed_vllm(self._model_instance):
            arguments.extend(["--distributed-executor-backend", "ray"])

        if self._model.extended_kv_cache and self._model.extended_kv_cache.enabled:
            arguments.extend(
                [
                    "--kv-transfer-config",
                    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
                ]
            )

        # For Ascend 310P, we need to enforce eager execution and default dtype to float16
        if is_ascend_310p(self._worker):
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

    def _derive_max_model_len(self) -> Optional[int]:
        """
        Derive max model length from model config.
        Returns None if unavailable.
        """
        try:
            pretrained_config = get_pretrained_config(self._model)
            pretrained_or_hf_text_config = get_hf_text_config(pretrained_config)
            return get_max_model_len(pretrained_or_hf_text_config)
        except Exception as e:
            logger.error(f"Failed to derive max model length: {e}")

        return None

    def _handle_error(self, error: Exception):
        """
        Handle errors during server startup.
        """
        cause = getattr(error, "__cause__", None)
        cause_text = f": {cause}" if cause else ""
        error_message = f"Failed to run vLLM: {error}{cause_text}"

        try:
            patch_dict = {
                "state_message": error_message,
                "state": ModelInstanceStateEnum.ERROR,
            }
            self._update_model_instance(self._model_instance.id, **patch_dict)
        except Exception as ue:
            logger.error(f"Failed to update model instance: {ue}")

        raise error


def get_auto_parallelism_arguments(
    backend_parameters: List[str], model_instance: ModelInstance
) -> List[str]:
    parallelism = find_parameter(
        backend_parameters,
        ["tensor-parallel-size", "tp", "pipeline-parallel-size", "pp"],
    )

    if parallelism is not None:
        return []

    if is_distributed_vllm(model_instance):
        # distributed across multiple workers
        pp = len(model_instance.distributed_servers.subordinate_workers) + 1
        tp = len(model_instance.gpu_indexes) if model_instance.gpu_indexes else 1
        uneven_pp = tp
        uneven = False
        for (
            subordinate_worker
        ) in model_instance.distributed_servers.subordinate_workers:
            num_gpus = len(subordinate_worker.gpu_indexes)
            uneven_pp += num_gpus
            if num_gpus != tp:
                uneven = True

        if uneven:
            tp = 1
            pp = uneven_pp
            logger.warning(
                f"The number of GPUs selected for each worker is not equal: {num_gpus} != {tp}, fallback to using pipeline parallelism."
            )
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


def is_distributed_vllm(model_instance: ModelInstance) -> bool:
    return (
        model_instance.distributed_servers
        and model_instance.distributed_servers.subordinate_workers
    )
