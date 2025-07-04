import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional
from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum
from gpustack.utils.command import (
    find_parameter,
    get_versioned_command,
    get_command_path,
)
from gpustack.utils.hub import (
    get_hf_text_config,
    get_max_model_len,
    get_pretrained_config,
)
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class VLLMServer(InferenceServer):
    def start(self):
        try:
            command_path = get_command_path("vllm")
            if self._model.backend_version:
                command_path = os.path.join(
                    self._config.bin_dir,
                    get_versioned_command("vllm", self._model.backend_version),
                )
            arguments = [
                "serve",
                self._model_path,
            ]

            derived_max_model_len = self._derive_max_model_len()
            if derived_max_model_len and derived_max_model_len > 8192:
                arguments.extend(["--max-model-len", "8192"])

            auto_parallism_arguments = get_auto_parallelism_arguments(
                self._model.backend_parameters, self._model_instance
            )
            arguments.extend(auto_parallism_arguments)

            if is_distributed_vllm(self._model_instance):
                arguments.extend(["--distributed-executor-backend", "ray"])

            if self._model.backend_parameters:
                arguments.extend(self._model.backend_parameters)

            built_in_arguments = [
                "--host",
                "0.0.0.0",
                "--port",
                str(self._model_instance.port),
                "--served-model-name",
                self._model_instance.model_name,
            ]

            # Extend the built-in arguments at the end so that
            # they cannot be overridden by the user-defined arguments
            arguments.extend(built_in_arguments)

            logger.info("Starting vllm server")
            logger.debug(f"Run vllm with arguments: {' '.join(arguments)}")
            if self._model.env:
                logger.debug(
                    f"Model environment variables: {', '.join(f'{key}={value}' for key, value in self._model.env.items())}"
                )
            env = os.environ.copy()
            self.set_vllm_distributed_env(env)
            env = self.get_inference_running_env(env)
            result = subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )
            self.exit_with_code(result.returncode)
        except Exception as e:
            error_message = f"Failed to run the vllm server: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(self._model_instance.id, **patch_dict)
            except Exception as ue:
                logger.error(f"Failed to update model instance: {ue}")
            sys.exit(1)

    def set_vllm_distributed_env(self, env: Dict[str, str]):
        model_instance = self._model_instance
        worker = self._worker
        subordinate_workers = model_instance.distributed_servers.subordinate_workers
        if not subordinate_workers:
            return

        device_str = "GPU"

        ray_placement_group_bundles: List[Dict[str, float]] = []
        bundle_indexes = []
        bundle_index_offset = 0
        # we add all gpus of involved workers to the bundle, then pick gpus by index
        for i in range(len(worker.status.gpu_devices)):
            bundle = {
                device_str: 1,
                f"node:{worker.ip}": 0.001,
            }
            ray_placement_group_bundles.append(bundle)
        for gpu_index in model_instance.gpu_indexes:
            bundle_indexes.append(gpu_index)
        bundle_index_offset += len(worker.status.gpu_devices)

        for subordinate_worker in subordinate_workers:
            for i in range(subordinate_worker.total_gpus):
                bundle = {
                    device_str: 1,
                    f"node:{subordinate_worker.worker_ip}": 0.001,
                }
                ray_placement_group_bundles.append(bundle)
            for gpu_index in subordinate_worker.gpu_indexes:
                bundle_indexes.append(bundle_index_offset + gpu_index)
            bundle_index_offset += subordinate_worker.total_gpus

        # encoded to json and set in GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES env
        env["GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES"] = json.dumps(
            ray_placement_group_bundles
        )
        # helps to pick specific gpus
        env["VLLM_RAY_BUNDLE_INDICES"] = ", ".join([str(x) for x in bundle_indexes])

        logger.debug(
            f"Set GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES: {env['GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES']}. "
            f"Set VLLM_RAY_BUNDLE_INDICES: {env['VLLM_RAY_BUNDLE_INDICES']}"
        )

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
