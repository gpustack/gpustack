import logging
import os
import subprocess
import sys
import sysconfig
from typing import Optional
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.command import find_parameter, get_versioned_command
from gpustack.utils.hub import get_max_model_len, get_pretrained_config
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class VLLMServer(InferenceServer):
    def start(self):
        try:
            command_path = os.path.join(sysconfig.get_path("scripts"), "vllm")
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

            parallelism = find_parameter(
                self._model.backend_parameters,
                ["tensor-parallel-size", "tp", "pipeline-parallel-size", "pp"],
            )

            if (
                self._model_instance.gpu_indexes is not None
                and len(self._model_instance.gpu_indexes) > 1
                and parallelism is None
            ):
                built_in_arguments.extend(
                    [
                        "--tensor-parallel-size",
                        str(len(self._model_instance.gpu_indexes)),
                    ]
                )

            worker_gpu_devices = None
            worker = self._clientset.workers.get(self._model_instance.worker_id)
            if worker and worker.status.gpu_devices:
                worker_gpu_devices = worker.status.gpu_devices

            # Extend the built-in arguments at the end so that
            # they cannot be overridden by the user-defined arguments
            arguments.extend(built_in_arguments)

            env = self.get_inference_running_env(
                self._model_instance.gpu_indexes, worker_gpu_devices, "vllm"
            )
            logger.info("Starting vllm server")
            logger.debug(f"Run vllm with arguments: {' '.join(arguments)}")
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

    def _derive_max_model_len(self) -> Optional[int]:
        """
        Derive max model length from model config.
        Returns None if unavailable.
        """
        try:
            pretrained_config = get_pretrained_config(self._model)
            return get_max_model_len(pretrained_config)
        except Exception as e:
            logger.error(f"Failed to derive max model length: {e}")

        return None
