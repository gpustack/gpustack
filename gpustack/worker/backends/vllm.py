import logging
import os
import subprocess
import sys
import sysconfig
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.command import find_parameter
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class VLLMServer(InferenceServer):
    def start(self):
        command_path = os.path.join(sysconfig.get_path("scripts"), "vllm")
        arguments = [
            "serve",
            self._model_path,
        ]

        if self._model.backend_parameters:
            arguments.extend(self._model.backend_parameters)

        built_in_arguments = [
            "--host",
            "0.0.0.0",
            "--port",
            str(self._model_instance.port),
            "--served-model-name",
            self._model_instance.model_name,
            "--trust-remote-code",
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
                ["--tensor-parallel-size", str(len(self._model_instance.gpu_indexes))]
            )

        # Extend the built-in arguments at the end so that
        # they cannot be overridden by the user-defined arguments
        arguments.extend(built_in_arguments)

        env = self.get_inference_running_env(self._model_instance.gpu_indexes)
        try:
            logger.info("Starting vllm server")
            logger.debug(f"Run vllm with arguments: {' '.join(arguments)}")
            subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )
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
