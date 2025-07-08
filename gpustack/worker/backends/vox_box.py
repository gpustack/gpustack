import logging
import os
import subprocess
import sys
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.command import get_versioned_command, get_command_path
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class VoxBoxServer(InferenceServer):
    def start(self):
        try:
            command_path = get_command_path("vox-box")
            if self._model.backend_version:
                command_path = os.path.join(
                    self._config.bin_dir,
                    get_versioned_command("vox-box", self._model.backend_version),
                )
            arguments = [
                "start",
                "--model",
                self._model_path,
                "--data-dir",
                self._config.data_dir,
            ]

            built_in_arguments = [
                "--host",
                "0.0.0.0",
                "--port",
                str(self._model_instance.port),
            ]

            if self._model.backend_parameters:
                arguments.extend(self._model.backend_parameters)

            arguments.extend(built_in_arguments)

            env = os.environ.copy()
            env.update(self._model.env or {})

            if self._model_instance.gpu_indexes is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self._model_instance.gpu_indexes[0])
                logger.info(
                    f"Set CUDA_VISIBLE_DEVICES = {self._model_instance.gpu_indexes[0]}"
                )

            logger.info("Starting vox-box server")
            logger.debug(f"Run vox-box with arguments: {' '.join(arguments)}")
            if self._model.env:
                logger.debug(
                    f"Model environment variables: {', '.join(f'{key}={value}' for key, value in self._model.env.items())}"
                )

            result = subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )
            self.exit_with_code(result.returncode)
        except Exception as e:
            error_message = f"Failed to run the vox-box server: {e}"
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
