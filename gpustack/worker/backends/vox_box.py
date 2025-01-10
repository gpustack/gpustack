import logging
import os
import subprocess
import sys
import sysconfig
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.command import get_versioned_command
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class VoxBoxServer(InferenceServer):
    def start(self):
        try:
            command_path = os.path.join(sysconfig.get_path("scripts"), "vox-box")
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

            if self._model_instance.gpu_indexes is not None:
                arguments.extend(
                    [
                        "--device",
                        f"cuda:{self._model_instance.gpu_indexes[0]}",
                    ]
                )

            arguments.extend(built_in_arguments)

            env = os.environ.copy()

            logger.info("Starting vox-box server")
            logger.debug(f"Run vox-box with arguments: {' '.join(arguments)}")
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
