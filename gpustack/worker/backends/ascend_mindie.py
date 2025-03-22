import logging
import subprocess
import sys
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class AscendMindIEServer(InferenceServer):
    def start(self):

        command_path = ""
        arguments = []

        try:
            logger.info("Starting Ascend MindIE")
            logger.debug(
                f"Run Ascend MindIE: {command_path} with arguments: {' '.join(arguments)}"
            )
            if self._model.env:
                logger.debug(
                    f"Model environment variables: {', '.join(f'{key}={value}' for key, value in self._model.env.items())}"
                )

            env = self.get_inference_running_env(version=self._model.backend_version)
            proc = subprocess.Popen(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )

            exit_code = proc.wait()
            self.exit_with_code(exit_code)

        except Exception as e:
            error_message = f"Failed to run the Ascend MindIE: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(self._model_instance.id, **patch_dict)
            except Exception as e:
                logger.error(f"Failed to update model instance state: {e}")
            sys.exit(1)
