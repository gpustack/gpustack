import logging
import subprocess
import sys
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class VoxBoxServer(InferenceServer):
    def start(self):
        try:
            command_path = "vox-box"
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

            if self._model_instance.gpu_indexes is not None:
                arguments.extend(
                    [
                        "--device",
                        f"cuda:{self._model_instance.gpu_indexes[0]}",
                    ]
                )

            worker_gpu_devices = None
            worker = self._clientset.workers.get(self._model_instance.worker_id)
            if worker and worker.status.gpu_devices:
                worker_gpu_devices = worker.status.gpu_devices

            arguments.extend(built_in_arguments)

            env = self.get_inference_running_env(
                self._model_instance.gpu_indexes, worker_gpu_devices
            )

            logger.info("Starting vox-box server")
            logger.debug(f"Run vox-box with arguments: {' '.join(arguments)}")
            subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )
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
