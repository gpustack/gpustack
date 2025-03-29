import json
import logging
import subprocess
import sys
import time
from typing import Optional
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils import envs
from gpustack.worker.backends.base import InferenceServer
from gpustack.utils.hub import (
    get_hf_text_config,
    get_max_model_len,
    get_pretrained_config,
)

logger = logging.getLogger(__name__)


class AscendMindIEServer(InferenceServer):
    def start(self):

        version = self._model.backend_version
        if not version:
            version = "1.0.0"

        root_path = envs.get_unix_root_path_of_ascend()
        install_path = root_path.joinpath("mindie", version, "mindie-service")

        # Load envs and configuration

        env = self.get_inference_running_env(env=self._model.env, version=version)
        with open(install_path.joinpath("conf", "config.json"), "r") as f:
            config = json.load(f)
        service_config = config["ServiceConfig"]
        backend_config = config["BackendConfig"]
        model_deploy_config = backend_config["ModelDeployConfig"]
        model_config = model_deploy_config["ModelConfig"][0]

        # Mutate envs and configuration

        # - Pin installation path, which helps to locate other resources.
        env["MIES_INSTALL_PATH"] = str(install_path)

        # - Logging configuration
        # -- Ascend MindIE Service
        config["logLevel"] = "Info"
        env["MINDIE_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LOG_TO_FILE"] = "0"
        # -- scend MindIE Runtime
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "2"  # WARN
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
        env["ASCEND_SLOG_PRINT_TO_FILE"] = "0"
        # -- Ascend MindIE ATB
        env["ATB_LOG_LEVEL"] = "WARN"
        env["ATB_LOG_TO_STDOUT"] = "1"
        env["ATB_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Model
        env["ASDOPS_LOG_LEVEL"] = "WARN"
        env["ASDOPS_LOG_TO_STDOUT"] = "1"
        env["ASDOPS_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE OCK
        env["OCK_LOG_LEVEL"] = "WARN"
        env["OCK_LOG_TO_STDOUT"] = "1"
        env["OCK_LOG_TO_FILE"] = "0"

        # - Listening configuration
        service_config["ipAddress"] = "0.0.0.0"
        service_config["port"] = self._model_instance.port
        service_config["httpsEnabled"] = False

        # - Device configuration
        env.pop("ASCEND_RT_VISIBLE_DEVICES")
        backend_config["npuDeviceIds"] = [self._model_instance.gpu_indexes]
        backend_config["multiNodesInferEnabled"] = False
        backend_config["multiNodesInferEnabled"] = False
        model_config["worldSize"] = len(self._model_instance.gpu_indexes)

        # - Model configuration
        model_deploy_config["truncation"] = True
        if max_model_len := self._get_model_max_seq_len():
            model_deploy_config["maxSeqLen"] = max_model_len
            model_deploy_config["maxInputTokenLen"] = max_model_len
        model_config["modelName"] = self._model.name
        model_config["modelWeightPath"] = self._model_path
        model_config["trustRemoteCode"] = True

        # Generate configuration file

        config_path = install_path.joinpath("conf", f"config-{int(time.time())}.json")
        with open(config_path, "w") as f:
            json.dump(config, f.name)

        # Run
        env["MIES_CONFIG_JSON_PATH"] = str(config_path)
        ascend_mindie_service_bin_path = root_path.joinpath(
            "mindie", version, "mindie-service", "bin", "mindieservice_daemon"
        )

        try:
            logger.info("Starting Ascend MindIE")
            logger.debug(f"Run Ascend MindIE: {ascend_mindie_service_bin_path}")
            if self._model.env:
                logger.debug(
                    f"Model environment variables: {', '.join(f'{key}={value}' for key, value in self._model.env.items())}"
                )

            proc = subprocess.Popen(
                [ascend_mindie_service_bin_path],
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
            except Exception as ue:
                logger.error(f"Failed to update model instance state: {ue}")

            raise e

        finally:
            # Remove configuration file
            config_path.unlink(missing_ok=True)

    def _get_model_max_seq_len(self) -> Optional[int]:
        """Get the maximum sequence length of the model."""
        try:
            pretrained_config = get_pretrained_config(self._model)
            pretrained_or_hf_text_config = get_hf_text_config(pretrained_config)
            return get_max_model_len(pretrained_or_hf_text_config)
        except Exception as e:
            logger.error(f"Failed to derive max model length: {e}")

        return 8192
