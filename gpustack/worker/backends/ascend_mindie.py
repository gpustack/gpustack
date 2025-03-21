import json
import logging
import subprocess
import sys
import os
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

        # Select root path
        root_path = next(
            (
                rp
                for rp in envs.get_unix_available_root_paths_of_ascend()
                if rp.joinpath("mindie", version).is_dir()
            ),
            None,
        )
        install_path = root_path.joinpath("mindie", version, "mindie-service")

        # Load envs and configuration
        logger.info("Loading Ascend MindIE config")
        env = self.get_inference_running_env(version=version)
        with open(
            install_path.joinpath("conf", "config.json"), "r", encoding="utf-8"
        ) as f:
            config = json.load(f)
        server_config = config["ServerConfig"]
        backend_config = config["BackendConfig"]
        model_deploy_config = backend_config["ModelDeployConfig"]
        model_config = model_deploy_config["ModelConfig"][0]
        schedule_config = backend_config["ScheduleConfig"]

        # Mutate envs and configuration
        logger.info("Mutating Ascend MindIE config")

        # - Pin installation path, which helps to locate other resources.
        env["MIES_INSTALL_PATH"] = str(install_path)

        # - Logging configuration
        # -- Ascend MindIE Service
        config["logLevel"] = "Info"
        env["MINDIE_LOG_LEVEL"] = "INFO"
        env["MINDIE_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LOG_TO_FILE"] = "0"
        env["MINDIE_LLM_LOG_LEVEL"] = "WARN"
        env["MINDIE_LLM_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LLM_LOG_TO_FILE"] = "0"
        env["MINDIE_LLM_PYTHON_LOG_LEVEL"] = "WARN"
        env["MINDIE_LLM_PYTHON_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LLM_PYTHON_LOG_TO_FILE"] = "0"
        # -- scend MindIE Runtime
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
        env["ASCEND_SLOG_PRINT_TO_FILE"] = "0"
        # -- Ascend MindIE ATB
        env["ATB_LOG_LEVEL"] = "ERROR"
        env["ATB_LOG_TO_STDOUT"] = "1"
        env["ATB_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Model
        env["ASDOPS_LOG_LEVEL"] = "ERROR"
        env["ASDOPS_LOG_TO_STDOUT"] = "1"
        env["ASDOPS_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE OCK
        env["OCK_LOG_LEVEL"] = "ERROR"
        env["OCK_LOG_TO_STDOUT"] = "1"
        env["OCK_LOG_TO_FILE"] = "0"

        # - Listening configuration
        server_config["ipAddress"] = "0.0.0.0"
        server_config["allowAllZeroIpListening"] = True
        server_config["port"] = self._model_instance.port
        server_config["httpsEnabled"] = False

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
            schedule_config["maxPrefillTokens"] = max_model_len
        model_config["modelName"] = self._model.name
        model_config["modelWeightPath"] = self._model_path
        model_config["trustRemoteCode"] = True

        # Generate configuration file by model instance id
        config_path = install_path.joinpath(
            "conf", f"config-{self._model_instance.id}.json"
        )
        logger.info(f"Writing Ascend MindIE config to {config_path}")
        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        # Run
        env["MIES_CONFIG_JSON_PATH"] = str(config_path)
        ascend_mindie_service_path = root_path.joinpath(
            "mindie", version, "mindie-service"
        )
        ascend_mindie_service_bin_path = ascend_mindie_service_path.joinpath(
            "bin", "mindieservice_daemon"
        )

        try:
            debugging = logger.isEnabledFor(logging.DEBUG)

            if debugging:
                envs_view = env
            else:
                envs_view = self._model.env
            logger.info(
                f"Starting Ascend MindIE: {ascend_mindie_service_bin_path}, "
                f"with envs: {os.linesep}{os.linesep.join(f'  {key}={value}' for key, value in sorted(envs_view.items()))}"
            )

            # Fork
            proc = subprocess.Popen(
                [str(ascend_mindie_service_bin_path)],
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
                cwd=ascend_mindie_service_path,
            )
            exit_code = proc.wait()

            # Remove configuration file
            if not debugging:
                config_path.unlink(missing_ok=True)

            self.exit_with_code(exit_code)

        except Exception as e:
            error_message = f"Failed to run Ascend MindIE: {e}"
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

    def _get_model_max_seq_len(self) -> Optional[int]:
        """Get the maximum sequence length of the model."""
        try:
            pretrained_config = get_pretrained_config(self._model)
            pretrained_or_hf_text_config = get_hf_text_config(pretrained_config)
            return get_max_model_len(pretrained_or_hf_text_config)
        except Exception as e:
            logger.error(f"Failed to get model max seq length: {e}")

        return 8192
