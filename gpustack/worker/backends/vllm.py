import logging
import os
import subprocess
import sys
import sysconfig
from typing import Optional
from gpustack.schemas.models import ModelInstanceStateEnum, SourceEnum
from gpustack.utils.command import find_parameter
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class VLLMServer(InferenceServer):
    def start(self):
        try:
            command_path = os.path.join(sysconfig.get_path("scripts"), "vllm")
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
                self._model_instance.gpu_indexes, worker_gpu_devices
            )
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

    def _derive_max_model_len(self) -> Optional[int]:
        """
        Derive max model length from model config.
        Returns None if unavailable.
        """
        trust_remote_code = False
        if (
            self._model.backend_parameters
            and "--trust-remote-code" in self._model.backend_parameters
        ):
            trust_remote_code = True

        pretrained_config = None
        if self._model.source == SourceEnum.HUGGING_FACE:
            from transformers import AutoConfig

            pretrained_config = AutoConfig.from_pretrained(
                self._model.huggingface_repo_id,
                token=self._config.huggingface_token,
                trust_remote_code=trust_remote_code,
            )
        elif self._model.source == SourceEnum.MODEL_SCOPE:
            from modelscope import AutoConfig

            pretrained_config = AutoConfig.from_pretrained(
                self._model.model_scope_model_id,
                trust_remote_code=trust_remote_code,
            )
        else:
            # Should not reach here.
            raise ValueError(f"Unsupported model source: {self._model.source}")

        if pretrained_config:
            return get_max_model_len(pretrained_config)

        return None


# Simplified from vllm.config._get_and_verify_max_len
# Keep in our codebase to avoid dependency on vllm's internal
# APIs which may change unexpectedly.
# https://github.com/vllm-project/vllm/blob/v0.6.2/vllm/config.py#L1668
def get_max_model_len(hf_config) -> int:  # noqa: C901
    """Get the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.",
            possible_keys,
            default_max_len,
        )
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        if "type" in rope_scaling:
            rope_type = rope_scaling["type"]
        elif "rope_type" in rope_scaling:
            rope_type = rope_scaling["rope_type"]
        else:
            raise ValueError("rope_scaling must have a 'type' or 'rope_type' key.")

        # The correct one should be "longrope", kept "su" here
        # to be backward compatible
        if rope_type not in ("su", "longrope", "llama3"):
            scaling_factor = 1
            if "factor" in rope_scaling:
                scaling_factor = rope_scaling["factor"]
            if rope_type == "yarn":
                derived_max_model_len = rope_scaling["original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    logger.debug(f"Derived max model length: {derived_max_model_len}")
    return int(derived_max_model_len)
