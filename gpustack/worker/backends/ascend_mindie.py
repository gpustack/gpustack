import argparse
import dataclasses
import json
import logging
import shutil
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from typing import Optional, List
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils import envs
from gpustack.worker.backends.base import InferenceServer
from gpustack.utils.hub import (
    get_hf_text_config,
    get_max_model_len,
    get_pretrained_config,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AscendMindIEParameters:
    log_level: str = "Info"
    max_link_num: int = 1000
    token_timeout: int = 600
    e2e_timeout: int = 600
    max_seq_len: int = 8192
    max_input_token_len: int = -1
    truncation: bool = False
    cpu_mem_size: int = 5
    npu_memory_fraction: float = 0.9
    trust_remote_code: bool = False
    cache_block_size: int = 128
    max_prefill_batch_size: int = 50
    prefill_time_ms_per_req: int = 150
    prefill_policy_type: int = 0
    max_batch_size: int = 200  # FIXME: Calculate this
    decode_time_ms_per_req: int = 50
    decode_policy_type: int = 0
    max_preempt_count: int = 0
    support_select_batch: bool = False
    max_queue_delay_microseconds: int = 5000
    enable_prefix_caching: bool = False
    metrics: bool = False
    enforce_eager: bool = False
    dtype: str = "auto"
    rope_scaling: Optional[str] = None
    rope_scaling_parsed: Optional[any] = None  # store JSON parsed result
    rope_theta: Optional[float] = None
    override_generation_config: Optional[str] = None
    override_generation_config_parsed: Optional[any] = None  # store JSON parsed result

    def from_args(self, args: List[str]):
        parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False)
        #
        # Log config
        #
        parser.add_argument(
            "--log-level",
            type=str,
            default="Info",
            choices=['Verbose', 'Info', 'Warning', 'Warn', 'Error', 'Debug'],
            help="Log level for MindIE.",
        )
        #
        # Server config
        #
        parser.add_argument(
            "--max-link-num",
            type=int,
            default=self.max_link_num,
            help="Maximum parallel requests",
        )
        parser.add_argument(
            "--token-timeout",
            type=int,
            default=self.token_timeout,
            help="Timeout for a token generation in seconds.",
        )
        parser.add_argument(
            "--e2e-timeout",
            type=int,
            default=self.e2e_timeout,
            help="E2E (from request accepted to inference stopped) timeout in seconds.",
        )
        #
        # Model deploy config
        #
        parser.add_argument(
            "--max-seq-len",
            type=int,
            default=self.max_seq_len,
            help="Model context length. "
            "If unspecified, will be automatically derived from the model config.",
        )
        parser.add_argument(
            "--max-input-token-len",
            type=int,
            default=self.max_input_token_len,
            help="Max input token length. "
            "If unspecified, will be automatically derived from `--max-seq-len`.",
        )
        parser.add_argument(
            "--truncation",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Truncate the input token length, "
            "when the length is larger than the minimum between `--max-input-token-len` and `--max-seq-len` - 1.",
        )
        #
        # Model config
        #
        parser.add_argument(
            "--cpu-mem-size",
            type=int,
            default=self.cpu_mem_size,
            help="CPU swap space size (GiB). "
            f"If unspecified, will use the default value of {self.cpu_mem_size}.",
        )
        parser.add_argument(
            "--npu-memory-fraction",
            type=float,
            default=self.npu_memory_fraction,
            help="The fraction of NPU memory to be used for the model executor, "
            "which can range from 0 to 1 (included). "
            "For example, a value of 0.5 would imply 50% NPU memory utilization. "
            f"If unspecified, will use the default value of {self.npu_memory_fraction}.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action='store_true',
            help="Trust remote code.",
        )
        #
        # Schedule config
        #
        parser.add_argument(
            "--cache-block-size",
            type=int,
            default=self.cache_block_size,
            help="KV cache block size, which must be powers of 2. "
            f"If unspecified, will use the default value of {self.cache_block_size}.",
        )
        parser.add_argument(
            "--max-prefill-batch-size",
            type=int,
            default=self.max_prefill_batch_size,
            help="During prefilling stage, the maximum requests can be batched, "
            "which must be less than `--max-batch-size`.",
        )
        parser.add_argument(
            "--prefill-time-ms-per-req",
            type=int,
            default=self.prefill_time_ms_per_req,
            help="Compare with --decode-time-ms-per-req to select prefilling or decoding, "
            "works with `--support-select-batch`.",
        )
        parser.add_argument(
            "--prefill-policy-type",
            type=int,
            choices=[0, 1, 2, 3],
            default=self.prefill_policy_type,
            help="Strategy of prefilling stage. "
            "0: FCFS, first come first serving, "
            "1: STATE, same as FCFS, "
            "2: PRIORITY, priority queue, "
            "3: MLFQ, multi-levels feedback queue.",
        )
        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=self.max_batch_size,
            help="During decoding stage, the maximum requests can be batched.",
        )
        parser.add_argument(
            "--decode-time-ms-per-req",
            type=int,
            default=self.decode_time_ms_per_req,
            help="Compare with `--prefill-time-ms-per-req` to select prefilling or decoding, "
            "works with `--support-select-batch`.",
        )
        parser.add_argument(
            "--decode-policy-type",
            type=int,
            choices=[0, 1, 2, 3],
            default=self.decode_policy_type,
            help="Strategy of decoding stage. "
            "0: FCFS, first come first serving, "
            "1: STATE, process those requests have been preempted or swapped at first, "
            "2: PRIORITY, priority queue, "
            "3: MLFQ, multi-levels feedback queue.",
        )
        parser.add_argument(
            "--max-preempt-count",
            type=int,
            default=self.max_preempt_count,
            help="Maximum preempt requests during decoding stage, which must be less than `--max-batch-size`.",
        )
        parser.add_argument(
            "--support-select-batch",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Enable batch selecting. "
            "According to `--prefill-time-ms-per-req` and `--decode-time-ms-per-req`, "
            "select the execution priority for this batch.",
        )
        parser.add_argument(
            "--max-queue-delay-microseconds",
            type=int,
            help="Maximum microseconds of queue waiting.",
        )
        #
        # Features
        #
        parser.add_argument(
            "--enable-prefix-caching",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Enable prefix caching. "
            "Use `--no-enable-prefix-caching` to disable explicitly.",
        )
        parser.add_argument(
            "--metrics",
            action='store_true',
            help="Expose metrics in /metrics router.",
        )
        parser.add_argument(
            "--enforce-eager",
            action='store_true',
            help="Emit operators in eager mode.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=self.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations. "
            "- `auto`: use the default data type of the model config, "
            "- `half`: for FP16, "
            "- `float16`: is the same as `half`, "
            "- `bfloat16`: for BF16, "
            "- `float`: is the shorthand for `float32`, "
            "- `float32`: for FP32. ",
        )
        parser.add_argument(
            "--rope-scaling",
            type=str,
            required=False,
            help="RoPE scaling configuration in JSON format. "
            "For example: `{\"type\": \"yarn\", \"factor\" :4.0, \"original_max_position_embeddings\": 32768}`. "
            "This will merge into the `config.json` of the model structure.",
        )
        parser.add_argument(
            "--rope-theta",
            type=float,
            required=False,
            help="RoPE theta configuration. "
            "This will merge into the `config.json` of the model structure.",
        )
        parser.add_argument(
            "--override-generation-config",
            type=str,
            required=False,
            help="Overrides or sets generation config in JSON format. "
            "For example: `{\"temperature\": 0.5}`. "
            "This will merge into the `generation_config.json` of the model structure.",
        )

        args_parsed = parser.parse_known_args(args=args)
        for attr_name in [attr.name for attr in dataclasses.fields(self.__class__)]:
            if attr_name.endswith("_parsed"):
                continue
            if attr_value := getattr(args_parsed[0], attr_name):
                setattr(self, attr_name, attr_value)

        self._default()
        self._validate()

    def _default(self):
        if self.max_input_token_len <= 0:
            self.max_input_token_len = self.max_seq_len

    def _validate(self):  # noqa: max-complexity=14
        if not (1 <= self.max_link_num <= 1000):
            raise argparse.ArgumentTypeError(
                "--max-link-num must be in the range [1, 1000]"
            )
        if not (1 <= self.token_timeout <= 3600):
            raise argparse.ArgumentTypeError(
                "--token-timeout must be in the range [1, 3600]"
            )
        if not (1 <= self.e2e_timeout <= 3600):
            raise argparse.ArgumentTypeError(
                "--e2e-timeout must be in the range [1, 3600]"
            )
        if self.max_seq_len <= 0:
            raise argparse.ArgumentTypeError("--max-seq-len must be greater than 0")
        if self.max_input_token_len > self.max_seq_len:
            raise argparse.ArgumentTypeError(
                "--max-input-token-len must be less or equal than --max-seq-len"
            )
        if not (0 < self.npu_memory_fraction < 1):
            raise argparse.ArgumentTypeError(
                "--npu-memory-fraction must be in the range (0, 1]"
            )
        if self.cache_block_size & (self.cache_block_size - 1) != 0:
            raise argparse.ArgumentTypeError("--cache-block-size must be powers of 2")
        if not (1 <= self.max_batch_size <= 5000):
            raise argparse.ArgumentTypeError(
                "--max-batch-size must be in the range [1, 5000]"
            )
        if not (0 <= self.max_preempt_count <= self.max_batch_size):
            raise argparse.ArgumentTypeError(
                "--max-preempt-count must be in the range [0, --max-batch-size]"
            )
        if not (1 <= self.max_prefill_batch_size <= self.max_batch_size):
            raise argparse.ArgumentTypeError(
                "--max-prefill-batch-size must be in the range [1, --max-batch-size]"
            )
        if not (0 <= self.prefill_time_ms_per_req <= 1000):
            raise argparse.ArgumentTypeError(
                "--prefill-time-ms-per-req must be in the range [0, 1000]"
            )
        if not (0 <= self.decode_time_ms_per_req <= 1000):
            raise argparse.ArgumentTypeError(
                "--decode-time-ms-per-req must be in the range [0, 1000]"
            )
        if not (500 <= self.max_queue_delay_microseconds <= 1000000):
            raise argparse.ArgumentTypeError(
                "--max-queue-delay-microseconds must be in the range [500, 1000000]"
            )
        if self.rope_scaling:
            try:
                self.rope_scaling_parsed = json.loads(self.rope_scaling)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"--rope-scaling must be a valid JSON string: {self.rope_scaling_parsed}"
                ) from e
        if self.override_generation_config:
            try:
                self.override_generation_config_parsed = json.loads(
                    self.override_generation_config
                )
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"--override-generation-config must be a valid JSON string: {self.override_generation_config}"
                ) from e


class AscendMindIEServer(InferenceServer):
    model_path_mapped: Optional[Path] = None

    def __del__(self):
        self.cleanup()

    def start(self):
        # Prepare
        self.model_path_mapped = self._map_model_path()
        # Start
        try:
            self._start()
        except Exception as e:
            self._report_error(e)
            raise e
        finally:
            self.cleanup()

    def cleanup(self):
        # Clean up
        if self.model_path_mapped:
            shutil.rmtree(self.model_path_mapped)
        self.model_path_mapped = None

    def _start(self):  # noqa: max-complexity=15
        """
        Start Ascend MindIE service.
        """
        version = self._model.backend_version
        if not version:
            # Allow to control the version installed by user,
            # this relies on the environment setting.
            # There is a risk of failure, but flexible.
            # When error happens, specify a version to avoid this.
            version = "latest"

        # Select root path
        root_path = next(
            (
                rp
                for rp in envs.get_unix_available_root_paths_of_ascend()
                if rp.joinpath("mindie", version).is_dir()
            ),
            None,
        )
        if not root_path:
            e = FileNotFoundError(
                f"Ascend MindIE version {version} is not installed. "
                "Please install it first."
            )
            raise e

        install_path = root_path.joinpath("mindie", version, "mindie-service")

        # Load config,
        # the config includes two parts: environment variables and a JSON configuration file.
        logger.info("Loading Ascend MindIE config")

        # - Load environment variables,
        #   see https://www.hiascend.com/document/detail/zh/mindie/100/mindiellm/llmdev/mindie_llm0416.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0300.html.
        env = self.get_inference_running_env(version=version)

        # - Load JSON configuration,
        #   see https://www.hiascend.com/document/detail/zh/mindie/100/mindiellm/llmdev/mindie_llm0004.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0285.html.
        with open(
            install_path.joinpath("conf", "config.json"), "r", encoding="utf-8"
        ) as f:
            config = json.load(f)
        log_config = config.get("LogConfig", {})  # Deprecated since MindIE 2.0.RC1
        server_config = config["ServerConfig"]
        backend_config = config["BackendConfig"]
        model_deploy_config = backend_config["ModelDeployConfig"]
        model_config = model_deploy_config["ModelConfig"][0]
        schedule_config = backend_config["ScheduleConfig"]

        # Mutate config
        logger.info("Mutating Ascend MindIE config")

        # - Global config
        # -- Pin installation path, which helps to locate other resources.
        env["MIES_INSTALL_PATH"] = str(install_path)
        # -- Enable high performance swapper.
        # env["MIES_USE_MB_SWAPPER"] = "1"  # Atlas 300I Duo needs to unset this.
        env["MIES_RECOMPUTE_THRESHOLD"] = "0.5"
        # env["MINDIE_LLM_USE_MB_SWAPPER"] = "1"  # Atlas 300I Duo needs to unset this.
        env["MINDIE_LLM_RECOMPUTE_THRESHOLD"] = "0.5"
        # -- Enforce continues batching.
        env["MINDIE_LLM_CONTINUOUS_BATCHING"] = "1"
        # -- Disable checking files permission.
        env["MINDIE_CHECK_INPUTFILES_PERMISSION"] = "0"
        # -- Enforce using ATB as backend
        env["MINDIE_LLM_FRAMEWORK_BACKEND"] = "ATB"
        # -- Asynchronous ATB execution
        env["ATB_OPERATION_EXECUTE_ASYNC"] = "1"
        # -- Asynchronous operators emitting
        env["TASK_QUEUE_ENABLE"] = "1"
        # -- Enforce using 90% of GPU memory
        env["NPU_MEMORY_FRACTION"] = "0.9"
        # -- Pop conflict configuration items.
        env.pop("RANKTABLEFILE", "")  # TODO need for host-across deployment.
        env.pop("RANK_TABLE_FILE", "")  # TODO need for host-across deployment.
        env.pop("NPU_VISIBLE_DEVICES", "")
        env.pop("NPU-VISIBLE-DEVICES", "")
        env.pop("NPU_DEVICE_IDS", "")
        env.pop("ASCEND_RT_VISIBLE_DEVICES", "")
        env.pop("MIES_CONTAINER_IP", "")
        env.pop("MIES_CONTAINER_MANAGEMENT_IP", "")

        # - Logging config
        # -- Ascend MindIE
        env["MINDIE_LOG_LEVEL"] = "INFO"
        env["MINDIE_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Service
        env["MIES_CERTS_LOG_LEVEL"] = "INFO"
        env["MIES_CERTS_LOG_TO_STDOUT"] = "1"
        env["MIES_CERTS_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE LLM
        env["MINDIE_LLM_LOG_LEVEL"] = "WARN"
        env["MINDIE_LLM_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LLM_LOG_TO_FILE"] = "0"
        env["MINDIE_LLM_PYTHON_LOG_LEVEL"] = "WARN"
        env["MINDIE_LLM_PYTHON_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LLM_PYTHON_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Runtime
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
        env["ASCEND_SLOG_LEVEL"] = "WARN"
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
        env["ASCEND_SLOG_PRINT_TO_FILE"] = "0"
        env["MINDIE_RT_LOG_LEVEL"] = "3"  # 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
        env["MINDIE_RT_LOG_PRINT_TO_STDOUT"] = "1"
        env["MINDIE_RT_LOG_PRINT_TO_FILE"] = "0"
        # -- Ascend MindIE ATB
        env["ATB_LOG_LEVEL"] = "ERROR"
        env["ATB_LOG_TO_STDOUT"] = "1"
        env["ATB_LOG_TO_FILE"] = "0"
        env["LOG_LEVEL"] = "ERROR"
        env["LOG_TO_STDOUT"] = "1"
        env["LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Model
        env["ASDOPS_LOG_LEVEL"] = "ERROR"
        env["ASDOPS_LOG_TO_STDOUT"] = "1"
        env["ASDOPS_LOG_TO_FILE"] = "0"
        env["ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE"] = "0"
        # -- Ascend MindIE OCK
        env["OCK_LOG_LEVEL"] = "ERROR"
        env["OCK_LOG_TO_STDOUT"] = "1"
        env["OCK_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Torch
        env["TORCH_AIE_LOG_LEVEL"] = "3"  # 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
        env["TORCH_AIE_PRINT_TO_STDOUT"] = "1"
        env["TORCH_AIE_PRINT_TO_FILE"] = "0"

        # - Listening config
        server_config["ipAddress"] = "0.0.0.0"
        server_config.pop("managementIpAddress", None)
        server_config["allowAllZeroIpListening"] = True
        server_config["maxLinkNum"] = 1000
        server_config["port"] = self._model_instance.port
        server_config["managementPort"] = self._model_instance.port
        server_config["metricsPort"] = self._model_instance.port
        server_config["httpsEnabled"] = False
        server_config["interCommTLSEnabled"] = False

        # - Device config
        backend_config["npuDeviceIds"] = [self._model_instance.gpu_indexes]
        backend_config["multiNodesInferEnabled"] = False
        backend_config["interNodeTLSEnabled"] = False
        model_config["worldSize"] = len(self._model_instance.gpu_indexes)

        # - Model config
        max_seq_len = self._get_model_max_seq_len()
        # -- Mutate default max sequence length (aka. context length),
        #    but allow to change it with below advanced parameters.
        if max_seq_len > 8192:
            max_seq_len = 8192
        model_deploy_config["maxSeqLen"] = max_seq_len
        model_deploy_config["maxInputTokenLen"] = max_seq_len
        model_deploy_config["truncation"] = False
        schedule_config["maxIterTimes"] = max_seq_len
        schedule_config["maxPrefillTokens"] = max_seq_len
        model_config["modelName"] = self._model.name
        model_config["modelWeightPath"] = str(self.model_path_mapped)

        # - Customize config, translate to Ascend MindIE configuration language,
        #   see https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0285.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/100/mindiellm/llmdev/mindie_llm0302.html,
        #       ttps://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0300.html.
        if self._model.backend_parameters:
            logger.debug(
                f"Parsing given parameters: {os.linesep}{os.linesep.join(self._model.backend_parameters)}"
            )
            params = AscendMindIEParameters(max_seq_len=max_seq_len)
            params.from_args(self._model.backend_parameters)

            # -- Log config
            log_config["logLevel"] = params.log_level
            env["MINDIE_LOG_LEVEL"] = params.log_level.upper()
            # -- Server config
            server_config["maxLinkNum"] = params.max_link_num
            # -- Model deploy config.
            model_deploy_config["maxSeqLen"] = params.max_seq_len
            model_deploy_config["maxInputTokenLen"] = params.max_input_token_len
            schedule_config["maxIterTimes"] = params.max_seq_len
            schedule_config["maxPrefillTokens"] = params.max_seq_len
            model_deploy_config["truncation"] = params.truncation
            # -- Model config.
            model_config["cpuMemSize"] = params.cpu_mem_size
            env["NPU_MEMORY_FRACTION"] = str(params.npu_memory_fraction)
            model_config["trustRemoteCode"] = params.trust_remote_code
            # -- Schedule config.
            schedule_config["cacheBlockSize"] = params.cache_block_size
            schedule_config["maxPrefillBatchSize"] = params.max_prefill_batch_size
            schedule_config["prefillTimeMsPerReq"] = params.prefill_time_ms_per_req
            schedule_config["prefillPolicyType"] = params.prefill_policy_type
            schedule_config["maxBatchSize"] = params.max_batch_size
            schedule_config["decodeTimeMsPerReq"] = params.decode_time_ms_per_req
            schedule_config["decodePolicyType"] = params.decode_policy_type
            schedule_config["maxPreemptCount"] = params.max_preempt_count
            schedule_config["supportSelectBatch"] = params.support_select_batch
            schedule_config["maxQueueDelayMicroseconds"] = (
                params.max_queue_delay_microseconds
            )
            # -- Features
            # --- Prefix cache.
            if params.enable_prefix_caching:
                schedule_config["enablePrefixCache"] = True
                model_config["plugin_params"] = json.dumps(
                    {
                        "plugin_type": "prefix_cache",
                    }
                )
            # --- Exposing metrics.
            if params.metrics:
                env["MIES_SERVICE_MONITOR_MODE"] = "1"
            # --- Emitting operators in synchronous way.
            env["TASK_QUEUE_ENABLE"] = "0" if params.enforce_eager else "1"
            # --- Mutating model config.
            model_config_path = self.model_path_mapped.joinpath("config.json")
            with open(
                model_config_path,
                "r",
                encoding="utf-8",
            ) as f:
                model_path_config = json.load(f)
            # Merge the updated model config with the existing one
            if params.dtype != "auto":
                dtype = params.dtype
                if dtype == "half":
                    dtype = "float16"
                elif dtype == "float":
                    dtype = "float32"
                model_path_config["torch_dtype"] = dtype
            if params.rope_scaling_parsed:
                rope_scaling = model_path_config.get("rope_scaling")
                if rope_scaling:
                    # Merge the updated RoPE scaling config with the existing one
                    rope_scaling.update(params.rope_scaling_parsed)
                else:
                    # Override the RoPE scaling config
                    rope_scaling = params.rope_scaling_parsed
                model_path_config["rope_scaling"] = rope_scaling
            if params.rope_theta:
                model_path_config["rope_theta"] = params.rope_theta
            # Save the mutated model config
            with open(
                model_config_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(model_path_config, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved model config to {model_config_path}")
            # --- Mutating model generation config.
            model_generation_config_path = self.model_path_mapped.joinpath(
                "generation_config.json"
            )
            if params.override_generation_config_parsed:
                if model_generation_config_path.exists():
                    with open(
                        model_generation_config_path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        generation_config = json.load(f)
                    # Merge the updated generation config with the existing one
                    generation_config.update(params.override_generation_config_parsed)
                else:
                    # Override the generation config
                    generation_config = params.override_generation_config_parsed
                # Save the new generation config
                with open(
                    model_generation_config_path,
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(generation_config, f, indent=4, ensure_ascii=False)
                logger.info(
                    f"Saved model generation config to {model_generation_config_path}"
                )

        # Generate JSON configuration file by model instance id
        config_path = install_path.joinpath(
            "conf", f"config-{self._model_instance.id}.json"
        )
        config_str = json.dumps(config, indent=4, ensure_ascii=False)
        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(config_str)
        logger.info(f"Saved Ascend MindIE config to {config_path}")

        # Start, configure environment variable to indicate the JSON configuration file.
        env["MIES_CONFIG_JSON_PATH"] = str(config_path)
        service_path = root_path.joinpath("mindie", version, "mindie-service")
        service_bin_path = service_path.joinpath("bin", "mindieservice_daemon")

        try:
            # Display environment variables and JSON configuration.
            logger.info(f"Starting Ascend MindIE: {service_bin_path}")
            env_view = None
            if logger.isEnabledFor(logging.DEBUG):
                env_view = env
            elif self._model.env:
                # If the model instance has its own environment variables,
                # display the mutated environment variables.
                env_view = self._model.env
                for k, v in self._model.env.items():
                    env_view[k] = env.get(k, v)
            if env_view:
                logger.info(
                    f"With environment variables(inconsistent input items mean unchangeable):{os.linesep}{os.linesep.join(f'{k}={v}' for k, v in sorted(env_view.items()))}"
                )
            logger.info(
                f"With JSON configuration(inconsistent input items mean unchangeable):{os.linesep}{config_str}"
            )

            # Fork, inject environment variables and set working directory.
            proc = subprocess.Popen(
                [str(service_bin_path)],
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
                cwd=service_path,
            )
            exit_code = proc.wait()

            self.exit_with_code(exit_code)

        except Exception as e:
            raise e

        finally:
            # Finally, remove JSON configuration file.
            config_path.unlink(missing_ok=True)

    def _map_model_path(self) -> Path:
        """
        Map model path.

        Since MindIE doesn't support fine-grained model configuration,
        we need to construct a temporary model path with weight sort links,
        and copy the model configurations, like "config.json", "generation_config.json".

        First, create a temporary directory,
        it should be a subdirectory of system temporary directory,
        e.g. "/tmp/ascend-mindie-<model_id>-xxx".
        Then, link all files and directories to the temporary directory,
        excluding: "config.json" and "generation_config.json".
        Finally, copy "config.json" and "generation_config.json" files to the temporary directory.
        """

        raw = Path(self._model_path)

        # Check if the model path exists
        if not raw.is_dir():
            raise FileNotFoundError(f"Model path {raw} does not exist.")

        # Create a temporary directory
        mapped = tempfile.mkdtemp(prefix=f"ascend-mindie-{self._model_instance.id}-")
        mapped = Path(mapped)

        mutable_files = ["config.json", "generation_config.json"]

        # Link all files and directories to the temporary directory, excluding mutable files
        for item in raw.iterdir():
            if item.name in mutable_files:
                continue
            link_path = mapped.joinpath(item.name)
            try:
                os.symlink(item, link_path)
            except FileExistsError:
                # If the link already exists, remove it and create a new one
                link_path.unlink(missing_ok=True)
                os.symlink(item, link_path)

        # Copy mutable files to the temporary directory
        for item in mutable_files:
            src = raw.joinpath(item)
            if not src.is_file():
                continue
            dst = mapped.joinpath(item)
            # Copy the file to the temporary directory
            with open(src, "rb") as src_file:
                with open(dst, "wb") as dst_file:
                    dst_file.write(src_file.read())

        logger.info(f"Mapped original model path {self._model_path} to {mapped}")
        return mapped

    def _report_error(self, ex: Exception):
        """
        Report error message to the model instance.
        """
        error_message = f"Failed to run Ascend MindIE: {ex}"
        logger.error(error_message)
        try:
            patch_dict = {
                "state_message": error_message,
                "state": ModelInstanceStateEnum.ERROR,
            }
            self._update_model_instance(self._model_instance.id, **patch_dict)
        except Exception as e:
            logger.error(f"Failed to update model instance state: {e}")

    def _get_model_max_seq_len(self) -> Optional[int]:
        """
        Get the maximum sequence length of the model.
        """
        try:
            pretrained_config = get_pretrained_config(self._model)
            pretrained_or_hf_text_config = get_hf_text_config(pretrained_config)
            return get_max_model_len(pretrained_or_hf_text_config)
        except Exception as e:
            logger.error(f"Failed to get model max seq length: {e}")

        return 8192
