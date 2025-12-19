import argparse
import dataclasses
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any

from gpustack_runtime.deployer import (
    Container,
    ContainerProfileEnum,
    ContainerExecution,
    ContainerEnv,
    WorkloadPlan,
    create_workload,
    ContainerFile,
    ContainerRestartPolicyEnum,
)
from gpustack_runtime.envs import to_bool

from gpustack.schemas.models import ModelInstanceDeploymentMetadata
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.base import InferenceServer, is_ascend_310p

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AscendMindIEParameters:
    #
    # Log config
    #
    log_level: str = "Info"
    #
    # Server config
    #
    max_link_num: int = 1000
    token_timeout: int = 600
    e2e_timeout: int = 600
    #
    # Backend config
    #
    kv_pool_config: Optional[str] = None
    kv_pool_config_parsed: Optional[Dict[str, Any]] = None  # store JSON parsed result
    #
    # Model deploy config
    #
    max_seq_len: int = 8192
    max_input_token_len: int = -1
    truncation: bool = False
    #
    # Model config
    #
    cpu_mem_size: int = 0
    npu_memory_fraction: float = 0.8
    trust_remote_code: bool = False
    models: Optional[str] = None
    models_parsed: Optional[any] = None  # store JSON parsed result
    async_scheduler_wait_time: int = 120
    #
    # Schedule config
    #
    cache_block_size: int = 128
    max_prefill_batch_size: int = 50
    max_prefill_tokens: int = -1
    prefill_time_ms_per_req: int = 150
    prefill_policy_type: int = 0
    max_batch_size: int = 200
    max_iter_times: int = -1
    decode_time_ms_per_req: int = 50
    decode_policy_type: int = 0
    max_preempt_count: int = 0
    support_select_batch: bool = False
    max_queue_delay_microseconds: int = 5000
    max_first_token_wait_time: int = 2500
    #
    # Extends or Features
    #
    override_generation_config: Optional[str] = None
    override_generation_config_parsed: Optional[any] = None  # store JSON parsed result
    enforce_eager: bool = False
    no_metrics: bool = False
    dtype: str = "auto"
    rope_scaling: Optional[str] = None
    rope_scaling_parsed: Optional[any] = None  # store JSON parsed result
    rope_theta: Optional[float] = None
    enable_split: bool = False
    policy_type: int = 0
    split_chunk_tokens: int = 512
    split_start_batch_size: int = 16
    enable_memory_decoding: bool = False
    memory_decoding_length: int = 16
    memory_decoding_dynamic_algo: bool = False
    enable_lookahead: bool = False
    lookahead_level: int = 4
    lookahead_window: int = 5
    lookahead_guess_set_size: int = 5
    enable_multi_token_prediction: bool = False
    multi_token_prediction_tokens: int = 1
    enable_prefix_caching: bool = False
    local_world_size: int = -1  # store validation input
    world_size: int = -1  # store validation input
    pipeline_parallel_size: int = 1
    data_parallel_size: int = -1
    context_parallel_size: int = -1
    tensor_parallel_size: int = -1
    sequence_parallel_size: int = -1
    moe_expert_parallel_size: int = -1
    moe_tensor_parallel_size: int = -1
    enable_buffer_response: bool = False
    prefill_expected_time_ms: Optional[int] = None
    decode_expected_time_ms: Optional[int] = None

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
        # Backend config
        #
        parser.add_argument(
            "--kv-pool-config",
            type=str,
            default=self.kv_pool_config,
            help="KV pool configuration in JSON format. "
            "For example: `{\"backend\":\"<KV pool backend name>\", \"configPath\":\"/path/to/your/config/file\"}`.",
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
            "Works when specified `--max-preempt-count`.",
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
        parser.add_argument(
            "--models",
            type=str,
            required=False,
            help="Models configuration in JSON format, for certain specific configurations, like Expert Parallelism Implementation Method, Tensor Parallelism LM Header/Output Attention Split.",
        )
        parser.add_argument(
            "--async-scheduler-wait-time",
            type=int,
            default=self.async_scheduler_wait_time,
            help="The wait time (in seconds) for the asynchronous scheduler to start.",
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
            "--max-prefill-tokens",
            type=int,
            default=self.max_prefill_tokens,
            help="During each prefill, the total number of all input tokens in the current batch cannot exceed `--max-prefill-tokens`. Default is same as `--max-seq-len`.",
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
            "--max-iter-times",
            type=int,
            default=self.max_iter_times,
            help="Maximum iterations for decoding stage. Default is same as `--max-seq-len`.",
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
            "select the execution priority for this batch. "
            "Use `--no-support-select-batch` to disable explicitly.",
        )
        parser.add_argument(
            "--max-queue-delay-microseconds",
            type=int,
            default=self.max_queue_delay_microseconds,
            help="Maximum microseconds of queue waiting.",
        )
        parser.add_argument(
            "--max-first-token-wait-time",
            type=int,
            default=self.max_first_token_wait_time,
            help="Maximum milliseconds to wait for the first token generation.",
        )
        #
        # Extends or Features
        #
        parser.add_argument(
            "--override-generation-config",
            type=str,
            required=False,
            help="Overrides or sets generation config in JSON format. "
            "For example: `{\"temperature\": 0.5}`. "
            "This will merge into the `generation_config.json` of the model structure.",
        )
        parser.add_argument(
            "--enable-memory-decoding",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Enable memory decoding speculation. "
            "Use `--no-enable-memory-decoding` to disable explicitly.",
        )
        parser.add_argument(
            "--memory-decoding-length",
            type=int,
            default=self.memory_decoding_length,
            help="Length for memory decoding speculation.",
        )
        parser.add_argument(
            "--memory-decoding-dynamic-algo",
            action="store_true",
            help="Enable dynamic algorithm for memory decoding speculation.",
        )
        parser.add_argument(
            "--enable-lookahead",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Enable lookahead speculation. "
            "Use `--no-enable-lookahead` to disable explicitly.",
        )
        parser.add_argument(
            "--lookahead-level",
            type=int,
            default=self.lookahead_level,
            help="Level for lookahead speculation.",
        )
        parser.add_argument(
            "--lookahead-window",
            type=int,
            default=self.lookahead_window,
            help="Window size for lookahead speculation.",
        )
        parser.add_argument(
            "--lookahead-guess-set-size",
            type=int,
            default=self.lookahead_guess_set_size,
            help="Guess set size for lookahead speculation.",
        )
        parser.add_argument(
            "--enable-buffer-response",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Enable buffer response. "
            "Use `--no-enable-buffer-response` to disable explicitly.",
        )
        parser.add_argument(
            "--prefill-expected-time-ms",
            type=int,
            required=False,
            help="Expected latency (SLO) for Time to First Token (TTFT) in milliseconds.",
        )
        parser.add_argument(
            "--decode-expected-time-ms",
            type=int,
            required=False,
            help="Expected latency (SLO) for Time Per Output Token (TPOT) in milliseconds.",
        )
        parser.add_argument(
            "--enable-split",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Enable split fuse, something like chunked prefill. "
            "Use `--no-enable-split` to disable explicitly.",
        )
        parser.add_argument(
            "--policy-type",
            type=int,
            choices=[0, 4, 5, 6, 7],
            default=self.policy_type,
            help="Strategy of split fuse. "
            "- `0`: FCFS, first come first serving, "
            "- `4`: SJF, shortest job first, "
            "- `5`: LJF, longest job first, "
            "- `6`: Skip-Join MLFQ, skip-Join multi-levels feedback queue, "
            "- `7`: SJF-MLFQ, shortest job first and multi-levels feedback queue.",
        )
        parser.add_argument(
            "--split-chunk-tokens",
            type=int,
            default=self.split_chunk_tokens,
            help="Tokens size to batch for split fuse. Multiple of 16.",
        )
        parser.add_argument(
            "--split-start-batch-size",
            type=int,
            default=self.split_start_batch_size,
            help="Batch size to start splitting for split fuse.",
        )
        parser.add_argument(
            "--enable-multi-token-prediction",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Enable multi-token prediction. "
            "Use `--no-enable-multi-token-prediction` to disable explicitly.",
        )
        parser.add_argument(
            "--multi-token-prediction-tokens",
            type=int,
            default=self.multi_token_prediction_tokens,
            help="Number of multi-token prediction tokens. "
            "This is only effective when `--enable-multi-token-prediction` is enabled.",
        )
        parser.add_argument(
            "--enable-prefix-caching",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Enable prefix caching. "
            "Use `--no-enable-prefix-caching` to disable explicitly.",
        )
        parser.add_argument(
            "--no-metrics",
            action='store_true',
            help="Disable exposing metrics in /metrics router.",
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
            "--pipeline-parallel-size",
            "-pp",
            type=int,
            default=self.pipeline_parallel_size,
            required=False,
            help="Number of pipeline parallel groups.",
        )
        parser.add_argument(
            "--data-parallel-size",
            "-dp",
            type=int,
            default=self.data_parallel_size,
            required=False,
            help="Number of data parallel groups for Attention layers. "
            "`-1` means disabling data parallelism, otherwise, must be a power of 2.",
        )
        parser.add_argument(
            "--context-parallel-size",
            "-cp",
            type=int,
            default=self.context_parallel_size,
            required=False,
            help="Number of context parallel groups for Attention layers."
            "`-1` means disabling context parallelism, otherwise, must be power of 2.",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=self.tensor_parallel_size,
            required=False,
            help="Number of tensor parallel groups for Attention layers."
            "`-1` means using world size as tensor parallel size, otherwise, must be a power of 2.",
        )
        parser.add_argument(
            "--sequence-parallel-size",
            "-sp",
            type=int,
            default=self.sequence_parallel_size,
            required=False,
            help="Number of sequence parallel groups for MLP layers. "
            "`-1` means disabling sequence parallelism, otherwise, must be power of 2.",
        )
        parser.add_argument(
            "--moe-expert-parallel-size",
            "-moe-ep",
            type=int,
            default=self.moe_expert_parallel_size,
            required=False,
            help="Number of expert parallel groups. "
            "`-1` means disabling MoE expert parallelism, otherwise, must be power of 2.",
        )
        parser.add_argument(
            "--moe-tensor-parallel-size",
            "-moe-tp",
            type=int,
            default=self.moe_tensor_parallel_size,
            required=False,
            help="Number of tensor parallel groups for MoE MLP layers. "
            "`-1` and means using world size as MoE tensor parallel size, otherwise, must be power of 2. ",
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

        if args:
            args_parsed = parser.parse_known_args(args=args)
            for attr_name in [attr.name for attr in dataclasses.fields(self.__class__)]:
                try:
                    attr_value = getattr(args_parsed[0], attr_name, None)
                    if attr_value is not None:
                        try:
                            setattr(self, attr_name, attr_value)
                        except ValueError as e:
                            # Never reach here, but just in case.
                            raise argparse.ArgumentTypeError(
                                f"Invalid value for --{attr_name.replace('_', '-')} {attr_value}"
                            ) from e
                except AttributeError:
                    # If reach here, that means the field is an internal property,
                    # which would not register in the argument parser.
                    pass

        self._default()
        self._validate()

    def _default(self):  # noqa: C901
        # Model deploy config
        if self.max_input_token_len <= 0:
            if self.max_prefill_tokens > 0:
                self.max_input_token_len = min(
                    self.max_seq_len, self.max_prefill_tokens
                )
            else:
                self.max_input_token_len = self.max_seq_len
        # Model config
        self.max_prefill_batch_size = min(
            self.max_prefill_batch_size, self.max_batch_size
        )
        # Schedule config
        if self.max_prefill_tokens <= 0:
            self.max_prefill_tokens = self.max_seq_len
        if self.max_iter_times <= 0:
            self.max_iter_times = self.max_seq_len
        if self.max_preempt_count == 0 and self.cpu_mem_size > 0:
            self.cpu_mem_size = 0
        # Extends or Features
        # -- Parallelism
        if self.world_size > 0:
            # Base on the world size to infer other parallel sizes.
            #
            if self.tensor_parallel_size < 0:
                if self.pipeline_parallel_size > 1:
                    self.tensor_parallel_size = (
                        self.world_size // self.pipeline_parallel_size
                    )
                else:
                    self.tensor_parallel_size = self.world_size
                    if self.data_parallel_size > 1:
                        self.tensor_parallel_size //= self.data_parallel_size
                    elif self.context_parallel_size > 1:
                        self.tensor_parallel_size //= self.context_parallel_size
                        self.data_parallel_size = 1
            if self.moe_tensor_parallel_size < 0 and self.pipeline_parallel_size <= 1:
                if self.moe_expert_parallel_size > 1:
                    self.moe_tensor_parallel_size = (
                        self.world_size // self.moe_expert_parallel_size
                    )
                else:
                    self.moe_tensor_parallel_size = self.world_size
        else:
            # Infer the world size from other parallel sizes.
            #
            if self.pipeline_parallel_size > 1:
                if self.tensor_parallel_size < 0:
                    self.tensor_parallel_size = 1
                self.local_world_size = self.tensor_parallel_size
                self.world_size = (
                    self.pipeline_parallel_size * self.tensor_parallel_size
                )
            else:
                self.world_size = self.tensor_parallel_size
                if self.data_parallel_size > 1:
                    if self.tensor_parallel_size < 0:
                        self.tensor_parallel_size = 1
                    if self.local_world_size < 0:
                        self.local_world_size = self.tensor_parallel_size
                    self.world_size = (
                        self.data_parallel_size * self.tensor_parallel_size
                    )
                elif self.context_parallel_size > 1:
                    if self.tensor_parallel_size < 0:
                        self.tensor_parallel_size = 1
                    if self.local_world_size < 0:
                        self.local_world_size = self.tensor_parallel_size
                    self.world_size = (
                        self.context_parallel_size * self.tensor_parallel_size
                    )
                    self.data_parallel_size = 1
                if self.moe_expert_parallel_size > 1:
                    if self.moe_tensor_parallel_size < 0:
                        self.moe_tensor_parallel_size = 1
                    if self.tensor_parallel_size < 0:
                        self.tensor_parallel_size = self.moe_tensor_parallel_size
                    if self.local_world_size < 0:
                        self.local_world_size = self.tensor_parallel_size
                    self.world_size = (
                        self.moe_expert_parallel_size * self.moe_tensor_parallel_size
                    )
                elif self.moe_tensor_parallel_size < 0:
                    self.moe_tensor_parallel_size = self.world_size

    def _validate(self):  # noqa: C901
        # Server config
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
        # Backend config
        if self.kv_pool_config:
            try:
                self.kv_pool_config_parsed = json.loads(self.kv_pool_config)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"--kv-pool-config must be a valid JSON string: {self.kv_pool_config}"
                ) from e
        # Model deploy config
        if self.max_seq_len <= 0:
            raise argparse.ArgumentTypeError("--max-seq-len must be greater than 0")
        if not (0 < self.max_input_token_len <= self.max_seq_len):
            raise argparse.ArgumentTypeError(
                "--max-input-token-len must be in the range (0, --max-seq-len]"
            )
        # Model config
        if self.cpu_mem_size < 0:
            raise argparse.ArgumentTypeError(
                "--cpu-mem-size must be greater than or equal to 0"
            )
        if not (0 < self.npu_memory_fraction <= 1):
            raise argparse.ArgumentTypeError(
                "--npu-memory-fraction must be in the range (0, 1]"
            )
        if self.models:
            try:
                self.models_parsed = json.loads(self.models)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"--models must be a valid JSON string: {self.models}"
                ) from e
        if not (1 <= self.async_scheduler_wait_time <= 3600):
            raise argparse.ArgumentTypeError(
                "--async-scheduler-wait-time must be in the range [1, 3600]"
            )
        # Schedule config
        if self.cache_block_size & (self.cache_block_size - 1) != 0:
            raise argparse.ArgumentTypeError("--cache-block-size must be powers of 2")
        if not (1 <= self.max_prefill_batch_size <= self.max_batch_size):
            raise argparse.ArgumentTypeError(
                "--max-prefill-batch-size must be in the range [1, --max-batch-size]"
            )
        if not (0 <= self.prefill_time_ms_per_req <= 1000):
            raise argparse.ArgumentTypeError(
                "--prefill-time-ms-per-req must be in the range [0, 1000]"
            )
        if not (1 <= self.max_batch_size <= 5000):
            raise argparse.ArgumentTypeError(
                "--max-batch-size must be in the range [1, 5000]"
            )
        if not (
            self.max_input_token_len <= self.max_prefill_tokens <= self.max_seq_len
        ):
            raise argparse.ArgumentTypeError(
                "--max-prefill-tokens must be in the range [--max-input-token-len, --max-seq-len]"
            )
        if not (1 <= self.max_iter_times <= self.max_seq_len):
            raise argparse.ArgumentTypeError(
                "--max-iter-times must be in the range [1, --max-seq-len]"
            )
        if not (0 <= self.decode_time_ms_per_req <= 1000):
            raise argparse.ArgumentTypeError(
                "--decode-time-ms-per-req must be in the range [0, 1000]"
            )
        if not (0 <= self.max_preempt_count <= self.max_batch_size):
            raise argparse.ArgumentTypeError(
                "--max-preempt-count must be in the range [0, --max-batch-size]"
            )
        if not (500 <= self.max_queue_delay_microseconds <= 1000000):
            raise argparse.ArgumentTypeError(
                "--max-queue-delay-microseconds must be in the range [500, 1000000]"
            )
        if not (0 <= self.max_first_token_wait_time <= 3600000):
            raise argparse.ArgumentTypeError(
                "--max-first-token-wait-time must be in the range [0, 3600000]"
            )
        # Extends or Features
        if self.override_generation_config:
            try:
                self.override_generation_config_parsed = json.loads(
                    self.override_generation_config
                )
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"--override-generation-config must be a valid JSON string: {self.override_generation_config}"
                ) from e
        # -- Extending context size
        if self.rope_scaling:
            try:
                self.rope_scaling_parsed = json.loads(self.rope_scaling)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"--rope-scaling must be a valid JSON string: {self.rope_scaling}"
                ) from e
        # -- Split fuse
        if self.enable_split:
            if not (1 <= self.split_chunk_tokens <= 8192):
                raise argparse.ArgumentTypeError(
                    "--split-chunk-tokens must be in the range [1, 8192]"
                )
            elif self.split_chunk_tokens % 16 != 0:
                raise argparse.ArgumentTypeError(
                    "--split-chunk-tokens must be the multiple of 16"
                )
            if not (0 <= self.split_start_batch_size <= self.max_batch_size):
                raise argparse.ArgumentTypeError(
                    "--split-start-batch-size must be in the range [0, --max-batch-size]"
                )
        # -- Parallelism
        pp, tp, dp, cp, sp, moe_tp, moe_ep, ws, local_ws = (
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.data_parallel_size,
            self.context_parallel_size,
            self.sequence_parallel_size,
            self.moe_tensor_parallel_size,
            self.moe_expert_parallel_size,
            self.world_size,
            self.local_world_size,
        )
        if pp <= 0:
            raise argparse.ArgumentTypeError(
                "--pipeline-parallel-size must be greater than 0"
            )
        if tp > 0 and tp & (tp - 1) != 0:
            raise argparse.ArgumentTypeError(
                "--tensor-parallel-size must be the power of 2"
            )
        if dp > 0 and dp & (dp - 1) != 0:
            raise argparse.ArgumentTypeError(
                "--data-parallel-size must be the power of 2"
            )
        if cp > 0 and cp & (cp - 1) != 0:
            raise argparse.ArgumentTypeError(
                "--context-parallel-size must be the power of 2"
            )
        if sp > 0 and sp & (sp - 1) != 0:
            raise argparse.ArgumentTypeError(
                "--sequence-parallel-size must be the power of 2"
            )
        if moe_tp > 0 and moe_tp & (moe_tp - 1) != 0:
            raise argparse.ArgumentTypeError(
                "--moe-tensor-parallel-size must be the power of 2"
            )
        if moe_ep > 0 and moe_ep & (moe_ep - 1) != 0:
            raise argparse.ArgumentTypeError(
                "--moe-expert-parallel-size must be the power of 2"
            )
        if pp != 1 and dp != -1:
            raise argparse.ArgumentTypeError(
                f"--pipeline-parallel-size {pp} "
                f"and --data-parallel-size {dp} "
                "are incompatible, "
                "set --pipeline-parallel-size to 1 or disable data parallelism"
            )
        if dp > 1 and cp > 1:
            raise argparse.ArgumentTypeError(
                f"--data-parallel-size {dp} "
                f"and --context-parallel-size {cp} "
                "are incompatible, "
                "set --data-parallel-size to 1 or disable context parallelism"
            )
        # Check pp * tp == world size if enable pipeline parallelism
        if pp > 1:
            if 0 < ws != pp * tp:
                raise argparse.ArgumentTypeError(
                    f"--pipeline-parallel-size {pp} "
                    f"and --tensor-parallel-size {tp} "
                    f"must be multiples of world size: {ws}"
                )
        else:
            # Check tp == world size or tp <= local world size
            if 0 < local_ws < tp and 0 < ws != tp:
                raise argparse.ArgumentTypeError(
                    f"--tensor-parallel-size {tp} "
                    f"must be less or equal to local world size: {local_ws} "
                    f"or equal to world size: {ws}"
                )
            # Check dp * tp == world size if enable data parallelism
            if dp > 1:
                if 0 < ws != dp * tp:
                    raise argparse.ArgumentTypeError(
                        f"--data-parallel-size {dp} "
                        f"and --tensor-parallel-size {tp} "
                        f"must be multiples of world size: {ws}"
                    )
            # Check cp * tp == world size if enable context parallelism
            elif cp > 1:
                if 0 < ws != cp * tp:
                    raise argparse.ArgumentTypeError(
                        f"--context-parallel-size {cp} "
                        f"and --tensor-parallel-size {tp} "
                        f"must be multiples of world size: {ws}"
                    )
            # Check moe_tp * moe_ep == world size if enable expert parallelism
            if moe_ep > 1:
                # Check moe_tp == world size or moe_tp <= local world size
                if 0 < local_ws < moe_tp and 0 < ws != moe_tp:
                    raise argparse.ArgumentTypeError(
                        f"--moe-tensor-parallel-size {moe_tp} "
                        f"must be less or equal to local world size: {local_ws} "
                        f"or equal to world size: {ws}"
                    )
                if 0 < ws != moe_ep * moe_tp:
                    raise argparse.ArgumentTypeError(
                        f"--moe-expert-parallel-size {moe_ep}"
                        f"and --moe-tensor-parallel-size {moe_tp} "
                        f"must be multiples of world size: {ws}"
                    )
            # Otherwise, check moe_tp == world size
            else:
                if 0 < ws != moe_tp:
                    raise argparse.ArgumentTypeError(
                        f"--moe-tensor-parallel-size {moe_tp} "
                        f"must be equal to world size: {ws}"
                    )
            # Check sp == tp if enable sequence parallelism
            if sp > 1:
                if sp != tp:
                    raise argparse.ArgumentTypeError(
                        f"--sequence-parallel-size {sp} "
                        f"must be equal to --tensor-parallel-size {tp}"
                    )
        # -- Speculative decoding
        if self.enable_memory_decoding:
            if not (1 <= self.memory_decoding_length <= 16):
                raise argparse.ArgumentTypeError(
                    "--memory-decoding-length must be in the range [1, 16]"
                )
        if self.enable_lookahead:
            if not (3 <= self.lookahead_level <= 16):
                raise argparse.ArgumentTypeError(
                    "--lookahead-level must be in the range [3, 16]"
                )
            if not (1 <= self.lookahead_window <= 16):
                raise argparse.ArgumentTypeError(
                    "--lookahead-window must be in the range [1, 16]"
                )
            if not (1 <= self.lookahead_guess_set_size <= 16):
                raise argparse.ArgumentTypeError(
                    "--lookahead-guess-set-size must be in the range [1, 16]"
                )
        if self.enable_multi_token_prediction:
            if self.multi_token_prediction_tokens <= 0:
                raise argparse.ArgumentTypeError(
                    "--multi-token-prediction-tokens must be greater than 0"
                )
        # -- Buffer response
        if self.enable_buffer_response:
            if self.prefill_expected_time_ms is None:
                raise argparse.ArgumentTypeError(
                    "--prefill-expected-time-ms is required when --enable-buffer-response is enabled"
                )
            elif self.prefill_expected_time_ms <= 0:
                raise argparse.ArgumentTypeError(
                    "--prefill-expected-time-ms must be greater than 0"
                )
            if self.decode_expected_time_ms is None:
                raise argparse.ArgumentTypeError(
                    "--decode-expected-time-ms is required when --enable-buffer-response is enabled"
                )
            elif self.decode_expected_time_ms <= 0:
                raise argparse.ArgumentTypeError(
                    "--decode-expected-time-ms must be greater than 0"
                )

        # Feature compatibility check
        if self.enable_split:
            if self.enable_memory_decoding or self.enable_lookahead:
                raise argparse.ArgumentTypeError(
                    "--enable-memory-decoding and --enable-lookahead are not supported when --enable-split is enabled"
                )
            if self.rope_scaling:
                raise argparse.ArgumentTypeError(
                    "--rope-scaling is not supported when --enable-split is enabled"
                )
        if self.enable_memory_decoding:
            if self.enable_lookahead:
                raise argparse.ArgumentTypeError(
                    "--enable-lookahead is not supported when --enable-memory-decoding is enabled"
                )
            if self.rope_scaling:
                raise argparse.ArgumentTypeError(
                    "--rope-scaling is not supported when --enable-memory-decoding is enabled"
                )
        elif self.enable_lookahead:
            if self.rope_scaling:
                raise argparse.ArgumentTypeError(
                    "--rope-scaling is not supported when --enable-lookahead is enabled"
                )
        if self.enable_multi_token_prediction:
            if self.enable_memory_decoding or self.enable_lookahead:
                raise argparse.ArgumentTypeError(
                    "--enable-memory-decoding and --enable-lookahead are not supported when --enable-multi-token-prediction is enabled"
                )
            if self.enable_split:
                raise argparse.ArgumentTypeError(
                    "--enable-split is not supported when --enable-multi-token-prediction is enabled"
                )
            if self.rope_scaling:
                raise argparse.ArgumentTypeError(
                    "--rope-scaling is not supported when --enable-multi-token-prediction is enabled"
                )
        if self.enable_prefix_caching:
            if self.rope_scaling:
                raise argparse.ArgumentTypeError(
                    "--rope-scaling is not supported when --enable-prefix-caching is enabled"
                )
        if self.data_parallel_size > 1:
            if self.enable_memory_decoding or self.enable_lookahead:
                raise argparse.ArgumentTypeError(
                    "--enable-memory-decoding and --enable-lookahead are not supported when --data-parallel-size > 1"
                )
            if self.enable_split:
                raise argparse.ArgumentTypeError(
                    "--enable-split is not supported when --data-parallel-size > 1"
                )
            if self.enable_prefix_caching:
                raise argparse.ArgumentTypeError(
                    "--enable-prefix-caching is not supported when --data-parallel-size > 1"
                )


class AscendMindIEServer(InferenceServer):
    """
    Containerized Ascend MindIE inference server backend using gpustack-runtime.

    This backend preserves all the original logic from AscendMindIEServer but runs
    the final service in a Docker container instead of a subprocess.
    """

    def start(self):
        try:
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):  # noqa: C901
        logger.info(
            f"Starting Ascend MindIE model instance: {self._model_instance.name}"
        )
        # Prepare distributed information.
        dservers = self._model_instance.distributed_servers
        subworkers = (
            dservers.subordinate_workers
            if dservers and dservers.subordinate_workers
            else []
        )
        deployment_metadata = self._get_deployment_metadata()

        # Root path is defined by in Dockerfile ENV
        # https://github.com/gpustack/runner/blob/main/pack/cann/Dockerfile#L273
        root_path = Path("/usr/local/Ascend")
        install_path = root_path.joinpath("mindie", "latest", "mindie-service")

        # Load config,
        # the config includes two parts: environment variables and a JSON configuration file.
        logger.debug("Loading Ascend MindIE config")

        # - Load environment variables.
        env = self._get_configured_env()
        config_files: list[ContainerFile] = []

        # - Load JSON configuration,
        #   see https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0004.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html.
        config = self._get_mindie_config_json()
        log_config = config.get("LogConfig", {})  # Deprecated since MindIE 2.0.RC1
        server_config = config.get("ServerConfig", {})
        backend_config = config.get("BackendConfig", {})
        model_deploy_config = backend_config.get("ModelDeployConfig", {})
        model_config = model_deploy_config.get("ModelConfig", [{}])[0]
        schedule_config = backend_config.get("ScheduleConfig", {})

        # Mutate config
        logger.debug("Mutating Ascend MindIE config")

        # - Global config
        # -- Pin installation path, which helps to locate other resources.
        env["MIES_INSTALL_PATH"] = str(install_path)
        # -- Enable exposing metircs.
        env["MIES_SERVICE_MONITOR_MODE"] = env.pop("MIES_SERVICE_MONITOR_MODE", "1")
        # -- Enable high performance swapper.
        env["MIES_RECOMPUTE_THRESHOLD"] = env.pop("MIES_RECOMPUTE_THRESHOLD", "0.5")
        # env["MINDIE_LLM_USE_MB_SWAPPER"] = "1"  # Atlas 300I Duo needs to unset this.
        env["MINDIE_LLM_RECOMPUTE_THRESHOLD"] = env.pop(
            "MINDIE_LLM_RECOMPUTE_THRESHOLD", "0.5"
        )
        # -- Enforce continues batching.
        env["MINDIE_LLM_CONTINUOUS_BATCHING"] = env.pop(
            "MINDIE_LLM_CONTINUOUS_BATCHING", "1"
        )
        # -- Disable checking files permission.
        env["MINDIE_CHECK_INPUTFILES_PERMISSION"] = "0"
        # -- Enforce using ATB as backend
        env["MINDIE_LLM_FRAMEWORK_BACKEND"] = "ATB"
        # -- Enforce using 80% of GPU memory.
        env["NPU_MEMORY_FRACTION"] = "0.8"
        # -- Disable OpenMP parallelism, speed up model loading.
        env["OMP_NUM_THREADS"] = env.pop("OMP_NUM_THREADS", "1")
        # -- Enable safetensors GPU loading pass-through for faster model loading.
        env["SAFETENSORS_FAST_GPU"] = env.pop("SAFETENSORS_FAST_GPU", "1")
        # -- Improve performance.
        env["MINDIE_ASYNC_SCHEDULING_ENABLE"] = env.pop(
            "MINDIE_ASYNC_SCHEDULING_ENABLE", "1"
        )
        env["TASK_QUEUE_ENABLE"] = env.pop("TASK_QUEUE_ENABLE", "1")
        env["CPU_AFFINITY_CONF"] = env.pop("CPU_AFFINITY_CONF", "1")
        env["ATB_OPERATION_EXECUTE_ASYNC"] = "1"
        env["ATB_LAYER_INTERNAL_TENSOR_REUSE"] = env.pop(
            "ATB_LAYER_INTERNAL_TENSOR_REUSE", "1"
        )
        env["INF_NAN_MODE_ENABLE"] = env.pop("INF_NAN_MODE_ENABLE", "0")
        env["ATB_LLM_ENABLE_AUTO_TRANSPOSE"] = env.pop(
            "ATB_LLM_ENABLE_AUTO_TRANSPOSE", "1"
        )
        env["ATB_CONVERT_NCHW_TO_ND"] = env.pop("ATB_CONVERT_NCHW_TO_ND", "1")
        env["ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE"] = env.pop(
            "ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "3"
        )
        env["ATB_WORKSPACE_MEM_ALLOC_GLOBAL"] = env.pop(
            "ATB_WORKSPACE_MEM_ALLOC_GLOBAL", "1"
        )
        env["PYTORCH_NPU_ALLOC_CONF"] = env.pop(
            "PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"
        )
        # -- Pop conflict configuration items.
        env.pop("NPU_VISIBLE_DEVICES", "")
        env.pop("NPU-VISIBLE-DEVICES", "")
        env.pop("NPU_DEVICE_IDS", "")
        env.pop("ASCEND_LAUNCH_BLOCKING", "")
        env.pop("ASCEND_RT_VISIBLE_DEVICES", "")
        env.pop("MIES_CONTAINER_MANAGEMENT_IP", "")
        env.pop("WORLD_SIZE", "")
        env.pop("RANKTABLEFILE", "")
        env.pop("RANK_TABLE_FILE", "")
        if not deployment_metadata.distributed:
            env.pop("MIES_CONTAINER_IP", "")
            env.pop("HOST_IP", "")

        # - Listening config
        serving_port = self._get_serving_port()
        server_config["ipAddress"] = self._worker.ip
        server_config.pop("managementIpAddress", None)
        server_config["allowAllZeroIpListening"] = True
        server_config["maxLinkNum"] = 1000
        server_config["port"] = serving_port
        server_config["managementPort"] = serving_port
        server_config["metricsPort"] = serving_port
        server_config["httpsEnabled"] = False
        server_config["interCommTLSEnabled"] = False

        # - Device config
        backend_config["interNodeTLSEnabled"] = False
        backend_config["npuDeviceIds"] = [
            # Use logic(count) device indexes as NPU device IDs,
            # which is friendly to virtualized environments.
            list(range(len(self._model_instance.gpu_indexes)))
        ]
        model_config["worldSize"] = len(self._model_instance.gpu_indexes)
        backend_config["multiNodesInferEnabled"] = False
        if deployment_metadata.distributed:
            # Add distributed config if in distributed mode.
            backend_config["multiNodesInferEnabled"] = True
            # During distributed setup,
            # we must get more than one port here,
            # so we use ports[1] for distributed initialization.
            backend_config["multiNodesInferPort"] = self._model_instance.ports[1]
        if deployment_metadata.distributed_follower:
            subworker = subworkers[deployment_metadata.distributed_follower_index]
            # Override device config if is a subordinate worker.
            backend_config["npuDeviceIds"] = [
                # Use logic(count) device indexes as NPU device IDs,
                # which is friendly to virtualized environments.
                list(range(len(subworker.gpu_indexes)))
            ]
            model_config["worldSize"] = len(subworker.gpu_indexes)

        # - Model config
        derived_max_seq_len = self._derive_max_model_len(default=8192)
        max_seq_len = derived_max_seq_len
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
        model_config["modelWeightPath"] = self._model_path

        # - Customize config, translate to Ascend MindIE configuration language,
        #   see https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0300.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0302.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0424.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0009.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0300.html,
        #       https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0425.html.
        local_world_size = len(self._model_instance.gpu_indexes)
        world_size = local_world_size
        if deployment_metadata.distributed:
            world_size = local_world_size * (len(subworkers) + 1)
        params = AscendMindIEParameters(
            local_world_size=local_world_size,
            world_size=world_size,
            max_seq_len=max_seq_len,
        )
        # For Ascend 310P, we need to default dtype to float16.
        # As a workaround, we should allow users to override this with backend parameters.
        if is_ascend_310p(self._get_selected_gpu_devices()):
            original_params = self._model.backend_parameters or []
            self._model.backend_parameters = ["--dtype=float16"]
            self._model.backend_parameters.extend(original_params)
        if self._model.backend_parameters:
            logger.debug(
                f"Parsing given parameters: {os.linesep}{os.linesep.join(self._model.backend_parameters)}"
            )
            params.from_args(self._flatten_backend_param())

            # -- Log config
            log_config["logLevel"] = params.log_level
            env["MINDIE_LOG_LEVEL"] = params.log_level.upper()
            # -- Server config
            server_config["maxLinkNum"] = params.max_link_num
            # -- Backend config
            if params.kv_pool_config_parsed:
                backend_config["kvPoolConfig"] = params.kv_pool_config_parsed
            # -- Model deploy config
            model_deploy_config["maxSeqLen"] = params.max_seq_len
            model_deploy_config["maxInputTokenLen"] = params.max_input_token_len
            schedule_config["maxIterTimes"] = params.max_iter_times
            schedule_config["maxPrefillTokens"] = params.max_prefill_tokens
            model_deploy_config["truncation"] = params.truncation
            # -- Model config
            model_config["cpuMemSize"] = params.cpu_mem_size
            env["MIES_USE_MB_SWAPPER"] = "1" if params.cpu_mem_size > 0 else "0"
            env["NPU_MEMORY_FRACTION"] = str(params.npu_memory_fraction)
            model_config["trustRemoteCode"] = params.trust_remote_code
            if params.models_parsed:
                model_config["models"] = params.models_parsed
            model_config["async_scheduler_wait_time"] = params.async_scheduler_wait_time
            # -- Schedule config
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
            schedule_config["maxFirstTokenWaitTime"] = params.max_first_token_wait_time
            # -- Extends or Features
            # --- Disable exposing metrics
            if params.no_metrics:
                env["MIES_SERVICE_MONITOR_MODE"] = "0"
            # --- Emitting operators in synchronous way.
            if params.enforce_eager:
                env["MINDIE_ASYNC_SCHEDULING_ENABLE"] = "0"
                env["TASK_QUEUE_ENABLE"] = "0"
                env["ATB_OPERATION_EXECUTE_ASYNC"] = "0"
                env["ASCEND_LAUNCH_BLOCKING"] = "1"
            # --- Mutating model config.
            model_config_path = Path(self._model_path).joinpath("config.json")
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
            model_config_str = json.dumps(
                model_path_config,
                indent=4,
                ensure_ascii=False,
            )
            config_files.append(
                ContainerFile(
                    path=str(model_config_path),
                    content=model_config_str,
                    mode=0o750,
                ),
            )
            # --- Mutating model generation config
            model_generation_config_path = Path(self._model_path).joinpath(
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
                model_generation_config_str = json.dumps(
                    generation_config,
                    indent=4,
                    ensure_ascii=False,
                )
                config_files.append(
                    ContainerFile(
                        path=str(model_generation_config_path),
                        content=model_generation_config_str,
                    ),
                )
            # --- Split fuse
            if params.enable_split:
                schedule_config["enableSplit"] = True
                schedule_config["templateType"] = "Mix"
                schedule_config["policyType"] = params.policy_type
                schedule_config["splitType"] = False
                schedule_config["splitStartType"] = False
                schedule_config["splitChunkTokens"] = params.split_chunk_tokens
                schedule_config["splitStartBatchSize"] = params.split_start_batch_size
                model_config["plugin_params"] = json.dumps(
                    {
                        "plugin_type": "splitfuse",
                    }
                )
            # --- Speculative decoding
            if params.enable_memory_decoding:
                model_deploy_config["speculationGamma"] = params.memory_decoding_length
                if derived_max_seq_len > max_seq_len == schedule_config["maxIterTimes"]:
                    schedule_config["maxIterTimes"] = (
                        max_seq_len + params.memory_decoding_length
                    )
                model_config["plugin_params"] = json.dumps(
                    {
                        "plugin_type": "memory_decoding",
                        "decoding_length": params.memory_decoding_length,
                        "dynamic_algo": params.memory_decoding_dynamic_algo,
                    }
                )
            if params.enable_lookahead:
                model_deploy_config["speculationGamma"] = (
                    params.lookahead_level - 1
                ) * (params.lookahead_window + params.lookahead_guess_set_size)
                model_config["plugin_params"] = json.dumps(
                    {
                        "plugin_type": "la",
                        "level": params.lookahead_level,
                        "window": params.lookahead_window,
                        "guess_set_size": params.lookahead_guess_set_size,
                    }
                )
            # --- Multi-token prediction
            if params.enable_multi_token_prediction:
                model_config["plugin_params"] = json.dumps(
                    {
                        "plugin_type": "mtp",
                        "num_speculative_tokens": params.multi_token_prediction_tokens,
                    }
                )
            # --- Prefix cache
            if params.enable_prefix_caching:
                schedule_config["enablePrefixCache"] = True
                model_config["plugin_params"] = json.dumps(
                    {
                        "plugin_type": "prefix_cache",
                    }
                )
            # --- Parallelism
            if params.pipeline_parallel_size > 1:
                model_config["pp"] = params.pipeline_parallel_size
                model_config["tp"] = params.tensor_parallel_size
            else:
                if params.data_parallel_size > 0:
                    model_config["dp"] = params.data_parallel_size
                if params.context_parallel_size > 0:
                    model_config["cp"] = params.context_parallel_size
                if params.tensor_parallel_size > 0:
                    model_config["tp"] = params.tensor_parallel_size
                    model_config["moe_tp"] = params.moe_tensor_parallel_size
                if params.moe_expert_parallel_size > 0:
                    model_config["moe_ep"] = params.moe_expert_parallel_size
                    model_config["moe_tp"] = params.moe_tensor_parallel_size
                if params.sequence_parallel_size > 0:
                    model_config["sp"] = params.sequence_parallel_size
            # --- Asynchronous scheduling
            if params.max_batch_size <= 50:
                env["MINDIE_ASYNC_SCHEDULING_ENABLE"] = "0"
            # --- Buffer response
            if params.enable_buffer_response:
                schedule_config["bufferResponseEnabled"] = True
                schedule_config["prefillExpectedTime"] = params.prefill_expected_time_ms
                schedule_config["decodeExpectedTime"] = params.decode_expected_time_ms

        # Generate rank table file if needed,
        # see https://www.hiascend.com/document/detail/zh/mindie/20RC2/envdeployment/instg/mindie_instg_0027.html,
        #     https://www.hiascend.com/forum/thread-0237183374051498211-1-1.html
        if deployment_metadata.distributed:
            server_count = f"{len(subworkers) + 1}"
            server_list = [
                {
                    "server_id": self._model_instance.worker_ip,
                    "container_ip": self._model_instance.worker_ip,
                    "device": [
                        {
                            # Unlike above npuDeviceIds,
                            # here we must use real device indexes as device IDs.
                            # I guess Ascend needs to construct the communication topology based on real device IDs,
                            # see https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/hccl/hcclug/hcclug_000014.html#ZH-CN_TOPIC_0000002479883061__zh-cn_topic_0000001463640385_section10882094214.
                            #
                            # Since rank table will in charge of device mapping in distributed mode,
                            # the above logic(count) device indexes will not affect distributed deployment,
                            # see https://www.hiascend.com/document/detail/zh/mindie/21RC2/mindiellm/llmdev/mindie_llm0004.html#ZH-CN_TOPIC_0000002366997374__section7821428101811.
                            "device_id": str(self._model_instance.gpu_indexes[i]),
                            "device_ip": self._model_instance.gpu_addresses[i],
                            "rank_id": str(i),
                        }
                        for i in range(len(self._model_instance.gpu_indexes))
                    ],
                },
            ]
            for i, sw in enumerate(subworkers):
                server_list.append(
                    {
                        "server_id": sw.worker_ip,
                        "container_ip": sw.worker_ip,
                        "device": [
                            {
                                # Unlike above npuDeviceIds,
                                # here we must use real device indexes as device IDs.
                                # I guess Ascend needs to construct the communication topology based on real device IDs,
                                # see https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/hccl/hcclug/hcclug_000014.html#ZH-CN_TOPIC_0000002479883061__zh-cn_topic_0000001463640385_section10882094214.
                                #
                                # Since rank table will in charge of device mapping in distributed mode,
                                # the above logic(count) device indexes will not affect distributed deployment,
                                # see https://www.hiascend.com/document/detail/zh/mindie/21RC2/mindiellm/llmdev/mindie_llm0004.html#ZH-CN_TOPIC_0000002366997374__section7821428101811.
                                "device_id": str(sw.gpu_indexes[j]),
                                "device_ip": sw.gpu_addresses[j],
                                "rank_id": str(j + len(sw.gpu_indexes) * (i + 1)),
                            }
                            for j in range(len(sw.gpu_indexes))
                        ],
                    }
                )
            # Save rank table to a JSON file.
            rank_table = {
                "version": "1.0",
                "server_count": server_count,
                "server_list": server_list,
                "status": "completed",
            }
            rank_table_path = install_path.joinpath("conf", "ranktable.json")
            rank_table_str = json.dumps(rank_table, indent=4, ensure_ascii=False)
            config_files.append(
                ContainerFile(
                    path=str(rank_table_path),
                    content=rank_table_str,
                    mode=0o640,
                )
            )
            # - Set environment variables.
            env["WORLD_SIZE"] = str(
                len(self._model_instance.gpu_indexes) * (len(subworkers) + 1)
            )
            env["RANKTABLEFILE"] = str(rank_table_path)
            env["RANK_TABLE_FILE"] = str(rank_table_path)
            env["MIES_CONTAINER_IP"] = env.pop("MIES_CONTAINER_IP", self._worker.ip)
            env["HOST_IP"] = env.pop("HOST_IP", self._worker.ip)
            env["ATB_LLM_HCCL_ENABLE"] = env.pop("ATB_LLM_HCCL_ENABLE", "1")
            env["ATB_LLM_COMM_BACKEND"] = env.pop("ATB_LLM_COMM_BACKEND", "hccl")
            env["HCCL_CONNECT_TIMEOUT"] = env.pop("HCCL_CONNECT_TIMEOUT", "7200")
            env["HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT"] = env.pop(
                "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT", "TRUE"
            )
            if not is_ascend_310p(self._get_selected_gpu_devices()):
                env["HCCL_EXEC_TIMEOUT"] = env.pop("HCCL_EXEC_TIMEOUT", "0")
                env["HCCL_OP_EXPANSION_MODE"] = env.pop("HCCL_OP_EXPANSION_MODE", "AIV")
            # NB(thxCode): For deterministic calculation, needs the following environment variables.
            # LCCL_DETERMINISTIC=1
            # ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
            # HCCL_DETERMINISTIC=true
            # ATB_MATMUL_SHUFFLE_K_ENABLE=0
            # ATB_LLM_LCOC_ENABLE=0
            # HCCL_OP_EXPANSION_MODE=""
            logger.info(
                f"With rank table JSON configuration:{os.linesep}{rank_table_str}"
            )

        # Generate JSON configuration file by model instance id.
        config_path = str(install_path.joinpath("conf", "config.json"))
        config_str = json.dumps(config, indent=4, ensure_ascii=False)
        config_files.append(
            ContainerFile(
                path=config_path,
                content=config_str,
                mode=0o640,
            ),
        )
        logger.info(
            f"With JSON configuration(inconsistent input items mean unchangeable):{os.linesep}{config_str}"
        )

        # Indicate the JSON configuration file.
        env["MIES_CONFIG_JSON_PATH"] = str(config_path)

        command_script = self._get_serving_command_script(env)

        command_args = self.build_versioned_command_args(
            [
                str(install_path.joinpath("bin", "mindieservice_daemon")),
            ]
        )

        self._create_workload(
            deployment_metadata=deployment_metadata,
            command_script=command_script,
            command_args=command_args,
            env=env,
            config_files=config_files,
            working_dir=str(install_path.joinpath("bin")),
        )

    def _create_workload(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
        command_script: Optional[str],
        command_args: List[str],
        env: Dict[str, str],
        config_files: List[ContainerFile],
        working_dir: Optional[str],
    ):
        image = self._get_configured_image(backend="cann")
        if not image:
            raise ValueError("Failed to get Ascend MindIE backend image")

        resources = self._get_configured_resources(
            mount_all_devices=deployment_metadata.distributed,
        )

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        # Get container entrypoint from inference backend configuration
        container_entrypoint = None
        if self.inference_backend:
            container_entrypoint = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=True,
                command=container_entrypoint,
                command_script=command_script,
                args=command_args,
                working_dir=working_dir,
            ),
            envs=[
                ContainerEnv(
                    name=name,
                    value=value,
                )
                for name, value in env.items()
            ],
            resources=resources,
            mounts=mounts,
            files=config_files,
            ports=ports,
        )

        logger.info(
            f"Creating Ascend MindIE container workload: {deployment_metadata.name}"
        )
        logger.info(
            f"With image: {image}, "
            f"{('entrypoint: ' + str(container_entrypoint) + ', ') if container_entrypoint else ''}"
            f"arguments: [{' '.join(command_args)}], "
            f"ports: [{','.join([str(port.internal) for port in ports])}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=deployment_metadata.name,
            host_network=True,
            shm_size=10 * 1 << 30,  # 10 GiB
            containers=[run_container],
        )

        create_workload(self._transform_workload_plan(workload_plan))

        logger.info(
            f"Created Ascend MindIE container workload: {deployment_metadata.name}"
        )

    @staticmethod
    def _get_serving_command_script(env: dict[str, str]) -> Optional[str]:
        """
        Get serving command script for the MindIE service.
        """

        # Skip if explicitly disabled.
        if env and to_bool(
            env.get("GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED", "0")
        ):
            return None

        return """#!/usr/bin/bash

#
# Prepare
#

if [ -n "${PYPI_PACKAGES_INSTALL:-}" ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "Installing additional PyPi packages: ${PYPI_PACKAGES_INSTALL}"
        export UV_PRERELEASE=allow
        export UV_HTTP_TIMEOUT=500
        export UV_NO_CACHE=1
        if [ -n "${PIP_INDEX_URL:-}" ]; then
            export UV_DEFAULT_INDEX="${PIP_INDEX_URL}"
            export UV_INDEX_URL="${PIP_INDEX_URL}"
        fi
        if [ -n "${PIP_EXTRA_INDEX_URL:-}" ]; then
            export UV_INDEX="${PIP_EXTRA_INDEX_URL}"
            export UV_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL}"
        fi
        uv pip install --system ${PYPI_PACKAGES_INSTALL}
        uv pip tree --system
    elif command -v pip >/dev/null 2>&1; then
        echo "Installing additional PyPi packages: ${PYPI_PACKAGES_INSTALL}"
        export PIP_DISABLE_PIP_VERSION_CHECK=1
        export PIP_ROOT_USER_ACTION=ignore
        export PIP_PRE=1
        export PIP_TIMEOUT=500
        export PIP_NO_CACHE_DIR=1
        pip install ${PYPI_PACKAGES_INSTALL}
        pip freeze
    fi
    unset PYPI_PACKAGES_INSTALL
fi

#
# Execute
#

## Cache Envs Configured by GPUStack

MINDIE_LOG_LEVEL=${MINDIE_LOG_LEVEL:-INFO}
MIES_CERTS_LOG_LEVEL=${MIES_CERTS_LOG_LEVEL:-INFO}
MINDIE_LLM_LOG_LEVEL=${MINDIE_LLM_LOG_LEVEL:-WARN}
MINDIE_LLM_PYTHON_LOG_LEVEL=${MINDIE_LLM_PYTHON_LOG_LEVEL:-WARN}
ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
ASCEND_SLOG_LEVEL=${ASCEND_SLOG_LEVEL:-WARN}
MINDIE_RT_LOG_LEVEL=${MINDIE_RT_LOG_LEVEL:-3}
ATB_LOG_LEVEL=${ATB_LOG_LEVEL:-ERROR}
ASDOPS_LOG_LEVEL=${ASDOPS_LOG_LEVEL:-ERROR}
OCK_LOG_LEVEL=${OCK_LOG_LEVEL:-ERROR}
LOG_LEVEL=${LOG_LEVEL:-ERROR}
TORCH_AIE_LOG_LEVEL=${TORCH_AIE_LOG_LEVEL:-3}
CANN_HOME=${CANN_HOME:-/usr/local/Ascend}

## Activate Ascend Envs

PYTHON_LIB_PREFIX=$(python3 -c "import sys; print(sys.base_prefix);")
export LD_LIBRARY_PATH=${PYTHON_LIB_PREFIX}/lib:${PYTHON_LIB_PREFIX}/lib64:${LD_LIBRARY_PATH}
source ${CANN_HOME}/ascend-toolkit/set_env.sh
source ${CANN_HOME}/nnal/atb/set_env.sh
source ${CANN_HOME}/atb-models/set_env.sh
source ${CANN_HOME}/mindie/set_env.sh

## Override Envs Configured by GPUStack

export MINDIE_LOG_LEVEL=${MINDIE_LOG_LEVEL}
export MINDIE_LOG_TO_STDOUT=1
export MINDIE_LOG_TO_FILE=0
export MIES_CERTS_LOG_LEVEL=${MIES_CERTS_LOG_LEVEL}
export MIES_CERTS_LOG_TO_STDOUT=1
export MIES_CERTS_LOG_TO_FILE=0
export MINDIE_LLM_LOG_LEVEL=${MINDIE_LLM_LOG_LEVEL}
export MINDIE_LLM_LOG_TO_STDOUT=1
export MINDIE_LLM_LOG_TO_FILE=0
export MINDIE_LLM_PYTHON_LOG_LEVEL=${MINDIE_LLM_PYTHON_LOG_LEVEL}
export MINDIE_LLM_PYTHON_LOG_TO_STDOUT=1
export MINDIE_LLM_PYTHON_LOG_TO_FILE=0
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL}
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_LEVEL=${ASCEND_SLOG_LEVEL}
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_SLOG_PRINT_TO_FILE=0
export MINDIE_RT_LOG_LEVEL=${MINDIE_RT_LOG_LEVEL}
export MINDIE_RT_LOG_PRINT_TO_STDOUT=1
export MINDIE_RT_LOG_PRINT_TO_FILE=0
export ATB_LOG_LEVEL=${ATB_LOG_LEVEL}
export ATB_LOG_TO_STDOUT=1
export ATB_LOG_TO_FILE=0
export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
export ATB_LOG_TO_FILE_FLUSH=0
export ASDOPS_LOG_LEVEL=${ASDOPS_LOG_LEVEL}
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_FILE=0
export OCK_LOG_LEVEL=${OCK_LOG_LEVEL}
export OCK_LOG_TO_STDOUT=1
export OCK_LOG_TO_FILE=0
export LOG_LEVEL=${LOG_LEVEL}
export LOG_TO_STDOUT=1
export LOG_TO_FILE=0
export TORCH_AIE_LOG_LEVEL=${TORCH_AIE_LOG_LEVEL}
export TORCH_AIE_PRINT_TO_STDOUT=1
export TORCH_AIE_PRINT_TO_FILE=0

## Execute the binary preprocessed by GPUStack Runner if exists,
## otherwise, execute the original binary.

if [ -x ${CANN_HOME}/mindie/latest/mindie-service/bin/mindieservice_daemon_ ]; then
    ${CANN_HOME}/mindie/latest/mindie-service/bin/mindieservice_daemon_
else
    $@
fi
"""

    @staticmethod
    @lru_cache
    def _get_mindie_config_json() -> dict[str, Any]:
        config_str = """
{
    "Version" : "1.0.0",

    "ServerConfig" :
    {
        "ipAddress" : "127.0.0.1",
        "managementIpAddress" : "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000,
        "httpsEnabled" : true,
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsPkPwd" : "security/pass/key_pwd.txt",
        "tlsCrlPath" : "security/certs/",
        "tlsCrlFiles" : ["server_crl.pem"],
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrlPath" : "security/management/certs/",
        "managementTlsCrlFiles" : ["server_crl.pem"],
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "interCommTLSEnabled" : true,
        "interCommPort" : 1121,
        "interCommTlsCaPath" : "security/grpc/ca/",
        "interCommTlsCaFiles" : ["ca.pem"],
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrlPath" : "security/grpc/certs/",
        "interCommTlsCrlFiles" : ["server_crl.pem"],
        "openAiSupport" : "vllm",
        "tokenTimeout" : 600,
        "e2eTimeout" : 600,
        "distDPServerEnabled":false
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : false,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : true,
        "interNodeTlsCaPath" : "security/grpc/ca/",
        "interNodeTlsCaFiles" : ["ca.pem"],
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrlPath" : "security/grpc/certs/",
        "interNodeTlsCrlFiles" : ["server_crl.pem"],
        "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 2560,
            "maxInputTokenLen" : 2048,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "llama_65b",
                    "modelWeightPath" : "/data/atb_testdata/weights/llama1-65b-safetensors",
                    "worldSize" : 4,
                    "cpuMemSize" : 0,
                    "npuMemSize" : -1,
                    "backendType" : "atb",
                    "trustRemoteCode" : false,
                    "async_scheduler_wait_time": 120,
                    "kv_trans_timeout": 10,
                    "kv_link_timeout": 1080
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 50,
            "maxPrefillTokens" : 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 200,
            "maxIterTimes" : 512,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000,
            "maxFirstTokenWaitTime": 2500
        }
    }
}
"""
        return json.loads(config_str)
