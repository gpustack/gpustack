import argparse
import dataclasses
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Iterator, Any

from gpustack_runtime.deployer import (
    Container,
    ContainerProfileEnum,
    ContainerExecution,
    ContainerEnv,
    WorkloadPlan,
    create_workload,
    logs_workload,
    WorkloadStatus,
    get_workload,
    delete_workload,
    ContainerFile,
    ContainerPort,
)

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.base import InferenceServer, is_ascend_310p
from gpustack.utils.hub import (
    get_hf_text_config,
    get_max_model_len,
    get_pretrained_config,
)

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
    # Model deploy config
    #
    max_seq_len: int = 8192
    max_input_token_len: int = -1
    truncation: bool = False
    #
    # Model config
    #
    cpu_mem_size: int = 0
    npu_memory_fraction: float = 0.9
    trust_remote_code: bool = False
    #
    # Schedule config
    #
    cache_block_size: int = 128
    max_prefill_batch_size: int = 50
    prefill_time_ms_per_req: int = 150
    prefill_policy_type: int = 0
    max_batch_size: int = 200
    decode_time_ms_per_req: int = 50
    decode_policy_type: int = 0
    max_preempt_count: int = 0
    support_select_batch: bool = False
    max_queue_delay_microseconds: int = 5000
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
            "select the execution priority for this batch. "
            "Use `--no-support-select-batch` to disable explicitly.",
        )
        parser.add_argument(
            "--max-queue-delay-microseconds",
            type=int,
            help="Maximum microseconds of queue waiting.",
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
            help="Tokens size to batch for split fuse.",
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
            self.max_input_token_len = self.max_seq_len
        # Schedule config
        if self.max_preempt_count == 0 and self.cpu_mem_size > 0:
            self.cpu_mem_size = 0
        # Extends or Features
        # -- Parallelism
        if self.world_size > 0:
            if self.tensor_parallel_size < 0:
                self.tensor_parallel_size = self.world_size
            if self.moe_tensor_parallel_size < 0:
                self.moe_tensor_parallel_size = self.world_size
        else:
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
                    f"--rope-scaling must be a valid JSON string: {self.rope_scaling_parsed}"
                ) from e
        # -- Split fuse
        if self.enable_split:
            if not (512 <= self.split_chunk_tokens <= self.max_input_token_len):
                raise argparse.ArgumentTypeError(
                    "--split-chunk-tokens must be in the range [512, --max-input-token-len]"
                )
            if not (0 <= self.split_start_batch_size <= self.max_batch_size):
                raise argparse.ArgumentTypeError(
                    "--split-start-batch-size must be in the range [0, --max-batch-size]"
                )
        # -- Parallelism
        pp, tp, dp, sp, moe_tp, moe_ep, ws, local_ws = (
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.data_parallel_size,
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
                f"cannot be set at the same time, "
                f"which means enabling both pipeline and data parallelism."
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

    _inference_backend: str = "mindie"  # adapt the definition of runtime

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._workload_name: Optional[str] = None

    def start(self):
        # Start
        try:
            self._start()
        except Exception as e:
            self._report_error(e)
            raise e

    def _start(self):  # noqa: max-complexity=15
        """
        Start Ascend MindIE service in container.
        This method preserves all the original logic from AscendMindIEServer._start()
        but launches the service in a container instead of subprocess.
        """
        version = self._model.backend_version
        if not version:
            # Allow to control the version installed by user,
            # this relies on the environment setting.
            # There is a risk of failure, but flexible.
            # When error happens, specify a version to avoid this.
            version = "latest"

        minstance = self._model_instance
        dservers = minstance.distributed_servers
        subworkers = (
            dservers.subordinate_workers
            if dservers and dservers.subordinate_workers
            else []
        )
        subworker_pos = None
        for i, sw in enumerate(subworkers):
            if sw.worker_id == self._worker.id:
                subworker_pos = i
                break

        # Root path is defined by in Dockerfile ENV
        # https://github.com/gpustack/runner/blob/main/pack/cann/Dockerfile#L273
        root_path = Path("/usr/local/Ascend")

        install_path = root_path.joinpath("mindie", version, "mindie-service")

        # Load config,
        # the config includes two parts: environment variables and a JSON configuration file.
        logger.info("Loading Ascend MindIE config")

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
        logger.info("Mutating Ascend MindIE config")

        # - Global config
        # -- Pin installation path, which helps to locate other resources.
        env["MIES_INSTALL_PATH"] = str(install_path)
        # -- Enable exposing metircs.
        env["MIES_SERVICE_MONITOR_MODE"] = "1"
        # -- Enable high performance swapper.
        env["MIES_RECOMPUTE_THRESHOLD"] = "0.5"
        # env["MINDIE_LLM_USE_MB_SWAPPER"] = "1"  # Atlas 300I Duo needs to unset this.
        env["MINDIE_LLM_RECOMPUTE_THRESHOLD"] = "0.5"
        # -- Enforce continues batching.
        env["MINDIE_LLM_CONTINUOUS_BATCHING"] = "1"
        # -- Disable checking files permission.
        env["MINDIE_CHECK_INPUTFILES_PERMISSION"] = "0"
        # -- Enforce using ATB as backend
        env["MINDIE_LLM_FRAMEWORK_BACKEND"] = "ATB"
        # -- Enforce using 90% of GPU memory
        env["NPU_MEMORY_FRACTION"] = "0.9"
        # -- Disable OpenMP parallelism, speed up model loading.
        env["OMP_NUM_THREADS"] = env.pop("OMP_NUM_THREADS", "1")
        # -- Improve performance.
        env["MINDIE_ASYNC_SCHEDULING_ENABLE"] = "1"
        env["TASK_QUEUE_ENABLE"] = env.pop("TASK_QUEUE_ENABLE", "2")
        env["ATB_OPERATION_EXECUTE_ASYNC"] = "1"
        env["ATB_LAYER_INTERNAL_TENSOR_REUSE"] = env.pop(
            "ATB_LAYER_INTERNAL_TENSOR_REUSE", "1"
        )
        env["INF_NAN_MODE_ENABLE"] = env.pop("INF_NAN_MODE_ENABLE", "0")
        env["ATB_LLM_ENABLE_AUTO_TRANSPOSE"] = env.pop(
            "ATB_LLM_ENABLE_AUTO_TRANSPOSE", "0"
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
        if not subworkers:
            env.pop("MIES_CONTAINER_IP", "")
            env.pop("HOST_IP", "")

        # - Logging config
        # -- Ascend MindIE
        env["MINDIE_LOG_LEVEL"] = "INFO"
        env["MINDIE_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Service
        env["MIES_CERTS_LOG_LEVEL"] = env.pop("MIES_CERTS_LOG_LEVEL", "INFO")
        env["MIES_CERTS_LOG_TO_STDOUT"] = "1"
        env["MIES_CERTS_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE LLM
        env["MINDIE_LLM_LOG_LEVEL"] = env.pop("MINDIE_LLM_LOG_LEVEL", "WARN")
        env["MINDIE_LLM_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LLM_LOG_TO_FILE"] = "0"
        env["MINDIE_LLM_PYTHON_LOG_LEVEL"] = env.pop(
            "MINDIE_LLM_PYTHON_LOG_LEVEL", "WARN"
        )
        env["MINDIE_LLM_PYTHON_LOG_TO_STDOUT"] = "1"
        env["MINDIE_LLM_PYTHON_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Runtime
        env["ASCEND_GLOBAL_LOG_LEVEL"] = env.pop(
            "ASCEND_GLOBAL_LOG_LEVEL", "3"
        )  # 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
        env["ASCEND_SLOG_LEVEL"] = env.pop("ASCEND_SLOG_LEVEL", "WARN")
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
        env["ASCEND_SLOG_PRINT_TO_FILE"] = "0"
        env["MINDIE_RT_LOG_LEVEL"] = env.pop(
            "MINDIE_RT_LOG_LEVEL", "3"
        )  # 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
        env["MINDIE_RT_LOG_PRINT_TO_STDOUT"] = "1"
        env["MINDIE_RT_LOG_PRINT_TO_FILE"] = "0"
        # -- Ascend MindIE ATB
        env["ATB_LOG_LEVEL"] = env.pop("ATB_LOG_LEVEL", "ERROR")
        env["ATB_LOG_TO_STDOUT"] = "1"
        env["ATB_LOG_TO_FILE"] = "0"
        env["ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE"] = env.pop(
            "ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE", "0"
        )
        env["LOG_LEVEL"] = env.pop("LOG_LEVEL", "ERROR")
        env["LOG_TO_STDOUT"] = "1"
        env["LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Model
        env["ASDOPS_LOG_LEVEL"] = env.pop("ASDOPS_LOG_LEVEL", "ERROR")
        env["ASDOPS_LOG_TO_STDOUT"] = "1"
        env["ASDOPS_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE OCK
        env["OCK_LOG_LEVEL"] = env.pop("OCK_LOG_LEVEL", "ERROR")
        env["OCK_LOG_TO_STDOUT"] = "1"
        env["OCK_LOG_TO_FILE"] = "0"
        # -- Ascend MindIE Torch
        env["TORCH_AIE_LOG_LEVEL"] = env.pop(
            "TORCH_AIE_LOG_LEVEL", "3"
        )  # 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
        env["TORCH_AIE_PRINT_TO_STDOUT"] = "1"
        env["TORCH_AIE_PRINT_TO_FILE"] = "0"

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
        backend_config["npuDeviceIds"] = [minstance.gpu_indexes]
        model_config["worldSize"] = len(minstance.gpu_indexes)
        backend_config["multiNodesInferEnabled"] = False
        if subworkers:
            connecting_port = minstance.ports[1] if len(minstance.ports) > 1 else None
            backend_config["multiNodesInferEnabled"] = True
            backend_config["multiNodesInferPort"] = connecting_port
        if minstance.worker_id != self._worker.id:
            backend_config["npuDeviceIds"] = [subworkers[subworker_pos].gpu_indexes]
            model_config["worldSize"] = len(subworkers[subworker_pos].gpu_indexes)

        # - Model config
        derived_max_seq_len = self._get_model_max_seq_len()
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
        local_world_size = len(minstance.gpu_indexes)
        world_size = local_world_size
        if subworkers:
            world_size = local_world_size * (len(subworkers) + 1)
        params = AscendMindIEParameters(
            local_world_size=local_world_size,
            world_size=world_size,
            max_seq_len=max_seq_len,
        )
        # For Ascend 310P, we need to default dtype to float16.
        # As a workaround, we should allow users to override this with backend parameters.
        if is_ascend_310p(self._worker):
            original_params = self._model.backend_parameters or []
            self._model.backend_parameters = ["--dtype=float16"]
            self._model.backend_parameters.extend(original_params)
        if self._model.backend_parameters:
            logger.debug(
                f"Parsing given parameters: {os.linesep}{os.linesep.join(self._model.backend_parameters)}"
            )
            params.from_args(self._model.backend_parameters)

            # -- Log config
            log_config["logLevel"] = params.log_level
            env["MINDIE_LOG_LEVEL"] = params.log_level.upper()
            # -- Server config
            server_config["maxLinkNum"] = params.max_link_num
            # -- Model deploy config
            model_deploy_config["maxSeqLen"] = params.max_seq_len
            model_deploy_config["maxInputTokenLen"] = params.max_input_token_len
            schedule_config["maxIterTimes"] = params.max_seq_len
            schedule_config["maxPrefillTokens"] = params.max_seq_len
            model_deploy_config["truncation"] = params.truncation
            # -- Model config
            model_config["cpuMemSize"] = params.cpu_mem_size
            env["MIES_USE_MB_SWAPPER"] = "1" if params.cpu_mem_size > 0 else "0"
            env["NPU_MEMORY_FRACTION"] = str(params.npu_memory_fraction)
            model_config["trustRemoteCode"] = params.trust_remote_code
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
                if max_seq_len < derived_max_seq_len:
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
        rank_table_str = None
        if subworkers:
            server_count = f"{len(subworkers) + 1}"
            server_list = [
                {
                    "server_id": minstance.worker_ip,
                    "container_ip": minstance.worker_ip,
                    "device": [
                        {
                            "device_id": str(minstance.gpu_indexes[i]),
                            "device_ip": minstance.gpu_addresses[i],
                            "rank_id": str(i),
                        }
                        for i in range(len(minstance.gpu_indexes))
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
            rank_table_path = Path(self._model_path).joinpath("ranktable.json")
            rank_table_str = json.dumps(rank_table, indent=4, ensure_ascii=False)
            config_files.append(
                ContainerFile(
                    path=str(rank_table_path),
                    content=rank_table_str,
                    mode=0o640,
                )
            )
            # - Set environment variables.
            env["WORLD_SIZE"] = str(len(minstance.gpu_indexes) * (len(subworkers) + 1))
            env["RANKTABLEFILE"] = str(rank_table_path)
            env["RANK_TABLE_FILE"] = str(rank_table_path)
            env["MIES_CONTAINER_IP"] = env.pop("MIES_CONTAINER_IP", self._worker.ip)
            env["HOST_IP"] = env.pop("HOST_IP", self._worker.ip)
            env["ATB_LLM_HCCL_ENABLE"] = env.pop("ATB_LLM_HCCL_ENABLE", "1")
            env["ATB_LLM_COMM_BACKEND"] = env.pop("ATB_LLM_COMM_BACKEND", "hccl")
            env["HCCL_CONNECT_TIMEOUT"] = env.pop("HCCL_CONNECT_TIMEOUT", "7200")
            env["HCCL_EXEC_TIMEOUT"] = env.pop("HCCL_EXEC_TIMEOUT", "0")
            env["HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT"] = env.pop(
                "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT", "TRUE"
            )
            if env.get("CANN_CHIP", "310p") != "310p":
                env["HCCL_OP_EXPANSION_MODE"] = env.pop("HCCL_OP_EXPANSION_MODE", "AIV")
            # NB(thxCode): For deterministic calculation, needs the following environment variables.
            # LCCL_DETERMINISTIC=1
            # ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
            # HCCL_DETERMINISTIC=true
            # ATB_MATMUL_SHUFFLE_K_ENABLE=0
            # ATB_LLM_LCOC_ENABLE=0
            # HCCL_OP_EXPANSION_MODE=""
            logger.info(f"Saved Ascend MindIE rank table config to {rank_table_path}")

        # Generate JSON configuration file by model instance id.
        config_path = str(install_path.joinpath("conf", f"config-{minstance.id}.json"))
        config_str = json.dumps(config, indent=4, ensure_ascii=False)
        config_files.append(
            ContainerFile(
                path=config_path,
                content=config_str,
            ),
        )

        # Start, configure environment variable to indicate the JSON configuration file.
        env["MIES_CONFIG_JSON_PATH"] = str(config_path)
        service_path = root_path.joinpath("mindie", version, "mindie-service")
        service_bin_path = service_path.joinpath("bin", "mindieservice_daemon")

        try:
            # Display environment variables and JSON configuration.
            logger.info(
                f"Starting Ascend MindIE container: {self._model_instance.name}"
            )
            env_view = None
            if logger.isEnabledFor(logging.DEBUG):
                env_view = sanitize_env(env)
            elif self._model.env:
                # If the model instance has its own environment variables,
                # display the mutated environment variables.
                env_view = self._model.env
                for k, v in self._model.env.items():
                    env_view[k] = env.get(k, v)
            if env_view:
                logger.info(
                    f"With environment variables(inconsistent input items mean unchangeable):{os.linesep}"
                    f"{os.linesep.join(f'{k}={v}' for k, v in sorted(env_view.items()))}"
                )
            logger.info(
                f"With JSON configuration(inconsistent input items mean unchangeable):{os.linesep}{config_str}"
            )
            if rank_table_str:
                logger.info(
                    f"With rank table JSON configuration:{os.linesep}{rank_table_str}"
                )

            # Container startup - this replaces the subprocess.Popen call
            self._start_container(env, str(service_bin_path), config_files)

        except Exception as e:
            raise e

        finally:
            # Finally, remove JSON configuration file.
            Path(config_path).unlink(missing_ok=True)

    def _start_container(
        self, env: Dict[str, str], service_bin_path: str, files: List[ContainerFile]
    ):
        """
        Start the Ascend MindIE service in a container.
        This replaces the subprocess.Popen call from the original implementation.
        """
        # Setup container mounts
        mounts = self._get_configured_mounts()

        # Get resources configuration
        resources = self._get_configured_resources()

        image_name = self._get_backend_image_name(backend_type="cann")

        # Build command arguments
        arguments = service_bin_path.split()

        arguments = self.build_versioned_command_args(arguments)

        # Create container configuration
        run_container = Container(
            image=image_name,
            name=self._model_instance.name,
            profile=ContainerProfileEnum.RUN,
            execution=ContainerExecution(
                privileged=True,
                args=arguments,
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
            files=files,
            ports=[
                ContainerPort(
                    internal=self._get_serving_port(),
                ),
            ],
        )

        # Store workload name for management operations
        self._workload_name = self._model_instance.name

        workload_plan = WorkloadPlan(
            name=self._workload_name,
            host_network=True,
            containers=[run_container],
        )

        logger.info(f"Creating Ascend MindIE container workload: {self._workload_name}")
        logger.info(f"Container image name: {image_name} arguments: {arguments}")
        create_workload(workload_plan)

        logger.info(
            f"Ascend MindIE container workload {self._workload_name} created successfully"
        )

    def _report_error(self, ex: Exception):
        """
        Report error message to the model instance.
        """
        cause = getattr(ex, "__cause__", None)
        cause_text = f": {cause}" if cause else ""
        error_message = f"Failed to run Ascend MindIE: {ex}{cause_text}"
        logger.error(error_message, exc_info=True)
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

    def get_container_logs(
        self,
        tail: Optional[int] = 100,
        follow: bool = False,
        timestamps: bool = True,
        since: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Get container logs.

        Args:
            tail: Number of lines to show from the end of the logs
            follow: Whether to follow log output (stream new logs)
            timestamps: Whether to include timestamps in log output
            since: Show logs since timestamp (Unix timestamp)

        Returns:
            Iterator of log lines
        """
        if not self._workload_name:
            logger.warning(
                "No workload name available. Container may not be started yet."
            )
            return iter([])

        try:
            logger.info(
                f"Getting logs for Ascend MindIE container: {self._workload_name}"
            )
            return logs_workload(
                name=self._workload_name,
                tail=tail,
                follow=follow,
                timestamps=timestamps,
                since=since,
            )
        except Exception as e:
            logger.error(f"Failed to get container logs: {e}")
            return iter([])

    def get_container_status(self) -> Optional[WorkloadStatus]:
        """
        Get the current status of the container.

        Returns:
            WorkloadStatus object containing container information, or None if not found.
        """
        if not self._workload_name:
            logger.warning(
                "No workload name available. Container may not be started yet."
            )
            return None

        try:
            status = get_workload(self._workload_name)
            if status:
                logger.info(
                    f"Ascend MindIE container {self._workload_name} status: {status.state}"
                )
            else:
                logger.warning(
                    f"Ascend MindIE container {self._workload_name} not found"
                )
            return status
        except Exception as e:
            logger.error(f"Failed to get container status: {e}")
            return None

    def stop_container(self) -> bool:
        """
        Stop the container.

        Returns:
            True if container was stopped successfully, False otherwise.
        """
        if not self._workload_name:
            logger.warning(
                "No workload name available. Container may not be started yet."
            )
            return False

        try:
            logger.info(f"Stopping Ascend MindIE container: {self._workload_name}")
            result = delete_workload(self._workload_name)

            if result:
                logger.info(
                    f"Ascend MindIE container {self._workload_name} stopped successfully"
                )
                # Update model instance state
                try:
                    patch_dict = {
                        "state_message": "Container stopped by user request",
                        "state": ModelInstanceStateEnum.ERROR,
                    }
                    self._update_model_instance(self._model_instance.id, **patch_dict)
                except Exception as ue:
                    logger.error(f"Failed to update model instance state: {ue}")
                return True
            else:
                logger.warning(
                    f"Ascend MindIE container {self._workload_name} was not found or already stopped"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    def restart_container(self) -> bool:
        """
        Restart the container by stopping and starting it again.

        Returns:
            True if container was restarted successfully, False otherwise.
        """
        logger.info(f"Restarting Ascend MindIE container: {self._workload_name}")

        # Stop the container first
        if not self.stop_container():
            logger.error("Failed to stop container for restart")
            return False

        # Wait a moment for cleanup
        import time

        time.sleep(2)

        # Start the container again
        try:
            self.start()
            logger.info(
                f"Ascend MindIE container {self._workload_name} restarted successfully"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            return False

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
                    "cpuMemSize" : 5,
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
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
"""
        return json.loads(config_str)
