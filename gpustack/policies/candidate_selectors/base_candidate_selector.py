from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import enum
import logging
from sqlalchemy.ext.asyncio import AsyncEngine
from transformers import PretrainedConfig
from typing import Dict, List, Optional
from gpustack.config import Config
from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker
from gpustack.utils.convert import safe_int
from gpustack.utils.hub import (
    get_pretrained_config,
    get_hf_text_config,
    get_max_model_len,
)
from gpustack.utils.gpu import parse_gpu_id, parse_gpu_ids_by_worker
from gpustack.server.db import get_engine
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.utils import (
    get_model_num_attention_heads,
)

logger = logging.getLogger(__name__)


class ModelAttentionTypeEnum(enum.Enum):
    UNK = "unknown"
    MHA = "multi_head_attention"
    GQA = "grouped_query_attention"
    MQA = "multi_query_attention"
    MLA = "multi_head_latent_attention"


@dataclass
class ModelParameters:
    architectures: List[str] = None
    derived_max_seq_len: int = 0
    num_hidden_layers: int = 0
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: int = 1
    n_group: Optional[int] = None
    head_dim: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    torch_dtype: str = "bfloat16"
    quantize: Optional[str] = None
    quantization_config: Optional[Dict] = None
    moe_num_experts: Optional[int] = None
    moe_num_shared_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None

    def from_model(self, model: Model):  # noqa: C901
        """
        Parse the model's (hyper)parameters from the model.
        """

        pretrained_config = PretrainedConfig()
        try:
            pretrained_config = get_pretrained_config(model, trust_remote_code=True)
        except ValueError as e:
            if "architecture" in e.args[0] and model.backend_version:
                # In the AutoConfig.from_pretrained method, the architecture field in config undergoes validation.
                # For custom backend versions, exceptions caused by unrecognized architectures should be allowed
                # to prevent startup failures of valid new models with properly customized versions.
                pass
            else:
                raise e

        pretrained_config = get_hf_text_config(pretrained_config)
        if pretrained_config is None:
            # Exclude empty dict cases, as they indicate the locally-sourced model is not local to the server node.
            raise ValueError(f"Failed to get model {model.name} pretrained config")

        for attr_name in [attr.name for attr in fields(self.__class__)]:
            try:
                attr_value = getattr(pretrained_config, attr_name, None)
                if attr_value is not None:
                    setattr(self, attr_name, attr_value)
            except AttributeError:
                # If reach here, that means the field is an internal property,
                # which would not register in the argument parser.
                pass

        # Default
        self.derived_max_seq_len = get_max_model_len(pretrained_config)
        if not self.num_attention_heads:
            # For backward compatibility, try to get num_attention_heads from llm_config.
            self.num_attention_heads = get_model_num_attention_heads(pretrained_config)
        if not self.head_dim and self.hidden_size and self.num_attention_heads:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if not self.moe_num_experts:
            for key in [
                "n_routed_experts",
                "num_local_experts",
                "num_experts",
            ]:
                if value := getattr(pretrained_config, key, None):
                    setattr(self, "moe_num_experts", value)
                    break
        if self.moe_num_experts and not self.moe_num_shared_experts:
            for key in [
                "n_shared_experts",
                "num_shared_experts",
            ]:
                if value := getattr(pretrained_config, key, None):
                    setattr(self, "moe_num_shared_experts", value)
                    break

    def get_attention_type(self) -> ModelAttentionTypeEnum:
        """
        Get the attention type based on the hyperparameters.
        """

        if self.num_attention_heads:
            if self.num_key_value_heads == 1:
                return ModelAttentionTypeEnum.MQA
            elif (
                1 < self.num_key_value_heads < self.num_attention_heads
                and self.num_attention_heads % self.num_key_value_heads == 0
            ):
                return ModelAttentionTypeEnum.GQA
            elif self.num_key_value_heads == self.num_attention_heads:
                if self.q_lora_rank and self.kv_lora_rank:
                    return ModelAttentionTypeEnum.MLA
                return ModelAttentionTypeEnum.MHA
        return ModelAttentionTypeEnum.UNK


class ScheduleCandidatesSelector(ABC):
    # GPUStack configuration.
    _config: Config
    # Model to be scheduled.
    _model: Model
    # Database engine.
    _engine: AsyncEngine
    # Model hyperparameters.
    _model_params: ModelParameters
    # Frequently used model parameter in selectors.
    _num_attention_heads: int
    # Number of GPUs required by the model.
    # Derived from GPU selectors(manual) or Parallelism parameters(auto).
    _gpu_count: int
    # Selected worker names if manual GPU selection is used.
    _selected_gpu_workers: Optional[List[str]]
    # Number of selected workers if manual GPU selection is used.
    _selected_gpu_worker_count: int
    # Selected GPU indexes by worker name if manual GPU selection is used.
    _selected_gpu_indexes_by_worker: Dict[str, List[int]]

    def __init__(
        self,
        config: Config,
        model: Model,
        parse_model_params: bool = True,
    ):
        self._config = config
        self._model = model
        self._engine = get_engine()
        self._model_params = ModelParameters()
        self._num_attention_heads = 0
        self._gpu_count = 0
        self._selected_gpu_workers = None
        self._selected_gpu_worker_count = 0
        self._selected_gpu_indexes_by_worker = {}

        if parse_model_params:
            self._set_model_parameters()
            self._num_attention_heads = self._model_params.num_attention_heads

    @abstractmethod
    def get_messages(self) -> List[str]:
        """
        Get diagnostic messages from the selector.
        :return: A list of diagnostic messages.
        """
        pass

    @abstractmethod
    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates.
        :param workers: The list of workers to select from.
        :return: A list of schedule candidates.
        """
        pass

    def _set_model_parameters(self):
        try:
            self._model_params.from_model(self._model)
        except Exception as e:
            raise ValueError(
                f"Failed to parse model {self._model.name} hyperparameters: {e}"
            )

    def _set_gpu_count(
        self,
        tp: Optional[int] = None,
        pp: Optional[int] = None,
        dp: Optional[int] = None,
    ):
        model = self._model
        if model.gpu_selector and model.gpu_selector.gpu_ids:
            gpu_ids_by_worker = parse_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            self._selected_gpu_workers = list(gpu_ids_by_worker.keys())
            self._selected_gpu_worker_count = len(self._selected_gpu_workers)
            for worker_name, gpu_ids in gpu_ids_by_worker.items():
                gpu_indexes = []
                for gpu_id in gpu_ids:
                    valid, matched = parse_gpu_id(gpu_id)
                    if valid:
                        gpu_index = safe_int(matched.get("gpu_index"))
                        gpu_indexes.append(gpu_index)
                self._selected_gpu_indexes_by_worker[worker_name] = gpu_indexes

            self._gpu_count = model.gpu_selector.gpus_per_replica or len(
                model.gpu_selector.gpu_ids
            )

        # When tp/pp/dp is set, the gpu count is calculated by tp * pp * dp.
        if tp or pp or dp:
            world_size = 1
            strategies = []
            if tp:
                strategies.append("tp")
                world_size *= tp
            if pp:
                strategies.append("pp")
                world_size *= pp
            if dp:
                strategies.append("dp")
                world_size *= dp

            if self._gpu_count and self._gpu_count != world_size:
                # Both gpu selector and parallelism are set, validate they match.
                strategies_str = "/".join(strategies)
                raise ValueError(
                    f"Model {model.name} has {strategies_str} set, but the selected gpu count ({self._gpu_count}) does not match the world size ({world_size})."
                )

            self._gpu_count = world_size
