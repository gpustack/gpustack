from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import enum
import logging
from sqlalchemy.ext.asyncio import AsyncEngine
from transformers import PretrainedConfig
from typing import Dict, List, Optional, Tuple
from gpustack.config import Config
from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker
from gpustack.utils.convert import safe_int
from gpustack.utils.hub import (
    get_pretrained_config,
    get_hf_text_config,
    get_max_model_len,
)
from gpustack.utils.gpu import (
    abbreviate_worker_gpu_indexes,
    parse_gpu_id,
    parse_gpu_ids_by_worker,
)
from gpustack.server.db import get_engine
from gpustack.policies.base import Allocatable, ModelInstanceScheduleCandidate
from gpustack.policies.utils import (
    ListMessageBuilder,
    get_model_num_attention_heads,
    get_worker_allocatable_resource,
)
from gpustack.utils.unit import byte_to_gib

logger = logging.getLogger(__name__)

EVENT_ACTION_DEFAULT = "default_scheduling_msg"
EVENT_ACTION_MANUAL_MULTI = "manual_multi_gpu_scheduling_msg"
EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU = "auto_multi_worker_multi_gpu_scheduling_msg"
EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU = (
    "auto_single_worker_multi_gpu_scheduling_msg"
)
EVENT_ACTION_AUTO_SINGLE_GPU = "auto_single_gpu_scheduling_msg"


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
    # Worker allocatable resource cache.
    _workers_allocatable_resource: Dict[int, Allocatable] = {}
    # Worker VRAM totals by GPU indexs cache.
    _worker_vram_totals_by_gpu_idxs: Dict[str, Dict[int, int]] = {}

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

    async def _get_worker_allocatable_resource(self, worker: Worker):
        if worker.id in self._workers_allocatable_resource:
            return self._workers_allocatable_resource[worker.id]

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        self._workers_allocatable_resource[worker.id] = allocatable
        return allocatable

    def _get_worker_vram_totals_by_gpu_idxs(self, worker: Worker) -> Dict[int, int]:
        """
        Get a mapping from GPU idxs to total VRAM for the given worker.
        Uses cache for efficiency.

        Args:
            worker: Worker object

        Returns:
            Dict[int, int]: Mapping from GPU idxs to total VRAM (bytes)
        """
        self._worker_vram_totals_by_gpu_idxs = {}
        if worker.name in self._worker_vram_totals_by_gpu_idxs:
            return self._worker_vram_totals_by_gpu_idxs[worker.name]

        if not worker.status or not worker.status.gpu_devices:
            return {}

        vram_by_idxs = {
            gpu.index: gpu.memory.total if gpu.memory and gpu.memory.total else 0
            for gpu in worker.status.gpu_devices
            if gpu.index is not None
        }
        self._worker_vram_totals_by_gpu_idxs[worker.name] = vram_by_idxs
        return vram_by_idxs

    async def _get_worker_vram_claim(
        self,
        worker: Worker,
        gpu_indexes: List[int],
        gpu_memory_utilization: float,
    ) -> Dict[int, int]:
        """
        Given a worker and gpu indexes, get the vram claim according to gpu_memory_utilization.
        Returns a dictionary of gpu index to vram claim in bytes.
        """
        vram_claim: Dict[int, int] = {}
        for gpu in worker.status.gpu_devices:
            if gpu.index not in gpu_indexes:
                continue

            vram_claim[gpu.index] = int(gpu.memory.total * gpu_memory_utilization)
        return vram_claim

    async def _generate_manual_selected_gpus_overcommit_message(  # noqa: C901
        self,
        candidate: ModelInstanceScheduleCandidate,
        workers: List[Worker],
        vram_claim: int,
        gpu_memory_utilization: float,
        # event_collector: EventCollector,
    ) -> Tuple[bool, str, float]:
        """Check whether the candidate is overcommit and generate overcommit message.

        Args:
            candidate (ModelInstanceScheduleCandidate): The candidate to check.
            workers (List[Worker]): The list of workers.
            vram_claim (int): The model VRAM claim in bytes.
            gpu_memory_utilization (float): The required GPU memory utilization ratio.
            event_collector (EventCollector): The event collector to record messages.

        Returns:
            _type_: _description_
        """
        if not self._model.gpu_selector or not self._model.gpu_selector.gpu_ids:
            return False

        # 1. Build worker set for easy access.
        worker_set = {w.name: w for w in workers}

        # 2. Summarize the total VRAM and the number that meets the utilization for selected GPUs.
        selected_gpu_count = len(self._model.gpu_selector.gpu_ids)
        satisfied_gpus_by_worker = {}
        satisfied_gpus_count = 0
        satisfied_gpus_total_allocatable_vram_for_non_llms = 0
        selected_gpus_total_allocatable_vram_for_non_llms = 0
        for wn, g in self._selected_gpu_indexes_by_worker.items():
            w = worker_set.get(wn)
            if w is None:
                continue

            wa = await self._get_worker_allocatable_resource(w)
            vram_total_by_index = self._get_worker_vram_totals_by_gpu_idxs(w)
            if wa.vram is None:
                continue
            for gpu_index in g:
                total_vram = vram_total_by_index.get(gpu_index)
                allocatable_vram = wa.vram.get(gpu_index)
                selected_gpus_total_allocatable_vram_for_non_llms += (
                    allocatable_vram or 0
                )
                if gpu_memory_utilization == 0:  # non LLMs
                    satisfied_gpus_count += 1
                    satisfied_gpus_total_allocatable_vram_for_non_llms += (
                        allocatable_vram or 0
                    )
                    if wn not in satisfied_gpus_by_worker:
                        satisfied_gpus_by_worker[wn] = {}
                    satisfied_gpus_by_worker[wn][gpu_index] = True
                else:  # LLMs
                    if total_vram is None or allocatable_vram is None:
                        continue

                    allocatable_gpu_memory_utilization = allocatable_vram / total_vram
                    if allocatable_gpu_memory_utilization >= gpu_memory_utilization:
                        satisfied_gpus_count += 1
                        if wn not in satisfied_gpus_by_worker:
                            satisfied_gpus_by_worker[wn] = {}
                        satisfied_gpus_by_worker[wn][gpu_index] = True

        # 3. Summarize the total VRAM and the number that meets the utilization for used GPUs.
        used_gpu_count = 0
        used_satisfied_count = 0
        used_gpus_total_allocatable_vram_for_non_llms = 0
        used_vram_claim_total = 0

        pairs = [
            (
                candidate.worker.name,
                candidate.gpu_indexes,
                candidate.computed_resource_claim,
            )
        ]
        if candidate.subordinate_workers:
            for sw in candidate.subordinate_workers:
                pairs.append(
                    (sw.worker_name, sw.gpu_indexes, sw.computed_resource_claim)
                )

        for wn, indexes, claim in pairs:
            w = worker_set.get(wn)
            wa = await self._get_worker_allocatable_resource(w)
            used_vram_claim_total += (
                sum(v for k, v in claim.vram.items() if claim.vram is not None) or 0
            )
            used_gpu_count += len(indexes)
            for idx in indexes:
                allocatable_vram = wa.vram.get(idx)
                used_gpus_total_allocatable_vram_for_non_llms += allocatable_vram or 0

                if satisfied_gpus_by_worker.get(wn, {}).get(idx, None) is not None:
                    used_satisfied_count += 1

        # 4. Determine if overcommit occurred.
        overcommit = (
            used_vram_claim_total < vram_claim or used_satisfied_count < used_gpu_count
        )

        if not overcommit:
            return (False, "", 0)

        # 5. Build scheduling message.
        using_partial = used_gpu_count < selected_gpu_count
        used_devices_msg = abbreviate_worker_gpu_indexes(
            candidate.worker.name,
            candidate.gpu_indexes,
            len(candidate.subordinate_workers or []),
            (used_gpu_count - len(candidate.gpu_indexes)),
            8,
        )

        scheduling_msg = ListMessageBuilder(
            "Manual GPU selection resulted in resource overcommit."
        )
        effective_vram = 0
        if gpu_memory_utilization == 0:
            # non-LLM case
            if using_partial:
                effective_vram = used_gpus_total_allocatable_vram_for_non_llms
                scheduling_msg.extend(
                    [
                        f"Using {used_devices_msg} out of {selected_gpu_count} selected devices.",
                        f"Used GPUs provide {byte_to_gib(effective_vram)} GiB allocatable VRAM.",
                    ]
                )
            else:
                effective_vram = selected_gpus_total_allocatable_vram_for_non_llms
                scheduling_msg.append(
                    f"Selected GPUs have {byte_to_gib(effective_vram)} GiB of VRAM."
                )
        else:
            # LLM case
            if using_partial:
                effective_vram = (
                    used_gpus_total_allocatable_vram_for_non_llms
                    * gpu_memory_utilization
                )
                scheduling_msg.extend(
                    [
                        f"Using {used_devices_msg} out of {selected_gpu_count} selected devices.",
                        f"Used GPUs provide {byte_to_gib(used_gpus_total_allocatable_vram_for_non_llms):.2f} GiB allocatable VRAM, "
                        f"{used_satisfied_count}/{used_gpu_count} of GPUs meet the VRAM utilization ratio, "
                        f"providing {byte_to_gib(effective_vram):.2f} GiB of allocatable VRAM.",
                    ]
                )
            else:
                effective_vram = (
                    selected_gpus_total_allocatable_vram_for_non_llms
                    * gpu_memory_utilization
                )
                scheduling_msg.extend(
                    [
                        f"Selected GPUs have {byte_to_gib(selected_gpus_total_allocatable_vram_for_non_llms):.2f} GiB allocatable VRAM, "
                        f"{satisfied_gpus_count}/{selected_gpu_count} of GPUs meet the VRAM utilization ratio, "
                        f"providing {byte_to_gib(effective_vram):.2f} GiB of allocatable VRAM."
                    ]
                )

        return (True, str(scheduling_msg), effective_vram)

    async def _get_non_overcommit_and_best_overcommit_candidates(
        self,
        candidates: List[ModelInstanceScheduleCandidate],
        workers: List[Worker],
        vram_claim: int,
        gpu_memory_utilization: float,
    ) -> Tuple[
        List[ModelInstanceScheduleCandidate],
        Optional[ModelInstanceScheduleCandidate],
        Optional[str],
    ]:

        non_overcommits_candidates = []
        max_effective_vram = 0
        best_overcommit_candidate = None
        overcommit_msg = None
        for c in candidates:
            overcommit, msg, effective_vram = (
                await self._generate_manual_selected_gpus_overcommit_message(
                    c, workers, vram_claim, gpu_memory_utilization
                )
            )
            c.overcommit = overcommit

            if not overcommit:
                non_overcommits_candidates.append(c)
            elif effective_vram >= max_effective_vram:
                max_effective_vram = effective_vram
                best_overcommit_candidate = c
                overcommit_msg = msg

        if non_overcommits_candidates:
            return non_overcommits_candidates, None, None

        return non_overcommits_candidates, best_overcommit_candidate, overcommit_msg
