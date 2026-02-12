from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import enum
import logging
from typing import Any, Dict, List, Optional, Tuple
from gpustack.config import Config
from gpustack.policies.event_recorder.recorder import EventCollector, EventLevelEnum
from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    ComputedResourceClaim,
    Model,
    ModelInstance,
    ModelInstanceSubordinateWorker,
    is_omni_model,
)
from gpustack.schemas.workers import Worker
from gpustack.utils.hub import get_hf_text_config, get_max_model_len
from gpustack.scheduler.calculator import get_pretrained_config_with_workers
from gpustack.utils.gpu import (
    abbreviate_worker_gpu_indexes,
    group_gpu_ids_by_worker,
    group_gpu_indexes_by_gpu_type_and_worker,
)
from gpustack.policies.base import Allocatable, ModelInstanceScheduleCandidate
from gpustack.policies.utils import (
    ListMessageBuilder,
    get_computed_ram_claim,
    get_model_num_attention_heads,
    get_worker_allocatable_resource,
    get_worker_model_instances,
    sort_gpu_indexes_by_allocatable_rate,
    sort_selected_workers_by_gpu_type_and_resource,
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


@dataclass
class RequestEstimateUsage:
    ram: int
    vram: int


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
    vocab_size: Optional[int] = None
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

    is_multimodel: bool = False
    vision_config: Optional[Dict] = None

    def from_model_pretrained_config(  # noqa: C901
        self, model: Model, pretrained_config: Any
    ):
        """
        Parse the model's (hyper)parameters from the model.
        """
        if hasattr(pretrained_config, "vision_config"):
            self.is_multimodel = True
            self.vision_config = pretrained_config.vision_config

        # Get architectures first, it is not available in text_config.
        if hasattr(pretrained_config, "architectures"):
            self.architectures = pretrained_config.architectures

        pretrained_config = get_hf_text_config(pretrained_config)
        if pretrained_config is None and CategoryEnum.LLM in model.categories:
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
    # Model hyperparameters.
    _model_params: ModelParameters
    # Frequently used model parameter in selectors.
    _num_attention_heads: int
    # Number of GPUs required by the model.
    # Derived from GPU selectors(manual) or Parallelism parameters(auto).
    _gpu_count: int
    # Selected worker names if manual GPU selection is used.
    _selected_gpu_workers: Optional[List[str]]
    # Selected GPU indexes by gpu type and worker name if manual GPU selection is used.
    _selected_gpu_indexes_by_gpu_type_and_worker: Dict[str, Dict[str, List[int]]]
    # Worker VRAM totals by GPU indexs cache.
    _vram_totals_by_gpu_type_and_worker_and_gpu_idxs: Dict[
        str, Dict[str, Dict[int, int]]
    ]
    # Worker allocatable resource cache by gpu type.
    _workers_allocatable_resource_by_gpu_type: Dict[str, Dict[int, Allocatable]]

    def __init__(
        self,
        config: Config,
        model: Model,
        model_instances: List[ModelInstance],
    ):
        self._config = config
        self._model = model
        self._model_instances = model_instances
        self._model_params = ModelParameters()
        self._num_attention_heads = 0
        self._gpu_count = 0
        self._selected_gpu_workers = None
        self._selected_gpu_indexes_by_gpu_type_and_worker = {}
        self._vram_totals_by_gpu_type_and_worker_and_gpu_idxs = {}
        self._workers_allocatable_resource_by_gpu_type = {}

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

    async def _init_model_parameters(self, workers: List[Worker]):
        if is_omni_model(self._model):
            # Current model parameters are for llm-like models.
            return

        try:
            pretrained_config = await get_pretrained_config_with_workers(
                self._model,
                workers,
                trust_remote_code=True,
            )

            self._model_params.from_model_pretrained_config(
                self._model, pretrained_config
            )
            self._num_attention_heads = self._model_params.num_attention_heads
        except Exception as e:
            raise ValueError(
                f"Failed to parse model {self._model.name} hyperparameters: {e}"
            )

    def _set_gpu_count(
        self,
        world_size: Optional[int] = None,
        strategies: Optional[List[str]] = None,
    ):
        model = self._model
        if model.gpu_selector and model.gpu_selector.gpu_ids:
            gpu_indexes_by_gpu_type_and_worker = (
                group_gpu_indexes_by_gpu_type_and_worker(model.gpu_selector.gpu_ids)
            )
            gpu_ids_by_worker = group_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            self._selected_gpu_workers = list(gpu_ids_by_worker.keys())
            self._selected_gpu_indexes_by_gpu_type_and_worker = (
                gpu_indexes_by_gpu_type_and_worker
            )

            self._gpu_count = model.gpu_selector.gpus_per_replica or len(
                model.gpu_selector.gpu_ids
            )

        # When world_size is set.
        if world_size and world_size > 0:
            if self._gpu_count and self._gpu_count != world_size:
                # Both gpu selector and parallelism are set, validate they match.
                strategies_str = "/".join(strategies) if strategies else "parallelism"
                raise ValueError(
                    f"Model {model.name} has {strategies_str} set, but the selected gpu count ({self._gpu_count}) does not match the world size ({world_size})."
                )

            self._gpu_count = world_size

    def get_worker_allocatable_resource(
        self, worker: Worker, gpu_type: Optional[str] = None
    ) -> Allocatable:
        """
        Get the worker's allocatable resource, with caching for efficiency.
        Args:
            worker: Worker object
            gpu_type: Optional GPU type to filter
        Returns:
            Allocatable: The allocatable resource of the worker
        """

        allocatable = self._workers_allocatable_resource_by_gpu_type.get(
            gpu_type, {}
        ).get(worker.id)
        if allocatable is not None:
            return allocatable

        allocatable = get_worker_allocatable_resource(
            self._model_instances, worker, gpu_type
        )
        self._workers_allocatable_resource_by_gpu_type.setdefault(
            gpu_type, {}
        ).setdefault(worker.id, allocatable)
        return allocatable

    def _get_worker_vram_totals(
        self, worker: Worker, gpu_type: Optional[str] = None
    ) -> Dict[int, int]:
        """
        Get a mapping from GPU idxs to total VRAM for the given worker, filtered by gpu_type if provided.
        Uses cache for efficiency.

        Args:
            worker: Worker object

        Returns:
            Dict[int, int]: Mapping from GPU idxs to total VRAM (bytes)
        """
        vram_by_ids = self._vram_totals_by_gpu_type_and_worker_and_gpu_idxs.get(
            gpu_type, {}
        ).get(worker.name)
        if vram_by_ids is not None:
            return vram_by_ids

        if not worker.status or not worker.status.gpu_devices:
            return {}

        vram_by_idxs = {
            gpu.index: gpu.memory.total if gpu.memory and gpu.memory.total else 0
            for gpu in worker.status.gpu_devices
            if gpu.index is not None and (gpu_type is None or gpu.type == gpu_type)
        }
        self._vram_totals_by_gpu_type_and_worker_and_gpu_idxs.setdefault(gpu_type, {})[
            worker.name
        ] = vram_by_idxs
        return vram_by_idxs

    def _get_worker_resource_claim(
        self,
        worker: Worker,
        gpu_indexes: List[int],
        gpu_memory_utilization: float,
        request: Optional[RequestEstimateUsage] = None,
        gpu_type: Optional[str] = None,
    ) -> Dict[int, int]:
        """
        Given a worker and gpu indexes, get the vram claim according to gpu_memory_utilization.
        Returns a dictionary of gpu index to vram claim in bytes.
        """
        vram_claim: Dict[int, int] = {}
        for gpu in worker.status.gpu_devices:
            if gpu.index not in gpu_indexes or (
                gpu_type is not None and gpu.type != gpu_type
            ):
                continue

            claim = 0
            # LLMs
            if gpu_memory_utilization > 0:
                claim = int(gpu.memory.total * gpu_memory_utilization)
            # non LLMs
            else:
                if request is not None:
                    claim = int(request.vram / len(gpu_indexes))
                else:
                    claim = 0

            vram_claim[gpu.index] = claim
        return vram_claim

    def _get_worker_gpu_addresses(
        self,
        worker: Worker,
        gpu_indexes: List[int],
        gpu_type: Optional[str] = None,
    ) -> List[str]:
        """
        Given a worker and gpu indexes, get the gpu addresses according to gpu_indexes.
        Returns a list of GPU addresses.
        """
        gpu_addresses = []

        if self._model.backend != BackendEnum.ASCEND_MINDIE.value:
            return gpu_addresses

        for gpu in worker.status.gpu_devices:
            if gpu.index not in gpu_indexes and (
                gpu_type is not None and gpu.type != gpu_type
            ):
                continue

            addr = (
                gpu.network.inet
                if gpu.network and gpu.network.status == 'up' and gpu.network.inet
                else "-.-.-.-"
            )
            gpu_addresses.append(addr)
        return gpu_addresses

    def _generate_manual_selected_gpus_overcommit_message(  # noqa: C901
        self,
        candidate: ModelInstanceScheduleCandidate,
        workers: List[Worker],
        request: RequestEstimateUsage,
        gpu_memory_utilization: float,
    ) -> Tuple[bool, str, float]:
        """Check whether the candidate is overcommit and generate overcommit message.

        Args:
            candidate (ModelInstanceScheduleCandidate): The candidate to check.
            workers (List[Worker]): The list of workers.
            vram_claim (int): The model VRAM claim in bytes.
            gpu_memory_utilization (float): The required GPU memory utilization ratio.

        Returns:
            Tuple[bool, str, float]: A tuple of (is_overcommit, message, effective_vram).
        """

        if not self._model.gpu_selector or not self._model.gpu_selector.gpu_ids:
            return (False, "", 0)

        # 1. Build worker set for easy access.
        worker_set = {w.name: w for w in workers}

        # 2. Summarize the total VRAM and the number that meets the utilization for selected GPUs.
        #    Generate:
        #    - total_vram
        #    - satisfied_gpus_by_worker
        #    - satisfied_gpus_total_allocatable_vram_for_non_llms
        #    - selected_gpus_total_allocatable_vram_for_non_llms
        selected_gpu_count = len(self._model.gpu_selector.gpu_ids)
        satisfied_gpus_by_worker = {}
        satisfied_gpus_count = 0
        satisfied_gpus_total_allocatable_vram_for_non_llms = 0
        selected_gpus_total_allocatable_vram_for_non_llms = 0
        for (
            gpu_type,
            worker_gpu_indexes,
        ) in self._selected_gpu_indexes_by_gpu_type_and_worker.items():
            for wn, g in worker_gpu_indexes.items():
                w = worker_set.get(wn)
                if w is None:
                    continue

                wa = self.get_worker_allocatable_resource(w, gpu_type)
                vram_total_by_index = self._get_worker_vram_totals(w, gpu_type)
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
                        satisfied_gpus_by_worker.setdefault(gpu_type, {}).setdefault(
                            wn, {}
                        )[gpu_index] = True
                    else:  # LLMs
                        if total_vram is None or allocatable_vram is None:
                            continue

                        allocatable_gpu_memory_utilization = (
                            allocatable_vram / total_vram
                        )
                        if allocatable_gpu_memory_utilization >= gpu_memory_utilization:
                            satisfied_gpus_count += 1
                            satisfied_gpus_by_worker.setdefault(
                                gpu_type, {}
                            ).setdefault(wn, {})[gpu_index] = True

        # 3. Summarize the total VRAM and the number that meets the utilization for used GPUs.
        used_gpu_count = 0
        used_satisfied_count = 0
        used_gpus_total_allocatable_vram_for_non_llms = 0
        used_vram_claim_total = 0
        used_ram_claim_total = 0

        pairs = [
            (
                candidate.worker.name,
                candidate.gpu_type,
                candidate.gpu_indexes,
                candidate.computed_resource_claim,
            )
        ]
        if candidate.subordinate_workers:
            for sw in candidate.subordinate_workers:
                pairs.append(
                    (
                        sw.worker_name,
                        sw.gpu_type,
                        sw.gpu_indexes,
                        sw.computed_resource_claim,
                    )
                )

        for wn, gpu_type, indexes, claim in pairs:
            w = worker_set.get(wn)
            wa = self.get_worker_allocatable_resource(w, gpu_type)
            used_vram_claim_total += (
                sum(v for k, v in claim.vram.items() if claim.vram is not None) or 0
            )
            used_ram_claim_total += claim.ram or 0

            used_gpu_count += len(indexes)
            for idx in indexes:
                allocatable_vram = wa.vram.get(idx)
                used_gpus_total_allocatable_vram_for_non_llms += allocatable_vram or 0

                if (
                    satisfied_gpus_by_worker.get(gpu_type, {})
                    .get(wn, {})
                    .get(idx, None)
                    is not None
                ):
                    used_satisfied_count += 1

        # 4. Determine if overcommit occurred.
        overcommit = (
            used_vram_claim_total < request.vram
            or (request.ram > 0 and used_ram_claim_total < request.ram)
            or used_satisfied_count < used_gpu_count
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

    def _get_non_overcommit_and_best_overcommit_candidates(
        self,
        candidates: List[ModelInstanceScheduleCandidate],
        workers: List[Worker],
        request: RequestEstimateUsage,
        gpu_memory_utilization: Dict[str, float],
    ) -> Tuple[
        List[ModelInstanceScheduleCandidate],
        Optional[ModelInstanceScheduleCandidate],
        Optional[str],
    ]:
        """Separate non-overcommit candidates and find the best overcommit candidate.
        Args:
            candidates (List[ModelInstanceScheduleCandidate]): The list of candidates to check.
            workers (List[Worker]): The list of workers.
            request (RequestEstimateUsage): The estimated resource usage request.
            gpu_memory_utilization (Dict[str, float]): The required GPU memory utilization ratio by GPU type, key is the GPU type, * is for all types.
        Returns:
            Tuple[List[ModelInstanceScheduleCandidate], Optional[ModelInstanceScheduleCandidate], Optional[str]]:
                A tuple of (non_overcommit_candidates, best_overcommit_candidate, overcommit_message).
        """

        non_overcommits_candidates = []
        max_effective_vram = 0
        best_overcommit_candidate = None
        overcommit_msg = None
        for c in candidates:
            gpu_memory_utilization_for_type = gpu_memory_utilization.get(
                c.gpu_type,
                gpu_memory_utilization.get("*", 0),
            )
            overcommit, msg, effective_vram = (
                self._generate_manual_selected_gpus_overcommit_message(
                    c, workers, request, gpu_memory_utilization_for_type
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

    def _find_manual_gpu_selection_candidates(  # noqa: C901
        self,
        workers: List[Worker],
        gpu_memory_utilization: Dict[str, float],
        request: RequestEstimateUsage,
        event_collector: EventCollector,
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates for manual GPU selection based on user-specified GPU IDs.
        This function handles all manual GPU selection scenarios for vLLM.

        args:
            workers: List of available workers.
            gpu_memory_utilization: Required GPU memory utilization ratio by GPU type, key is the GPU type, * is for all types.
            request: The estimated resource usage request.
            event_collector: Event collector for logging events.
        """
        # Skip if no manual GPU selection is specified
        if not self._selected_gpu_workers:
            return []

        # Not allow heterogeneous gpu types
        if (
            self._gpu_count == len(self._model.gpu_selector.gpu_ids)
            and len(self._selected_gpu_indexes_by_gpu_type_and_worker.keys()) > 1
        ):
            event_collector.add(
                EventLevelEnum.ERROR,
                EVENT_ACTION_MANUAL_MULTI,
                str(
                    ListMessageBuilder(
                        "Deployment with heterogeneous GPU types is not supported, please select GPUs of the same type or update GPUs per replica.",
                    )
                ),
            )
            return []

        logger.debug(
            f"Manual GPU selection: workers={self._selected_gpu_workers}, "
            f"worker_count={len(self._selected_gpu_workers)}, "
            f"gpu_indexes_by_worker_and_type={self._selected_gpu_indexes_by_gpu_type_and_worker}"
        )

        candidates = []

        # Filter and sort selected workers by resource
        selected_workers_by_gpu_type = sort_selected_workers_by_gpu_type_and_resource(
            workers,
            self._selected_gpu_indexes_by_gpu_type_and_worker,
            self.get_worker_allocatable_resource,
        )

        # Handle single-worker single gpu scenarios
        if self._gpu_count == 1:
            for gpu_type, workers_of_type in selected_workers_by_gpu_type.items():
                gpu_memory_utilization_for_type = gpu_memory_utilization.get(
                    gpu_type,
                    gpu_memory_utilization.get("*", 0),
                )
                for worker in workers_of_type:
                    for gpu in worker.status.gpu_devices:
                        worker_candidates = (
                            self._manual_select_single_worker_multi_gpu_candidates(
                                worker,
                                [gpu.index],
                                gpu_memory_utilization_for_type,
                                request,
                                gpu_type,
                            )
                        )
                        candidates.extend(worker_candidates)

        # Handle single-worker multi-GPU and multi-worker scenarios
        elif self._gpu_count > 1:

            # Single-worker multi-GPU selection
            for gpu_type, workers_of_type in selected_workers_by_gpu_type.items():
                gpu_memory_utilization_for_type = gpu_memory_utilization.get(
                    gpu_type,
                    gpu_memory_utilization.get("*", 0),
                )
                for worker in workers_of_type:
                    selected_gpu_indexes = [
                        gpu.index for gpu in worker.status.gpu_devices
                    ]
                    if selected_gpu_indexes is None:
                        continue

                    worker_candidates = (
                        self._manual_select_single_worker_multi_gpu_candidates(
                            worker,
                            selected_gpu_indexes,
                            gpu_memory_utilization_for_type,
                            request,
                            gpu_type,
                        )
                    )
                    candidates.extend(worker_candidates)

            # Multi-worker multi-GPU selection
            if not candidates:
                for gpu_type, selected_workers in selected_workers_by_gpu_type.items():
                    gpu_memory_utilization_for_type = gpu_memory_utilization.get(
                        gpu_type,
                        gpu_memory_utilization.get("*", 0),
                    )
                    worker_candidates = (
                        self._manual_select_multi_worker_multi_gpu_candidates(
                            selected_workers,
                            gpu_memory_utilization_for_type,
                            request,
                            gpu_type,
                        )
                    )
                    candidates.extend(worker_candidates)

        if not candidates:
            return []

        # Separate non-overcommit and overcommit candidates
        non_overcommits_candidates, best_overcommit_candidate, overcommit_msg = (
            self._get_non_overcommit_and_best_overcommit_candidates(
                candidates, workers, request, gpu_memory_utilization
            )
        )

        # Return non-overcommit candidates if any
        if non_overcommits_candidates:
            return non_overcommits_candidates

        # Handle overcommit candidates
        if self._model.replicas > 1:
            event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_MANUAL_MULTI,
                str(
                    ListMessageBuilder(
                        f"Found {len(candidates) - len(non_overcommits_candidates)} candidate, manual scheduling for multi-replica model instances does not allow overcommit or heterogeneous deployment topologies.",
                    )
                ),
            )
            return []

        if best_overcommit_candidate:
            event_collector.add(
                EventLevelEnum.INFO, EVENT_ACTION_MANUAL_MULTI, overcommit_msg
            )
            return [best_overcommit_candidate]

        return []

    def _manual_select_single_worker_multi_gpu_candidates(
        self,
        worker: Worker,
        gpu_indexes: List[int],
        gpu_memory_utilization: float,
        request: RequestEstimateUsage,
        gpu_type: Optional[str] = None,
    ) -> List[ModelInstanceScheduleCandidate]:
        """Manually select multi GPU candidates."""

        # Early exit
        if len(gpu_indexes) < self._gpu_count:
            return []

        allocatable = self.get_worker_allocatable_resource(worker, gpu_type)
        vram_totals_by_gpu_idx = self._get_worker_vram_totals(worker, gpu_type)

        # Check if the GPU is satisfied the requirement
        satisfied_gpu_indexes = []
        unsatisfied_gpu_indexes = []
        for gpu_index in gpu_indexes:
            allocatable_vram = allocatable.vram.get(gpu_index, 0)
            total_vram = vram_totals_by_gpu_idx.get(gpu_index, 0)

            # LLMs
            if gpu_memory_utilization > 0:
                if (
                    total_vram > 0
                    and allocatable_vram / total_vram >= gpu_memory_utilization
                ):
                    satisfied_gpu_indexes.append(gpu_index)
                else:
                    unsatisfied_gpu_indexes.append(gpu_index)
            # non LLMs
            else:
                satisfied_gpu_indexes.append(gpu_index)

            if len(satisfied_gpu_indexes) >= self._gpu_count:
                break

        # Extend with unsatisfied gpu indexes if not enough satisfied gpus
        used_gpu_indexes = satisfied_gpu_indexes.copy()
        used_gpu_indexes.extend(
            unsatisfied_gpu_indexes[: self._gpu_count - len(satisfied_gpu_indexes)]
        )

        # Get vram claims for used gpus
        vram_claims = self._get_worker_resource_claim(
            worker, used_gpu_indexes, gpu_memory_utilization, request, gpu_type
        )
        return [
            ModelInstanceScheduleCandidate(
                worker=worker,
                gpu_type=gpu_type,
                gpu_indexes=used_gpu_indexes,
                gpu_addresses=self._get_worker_gpu_addresses(
                    worker, used_gpu_indexes, gpu_type
                ),
                computed_resource_claim=ComputedResourceClaim(
                    vram=vram_claims,
                    ram=get_computed_ram_claim(self._model, vram_claims, request.ram),
                    vram_utilization=gpu_memory_utilization,
                ),
            )
        ]

    def _manual_select_multi_worker_multi_gpu_candidates(
        self,
        workers: List[Worker],
        gpu_memory_utilization: float,
        request: RequestEstimateUsage,
        gpu_type: Optional[str] = None,
    ) -> List[ModelInstanceScheduleCandidate]:
        """Manual select multi worker multi GPU candidates."""
        if len(workers) < 2:
            return []

        # Main worker is the first one (with most GPUs)
        main_worker = workers[0]
        main_worker_name = main_worker.name
        main_gpu_indexes = self._selected_gpu_indexes_by_gpu_type_and_worker.get(
            gpu_type, {}
        ).get(main_worker_name, [])
        main_vram_claim = self._get_worker_resource_claim(
            main_worker, main_gpu_indexes, gpu_memory_utilization, request, gpu_type
        )

        # Handle subordinate workers
        subordinate_workers: List[ModelInstanceSubordinateWorker] = []
        for worker in workers:

            # Skip if the worker is not selected
            if (
                self._selected_gpu_indexes_by_gpu_type_and_worker.get(gpu_type, {}).get(
                    worker.name
                )
                is None
            ):
                continue
            if worker.name == main_worker_name:
                continue
            if not self._validate_distributed_limit_per_worker(worker):
                continue

            # Sort GPUs by allocatable rate
            gpu_indexes = self._selected_gpu_indexes_by_gpu_type_and_worker.get(
                gpu_type, {}
            ).get(worker.name, [])
            vram_allocatable = self._get_worker_resource_claim(
                worker, gpu_indexes, gpu_memory_utilization, request, gpu_type
            )
            sorted_gpu_indexes = sort_gpu_indexes_by_allocatable_rate(
                worker, vram_allocatable, gpu_type
            )

            # Calculate how many GPUs can be assigned to this subordinate worker
            current_gpu_count = len(main_gpu_indexes) + sum(
                len(sw.gpu_indexes) for sw in subordinate_workers
            )
            remaining_gpu_count = max(self._gpu_count - current_gpu_count, 0)
            assign_count = min(len(gpu_indexes), remaining_gpu_count)

            # Assign GPUs to the subordinate worker
            sw_gpu_indexes = sorted_gpu_indexes[:assign_count]
            vram_claim = self._get_worker_resource_claim(
                worker, sw_gpu_indexes, gpu_memory_utilization, request, gpu_type
            )

            subordinate_workers.append(
                ModelInstanceSubordinateWorker(
                    worker_id=worker.id,
                    worker_name=worker.name,
                    worker_ip=worker.ip,
                    worker_ifname=worker.ifname,
                    total_gpus=len(worker.status.gpu_devices),
                    gpu_type=gpu_type,
                    gpu_indexes=sw_gpu_indexes,
                    gpu_addresses=self._get_worker_gpu_addresses(
                        worker, sw_gpu_indexes
                    ),
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=get_computed_ram_claim(
                            self._model, vram_claim, request.ram
                        ),
                        vram_utilization=gpu_memory_utilization,
                    ),
                )
            )

            current_gpu_count = len(main_gpu_indexes) + sum(
                len(sw.gpu_indexes) for sw in subordinate_workers
            )
            if self._gpu_count and current_gpu_count >= self._gpu_count:
                break

        if not subordinate_workers:
            return []

        return [
            ModelInstanceScheduleCandidate(
                worker=main_worker,
                gpu_type=gpu_type,
                gpu_indexes=main_gpu_indexes,
                gpu_addresses=self._get_worker_gpu_addresses(
                    main_worker, main_gpu_indexes
                ),
                computed_resource_claim=ComputedResourceClaim(
                    vram=main_vram_claim,
                    ram=get_computed_ram_claim(
                        self._model, main_vram_claim, request.ram
                    ),
                ),
                subordinate_workers=subordinate_workers,
            )
        ]

    def _validate_distributed_limit_per_worker(self, worker: Worker) -> bool:
        """
        Validate that there is no more than one distributed vLLM instance per worker.
        """
        instances = get_worker_model_instances(self._model_instances, worker)
        for instance in instances:
            if (
                instance.distributed_servers
                and instance.distributed_servers.subordinate_workers
                and (
                    instance.model
                    and instance.model.backend
                    and instance.model.backend == self._model.backend
                )
            ):
                self._messages = [
                    str(
                        ListMessageBuilder(
                            f"Each worker can run only one distributed vLLM instance. Worker '{worker.name}' already has '{instance.name}'."
                        )
                    ),
                ]

                return False

        return True

    def _is_tp_size_divisible(self, tp_size: int) -> bool:
        """
        Check whether InferenceBackend's constraint of parameter divisibility is satisfied.
        1. num_attention_heads
        2. vocab_size

        Notes on `tp_size` (tensor parallel size) usage in auto scheduling:
        - Single-worker multi-GPU: `tp_size` is the number of GPUs currently selected
          in the traversal (i.e., `gpu_sum`). The scheduler grows the candidate set
          incrementally and validates divisibility at each step.
        - Multi-worker multi-GPU: selected workers are constrained to have the same
          number of GPUs, so `tp_size` equals the per-worker GPU count (i.e., `gpu_count`)
        """
        if not tp_size:
            return False
        if self._num_attention_heads and self._num_attention_heads % tp_size != 0:
            return False

        if (
            self._model_params.vocab_size
            and self._model_params.vocab_size % tp_size != 0
        ):
            return False

        return True

    def _check_tp_size_divisibility(
        self,
        tp_size: int,
    ) -> Optional[str]:
        """
        Check whether InferenceBackend's constraint of parameter divisibility is satisfied.
        1. num_attention_heads
        2. vocab_size

        Return:
            None if divisibility is satisfied, otherwise an error message.
        """
        if not tp_size:
            return None
        if self._num_attention_heads and self._num_attention_heads % tp_size != 0:
            return (
                f"Total number of attention heads ({self._num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tp_size})."
            )

        if (
            self._model_params.vocab_size
            and self._model_params.vocab_size % tp_size != 0
        ):
            return (
                f"Vocabulary size ({self._model_params.vocab_size})"
                " must be divisible by tensor parallel size "
                f"({tp_size})."
            )

        return None
