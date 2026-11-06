import json
from collections import defaultdict
from itertools import accumulate
import logging
from math import ceil
import re
from typing import Dict, List, Optional, Tuple
from gpustack.policies.resource_view import ResourceView
from gpustack.policies.lookup_table import estimate_offloaded_lookup_bytes
from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScheduleCandidate,
)
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
    EVENT_ACTION_AUTO_SINGLE_GPU,
    EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
    EVENT_ACTION_DEFAULT,
    EVENT_ACTION_MANUAL_MULTI,
    RequestEstimateUsage,
    ScheduleCandidatesSelector,
)
from gpustack.policies.event_recorder.recorder import EventCollector, EventLevelEnum
from gpustack.policies.utils import (
    ListMessageBuilder,
    estimate_model_vram,
    group_workers_by_gpu_type,
    get_model_ram_claim,
    get_computed_ram_claim,
    get_ram_claim_from_model_env,
    sort_workers_by_gpu_count,
)
from gpustack.schemas.models import (
    ModelInstance,
    CategoryEnum,
    ComputedResourceClaim,
    Model,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.workers import GPUDevicesStatus, Worker
from gpustack.config import Config
from gpustack.utils.command import (
    find_bool_parameter,
    find_parameter,
    find_int_parameter,
    resolve_executor_backend,
)
from gpustack.utils.unit import byte_to_gib
from gpustack.validation.pd.roles import validate_managed_pd_topology
from gpustack.utils.vllm_topology import (
    parse_user_parallelism,
    validate_multinode_topology,
    vllm_lb_mode,
    managed_lb_parallelism,
    manual_dp_enabled,
)

logger = logging.getLogger(__name__)

# Bound synchronous subset search across all GPU counts for one worker.
_MAX_GPU_SUBSET_SEARCH_STATES = 4096
_EVENT_ACTION_GPU_SUBSET_SEARCH_LIMIT = "gpu_subset_search_limit"
_EVENT_ACTION_RAM_NOT_ENOUGH = "ram_not_enough"


def parse_model_size_by_name(model_name: str) -> int:
    """
    Parse the model size from the model name.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model_name)
    if match:
        size_in_billions = float(match.group(1))
        return int(size_in_billions * 1e9)
    else:
        raise ValueError(f"Cannot parse model size from model name: {model_name}")


class VLLMResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        cfg: Config,
        model: Model,
        model_instances: List[ModelInstance],
        *,
        resource_view: ResourceView,
    ):
        super().__init__(cfg, model, model_instances, resource_view=resource_view)

        self._vram_claim = 0
        self._ram_claim = 0
        self._offloaded_ram = 0
        self._largest_single_gpu_vram = 0
        self._largest_single_gpu_vram_utilization = 0
        self._largest_multi_gpu_vram = 0
        self._largest_multi_gpu_total = 0
        self._largest_multi_gpu_utilization_satisfied_count = 0
        # What one member would actually reserve on a card it lands on, as
        # opposed to what the model's weights need. For an engine that takes a
        # fraction of the whole card (vLLM's `--gpu-memory-utilization`, SGLang's
        # `--mem-fraction-static`) the two differ by an order of magnitude: a
        # 0.6B model whose weights want 3.68 GiB still books 86% of a 48 GiB card.
        # Reporting the weights figure as "what this member costs" told an
        # operator their group needed 7 GiB on a cluster of 48 GiB cards, which
        # reads as room to spare.
        #
        # Recorded while scanning cards rather than taken off a candidate,
        # because the refusal path has no candidate -- that is the whole reason
        # it needs this.
        self._reserved_vram: int = 0

        self._messages = []
        self._event_collector = EventCollector(self._model, logger)
        self._workers_allocatable_resource: Dict[int, Allocatable] = {}
        self._worker_name_to_vram: Dict[str, Dict[int, int]] = {}

        self._unsatisfied_gpu_messages: Dict[str, List[int]] = {}

        world_size, strategies = (
            VLLMResourceFitSelector.get_world_size_from_backend_parameters(model)
        )
        self._set_gpu_count(world_size, strategies)

    async def _init_model_parameters(self, workers: List[Worker]):
        await super()._init_model_parameters(workers)
        self._validate_arguments()
        # GMU relies on architecture info in model parameters. Set it after model parameters are initialized.
        self._set_gpu_memory_utilization()

    def _should_check_vision_tp_divisibility(self) -> bool:
        if not self._model.backend_parameters:
            return True

        language_only = find_bool_parameter(
            self._model.backend_parameters, ["language-model-only"]
        )
        return not language_only

    @staticmethod
    def get_world_size_from_backend_parameters(
        model: Model,
    ) -> Tuple[Optional[int], Optional[List[str]]]:
        parallelism = parse_user_parallelism(model.backend_parameters, last_wins=True)
        tp = parallelism.tp
        pp = parallelism.pp
        dp = parallelism.dp
        dpl = parallelism.dpl
        pcp = parallelism.pcp

        # A non-positive value would silently yield a zero/negative world_size below.
        for name, value in (
            ("tensor-parallel-size", tp),
            ("pipeline-parallel-size", pp),
            ("prefill-context-parallel-size", pcp),
            ("data-parallel-size", dp),
            ("data-parallel-size-local", dpl),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"vLLM: --{name} {value} must be positive.")

        # Each parallel dimension multiplies the world size in a fixed order
        # (tp, pp, pcp, dp).
        # DP-Local is a strategy label only — its ranks live inside the dp
        # world, so it never multiplies.
        world_dimensions = (("tp", tp), ("pp", pp), ("pcp", pcp), ("dp", dp))
        if any(value for _, value in world_dimensions):
            world_size = 1
            strategies = []
            for name, value in world_dimensions:
                if value:
                    strategies.append(name)
                    world_size *= value

            if dpl:
                strategies.append("dpl")

            return world_size, strategies

        return None, None

    def _set_gpu_memory_utilization(self):
        self._gpu_memory_utilization = 0.9
        model = self._model
        if self._disable_gpu_memory_utilization():
            # gpu memory utilization is not used for non-LLM models
            self._gpu_memory_utilization = 0

        self._gpu_memory_utilization_parameter_name = "gpu-memory-utilization"
        gmu = find_parameter(
            model.backend_parameters, [self._gpu_memory_utilization_parameter_name]
        )
        if gmu:
            self._gpu_memory_utilization = float(gmu)

    def _disable_gpu_memory_utilization(self) -> bool:
        """
        Determine whether GPU memory utilization should be disabled. vLLM does not use --gpu-memory-utilization for non-LLM models
        like embedding and reranker, except for some specific models like Qwen3-Embedding and Qwen3-Reranker.

        Rules:
        1. For non-LLM models, GPU memory utilization is DISABLED (return True) unless they are in the exception list.
        2. Otherwise, GPU memory utilization is ENABLED (return False).
        """
        if not self._model.categories:
            return False

        architectures = self._model_params.architectures or []

        # Non-LLM models that vLLM still uses GPU memory utilization
        NON_LLM_GMU_EXCEPTIONS = {
            "Qwen3ForCausalLM",
            "Qwen3ForSequenceClassification",  # Qwen3-Embedding & Qwen3-Reranker
            "Qwen3VLForConditionalGeneration",  # Qwen3-VL-Embedding & Qwen3-VL-Reranker
            "Qwen3VLForSequenceClassification",
        }

        use_gmu_categories = [CategoryEnum.LLM, CategoryEnum.SPEECH_TO_TEXT]
        if any(cat in self._model.categories for cat in use_gmu_categories):
            return False

        # Disable for non-LLM models unless they are in the exception list
        return not any(arch in NON_LLM_GMU_EXCEPTIONS for arch in architectures)

    def _set_model_parameters(self):
        super()._set_model_parameters()

        # Get the architectures from hf-overrides. This helps make resource allocation
        # decisions for specific models like Qwen3-Embedding and Qwen3-Reranker.
        hf_overrides = find_parameter(self._model.backend_parameters, ["hf-overrides"])
        if hf_overrides:
            overrides_dict = json.loads(hf_overrides)
            if isinstance(overrides_dict, dict) and "architectures" in overrides_dict:
                self._model_params.architectures = overrides_dict["architectures"]

        self._num_attention_heads = self._model_params.num_attention_heads

    def _cal_effective_vram(self) -> float:
        if self._largest_multi_gpu_total == 0:
            return 0.0
        return (
            byte_to_gib(self._largest_multi_gpu_vram)
            * self._gpu_memory_utilization
            * self._largest_multi_gpu_utilization_satisfied_count
            / self._largest_multi_gpu_total
        )

    def _set_messages(self):
        if self._messages:
            return

        event_messages = {
            EVENT_ACTION_DEFAULT: "",
            EVENT_ACTION_MANUAL_MULTI: "",
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU: "",
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU: "",
            EVENT_ACTION_AUTO_SINGLE_GPU: "",
        }

        ram_messages = []
        for event in self._event_collector.events:
            if event.action == _EVENT_ACTION_RAM_NOT_ENOUGH:
                ram_messages.append(event.message)
            else:
                event_messages[event.action] = event.message

        messages = event_messages[EVENT_ACTION_DEFAULT] + "\n"
        if ram_messages:
            messages += str(ListMessageBuilder(ram_messages)) + "\n"
        for action in [
            EVENT_ACTION_MANUAL_MULTI,
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            EVENT_ACTION_AUTO_SINGLE_GPU,
        ]:
            if event_messages[action]:
                messages += event_messages[action]
                break

        if search_limit := event_messages.get(_EVENT_ACTION_GPU_SUBSET_SEARCH_LIMIT):
            messages += "\n" + search_limit

        self._messages.append(messages)

    def _add_message(self, message: str):
        self._messages.append(message)

    def get_messages(self) -> List[str]:
        return self._messages

    def _get_worker_vram(self, worker: Worker) -> Dict[int, int]:
        if worker.name in self._worker_name_to_vram:
            return self._worker_name_to_vram[worker.name]

        if worker.status is None or not worker.status.gpu_devices:
            return {}

        vram_total_by_index = {}
        for gpu in worker.status.gpu_devices:
            total = gpu.memory.total if gpu.memory else 0
            vram_total_by_index[gpu.index] = total

        self._worker_name_to_vram[worker.name] = vram_total_by_index
        return vram_total_by_index

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement.
        """

        # Initialize model parameters.
        await self._init_model_parameters(workers)

        offloaded_weights = estimate_offloaded_lookup_bytes(
            self._model, self._pretrained_config
        )
        estimate_kwargs = (
            {"offloaded_weights": offloaded_weights} if offloaded_weights else {}
        )
        self._vram_claim = await estimate_model_vram(
            self._model, self._config.huggingface_token, workers, **estimate_kwargs
        )
        # Keep the base table size separate from candidate PCP copies. Local DP
        # shares one table budget per PCP replica, without extra headroom.
        self._offloaded_ram = offloaded_weights
        self._ram_claim = get_model_ram_claim(self._model, self._offloaded_ram)
        logger.info(
            f"Calculated resource claim for model {self._model.readable_source}, "
            f"VRAM claim: {self._vram_claim}, RAM claim: {self._ram_claim}"
        )

        default_msg_list = ListMessageBuilder(
            f"The model requires approximately {byte_to_gib(self._vram_claim)} GiB of VRAM"
            f"{f' and {byte_to_gib(self._ram_claim)} GiB of RAM' if self._ram_claim > 0 else ''}."
        )
        if self._gpu_memory_utilization != 0:
            default_msg_list.append(
                f"With --{self._gpu_memory_utilization_parameter_name}={self._gpu_memory_utilization}, "
                f"all GPUs combined need to provide at least {byte_to_gib(int(self._vram_claim / self._gpu_memory_utilization))} GiB of total VRAM "
                f"and each GPU needs {int(self._gpu_memory_utilization * 100)}% of allocatable VRAM."
            )
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_DEFAULT,
            str(default_msg_list),
        )

        candidate_functions = [
            self.find_manual_gpu_selection_candidates,
            self.find_single_worker_single_gpu_full_offloading_candidates,
            self.find_single_worker_multi_gpu_full_offloading_candidates,
            self.find_multi_worker_multi_gpu_candidates,
        ]

        for candidate_func in candidate_functions:
            if self.should_skip_candidate_func(candidate_func):
                continue

            logger.debug(
                f"model {self._model.readable_source}, filter candidates with resource fit selector: {candidate_func.__name__}"
            )

            candidates = candidate_func(workers)
            if not manual_dp_enabled(self._model.env) and vllm_lb_mode(
                self._model.backend_parameters
            ) in (
                "external",
                "multi_port_external",
                "hybrid",
            ):
                candidates = self._validate_managed_candidates(candidates)
            for candidate in candidates:
                self._finalize_candidate_ram(candidate)

            if len(candidates) >= 1 and candidates[0].overcommit:
                # Manually selected candidate with overcommit. Also add the message.
                # It's useful for compatibility check.
                self._set_messages()

            if candidates:
                return candidates

        self._set_messages()
        return []

    def _validate_managed_candidates(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """Apply the complete-group layout to manual and automatic placements."""
        accepted = []
        for candidate in candidates:
            members = [candidate, *(candidate.subordinate_workers or [])]
            counts = [len(member.gpu_indexes or []) for member in members]
            try:
                topology = managed_lb_parallelism(
                    self._model.backend_parameters, counts
                )
                validate_managed_pd_topology(self._model, topology)
            except ValueError as error:
                self._event_collector.add(
                    EventLevelEnum.INFO, EVENT_ACTION_DEFAULT, str(error)
                )
                continue
            self._reserved_vram = max(
                self._reserved_vram,
                sum(
                    sum((member.computed_resource_claim.vram or {}).values())
                    for member in members
                ),
            )
            accepted.append(candidate)
        return accepted

    def should_skip_candidate_func(self, candidate_func) -> bool:
        # Skip conditions for manual GPU selection.
        if (
            self._selected_gpu_workers
            and candidate_func != self.find_manual_gpu_selection_candidates
        ):
            return True

        # Skip conditions for distributed inference.
        if (
            not self._model.distributed_inference_across_workers
            and candidate_func == self.find_multi_worker_multi_gpu_candidates
        ):
            return True

        return False

    def find_manual_gpu_selection_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        request = RequestEstimateUsage(
            ram=self._ram_claim,
            vram=self._vram_claim,
        )

        return self._find_manual_gpu_selection_candidates(
            workers,
            {"*": self._gpu_memory_utilization},
            request,
            self._event_collector,
        )

    def _compute_ram_claim(
        self,
        vram_claim: Dict[int, int],
        static_ram: Optional[int] = None,
        *,
        gpu_per_node: Optional[List[int]] = None,
        node_index: int = 0,
    ) -> Optional[int]:
        """Estimate one worker's table and LOCAL KV cache RAM in bytes.

        Local DP replicas are assumed to share table storage; the complete
        candidate layout determines independent PCP copies. An explicit RAM
        claim overrides the estimate. static_ram is accepted for the base
        interface but unused: table copies and ratio-based cache RAM depend on
        this worker's GPU allocation.

        gpu_per_node and node_index identify the worker in a complete candidate
        or projected homogeneous layout. Without them, estimate a single node.

        This is an approximate scheduling reservation without extra headroom,
        intended to avoid rejecting deployments solely on worst-case estimates.
        The DP sharing assumption does not enable sharing in the engine and
        does not cover architecture, version, Hybrid DP, or Elastic EP sharing
        conditions. Private DP copies and other engine allocations or loading
        peaks may exceed this budget; passing RAM admission does not guarantee
        sufficient runtime memory. Set GPUSTACK_MODEL_RAM_CLAIM to override it.
        """
        copies = (
            self._lookup_table_copies(gpu_per_node or [len(vram_claim)])[node_index]
            if self._offloaded_ram
            else 0
        )
        return get_computed_ram_claim(
            self._model, vram_claim, offloaded_ram=self._offloaded_ram * copies
        )

    def _lookup_table_copies(self, gpu_per_node: List[int]) -> List[int]:
        """Assume one shared DP table budget per local PCP replica.

        This scheduling convention avoids engine-specific DP sharing rules.
        PCP replicas retain independent TP-sharded tables; when PCP spans nodes,
        count only the replicas that can reside on each worker.
        """
        user = parse_user_parallelism(self._model.backend_parameters)
        pcp = user.pcp or 1
        if pcp == 1:
            return [1] * len(gpu_per_node)
        tp = user.tp or 1
        if (
            len(gpu_per_node) > 1
            and resolve_executor_backend(
                self._model.backend_parameters,
                self._model.backend_version,
                self._model.image_name,
            )
            == "mp"
        ):
            try:
                tp = validate_multinode_topology(gpu_per_node, user).tp
            except ValueError:
                if not self._selected_gpu_workers:
                    raise
                # Manual placements leave execution topology validation to the
                # backend. Keep explicit TP (or 1) for the approximate PCP budget.
        elif user.tp is None and user.pp is None and user.dp is None:
            # Match single-node defaults and the Ray worker's TP fallback.
            tp = gpu_per_node[0] if len(set(gpu_per_node)) == 1 else 1
        return [min(pcp, max(1, ceil(count / tp))) for count in gpu_per_node]

    def _auto_topology_valid(self, gpu_per_node: List[int]) -> bool:
        """Validate automatic allocations against the MP executor's topology rules."""
        if (
            resolve_executor_backend(
                self._model.backend_parameters,
                self._model.backend_version,
                self._model.image_name,
            )
            != "mp"
        ):
            return True
        try:
            validate_multinode_topology(
                gpu_per_node, parse_user_parallelism(self._model.backend_parameters)
            )
        except ValueError as e:
            logger.debug("Skipping automatic GPU allocation %s: %s", gpu_per_node, e)
            return False
        return True

    def _worker_ram_fits(
        self,
        worker: Worker,
        vram_claim: Dict[int, int],
        gpu_type: Optional[str],
        gpu_per_node: Optional[List[int]] = None,
    ) -> bool:
        """Check the full worker RAM claim and record a scheduling reason on failure."""
        message = self._worker_ram_overcommit_message(
            worker,
            self._compute_ram_claim(vram_claim, gpu_per_node=gpu_per_node),
            gpu_type,
        )
        if message:
            self._event_collector.add(
                EventLevelEnum.INFO, _EVENT_ACTION_RAM_NOT_ENOUGH, message
            )
        return not message

    def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with workers.
        """
        if self._gpu_count is not None and self._gpu_count > 1:
            # Skip multi-GPU selection
            return []

        candidates = []
        workers_of_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_of_type.items():
            for worker in workers_of_type:
                if not worker.status.gpu_devices:
                    continue

                result = self._find_single_worker_single_gpu_full_offloading_candidates(
                    worker,
                    gpu_type,
                )
                if result:
                    candidates.extend(result)

        return candidates

    def _find_single_worker_single_gpu_full_offloading_candidates(
        self,
        worker: Worker,
        gpu_type: Optional[str] = None,
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with worker.
        requires: worker.status.gpu_devices is not None
        """

        candidates = []

        allocatable = self.get_worker_allocatable_resource(worker, gpu_type)

        if not worker.status.gpu_devices:
            return []

        for _, gpu in enumerate(worker.status.gpu_devices):

            gpu_index = gpu.index
            allocatable_vram = allocatable.vram.get(gpu_index, 0)
            allocatable_gpu_memory_utilization = allocatable_vram / gpu.memory.total

            if allocatable_vram > self._largest_single_gpu_vram:
                self._largest_single_gpu_vram = allocatable_vram
                self._largest_single_gpu_vram_utilization = (
                    allocatable_gpu_memory_utilization
                )

            if gpu.memory is None or gpu.memory.total == 0:
                continue

            if self._gpu_memory_utilization > 0:
                self._reserved_vram = max(
                    self._reserved_vram,
                    int(gpu.memory.total * self._gpu_memory_utilization),
                )

            exceeds_vram = (
                self._vram_claim > gpu.memory.total * self._gpu_memory_utilization
                if self._gpu_memory_utilization > 0  # LLMs
                else self._vram_claim > allocatable_vram  # non LLMs
            )
            exceeds_memory_utilization = (
                self._gpu_memory_utilization > 0
                and allocatable_gpu_memory_utilization < self._gpu_memory_utilization
            )
            if exceeds_vram or exceeds_memory_utilization:
                continue

            vram_claim_bytes = (
                int(gpu.memory.total * self._gpu_memory_utilization)
                if self._gpu_memory_utilization > 0  # LLMs
                else int(self._vram_claim)  # non LLMs
            )

            vram_claim = {gpu_index: vram_claim_bytes}
            if not self._worker_ram_fits(worker, vram_claim, gpu_type):
                continue
            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=[gpu_index],
                    gpu_type=gpu.type,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=self._compute_ram_claim(vram_claim),
                    ),
                )
            )

        if not candidates or (len(candidates) == 1 and candidates[0].overcommit):
            event_msg = f"The current available GPU only has {byte_to_gib(self._largest_single_gpu_vram)} GiB allocatable VRAM."
            if self._gpu_memory_utilization != 0:
                event_msg = (
                    event_msg.rstrip(".")
                    + f" ({(self._largest_single_gpu_vram_utilization * 100):.2f}%)."
                )
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_AUTO_SINGLE_GPU,
                str(ListMessageBuilder(event_msg)),
            )

        return candidates

    def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        if self._gpu_count == 1:
            return []

        candidates = []
        workers_of_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_of_type.items():
            for worker in workers_of_type:
                if not worker.status.gpu_devices:
                    continue

                result = self._find_single_worker_multi_gpu_full_offloading_candidates(
                    worker, gpu_type
                )
                if result:
                    candidates.extend(result)

        if not candidates:
            return []

        min_gpu_count = min(len(candidate.gpu_indexes) for candidate in candidates)
        final_candidates = [
            candidate
            for candidate in candidates
            if len(candidate.gpu_indexes) == min_gpu_count
        ]
        return final_candidates

    @staticmethod
    def _find_resource_fit_gpu_subset(
        groups: List[List[int]],
        claims: Dict[int, int],
        gpu_count: int,
        vram_range: Tuple[int, int],
        search_budget: int,
    ) -> Tuple[Optional[List[int]], int]:
        """Find a subset within joint resource bounds and return states visited.

        Equal-claim GPUs are interchangeable. Suffix sums bound every partial
        allocation by its cheapest and most expensive possible completion.
        The iterative search preserves group preference and stops at its budget;
        None means no fit was found, including when the budget was exhausted.
        """
        suffix_sums = [[0] for _ in range(len(groups) + 1)]
        values = []
        for position in range(len(groups) - 1, -1, -1):
            values.extend(claims[index] for index in groups[position])
            values.sort()
            suffix_sums[position] = list(accumulate(values, initial=0))

        lower, upper = vram_range
        stack = [(0, gpu_count, 0, [])]
        visited = 0
        while stack and visited < search_budget:
            position, remaining, total, indexes = stack.pop()
            visited += 1
            sums = suffix_sums[position]
            if remaining >= len(sums):
                continue
            if (
                total + sums[remaining] > upper
                or total + sums[-1] - sums[-remaining - 1] < lower
            ):
                continue
            if remaining == 0:
                return indexes, visited
            group = groups[position]
            for count in range(min(len(group), remaining) + 1):
                stack.append(
                    (
                        position + 1,
                        remaining - count,
                        total + claims[group[0]] * count,
                        indexes + group[:count],
                    )
                )
        return None, visited

    def _single_worker_vram_bounds(
        self,
        worker: Worker,
        claims: Dict[int, int],
        count: int,
        gpu_type: Optional[str],
    ) -> Optional[Tuple[int, int]]:
        """Find the feasible VRAM sum interval for a fixed GPU count.

        RAM is monotonic in total VRAM at a fixed topology. Synthetic claims
        preserve the GPU count while varying the sum, so table copies, explicit
        overrides, cache sizes, and ratio rounding use the normal RAM estimator.
        """
        indexes = sorted(claims, key=claims.get)
        sample = {index: claims[index] for index in indexes[:count]}
        lower = max(ceil(self._vram_claim), sum(sample.values()))
        upper = sum(claims[index] for index in indexes[-count:])
        if lower > upper:
            return None
        first = indexes[0]
        other_vram = sum(sample.values()) - sample[first]
        sample[first] = lower - other_vram
        if not self._worker_ram_fits(worker, sample, gpu_type):
            return None

        available_ram = self.get_worker_allocatable_resource(worker, gpu_type).ram

        def ram_fits(total: int) -> bool:
            sample[first] = total - other_vram
            return (self._compute_ram_claim(sample) or 0) <= available_ram

        if ram_fits(upper):
            return lower, upper
        left, right = lower, upper
        while left < right:
            middle = (left + right + 1) // 2
            if ram_fits(middle):
                left = middle
            else:
                right = middle - 1
        return lower, left

    def _find_single_worker_gpu_claim(
        self,
        worker: Worker,
        gpu_devices: GPUDevicesStatus,
        allocatable: Allocatable,
        gpu_type: Optional[str],
    ) -> Dict[int, int]:
        """Find an automatic multi-GPU allocation satisfying both VRAM and RAM.

        Input GPUs must already satisfy per-device utilization requirements.
        Prefer fewer GPUs unless a count is specified; prune impossible counts
        and equivalent subsets. Return per-GPU VRAM bytes, or an empty mapping
        when no fit is found within the per-worker search budget.
        """
        claims = {
            gpu.index: (
                int(gpu.memory.total * self._gpu_memory_utilization)
                if self._gpu_memory_utilization > 0
                else allocatable.vram.get(gpu.index, 0)
            )
            for gpu in gpu_devices
        }
        groups = {}
        for index, claim in claims.items():
            groups.setdefault(claim, []).append(index)
        counts = [self._gpu_count] if self._gpu_count else range(2, len(claims) + 1)
        remaining_budget = _MAX_GPU_SUBSET_SEARCH_STATES
        for count in counts:
            if count > len(claims) or not self._is_tp_size_divisible(count):
                continue
            bounds = self._single_worker_vram_bounds(worker, claims, count, gpu_type)
            if bounds is None:
                continue
            indexes, visited = self._find_resource_fit_gpu_subset(
                list(groups.values()), claims, count, bounds, remaining_budget
            )
            remaining_budget -= visited
            if indexes is not None:
                return {index: claims[index] for index in indexes}
            if remaining_budget == 0:
                self._event_collector.add(
                    EventLevelEnum.INFO,
                    _EVENT_ACTION_GPU_SUBSET_SEARCH_LIMIT,
                    f"Automatic GPU selection on {worker.name} reached its search limit. "
                    "Select GPUs manually to use a known feasible combination.",
                )
                break
        return {}

    def _find_single_worker_multi_gpu_full_offloading_candidates(  # noqa: C901
        self, worker: Worker, gpu_type: Optional[str] = None
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi gpu full offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """

        total_gpu = len(worker.status.gpu_devices)
        if total_gpu < 2:
            return None

        allocatable = self.get_worker_allocatable_resource(worker, gpu_type)

        gpu_list = []
        total_allocatable_vram = 0
        satisfied_gpu_count = 0

        for gpu in worker.status.gpu_devices:
            if gpu.memory is None or gpu.memory.total is None:
                continue

            allocatable_vram = allocatable.vram.get(gpu.index, 0)
            total_allocatable_vram += allocatable_vram

            if allocatable_vram / gpu.memory.total > self._gpu_memory_utilization:
                satisfied_gpu_count += 1
                gpu_list.append(gpu)

        if total_allocatable_vram > self._largest_multi_gpu_vram:
            self._largest_multi_gpu_vram = total_allocatable_vram
            self._largest_multi_gpu_utilization_satisfied_count = satisfied_gpu_count
            self._largest_multi_gpu_total = len(worker.status.gpu_devices)

        # Sort by vram in descending order
        sorted_gpu_devices: GPUDevicesStatus = sorted(
            gpu_list,
            key=lambda gpu: allocatable.vram.get(gpu.index, 0),
            reverse=True,
        )

        vram_claim = self._find_single_worker_gpu_claim(
            worker, sorted_gpu_devices, allocatable, gpu_type
        )
        if vram_claim:
            vram_sum = sum(vram_claim.values())
            gpu_indexes = list(vram_claim)
            # The reservation this member would hold across all of its cards,
            # recorded for the same reason the single-GPU scan records its own:
            # `get_resource_claim` quotes it in a group's refusal, and without
            # it a multi-GPU member falls back to the weights estimate — which
            # is materially smaller than what the engine seizes, on the one
            # path where the difference is multiplied by the card count.
            self._reserved_vram = max(self._reserved_vram, vram_sum)
            return [
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_type=gpu_type,
                    gpu_indexes=gpu_indexes,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=self._compute_ram_claim(vram_claim),
                    ),
                )
            ]
        event_msg_list = []
        if msg := self._check_tp_size_divisibility(
            self._largest_multi_gpu_utilization_satisfied_count
        ):
            event_msg_list.append(msg)
        # Nothing was measured: no worker offered a GPU set this branch
        # could size, so the counters are still at their initial zero.
        # Printing them says "this cluster has 0 of 0 GPUs and 0.00 GiB",
        # which reads as an empty fleet and contradicts the filter lines
        # above it ("Matched 1/2 workers by READY status"). An absence of
        # measurement has to say so.
        if self._largest_multi_gpu_total == 0:
            event_msg = (
                "No worker offered a GPU set that could be sized for this " "model."
            )
        else:
            event_msg = f"The largest available worker has {byte_to_gib(self._largest_multi_gpu_vram)} GiB allocatable VRAM."
            if self._gpu_memory_utilization != 0:
                event_msg = (
                    event_msg.rstrip(".")
                    + f", {self._largest_multi_gpu_utilization_satisfied_count}/{self._largest_multi_gpu_total} of GPUs meet the VRAM utilization ratio, providing {self._cal_effective_vram():.2f} GiB of allocatable VRAM."
                )
        event_msg_list.append(event_msg)

        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            str(ListMessageBuilder(event_msg_list)),
        )

        return []

    def find_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        candidates = []
        workers_of_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_of_type.items():
            result = self.auto_select_multi_worker_multi_gpu_candidates(
                workers_of_type, gpu_type
            )
            if result:
                candidates.extend(result)

        return candidates

    def auto_select_multi_worker_multi_gpu_candidates(  # noqa: C901
        self, workers: List[Worker], gpu_type: Optional[str] = None
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Auto select multi worker multi gpu candidates.
        Currently, a candidate should match the following conditions:
        1. Workers in the candidate have the same number of GPUs.
        2. All GPUs in the worker satisfy the gpu_memory_utilization requirement.
        3. TP size can be divided by the number of attention heads.
        4. The total VRAM claim is greater than the estimated VRAM claim.
        5. If gpu_count is set via parallelism, the total GPU count should be equal to gpu_count.
        """

        if not workers or len(workers) < 2:
            return []

        sort_workers_by_gpu_count(workers)

        workers_by_gpu_count_dict = defaultdict(list)
        for worker in workers:
            if not worker.status or not worker.status.gpu_devices:
                continue

            workers_by_gpu_count_dict[len(worker.status.gpu_devices)].append(worker)

        # Store the optimal combination info to show
        workers_combination: List[Worker] = []
        largest_vram = 0
        worker_count = 0
        device_count_per_worker = 0

        event_message = ListMessageBuilder([])
        seen_skip_reasons: set[str] = set()

        # Loop through worker groups with the same number of GPUs.
        for gpu_count, worker_group in workers_by_gpu_count_dict.items():
            if len(worker_group) < 2:
                continue

            if not self._is_tp_size_divisible(gpu_count):
                continue

            if self._gpu_count and self._gpu_count % gpu_count:
                continue
            # Automatic multi-node candidates use equal GPU counts. This gives
            # each worker the complete layout needed for RAM admission before
            # selecting it, so an insufficient worker cannot hide later fits.
            projected_layout = [gpu_count] * (
                self._gpu_count // gpu_count if self._gpu_count else 2
            )
            if not self._auto_topology_valid(projected_layout):
                continue

            selected_workers: List[Worker] = []
            gpu_sum = 0
            vram_sum = 0
            for worker in worker_group:
                allocatable = self.get_worker_allocatable_resource(worker, gpu_type)

                if any(
                    gpu.memory is None
                    or gpu.memory.total is None
                    or (
                        allocatable.vram.get(gpu.index, 0) / gpu.memory.total
                        < self._gpu_memory_utilization
                    )
                    for gpu in worker.status.gpu_devices
                ):
                    # Skip the worker if any GPU does not satisfy the gpu_memory_utilization requirement.
                    continue
                vram_claim = {
                    gpu.index: int(gpu.memory.total * self._gpu_memory_utilization)
                    for gpu in worker.status.gpu_devices
                }
                if not self._worker_ram_fits(
                    worker, vram_claim, gpu_type, projected_layout
                ):
                    continue
                selected_workers.append(worker)
                gpu_sum += gpu_count
                vram_sum += sum(
                    int(gpu.memory.total * (self._gpu_memory_utilization or 1))
                    for gpu in worker.status.gpu_devices
                )

                if self._gpu_count:
                    # Parallelism is set. Proceed until we match the exact GPU count.
                    if gpu_sum < self._gpu_count:
                        continue
                    elif gpu_sum > self._gpu_count:
                        break

                if vram_sum >= self._vram_claim:
                    candidate, skip_reason = _create_candidate(
                        self._model,
                        selected_workers,
                        self._gpu_memory_utilization,
                    )
                    if candidate is None:
                        if skip_reason and skip_reason not in seen_skip_reasons:
                            seen_skip_reasons.add(skip_reason)
                            event_message.append(skip_reason)
                        continue
                    return [candidate]
            if vram_sum > largest_vram:
                workers_combination = selected_workers
                largest_vram = vram_sum
                worker_count = len(worker_group)
                device_count_per_worker = gpu_count

        # Nothing can be return, construct scheduling message
        if self._gpu_memory_utilization == 0:
            event_message.append(
                f"The largest available worker has {byte_to_gib(largest_vram)} GiB of VRAM."
            )
        elif workers_combination:
            worker_names = [worker.name for worker in workers_combination]
            worker_names_msg = (
                str(worker_names[:3]).rstrip("]")
                + f"...(more {len(worker_names) - 3})]"
                if len(worker_names) > 3
                else str(worker_names)
            )
            message = f"The optimal combination {worker_names_msg} provides {byte_to_gib(largest_vram)} GiB of allocatable VRAM."
            if worker_count - len(workers_combination) > 0:
                message += f" There are {worker_count - len(workers_combination)} {'workers' if worker_count - len(workers_combination) > 1 else 'worker'} that can provide {device_count_per_worker} {'GPUs' if device_count_per_worker > 1 else 'GPU'}, as the workers in the combination, but some GPUs among them fail to meet requirements."
            event_message.append(message)

        event_message.append(
            "Cannot find a suitable worker combination to run the model in distributed mode. "
            "If you are confident that the resources are sufficient, you may manually schedule the model by selecting the workers and GPUs."
        )
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
            str(event_message),
        )

        return []

    def _finalize_candidate_ram(
        self, candidate: ModelInstanceScheduleCandidate
    ) -> None:
        """Record topology-aware RAM before admission or overcommit classification."""
        if (
            not self._offloaded_ram
            and get_ram_claim_from_model_env(self._model) is None
        ):
            return
        claims = [candidate.computed_resource_claim]
        claims.extend(
            worker.computed_resource_claim
            for worker in candidate.subordinate_workers or []
        )
        gpu_per_node = [len(claim.vram) for claim in claims]
        for index, claim in enumerate(claims):
            claim.ram = self._compute_ram_claim(
                claim.vram, gpu_per_node=gpu_per_node, node_index=index
            )

    def _validate_arguments(self):
        tp = find_int_parameter(
            self._model.backend_parameters, ["tensor-parallel-size", "tp"]
        )
        if msg := self._check_tp_size_divisibility(tp):
            raise ValueError(
                msg + " Consider adjusting your tensor-parallel-size value."
            )

        if (
            resolve_executor_backend(
                self._model.backend_parameters,
                self._model.backend_version,
                self._model.image_name,
            )
            == "mp"
        ):
            self._validate_mp_multinode_arguments()

    def _validate_mp_multinode_arguments(self):
        """
        Sanity-check user-provided MP multi-node parallelism arguments
        before worker selection. Cross-node-topology invariants
        (``workers_per_dp`` fits the cluster, homogeneous cross-node layout,
        etc.) are enforced later in :func:`cal_multinode_topology` once the
        deployment metadata is known; here we only catch contradictions
        detectable from backend_parameters alone.
        """
        dp = find_int_parameter(
            self._model.backend_parameters, ["data-parallel-size", "dp"]
        )
        dpl = find_int_parameter(
            self._model.backend_parameters, ["data-parallel-size-local", "dpl"]
        )

        # Positivity of dp/dpl is already enforced at construction time by
        # get_world_size_from_backend_parameters (all backends), so here we only
        # check the cross-argument relationship.
        if dp is not None and dpl is not None and dp % dpl != 0:
            raise ValueError(
                f"vLLM multi-node: --data-parallel-size {dp} must be a multiple "
                f"of --data-parallel-size-local {dpl}."
            )


def _create_candidate(
    model: Model,
    selected_workers: List[Worker],
    gpu_memory_utilization: float = 0.9,
) -> Tuple[Optional[ModelInstanceScheduleCandidate], Optional[str]]:
    """
    Create a candidate with all GPUs from the selected workers.

    Returns ``(None, reason)`` when the worker combination fails vLLM's MP
    multi-node topology constraints (e.g. heterogeneous nodes requested for
    cross-node TP/PP). The outer worker-combination loop skips this group and
    tries the next one — without raising, because a different combination
    may still satisfy the requirements.
    """
    try:
        mode = (
            'internal'
            if manual_dp_enabled(model.env)
            else vllm_lb_mode(model.backend_parameters)
        )
        if mode in ("external", "multi_port_external", "hybrid") or (
            len(selected_workers) > 1
            and (
                resolve_executor_backend(
                    model.backend_parameters, model.backend_version, model.image_name
                )
                == "mp"
            )
        ):
            gpu_per_node = [len(w.status.gpu_devices or []) for w in selected_workers]
            if mode in ("external", "multi_port_external", "hybrid"):
                topology = managed_lb_parallelism(
                    model.backend_parameters, gpu_per_node
                )
                validate_managed_pd_topology(model, topology)
            else:
                validate_multinode_topology(
                    gpu_per_node, parse_user_parallelism(model.backend_parameters)
                )
    except ValueError as e:
        logger.info(
            f"Skipping worker combination {[w.name for w in selected_workers]} "
            f"for {model.name}: {e}"
        )
        return None, str(e)

    main_worker = selected_workers[0]
    vram_claim_main = {
        gpu.index: int(gpu.memory.total * gpu_memory_utilization)
        for gpu in main_worker.status.gpu_devices
    }
    gpu_type = main_worker.status.gpu_devices[0].type
    candidate = ModelInstanceScheduleCandidate(
        worker=main_worker,
        gpu_type=gpu_type,
        gpu_indexes=[gpu.index for gpu in main_worker.status.gpu_devices],
        computed_resource_claim=ComputedResourceClaim(
            vram=vram_claim_main,
            ram=get_computed_ram_claim(model, vram_claim_main),
        ),
    )
    candidate.subordinate_workers = []
    for worker in selected_workers[1:]:
        vram_claim_subworker = {
            gpu.index: int(gpu.memory.total * gpu_memory_utilization)
            for gpu in worker.status.gpu_devices
        }
        candidate.subordinate_workers.append(
            ModelInstanceSubordinateWorker(
                worker_id=worker.id,
                worker_name=worker.name,
                worker_ip=worker.ip,
                worker_ifname=worker.ifname,
                total_gpus=len(worker.status.gpu_devices),
                gpu_type=gpu_type,
                gpu_indexes=[gpu.index for gpu in worker.status.gpu_devices],
                computed_resource_claim=ComputedResourceClaim(
                    vram=vram_claim_subworker,
                    ram=get_computed_ram_claim(model, vram_claim_subworker),
                ),
            )
        )

    return candidate, None
