import json
import logging
from collections import defaultdict
from typing import List, Optional, Dict

from transformers import PretrainedConfig
from transformers.utils import strtobool

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
)
from gpustack.policies.event_recorder.recorder import EventCollector, EventLevelEnum
from gpustack.policies.utils import (
    get_computed_ram_claim,
    get_worker_allocatable_resource,
    ListMessageBuilder,
    get_model_num_attention_heads,
    group_worker_gpu_by_memory,
    WorkerGPUInfo,
    estimate_model_vram,
    get_model_ram_claim,
    ram_not_enough,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.workers import Worker
from gpustack.config import Config
from gpustack.server.db import get_engine
from gpustack.utils.command import find_parameter
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_ids_by_worker, parse_gpu_id
from gpustack.utils.hub import get_pretrained_config
from gpustack.utils.unit import byte_to_gib

logger = logging.getLogger(__name__)

EVENT_ACTION_DEFAULT = "default_scheduling_msg"
EVENT_ACTION_MANUAL_MULTI = "manual_multi_gpu_scheduling_msg"
EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU = "auto_multi_worker_multi_gpu_scheduling_msg"
EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU = (
    "auto_single_worker_multi_gpu_scheduling_msg"
)
EVENT_ACTION_AUTO_SINGLE_GPU = "auto_single_gpu_scheduling_msg"


class SGLangResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        cfg: Config,
        model: Model,
    ):
        self._engine = get_engine()
        self._cfg = cfg
        self._model = model
        self._vram_claim = 0
        self._ram_claim = 0
        self._mem_fraction_static = 0.9  # SGLang default
        self._set_mem_fraction_static()
        self._pretrained_config: Optional[PretrainedConfig] = None
        self._num_attention_heads = 0
        self._effective_vram = 0
        self._messages = []
        self._event_collector = EventCollector(self._model, logger)
        self._nnodes = 1
        self._tp_size = 1
        self._gpu_count = 0

        # for multi worker schedule
        self._largest_multi_gpu_vram: int = 0
        self._largest_multi_gpu_total = 0
        self._largest_multi_gpu_utilization_satisfied_count = 0
        self._unsatisfied_gpu_messages: Dict[str, List[int]] = {}

        self._per_gpu_vram = 0
        self._selected_gpu_workers: Optional[List[str]] = None
        self._selected_gpu_worker_count = 0
        self._selected_gpu_indexes_by_worker: Dict[str, List[int]] = {}

        if self._model.gpu_selector and self._model.gpu_selector.gpu_ids:
            gpu_ids_by_worker = parse_gpu_ids_by_worker(
                self._model.gpu_selector.gpu_ids
            )
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

            # When user defined gpu selector, we use the gpu count from it.
            self._gpu_count = len(self._model.gpu_selector.gpu_ids)

        self._set_nnodes()

        tp_size = find_parameter(model.backend_parameters, ["tp-size", "tp"])
        self._tp_size = int(tp_size or 1)
        pp_size = find_parameter(model.backend_parameters, ["pp-size", "pp"])
        dp_size = find_parameter(model.backend_parameters, ["dp-size", "dp"])
        speculative_algorithm = find_parameter(
            model.backend_parameters, ["speculative-algorithm"]
        )
        enable_mixed_chunk_param = find_parameter(
            model.backend_parameters, ["enable-mixed-chunk"]
        )
        enable_mixed_chunk = (
            strtobool(enable_mixed_chunk_param)
            if enable_mixed_chunk_param is not None
            else False
        )
        if tp_size or pp_size:
            world_size = int(tp_size or 1) * int(pp_size or 1)
            if world_size % self._nnodes != 0:
                raise ValueError(f"tp-size {world_size} must be divisible by nnodes")
            if self._gpu_count and self._gpu_count != world_size:
                # Both gpu selector and tp/pp are set, validate they match.
                raise ValueError(
                    f"Model {model.name} has -tp/-pp set, but the selected gpu count ({self._gpu_count}) does not match the world size ({world_size})."
                )
            else:
                self._gpu_count = world_size
                self._vram_claim = 0

        if pp_size and int(pp_size) > 1:
            disable_overlap_schedule_param = find_parameter(
                model.backend_parameters, ["disable-overlap-schedule"]
            )
            disable_overlap_schedule = (
                strtobool(disable_overlap_schedule_param)
                if disable_overlap_schedule_param is not None
                else False
            )
            if (
                not disable_overlap_schedule
                or speculative_algorithm is not None
                or enable_mixed_chunk
            ):
                raise ValueError(
                    "Pipeline parallelism is not compatible with overlap schedule, speculative decoding, mixed chunked prefill."
                )

        if dp_size and int(dp_size) > 1 and self._nnodes != 1:
            enable_dp_attention = find_parameter(
                model.backend_parameters, ["enable-dp-attention"]
            )
            if not enable_dp_attention:
                raise ValueError(
                    "multi-node data parallel is not supported unless dp attention!"
                )

        if speculative_algorithm is not None and enable_mixed_chunk:
            raise ValueError("enable_mixed_chunk is required for speculative decoding")

        self._set_pretrained_config()
        self._set_num_attention_heads()

    def _set_mem_fraction_static(self):
        """Set memory fraction static parameter for SGLang."""
        # SGLang's argument `--mem-fraction-static`, it may
        if self._model.backend_parameters:
            mem_fraction_static = find_parameter(
                self._model.backend_parameters, ["mem-fraction-static"]
            )
            if mem_fraction_static:
                self._mem_fraction_static = float(mem_fraction_static)

    def _set_nnodes(self):
        if (
            self._model.gpu_selector
            and self._model.gpu_selector.gpu_ids
            and len(self._model.gpu_selector.gpu_ids) > 1
        ):
            self._nnodes = len(self._model.gpu_selector.gpu_ids)
        if self._model.backend_parameters:
            nnodes_param = find_parameter(self._model.backend_parameters, ["nnodes"])
            if nnodes_param:
                self._nnodes = int(nnodes_param)

    def _set_pretrained_config(self):
        """Set pretrained config for the model."""
        try:
            self._pretrained_config = get_pretrained_config(
                self._model, trust_remote_code=True
            )
        except ValueError as e:
            if "architecture" in e.args[0] and self._model.backend_version:
                # In the AutoConfig.from_pretrained method, the architecture field in config undergoes validation.
                # For custom backend versions, exceptions caused by unrecognized architectures should be allowed
                # to prevent startup failures of valid new models with properly customized versions.
                self._pretrained_config = PretrainedConfig()

                # We can also try to get the architectures from hf-overrides
                hf_overrides = find_parameter(
                    self._model.backend_parameters, ["hf-overrides"]
                )
                if hf_overrides:
                    overrides_dict = json.loads(hf_overrides)
                    self._pretrained_config.architectures = overrides_dict.get(
                        "architectures", []
                    )
            else:
                raise e
        except Exception as e:
            raise Exception(
                f"Cannot get pretrained config for model {self._model.readable_source}: {e}"
            ) from e

    def _set_num_attention_heads(self):
        self._num_attention_heads = get_model_num_attention_heads(
            self._pretrained_config
        )
        if (
            self._tp_size
            and self._num_attention_heads
            and self._num_attention_heads % self._tp_size != 0
        ):
            raise ValueError(
                f"Total number of attention heads ({self._num_attention_heads})"
                " must be divisible by tp-size "
                f"({self._tp_size})."
            )

    def _cal_effective_vram(self) -> float:
        """Calculate effective VRAM considering SGLang's memory management."""
        if self._mem_fraction_static == 0:
            return self._vram_claim
        return self._vram_claim / self._mem_fraction_static

    def _set_messages(self):
        """Set scheduling messages for SGLang."""
        if self._messages:
            return

        event_messages = {
            EVENT_ACTION_DEFAULT: "",
            EVENT_ACTION_MANUAL_MULTI: "",
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU: "",
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU: "",
            EVENT_ACTION_AUTO_SINGLE_GPU: "",
        }

        for event in self._event_collector.events:
            event_messages[event.action] = event.message

        messages = event_messages[EVENT_ACTION_DEFAULT] + "\n"
        for action in [
            EVENT_ACTION_MANUAL_MULTI,
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            EVENT_ACTION_AUTO_SINGLE_GPU,
        ]:
            if event_messages[action]:
                messages += event_messages[action]
                break

        self._messages.append(messages)

    def _add_message(self, message: str):
        self._messages.append(message)

    def get_messages(self) -> List[str]:
        return self._messages

    async def _get_worker_allocatable_resource(self, worker: Worker):
        return await get_worker_allocatable_resource(self._engine, worker)

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement for SGLang.
        """
        self._vram_claim = await estimate_model_vram(
            self._model, self._cfg.huggingface_token
        )
        self._ram_claim = get_model_ram_claim(self._model)
        self._effective_vram = self._cal_effective_vram()
        logger.info(
            f"Calculated SGLang resource claim for model {self._model.readable_source}, "
            f"VRAM claim: {self._vram_claim}, RAM claim: {self._ram_claim}"
        )

        default_msg_list = ListMessageBuilder(
            f"The model requires approximately {byte_to_gib(self._vram_claim)} GiB of VRAM"
            f"{f' and {byte_to_gib(self._ram_claim)} GiB of RAM' if self._ram_claim > 0 else ''}."
        )
        if self._mem_fraction_static != 0:
            default_msg_list.append(
                f"With --mem-fraction-static={self._mem_fraction_static}, "
                f"all GPUs combined need to provide at least {byte_to_gib(int(self._vram_claim / self._mem_fraction_static))} GiB of total VRAM "
                f"and each GPU needs {int(self._mem_fraction_static * 100)}% of allocatable VRAM."
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

        sort_workers_by_gpu_count(workers)

        for candidate_func in candidate_functions:
            logger.debug(
                f"SGLang model {self._model.readable_source}, filter candidates with resource fit selector: {candidate_func.__name__}"
            )

            candidates = await candidate_func(workers)

            if len(candidates) == 1 and candidates[0].overcommit:
                self._set_messages()

            if candidates:
                return candidates

        self._set_messages()
        return []

    async def find_manual_gpu_selection_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find candidates for manual GPU selection based on user-specified GPU IDs.
        This function handles all manual GPU selection scenarios for SGLang.
        """
        # Skip if no manual GPU selection is specified
        if not self._selected_gpu_workers:
            return []

        logger.debug(
            f"SGLang manual GPU selection: workers={self._selected_gpu_workers}, "
            f"worker_count={self._selected_gpu_worker_count}, "
            f"gpu_indexes_by_worker={self._selected_gpu_indexes_by_worker}"
        )

        candidates = []

        # Handle single worker scenarios
        if self._selected_gpu_worker_count == 1:
            selected_worker_name = self._selected_gpu_workers[0]
            selected_gpu_indexes = self._selected_gpu_indexes_by_worker[
                selected_worker_name
            ]

            # Find the worker
            target_worker = None
            for worker in workers:
                if worker.name == selected_worker_name:
                    target_worker = worker
                    break

            if not target_worker:
                self._add_message(
                    f"Selected worker '{selected_worker_name}' not found."
                )
                return []

            # Single GPU selection
            if len(selected_gpu_indexes) == 1:
                gpu_index = selected_gpu_indexes[0]
                worker_candidates = await self._find_single_worker_single_gpu_full_offloading_candidates(
                    target_worker, gpu_index
                )
                candidates.extend(worker_candidates)

            # Multi GPU selection on single worker
            elif len(selected_gpu_indexes) > 1:
                worker_candidates = (
                    await self.manually_select_single_worker_multi_gpu_candidates(
                        target_worker, selected_gpu_indexes
                    )
                )
                candidates.extend(worker_candidates)

        # Handle multi-worker scenarios
        elif self._selected_gpu_worker_count > 1:
            # Filter workers to only include selected ones
            selected_workers = []
            for worker in workers:
                if worker.name in self._selected_gpu_workers:
                    selected_workers.append(worker)

            if len(selected_workers) != self._selected_gpu_worker_count:
                missing_workers = set(self._selected_gpu_workers) - {
                    w.name for w in selected_workers
                }
                self._add_message(
                    f"Selected workers not found: {list(missing_workers)}"
                )
                return []

            # Multi-worker multi-GPU selection
            worker_candidates = (
                await self.manual_select_multi_worker_multi_gpu_candidates(
                    selected_workers
                )
            )
            candidates.extend(worker_candidates)

        # Add manual selection event message if overcommit
        if candidates and len(candidates) == 1 and candidates[0].overcommit:
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_MANUAL_MULTI,
                "Manual GPU selection resulted in resource overcommit. "
                "The selected GPUs may not have sufficient allocatable VRAM.",
            )

        return candidates

    async def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single GPU candidates for SGLang.
        This function only handles automatic GPU selection.
        """
        # Skip if manual GPU selection is specified
        if self._selected_gpu_workers:
            return []

        candidates = []

        # Auto selection only
        for worker in workers:
            worker_candidates = (
                await self._find_single_worker_single_gpu_full_offloading_candidates(
                    worker, None
                )
            )
            candidates.extend(worker_candidates)

        return candidates

    async def _find_single_worker_single_gpu_full_offloading_candidates(
        self, worker: Worker, selected_gpu_index: Optional[int]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with worker.
        requires: worker.status.gpu_devices is not None
        """

        candidates = []

        allocatable = await self._get_worker_allocatable_resource(worker)

        if ram_not_enough(self._ram_claim, allocatable):
            return []

        if not worker.status.gpu_devices:
            return []

        largest_single_gpu_vram = 0
        largest_single_gpu_utilization = 0

        for _, gpu in enumerate(worker.status.gpu_devices):
            if selected_gpu_index is not None and gpu.index != selected_gpu_index:
                continue
            gpu_index = gpu.index
            allocatable_vram = allocatable.vram.get(gpu_index, 0)
            allocatable_gpu_utilization = allocatable_vram / gpu.memory.total

            if allocatable_vram > largest_single_gpu_vram:
                largest_single_gpu_vram = allocatable_vram
                largest_single_gpu_utilization = allocatable_gpu_utilization

            overcommit = False
            exceeds_vram = (
                self._vram_claim > gpu.memory.total * self._mem_fraction_static
            )
            exceeds_memory_utilization = (
                allocatable_gpu_utilization < self._mem_fraction_static
            )
            if exceeds_vram or exceeds_memory_utilization:
                if selected_gpu_index is not None:
                    overcommit = True
                else:
                    continue

            vram_claim_bytes = int(gpu.memory.total * self._mem_fraction_static)
            vram_claim = {gpu_index: vram_claim_bytes}
            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=[gpu_index],
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=get_computed_ram_claim(self._model, vram_claim),
                    ),
                    overcommit=overcommit,
                )
            )

        if not candidates or (len(candidates) == 1 and candidates[0].overcommit):
            event_msg = f"The current available GPU only has {byte_to_gib(largest_single_gpu_vram)} GiB allocatable VRAM."
            if self._mem_fraction_static != 0:
                event_msg = (
                    event_msg.rstrip(".")
                    + f" ({(largest_single_gpu_utilization * 100):.2f}%)."
                )
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_AUTO_SINGLE_GPU,
                str(ListMessageBuilder(event_msg)),
            )

        return candidates

    async def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi GPU candidates for SGLang.
        This function only handles automatic GPU selection.
        """
        # Skip if manual GPU selection is specified
        if self._selected_gpu_workers:
            return []

        candidates = []

        for worker in workers:
            worker_candidates = (
                await self._find_single_worker_multi_gpu_full_offloading_candidates(
                    worker
                )
            )
            candidates.extend(worker_candidates)

        return candidates

    async def _find_single_worker_multi_gpu_full_offloading_candidates(  # noqa: C901
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """Find single worker multi GPU candidates for a specific worker."""
        if not worker.status.gpu_devices or len(worker.status.gpu_devices) < 2:
            return []

        # SGLang performs VRAM balancing checks. We group all GPUs based on available VRAM capacity
        gpu_group = await group_worker_gpu_by_memory(
            self._engine, [worker], ram_claim=self._ram_claim
        )

        for info in gpu_group:
            gpu_list = info
            if any(
                gpu.allocatable_vram / gpu.gpu_device.memory.total
                < self._mem_fraction_static
                for gpu in gpu_list
            ):
                continue
            total_allocatable_vram = sum(gpu.allocatable_vram for gpu in gpu_list)

            if not gpu_list or len(gpu_list) < 2:
                continue

            if total_allocatable_vram > self._largest_multi_gpu_total:
                self._largest_multi_gpu_vram = total_allocatable_vram
                self._largest_multi_gpu_utilization_satisfied_count = len(gpu_list)
                self._largest_multi_gpu_total = len(worker.status.gpu_devices)

            if total_allocatable_vram < self._effective_vram:
                continue

            # Sort by vram in descending order
            sorted_gpu_devices: list[WorkerGPUInfo] = sorted(
                gpu_list,
                key=lambda gpu: gpu.allocatable_vram,
                reverse=True,
            )

            vram_sum = 0
            gpu_sum = 0
            gpu_indexes = []
            vram_claim: Dict[int, int] = {}
            found_candidate = False
            for _, gpu_device in enumerate(sorted_gpu_devices):
                gpu = gpu_device.gpu_device
                gpu_indexes.append(gpu.index)
                vram_claim[gpu.index] = int(
                    gpu.memory.total * self._mem_fraction_static
                )
                gpu_sum += 1
                vram_sum += vram_claim[gpu.index]

                if (
                    self._num_attention_heads
                    and self._num_attention_heads % gpu_sum != 0
                ):
                    continue

                if self._gpu_count and gpu_sum >= self._gpu_count:
                    if vram_sum >= self._vram_claim:
                        found_candidate = True
                    # if self._gpu_count is set, cannot return more than gpu_count
                    break

                if self._gpu_count is None and vram_sum >= self._vram_claim:
                    found_candidate = True
                    break

            if found_candidate:
                return [
                    ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_indexes=gpu_indexes,
                        computed_resource_claim=ComputedResourceClaim(
                            vram=vram_claim,
                            ram=get_computed_ram_claim(self._model, vram_claim),
                        ),
                    )
                ]
        event_msg_list = []
        if (
            self._num_attention_heads
            and self._largest_multi_gpu_utilization_satisfied_count != 0
            and self._num_attention_heads
            % self._largest_multi_gpu_utilization_satisfied_count
            != 0
        ):
            event_msg_list.append(
                f"Total number of attention heads ({self._num_attention_heads})"
                " must be divisible by gpu count "
                f"({self._largest_multi_gpu_utilization_satisfied_count})."
            )
        if len(event_msg_list) == 0:
            event_msg = f"The largest available worker has {byte_to_gib(self._largest_multi_gpu_vram):.2f} GiB allocatable VRAM."
            if self._mem_fraction_static != 0:
                effective_vram = byte_to_gib(
                    int(
                        self._largest_multi_gpu_vram
                        * self._mem_fraction_static
                        * self._largest_multi_gpu_utilization_satisfied_count
                        / self._largest_multi_gpu_total
                    )
                )
                event_msg = (
                    event_msg.rstrip(".")
                    + f", {self._largest_multi_gpu_utilization_satisfied_count}/{self._largest_multi_gpu_total} of GPUs meet the VRAM utilization ratio, providing {effective_vram} GiB of allocatable VRAM."
                )
            event_msg_list.append(event_msg)

        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            str(ListMessageBuilder(event_msg_list)),
        )

        return []

    async def manually_select_single_worker_multi_gpu_candidates(
        self, worker: Worker, gpu_indexes: List[int]
    ) -> List[ModelInstanceScheduleCandidate]:
        """Manually select multi GPU candidates for SGLang."""
        candidates = []
        overcommit = False
        unsatisfied_gpu_messages: dict[str, list[int]] = {}
        satisfied_gpu_count = 0
        allocatable = await self._get_worker_allocatable_resource(worker)
        if len(gpu_indexes) <= 1:
            return []

        # model require total vram
        self._effective_vram = self._cal_effective_vram()

        total_vram = sum(allocatable.vram.get(gpu_idx, 0) for gpu_idx in gpu_indexes)
        if total_vram < self._effective_vram:
            return []

        # For SGLang, distribute memory evenly across GPUs for tensor parallelism
        vram_per_gpu = int(self._effective_vram / len(gpu_indexes))

        # Check if the GPU is overcommitted
        for gpu_idx in gpu_indexes:
            allocatable_vram = allocatable.vram.get(gpu_idx, 0)
            if allocatable_vram < vram_per_gpu:
                overcommit = True
            elif (
                allocatable_vram / worker.status.gpu_devices[gpu_idx].memory.total
                < self._mem_fraction_static
            ):
                overcommit = True

            if overcommit:
                if worker.name not in unsatisfied_gpu_messages:
                    unsatisfied_gpu_messages[worker.name] = []
                unsatisfied_gpu_messages[worker.name].append(gpu_idx)
            else:
                satisfied_gpu_count += 1

        if overcommit:
            scheduling_msg = ListMessageBuilder([])
            unsatisfied_devices_msg = str(unsatisfied_gpu_messages[worker.name])
            if len(unsatisfied_gpu_messages[worker.name]) > 3:
                unsatisfied_devices_msg = (
                    str(unsatisfied_gpu_messages[worker.name][:3]).rstrip("]")
                    + f"...(more {len(unsatisfied_gpu_messages[worker.name]) - 3})]"
                )
            effective_vram = byte_to_gib(
                total_vram
                * self._mem_fraction_static
                * satisfied_gpu_count
                / len(gpu_indexes)
            )
            scheduling_msg.extend(
                [
                    f"Worker {worker.name} GPU indexes {unsatisfied_devices_msg} fail to meet the {(self._mem_fraction_static * 100):.2f}% allocatable VRAM ratio or the per-GPU VRAM requirement {vram_per_gpu}.",
                    f"Selected GPUs have {byte_to_gib(total_vram)} GiB allocatable VRAM, {satisfied_gpu_count}/{len(gpu_indexes)} of GPUs meet the above requirements, providing {effective_vram:.2f} GiB of allocatable VRAM.",
                ]
            )
        else:
            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=gpu_indexes,
                    computed_resource_claim=ComputedResourceClaim(
                        vram={gpu_idx: vram_per_gpu for gpu_idx in gpu_indexes},
                    ),
                    overcommit=overcommit,
                )
            )
        return candidates

    async def find_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find multi worker multi GPU candidates for SGLang.
        This function only handles automatic multi-worker selection.
        """

        candidates = []

        if self._model.distributed_inference_across_workers:
            gpu_group = await group_worker_gpu_by_memory(
                self._engine, workers, ram_claim=self._ram_claim
            )
            for gpu_list in gpu_group:
                if len(gpu_list) <= 1:
                    continue
                if any(
                    gpu.allocatable_vram / gpu.gpu_device.memory.total
                    < self._mem_fraction_static
                    for gpu in gpu_list
                ):
                    continue

                worker_gpu_cnt = {}
                for gpu_info in gpu_list:
                    if not worker_gpu_cnt.get(gpu_info.worker_id):
                        worker_gpu_cnt[gpu_info.worker_id] = 0
                    worker_gpu_cnt[gpu_info.worker_id] += 1
                first_cnt = 0
                # workers must with the same number of GPUs
                for gpu_cnt in worker_gpu_cnt.values():
                    if not first_cnt:
                        first_cnt = gpu_cnt
                    if first_cnt != gpu_cnt:
                        first_cnt = 0
                        break
                if not first_cnt:
                    continue

                return await self.auto_select_multi_worker_multi_gpu_candidates(workers)

        return candidates

    async def _get_worker_vram_claim(
        self,
        worker: Worker,
        gpu_indexes: List[int],
        mem_fraction_static: float,
    ) -> Dict[int, int]:
        """
        Given a worker and gpu indexes, get the vram claim according to gpu_memory_utilization.
        Returns a dictionary of gpu index to vram claim in bytes, and a boolean indicating
        whether the claim exceeds the allocatable vram.
        """
        vram_claim: Dict[int, int] = {}

        allocatable = await self._get_worker_allocatable_resource(worker)
        for gpu in worker.status.gpu_devices:
            if gpu.index not in gpu_indexes:
                continue

            vram_claim[gpu.index] = self._per_gpu_vram
            # Record allocation info for scheduling message
            allocatable_vram = allocatable.vram.get(gpu.index, 0)
            allocatable_utilization = allocatable_vram / gpu.memory.total
            if (
                allocatable_utilization >= mem_fraction_static
                and allocatable_vram > self._per_gpu_vram
            ):
                self._largest_multi_gpu_utilization_satisfied_count += 1
                self._largest_multi_gpu_vram += allocatable_vram
            else:
                if worker.name not in self._unsatisfied_gpu_messages:
                    self._unsatisfied_gpu_messages[worker.name] = []
                self._unsatisfied_gpu_messages[worker.name].append(gpu.index)

        return vram_claim

    async def auto_select_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """Auto select multi worker multi GPU candidates for SGLang."""
        candidates = []

        if not workers or len(workers) < 2:
            return candidates

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

        # Loop through worker groups with the same number of GPUs.
        for gpu_count, worker_group in workers_by_gpu_count_dict.items():
            if len(worker_group) < 2:
                continue

            selected_workers: List[Worker] = []
            gpu_sum = 0
            vram_sum = 0
            for worker in worker_group:
                allocatable = await self._get_worker_allocatable_resource(worker)

                if ram_not_enough(self._ram_claim, allocatable):
                    # The RAM resource(for extended KV cache) is required per worker.
                    # Skip the worker if it does not satisfy the RAM requirement.
                    continue

                if any(
                    allocatable.vram.get(gpu.index, 0) / gpu.memory.total
                    < self._mem_fraction_static
                    for gpu in worker.status.gpu_devices
                ):
                    continue
                selected_workers.append(worker)
                gpu_sum += gpu_count
                vram_sum += sum(
                    int(gpu.memory.total * self._mem_fraction_static)
                    for gpu in worker.status.gpu_devices
                )

                if (
                    self._num_attention_heads
                    and self._num_attention_heads % gpu_sum == 0
                ) and (vram_sum >= self._vram_claim):
                    return [
                        _create_candidate(
                            self._model,
                            selected_workers,
                            self._mem_fraction_static,
                        )
                    ]
            if vram_sum > largest_vram:
                workers_combination = selected_workers
                largest_vram = vram_sum
                worker_count = len(worker_group)
                device_count_per_worker = gpu_count

        # Nothing can be return, construct scheduling message
        event_message = ListMessageBuilder([])
        if workers_combination:
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

    async def manual_select_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """Manual select multi worker multi GPU candidates for SGLang."""
        if len(workers) < 2:
            return []

        overcommit = False
        self._effective_vram = self._cal_effective_vram()

        for _, gpu_indexes in self._selected_gpu_indexes_by_worker.items():
            self._largest_multi_gpu_total += len(gpu_indexes)

        self._per_gpu_vram = int(self._effective_vram / self._largest_multi_gpu_total)

        # Main worker is the first one (with most GPUs)
        main_worker = workers[0]
        main_worker_name = main_worker.name
        main_gpu_indexes = self._selected_gpu_indexes_by_worker[main_worker_name]
        main_vram_claim = await self._get_worker_vram_claim(
            main_worker, main_gpu_indexes, self._mem_fraction_static
        )

        subordinate_workers: List[ModelInstanceSubordinateWorker] = []
        # Handle subordinate workers
        for worker in workers:
            if worker.name not in self._selected_gpu_workers:
                continue
            if worker.name == main_worker_name:
                continue

            gpu_indexes = self._selected_gpu_indexes_by_worker[worker.name]
            vram_claim = await self._get_worker_vram_claim(
                worker, gpu_indexes, self._mem_fraction_static
            )

            subordinate_workers.append(
                ModelInstanceSubordinateWorker(
                    worker_id=worker.id,
                    worker_name=worker.name,
                    worker_ip=worker.ip,
                    worker_ifname=worker.ifname,
                    total_gpus=len(worker.status.gpu_devices),
                    gpu_indexes=gpu_indexes,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=get_computed_ram_claim(self._model, vram_claim),
                    ),
                )
            )

        if (
            self._largest_multi_gpu_utilization_satisfied_count
            < self._largest_multi_gpu_total
            or len(self._unsatisfied_gpu_messages) > 0
        ):
            overcommit = True

        if overcommit:
            scheduling_msg = ListMessageBuilder([])
            for worker_name, gpu_indexes in self._unsatisfied_gpu_messages.items():
                # Some seleted gpu is not meet the utilization
                devices_msg = (
                    str(gpu_indexes[:3]).rstrip("]")
                    + f"...(more {len(gpu_indexes) - 3})]"
                    if len(gpu_indexes) > 3
                    else str(gpu_indexes)
                )
                unsatisfied_msg = f"Worker {worker_name} GPU indexes {devices_msg}"
                if len(self._unsatisfied_gpu_messages) > 1:
                    unsatisfied_msg += f" and other {len(self._unsatisfied_gpu_messages) - 1} {'workers' if len(self._unsatisfied_gpu_messages) > 2 else 'worker'}"
                unsatisfied_msg += f" {'fail' if len(self._unsatisfied_gpu_messages) > 1 else 'fails'} to meet the {(self._mem_fraction_static * 100):.2f}% allocatable VRAM ratio or the per-GPU VRAM requirement {self._per_gpu_vram}."
                scheduling_msg.append(unsatisfied_msg)
                break

            effective_vram = byte_to_gib(
                int(
                    self._largest_multi_gpu_vram
                    * self._mem_fraction_static
                    * self._largest_multi_gpu_utilization_satisfied_count
                    / self._largest_multi_gpu_total
                )
            )
            # All selected gpus combined VRAM is not enough.
            scheduling_msg.append(
                f"Selected GPUs have {byte_to_gib(self._largest_multi_gpu_vram)} GiB allocatable VRAM, "
                f"{self._largest_multi_gpu_utilization_satisfied_count}/{self._largest_multi_gpu_total} of GPUs meet the VRAM utilization ratio, providing {effective_vram:.2f} GiB of allocatable VRAM."
            )

            self._event_collector.add(
                EventLevelEnum.INFO, EVENT_ACTION_MANUAL_MULTI, str(scheduling_msg)
            )

        return [
            ModelInstanceScheduleCandidate(
                worker=main_worker,
                gpu_indexes=main_gpu_indexes,
                computed_resource_claim=ComputedResourceClaim(
                    vram=main_vram_claim,
                ),
                subordinate_workers=subordinate_workers,
                overcommit=overcommit,
            )
        ]


def _create_candidate(
    model: Model,
    selected_workers: List[Worker],
    mem_fraction_static: float = 0.9,
) -> ModelInstanceScheduleCandidate:
    """Create a candidate with SGLang-specific parameters and primary node confirmation."""
    if not selected_workers:
        raise ValueError("No workers provided for candidate creation")

    # Primary worker is the first one (with most GPUs)
    primary_worker = selected_workers[0]
    subordinate_workers = []

    # Calculate primary worker resources
    primary_gpu_indexes = []
    primary_vram_claim = {}

    for gpu in primary_worker.status.gpu_devices:
        primary_gpu_indexes.append(gpu.index)
        primary_vram_claim[gpu.index] = int(gpu.memory.total * mem_fraction_static)

    # Process subordinate workers if any
    if len(selected_workers) > 1:
        for worker in selected_workers[1:]:
            gpu_indexes = []
            vram_claim = {}

            if worker.status.gpu_devices:
                for gpu in worker.status.gpu_devices:
                    gpu_indexes.append(gpu.index)
                    vram_claim[gpu.index] = int(gpu.memory.total * mem_fraction_static)

            subordinate_worker = ModelInstanceSubordinateWorker(
                worker_id=worker.id,
                worker_name=worker.name,
                worker_ip=worker.ip,
                worker_ifname=worker.ifname,
                total_gpus=len(gpu_indexes),
                gpu_indexes=gpu_indexes,
                computed_resource_claim=ComputedResourceClaim(
                    vram=vram_claim,
                    ram=get_computed_ram_claim(model, vram_claim),
                ),
            )
            subordinate_workers.append(subordinate_worker)

    computed_resource_claim = ComputedResourceClaim(
        vram=primary_vram_claim,
        ram=get_computed_ram_claim(model, primary_vram_claim),
    )

    return ModelInstanceScheduleCandidate(
        worker=primary_worker,
        gpu_indexes=primary_gpu_indexes,
        computed_resource_claim=computed_resource_claim,
        subordinate_workers=subordinate_workers if subordinate_workers else None,
    )


def sort_workers_by_gpu_count(workers: List[Worker]):
    """Sort workers by GPU count for SGLang scheduling (in-place sorting)."""
    workers.sort(
        key=lambda w: len(w.status.gpu_devices) if w.status.gpu_devices else 0,
        reverse=True,
    )
