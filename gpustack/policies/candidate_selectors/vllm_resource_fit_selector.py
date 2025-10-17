import asyncio
import json
from collections import defaultdict
import logging
import os
import re
from typing import Dict, List, Optional

from transformers import PretrainedConfig

from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
)
from gpustack.policies.event_recorder.recorder import EventCollector, EventLevelEnum
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
    get_worker_model_instances,
    ListMessageBuilder,
    get_model_num_attention_heads,
    get_local_model_weight_size,
)
from gpustack.schemas.models import (
    CategoryEnum,
    ComputedResourceClaim,
    Model,
    ModelInstanceSubordinateWorker,
    SourceEnum,
)
from gpustack.schemas.workers import GPUDevicesInfo, Worker
from gpustack.config import Config
from gpustack.server.db import get_engine
from gpustack.utils.command import find_parameter
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_id, parse_gpu_ids_by_worker
from gpustack.utils.hub import get_model_weight_size, get_pretrained_config
from gpustack.utils.unit import byte_to_gib

logger = logging.getLogger(__name__)

EVENT_ACTION_DEFAULT = "default_scheduling_msg"
EVENT_ACTION_MANUAL_MULTI = "manual_multi_gpu_scheduling_msg"
EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU = "auto_multi_worker_multi_gpu_scheduling_msg"
EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU = (
    "auto_single_worker_multi_gpu_scheduling_msg"
)
EVENT_ACTION_AUTO_SINGLE_GPU = "auto_single_gpu_scheduling_msg"


async def estimate_model_vram(model: Model, token: Optional[str] = None) -> int:
    """
    Estimate the vram requirement in bytes heuristically.
    This is the minimum requirement to help us decide how many GPUs are needed for the model.
    If users explicitly set parameters like tp & pp, this estimation is not needed.

    Formula:

        VRAM = WEIGHT * 1.2 + RESERVERD_FOOTPRINT

    Reference for the 20% overhead: https://blog.eleuther.ai/transformer-math/#total-inference-memory

    For example, using bfloat16,
    - 0.5B requires 3.1 GiB
    - 3B requires 8.9 GiB
    - 7B requires 19.0 GiB
    - 72B requires 164.5 GiB

    """
    if model.env and 'GPUSTACK_MODEL_VRAM_CLAIM' in model.env:
        # Use as a potential workaround if the empirical vram estimation is far beyond the expected value.
        return int(model.env['GPUSTACK_MODEL_VRAM_CLAIM'])

    # CUDA graphs can take additional 1~3 GiB memory
    # https://github.com/vllm-project/vllm/blob/v0.6.1/vllm/worker/model_runner.py#L1313
    # For non-LLM models like embedding, set a smaller overhead
    framework_overhead = (
        2 * 1024**3
        if not model.categories or CategoryEnum.LLM in model.categories
        else 512 * 1024**2
    )
    weight_size = 0
    timeout_in_seconds = 15

    try:
        if (
            model.source == SourceEnum.HUGGING_FACE
            or model.source == SourceEnum.MODEL_SCOPE
        ):
            weight_size = await asyncio.wait_for(
                asyncio.to_thread(get_model_weight_size, model, token),
                timeout=timeout_in_seconds,
            )
        elif model.source == SourceEnum.LOCAL_PATH and os.path.exists(model.local_path):
            weight_size = get_local_model_weight_size(model.local_path)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout when getting weight size for model {model.name}")
    except Exception as e:
        logger.warning(f"Cannot get weight size for model {model.name}: {e}")

    # Reference: https://blog.eleuther.ai/transformer-math/#total-inference-memory
    return weight_size * 1.2 + framework_overhead


def get_model_ram(model: Model) -> int:
    """
    Get the RAM requirement for the model in bytes.
    """
    if (
        model.extended_kv_cache
        and model.extended_kv_cache.enabled
        and model.extended_kv_cache.max_local_cpu_size > 0
    ):
        # When extended kv cache is enabled, reserve the max_local_cpu_size for RAM.
        return model.extended_kv_cache.max_local_cpu_size * 1024**3
    return 0


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
    ):
        self._engine = get_engine()
        self._cfg = cfg
        self._model = model
        self._gpu_count = None
        self._vram_claim = 0
        self._largest_single_gpu_vram = 0
        self._largest_single_gpu_vram_utilization = 0
        self._largest_multi_gpu_vram = 0
        self._largest_multi_gpu_total = 0
        self._largest_multi_gpu_utilization_satisfied_count = 0

        self._num_attention_heads = None
        self._messages = []
        self._event_collector = EventCollector(self._model, logger)
        self._workers_allocatable_resource: Dict[int, Allocatable] = {}

        self._selected_gpu_workers: List[str] = None
        self._selected_gpu_worker_count = 0
        self._selected_gpu_indexes_by_worker: Dict[str, List[int]] = {}
        self._unsatisfied_gpu_messages: Dict[str, List[int]] = {}

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

            self._gpu_count = self._model.gpu_selector.gpus_per_replica or len(
                self._model.gpu_selector.gpu_ids
            )

        # When tp/pp is set, the gpu count is calculated by tp * pp.
        # Pick the candidate with satisfied gpu count.
        # Otherwise, estimate gpu count by vram requirement heuristically.
        tp = find_parameter(model.backend_parameters, ["tensor-parallel-size", "tp"])
        pp = find_parameter(model.backend_parameters, ["pipeline-parallel-size", "pp"])
        if tp or pp:
            world_size = int(tp or 1) * int(pp or 1)

            if self._gpu_count and self._gpu_count != world_size:
                # Both gpu selector and tp/pp are set, validate they match.
                raise ValueError(
                    f"Model {model.name} has -tp/-pp set, but the selected gpu count ({self._gpu_count}) does not match the world size ({world_size})."
                )
            else:
                self._gpu_count = world_size
                self._vram_claim = 0

        self._set_pretrained_config()
        # _pretrained_config may be None or an empty dictionary, but subsequent methods are designed to handle this case.
        self._set_gpu_memory_utilization()
        self._set_num_attention_heads()

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

        architectures = getattr(self._pretrained_config, "architectures", []) or []

        # Non-LLM models that vLLM still uses GPU memory utilization
        NON_LLM_GMU_EXCEPTIONS = {
            "Qwen3ForCausalLM",
            "Qwen3ForSequenceClassification",  # Qwen3-Embedding & Qwen3-Reranker
        }

        if CategoryEnum.LLM not in self._model.categories:
            # Disable for non-LLM models unless they are in the exception list
            return not any(arch in NON_LLM_GMU_EXCEPTIONS for arch in architectures)

        return False

    def _set_pretrained_config(self):
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
            self._gpu_count
            and self._num_attention_heads
            and self._num_attention_heads % self._gpu_count != 0
        ):
            raise ValueError(
                f"Total number of attention heads ({self._num_attention_heads})"
                " must be divisible by gpu count "
                f"({self._gpu_count})."
            )

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
        if worker.id in self._workers_allocatable_resource:
            return self._workers_allocatable_resource[worker.id]

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        self._workers_allocatable_resource[worker.id] = allocatable
        return allocatable

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement.
        """

        self._vram_claim = await estimate_model_vram(
            self._model, self._cfg.huggingface_token
        )
        self._ram_claim = get_model_ram(self._model)
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
            self.find_single_worker_single_gpu_full_offloading_candidates,
            self.find_single_worker_multi_gpu_full_offloading_candidates,
            self.find_multi_worker_multi_gpu_candidates,
        ]

        for candidate_func in candidate_functions:
            logger.debug(
                f"model {self._model.readable_source}, filter candidates with resource fit selector: {candidate_func.__name__}"
            )

            candidates = await candidate_func(workers)

            if len(candidates) == 1 and candidates[0].overcommit:
                # Manually selected candidate with overcommit. Also add the message.
                # It's useful for compatibility check.
                self._set_messages()

            if candidates:
                return candidates

        self._set_messages()
        return []

    async def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with workers.
        """
        if self._gpu_count is not None and self._gpu_count > 1:
            # Skip multi-GPU selection
            return []

        if (
            self._selected_gpu_worker_count > 1
            and self._gpu_count
            and self._gpu_count > 1
        ):
            # Skip multi-worker selection
            return []

        selected_gpu_worker = None
        filtered_workers = []

        if self._selected_gpu_workers:
            # Handle manual scheduling
            for worker in workers:
                if worker.name in self._selected_gpu_workers:
                    filtered_worker = worker
                    filtered_devices = []
                    indexes = self._selected_gpu_indexes_by_worker.get(worker.name, [])
                    for gpu in worker.status.gpu_devices:
                        if gpu.index in indexes:
                            filtered_devices.append(gpu)
                    filtered_worker.status.gpu_devices = filtered_devices
                    filtered_workers.append(filtered_worker)
        else:
            filtered_workers = workers

        candidates = []
        for worker in filtered_workers:
            if not worker.status.gpu_devices:
                continue

            if selected_gpu_worker and worker.name != selected_gpu_worker:
                continue

            result = (
                await self._find_single_worker_single_gpu_full_offloading_candidates(
                    worker,
                )
            )
            if result:
                candidates.extend(result)

        return candidates

    async def _find_single_worker_single_gpu_full_offloading_candidates(
        self,
        worker: Worker,
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu full offloading candidates for the model instance with worker.
        requires: worker.status.gpu_devices is not None
        """

        candidates = []

        allocatable = await self._get_worker_allocatable_resource(worker)

        if self._ram_claim > 0 and allocatable.ram < self._ram_claim:
            return []

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

            overcommit = False
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
                if self._selected_gpu_worker_count != 0:
                    overcommit = True
                else:
                    continue

            vram_claim = (
                int(gpu.memory.total * self._gpu_memory_utilization)
                if self._gpu_memory_utilization > 0  # LLMs
                else int(self._vram_claim)  # non LLMs
            )

            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=[gpu_index],
                    computed_resource_claim=ComputedResourceClaim(
                        vram={
                            gpu_index: vram_claim,
                        },
                        ram=self._ram_claim if self._ram_claim > 0 else None,
                    ),
                    overcommit=overcommit,
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

    async def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        if self._gpu_count == 1:
            return []

        should_skip_worker = 0
        if (
            self._selected_gpu_worker_count > 1
            and self._model.gpu_selector
            and self._model.gpu_selector.gpus_per_replica
        ):
            for (
                worker_name,
                selected_gpus,
            ) in self._selected_gpu_indexes_by_worker.items():
                if len(selected_gpus) < self._model.gpu_selector.gpus_per_replica:
                    should_skip_worker += 1
            if should_skip_worker == self._selected_gpu_worker_count:
                return []

        candidates = []
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            result = (
                await self._find_single_worker_multi_gpu_full_offloading_candidates(
                    worker
                )
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

    async def _find_single_worker_multi_gpu_full_offloading_candidates(  # noqa: C901
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi gpu full offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """

        total_gpu = len(worker.status.gpu_devices)
        if total_gpu < 2:
            return None

        if self._selected_gpu_workers:
            return await self.manual_select_single_worker_multi_gpu_candidates(worker)

        allocatable = await self._get_worker_allocatable_resource(worker)

        if self._ram_claim > 0 and allocatable.ram < self._ram_claim:
            return []

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

        if total_allocatable_vram > self._largest_multi_gpu_total:
            self._largest_multi_gpu_vram = total_allocatable_vram
            self._largest_multi_gpu_utilization_satisfied_count = satisfied_gpu_count
            self._largest_multi_gpu_total = len(worker.status.gpu_devices)

        # Sort by vram in descending order
        sorted_gpu_devices: GPUDevicesInfo = sorted(
            gpu_list,
            key=lambda gpu: allocatable.vram.get(gpu.index, 0),
            reverse=True,
        )

        vram_sum = 0
        gpu_sum = 0
        gpu_indexes = []
        vram_claim: Dict[int, int] = {}
        found_candidate = False
        for _, gpu in enumerate(sorted_gpu_devices):
            gpu_indexes.append(gpu.index)
            vram_claim[gpu.index] = (
                int(gpu.memory.total * self._gpu_memory_utilization)
                if self._gpu_memory_utilization > 0  # LLMs
                else allocatable.vram.get(gpu.index, 0)  # non LLMs
            )
            gpu_sum += 1
            vram_sum += vram_claim[gpu.index]

            if self._num_attention_heads and self._num_attention_heads % gpu_sum != 0:
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
                        ram=self._ram_claim if self._ram_claim > 0 else None,
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

    async def manual_select_single_worker_multi_gpu_candidates(  # noqa: C901
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:

        if not worker.status or not worker.status.gpu_devices:
            return []

        allocatable = await self._get_worker_allocatable_resource(worker)

        selected_gpu_indexes = self._selected_gpu_indexes_by_worker.get(worker.name)
        if selected_gpu_indexes is None:
            return []

        sorted_gpu_indexes_by_allocatable_rate = sort_gpu_indexes_by_allocatable_rate(
            worker, allocatable.vram
        )
        gpu_set = {gpu.index: gpu for gpu in worker.status.gpu_devices}

        overcommit = False
        total_allocatable_vram = 0
        satisfied_gpu_count = 0
        satisfied_gpu_indexes = []
        vram_claims = {}
        for gpu_index in sorted_gpu_indexes_by_allocatable_rate:
            if gpu_index not in selected_gpu_indexes:
                continue

            gpu = gpu_set.get(gpu_index)
            if not gpu:
                continue

            if gpu.memory is None or gpu.memory.total is None:
                continue
            vram_claim = (
                int(gpu.memory.total * self._gpu_memory_utilization)
                if self._gpu_memory_utilization > 0  # LLMs
                else int(self._vram_claim / len(selected_gpu_indexes))  # non LLMs
            )
            vram_claims[gpu.index] = vram_claim

            # Check if the GPU is overcommitted
            allocatable_vram = allocatable.vram.get(gpu.index, 0)
            allocatable_gpu_memory_utilization = allocatable_vram / gpu.memory.total
            if (
                self._gpu_memory_utilization > 0
                and allocatable_gpu_memory_utilization < self._gpu_memory_utilization
            ):
                overcommit = True
                if worker.name not in self._unsatisfied_gpu_messages:
                    self._unsatisfied_gpu_messages[worker.name] = []
                self._unsatisfied_gpu_messages[worker.name].append(gpu.index)

            # Record allocation info for scheduling message
            total_allocatable_vram += allocatable_vram
            if allocatable_vram / gpu.memory.total > self._gpu_memory_utilization:
                satisfied_gpu_count += 1
                satisfied_gpu_indexes.append(gpu.index)

            if self._gpu_count and satisfied_gpu_count >= self._gpu_count:
                break

        self._largest_multi_gpu_vram = total_allocatable_vram
        self._largest_multi_gpu_utilization_satisfied_count = satisfied_gpu_count
        self._largest_multi_gpu_total = len(selected_gpu_indexes)

        if sum(vram_claims.values()) < self._vram_claim:
            overcommit = True
            # Construct event collector.
            scheduling_msg = ListMessageBuilder([])
            if self._gpu_memory_utilization != 0:
                scheduling_msg.append(
                    f"Selected GPUs have {byte_to_gib(self._largest_multi_gpu_vram)} GiB of VRAM."
                )
            else:
                unsatisfied_devices_msg = str(
                    self._unsatisfied_gpu_messages[worker.name]
                )
                if len(self._unsatisfied_gpu_messages[worker.name]) > 3:
                    unsatisfied_devices_msg = (
                        str(self._unsatisfied_gpu_messages[worker.name][:3]).rstrip("]")
                        + f"...(more {len(self._unsatisfied_gpu_messages[worker.name]) - 3})]"
                    )
                scheduling_msg.extend(
                    [
                        f"Worker {worker.name} GPU indexes {unsatisfied_devices_msg} fail to meet the {(self._gpu_memory_utilization * 100):.2f}% allocatable VRAM ratio.",
                        f"Selected GPUs have {byte_to_gib(self._largest_multi_gpu_vram)} GiB allocatable VRAM, {self._largest_multi_gpu_utilization_satisfied_count}/{self._largest_multi_gpu_total} of GPUs meet the VRAM utilization ratio, providing {self._cal_effective_vram():.2f} GiB of allocatable VRAM.",
                    ]
                )
            self._event_collector.add(
                EventLevelEnum.INFO, EVENT_ACTION_MANUAL_MULTI, str(scheduling_msg)
            )

        return [
            ModelInstanceScheduleCandidate(
                worker=worker,
                gpu_indexes=satisfied_gpu_indexes,
                computed_resource_claim=ComputedResourceClaim(
                    vram=vram_claims,
                ),
                overcommit=overcommit,
            )
        ]

    async def find_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        if not self._cfg.enable_ray:
            return []

        if self._model.backend_version:
            # Using custom backend version to run vLLM across multiple workers is not supported.
            return []

        if self._selected_gpu_workers:
            return await self.manual_select_multi_worker_multi_gpu_candidates(workers)

        if self._model.distributed_inference_across_workers:
            return await self.auto_select_multi_worker_multi_gpu_candidates(workers)

        return []

    async def auto_select_multi_worker_multi_gpu_candidates(  # noqa: C901
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Auto select multi worker multi gpu candidates.
        Currently, a candidate should match the following conditions:
        1. Workers in the candidate have the same number of GPUs.
        2. All GPUs in the worker satisfy the gpu_memory_utilization requirement.
        3. The total number of GPUs can be divided by the number of attention heads.
        4. The total VRAM claim is greater than the estimated VRAM claim.
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

        # Loop through worker groups with the same number of GPUs.
        for gpu_count, worker_group in workers_by_gpu_count_dict.items():
            if len(worker_group) < 2:
                continue

            selected_workers: List[Worker] = []
            gpu_sum = 0
            vram_sum = 0
            for worker in worker_group:
                allocatable = await self._get_worker_allocatable_resource(worker)

                if self._ram_claim > 0 and allocatable.ram < self._ram_claim:
                    # The RAM resource(for extended KV cache) is required per worker.
                    # Skip the worker if it does not satisfy the RAM requirement.
                    continue

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
                selected_workers.append(worker)
                gpu_sum += gpu_count
                vram_sum += sum(
                    int(gpu.memory.total * (self._gpu_memory_utilization or 1))
                    for gpu in worker.status.gpu_devices
                )

                if (
                    self._num_attention_heads
                    and self._num_attention_heads % gpu_sum == 0
                ) and (vram_sum >= self._vram_claim):
                    return [
                        _create_candidate(
                            selected_workers,
                            self._gpu_memory_utilization,
                            self._ram_claim,
                        )
                    ]
            if vram_sum > largest_vram:
                workers_combination = selected_workers
                largest_vram = vram_sum
                worker_count = len(worker_group)
                device_count_per_worker = gpu_count

        # Nothing can be return, construct scheduling message
        event_message = ListMessageBuilder([])
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

    async def manual_select_multi_worker_multi_gpu_candidates(  # noqa: C901
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get manually selected multi worker multi gpu candidates.
        """
        if not self._selected_gpu_workers or len(self._selected_gpu_workers) < 2:
            return []

        if not workers:
            return []

        sort_workers_by_gpu_count(workers)

        main_worker = None
        main_gpu_indexes = None
        main_vram_claim = None

        subordinate_workers: List[ModelInstanceSubordinateWorker] = []
        for worker in workers:
            if worker.name not in self._selected_gpu_workers:
                continue

            if not await self._validate_distributed_vllm_limit_per_worker(worker):
                continue

            if main_worker is None:
                main_worker = worker
                main_gpu_indexes = self._selected_gpu_indexes_by_worker[
                    main_worker.name
                ]
                main_vram_claim = await self._get_worker_vram_claim(
                    main_worker, main_gpu_indexes, self._gpu_memory_utilization
                )
                continue

            if worker.name == main_worker.name:
                continue

            selected_gpu_indexes = self._selected_gpu_indexes_by_worker[worker.name]
            selected_vram_claim = await self._get_worker_vram_claim(
                worker, selected_gpu_indexes, self._gpu_memory_utilization
            )

            gpu_indexes = []
            vram_claim = {}
            sorted_gpu_indexes = [
                idx
                for idx in sort_gpu_indexes_by_allocatable_rate(
                    worker, selected_vram_claim
                )
                if idx in selected_gpu_indexes
            ]

            if self._model.gpu_selector and self._model.gpu_selector.gpus_per_replica:
                current_gpu_count = len(main_gpu_indexes) + sum(
                    len(sw.gpu_indexes) for sw in subordinate_workers
                )
                current = min(
                    len(selected_gpu_indexes),
                    self._model.gpu_selector.gpus_per_replica - current_gpu_count,
                )
                gpu_indexes = sorted_gpu_indexes[:current]
                vram_claim = {idx: selected_vram_claim[idx] for idx in gpu_indexes}

            subordinate_workers.append(
                ModelInstanceSubordinateWorker(
                    worker_id=worker.id,
                    worker_name=worker.name,
                    worker_ip=worker.ip,
                    total_gpus=len(worker.status.gpu_devices),
                    gpu_indexes=gpu_indexes,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=self._ram_claim if self._ram_claim > 0 else None,
                    ),
                )
            )

        overcommit = (
            self._largest_multi_gpu_utilization_satisfied_count
            < self._largest_multi_gpu_total
            or self._largest_multi_gpu_vram * self._gpu_memory_utilization
            < self._vram_claim
        )

        if overcommit:
            scheduling_msg = ListMessageBuilder([])
            if self._gpu_memory_utilization == 0:
                scheduling_msg.append(
                    f"Selected GPUs have {byte_to_gib(self._largest_multi_gpu_vram)} GiB of VRAM."
                )
            else:
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
                    unsatisfied_msg += f" {'fail' if len(self._unsatisfied_gpu_messages) > 1 else 'fails'} to meet the {(self._gpu_memory_utilization * 100):.2f}% allocatable VRAM ratio."
                    scheduling_msg.append(unsatisfied_msg)
                    break

                # All selected gpus combined VRAM is not enough.
                scheduling_msg.append(
                    f"Selected GPUs have {byte_to_gib(self._largest_multi_gpu_vram)} GiB allocatable VRAM, "
                    f"{self._largest_multi_gpu_utilization_satisfied_count}/{self._largest_multi_gpu_total} of GPUs meet the VRAM utilization ratio, providing {self._cal_effective_vram():.2f} GiB of allocatable VRAM."
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
                    ram=self._ram_claim if self._ram_claim > 0 else None,
                ),
                subordinate_workers=subordinate_workers,
                overcommit=overcommit,
            )
        ]

    async def _validate_distributed_vllm_limit_per_worker(self, worker: Worker) -> bool:
        """
        Validate that there is no more than one distributed vLLM instance per worker.
        """
        instances = await get_worker_model_instances(self._engine, worker)
        for instance in instances:
            if (
                instance.distributed_servers
                and instance.distributed_servers.subordinate_workers
            ):
                self._messages = [
                    f"Each worker can run only one distributed vLLM instance. Worker '{worker.name}' already has '{instance.name}'."
                ]
                return False

        return True

    async def _get_worker_vram_claim(
        self,
        worker: Worker,
        gpu_indexes: List[int],
        gpu_memory_utilization: float = 0.9,
    ) -> Dict[int, int]:
        """
        Given a worker and gpu indexes, get the vram claim according to gpu_memory_utilization.
        Returns a dictionary of gpu index to vram claim in bytes, and a boolean indicating
        whether the claim exceeds the allocatable vram.
        """
        vram_claim: Dict[int, int] = {}

        allocatable = await self._get_worker_allocatable_resource(worker)
        for gpu in worker.status.gpu_devices:
            if gpu.index in gpu_indexes:
                gpu_vram_claim = int(gpu.memory.total * gpu_memory_utilization)
                vram_claim[gpu.index] = gpu_vram_claim

                # Record allocation info for scheduling message
                allocatable_vram = allocatable.vram.get(gpu.index, 0)
                self._largest_multi_gpu_vram += allocatable_vram
                if gpu_vram_claim < allocatable_vram:
                    self._largest_multi_gpu_utilization_satisfied_count += 1
                else:
                    if worker.name not in self._unsatisfied_gpu_messages:
                        self._unsatisfied_gpu_messages[worker.name] = []
                    self._unsatisfied_gpu_messages[worker.name].append(gpu.index)

        self._largest_multi_gpu_total += len(gpu_indexes)

        return vram_claim


def _create_candidate(
    selected_workers: List[Worker],
    gpu_memory_utilization: float = 0.9,
    ram_claim: int = 0,
) -> ModelInstanceScheduleCandidate:
    """
    Create a candidate with all GPUs from the selected workers.
    """
    main_worker = selected_workers[0]
    candidate = ModelInstanceScheduleCandidate(
        worker=main_worker,
        gpu_indexes=[gpu.index for gpu in main_worker.status.gpu_devices],
        computed_resource_claim=ComputedResourceClaim(
            vram={
                gpu.index: int(gpu.memory.total * gpu_memory_utilization)
                for gpu in main_worker.status.gpu_devices
            },
            ram=ram_claim if ram_claim > 0 else None,
        ),
    )
    candidate.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=worker.id,
            worker_name=worker.name,
            worker_ip=worker.ip,
            total_gpus=len(worker.status.gpu_devices),
            gpu_indexes=[gpu.index for gpu in worker.status.gpu_devices],
            computed_resource_claim=ComputedResourceClaim(
                vram={
                    gpu.index: int(gpu.memory.total * gpu_memory_utilization)
                    for gpu in worker.status.gpu_devices
                },
                ram=ram_claim if ram_claim > 0 else None,
            ),
        )
        for worker in selected_workers[1:]
    ]
    return candidate


def sort_workers_by_gpu_count(workers: List[Worker]):
    """
    Sort workers by the number of GPUs.
    """
    workers.sort(
        key=lambda worker: (
            len(worker.status.gpu_devices)
            if worker.status and worker.status.gpu_devices
            else 0
        ),
        reverse=True,
    )


def sort_gpu_indexes_by_allocatable_rate(
    worker: Worker, allocatable: dict
) -> List[int]:
    """
    Sort GPU indexes of a worker by allocatable VRAM rate (allocatable_vram / total_vram), ascending.
    """
    allocatable_rate = {
        gpu.index: (
            allocatable.get(gpu.index, 0) / gpu.memory.total
            if gpu.memory and gpu.memory.total
            else 0
        )
        for gpu in worker.status.gpu_devices
    }

    # return a list of gpu indexes sorted by allocatable rate in ascending order
    return sorted(allocatable_rate, key=lambda idx: allocatable_rate[idx])
