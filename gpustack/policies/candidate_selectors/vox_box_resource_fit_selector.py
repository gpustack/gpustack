from abc import ABC, abstractmethod
import asyncio

from gpustack_runtime.detector import ManufacturerEnum

from gpustack.config.config import Config
from gpustack.utils.convert import safe_int
from gpustack.utils.file import get_local_file_size_in_byte
from gpustack.utils.gpu import parse_gpu_id
from gpustack.utils.hub import get_model_weight_size
from typing import Dict, List
import os
import logging

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
)
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    ScheduleCandidatesSelector,
)
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
)
from gpustack.schemas.workers import Worker

logger = logging.getLogger(__name__)

Mib = 1024 * 1024
Gib = 1024 * 1024 * 1024


class VoxBoxResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        config: Config,
        model: Model,
        cache_dir: str,
    ):
        super().__init__(config, model, parse_model_params=False)
        self._cache_dir = os.path.join(cache_dir, "vox-box")
        self._messages = []

        self._gpu_ram_claim = 0
        self._gpu_vram_claim = 0
        self._cpu_ram_claim = 0
        self._required_os = None
        self._selected_gpu_worker = None
        self._selected_gpu_index = None

        if self._model.gpu_selector and self._model.gpu_selector.gpu_ids:
            valid, match = parse_gpu_id(self._model.gpu_selector.gpu_ids[0])
            if valid:
                self._selected_gpu_worker = match.get("worker_name")
                self._selected_gpu_index = safe_int(match.get("gpu_index"))

    def _set_messages(self):
        self._messages = ["No workers meet the resource requirements."]

    def get_messages(self) -> List[str]:
        return self._messages

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement.
        """

        timeout_in_seconds = 15
        resource_claim = await asyncio.wait_for(
            asyncio.to_thread(
                estimate_model_resource, self._config, self._model, self._cache_dir
            ),
            timeout=timeout_in_seconds,
        )
        if resource_claim is not None:
            self._gpu_ram_claim = resource_claim.get("cuda", {}).get("ram", 0)
            self._gpu_vram_claim = resource_claim.get("cuda", {}).get("vram", 0)
            self._cpu_ram_claim = resource_claim.get("cpu", {}).get("ram", 0)
            self._required_os = resource_claim.get("os", None)

            logger.info(
                f"Calculated resource claim for model {self._model.readable_source}, "
                f"gpu vram claim: {self._gpu_vram_claim}, gpu ram claim: {self._gpu_ram_claim}, cpu ram claim: {self._cpu_ram_claim}"
            )

        candidate_functions = [
            self.find_single_worker_single_gpu_candidates,
            self.find_single_worker_cpu_candidates,
        ]

        for candidate_func in candidate_functions:
            logger.debug(
                f"model {self._model.readable_source}, filter candidates with resource fit selector: {candidate_func.__name__}"
            )

            candidates = await candidate_func(workers)
            if candidates:
                return candidates

        self._set_messages()
        return []

    async def find_single_worker_single_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu candidates for the model instance with workers.
        """

        candidates = []
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            result = await self._find_single_worker_single_gpu_candidates(worker)
            if result:
                candidates.extend(result)

        return candidates

    async def _find_single_worker_single_gpu_candidates(
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu candidates for the model instance with worker.
        requires: worker.status.gpu_devices is not None
        """

        if self._selected_gpu_worker and worker.name != self._selected_gpu_worker:
            return []

        candidates = []

        if (
            self._required_os
            and worker.labels.get("os")
            and worker.labels["os"].lower() not in self._required_os
        ):
            return []

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        is_unified_memory = worker.status.memory.is_unified_memory

        if self._gpu_ram_claim > allocatable.ram:
            return []

        if worker.status.gpu_devices:
            for _, gpu in enumerate(worker.status.gpu_devices):
                if (
                    self._selected_gpu_index is not None
                    and gpu.index != self._selected_gpu_index
                ):
                    continue

                if gpu.vendor != ManufacturerEnum.NVIDIA.value:
                    continue

                gpu_index = gpu.index
                allocatable_vram = allocatable.vram.get(gpu_index, 0)

                if gpu.memory is None or gpu.memory.total == 0:
                    continue

                if self._gpu_vram_claim > allocatable_vram:
                    continue

                candidates.append(
                    ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_indexes=[gpu_index],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={gpu_index: int(self._gpu_vram_claim)},
                            ram=self._gpu_ram_claim,
                            is_unified_memory=is_unified_memory,
                        ),
                    )
                )

        return candidates

    async def find_single_worker_cpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker without offloading candidates for the model instance with workers.
        """
        candidates = []
        for worker in workers:
            result = await self._find_single_worker_with_cpu_candidates(worker)
            if result:
                candidates.extend(result)
        return candidates

    async def _find_single_worker_with_cpu_candidates(
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker without offloading candidates for the model instance.
        """

        if (
            self._required_os
            and worker.labels.get("os")
            and worker.labels["os"].lower() not in self._required_os
        ):
            return []

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        is_unified_memory = worker.status.memory.is_unified_memory

        if self._cpu_ram_claim > allocatable.ram:
            return []

        return [
            ModelInstanceScheduleCandidate(
                worker=worker,
                gpu_indexes=None,
                computed_resource_claim=ComputedResourceClaim(
                    is_unified_memory=is_unified_memory,
                    vram=None,
                    ram=self._cpu_ram_claim,
                ),
            )
        ]


def estimate_model_resource(cfg: Config, model: Model, cache_dir: str) -> dict:
    try:
        from vox_box.estimator.estimate import estimate_model
        from vox_box.config import Config as VoxBoxConfig
    except ImportError:
        raise Exception("vox_box is not installed.")

    if model.local_path is not None and not os.path.exists(model.local_path):
        logger.debug(f"Model {model.name} local path {model.local_path} does not exist")
        return

    box_cfg = VoxBoxConfig()
    box_cfg.cache_dir = cache_dir
    box_cfg.model = model.local_path
    box_cfg.huggingface_repo_id = model.huggingface_repo_id
    box_cfg.model_scope_model_id = model.model_scope_model_id

    try:
        model_dict = estimate_model(box_cfg)
    except Exception as e:
        logger.error(f"Failed to estimate model {model.name}: {e}")
        return

    if model_dict is None:
        logger.debug(f"model_dict is empty after estimate model {model.name}")
        return

    framework = model_dict.get("backend_framework", "")
    if framework == "":
        logger.error(f"Unsupported audio model {model.name}")
        return

    framework_mapping = {
        "Bark": Bark,
        "CosyVoice": CosyVoice,
        "Dia": Dia,
        "FasterWhisper": FasterWhisper,
        "FunASR": FunASR,
    }

    framework_class = framework_mapping.get(framework)
    if not framework_class:
        logger.error(f"Unsupported framework {framework} for model {model.name}")
        return {}

    return framework_class(cfg, model, model_dict).get_required_resource()


class BaseModelResourceEstimator(ABC):
    def __init__(self, cfg: Config, model: Model, model_info: Dict):
        self._cfg = cfg
        self._model = model
        self._model_info = model_info

    @abstractmethod
    def get_required_resource(self) -> Dict:
        pass


class FasterWhisper(BaseModelResourceEstimator):
    def get_required_resource(self) -> Dict:
        """
        File size from https://huggingface.co/Systran
        | large            | Size   | Size (MiB/GiB) |
        | ---------------- | ------ | -------------- |
        | tiny en          | 75MB   | 71.53 MiB      |
        | tiny             | 75MB   | 71.53 MiB      |
        | base en          | 145MB  | 138.67 MiB     |
        | base             | 145MB  | 138.67 MiB     |
        | distil small en  | 332MB  | 316.41 MiB     |
        | small en         | 484MB  | 461.91 MiB     |
        | small            | 484MB  | 461.91 MiB     |
        | distil medium en | 789MB  | 752.93 MiB     |
        | medium en        | 1.53G  | 1.42 GiB       |
        | medium           | 1.52G  | 1.41 GiB       |
        | distil large v2  | 1.51G  | 1.41 GiB       |
        | distil large v3  | 1.51G  | 1.41 GiB       |
        | turbo            | 1.62G  | 1.51 GiB       |
        | large v3         | 3.09GB | 2.88 GiB       |
        | large v2         | 3.09GB | 2.88 GiB       |
        | large v1         | 3.09GB | 2.88 GiB       |

        Resource required from:
        https://github.com/openai/whisper?tab=readme-ov-file
        | Size         | Parameters | English-only model | Multilingual model | Required VRAM  | Relative speed |
        | ------------ | ---------- | ------------------ | ------------------ | -------------  | -------------- |
        | tiny         | 39 M       | tiny.en            | tiny               | ~1 GB          | ~10x           |
        | base         | 74 M       | base.en            | base               | ~1 GB          | ~7x            |
        | small        | 244 M      | small.en           | small              | ~2 GB          | ~4x            |
        | medium       | 769 M      | medium.en          | medium             | ~5 GB          | ~2x            |
        | distil-large | 756 M      | N/A                | distil-large       |                |                |
        | large        | 1550 M     | N/A                | large              | ~10 GB         | 1x             |
        | turbo        | 809 M      | N/A                | turbo              | ~6 GB          | 8x             |
        """

        # Here we simply assume that the resources used for CPU inference are 1.5 times those used for GPU inference.
        resource_requirements = {
            # tiny
            100 * Mib: {"cuda": {"vram": 1 * Gib}, "cpu": {"ram": 1.5 * Gib}},
            # base
            200 * Mib: {"cuda": {"vram": 1 * Gib}, "cpu": {"ram": 1.5 * Gib}},
            # small
            500 * Mib: {"cuda": {"vram": 2 * Gib}, "cpu": {"ram": 3 * Gib}},
            # medium
            1.5 * Gib: {"cuda": {"vram": 5 * Gib}, "cpu": {"ram": 7.5 * Gib}},
            # turbo
            1.6 * Gib: {"cuda": {"vram": 6 * Gib}, "cpu": {"ram": 9 * Gib}},
            # large
            3.1 * Gib: {"cuda": {"vram": 10 * Gib}, "cpu": {"ram": 15 * Gib}},
        }

        return get_model_resource_requirement_from_file_size(
            self._model, self._cfg, resource_requirements, "model.bin"
        )


class Bark(BaseModelResourceEstimator):
    def get_required_resource(self) -> Dict:
        """
        Main model file size from https://huggingface.co/suno
        | large            | Size   | Size (MiB/GiB) |
        | ---------------- | ------ | -------------- |
        | bark             | 4.49G  | 4.28 GiB       |
        | bark-small       | 1.68G  | 1.60 GiB       |

        Resource required from:
        https://github.com/suno-ai/bark?tab=readme-ov-file#how-much-vram-do-i-need
        | Size         | Required VRAM  |
        | ------------ | -------------  |
        | bark-small   | ~2 GB          |
        | bark         | ~12 GB         |
        """

        # Here we simply assume that the resources used for CPU inference are 1.5 times those used for GPU inference.
        resource_requirements = {
            # small
            2 * Gib: {"cuda": {"vram": 2 * Gib}, "cpu": {"ram": 3 * Gib}},
            # large
            5 * Gib: {"cuda": {"vram": 12 * Gib}, "cpu": {"ram": 18 * Gib}},
        }
        return get_model_resource_requirement_from_file_size(
            self._model, self._cfg, resource_requirements, "pytorch_model.bin"
        )


class CosyVoice(BaseModelResourceEstimator):
    def get_required_resource(self) -> Dict:
        # The required resource values used here are based on test estimates
        # and may not accurately reflect actual requirements. Adjustments might be
        # necessary based on real-world scenarios in feature.
        return {
            "cuda": {"vram": 3 * Gib, "ram": 7 * Gib},
            "cpu": {"ram": 6 * Gib},
        }


class Dia(BaseModelResourceEstimator):
    def get_required_resource(self) -> Dict:
        # The required resource values used here are based on test estimates
        # and may not accurately reflect actual requirements. Adjustments might be
        # necessary based on real-world scenarios in feature.
        return {
            "cuda": {"vram": 10 * Gib, "ram": 1 * Gib},
            "cpu": {"ram": 10 * Gib},
        }


class FunASR(BaseModelResourceEstimator):
    def get_required_resource(self) -> Dict:
        # TODO: Update the resource requirements based on the test.
        return {}


def get_model_resource_requirement_from_file_size(
    model: Model, cfg: Config, resource_requirements: dict, file_path: str
) -> int:
    file_size_in_byte = -1
    if model.local_path is not None:
        file_size_in_byte = get_local_file_size_in_byte(
            os.path.join(model.local_path, file_path)
        )
    else:
        file_size_in_byte = get_model_weight_size(model, cfg.huggingface_token)
    for size, resource in resource_requirements.items():
        if file_size_in_byte <= size:
            return resource
