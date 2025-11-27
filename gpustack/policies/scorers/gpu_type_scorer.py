from typing import List
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.schemas.models import Model


class GPUTypeScorer:
    # GPU type is from runtime MANUFACTURER_BACKEND_MAPPING, update this once runtime updates
    # https://github.com/gpustack/runtime/blob/696d00f428d52a5b4e18b24ac73990871f04d8bb/gpustack_runtime/detector/__types__.py#L57
    GPU_TYPE_SCORE_MAP = {
        "cuda": 100,
        "cann": 90,
        "rocm": 80,
        "dtk": 70,
        "musa": 60,
        "neuware": 50,
        "corex": 50,
        "maca": 50,
    }
    DEFAULT_SCORE = 40

    def __init__(self, model: Model):
        self._model = model

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        for candidate in candidates:
            gpu_type = candidate.gpu_type
            score = self.GPU_TYPE_SCORE_MAP.get(gpu_type, self.DEFAULT_SCORE)
            candidate.score = score
        return candidates
