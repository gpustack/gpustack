import logging
from typing import List, Optional, Set

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesScorer,
)
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.models import Model, ModelSource, SourceEnum
from gpustack.server.db import async_session
from gpustack.server.services import ModelFileService

logger = logging.getLogger(__name__)


class ModelFileLocalityScorer(ScheduleCandidatesScorer):
    def __init__(
        self,
        model: Model,
        draft_model_source: Optional[ModelSource] = None,
        max_score: float = 5.0,
    ):
        self._model = model
        self._draft_model_source = draft_model_source
        self._max_score = max_score

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        if not candidates or self._max_score <= 0:
            return candidates

        try:
            async with async_session() as session:
                ready_main_workers = await self._get_ready_worker_ids(
                    session, self._model
                )
                ready_draft_workers: Set[int] = set()
                if self._draft_model_source is not None:
                    ready_draft_workers = await self._get_ready_worker_ids(
                        session, self._draft_model_source
                    )
        except Exception as e:
            logger.warning(
                "Failed to load model file locality data, skip locality score: %s", e
            )
            return candidates

        for candidate in candidates:
            candidate.score = self._calculate_score(
                candidate, ready_main_workers, ready_draft_workers
            )

        return candidates

    async def _get_ready_worker_ids(self, session, source: ModelSource) -> Set[int]:
        source_index = source.model_source_index
        if not source_index:
            return set()

        model_files = (
            await ModelFileService(session).get_by_source_index(source_index) or []
        )

        if source.source == SourceEnum.LOCAL_PATH and source.local_path:
            local_path_files = await ModelFileService(session).get_by_resolved_path(
                source.local_path
            )
            if local_path_files:
                model_files = model_files + local_path_files

        ready_worker_ids = set()
        for model_file in model_files:
            if (
                model_file.state == ModelFileStateEnum.READY
                and model_file.worker_id is not None
            ):
                ready_worker_ids.add(model_file.worker_id)

        return ready_worker_ids

    def _calculate_score(
        self,
        candidate: ModelInstanceScheduleCandidate,
        ready_main_workers: Set[int],
        ready_draft_workers: Set[int],
    ) -> float:
        worker_ids = self._get_candidate_worker_ids(candidate)
        if not worker_ids:
            return 0.0

        main_ratio = 0.0
        if ready_main_workers:
            main_ratio = len(worker_ids & ready_main_workers) / len(worker_ids)

        draft_ratio = 0.0
        if ready_draft_workers:
            draft_ratio = len(worker_ids & ready_draft_workers) / len(worker_ids)

        if main_ratio == 0.0 and draft_ratio == 0.0:
            return 0.0

        # Prefer main model locality; draft is a secondary signal.
        draft_weight = 0.5 if ready_draft_workers else 0.0
        denom = 1.0 + draft_weight
        ratio = (main_ratio + draft_weight * draft_ratio) / denom
        locality_score = min(self._max_score, ratio * self._max_score)
        return locality_score

    def _get_candidate_worker_ids(
        self, candidate: ModelInstanceScheduleCandidate
    ) -> Set[int]:
        worker_ids: Set[int] = set()
        if candidate.worker and candidate.worker.id is not None:
            worker_ids.add(candidate.worker.id)

        if candidate.subordinate_workers:
            for subworker in candidate.subordinate_workers:
                if subworker.worker_id is not None:
                    worker_ids.add(subworker.worker_id)

        return worker_ids
