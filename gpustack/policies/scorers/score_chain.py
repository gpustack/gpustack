from dataclasses import dataclass
from typing import List, Optional

from gpustack.policies.base import (
    ModelInstance,
    ModelInstanceScore,
    ModelInstanceScorer,
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesScorer,
)


@dataclass
class CandidateScoreChain:
    scorers: List[ScheduleCandidatesScorer]

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        if not candidates:
            return candidates

        total_scores = {id(candidate): 0.0 for candidate in candidates}

        for scorer in self.scorers:
            # Reset candidate scores so each scorer computes its own score.
            for candidate in candidates:
                candidate.score = None

            scored = await scorer.score(candidates)
            for candidate in scored:
                total_scores[id(candidate)] += candidate.score or 0

        for candidate in candidates:
            candidate.score = total_scores[id(candidate)]

        return candidates


class ModelInstanceScoreChain:
    def __init__(
        self,
        scorers: List[ModelInstanceScorer],
        total_max_score: Optional[float] = None,
    ):
        self._scorers = [s for s in scorers if s.max_score and s.max_score > 0]
        self._total_max_score = total_max_score
        self._sum_max_score = sum(s.max_score for s in self._scorers)
        if self._total_max_score is None or self._sum_max_score == 0:
            self._scale_factor = 1.0
        else:
            self._scale_factor = self._total_max_score / self._sum_max_score

    async def score(self, instances: List[ModelInstance]) -> List[ModelInstanceScore]:
        if not instances:
            return []

        if not self._scorers:
            return [
                ModelInstanceScore(model_instance=instance, score=0)
                for instance in instances
            ]

        if self._sum_max_score == 0:
            return [
                ModelInstanceScore(model_instance=instance, score=0)
                for instance in instances
            ]

        score_sum = {id(instance): 0.0 for instance in instances}
        instance_map = {id(instance): instance for instance in instances}

        for scorer in self._scorers:
            scores = await scorer.score_instances(instances)
            for item in scores:
                key = id(item.model_instance)
                if key in score_sum:
                    score_sum[key] += item.score or 0

        results = []
        for key, instance in instance_map.items():
            score = score_sum[key] * self._scale_factor
            results.append(ModelInstanceScore(model_instance=instance, score=score))

        return results
