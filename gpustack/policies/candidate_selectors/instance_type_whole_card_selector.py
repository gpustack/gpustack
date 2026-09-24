"""Several whole cards from one InstanceType pool, on one worker.

`VGPUResourceFitSelector` answers "one slice per worker" and that is correct
for what it was written for: a slice is a fraction of a card the node's device
plugin picks at allocation time, so there is no way to say "and another one on
the same host" — the caller does not know which card the first one landed on.
Needing more than one slice therefore means more workers, one slice each.

**Whole-card exclusive mode has none of that difficulty, and inherits the
restriction anyway.** With all slicing percentages at zero the claim is for
entire cards, the operator's own resource model hands out several at once
(`Accelerator` is documented as `"1"`, `"4"`), and the container ends up seeing
exactly the devices the plugin allocated — so an engine told `tp=4` finds four
cards without anyone naming them. What was missing was a claim shaped like four
cards instead of one.

**Subclassed rather than copied.** Pool matching, the node-capability check
against the cluster's `Devices`, and the message wording are the parts most
likely to drift if duplicated, and they are identical here. What differs is two
things: how many cards the claim covers, and that they must be on *one* worker.

Reads `gpu_indexes=None` like its parent, deliberately. Which physical cards
serve the claim is still the device plugin's call; the claim's keys are
placeholders `0..N-1` and the allocation is re-keyed on read-back. That is the
existing contract for this path, not a new compromise — the one-card case has
always used placeholder index 0.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from gpustack.config.config import Config
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.candidate_selectors.vgpu_resource_fit_selector import (
    VGPUResourceFitSelector,
)
from gpustack.schemas.gpu_instance_types import GPUInstanceTypeDetail
from gpustack.schemas.models import (
    ComputedResourceClaim,
    GPUTypeSelector,
    Model,
    ModelInstance,
)
from gpustack.schemas.workers import Worker

logger = logging.getLogger(__name__)


def is_whole_card_claim(selector: Optional[GPUTypeSelector]) -> bool:
    """Whether this `gpu_type_selector` asks for entire cards.

    All three slicing dimensions at rest: no partition profile, and both
    percentages zero or unset. This is the same condition
    `VGPUResourceFitSelector._get_slice_vram` uses to return the full card
    VRAM, kept as a function so the factory and the selector cannot disagree
    about which mode they are in.
    """
    if selector is None:
        return False
    if selector.accelerator_partitioned_profile:
        return False
    memory = selector.accelerator_sliced_memory_percentage or 0
    cores = selector.accelerator_sliced_cores_percentage or 0
    return memory == 0 and cores == 0


class InstanceTypeWholeCardSelector(VGPUResourceFitSelector):
    """Whole cards from a pool, `cards_per_member` of them on one worker."""

    def __init__(
        self,
        config: Config,
        model: Model,
        model_instances: List[ModelInstance],
        cards_per_member: int = 1,
    ):
        super().__init__(config, model, model_instances)
        # Never below one: a member that occupies an accelerator occupies at
        # least one, and a zero here would make the claim empty and the worker
        # look free.
        self._cards = max(int(cards_per_member or 1), 1)

    def get_messages(self) -> List[str]:
        return self._messages

    def _cards_per_member(self) -> int:
        return self._cards

    def _member_vram_capacity(self) -> int:
        """Every card the member takes, not one of them.

        A whole-card member is `cards_per_member` cards on ONE host, so its
        capacity there is their sum. Measured against a single card, a member
        needing two or more failed the parent's single-host test and fell into
        the spread-across-hosts branch — whose candidates this selector then
        refuses by design, leaving exactly the multi-card members it exists to
        place with nowhere to go.
        """
        return self._cards * self._slice_vram

    def _create_slice_claim(self) -> ComputedResourceClaim:
        """One entry per card, not one entry.

        The keys are placeholder indexes, exactly as the parent's single entry
        is a placeholder — `compute_worker_allocated` sums them, and the sum is
        what decides whether the next claim fits. Per-index attribution is
        wrong either way on this path and is corrected from the allocation
        read-back; the total is what has to be right, and with one entry it was
        short by a factor of `cards_per_member`.
        """
        return ComputedResourceClaim(
            vram={index: self._slice_vram for index in range(self._cards)},
            ram=self._ram_claim,
        )

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """Whole-card claims never spread a member across workers.

        The parent falls back to one slice per worker across
        `slices_needed` hosts when a model outgrows a single slice. For whole
        cards that fallback would put a member's tensor-parallel group on
        several machines, which is the most expensive way to split it — two
        all-reduces per layer over the network. A member that does not fit on
        one host is refused here, and crossing hosts stays something the user
        asks for explicitly.
        """
        candidates = await super().select_candidates(workers)
        if not candidates:
            return candidates

        single_host = [c for c in candidates if not c.subordinate_workers]
        if not single_host and candidates:
            self._messages = self._messages or [
                f"The group's members need {self._cards} whole cards each on a "
                "single worker, and no worker in the pool has that many free. "
                "Spreading one member across workers would put its "
                "tensor-parallel group on several machines; enable "
                "distributed inference across workers if that is intended."
            ]
            return []
        return single_host

    def _create_candidate(
        self,
        worker: Worker,
        detail: GPUInstanceTypeDetail,
        subordinate_workers: Optional[List[Worker]] = None,
    ) -> ModelInstanceScheduleCandidate:
        # Subordinates dropped rather than passed through: this selector's
        # whole contract is "on one worker", and a candidate carrying
        # subordinates would be filtered out by `select_candidates` anyway.
        # Building it and then discarding it costs a claim per host.
        return super()._create_candidate(worker, detail, None)
