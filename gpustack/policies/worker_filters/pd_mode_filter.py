import logging
from typing import List, Optional, Set, Tuple

from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker
from gpustack.server.pd_mode_catalog import get_pd_mode

logger = logging.getLogger(__name__)


def required_vendors(model: Model) -> Optional[Set[str]]:
    """Which accelerator vendors may host this model's PD group.

    Two independent sources, and the group has to satisfy both:

    - the **recipe's** declaration (``gpu_filters.vendor``) -- what the
      connector can talk to at all;
    - the **group's** explicit ``disaggregation.vendor`` -- the partition a
      user picked in a mixed cluster, which is a placement constraint rather
      than a preference, since a PD group cannot span vendors.

    None means unconstrained, and that is the state ``custom`` is in by
    design: it injects nothing, so it must stay runnable on every
    accelerator. An unsupported engine × accelerator pair means "no built-in
    recipe", never "no PD".
    """
    disaggregation = getattr(model, "disaggregation", None)
    if not disaggregation:
        return None

    mode_name = disaggregation.mode.value
    mode = get_pd_mode(mode_name)
    from_recipe = (
        {v.lower() for v in mode.gpu_filters.vendor}
        if mode is not None and mode.gpu_filters and mode.gpu_filters.vendor
        else None
    )
    chosen = getattr(disaggregation, "vendor", None)
    from_group = {chosen.lower()} if chosen else None

    if from_recipe is None:
        return from_group
    if from_group is None:
        return from_recipe
    # Both present: the group's choice narrows the recipe, never widens it.
    # An empty intersection is a contradiction the request-time check already
    # refuses; keeping it empty here makes the scheduler agree rather than
    # quietly fall back to the recipe's wider set.
    return from_recipe & from_group


def worker_vendors(worker: Worker) -> Set[str]:
    """Accelerator manufacturer slugs this worker reports.

    Empty when the worker has not reported devices yet -- absence of evidence,
    not evidence of a mismatch. Callers decide what to do with that.
    """
    devices = (worker.status and worker.status.gpu_devices) or []
    return {(device.vendor or "").lower() for device in devices if device.vendor}


class PDModeRuntimeFilter:
    """Keep only workers whose accelerator matches the pd mode's declaration.

    A recipe like `vllm-ascend-mooncake` injects an Ascend-only connector plus
    HCCL variables, and the three NVIDIA recipes inject connectors no AMD
    runtime can read. Nothing else enforces this -- the catalog loader only
    asserts `backends` -- and a mismatch surfaces as an engine that starts and
    then fails inside the connector, which is far harder to read than an empty
    candidate list.

    `custom` is never narrowed. It declares no `gpu_filters`, so
    `required_vendors` returns None and this filter is a no-op: an
    unsupported engine × accelerator pair means "no built-in recipe", never
    "no PD".
    """

    def __init__(self, model: Model):
        self._model = model
        self._wanted = required_vendors(model)

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        if self._wanted is None:
            return workers, []

        wanted = self._wanted
        candidates = [worker for worker in workers if worker_vendors(worker) & wanted]

        messages = []
        if len(candidates) != len(workers):
            messages = [
                f"Matched {len(candidates)}/{len(workers)} workers by "
                f"{'/'.join(sorted(wanted)) or 'no'} accelerator, required by "
                f"pd mode {self._model.disaggregation.mode}."
            ]
        return candidates, messages
