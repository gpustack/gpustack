import logging
from typing import List, Optional, Sequence, Tuple

from gpustack.policies.base import WorkerFilter
from gpustack.schemas.models import ModelInstance, member_worker_ids
from gpustack.schemas.workers import Worker
from gpustack.topology.view import TopologyView

logger = logging.getLogger(__name__)


class GatherFloorFilter(WorkerFilter):
    """Keep a group's later members inside the floor its formation honoured.

    **The half of `MustGather` that was never enforced.** The floor is a
    hard refusal at formation -- `solve_group_placement` walks up to the named
    layer and stops -- but only the formation goes through the solver. Every
    member created afterwards (a scaled-out prefill, a scaled-out decode, and
    the router, which is created a pass later by the dependency gate on *every*
    deployment) is placed by the ordinary per-instance path, and that path had
    no idea a floor had been asked for. So a group the operator told the
    scheduler to refuse rather than spread was spread anyway, one member at a
    time, with nothing said: `_gather_unmet` computes for `PreferGather` only,
    on the reasoning that `MustGather` "refused at admission instead".

    The router made it worse rather than better. It is deliberately outside the
    gang, so the solver checks a worker in the domain could host it and then
    the ordinary path puts it wherever it likes -- a promise the placement did
    not keep.

    **A filter, and only here.** Turning locality into a filter for the router
    would let one full worker hold a group hostage, which is why every other
    locality policy here is a scorer, and that is right *by default*. It stops being right when the operator has
    said, in as many words, that they would rather the deployment be refused
    than placed below the floor. That sentence is what `MustGather` is; without
    this, it was decoration on everything except the first four rows.

    Silent when there is nothing to enforce: no floor, no group, or no member
    placed yet leaves every worker exactly where it was.
    """

    def __init__(
        self,
        layer: Optional[str],
        group_id: Optional[str],
        model_instances: Sequence[ModelInstance],
        view: Optional[TopologyView],
        weight_bearing: Sequence[str] = (),
    ):
        self._layer = layer
        self._group_id = group_id
        self._model_instances = list(model_instances)
        self._view = view
        # Roles whose placement defines where the group *is*. The router is
        # excluded from the definition for the same reason `role_demands`
        # excludes it -- it holds no weights, so it cannot be what anchors the
        # group -- while still being subject to the answer.
        self._weight_bearing = set(weight_bearing)

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        if not self._layer or not self._group_id or self._view is None:
            return workers, []

        anchors = {
            worker_id
            for instance in self._model_instances
            if instance.group_id == self._group_id
            and instance.role in self._weight_bearing
            # Every machine the member holds, not only the one its row is
            # filed under: a member that spans machines is *in* the domain on
            # all of them, and anchoring on the primary alone would let the
            # floor be computed from a partial picture of where the group is.
            for worker_id in member_worker_ids(instance)
        }
        if not anchors:
            # Nothing placed yet. Either this is the formation -- which does not
            # come through here at all -- or the group is being rebuilt, and in
            # both cases the solver is the one applying the floor.
            return workers, []

        domains = [
            node
            for node in self._view.nodes(self._layer)
            if not node.is_unclassified
            and anchors.intersection(node.descendant_worker_ids())
        ]
        if len(domains) != 1:
            # Deliberately not a refusal. Zero domains means the members sit
            # in the unclassified bucket at this layer, where "together" is not
            # a fact anyone established; more than one means the floor is
            # already broken, by a rescheduled member or a topology edited
            # underneath a running group. Picking one of several would be
            # arbitrary, and refusing everything would not put back a floor that
            # is already gone -- so the placement proceeds and the breach is
            # reported on the model instead (`_gather_unmet` covers
            # `MustGather` for exactly this case).
            logger.warning(
                "Group %s asked to stay within one %r but its placed members "
                "resolve to %d domains there; not constraining this member.",
                self._group_id,
                self._layer,
                len(domains),
            )
            return workers, []

        domain = domains[0]
        inside = set(domain.descendant_worker_ids())
        kept = [worker for worker in workers if worker.id in inside]
        if len(kept) == len(workers):
            return kept, []

        return kept, [
            f"Kept {len(kept)}/{len(workers)} workers inside {domain.name!r}: "
            f"this deployment asks to stay within one {self._layer!r} and be "
            f"refused rather than placed outside it, and its running members "
            f"are there.\n"
        ]
