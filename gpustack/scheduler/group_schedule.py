"""Placing a whole group at once, and the gate that keeps it away from
everything else.

This is the scheduler's entry point into the group solver, the topology tree
and the capacity bridge, plus the half the solver does not answer — which
worker is not enough, a member also needs its cards.

**The safety property is the gate, not the algorithm.** Group scheduling runs
only for a model that has `roles`, and only the first time that generation
forms. Everything else — every single-role deployment, and every later
per-role scale-out — falls through to `_schedule_one` on a path this module
does not touch. That is what makes "the group is never spread" a decision
confined to groups: it needs no compatibility argument for existing models,
because existing models never reach here.

**Why one arriving instance places all of them.** The schedule queue delivers
instances one at a time, so a 2P2D group arrives as four separate items. Left
to itself each would be placed independently, which is exactly the greedy trap
the solver exists to remove ("place two, and the other two no longer fit").
So the first member of an unplaced group solves for the whole group and writes
every member's row; the siblings that arrive afterwards find themselves already
scheduled and are skipped.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.config.config import Config
from gpustack.policies.scorers.model_file_locality_scorer import ready_worker_ids
from gpustack.schemas.clusters import Cluster, GatherStrategyEnum
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    RoleNameEnum,
)
from gpustack.schemas.workers import Worker
from gpustack.scheduler.group_capacity import (
    GroupCapacity,
    attendant_demands,
    role_demands,
)
from gpustack.policies.scorers.group_placement_scorer import group_scorer
from gpustack.scheduler.group_solver import (
    GatherRequest,
    GroupPlacement,
    RoleDemand,
    solve_group_placement,
)
from gpustack.topology.tree import TopologyError
from gpustack.topology.view import build_view

logger = logging.getLogger(__name__)


def is_group_forming(model: Model, instances: Sequence[ModelInstance]) -> bool:
    """Whether this generation of `model` is a group that has not been placed.

    Three conditions, and each excludes a case that must keep its current
    behaviour:

    - **`roles` is non-empty.** A role-less model is not a group; it has
      replicas, and replicas are interchangeable. This is the condition that
      keeps every existing deployment out of here.
    - **At least one GPU-bearing member is unplaced.** Nothing to solve
      otherwise.
    - **No GPU-bearing member is already placed.** A group with some members
      on workers is a scale-out, not a forming: solving the whole group again
      would either move running members (it cannot) or place the new one
      against a stale picture. Scale-out stays on the per-instance path.
    """
    if not model.roles:
        return False

    gpu_members = [i for i in instances if i.role != RoleNameEnum.ROUTER.value]
    if not gpu_members:
        return False
    if all(i.worker_id is not None for i in gpu_members):
        return False
    return not any(i.worker_id is not None for i in gpu_members)


def gather_request(model: Model) -> GatherRequest:
    """The group's gather requirement — the model's, and only the model's.

    Deliberately no cluster-level default underneath it, because the two
    directions of being wrong are not the same size. Without a default, a group
    that wanted `rack` and said nothing is placed looser than ideal — it runs,
    slower. With one, a group inherits `MustGather` at a layer the deploy form
    never mentioned and the deployment is *refused*, citing a floor the
    deployer did not set and cannot see. A cluster-level failure policy is an
    operator arming a rejection on someone else's behalf.

    So the requirement comes from one place, and an empty `GatherRequest`
    means exactly what it says: no constraint, place it wherever it fits.
    """
    spec = getattr(model, "gather", None)
    if spec and spec.strategy:
        return GatherRequest(
            layer=spec.layer,
            must=spec.strategy == GatherStrategyEnum.MUST_GATHER,
        )
    return GatherRequest()


async def schedule_group(
    session: AsyncSession,
    config: Config,
    model: Model,
    workers: List[Worker],
    model_instances: List[ModelInstance],
    group_instances: List[ModelInstance],
) -> Tuple[Optional[Dict[int, object]], List[str]]:
    """Solve the whole group, and return one candidate per member row.

    Returns `(by_instance_id, messages)`. A `None` mapping means the group
    cannot be placed — all-or-nothing, so the caller must not place a
    subset. `messages` carries the refusal in the solver's own words, which
    name the shortfall and the roomiest domain rather than saying "no room".
    """
    cluster = (
        await Cluster.one_by_id(session, model.cluster_id) if model.cluster_id else None
    )
    # The model's own cluster, and only it. `workers` arrives from
    # `Worker.all(session)` — every worker this server knows, across every
    # cluster — and both sibling paths narrow it before they use it: the
    # per-instance topology read spells the same comprehension, and
    # `evaluate_group` is never handed more than one cluster's workers because
    # its caller groups them first. Left whole, the tree is built over hosts
    # this deployment can never land on: the solver walks a foreign single-host
    # domain per foreign worker and pays a full selector sweep on each, and the
    # refusal then counts them. An e2e run on a 3-worker cluster inside a
    # 17-worker fleet was refused with "capacity could not be measured on 17
    # worker(s)" — fourteen of which were never candidates and nothing an
    # operator did to them could have changed the answer.
    in_cluster = [w for w in workers if w.cluster_id == model.cluster_id]
    try:
        view = build_view(cluster.topology if cluster else None, in_cluster)
    except TopologyError as e:
        # A declaration that cannot become a tree is an operator error, not a
        # capacity one. Refusing the group with the reason beats placing it
        # against a tree built from a guess.
        return None, [f"Cluster topology is invalid: {e}"]

    demands = [RoleDemand(**d) for d in role_demands(model)]
    if not demands:
        return None, ["The group has no member that occupies an accelerator."]

    # Cache servers draw from the same `service_port_range` the members do, so
    # a host running one has fewer ports for the group — and a group is placed
    # onto cache-bearing hosts on purpose, not by accident.
    cache_instances = await cache_instances_in(session, model.cluster_id)
    # The same narrowed list the tree was built from. `ClusterFilter` inside
    # `GroupCapacity` would reach the same verdict, but it would reach it after
    # the foreign workers have already been counted into the "matched N by
    # cluster selector" line the refusal now carries — and a capacity function
    # that measures workers the solver was never offered is one more place for
    # the two to disagree.
    capacity = GroupCapacity(
        config, model, in_cluster, model_instances, cache_instances
    )
    # One chain, walked from the host upward. Picking which chain to walk used
    # to be a step here — the layer name was looked up to decide whether the
    # search followed the network rungs or the accelerator ones — and it is
    # gone with the second chain: there is one tree, so there is one search.
    request = gather_request(model)
    # Where this model's files already sit, read once and spent twice: it
    # orders the workers *inside* a domain and it ranks whole placements
    # *between* domains. The two pull the same way, so a placement assembled
    # under the first scores better under the second rather than fighting it.
    #
    # Best-effort, like the scorer that shares the query: a fleet whose file
    # table cannot be read is placed on capacity alone, which is the answer
    # this path gave before locality was consulted at all.
    try:
        warm = await ready_worker_ids(session, model)
    except Exception as e:
        logger.warning(
            "Could not read model file locality for group %s; placing on "
            "capacity alone: %s",
            model.name,
            e,
        )
        warm = set()
    placement = await solve_group_placement(
        view.root,
        demands,
        capacity,
        view.scopes(),
        request,
        # The router, checked but never assigned here: it is created a pass
        # later by the dependency gate, from peer addresses that do not exist
        # yet. Passing it in is what stops a group being admitted onto hardware
        # with no room for the one member that answers requests.
        attendants=[RoleDemand(**d) for d in attendant_demands(model)],
        # Rank, not decide: a worker the solver has already found roomier or
        # less loaded wins whatever this says.
        preference={worker_id: 0 for worker_id in warm},
        score=group_scorer(
            warm,
            pair_weight=envs.SCHEDULER_GROUP_PAIR_LOCALITY_WEIGHT,
            file_weight=envs.SCHEDULER_GROUP_FILE_LOCALITY_WEIGHT,
        ),
        limit=envs.SCHEDULER_GROUP_CANDIDATE_LIMIT,
    )
    if not isinstance(placement, GroupPlacement):
        # The count and the size, together. The solver speaks in placements
        # ("needs 4, the cluster has room for 2") because that is the unit it
        # reasons in; the selectors speak in GiB. A refusal with only the first
        # leaves the reader unable to tell a group that is slightly too big
        # from one that was never going to fit, which is the difference between
        # freeing a card and choosing another model.
        reason = getattr(placement, "reason", "The group does not fit.")
        notes = capacity.notes_for(getattr(placement, "role", None))
        return None, [reason] + [f"\n{note}" for note in notes]

    logger.info(
        "Group %s placed in %s %r (%s)",
        model.name,
        placement.layer,
        placement.domain,
        # Read off the placement, so the line records what the ranking used
        # rather than a second evaluation of the same terms. "not ranked" is
        # itself worth logging: it says one domain fitted and there was
        # nothing to choose between.
        placement.score.describe() if placement.score else "not ranked",
    )

    # Step 6's second half. `already` accumulates across roles for the same
    # reason the count does: the second role has to see what the first took, or
    # both are offered the same cards.
    already: List[object] = []
    by_instance: Dict[int, object] = {}
    for role, worker_ids in placement.assignments.items():
        candidates = await capacity.commit(role, worker_ids, already)
        if len(candidates) != len(worker_ids):
            # Deliberately not blamed on the cluster. `GroupCapacity` snapshots
            # the instance list when it is built, so both passes read the same
            # picture and a concurrent change is invisible to either -- which
            # means it cannot be what made them disagree. Reaching here is a
            # defect in this module, and saying so is what sends the reader to
            # the warning `commit` logs with the role, the worker and the two
            # counts rather than to `kubectl get` on a healthy fleet.
            return None, [
                "The group's placement could not be turned into GPU "
                f"assignments for role '{role}': the search and the commit "
                "disagreed about what fits. This is a bug in GPUStack, not a "
                "change in the cluster; the server log names the member."
            ]
        rows = [i for i in group_instances if i.role == role and i.worker_id is None]
        if len(rows) < len(candidates):
            # Fewer rows than the solve placed: the convergence loop has not
            # created them all yet. Refuse rather than place part of a role —
            # the next cycle sees a complete picture.
            return None, [
                f"Role '{role}' has {len(rows)} unplaced members but the "
                f"group was solved for {len(candidates)}; waiting for the "
                "rest to be created."
            ]
        for row, candidate in zip(rows, candidates):
            by_instance[row.id] = candidate
            already.append(stand_in(candidate))

    return by_instance, []


async def cache_instances_in(session: AsyncSession, cluster_id) -> List[object]:
    """Cache server instances of this cluster, for the port budget.

    Failure is not fatal here and deliberately so: the budget is a refinement
    of a capacity number that was already correct about cards. Refusing to
    schedule a group because the cache table could not be read would trade a
    slightly optimistic port count for an outage.
    """
    if not cluster_id:
        return []
    try:
        from gpustack.schemas.cache_services import CacheServiceInstance

        return list(
            await CacheServiceInstance.all_by_field(session, "cluster_id", cluster_id)
        )
    except Exception as e:
        logger.warning(
            "Could not read cache service instances for the port budget: %s", e
        )
        return []


def stand_in(candidate) -> object:
    """What the allocation accounting reads off a placed instance.

    The same four fields `offer_slot._PlacedStandIn` carries, and for the same
    reason: a real `ModelInstance` would mean either touching the session or
    filling in values that are lies.
    """
    from gpustack.scheduler.offer_slot import _PlacedStandIn

    subordinates = getattr(candidate, "subordinate_workers", None)
    return _PlacedStandIn(
        worker_id=candidate.worker.id,
        gpu_indexes=candidate.gpu_indexes,
        gpu_type=candidate.gpu_type,
        computed_resource_claim=candidate.computed_resource_claim,
        distributed_servers=(
            _Subordinates(subordinate_workers=list(subordinates))
            if subordinates
            else None
        ),
    )


class _Subordinates:
    def __init__(self, subordinate_workers):
        self.subordinate_workers = subordinate_workers
