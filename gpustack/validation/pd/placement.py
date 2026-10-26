"""Validate topology constraints and resolve per-role placement widths."""

import logging
from collections import Counter
from typing import List, Optional, Set, Tuple, Union

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.clusters import GatherStrategyEnum
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelCreate,
    ModelSpecBase,
    ModelUpdate,
    role_effective_model,
)
from gpustack.schemas.workers import Worker
from gpustack.validation.pd.common import effective_model_spec

logger = logging.getLogger(__name__)


async def validate_gather_layer(
    session: AsyncSession,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
    stored: Optional[Model] = None,
) -> None:
    """Refuse a `gather.layer` this cluster has no rung for.

    The failure this prevents is silence, not a crash. The solver's
    `_enforced_gather` stands an unknown layer down and places the group as if
    nothing had been asked for — correct behaviour there (a layer renamed under
    a *running* deployment must not take it down) and exactly the wrong
    behaviour at submit time, where it would accept a `MustGather` under a
    promise nothing enforces.

    It is checked here rather than on `GatherSpec` because the answer depends
    on the cluster: the layer names are the cluster's own declaration, which a
    field validator on the model has no access to. `accelerator_domain` is
    subject to the same rule as every other name — valid only for a cluster
    that actually declared a layer called that.

    A cluster that cannot be read is left alone rather than guessed at:
    scheduling still stands the requirement down, so the cost is a missed
    refusal, not a wrong one.
    """
    model_in = effective_model_spec(model_in, stored)
    gather = getattr(model_in, "gather", None)
    layer = getattr(gather, "layer", None)
    if not layer:
        return

    from gpustack.schemas.clusters import Cluster
    from gpustack.topology.tree import NODE_LAYER, TopologyError
    from gpustack.topology.vocabulary import (
        gather_layer_names,
        validate_declaration,
    )

    if layer == NODE_LAYER:
        await _refuse_a_floor_no_member_can_meet(session, model_in, cluster_id)
        return

    effective_cluster_id = cluster_id or getattr(model_in, "cluster_id", None)
    if effective_cluster_id is None:
        return
    cluster = await Cluster.one_by_id(session, effective_cluster_id)
    if cluster is None:
        return

    try:
        names = gather_layer_names(validate_declaration(cluster.topology))
    except TopologyError:
        # The cluster's own declaration is broken; refusing the *model* for it
        # would send the operator to the wrong page.
        return
    if layer not in names:
        raise BadRequestException(
            message=(
                f"gather layer {layer!r} is not a layer of this cluster. "
                f"Available: {', '.join(names)}."
            )
        )


async def _refuse_a_floor_no_member_can_meet(
    session: AsyncSession,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    cluster_id: Optional[int],
) -> None:
    """Refuse «all members on one machine» when a member cannot fit on one.

    **The one contradiction cross-machine members introduce.** A floor at the
    host rung says every member of the group sits on a single machine; a role
    whose tensor-parallel width exceeds any machine in the cluster needs two.
    Both are things the operator asked for, and no placement satisfies them
    together.

    Caught here rather than left to the solver because the two answers read
    completely differently. At placement time it surfaces as a group that never
    leaves PENDING with a capacity sentence beside it -- and the cluster is not
    short of capacity, so the reader goes looking for cards that are already
    there. At submit time it is one sentence naming the two settings that
    disagree, while both are still on screen.

    Silent about everything it cannot be sure of: a parallel width the engine
    was never told, a cluster whose workers cannot be read, a backend whose
    width this does not know how to ask for. A refusal is only worth issuing
    when the contradiction is certain.
    """
    gather = getattr(model_in, "gather", None)
    if getattr(gather, "strategy", None) != GatherStrategyEnum.MUST_GATHER:
        # `PreferGather` at the host rung is a target, not a promise. A member
        # that has to span simply misses it and says so afterwards.
        return

    spanning, widest = await roles_that_must_span(session, model_in, cluster_id)
    if spanning:
        requirements = ", ".join(
            f"{name!r} needs {width} GPUs" for name, width in spanning
        )
        raise BadRequestException(
            message=(
                f"Roles that must span machines: {requirements}. The widest worker "
                f"in this cluster has at most {widest} usable GPUs, "
                f"but this deployment also asks for every member to be on one "
                f"machine, and to be refused rather than placed outside it. "
                f"Lower the parallel size, or choose a looser topology floor."
            )
        )


async def roles_that_must_span(
    session: AsyncSession,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    cluster_id: Optional[int] = None,
) -> Tuple[List[Tuple[str, int]], int]:
    """Which roles cannot fit one machine, and how wide the widest machine is.

    Arithmetic, not a placement solve — which is the whole reason it is safe to
    ask while a form is still being typed. It compares a width the operator has
    already stated against a fact about the cluster; it does not select
    candidates, does not consult free capacity, and its answer does not move
    while the rest of the form is filled in. It must not become the other
    thing: a verdict about a finished configuration, asked of a half finished
    one, which conflates «this tier does not fit» with «I cannot tell yet».

    Empty is the answer for everything it cannot be sure of — a width the
    engine was never told, a cluster whose workers cannot be read, a backend
    whose width this does not know how to ask for. Both callers treat «not
    sure» as «not spanning».
    """
    effective_cluster_id = cluster_id or getattr(model_in, "cluster_id", None)
    if effective_cluster_id is None:
        return [], 0

    # Widths first: a deployment that never stated one cannot span anything,
    # and asking the cluster about it would be a query for nothing.
    widths = [
        (spec.name, _role_gpu_width(model_in, spec.name))
        for spec in getattr(model_in, "roles", None) or []
    ]
    widths = [(name, width) for name, width in widths if width]
    if not widths:
        return [], 0

    workers = await Worker.all_by_field(session, "cluster_id", effective_cluster_id)
    from gpustack.policies.worker_filters.pd_mode_filter import required_vendors

    wanted = required_vendors(model_in)
    widest = max(
        (_worker_gpu_capacity(worker, wanted) for worker in workers),
        default=0,
    )
    if widest == 0:
        return [], 0

    return [(name, width) for name, width in widths if width > widest], widest


def _worker_gpu_capacity(worker: Worker, wanted: Optional[Set[str]]) -> int:
    """Count one eligible vendor partition, leaving unknown vendors possible."""
    if wanted == set():
        return 0
    devices = (worker.status and worker.status.gpu_devices) or []
    counts = Counter(
        (getattr(device, "vendor", None) or "").lower() for device in devices
    )
    unknown = counts.pop("", 0)
    # One member cannot combine vendors. Unreported vendors could still belong
    # to the eligible partition, so include them in this capacity upper bound.
    return unknown + max(
        (
            count
            for vendor, count in counts.items()
            if wanted is None or vendor in wanted
        ),
        default=0,
    )


def _role_gpu_width(model_in, role_name: str) -> Optional[int]:
    """How many GPUs one member of this role wants, or None if not stated.

    Asked of the selectors rather than re-derived: they already turn a mixed
    bag of `--tensor-parallel-size` / `--tp-size` / `--pipeline-parallel-size`
    / data parallelism into a world size, per engine, and a second reading of
    those flags here is a second answer to one question.
    """
    from gpustack.policies.candidate_selectors import (
        SGLangResourceFitSelector,
        VLLMResourceFitSelector,
    )
    from gpustack.schemas.models import get_backend

    try:
        projected = role_effective_model(model_in, role_name)
        backend = get_backend(projected)
    except Exception:
        logger.debug(
            "Could not project model configuration for role '%s'",
            role_name,
            exc_info=True,
        )
        return None
    selectors = {
        BackendEnum.VLLM: VLLMResourceFitSelector,
        BackendEnum.SGLANG: SGLangResourceFitSelector,
    }
    selector = selectors.get(backend)
    if selector is None:
        return None
    try:
        world_size, _strategies = selector.get_world_size_from_backend_parameters(
            projected
        )
    except Exception:
        logger.debug(
            "Could not resolve GPU width for role '%s'", role_name, exc_info=True
        )
        return None
    return world_size
