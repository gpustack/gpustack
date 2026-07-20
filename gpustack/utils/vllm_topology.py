"""Pure topology validation for vLLM MP multi-node deployments.

Lives in ``utils`` (not in ``worker.backends``) so scheduler / policies code
can import these helpers without dragging in worker-only dependencies such as
``gpustack_runtime.deployer``. Anything that needs a specific node's identity
(``ModelInstance`` / ``ModelInstanceDeploymentMetadata``) stays on the worker
side — see ``gpustack.worker.backends.vllm.cal_multinode_topology``.
"""

from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import List, Literal, Optional

from gpustack.utils.command import (
    find_int_parameter,
    find_bool_parameter,
    find_parameter,
)


MultinodeShape = Literal["dp_only", "mp_only", "nested"]


@dataclass
class MultinodeUserParallelism:
    """User-supplied parallelism hints from backend_parameters.

    Any field left as ``None`` means GPUStack should derive a value from the
    physical topology. Non-None values are treated as hard constraints and
    rejected if they contradict the topology.
    """

    tp: Optional[int] = None
    pp: Optional[int] = None
    dp: Optional[int] = None
    dpl: Optional[int] = None
    # Prefill context parallel: multiplies the per-DP-group world like tp/pp
    # (vLLM world_size = pp * tp * pcp). Decode CP is deliberately absent — it
    # reuses the TP group's GPUs and never widens the topology.
    pcp: Optional[int] = None


@dataclass
class ValidatedTopology:
    """Cluster-level layout that has passed every topology-only validation.

    Captures the shape and per-node DP fragmentation so node-perspective
    callers can compute their own ``start_rank`` / ``dpl`` without redoing
    the inference work. Does **not** include node-perspective fields like
    ``my_idx`` or ``is_follower`` — those are filled in by
    ``cal_multinode_topology`` for a specific node.
    """

    shape: MultinodeShape
    tp: int
    pp: int
    dp: int  # cluster total DP ranks
    dpl_per_node: List[int]  # per-node --data-parallel-size-local values
    nnodes_within_dp: int  # how many physical nodes share one DP rank


def validate_multinode_topology(  # noqa: C901
    gpu_per_node: List[int],
    user: Optional[MultinodeUserParallelism] = None,
) -> ValidatedTopology:
    """Pure topology validator.

    Runs every check that does not depend on a specific node's identity.
    Callers (selector at schedule time, ``cal_multinode_topology`` at worker
    start time) get the same diagnostic ``ValueError`` so error messages are
    maintained in one place.

    Pivot: ``workers_per_dp = tp * pp * pcp`` versus per-node GPU count decides
    whether a single DP rank fits in one node (``dp_only``) or must span
    multiple nodes (``mp_only`` / ``nested``).
    """
    user = user or MultinodeUserParallelism()
    nnodes = len(gpu_per_node)

    # Reject non-positive parallelism before ``... or 1`` silently swaps a 0 for
    # 1 or a negative value corrupts workers_per_dp. Shared choke point for both
    # the scheduler selector and the worker's cal_multinode_topology; tp is
    # validated after its derivation below.
    for name, value in (
        ("pipeline-parallel-size", user.pp),
        ("prefill-context-parallel-size", user.pcp),
        ("data-parallel-size", user.dp),
        ("data-parallel-size-local", user.dpl),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"vLLM multi-node: --{name} {value} must be positive.")

    pp = user.pp or 1
    pcp = user.pcp or 1
    if user.tp is not None:
        tp = user.tp
    elif user.dp is not None:
        # vLLM constraint: tp * pp * pcp * dp == sum(gpu_per_node). When the user
        # pins dp (and optionally pp / prefill-CP), we can solve for tp instead of
        # falling back to gcd(gpu_per_node) — which would ignore the user's intent
        # and almost always conflict with their dp later.
        total_gpus = sum(gpu_per_node)
        denom = user.dp * pp * pcp
        if denom == 0 or total_gpus % denom != 0:
            raise ValueError(
                f"vLLM multi-node: cannot infer --tensor-parallel-size from "
                f"--data-parallel-size {user.dp}, --pipeline-parallel-size {pp} "
                f"and --prefill-context-parallel-size {pcp}: total GPUs "
                f"{total_gpus} is not a multiple of {user.dp} * {pp} * {pcp} "
                f"= {denom}."
            )
        tp = total_gpus // denom
    else:
        tp = reduce(gcd, gpu_per_node)

    if tp <= 0:
        raise ValueError(
            f"vLLM multi-node: --tensor-parallel-size {tp} must be positive."
        )

    workers_per_dp = tp * pp * pcp

    if workers_per_dp <= min(gpu_per_node):
        # dp_only: every DP rank fits in one node.
        dpl_per_node = [g // workers_per_dp for g in gpu_per_node]
        dp_total = sum(dpl_per_node)

        if user.dp is not None and user.dp != dp_total:
            raise ValueError(
                f"vLLM multi-node: --data-parallel-size {user.dp} does not match "
                f"derived total {dp_total} (per-node DP-Local = {dpl_per_node})."
            )
        if user.dpl is not None and any(d != user.dpl for d in dpl_per_node):
            raise ValueError(
                f"vLLM multi-node: --data-parallel-size-local {user.dpl} cannot "
                f"be applied uniformly to a heterogeneous worker group "
                f"(per-node GPUs = {gpu_per_node}, derived per-node DP-Local = "
                f"{dpl_per_node}). Remove --data-parallel-size-local from "
                "backend_parameters and let GPUStack derive it per node."
            )

        return ValidatedTopology(
            shape="dp_only",
            tp=tp,
            pp=pp,
            dp=dp_total,
            dpl_per_node=dpl_per_node,
            nnodes_within_dp=1,
        )

    # workers_per_dp > min(gpu_per_node): a DP rank must span multiple nodes.
    # vLLM requires this layout to be homogeneous because every node receives
    # the same --data-parallel-size-local value.
    if len(set(gpu_per_node)) != 1:
        raise ValueError(
            f"vLLM multi-node: cross-node TP/PP (workers_per_dp={workers_per_dp} "
            f"> min node capacity {min(gpu_per_node)}) requires a homogeneous "
            f"cluster, but workers have differing GPU counts {gpu_per_node}."
        )
    # Homogeneous guaranteed by the check above, so any index is equivalent.
    g_node = gpu_per_node[0]
    if workers_per_dp % g_node != 0:
        raise ValueError(
            f"vLLM multi-node: workers_per_dp={workers_per_dp} cannot evenly "
            f"distribute across nodes with {g_node} GPUs each."
        )
    nnodes_within_dp = workers_per_dp // g_node
    if nnodes % nnodes_within_dp != 0:
        raise ValueError(
            f"vLLM multi-node: cluster size {nnodes} cannot evenly distribute "
            f"DP ranks that each span {nnodes_within_dp} nodes."
        )
    inferred_dp = nnodes // nnodes_within_dp

    if user.dp is not None and user.dp != inferred_dp:
        raise ValueError(
            f"vLLM multi-node: --data-parallel-size {user.dp} does not match "
            f"derived total {inferred_dp} (cluster fits {inferred_dp} DP ranks "
            f"of {nnodes_within_dp} nodes each)."
        )
    if user.dpl is not None and user.dpl != 1:
        raise ValueError(
            f"vLLM multi-node: --data-parallel-size-local {user.dpl} is "
            "invalid in cross-node TP/PP layouts (must be 1 — each node hosts "
            "at most one DP rank fragment)."
        )

    shape: MultinodeShape = "mp_only" if inferred_dp == 1 else "nested"
    return ValidatedTopology(
        shape=shape,
        tp=tp,
        pp=pp,
        dp=inferred_dp,
        dpl_per_node=[1] * nnodes,
        nnodes_within_dp=nnodes_within_dp,
    )


def parse_user_parallelism(
    backend_parameters: Optional[List[str]],
) -> MultinodeUserParallelism:
    """Extract MultinodeUserParallelism from user-provided backend_parameters."""
    return MultinodeUserParallelism(
        tp=find_int_parameter(backend_parameters, ["tensor-parallel-size", "tp"]),
        pp=find_int_parameter(backend_parameters, ["pipeline-parallel-size", "pp"]),
        dp=find_int_parameter(backend_parameters, ["data-parallel-size", "dp"]),
        dpl=find_int_parameter(backend_parameters, ["data-parallel-size-local", "dpl"]),
        pcp=find_int_parameter(
            backend_parameters, ["prefill-context-parallel-size", "pcp"]
        ),
    )


def resolve_data_parallel_load_balance_mode(
    backend_parameters: Optional[List[str]],
) -> str:
    """Resolve the effective vLLM data-parallel load-balance mode from the
    user-supplied backend parameters.

    ``--data-parallel-hybrid-lb`` -> ``"hybrid"``; ``--data-parallel-external-lb``
    or a pinned ``--data-parallel-rank`` -> ``"external"`` (vLLM implies
    external-LB then). Defaults to ``"internal"``.
    """
    if find_bool_parameter(backend_parameters, ["data-parallel-hybrid-lb"]):
        return "hybrid"
    if (
        find_bool_parameter(backend_parameters, ["data-parallel-external-lb"])
        or find_parameter(backend_parameters, ["data-parallel-rank"]) is not None
    ):
        return "external"
    return "internal"


def subordinates_serve_api(backend_parameters: Optional[List[str]]) -> bool:
    """Whether each subordinate worker runs its own API server (non-headless) and
    must be registered as an independent gateway backend. True for any non-internal
    load-balance mode (hybrid-LB / external-LB; see
    ``resolve_data_parallel_load_balance_mode``); internal-LB followers stay
    ``--headless`` so only the leader serves.
    """
    return resolve_data_parallel_load_balance_mode(backend_parameters) != "internal"


def derive_dp_node_count(
    backend_parameters: Optional[List[str]],
) -> Optional[int]:
    """Number of DP nodes derivable from the data-parallel flags alone.

    external-LB serves one rank per node -> nodes == --data-parallel-size;
    hybrid-LB packs ``dpl`` ranks per node -> nodes == dp // dpl. Returns
    ``None`` for internal-LB / no LB flag. Raises ``ValueError`` when the flags
    are missing or inconsistent to derive a count.
    """
    load_balance_mode = resolve_data_parallel_load_balance_mode(backend_parameters)
    if load_balance_mode not in ("external", "hybrid"):
        return None

    parallelism = parse_user_parallelism(backend_parameters)
    dp = parallelism.dp
    dpl = parallelism.dpl

    if dp is None or dp <= 1:
        got = "omitted, vLLM defaults to 1" if dp is None else dp
        raise ValueError(
            f"vLLM {load_balance_mode} load-balance requires --data-parallel-size "
            f"> 1 (got {got}); each node serves its own DP rank(s)."
        )

    if load_balance_mode == "external":
        # external-LB pins exactly one DP rank per node (dpl == 1).
        return dp

    # hybrid-LB: dpl ranks per node, so nodes == dp // dpl. Without dpl the
    # per-node fan-out is topology-derived and can't be known from params alone.
    if dpl is None:
        raise ValueError(
            "vLLM hybrid load-balance requires --data-parallel-size-local so "
            "GPUStack can derive replicas (nodes = --data-parallel-size // "
            "--data-parallel-size-local)."
        )
    if dp % dpl != 0:
        raise ValueError(
            f"vLLM multi-node: --data-parallel-size {dp} must be a multiple of "
            f"--data-parallel-size-local {dpl}."
        )
    return dp // dpl


_MANUAL_DISTRIBUTED_PARAMS = [
    "data-parallel-address",
    "data-parallel-start-rank",
    "node-rank",
    "data-parallel-rank",
]


def matched_manual_distributed_params(
    backend_parameters: Optional[List[str]],
) -> List[str]:
    """The manual DP-wiring flags (``--data-parallel-address`` / ``--node-rank`` /
    rank) actually present in backend_parameters, as ``--flag`` strings. A non-empty
    result signals the user manages distribution themselves rather than letting
    GPUStack orchestrate it; empty means none are set.
    """
    return [
        f"--{param}"
        for param in _MANUAL_DISTRIBUTED_PARAMS
        if find_parameter(backend_parameters, [param]) is not None
    ]
