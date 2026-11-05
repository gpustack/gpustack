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
from typing import Dict, List, Literal, Optional, Tuple, Union

from gpustack.utils.command import (
    find_int_parameter,
    find_last_int_parameter,
    find_last_parameter,
    find_bool_parameter,
    flatten_to_argv,
)

# vLLM excludes this window from automatic ports when VLLM_DP_MASTER_PORT
# is set. Both placement budgeting and Worker reservations protect it.
VLLM_MP_CONNECTING_PORT_COUNT = 10


def normalize_vllm_parameters(
    parameters: Optional[List[str]], *, tokenized: bool = False
) -> List[str]:
    """Canonicalize long option names without changing values or dotted keys.

    Set tokenized for built argv so equal-form values are not split again.
    """
    result = []
    argv = list(parameters or []) if tokenized else flatten_to_argv(parameters)
    for token in argv:
        if token.startswith('--'):
            key, separator, value = token.partition('=')
            name, dot, field = key.partition('.')
            token = f'{name.replace("_", "-")}{dot}{field}{separator}{value}'
        result.append(token)
    return result


def manual_dp_enabled(env: Optional[Dict[str, str]]) -> bool:
    """Whether the operator owns the ordinary vLLM deployment's DP group."""
    return str((env or {}).get('GPUSTACK_MANUAL_DP', '')).strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def managed_lb_port_requests(
    parameters: Optional[List[str]], gpu_per_node: List[int], *, tokenized: bool = False
) -> List[Tuple[str, int]]:
    """Ordered platform port bands shared by scheduling and Worker allocation."""
    parameters = normalize_vllm_parameters(parameters, tokenized=tokenized)
    topology = managed_lb_parallelism(parameters, gpu_per_node, tokenized=True)
    mode = vllm_lb_mode(parameters, tokenized=True)
    hybrid = mode == "hybrid"
    requests = [
        ("_vllm_external_api", 1 if hybrid else topology.dp),
        ("_vllm_external_rpc", 1),
    ]
    if not hybrid or topology.nnodes_within_dp > 1:
        requests.insert(0, ("_vllm_mp_connecting", VLLM_MP_CONNECTING_PORT_COUNT))
    if topology.nnodes_within_dp > 1:
        requests.append(("_vllm_external_store", topology.dp))
    if (
        mode == "multi_port_external"
        and find_last_parameter(
            parameters, ["data-parallel-supervisor-port"], tokenized=True
        )
        is None
    ):
        requests.append(("_vllm_multi_port_external_supervisor", 1))
    return requests


def vllm_lb_mode(parameters: Optional[List[str]], *, tokenized: bool = False) -> str:
    """Identify explicitly selected load balancing and reject conflicting modes."""
    parameters = normalize_vllm_parameters(parameters, tokenized=tokenized)
    external = find_bool_parameter(
        parameters,
        ["data-parallel-external-lb", "dpe"],
        negative_param_names=["no-data-parallel-external-lb"],
        last_wins=True,
        tokenized=True,
    )
    multi_port_external = find_bool_parameter(
        parameters,
        ["data-parallel-multi-port-external-lb", "dpm"],
        last_wins=True,
        tokenized=True,
    )
    hybrid = find_bool_parameter(
        parameters,
        ["data-parallel-hybrid-lb", "dph"],
        negative_param_names=["no-data-parallel-hybrid-lb"],
        last_wins=True,
        tokenized=True,
    ) or (
        not multi_port_external
        and find_last_parameter(
            parameters, ["data-parallel-start-rank", "dpr", "dps"], tokenized=True
        )
        is not None
    )
    if sum((external, multi_port_external, hybrid)) > 1 or (
        multi_port_external
        and find_last_parameter(
            parameters, ["data-parallel-rank", "dpn"], tokenized=True
        )
        is not None
    ):
        raise ValueError("Choose only one vLLM data parallel load balancing mode.")
    for mode, enabled in (
        ("external", external),
        ("multi_port_external", multi_port_external),
        ("hybrid", hybrid),
    ):
        if enabled:
            return mode
    return "internal"


def external_lb_parallelism(
    parameters: List[str], gpu_count: Union[int, List[int]], *, tokenized: bool = False
) -> "ValidatedTopology":
    """Validate the complete allocation and split uniform rank fragments."""
    parameters = normalize_vllm_parameters(parameters, tokenized=tokenized)
    values = {}
    for field, names in (
        ("dp", ["data-parallel-size", "dp"]),
        ("tp", ["tensor-parallel-size", "tp"]),
        ("pp", ["pipeline-parallel-size", "pp"]),
        ("pcp", ["prefill-context-parallel-size", "pcp"]),
        ("dpl", ["data-parallel-size-local", "dpl"]),
    ):
        raw = find_last_parameter(parameters, names, tokenized=True)
        value = find_last_int_parameter(parameters, names, tokenized=True)
        if raw is not None and (value is None or value <= 0):
            raise ValueError(f"--{names[0]} must be a positive integer.")
        values[field] = value
    counts = [gpu_count] if isinstance(gpu_count, int) else gpu_count
    if not counts or min(counts) <= 0:
        raise ValueError("Managed vLLM External-LB requires allocated devices.")
    uniform = len(set(counts)) == 1
    local_ranks = values.pop('dpl')
    # Uniform External-LB slices can span Workers; heterogeneous allocations
    # must satisfy the generic per-node consistency check for an explicit dpl.
    result = validate_multinode_topology(
        counts,
        MultinodeUserParallelism(**values, dpl=local_ranks if not uniform else None),
    )
    if result.tp * result.pp * (values["pcp"] or 1) * result.dp != sum(counts):
        raise ValueError("Selected devices must exactly match the full DP/TP/PP world.")
    if uniform:
        local_ranks = local_ranks if local_ranks is not None else result.dpl_per_node[0]
        if (
            counts[0] % local_ranks
            or result.dp % local_ranks
            or len(counts) * local_ranks % result.dp
        ):
            raise ValueError(
                "External-LB local rank slices must evenly consume the allocation."
            )
        nodes = len(counts) * local_ranks // result.dp
        result.dpl_per_node = [local_ranks] * len(counts)
        result.nnodes_within_dp = nodes
    return result


def validate_managed_http_parameters(
    parameters: Optional[List[str]], *, tokenized: bool = False
) -> None:
    """Keep managed rank APIs compatible with platform HTTP routes and probes."""
    parameters = normalize_vllm_parameters(parameters, tokenized=tokenized)
    if any(token.partition("=")[0] == "--uds" for token in parameters):
        raise ValueError(
            "Managed rank routes and health checks require TCP HTTP; --uds is unsupported."
        )
    if any(
        find_last_parameter(parameters, [name], tokenized=True) is not None
        for name in ("ssl-keyfile", "ssl-certfile")
    ):
        raise ValueError(
            "Managed rank routes and health checks currently require HTTP; TLS is unsupported."
        )


def validate_multi_port_external_lb_parameters(
    parameters: Optional[List[str]], *, tokenized: bool = False
) -> None:
    """Reject multi-port External-LB options that cannot represent managed rank APIs."""
    parameters = normalize_vllm_parameters(parameters, tokenized=tokenized)
    if vllm_lb_mode(parameters, tokenized=True) != "multi_port_external":
        raise ValueError(
            "Multi-port External-LB requires --data-parallel-multi-port-external-lb."
        )
    for names in (["api-server-count"], ["prefill-context-parallel-size", "pcp"]):
        raw = find_last_parameter(parameters, names, tokenized=True)
        if (
            raw is not None
            and find_last_int_parameter(parameters, names, tokenized=True) != 1
        ):
            raise ValueError(f"Multi-port External-LB requires --{names[0]}=1.")
    for names, minimum in (
        (["data-parallel-size", "dp"], 2),
        (["data-parallel-size-local", "dpl"], 2),
    ):
        if find_last_parameter(parameters, names, tokenized=True) is not None:
            value = find_last_int_parameter(parameters, names, tokenized=True)
            if value is None or value < minimum:
                raise ValueError(
                    f"Multi-port External-LB requires --{names[0]} >= {minimum}."
                )
    nnodes = find_last_parameter(parameters, ["nnodes"], tokenized=True)
    if (
        nnodes is not None
        and find_last_int_parameter(parameters, ["nnodes"], tokenized=True) != 1
    ):
        raise ValueError(
            "Multi-port External-LB requires every rank to fit on one Worker."
        )
    for flag in ("headless", "grpc"):
        if find_bool_parameter(parameters, [flag], last_wins=True, tokenized=True):
            raise ValueError(f"Multi-port External-LB does not support --{flag}.")
    validate_managed_http_parameters(parameters, tokenized=True)
    for name, allowed in (
        ("distributed-executor-backend", ("mp", "uni")),
        ("data-parallel-backend", ("mp",)),
    ):
        if find_last_parameter(parameters, [name], tokenized=True) not in (
            None,
            *allowed,
        ):
            raise ValueError(f"Multi-port External-LB requires a local --{name}.")


def validate_managed_lb_parameters(
    parameters: Optional[List[str]], *, tokenized: bool = False
) -> None:
    """Validate an explicit LB mode before scheduling without device assumptions."""
    parameters = normalize_vllm_parameters(parameters, tokenized=tokenized)
    mode = vllm_lb_mode(parameters, tokenized=True)
    if mode != "internal":
        validate_managed_http_parameters(parameters, tokenized=True)
    if mode == "multi_port_external":
        validate_multi_port_external_lb_parameters(parameters, tokenized=True)


def multi_port_external_lb_parallelism(
    parameters: List[str], gpu_count: Union[int, List[int]], *, tokenized: bool = False
) -> "ValidatedTopology":
    """Resolve multi-port External-LB with complete, evenly allocated local DP replicas."""
    parameters = normalize_vllm_parameters(parameters, tokenized=tokenized)
    validate_multi_port_external_lb_parameters(parameters, tokenized=True)
    counts = [gpu_count] if isinstance(gpu_count, int) else gpu_count
    result = external_lb_parallelism(parameters, counts, tokenized=True)
    if (
        find_last_parameter(
            parameters, ["data-parallel-size-local", "dpl"], tokenized=True
        )
        is None
    ):
        width = result.tp * result.pp
        local = counts[0] // width
        if counts[0] % width or local < 2:
            raise ValueError(
                "Multi-port External-LB requires complete local ranks and local DP >= 2."
            )
        result = external_lb_parallelism(
            [*parameters, "--data-parallel-size-local", str(local)],
            counts,
            tokenized=True,
        )
    if result.dp < 2 or min(result.dpl_per_node) < 2:
        raise ValueError("Multi-port External-LB requires DP > 1 and local DP >= 2.")
    if result.nnodes_within_dp != 1 or len(set(result.dpl_per_node)) != 1:
        raise ValueError(
            "Multi-port External-LB requires every rank to fit on one Worker."
        )
    if (
        find_last_parameter(
            parameters, ["distributed-executor-backend"], tokenized=True
        )
        == "uni"
        and result.tp * result.pp > 1
    ):
        raise ValueError(
            "Multi-port External-LB TP/PP ranks require multiprocessing executors."
        )
    start = find_last_parameter(
        parameters, ["data-parallel-start-rank", "dpr", "dps"], tokenized=True
    )
    if start is not None:
        value = find_last_int_parameter(
            parameters, ["data-parallel-start-rank", "dpr", "dps"], tokenized=True
        )
        if value is None or value < 0 or value + result.dpl_per_node[0] > result.dp:
            raise ValueError(
                "Multi-port External-LB start rank exceeds the allocated DP world."
            )
    return result


def hybrid_lb_parallelism(
    parameters: List[str], gpu_count: Union[int, List[int]], *, tokenized: bool = False
) -> "ValidatedTopology":
    """Resolve a complete Hybrid-LB group with the MP topology constraints.

    Omitted local DP is derived from the allocation. Node roles and rank starts
    use ``external_lb_fragments`` in persisted allocation order.
    """
    parameters = normalize_vllm_parameters(parameters, tokenized=tokenized)
    counts = [gpu_count] if isinstance(gpu_count, int) else gpu_count
    external_lb_parallelism(parameters, counts, tokenized=True)
    return validate_multinode_topology(
        counts, parse_user_parallelism(parameters, last_wins=True, tokenized=True)
    )


def managed_lb_parallelism(
    parameters: List[str], gpu_count: Union[int, List[int]], *, tokenized: bool = False
) -> "ValidatedTopology":
    """Resolve the topology for the explicitly selected managed LB mode."""
    parameters = normalize_vllm_parameters(parameters, tokenized=tokenized)
    validate_managed_lb_parameters(parameters, tokenized=True)
    mode = vllm_lb_mode(parameters, tokenized=True)
    if mode == "multi_port_external":
        return multi_port_external_lb_parallelism(parameters, gpu_count, tokenized=True)
    if mode == "hybrid":
        return hybrid_lb_parallelism(parameters, gpu_count, tokenized=True)
    return external_lb_parallelism(parameters, gpu_count, tokenized=True)


@dataclass(frozen=True)
class ExternalRankFragment:
    dp_rank: int
    node_rank: int
    positions: List[int]
    main_worker_index: int


def external_lb_fragments(
    gpu_per_node: List[int], topology: "ValidatedTopology", worker_index: int
) -> List[ExternalRankFragment]:
    """Map Worker-local device positions to logical rank coordinates."""
    nodes = topology.nnodes_within_dp
    local_ranks = topology.dpl_per_node[worker_index]
    if nodes == 1:
        first_rank = sum(topology.dpl_per_node[:worker_index])
        main = worker_index
    else:
        main = worker_index // nodes * nodes
        first_rank = main // nodes * local_ranks
    width = gpu_per_node[worker_index] // local_ranks
    return [
        ExternalRankFragment(
            dp_rank=first_rank + index,
            node_rank=worker_index - main,
            positions=list(range(index * width, (index + 1) * width)),
            main_worker_index=main,
        )
        for index in range(local_ranks)
    ]


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
    *,
    last_wins: bool = False,
    tokenized: bool = False,
) -> MultinodeUserParallelism:
    """Extract MultinodeUserParallelism from user-provided backend_parameters."""
    backend_parameters = normalize_vllm_parameters(
        backend_parameters, tokenized=tokenized
    )
    read = find_last_int_parameter if last_wins else find_int_parameter
    return MultinodeUserParallelism(
        tp=read(backend_parameters, ["tensor-parallel-size", "tp"], tokenized=True),
        pp=read(backend_parameters, ["pipeline-parallel-size", "pp"], tokenized=True),
        dp=read(backend_parameters, ["data-parallel-size", "dp"], tokenized=True),
        dpl=read(
            backend_parameters, ["data-parallel-size-local", "dpl"], tokenized=True
        ),
        pcp=read(
            backend_parameters, ["prefill-context-parallel-size", "pcp"], tokenized=True
        ),
    )
