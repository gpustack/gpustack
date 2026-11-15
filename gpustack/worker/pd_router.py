"""Rendering a group's router from the pd-mode catalog.

There is no universal router, and that is a finding rather than a limitation:
SGLang ships one, vLLM ships a fork of that one, vllm-ascend ships a Python
proxy example, and TileRT expects you to bring your own. What generalises is
the *catalog format*, not the binary — so this module is one renderer driven by
declarations, not a family of per-engine adapters. Adding an engine whose
router reuses a known peer style is a YAML change; only a genuinely new wire
style needs a branch here, which is why an unknown style fails at catalog load
rather than being rendered wrong.

Deliberately separate from `pd_injection`: prefill and decode are *engines*
that receive injected configuration on top of a command GPUStack already knows
how to build, while a managed router is a command GPUStack builds from nothing.
The two produce different things and the injector says so — it returns None for
the router rather than pretending it has a role entry, since `vllm-nixl`'s
catalog entry keeps the router under `router:` and not under `roles:`.
"""

import logging
import shlex
from typing import TYPE_CHECKING, Dict, List, Mapping, NamedTuple, Optional, Sequence

from pydantic import BaseModel

from gpustack.utils.command import (
    find_last_int_parameter,
    find_last_parameter,
    flatten_to_argv,
    is_parameter_key,
)

from gpustack.schemas.pd_modes import (
    PDMode,
    PDPeerStyleEnum,
    PDRouter,
    PDRouterProtocolEnum,
)
from gpustack.utils.pd_router_launch import resolve_router_launch
from gpustack.utils.template import render, render_values
from gpustack.utils.pd_router_command import (
    render_router_command_template,
    router_environment,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gpustack.schemas.models import ModelBase, ModelInstance


class PeerAddress(NamedTuple):
    """One member of the group, as the router needs to address it.

    Carries the member's named port bands as well as its HTTP port, because
    some routers need a second port per peer and it is a *per-peer* value.
    SGLang's is the case that forced this: `--prefill URL BOOTSTRAP_PORT`
    takes each prefill's own bootstrap band, and the engine's fixed default
    (8998) collides the moment two prefills share a host — so the band is
    allocated per member and cannot come from the deployment scope, which has
    exactly one value for the whole render.
    """

    ip: str
    port: int
    ports: Mapping[str, int] = {}
    """Named port bands of this peer, base only, by declared name."""


PeerMap = Mapping[str, Sequence[PeerAddress]]


class RouterPlan(BaseModel):
    """What a managed router needs in order to start."""

    image: Optional[str] = None
    command: List[str] = []
    env: Dict[str, str] = {}
    """Environment the router binary needs to come up, from the catalog. Not
    KV-transfer configuration — see `PDRouter.env`."""
    health_path: Optional[str] = None
    metrics: bool = False
    """Whether the router serves a Prometheus exposition. False is not a
    default to fall back on — vllm-ascend's proxy example has neither /metrics
    nor /v1/models, and polling them produced a ~1/s 404 storm in its log plus
    a permanent false alarm. So health checks degrade to process liveness and
    the PD-effectiveness ratio loses its denominator, both knowingly."""
    models_endpoint: bool = False


class RouterPeersUnavailable(Exception):
    """Raised when a managed router is asked to render without its peers.

    A hard failure on purpose. Every other way of handling this produces
    a router that starts and answers requests wrongly: an empty `--prefill`
    list makes vllm-router accept traffic it has nowhere to send, and a
    partial one silently halves the group. PD failures are invisible from the
    outside, so degrading here would be lying.
    """


def static_hybrid_router_variables(
    model: "ModelBase",
    instances: Sequence["ModelInstance"],
    group_id: Optional[str],
) -> Dict[str, int]:
    """Derive the common Engine scope required by a static Mooncake router.

    Args:
        model: The original deployment specification, before role projection.
        instances: Current deployment instances, including their allocated tasks.
        group_id: The Router's communication group.

    Returns:
        Transient rendering variables for a managed Hybrid Mooncake Router.
    """
    from gpustack.schemas.models import (
        find_role,
        role_effective_model,
    )
    from gpustack.utils.model_instance_workers import (
        get_model_instance_workloads,
        model_instance_endpoints,
    )

    disaggregation = getattr(model, "disaggregation", None)
    if disaggregation is None or disaggregation.mode.value != "vllm-mooncake":
        return {}
    router = find_role(model, "router")
    if router is not None and router.run_command:
        return {}
    endpoints = [
        (instance, endpoint)
        for instance in instances
        if instance.group_id == group_id
        and instance.role in ("prefill", "decode")
        and getattr(instance, "deleted_at", None) is None
        for endpoint in model_instance_endpoints(instance)
    ]
    if not any(endpoint.component == "vllm_hybrid" for _, endpoint in endpoints):
        return {}
    sizes = set()
    for instance, endpoint in endpoints:
        if endpoint.component == "vllm_hybrid":
            task = next(
                (
                    task
                    for task in get_model_instance_workloads(instance) or []
                    if task.worker_id == endpoint.worker_id
                ),
                None,
            )
            if task is None:
                raise ValueError(
                    f"Hybrid endpoint for instance {instance.id} on "
                    f"Worker {endpoint.worker_id} has no matching workload."
                )
            sizes.add(task.component.dp_size)
        elif endpoint.dp_rank is not None:
            sizes.add(1)
        else:
            effective = role_effective_model(model, instance.role)
            sizes.add(_internal_api_dp_size(effective, instance))
    if len(sizes) != 1:
        raise ValueError(
            "CUDA Mooncake shared APIs must have the same local DP size; "
            "the static router has only one --intra-node-data-parallel-size."
        )
    return {"intra_node_data_parallel_size": sizes.pop()}


def _internal_api_dp_size(model, instance) -> int:
    """An Internal API addresses the full DP group, including remote Engines."""
    parameters = model.backend_parameters
    names = ["data-parallel-size", "dp"]
    raw = find_last_parameter(parameters, names)
    if raw is not None:
        value = find_last_int_parameter(parameters, names)
        if value is not None and value > 0:
            return value
        raise ValueError("Cannot determine the Internal API data parallel size.")
    from gpustack.utils.vllm_topology import (
        parse_user_parallelism,
        validate_multinode_topology,
    )

    servers = instance.distributed_servers
    members = [instance] + list(servers.subordinate_workers or [] if servers else [])
    counts = [len(member.gpu_indexes or []) for member in members]
    return validate_multinode_topology(counts, parse_user_parallelism(parameters)).dp


def _static_hybrid_router_parameters(parameters, expected: int) -> List[str]:
    """Keep one platform-rendered DP flag while retaining other user options."""
    flag = "--intra-node-data-parallel-size"
    tokens = flatten_to_argv(parameters)
    result, overrides = [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name, separator, value = token.partition("=")
        # Both supported wheels use argparse's unambiguous long-option prefixes.
        if len(name) > 2 and name.startswith("--") and flag.startswith(name):
            if not separator:
                index += 1
                value = tokens[index] if index < len(tokens) else ""
            try:
                overrides.append(int(value))
            except ValueError as exc:
                raise ValueError(f"{flag} requires an integer value.") from exc
        else:
            result.append(token)
        index += 1
    if overrides and overrides[-1] != expected:
        raise ValueError(f"{flag} must match the shared API local DP {expected}.")
    if not overrides:
        return list(parameters or [])
    # These fragments are parsed again when the Custom backend builds its argv.
    # Quote values inside their option rather than storing unquoted argv tokens.
    fragments = []
    for token in result:
        key, separator, value = token.partition('=')
        if is_parameter_key(key):
            fragments.append(f'{key}={shlex.quote(value)}' if separator else key)
        elif fragments and is_parameter_key(fragments[-1]):
            fragments[-1] += ' ' + shlex.quote(token)
        else:
            fragments.append(token)
    return fragments


def _peer_scope(peer: PeerAddress) -> Dict[str, object]:
    """The variables visible while rendering ONE peer's address.

    Deliberately not merged with the deployment scope. The catalog first spelled
    these `{{ip}}` / `{{port}}`, and `{{port}}` already means "this member's own
    HTTP port" out there — so a merged scope would resolve a peer's address to
    the *router's* port. That is a wrong address rather than a failure, which is
    the class of bug this whole design keeps having to defend against. The
    `peer.` prefix makes the two scopes unmergeable by construction.

    The same reasoning is why a peer's named ports are `peer.ports.<name>` and
    not `ports.<name>`: the deployment scope already has a `ports.<name>`
    meaning the *router's* band of that name, and for a two-prefill group there
    is no single right answer there at all — each prefill has its own.
    """
    scope: Dict[str, object] = {"peer.ip": peer.ip, "peer.port": peer.port}
    for name, base in (peer.ports or {}).items():
        scope[f"peer.ports.{name}"] = base
    return scope


def _repeated_flag(spec: Mapping[str, str], peers: Sequence[PeerAddress]) -> List[str]:
    """`--prefill http://a:1 --prefill http://b:2` (vLLM router, SGLang router)."""
    flag = spec.get("flag")
    value = spec.get("value")
    if not flag or not value:
        return []
    args: List[str] = []
    for peer in peers:
        args.append(flag)
        rendered = render(value, _peer_scope(peer), context="router peer address")
        # A peer address is a positional pair for some routers
        # (`--prefill URL BOOTSTRAP_PORT`), and argv carries the split, not the
        # string. Splitting here rather than in the catalog keeps the catalog
        # writing one readable value per peer.
        args.extend(rendered.split())
    return args


def _parallel_lists(spec: Mapping[str, str], peers: Sequence[PeerAddress]) -> List[str]:
    """`--prefiller-hosts a b --prefiller-ports 1 2` (vllm-ascend's proxy).

    The two lists are positional against each other, so they are built from one
    iteration rather than two — a filter applied to one and not the other would
    pair each host with the wrong port and still start.
    """
    host_flag = spec.get("host_flag")
    port_flag = spec.get("port_flag")
    if not host_flag or not port_flag:
        return []
    hosts = [peer.ip for peer in peers]
    ports = [str(peer.port) for peer in peers]
    return [host_flag, *hosts, port_flag, *ports]


def render_router(
    mode: PDMode,
    variables: Mapping[str, object],
    peers: PeerMap,
) -> Optional[RouterPlan]:
    """Render the router of `mode` for a group whose peers are `peers`.

    Returns None when the mode declares no router GPUStack manages — either it
    has none at all, or its protocol is `user_provided`, where the image and
    command come from the role spec and rendering one here would override what
    the user typed.

    `variables` is the deployment scope (the router's own `worker_ip`, `port`,
    `ports.<name>`, `runner_image`, `model_name`, `group_id`). Peer addresses
    are NOT in it — see `_peer_scope`.
    """
    router = mode.router
    if router is None:
        return None
    if router.protocol == PDRouterProtocolEnum.USER_PROVIDED:
        return None
    if router.peers.style == PDPeerStyleEnum.USER_PROVIDED:
        return None

    command = [
        render(part, variables, context="router command") for part in router.command
    ]
    if mode.name == "vllm-mooncake" and "intra_node_data_parallel_size" in variables:
        command.extend(
            [
                "--intra-node-data-parallel-size",
                str(variables["intra_node_data_parallel_size"]),
            ]
        )

    # Peer flags go after the declared command rather than interleaved: the
    # catalog's command is the invariant part of the invocation and the peer
    # list is the part that changes on every scale, and keeping them apart is
    # what lets a scale re-render the tail without re-deriving the head.
    peer_args = _router_peer_args(router, peers, "prefill") + _router_peer_args(
        router, peers, "decode"
    )

    return RouterPlan(
        image=render(router.image, variables, context="router image"),
        command=command + peer_args,
        env={
            key: render(value, variables, context=f"router env '{key}'")
            for key, value in router.env.items()
        },
        health_path=router.health_path,
        metrics=router.capabilities.metrics,
        models_endpoint=router.capabilities.models_endpoint,
    )


def _router_peer_args(router: PDRouter, peers: PeerMap, role: str) -> List[str]:
    spec = getattr(router.peers, role)
    if not spec:
        return []
    addresses = list(peers.get(role) or [])
    if not addresses:
        raise RouterPeersUnavailable(
            f"No {role} address is available for the router. The router is "
            f"created only after its dependencies report ready, so this "
            f"means a member was lost between that check and this render."
        )
    if router.peers.style == PDPeerStyleEnum.REPEATED_FLAG:
        return _repeated_flag(spec, addresses)
    if router.peers.style == PDPeerStyleEnum.PARALLEL_LISTS:
        return _parallel_lists(spec, addresses)
    raise RouterPeersUnavailable(f"Peer style '{router.peers.style}' has no renderer.")


def is_managed_router(model, role_name: Optional[str], inference_backend=None) -> bool:
    """Whether GPUStack assembles this Router's selected launch configuration."""
    return _managed_router(model, role_name, inference_backend) is not None


def _managed_router(
    model, role_name: Optional[str], inference_backend=None
) -> Optional[PDRouter]:
    from gpustack.schemas.models import RoleNameEnum, find_role
    from gpustack.server.pd_mode_catalog import get_pd_mode

    if role_name != RoleNameEnum.ROUTER.value or find_role(model, role_name) is None:
        return None
    disaggregation = getattr(model, "disaggregation", None)
    if disaggregation is None:
        return None
    mode = get_pd_mode(disaggregation.mode.value)
    if mode is None or mode.router is None:
        return None
    launch = resolve_router_launch(model, inference_backend)
    if (
        mode.router.protocol == PDRouterProtocolEnum.USER_PROVIDED
        and not launch.command
    ):
        return None
    return mode.router


def managed_router_health_path(
    model, role_name: Optional[str], inference_backend=None
) -> Optional[str]:
    """Read the readiness path from the Router command's owner.

    Role and backend commands use an explicitly registered backend path when
    available; otherwise they retain generic readiness. Only a recipe command
    inherits the recipe executable's health endpoint.

    Measured on vllm-router 0.20.2: `/health` answers 200 with "RouterManager
    is healthy". Deliberately NOT `/health_generate`, which answers 503 "No
    routers with healthy workers available" on a group that is serving
    correctly -- it counts `regular` workers, and a disaggregated group has
    only `prefill` and `decode` ones.

    Returns None when no path is declared for the selected command.
    """
    if role_name != 'router':
        return None
    launch = resolve_router_launch(model, inference_backend)
    if launch.command:
        return inference_backend.health_check_path if inference_backend else None
    router = _managed_router(model, role_name, inference_backend)
    return router.health_path if router is not None else None


def apply_managed_router(
    model,
    role_name: Optional[str],
    *,
    inference_backend=None,
    peers: Optional[PeerMap] = None,
    variables: Optional[Mapping[str, object]] = None,
):
    """Turn a managed router's role-effective model into the shape the existing
    launch path already runs: a custom backend with an image and a command.

    Nothing new is taught to the worker. A managed router IS an image plus a
    command line, which is precisely what the custom backend already deploys,
    so expressing it that way means the container build, the port assignment,
    the health probe and the log capture all keep working with no router-shaped
    branch anywhere below this line.

    Called at two depths with different information, on purpose. The serve
    manager asks before the child process exists and passes no peers. It needs
    the backend choice for ports/registry and environment defaults for health
    checks. Runtime values and the command are rendered by the backend itself
    once its placement and peers are known.

    The result is never persisted — same rule as the projection it extends.
    A router's command holds its peers' addresses, and those change on every
    scale, so storing one would be storing a value that is wrong as soon as
    anything moves.
    """
    from gpustack.schemas.models import (
        BackendEnum,
        RoleEffectiveModel,
        find_role,
        role_effective_model,
    )
    from gpustack.server.pd_mode_catalog import get_pd_mode

    if not is_managed_router(model, role_name, inference_backend):
        return model

    projected = (
        model
        if isinstance(model, RoleEffectiveModel)
        else role_effective_model(model, role_name)
    )
    # A router runs a router binary, not an inference engine, so it must not
    # inherit the group's backend — a vLLM-shaped launch would try to serve the
    # model weights from the router's container.
    projected.backend = BackendEnum.CUSTOM

    launch = resolve_router_launch(model, inference_backend)
    if launch.version:
        projected.backend_version = launch.version
    # No engine command or parameters may escape the projection before runtime
    # rendering, including for recipes without a default Router invocation.
    projected.run_command = launch.command

    mode = get_pd_mode(model.disaggregation.mode.value)
    # Resolve defaults for both launch phases and every command source. User
    # env overrides backend/recipe defaults; only the worker knows runtime values.
    projected.env = router_environment(
        model,
        {
            **(mode.router.env if not launch.uses_backend_command else {}),
            **launch.env,
        },
    )
    if peers is None or variables is None:
        return projected
    projected.env = render_values(projected.env, variables)

    disaggregation = model.disaggregation
    mode_name = disaggregation.mode.value
    role = find_role(model, role_name)
    declared = role.backend_parameters if role else None
    if (
        mode_name == "vllm-mooncake"
        and not launch.command
        and "intra_node_data_parallel_size" in variables
    ):
        declared = _static_hybrid_router_parameters(
            declared, variables["intra_node_data_parallel_size"]
        )
    custom_command = launch.command
    if custom_command:
        # User templates own every flag. Rendering catalog peer arguments here
        # would impose its CLI and bootstrap requirements on another binary.
        router = mode.router
        plan = RouterPlan(
            image=render(
                router.image or "{{runner_image}}", variables, context="router image"
            ),
        )
        projected.run_command = shlex.join(
            render_router_command_template(
                custom_command,
                variables,
                peers,
                env=projected.env,
            )
        )
    else:
        plan = render_router(mode, variables, peers)
        if plan is None:
            return projected
        projected.run_command = shlex.join(plan.command)
    if not (role and role.image_name):
        projected.image_name = plan.image
    # Two different things arrive in this field and only one of them belongs
    # to the router.
    #
    # *Inherited* parameters are the group's engine parameters, projected here
    # like every other Model-level field — and the custom backend appends them
    # to whatever command it is given, which put `--max-model-len=8192` on a
    # `vllm-router` invocation that has no such flag. Those are dropped.
    #
    # Parameters the deployment wrote *on the router role* are the opposite
    # case: someone asked for a specific routing policy or breaker threshold.
    # Appending them works because every tunable flag is last-wins — verified
    # against both shipped wheels, `--decode-policy round_robin
    # --decode-policy cache_aware` parses to `cache_aware`. The flags the
    # platform renders from placement facts are refused at admission instead,
    # because `--prefill` and `--decode` are `action="append"` there and a
    # second one adds a phantom peer rather than replacing the injected one.
    projected.backend_parameters = list(declared) if declared is not None else []

    return projected


def group_peer_addresses(
    instances: Sequence[object],
    group_id: Optional[str],
    worker_ip_by_id: Mapping[int, str],
) -> Dict[str, List[PeerAddress]]:
    """Collect `{role: [(ip, port), ...]}` for one generation.

    Scoped by `group_id` rather than by model id, and that is the pairing
    boundary doing its job: a router must not be able to resolve a member of
    another generation, which is what makes a cross-generation pair
    structurally impossible rather than merely unlikely (F7 3.3).

    Sorted, so that a re-render with an unchanged membership produces an
    unchanged command — otherwise every reconcile would look like a spec change
    to anything comparing commands.
    """
    from gpustack.utils.pd_router_membership import active_router_peers

    peers: Dict[str, List[PeerAddress]] = {}
    for instance, ip, port, named in active_router_peers(
        instances, group_id, worker_ip_by_id
    ):
        peers.setdefault(instance.role, []).append(PeerAddress(ip, port, named))

    for role in peers:
        # By address only. The named ports are a mapping and would not compare,
        # and two members never share an (ip, port) anyway — so sorting on the
        # pair is both total and stable.
        peers[role].sort(key=lambda peer: (peer.ip, peer.port))
    return peers
