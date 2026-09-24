"""The PD-mode catalog turned into what one member's engine actually starts with.

`pd-modes.yaml` declares a recipe per mode and, inside it, per role: which
named port bands the role needs, which connector descriptor it runs, and which
env / args / files carry them. Nothing consumed that declaration until this
module: a group came up with the roles scheduled, the ports allocated and not
one `--kv-transfer-config` or `VLLM_NIXL_SIDE_CHANNEL_HOST` on any command
line, which is a deployment that looks healthy while serving aggregated.

The rendering itself is `gpustack.utils.template` — the same substitution the
shared-cache provider catalog uses, with the same three rules that make a
missing value diagnosable:

1. an unknown placeholder is left verbatim and logged, never blanked (a
   blank host is a plausible-looking wrong value; `{{worker_ip}}` reaching
   the engine is a `ZMQError: No such device` that names itself);
2. `{{ x }}` with inner spaces is not a placeholder;
3. env values do not see each other.

What this module adds is the *context*: the renderer takes a flat map keyed by
the whole dotted name, and assembling it — the instance's port bands, the
worker's NIC, the other roles' fields, the connector's lease window — is the
caller's job. That is what `_pd_variables` below does.
"""

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from gpustack.schemas.models import role_effective_model
from gpustack.schemas.pd_modes import PDKVLeaseTargetEnum, PDMode, PDModeRole
from gpustack.server.pd_mode_catalog import get_pd_mode
from gpustack.server.pd_pairing import (
    selector_cards_per_replica,
    selector_spans_workers,
)
from gpustack.utils.command import find_int_parameter, flatten_to_argv
from gpustack.utils.template import render, render_values
from gpustack.worker.kv_transfer import KV_TRANSFER_CONFIG_FLAG

logger = logging.getLogger(__name__)

_WHOLE_PLACEHOLDER = re.compile(r"^\{\{[A-Za-z_][A-Za-z0-9_.]*\}\}$")
"""A value that is nothing but one placeholder. Such a value renders to its
source's own type — a port band's base is a number in a connector descriptor,
not the string "5600" — while a value with text around it stays a string."""

# Parallelism is not a `RoleSpec` field: it is an engine parameter, and the
# Ascend recipe needs each side's connector config to carry the other side's.
# Read from the role's effective backend parameters under the spellings the
# two engines accept. What an absent one means depends on the member: for one
# on a single worker the engine derives tp from the cards it was given and runs
# dp at 1, so `_implicit_parallelism` supplies exactly that rule. For a member
# spanning workers the shape decides, that decision is made further down the
# vLLM path, and a guess here would put a number in a Mooncake descriptor the
# engine then contradicts — a pairing that fails at handshake. So there it
# stays unresolved, and the launch refuses rather than starting wrong.
_PARALLELISM_ALIASES: Dict[str, List[str]] = {
    "tensor_parallel_size": ["tensor-parallel-size", "tp", "tp-size"],
    "data_parallel_size": ["data-parallel-size", "dp", "dp-size"],
    "pipeline_parallel_size": ["pipeline-parallel-size", "pp", "pp-size"],
}

ACCELERATOR_COUNT_KEY = "accelerator_count"
"""Deliberately not in `_PARALLELISM_ALIASES`: it is read off the schedule --
the cards the member actually received -- rather than parsed out of the
parameters, so it needs no aliases and cannot disagree with what was placed."""


def band_specs_for(model, role: Optional[str], *, context: str = "") -> List:
    """The port bands this role's connector declares, or none.

    Empty for a plain deployment, for a role with no declaration, and for a
    catalog that cannot answer -- the last one logged, because a band that
    should exist and does not is a launch that will fail further down naming
    the placeholder.

    One reader for the scheduler and the worker both. They ask the same
    question at two moments (budgeting room, then allocating it), and an answer
    that differed between them would be a scheduler promising a worker space
    the worker then cannot find.
    """
    from gpustack.schemas.models import RoleNameEnum

    disaggregation = getattr(model, "disaggregation", None)
    mode_name = getattr(disaggregation, "mode", None)
    if not mode_name or not role:
        return []

    try:
        mode = get_pd_mode(str(mode_name))
    except Exception as e:
        # A broken catalog fails the server at start-up. Here it must not take
        # down an otherwise startable instance, nor scheduling on its way past.
        logger.warning(
            "Failed to load the PD mode catalog while resolving port bands "
            "for %s: %s",
            context or f"role {role!r}",
            e,
        )
        return []
    if mode is None:
        logger.warning(
            "%s declares PD mode '%s', which is not in the catalog. No named "
            "ports assigned.",
            context or f"Role {role!r}",
            mode_name,
        )
        return []

    # A router's declaration lives on `mode.router`, not in `mode.roles` -- it
    # is a command line, not an engine.
    holder = mode.router if role == RoleNameEnum.ROUTER.value else mode.role(role)
    return list(getattr(holder, "ports", None) or [])


def band_count_key(spec) -> Optional[str]:
    """The placeholder a templated `count` names, or None for a literal width.

    Exposed so a caller can say *why* a band did not resolve without reparsing
    the template itself -- "no accelerators assigned" and "no such parallelism
    parameter" send an operator to two different places.
    """
    count = spec.count
    if isinstance(count, int):
        return None
    return count[2:-2] if count.startswith("{{") and count.endswith("}}") else count


def band_width(spec, *, cards: int, backend_parameters) -> Optional[int]:
    """How wide one declared band is, or None when nothing can say.

    A templated `count` is the connector's own rule about its base. Two
    spellings resolve here:

    `{{accelerator_count}}` -- the accelerators this member was scheduled onto.
    Mooncake's `kv_port` is a base and the connector binds one port per *worker
    rank*, not per tensor-parallel rank:

        handshake_port = kv_port
                       + data_parallel_rank * tp_size * pp_size [* pcp_size]
                       + (pp_rank + pcp_rank) * tp_size + tp_rank

    so the band a member holds is `dp x tp x pp` wide -- its own card count. A
    `{{tensor_parallel_size}}` reading of the same rule is indistinguishable at
    TP8/DP1 and wrong at DP4xTP4, where it reserves 4 ports for a member that
    binds 16.

    `{{tensor_parallel_size}}` and friends -- the parallelism *this role* starts
    with, read off the same parameter spellings `render_pd_injection` reads for
    the `tp_size` it renders into the connector descriptor. The two have to
    agree, which is why they share one alias table.

    **None, never a guess, when the parameter is absent.** vLLM's own default is
    1, but GPUStack injects a tensor-parallel size of its own further down the
    vLLM path, so "the user did not write -tp" does not mean "one rank". The two
    failure modes are not comparable either: a band one port wide where the
    connector binds eight is exactly the collision this mechanism exists to
    prevent, and it surfaces as an instance wedged in `starting`, while an
    unallocated band leaves `{{ports.<name>}}` in the launch for the engine to
    reject by name.

    The caller decides what None costs -- the budget counts it as zero, the
    allocator skips it and says so -- which is the one place the two sides
    legitimately differ.
    """
    if isinstance(spec.count, int):
        return max(spec.count, 1)

    key = band_count_key(spec)

    if key == ACCELERATOR_COUNT_KEY:
        # No accelerators means either a role that claims none (a router, which
        # declares no KV band) or a member not yet scheduled. Neither is a width.
        return cards if cards >= 1 else None

    aliases = _PARALLELISM_ALIASES.get(key)
    resolved = (
        find_int_parameter(backend_parameters or [], aliases) if aliases else None
    )
    if resolved is not None and resolved >= 1:
        return resolved
    return None


# The RoleSpec fields worth exposing as {{roles.<role>.<field>}}. The
# deployment-shaped overrides (env, backend_parameters, selectors) are
# deliberately not among them: they are lists and mappings, and a template can
# only ever want a scalar.
_ROLE_SCALAR_FIELDS = (
    "name",
    "replicas",
    "backend",
    "backend_version",
    "image_name",
)


class PDInjectionError(Exception):
    """A PD injection that cannot be rendered into a correct launch.

    Raised, not degraded. A shared cache that fails to attach leaves a
    servable instance; a PD member that starts without its connector state
    joins a group it cannot hand KV to, returns 200s, and reports every NIXL
    metric as zero. The two failure postures have to differ.
    """


class PDInjection(BaseModel):
    """What one member's engine launch gains from its PD role."""

    env: Dict[str, str] = {}
    args: List[str] = []
    files: Dict[str, str] = {}
    """Container path -> contents; written by the serving script before the
    engine starts (Mooncake's transfer-engine config is read only from a file
    the engine is pointed at, so nothing else can carry it)."""
    host_mounts: List[str] = []
    """Host paths to bind read-only at the same path inside the container.
    See `PDModeRole.host_mounts` for why this cannot be a `files` entry."""


def render_pd_injection(
    model,
    instance,
    variables: Dict[str, object],
    peers: Optional[Dict[str, List[Tuple[str, int]]]] = None,
) -> Optional[PDInjection]:
    """Render this instance's role of `pd-modes.yaml` into env / args / files.

    `model` is the **unprojected** Model: it has to answer for every role, not
    just this one. `{{roles.decode.tensor_parallel_size}}` read off a
    projection would resolve decode's inherited parameters against *prefill's*
    effective ones, because a projection has already pushed the running role's
    overrides up to the Model level — a wrong number rather than a failure.
    The current role's own effective values are re-derived here, so nothing is
    lost by taking the unprojected spec.

    `variables` is the deployment-level context (`_template_variables()`),
    already carrying `{{worker_ip}}` / `{{port}}` / `{{role}}` / `{{group_id}}`
    and the two values only the worker can resolve, `{{net_device}}` and
    `{{runner_image}}`. Everything else is assembled here.

    Returns None when this is not a PD instance — no `disaggregation`, no
    role, an unknown mode, or a role the recipe injects nothing into
    (`custom`, and the router, whose command is assembled by the router path
    rather than merged into an engine's). The caller then takes today's path
    byte for byte.

    `peers` is only meaningful for a router. It is accepted so the signature
    is the one the router path will call, and ignored here: a router's role is
    not in `mode.roles`, so this returns None before reaching it. P and D
    never read across roles for an address (they rendezvous through the
    connector), which is why nothing else needs it.
    """
    disaggregation = getattr(model, "disaggregation", None)
    role_name = getattr(instance, "role", None)
    if disaggregation is None or not role_name:
        return None

    mode_name = _enum_value(getattr(disaggregation, "mode", None))
    mode = get_pd_mode(mode_name) if mode_name else None
    if mode is None:
        # The loader asserts the catalog and PDModeEnum agree at start-up, so
        # this is only reachable from a caller holding a mode this process
        # never loaded. Loud, because the symptom is silent aggregation.
        logger.warning(
            "PD mode '%s' is not in the catalog; instance %s (role '%s') starts "
            "with no connector configuration at all.",
            mode_name,
            getattr(instance, "name", None),
            role_name,
        )
        return None

    role = mode.role(role_name)
    if role is None:
        return None

    if peers:
        logger.debug(
            "Ignoring peer addresses for role '%s': peers are the router's, and "
            "the router's command is not assembled here.",
            role_name,
        )

    effective = role_effective_model(model, role_name)
    user_owns_kv_config = _user_owns_kv_transfer_config(role, effective)

    context = _pd_variables(model, mode, instance, effective, variables)
    where = f"PD mode '{mode.name}' role '{role_name}'"

    env = dict(render_values(role.env, context) or {})
    _apply_kv_lease_env(mode, env, where)

    args = [render(token, context, context=f"{where} args") for token in role.args]
    descriptor = (
        None
        if user_owns_kv_config
        else _render_tree(role.connector, context, f"{where} connector")
    )
    if user_owns_kv_config:
        # Warning rather than info, and level with the env override beside it:
        # the recipe's connector is the whole of what makes the role talk to
        # its peer, so withholding it can leave a group that starts, serves and
        # transfers nothing. A deliberate override is worth one line of log; an
        # accidental one is worth finding.
        logger.warning(
            "Role '%s' sets %s itself; PD mode '%s' is not adding its own "
            "connector descriptor. The role's own value is what the engine "
            "gets.",
            role_name,
            KV_TRANSFER_CONFIG_FLAG,
            mode.name,
        )
    if descriptor:
        # Rendered whole, as this role's own connector. If an extended KV cache
        # also contributes one, `kv_transfer.compose_kv_transfer_config` folds
        # the two into a MultiConnector once the whole argv exists — the two
        # are complementary, and which of them a role should ask first is a
        # property of the role, not of this render.
        args += [
            KV_TRANSFER_CONFIG_FLAG,
            json.dumps(descriptor, separators=(",", ":")),
        ]

    files = {
        render(path, context, context=f"{where} file path"): render(
            content, context, context=f"{where} file {path}"
        )
        for path, content in (role.files or {}).items()
    }

    # Deduplicated in declaration order: two roles of one recipe naming the
    # same path is normal, and a repeated bind is an error in some runtimes.
    host_mounts = list(
        dict.fromkeys(
            render(path, context, context=f"{where} host mount")
            for path in (role.host_mounts or [])
        )
    )

    injection = PDInjection(env=env, args=args, files=files, host_mounts=host_mounts)
    _refuse_unrendered(injection, where)
    logger.info(
        "PD injection for role '%s' of mode '%s': %d env, %d args, %d files.",
        role_name,
        mode.name,
        len(injection.env),
        len(injection.args),
        len(injection.files),
    )
    return injection


_ANY_PLACEHOLDER = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_.]*\}\}")


def _refuse_unrendered(injection: "PDInjection", where: str) -> None:
    """Stop a launch carrying a placeholder that never got a value.

    The renderer deliberately leaves an unknown name in place rather than
    blanking it, because a blank is a plausible-looking wrong value. That is
    right for the render and wrong for the launch: the string then travels all
    the way into the engine.

    An *argument* survives that trip visibly — vLLM echoes its argv into the
    log, and the log scanner recognises the shape. An *environment variable*
    does not. `HCCL_SOCKET_IFNAME={{net_device}}` reaches HCCL as the literal
    name of an interface that does not exist, and what comes back is a
    transport that quietly never connects. A placeholder gets this far whenever
    a host has several candidate NICs and `derive_net_device` refuses to guess
    between them — a defensible refusal with an invisible consequence. (A
    recipe whose `{{net_device}}` is a control-plane socket is derived from
    `Worker.ifname` and never arrives here that way; every other route to an
    underivable NIC does.)

    So the check moves to where both halves are already in hand and neither
    has left the process yet. Raising rather than warning, for the reason
    `PDInjectionError` exists: a PD member that starts without its connection
    state joins a group it cannot hand KV to and answers 200 to everything.
    """
    unrendered = []
    for name, value in sorted(injection.env.items()):
        for match in _ANY_PLACEHOLDER.finditer(str(value)):
            unrendered.append(f"{name}={match.group(0)}")
    for token in injection.args:
        for match in _ANY_PLACEHOLDER.finditer(str(token)):
            unrendered.append(match.group(0))
    for path in injection.host_mounts:
        # An unrendered mount path is the same class of failure as an
        # unrendered env: the bind either fails or creates an empty directory
        # where the transport expects a file, and neither says why.
        for match in _ANY_PLACEHOLDER.finditer(str(path)):
            unrendered.append(match.group(0))
    for path, content in sorted(injection.files.items()):
        # The most invisible of the four. A placeholder here is in no argv echo
        # and no environment -- it is written into a config file the engine
        # reads, and the interface name or port band it was meant to carry is
        # exactly what a transfer engine fails on, far from anything that names
        # the cause.
        for match in _ANY_PLACEHOLDER.finditer(str(path)):
            unrendered.append(match.group(0))
        for match in _ANY_PLACEHOLDER.finditer(str(content)):
            unrendered.append(f"{path}: {match.group(0)}")

    if not unrendered:
        return

    # Named individually rather than counted: the operator has to know which
    # one to go and set, and `{{net_device}}` and `{{ports.kv_port}}` are
    # fixed in entirely different places.
    raise PDInjectionError(
        f"{where}: {len(unrendered)} configuration value(s) would reach the "
        f"engine unrendered — {', '.join(unrendered)}. A placeholder with no "
        "value is a port band that was not allocated, a network interface "
        "that could not be derived (set `kv_transfer_ifname` on the worker when the "
        "host has several), or a parallelism the role never declared."
    )


def _pd_variables(
    model,
    mode: PDMode,
    instance,
    effective,
    variables: Dict[str, object],
) -> Dict[str, object]:
    """The deployment context plus everything only the catalog's consumer can
    resolve. Flat and dotted, because that is the renderer's contract."""
    context: Dict[str, object] = dict(variables or {})
    context.update(_port_variables(instance))
    context.update(_disaggregation_variables(getattr(model, "disaggregation", None)))
    context.update(_kv_lease_variables(mode))
    context.update(_cross_role_variables(model, instance))
    # The running role's own fields, unprefixed: a declaration referring to
    # its own parallelism writes {{tensor_parallel_size}}, and only the
    # cross-role case needs the prefix.
    context.update(_role_fields(effective, getattr(instance, "role", None), instance))
    return context


def _port_variables(instance) -> Dict[str, object]:
    """`{{ports.<name>}}` is a band's base, `{{ports.<name>.count}}` its width.

    A band the allocator has not written yet is simply absent, so the
    placeholder survives into the launch with a warning beside it — which is
    what an unallocated connector port should look like, rather than a port 0
    the engine binds happily.
    """
    context: Dict[str, object] = {}
    for name, band in (getattr(instance, "named_ports", None) or {}).items():
        base = (
            band.get("base") if isinstance(band, dict) else getattr(band, "base", None)
        )
        count = (
            band.get("count", 1)
            if isinstance(band, dict)
            else getattr(band, "count", 1)
        )
        if base is None:
            continue
        context[f"ports.{name}"] = base
        context[f"ports.{name}.count"] = count
    return context


def _disaggregation_variables(disaggregation) -> Dict[str, object]:
    """`DisaggregationSpec`'s own fields, e.g. `{{kv_load_failure_policy}}`."""
    if disaggregation is None:
        return {}
    try:
        dumped = disaggregation.model_dump(mode="json")
    except Exception:  # pragma: no cover - a caller passing a stand-in
        return {}
    return {key: value for key, value in dumped.items() if value is not None}


def _kv_lease_variables(mode: PDMode) -> Dict[str, object]:
    """The connector's KV lease window, keyed by the name it is configured
    under (`{{kv_lease_duration}}` for NIXL).

    GPUStack's own default wins over the engine's when the catalog declares
    one: the point of the registry is one window across connectors instead of
    inheriting a 30s-to-480s spread from whichever connector a mode happens
    to use.
    """
    lease = mode.kv_lease
    if lease is None or not lease.settable or not lease.param:
        return {}
    value = lease.gpustack_default
    if value is None:
        value = lease.engine_default
    if value is None:
        return {}
    return {lease.param: value}


def _apply_kv_lease_env(mode: PDMode, env: Dict[str, str], where: str) -> None:
    """Set the lease window when the connector takes it as an env var.

    `inject_to: env` is otherwise a declaration nothing acts on: unlike
    `connector_extra_config`, no role declaration mentions the variable — an
    env-configured window would silently keep the engine's default (8 minutes
    for Mooncake, 16x NIXL's, with no Prometheus counter to notice it). A
    declaration that does mention it, or a per-model env, still wins.
    """
    lease = mode.kv_lease
    if lease is None or not lease.settable or not lease.param:
        return
    if lease.inject_to != PDKVLeaseTargetEnum.ENV:
        return
    if lease.param in env:
        return
    value = lease.gpustack_default
    if value is None:
        return
    logger.debug("%s: setting KV lease window %s=%s", where, lease.param, value)
    env[lease.param] = str(value)


def _cross_role_variables(model, instance=None) -> Dict[str, object]:
    """`{{roles.<role>.<field>}}` — the coupling Mooncake needs and NIXL does
    not: prefill's connector config carries decode's parallelism and vice
    versa. Each role is resolved through its own projection, so a role that
    overrides nothing reads the Model-level value rather than the running
    role's.

    `instance` is the running member and belongs to exactly one of these
    roles. Handing it to all of them made the *other* side's implicit tensor
    parallelism the running member's card count: a four-card prefill rendering
    `{{roles.decode.tensor_parallel_size}}` as 4 while decode runs on one card
    writes a number into the Mooncake descriptor that decode's engine then
    contradicts — the handshake failure `_implicit_parallelism` exists to
    avoid, produced by `_implicit_parallelism` itself.

    The peer's own member is not available to correct it with, and not by
    oversight: the injection is rendered on the worker as this member starts,
    and prefill routinely starts before decode has been placed at all, so its
    `gpu_indexes` may not exist yet. What the *spec* can still say is said —
    `pd_pairing.selector_cards_per_replica` reads a peer that pinned its own
    cards, which is the shape where the two sides provably differ. A peer that
    pinned nothing falls back to the running member's count, deliberately: with
    no per-role pin both roles are sized by the same auto-selection of the same
    model on the same engine, and refusing to render there would break the
    silent 1P1D that `_implicit_parallelism` was written for.
    """
    context: Dict[str, object] = {}
    running_role = getattr(instance, "role", None)
    for role in getattr(model, "roles", None) or []:
        name = getattr(role, "name", None)
        if not name:
            continue
        projected = role_effective_model(model, name)
        is_peer = name != running_role
        cards = selector_cards_per_replica(role) if is_peer else None
        # A peer pinned across machines is left unresolved rather than sized
        # from the member that happens to be starting. Its own pin says None
        # for the same reason an absent pin does, so without this the peer
        # borrows the running member's card count -- a four-card prefill
        # rendering `{{roles.decode.tensor_parallel_size}}` as 4 while decode
        # spans two workers, which is the number going into a Mooncake
        # descriptor that decode's engine then contradicts. Unresolved, the
        # placeholder survives to `_refuse_unrendered`, and the launch refuses
        # instead of pairing badly.
        borrows_from = None if (is_peer and selector_spans_workers(role)) else instance
        fields = _role_fields(projected, name, borrows_from, pinned_cards=cards)
        for field, value in fields.items():
            context[f"roles.{name}.{field}"] = value
    return context


def _role_fields(
    effective,
    role_name: Optional[str],
    instance=None,
    *,
    pinned_cards: Optional[int] = None,
) -> Dict[str, object]:
    """One role's referenceable fields, read off its effective Model.

    `pinned_cards` overrides what the running member's own cards would say, for
    the cross-role case where the role being described is not the one starting.
    """
    fields: Dict[str, object] = {}
    role = None
    for candidate in getattr(effective, "roles", None) or []:
        if getattr(candidate, "name", None) == role_name:
            role = candidate
            break
    for field in _ROLE_SCALAR_FIELDS:
        value = getattr(role, field, None)
        if value is None:
            value = getattr(effective, field, None)
        if value is not None:
            fields[field] = _enum_value(value)

    parameters = getattr(effective, "backend_parameters", None) or []
    for field, aliases in _PARALLELISM_ALIASES.items():
        try:
            value = find_int_parameter(parameters, aliases)
        except Exception:  # pragma: no cover - malformed parameters
            value = None
        if value is not None:
            fields[field] = value

    fields.update(
        _implicit_parallelism(fields, instance, role_name, pinned_cards=pinned_cards)
    )
    return fields


def _implicit_parallelism(
    declared: Dict[str, object],
    instance,
    role_name: Optional[str],
    *,
    pinned_cards: Optional[int] = None,
) -> Dict[str, object]:
    """The parallelism a single-worker member has whether or not it says so.

    A role that writes no `--tensor-parallel-size` is not a role with an
    unknown one: for a member on a single worker the engine path derives it
    from the cards the member was given, and a member with no data parallelism
    runs at one. Both are the rule the engine will apply, read at a point that
    already knows the inputs — not a default chosen here.

    Why this is needed at all: the Ascend recipe's connector config carries
    *both* sides' parallelism, so without this a 1P1D whose roles declare none
    leaves `{{roles.prefill.data_parallel_size}}` in the launch and fails,
    forcing the user to write out parameters that only restate what the engine
    was going to do anyway.

    `pinned_cards` is how a role *other* than the running one gets its own
    answer: `instance` is the member that is starting, and its card count is
    only this role's when this role is the one starting. See
    `_cross_role_variables` for why a peer that pinned nothing still borrows it.

    Deliberately silent for a member spanning workers. There the shape decides
    dp and dpl, that decision happens further down the vLLM path, and guessing
    here would put a number in a Mooncake descriptor that the engine then
    contradicts — a pairing that fails at handshake, which is worse than the
    launch refusing.
    """
    if instance is None or _spans_workers(instance):
        return {}

    out: Dict[str, object] = {}
    if "tensor_parallel_size" not in declared:
        cards = (
            pinned_cards
            if pinned_cards is not None
            else len(getattr(instance, "gpu_indexes", None) or [])
        )
        if cards:
            out["tensor_parallel_size"] = cards
    if "data_parallel_size" not in declared:
        out["data_parallel_size"] = 1
    return out


def _spans_workers(instance) -> bool:
    servers = getattr(instance, "distributed_servers", None)
    return bool(servers and getattr(servers, "subordinate_workers", None))


def _render_tree(value: Any, variables: Dict[str, object], where: str) -> Any:
    """Render a connector descriptor in place, keeping its structure.

    The descriptor is JSON the engine parses, not a string GPUStack pastes, so
    a value that is nothing but a placeholder comes back as the type its
    source had — `kv_port` and `tp_size` are numbers in the config Ascend was
    measured running, and a quoted "8" is a different document.
    """
    if isinstance(value, str):
        rendered = render(value, variables, context=where)
        return _coerce(value, rendered)
    if isinstance(value, dict):
        return {
            key: _render_tree(item, variables, f"{where}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_render_tree(item, variables, where) for item in value]
    return value


def _coerce(template: str, rendered: Optional[str]) -> Any:
    """Give a whole-placeholder value back its source's type.

    Only when something was actually substituted: a placeholder left
    unresolved stays the literal it is, which is the whole point of leaving it
    there.
    """
    if rendered is None or rendered == template:
        return rendered
    if not _WHOLE_PLACEHOLDER.match(template):
        return rendered
    try:
        return int(rendered)
    except (TypeError, ValueError):
        return rendered


def _user_owns_kv_transfer_config(role: PDModeRole, effective) -> bool:
    """Whether the role's own parameters already carry `--kv-transfer-config`.

    vLLM reads the flag once and every source of it expands into a whole JSON
    document rather than a fragment, so two of them means argparse keeps one
    and the other is silently absent.

    A user-supplied flag is legitimate: the recipe's descriptor is seeded into
    the role's parameter list as an ordinary editable row, so setting it is the
    documented way to change it. Editing it wrongly breaks the group, and that
    is the user's to own.

    What must not happen is BOTH. So the injection drops its own descriptor
    when the user supplies one: one assembler owns the flag either way, and the
    user is the one who asked to be it.

    Only recipes carrying a connector descriptor are affected. SGLang's
    disaggregation is configured through its own flags and never collides.
    """
    if not role.connector:
        return False

    parameters = flatten_to_argv(getattr(effective, "backend_parameters", None) or [])
    return any(
        token == KV_TRANSFER_CONFIG_FLAG
        or token.startswith(KV_TRANSFER_CONFIG_FLAG + "=")
        for token in parameters
    )


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value
