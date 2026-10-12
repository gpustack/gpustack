"""Validate PD role graphs and mode-specific deployment constraints."""

from typing import Optional, Union

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import (
    PD_BACKENDS,
    DisaggregationSpec,
    Model,
    ModelCreate,
    ModelSpecBase,
    ModelUpdate,
    PDModeEnum,
    RoleNameEnum,
)
from gpustack.server.pd_mode_catalog import get_pd_mode
from gpustack.utils.command import flatten_to_argv
from gpustack.validation.pd.common import model_field


def validate_roles(  # noqa: C901
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    stored: Optional[Model] = None,
) -> None:
    """Validate role structure and mode-specific constraints on the merged spec.

    A sparse update must still satisfy the rules for stored roles. This check
    reads effective fields without changing what the request persists.

    Args:
        model_in: The submitted deployment specification.
        stored: The stored deployment supplying omitted update fields.
    """

    def field(name: str):
        return model_field(model_in, stored, name)

    roles = field("roles")
    disaggregation = field("disaggregation")

    if not roles:
        if disaggregation is not None:
            raise BadRequestException(
                message="disaggregation requires roles: declare a prefill and a decode role."
            )
        return

    names = [role.name for role in roles]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise BadRequestException(
            message=f"Duplicate role name(s): {', '.join(sorted(duplicates))}."
        )

    allowed = {item.value for item in RoleNameEnum}
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise BadRequestException(
            message=(
                f"Unsupported role name(s): {', '.join(unknown)}. "
                f"Supported roles are {', '.join(sorted(allowed))}."
            )
        )

    for role in roles:
        if role.name == RoleNameEnum.ROUTER.value and role.replicas != 1:
            raise BadRequestException(
                message="The router role runs exactly one replica."
            )
        # Refused rather than ignored. Prefill and decode get CPU and memory
        # from sizing — the weights and the parallelism decide them — so a
        # hand-written value here would be a second source for the same number
        # that silently disagrees with the estimate.
        if role.resources is not None and role.name != RoleNameEnum.ROUTER.value:
            raise BadRequestException(
                message=(
                    f"Role '{role.name}' cannot declare CPU or memory: only the "
                    "router does, because it holds no weights. Every other "
                    "role's footprint is derived from the model."
                )
            )
        # The router registers each member under the base model's name, so
        # a role-level adapter cannot be addressed through its member table.
        if role.lora_list:
            raise BadRequestException(
                message=(
                    f"Role '{role.name}' cannot declare LoRA adapters: a "
                    "multi-role deployment is reached through its router, and "
                    "the router's member table is indexed by served-model "
                    "name, so an adapter name attached to one role never "
                    "resolves to that role. Declare lora_list on the model "
                    "instead, or deploy the adapters as their own model."
                )
            )
        if role.resources is not None:
            if role.resources.cpu is not None and role.resources.cpu <= 0:
                raise BadRequestException(
                    message="The router's CPU request must be greater than zero."
                )
            if role.resources.memory is not None and role.resources.memory <= 0:
                raise BadRequestException(
                    message="The router's memory request must be greater than zero."
                )

    # `dependencies` is a start order, so a cycle is a deployment that never
    # starts. Reject it here rather than letting the controller spin.
    known = set(names)
    graph = {role.name: list(role.dependencies or []) for role in roles}
    for name, deps in graph.items():
        for dep in deps:
            if dep not in known:
                raise BadRequestException(
                    message=f"Role '{name}' depends on '{dep}', which is not declared."
                )
            if dep == name:
                raise BadRequestException(
                    message=f"Role '{name}' cannot depend on itself."
                )
    visiting: set = set()
    done: set = set()

    def _walk(name: str) -> None:
        if name in done:
            return
        if name in visiting:
            raise BadRequestException(
                message=f"Role dependencies form a cycle through '{name}'."
            )
        visiting.add(name)
        for dep in graph.get(name, []):
            _walk(dep)
        visiting.discard(name)
        done.add(name)

    for name in graph:
        _walk(name)

    # `roles[].replicas` is the only scaling truth, so a model-level count
    # above one would be a second one. Refuse instead of quietly reading it as
    # a multiplier — an implicit mode switch is exactly what makes a
    # deployment stop matching its own spec.
    if field("replicas") not in (0, 1):
        raise BadRequestException(
            message=(
                "A model with roles uses replicas as an on/off switch (0 or 1). "
                "Scale a disaggregated deployment through roles[].replicas."
            )
        )

    # The scaling scheduler writes `model.replicas` directly, without passing
    # through this validation, so a window rule holding 3 would break the
    # deployment at its next tick rather than at submit time.
    schedule = field("scaling_schedule")
    if schedule and schedule.enabled:
        raise BadRequestException(
            message="Scheduled scaling is not supported for a model with roles."
        )

    if disaggregation is None:
        return

    counts = {name: names.count(name) for name in allowed}
    if counts[RoleNameEnum.PREFILL.value] != 1:
        raise BadRequestException(
            message="A disaggregated model needs exactly one prefill role."
        )
    if counts[RoleNameEnum.DECODE.value] != 1:
        raise BadRequestException(
            message="A disaggregated model needs exactly one decode role."
        )
    if counts[RoleNameEnum.ROUTER.value] > 1:
        raise BadRequestException(
            message="A disaggregated model has at most one router."
        )

    _reject_lora_under_disaggregation(field)
    _reject_an_engine_that_cannot_be_disaggregated(field, roles)
    _reject_cache_under_a_hand_written_mode(field, roles, disaggregation)
    _reject_a_policy_the_mode_cannot_apply(disaggregation)
    _reject_router_params_the_platform_owns(roles, disaggregation)

    # A recipe injects one engine's connector configuration into every role,
    # so a role on a different engine would receive settings it cannot read.
    #
    # Read off the recipe rather than a table restating it. The catalog is
    # already loaded by the time this line runs -- the two rejections above
    # resolve the same mode -- so the second source of truth was buying a
    # dict lookup and owing a start-up assertion that the two still agreed.
    mode = get_pd_mode(disaggregation.mode.value)
    permitted = list(mode.backends) if mode else []
    if permitted:
        for role in roles:
            role_backend = role.backend or field("backend")
            if role_backend and role_backend not in permitted:
                raise BadRequestException(
                    message=(
                        f"Role '{role.name}' runs backend '{role_backend}', which "
                        f"pd mode '{disaggregation.mode.value}' cannot configure "
                        f"(it targets {', '.join(permitted)}). Mixing engines "
                        f"across roles requires pd mode 'custom', where the "
                        f"connection parameters are yours to supply."
                    )
                )


def _reject_router_params_the_platform_owns(roles, disaggregation) -> None:
    """Refuse peer flags that would append unreachable members to the router.

    The catalog declares ownership of peer flags, which append rather than
    replace upstreams. Tunable and connection flags remain user-overridable.
    """
    router = next(
        (
            r
            for r in roles
            if r.name == RoleNameEnum.ROUTER.value and r.backend_parameters
        ),
        None,
    )
    if router is None:
        return
    mode = get_pd_mode(disaggregation.mode.value)
    if mode is None or mode.router is None:
        return
    owned = set(mode.router.platform_owned_flags)
    if not owned:
        return
    for param in flatten_to_argv(router.backend_parameters):
        # Both spellings a user can write: `--flag value` and `--flag=value`.
        name = str(param).split("=", 1)[0].strip()
        if name in owned:
            raise BadRequestException(
                message=(
                    f"'{name}' on the router is set by GPUStack from where the "
                    f"group was placed, so it cannot be given here. Adjustable "
                    f"router flags for this mode: "
                    f"{', '.join(a.flag for a in mode.router.tunable_args) or 'none'}."
                )
            )


def _mode_renders_connector_key(mode, key: str) -> bool:
    """Whether any of the mode's roles renders `key` into its connector
    descriptor.

    Reads the descriptors rather than searching the serialized mode for
    `{{key}}`: a full-text search over `model_dump_json()` also matches the
    word appearing in a description, and misses a recipe that supplies a
    literal value instead of a placeholder. Both are the wrong answer for the
    question being asked, which is whether the user's value reaches the engine.
    """
    for role in (mode.roles or {}).values():
        if key in (role.connector or {}):
            return True
    return False


def _reject_a_policy_the_mode_cannot_apply(disaggregation) -> None:
    """`kv_load_failure_policy` is a vLLM/NIXL setting, not a platform one.

    Only `vllm-nixl` renders it. The SGLang modes have no equivalent concept
    at all -- their KV lifecycle is a bootstrap timeout that aborts the
    request, not a load that can fail and be retried -- and Mooncake's
    connector does not read the key. So there is nothing to implement on the
    other three; what there is, is a value the user weighed and set that then
    quietly does nothing.

    Which is why this rejects rather than warns, and only for a non-default
    value. `fail` is what an engine that never sees the setting does anyway,
    so refusing it would break every group on those modes to no purpose;
    `recompute` is the deliberate choice -- trade a 500 for a silent
    recomputation -- and a user who made it and got neither is worse off than
    one who was told the mode cannot honour it.

    Derived from the recipe rather than a list of mode names: a mode that
    starts rendering the key is accepted the moment it does, with nothing here
    to remember to update.
    """
    policy = disaggregation.kv_load_failure_policy
    default = DisaggregationSpec.model_fields["kv_load_failure_policy"].default
    if policy == default:
        return

    mode_name = disaggregation.mode.value
    mode = get_pd_mode(mode_name)
    if mode is None or _mode_renders_connector_key(mode, "kv_load_failure_policy"):
        return

    if disaggregation.mode == PDModeEnum.CUSTOM:
        # The one mode where the setting may well be reachable, just not from
        # here: `custom` injects nothing, so every connector key is the user's
        # to write. Pointing them at another mode would be the wrong advice.
        raise BadRequestException(
            message=(
                f"pd mode 'custom' injects no connector configuration, so "
                f"kv_load_failure_policy='{policy}' would be stored and never "
                f"reach the engine. Set it inside your own "
                f"--kv-transfer-config instead."
            )
        )

    raise BadRequestException(
        message=(
            f"pd mode '{mode_name}' cannot apply kv_load_failure_policy="
            f"'{policy}': its KV connector has no such setting, so the value "
            f"would be stored and never reach the engine. Leave it at "
            f"'{default}', or use a mode whose connector reads it."
        )
    )


def _reject_lora_under_disaggregation(field) -> None:
    """LoRA adapters and disaggregation cannot be asked for together yet.

    **The combination deploys and then refuses every request**, which is why
    it is refused here instead of documented. The group reaches `running` with
    all three members, the adapter's route `<base>:<adapter>` is created and
    reports a ready target, a chat against the base name answers 200 — and a
    chat against the adapter name answers 503 `No available workers`. The
    engines are innocent: prefill and decode each list the adapter in their own
    `GET /v1/models` and each answer a direct chat on it.

    The wall is the router in front of the group. Its worker registry is indexed
    by served-model name, and a member registers itself under `model_id:
    "{{model_name}}"` — the base name, once. So the registry has no entry under
    the adapter's name to hand the request to, and the honest reading of that is
    that adapter names are not part of a group's addressing scheme at all.
    Making them so is a change to the router's membership protocol, not a field
    on this model.

    Judged against the merged state, so neither direction of an update slips
    past: adding adapters to a group that already disaggregates, and adding
    disaggregation to a model that already carries adapters, are the same
    combination arriving from opposite sides.

    Admission-time only. A row that already holds both keeps reconciling — this
    is never consulted outside create and update — because refusing to converge
    a deployment that exists would take away the running base model too, and
    the base model is the part that works.
    """
    if not field("lora_list"):
        return

    raise BadRequestException(
        message=(
            "A disaggregated deployment cannot serve LoRA adapters. A group is "
            "addressed through its router, whose member table is indexed by "
            "served-model name, and its members register under the base model's "
            "name only — so a request naming an adapter reaches no member and "
            "fails with 'No available workers', even though both engines have "
            "the adapter loaded. Either remove the adapters from this "
            "deployment and serve them from a non-disaggregated one, which is "
            "unaffected, or remove the disaggregation."
        )
    )


def _reject_an_engine_that_cannot_be_disaggregated(field, roles) -> None:
    """An engine PD does not apply to, whatever the mode.

    Distinct from the per-mode engine check further up, which asks whether a
    *recipe* can be injected into a role and lets `custom` through because it
    injects nothing. This one asks whether the engine can be disaggregated at
    all, and `custom` is not an exemption from it: supplying the connection
    parameters yourself does not give an engine a prompt KV cache to hand
    across. Without this, `custom` mode is a way around the check entirely --
    the `custom` recipe declares no `backends`, so the loop below does not run.

    Judged per role and against the model-level engine each role falls back to,
    because a group is only as disaggregable as the engine each member runs.
    """
    offenders = {
        role.backend or field("backend")
        for role in roles
        if (role.backend or field("backend")) not in PD_BACKENDS
    }
    offenders.discard(None)
    if not offenders:
        return

    raise BadRequestException(
        message=(
            f"Backend {', '.join(sorted(offenders))} cannot be disaggregated. "
            f"Prefill/decode splits the prompt KV cache across two engines, "
            f"which only {', '.join(PD_BACKENDS)} do here. Deploy this model "
            f"without disaggregation, or switch it to one of those backends."
        )
    )


def _reject_cache_under_a_hand_written_mode(field, roles, disaggregation) -> None:
    """`custom` mode and an extended KV cache cannot be asked for together.

    Everywhere else the two compose: GPUStack folds the mode's connector and
    the cache's into one `MultiConnector`, which is what makes a disaggregated
    deployment with a shared cache a supported combination rather than a
    choice between them.

    `custom` is the one mode that injects no connection state at all — its
    whole contract is that the parameters are the user's. So there is nothing
    to compose the cache with, and quietly injecting a connector under a mode
    that promises not to would be the surprise this rejection exists to
    prevent. Written by hand, both still fit in one flag; the engine composes
    connectors and the user is the one holding the pen.
    """
    if disaggregation.mode != PDModeEnum.CUSTOM:
        return

    model_cache = field("extended_kv_cache")
    offenders = []
    for role in roles:
        cache = (
            role.extended_kv_cache
            if role.extended_kv_cache is not None
            else model_cache
        )
        if cache is None or not getattr(cache, "enabled", False):
            continue
        offenders.append(role.name)
    if offenders:
        names = ", ".join(repr(name) for name in offenders)
        subject = "Role" if len(offenders) == 1 else "Roles"
        verb = "enables" if len(offenders) == 1 else "enable"
        raise BadRequestException(
            message=(
                f"{subject} {names} {verb} the extended KV cache under pd "
                "mode 'custom', which injects no connector configuration at "
                "all — so there is nothing for GPUStack to compose the cache "
                "into. Either choose a pd mode that configures a connector, "
                "where the two are combined for you, or keep 'custom' and "
                "write the combined configuration into backend_parameters."
            )
        )


def validate_pd_cache_provider(
    disaggregation: Optional[DisaggregationSpec],
    provider_name: str,
    *,
    role: Optional[str] = None,
) -> None:
    """Validate the mode's support for a visible cache service and role.

    Args:
        disaggregation: The selected PD mode, or None for a standalone model.
        provider_name: The provider of a cache service already checked for access.
        role: The member using the service, or None for a model-level declaration.
    """
    if not disaggregation:
        return
    where = f" on role '{role}'" if role else ""
    mode = get_pd_mode(disaggregation.mode)
    if mode is None:
        raise BadRequestException(
            message=f"Unknown PD mode '{disaggregation.mode}'" + where
        )
    reason = mode.cache_service_reason(provider_name, role)
    if reason:
        raise BadRequestException(message=reason + where)
