"""Structural validation of a multi-role (PD) model on create/update.

``validate_roles`` is the API-layer gate on the role graph. It rejects
rather than reinterprets: every rule here exists because folding the request
into something adjacent — reading ``replicas: 3`` as a multiplier, letting a
schedule overwrite it later, injecting one engine's connector into another —
produces a deployment that behaves unlike what the user typed.
"""

from contextlib import contextmanager

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.models import validate_roles
from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    ModelCreate,
    ModelUpdate,
    PDModeEnum,
    RoleSpec,
    ScalingSchedule,
    ScalingScheduleRule,
)


def _model(**kwargs):
    base = dict(
        name="m",
        source="local_path",
        local_path="/models/m",
        backend="vLLM",
    )
    base.update(kwargs)
    return ModelCreate(**base)


def _pd(mode=PDModeEnum.VLLM_NIXL):
    return DisaggregationSpec(mode=mode)


@contextmanager
def rejects(fragment):
    """The API's HTTPException carries its text on ``.message``, not on
    ``str()``, so ``pytest.raises(match=...)`` would match the empty string."""
    with pytest.raises(BadRequestException) as excinfo:
        yield
    assert fragment in excinfo.value.message, excinfo.value.message


def test_no_roles_is_untouched():
    validate_roles(_model())


def test_disaggregation_without_roles_is_rejected():
    with rejects("requires roles"):
        validate_roles(_model(disaggregation=_pd()))


def test_plain_multi_role_needs_no_disaggregation():
    # roles without disaggregation is orchestration, not PD.
    validate_roles(
        _model(roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")], replicas=1)
    )


def test_minimal_pd_group_is_accepted():
    validate_roles(
        _model(
            roles=[
                RoleSpec(name="prefill", replicas=3),
                RoleSpec(name="decode"),
                RoleSpec(
                    name="router", cpu_only=True, dependencies=["prefill", "decode"]
                ),
            ],
            disaggregation=_pd(),
            replicas=1,
        )
    )


def test_stopped_pd_group_is_accepted():
    # replicas is the on/off switch, so zero is a stopped group, not an error.
    validate_roles(
        _model(
            roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
            disaggregation=_pd(),
            replicas=0,
        )
    )


def test_duplicate_role_name_is_rejected():
    with rejects("Duplicate role name"):
        validate_roles(
            _model(roles=[RoleSpec(name="prefill"), RoleSpec(name="prefill")])
        )


def test_unknown_role_name_is_rejected():
    # The data model allows any name; this validation does not.
    with rejects("Unsupported role name"):
        validate_roles(_model(roles=[RoleSpec(name="encoder")]))


def test_router_is_pinned_to_one_replica():
    with rejects("exactly one replica"):
        validate_roles(
            _model(
                roles=[
                    RoleSpec(name="prefill"),
                    RoleSpec(name="decode"),
                    RoleSpec(name="router", replicas=2),
                ]
            )
        )


def test_zero_role_replicas_is_rejected_by_the_schema():
    # Not wanting a role means removing it. A zero would leave dependencies
    # pointing at a role that never appears.
    with pytest.raises(ValueError):
        RoleSpec(name="prefill", replicas=0)


def test_dependency_on_an_undeclared_role_is_rejected():
    with rejects("not declared"):
        validate_roles(
            _model(roles=[RoleSpec(name="prefill", dependencies=["decode"])])
        )


def test_self_dependency_is_rejected():
    with rejects("depend on itself"):
        validate_roles(
            _model(roles=[RoleSpec(name="prefill", dependencies=["prefill"])])
        )


def test_dependency_cycle_is_rejected():
    with rejects("cycle"):
        validate_roles(
            _model(
                roles=[
                    RoleSpec(name="prefill", dependencies=["decode"]),
                    RoleSpec(name="decode", dependencies=["router"]),
                    RoleSpec(name="router", dependencies=["prefill"]),
                ]
            )
        )


@pytest.mark.parametrize("replicas", [2, 3, 8])
def test_model_replicas_above_one_is_rejected(replicas):
    # Not silently read as a multiplier: an implicit mode switch is the thing
    # this rule exists to prevent.
    with rejects("on/off switch"):
        validate_roles(
            _model(
                roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
                replicas=replicas,
            )
        )


def test_enabled_scaling_schedule_is_rejected():
    # The scheduler writes model.replicas directly, bypassing this validation,
    # so a window rule holding 3 would break the group at its next tick.
    schedule = ScalingSchedule(
        enabled=True,
        baseline_replicas=1,
        rules=[
            ScalingScheduleRule(
                name="peak", start_cron="0 9 * * *", duration_seconds=32400, replicas=3
            )
        ],
    )
    with rejects("Scheduled scaling"):
        validate_roles(
            _model(
                roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
                replicas=1,
                scaling_schedule=schedule,
            )
        )


def test_disabled_scaling_schedule_is_allowed():
    schedule = ScalingSchedule(
        enabled=False,
        baseline_replicas=1,
        rules=[
            ScalingScheduleRule(
                name="peak", start_cron="0 9 * * *", duration_seconds=32400, replicas=3
            )
        ],
    )
    validate_roles(
        _model(
            roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
            replicas=1,
            scaling_schedule=schedule,
        )
    )


@pytest.mark.parametrize(
    "roles, expected",
    [
        ([RoleSpec(name="prefill")], "one decode role"),
        ([RoleSpec(name="decode")], "one prefill role"),
        (
            [
                RoleSpec(name="prefill"),
                RoleSpec(name="decode"),
                RoleSpec(name="router"),
            ],
            None,
        ),
    ],
)
def test_disaggregation_requires_one_prefill_and_one_decode(roles, expected):
    model = _model(roles=roles, disaggregation=_pd(), replicas=1)
    if expected is None:
        validate_roles(model)
    else:
        with rejects(expected):
            validate_roles(model)


def test_mismatched_role_backend_is_rejected():
    # vllm-nixl expands into NixlConnector plus VLLM_NIXL_* env; handing that
    # to another engine fails silently at run time, so reject it at submit.
    with rejects("pd mode 'vllm-nixl' cannot configure"):
        validate_roles(
            _model(
                roles=[
                    RoleSpec(name="prefill"),
                    RoleSpec(name="decode", backend="SGLang"),
                ],
                disaggregation=_pd(),
                replicas=1,
            )
        )


def test_model_level_backend_is_checked_too():
    with rejects("cannot configure"):
        validate_roles(
            _model(
                backend="SGLang",
                roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
                disaggregation=_pd(PDModeEnum.VLLM_NIXL),
                replicas=1,
            )
        )


def test_custom_mode_permits_a_mixed_engine_group():
    # #5871: vLLM prefill with a different decode engine. Allowed only here,
    # where the user supplies the connection state themselves.
    validate_roles(
        _model(
            roles=[
                RoleSpec(name="prefill", backend="vLLM"),
                RoleSpec(name="decode", backend="Custom"),
            ],
            disaggregation=_pd(PDModeEnum.CUSTOM),
            replicas=1,
        )
    )


def test_sglang_mode_accepts_sglang_roles():
    validate_roles(
        _model(
            backend="SGLang",
            roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
            disaggregation=_pd(PDModeEnum.SGLANG_MOONCAKE),
            replicas=1,
        )
    )


# --- a sparse PUT is judged against the merged state -----------------------


def _stored(**kwargs):
    """The row a PUT is updating."""
    base = dict(
        name="m",
        source="local_path",
        local_path="/models/m",
        backend="vLLM",
        replicas=1,
    )
    base.update(kwargs)
    return Model(**base)


def _enabled_schedule():
    return ScalingSchedule(
        enabled=True,
        baseline_replicas=1,
        rules=[
            ScalingScheduleRule(
                name="peak", start_cron="0 9 * * *", duration_seconds=32400, replicas=3
            )
        ],
    )


def test_a_sparse_put_cannot_add_a_schedule_to_a_stored_group():
    # The request never mentions `roles`, so judged on its own it looks like a
    # plain model and the schedule would be accepted onto a group — the exact
    # combination the rule forbids, reached by not mentioning the thing that
    # makes it illegal.
    stored = _stored(
        roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
        disaggregation=_pd(),
    )
    submitted = ModelUpdate(
        name="m",
        source="local_path",
        local_path="/models/m",
        backend="vLLM",
        replicas=1,
        scaling_schedule=_enabled_schedule(),
    )

    validate_roles(submitted)  # no stored row: passes, which is the hole
    with rejects("Scheduled scaling"):
        validate_roles(submitted, stored=stored)


def test_a_sparse_put_cannot_raise_replicas_on_a_stored_group():
    stored = _stored(
        roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
        disaggregation=_pd(),
    )
    submitted = ModelUpdate(
        name="m",
        source="local_path",
        local_path="/models/m",
        backend="vLLM",
        replicas=5,
    )

    with rejects("on/off switch"):
        validate_roles(submitted, stored=stored)


def test_a_submitted_value_still_wins_over_the_stored_one():
    # Removing the roles in the same request must be allowed to make the
    # schedule legal again — the merge is a fallback, not an override.
    stored = _stored(roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")])
    submitted = ModelUpdate(
        name="m",
        source="local_path",
        local_path="/models/m",
        backend="vLLM",
        replicas=3,
        roles=[],
        scaling_schedule=_enabled_schedule(),
    )

    validate_roles(submitted, stored=stored)


def test_a_stored_plain_model_is_unaffected():
    stored = _stored(replicas=3)
    submitted = ModelUpdate(
        name="m",
        source="local_path",
        local_path="/models/m",
        backend="vLLM",
        replicas=3,
        scaling_schedule=_enabled_schedule(),
    )

    validate_roles(submitted, stored=stored)


# --- one flag, two writers, composed --------------------------------------- #


def _cache(enabled=True):
    from gpustack.schemas.models import ExtendedKVCacheConfig

    return ExtendedKVCacheConfig(enabled=enabled)


def test_a_connector_mode_and_an_extended_cache_are_allowed_together():
    """They are complementary, not conflicting. A prefill that can ask a
    shared cache first skips the prefill work for a prefix it already holds,
    and what is left is what disaggregation exists to optimise — so the pair
    is worth more than either alone. GPUStack folds the two connectors into a
    MultiConnector; the engine has always been able to run them."""
    validate_roles(
        _model(
            extended_kv_cache=_cache(),
            roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
            disaggregation=_pd(),
        )
    )


def test_one_role_may_take_the_cache_and_the_other_not():
    """The composition is per role, so which sides take a cache is too — and
    the prefill side is the one where it pays."""
    validate_roles(
        _model(
            roles=[
                RoleSpec(name="prefill", extended_kv_cache=_cache()),
                RoleSpec(name="decode", extended_kv_cache=_cache(enabled=False)),
            ],
            disaggregation=_pd(),
        )
    )


def test_custom_mode_has_nothing_to_compose_the_cache_into():
    """`custom` injects no connection state at all — that is its whole
    contract. Quietly injecting a cache connector under a mode that promises
    not to inject anything is the surprise worth rejecting; by hand, both
    still fit in one flag."""
    with rejects("nothing for GPUStack to compose"):
        validate_roles(
            _model(
                backend=None,
                extended_kv_cache=_cache(),
                roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
                disaggregation=_pd(mode=PDModeEnum.CUSTOM),
            )
        )


def test_custom_mode_without_a_cache_is_fine():
    validate_roles(
        _model(
            backend=None,
            roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
            disaggregation=_pd(mode=PDModeEnum.CUSTOM),
        )
    )


def test_sglang_never_touched_that_flag_and_is_unaffected():
    """SGLang configures disaggregation through its own --disaggregation-*
    flags, so there was never a clash to compose or to reject."""
    validate_roles(
        _model(
            backend="SGLang",
            extended_kv_cache=_cache(),
            roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
            disaggregation=_pd(mode=PDModeEnum.SGLANG_MOONCAKE),
        )
    )
