"""A router parameter that would collide with an injected one.

The connection flags (``--host``, ``--port``, ``--prometheus-*``,
``--kv-connector``) are not refused: those are last-wins and a deployment's own
parameters are appended after the declared command, so setting one simply
overrides it — which is what every other role's injected parameter allows too.
Binding the router somewhere the gateway is not looking is the user's to own.

What is still refused is the one case where overriding is not what happens:
``--prefill`` / ``--decode`` are ``action="append"`` in both routers, so a
second one does not replace the injected peer. It adds one the router then
forwards to and cannot reach, and the only symptom is a member that quietly
never gets traffic — a failure no message on the flag itself could explain.
"""

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.models import _reject_router_params_the_platform_owns
from gpustack.schemas.models import DisaggregationSpec, PDModeEnum, RoleSpec


def _roles(router_params):
    return [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="decode", replicas=1),
        RoleSpec(name="router", replicas=1, backend_parameters=router_params),
    ]


def _spec(mode=PDModeEnum.VLLM_NIXL):
    return DisaggregationSpec(mode=mode)


def test_a_router_with_no_parameters_of_its_own_is_untouched():
    _reject_router_params_the_platform_owns(_roles(None), _spec())
    _reject_router_params_the_platform_owns(_roles([]), _spec())


@pytest.mark.parametrize("param", ["--prefill", "--decode"])
def test_a_peer_flag_is_refused(param):
    """The one kind of collision overriding cannot fix."""
    with pytest.raises(BadRequestException) as excinfo:
        _reject_router_params_the_platform_owns(_roles([param, "x"]), _spec())
    # The message has to name the flag and offer the alternative, or the user
    # is left guessing which of their parameters was the problem.
    assert param in excinfo.value.message
    assert "--prefill-policy" in excinfo.value.message


@pytest.mark.parametrize(
    "param",
    ["--host", "--port", "--prometheus-port", "--kv-connector"],
)
def test_a_connection_flag_is_now_the_users_to_set(param):
    """Asserted a refusal until the form started offering these as ordinary
    editable rows. Appended after the declared command, so the user's value is
    the one the router starts with."""
    _reject_router_params_the_platform_owns(_roles([param, "x"]), _spec())


def test_both_spellings_of_a_peer_flag_are_caught():
    """`--flag=value` is as valid on a command line as `--flag value`, and a
    check that only splits on whitespace lets the first one through."""
    for spelling in ("--prefill=1.2.3.4:8000", "--decode=1.2.3.4:8001"):
        with pytest.raises(BadRequestException):
            _reject_router_params_the_platform_owns(_roles([spelling]), _spec())


@pytest.mark.parametrize(
    "param",
    ["--prefill-policy", "--decode-policy", "--cb-failure-threshold"],
)
def test_a_tunable_flag_is_accepted(param):
    """The point of the split. `--prefill-policy` shares a prefix with the
    refused `--prefill`, so a check written as a prefix match would forbid
    exactly the flag the feature exists to allow."""
    _reject_router_params_the_platform_owns(_roles([param, "cache_aware"]), _spec())


def test_a_flag_the_platform_does_not_own_is_accepted():
    """Anything the recipe never mentions is the user's business — the refusal
    list is what the catalog declares, not an allowlist of known flags."""
    _reject_router_params_the_platform_owns(
        _roles(["--shutdown-grace-period-secs", "30"]), _spec()
    )


def test_the_refused_set_follows_the_mode():
    """Read off the chosen recipe rather than a list in Python, so each mode
    refuses its own router's peer flags."""
    # Connection flags are nobody's to refuse now, on any recipe.
    _reject_router_params_the_platform_owns(
        _roles(["--kv-connector", "nixl"]), _spec(PDModeEnum.SGLANG_MOONCAKE)
    )
    _reject_router_params_the_platform_owns(
        _roles(["--host", "1.2.3.4"]), _spec(PDModeEnum.SGLANG_MOONCAKE)
    )


def test_a_hand_written_mode_refuses_nothing():
    """`custom` injects no connection state at all, so every flag is the
    user's — refusing one would be refusing a value nothing else supplies."""
    _reject_router_params_the_platform_owns(
        _roles(["--host", "1.2.3.4"]), _spec(PDModeEnum.CUSTOM)
    )
