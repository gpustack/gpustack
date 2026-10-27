"""Which members can answer a whole request.

There are two places that route to an instance — the gateway's upstream
registration and the direct proxy — and this rule has to hold in both. It did
not: the gateway filtered to the router while the proxy balanced across every
running member, so a disaggregated group deployed on a gateway-less install
answered two thirds of its requests wrongly. Not with an error: a prefill
returns after one token and a decode runs without the prefix its KV was meant
to carry, and both reply 200 with plausible text.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.models import (
    Model,
    RoleSpec,
    SourceEnum,
    servable_instances,
)


def _model(roles=None):
    return Model(
        id=1,
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        roles=roles,
    )


def _pd_roles():
    return [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="decode", replicas=1),
        RoleSpec(name="router", replicas=1),
    ]


def _i(name, role=None):
    return SimpleNamespace(name=name, role=role)


def test_a_role_less_model_serves_from_every_replica():
    instances = [_i("a"), _i("b")]
    assert servable_instances(_model(), instances) == instances


def test_a_group_serves_only_from_its_router():
    router = _i("r", "router")
    got = servable_instances(
        _model(_pd_roles()), [_i("p", "prefill"), _i("d", "decode"), router]
    )
    assert got == [router]


def test_a_group_without_a_running_router_serves_from_nothing():
    """Falling back to the GPU members would be the same wrong answer by
    another route. An empty result lets the caller report the group as
    unavailable, which is true."""
    got = servable_instances(
        _model(_pd_roles()), [_i("p", "prefill"), _i("d", "decode")]
    )
    assert got == []


def test_several_routers_are_all_offered():
    """The router is fixed at one replica, but the rule is about role, not
    count — nothing here should have to change when that opens up."""
    routers = [_i("r1", "router"), _i("r2", "router")]
    got = servable_instances(_model(_pd_roles()), [_i("p", "prefill"), *routers])
    assert got == routers


def test_the_two_routing_paths_use_the_same_rule():
    """A rule this consequential must not be able to hold in one path and not
    the other, which is why there is one function rather than two filters."""
    import inspect

    from gpustack.routes import openai
    from gpustack.server import controllers

    assert "servable_instances" in inspect.getsource(openai.get_running_instance)
    assert "servable_instances" in inspect.getsource(
        controllers._gateway_registrable_instances
    )


@pytest.mark.parametrize("roles", [None, []])
def test_an_empty_role_list_is_not_a_group(roles):
    instances = [_i("a"), _i("b")]
    assert servable_instances(_model(roles), instances) == instances
