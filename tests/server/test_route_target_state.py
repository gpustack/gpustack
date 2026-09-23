"""A route target's state is derived, not remembered.

Remembered state strands a target whose recovery event nobody heard: if the
`modelroutetarget` subscription is down while the model goes UNAVAILABLE and
comes back RUNNING, the model has nothing left to transition and the target
sits UNAVAILABLE indefinitely — `/v1/models` returns an empty list while the
deployment serves correctly when addressed directly.

Edge-triggered state cannot survive a consumer that was not listening. These
assert the property that replaces it: given a target and its model, the answer
is computable, so a pass that runs later can repair one that was lost.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.model_routes import TargetStateEnum
from gpustack.schemas.models import ModelStateEnum
from gpustack.server.controllers import derive_route_target_state


def _target(**kwargs):
    fields = {"model_id": None, "provider_id": None, "state": TargetStateEnum.ACTIVE}
    fields.update(kwargs)
    return SimpleNamespace(name="t", **fields)


def _model(state=ModelStateEnum.RUNNING, ready_replicas=1):
    return SimpleNamespace(state=state, ready_replicas=ready_replicas)


def test_a_running_model_makes_its_target_active():
    target = _target(model_id=1, state=TargetStateEnum.UNAVAILABLE)
    assert derive_route_target_state(target, _model()) is TargetStateEnum.ACTIVE


def test_a_model_that_cannot_serve_makes_its_target_unavailable():
    target = _target(model_id=1, state=TargetStateEnum.ACTIVE)
    assert (
        derive_route_target_state(target, _model(state=ModelStateEnum.PARTIAL))
        is TargetStateEnum.UNAVAILABLE
    )


def test_the_answer_does_not_depend_on_the_state_already_stored():
    """The property the repair rests on.

    A derivation that consulted the current value could not correct a wrong
    one — it would agree with whatever it found, which is exactly how the
    measured outage persisted.
    """
    model = _model()
    for stored in (TargetStateEnum.ACTIVE, TargetStateEnum.UNAVAILABLE):
        assert (
            derive_route_target_state(_target(model_id=1, state=stored), model)
            is TargetStateEnum.ACTIVE
        )


def test_a_provider_target_is_active_regardless_of_any_model():
    """Its availability belongs to the provider, not to us."""
    target = _target(provider_id=9, state=TargetStateEnum.UNAVAILABLE)
    assert derive_route_target_state(target, None) is TargetStateEnum.ACTIVE
    assert (
        derive_route_target_state(target, _model(state=ModelStateEnum.PARTIAL))
        is TargetStateEnum.ACTIVE
    )


def test_a_target_pointing_at_nothing_is_unavailable_rather_than_undefined():
    """The previous code left its `target_state` variable unbound on this
    path, so whether the target changed at all depended on which branch had
    run before it."""
    assert derive_route_target_state(_target(), None) is TargetStateEnum.UNAVAILABLE


def test_a_model_target_whose_model_is_gone_is_unavailable():
    assert (
        derive_route_target_state(_target(model_id=404), None)
        is TargetStateEnum.UNAVAILABLE
    )


@pytest.mark.parametrize(
    "state, servable",
    [
        (ModelStateEnum.RUNNING, True),
        (ModelStateEnum.PARTIAL, False),
        (ModelStateEnum.PENDING, False),
        (ModelStateEnum.ERROR, False),
    ],
)
def test_every_model_state_maps_to_one_target_state(state, servable):
    """One predicate, no per-shape special case: RUNNING means servable and
    nothing else does."""
    expected = TargetStateEnum.ACTIVE if servable else TargetStateEnum.UNAVAILABLE
    assert (
        derive_route_target_state(_target(model_id=1), _model(state=state)) is expected
    )
