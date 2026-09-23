"""Model-level status aggregation.

``sync_model_status`` is the single owner of the Model row's status fields: one
scan of the model's instances answers the RUNNING count, the lifecycle value,
the per-role detail and the degradation markers, and one change gate writes
them.

Two properties are load-bearing and pinned here:

* a model with no ``roles`` behaves exactly as it did before the field
  existed — ``ready_replicas`` stays a plain count of RUNNING instances, and
  the servability gate stays equivalent to ``ready_replicas > 0``;
* a group is servable when *every* role has at least one ready member, so a
  3P1D group with the router down is four RUNNING instances and no service.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.cache_services import CacheConfigSnapshot
from gpustack.schemas.model_routes import TargetStateEnum
from gpustack.schemas.principals import Principal
from gpustack.schemas.models import (
    DegradationReasonEnum,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelStateEnum,
    RoleSpec,
    RoleStatus,
    SourceEnum,
)
from gpustack.server import controllers
from gpustack.server.controllers import (
    ModelRouteTargetController,
    derive_model_state,
    is_model_servable,
    sync_model_status,
    upstream_registration_ready,
)
from gpustack.server.bus import Event, EventType


def _model(replicas=1, roles=None, **kwargs) -> Model:
    return Model(
        id=1,
        name="m",
        replicas=replicas,
        ready_replicas=0,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        roles=roles,
        **kwargs,
    )


def _pd_roles(prefill=3, decode=1, router=1):
    return [
        RoleSpec(name="prefill", replicas=prefill),
        RoleSpec(name="decode", replicas=decode),
        RoleSpec(name="router", replicas=router),
    ]


def _instance(
    id,
    state=ModelInstanceStateEnum.RUNNING,
    role=None,
    cache_config=None,
) -> ModelInstance:
    return ModelInstance(
        id=id,
        name=f"m-{id}",
        model_id=1,
        model_name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        state=state,
        role=role,
        cache_config=cache_config,
    )


def _members(counts, state=ModelInstanceStateEnum.RUNNING):
    """One instance per unit of ``{role: count}``, numbered uniquely."""
    instances = []
    for role, count in counts.items():
        for _ in range(count):
            instances.append(_instance(len(instances) + 1, state=state, role=role))
    return instances


async def _sync(model, instances):
    """Run the owner against a fake session, returning (changed, update_mock)."""
    update = AsyncMock()
    service = MagicMock(return_value=SimpleNamespace(update=update))
    with (
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=instances),
        ),
        patch("gpustack.server.controllers.ModelService", service),
    ):
        changed = await sync_model_status(MagicMock(), model)
    return changed, update


# --- role-less models: the compatibility baseline -------------------------- #


@pytest.mark.asyncio
async def test_role_less_pending_when_nothing_ready():
    model = _model(replicas=2)
    changed, _ = await _sync(model, [_instance(1, ModelInstanceStateEnum.DOWNLOADING)])

    assert changed is True
    assert model.state == ModelStateEnum.PENDING
    assert model.ready_replicas == 0
    assert model.state_message is None
    # Nothing role-shaped is invented for a plain deployment.
    assert model.role_status is None
    assert model.degradations is None


@pytest.mark.asyncio
async def test_role_less_short_of_replicas_serves_and_is_marked_degraded():
    """One ready replica serves, so the model is RUNNING; being short of the
    requested count is `ratio_unmet` beside the state, not inside it. PARTIAL
    means "up but unservable", which a role-less model cannot be."""
    model = _model(replicas=3)
    changed, _ = await _sync(
        model,
        [
            _instance(1),
            _instance(2, ModelInstanceStateEnum.STARTING),
            _instance(3, ModelInstanceStateEnum.PENDING),
        ],
    )

    assert changed is True
    assert model.state == ModelStateEnum.RUNNING
    assert model.state_message == "1/3 replicas ready"
    assert model.degradations == [DegradationReasonEnum.RATIO_UNMET.value]
    assert model.ready_replicas == 1


@pytest.mark.asyncio
async def test_role_less_running_when_count_met():
    model = _model(replicas=2)
    changed, _ = await _sync(model, [_instance(1), _instance(2)])

    assert changed is True
    assert model.state == ModelStateEnum.RUNNING
    assert model.state_message is None
    assert model.ready_replicas == 2


@pytest.mark.asyncio
async def test_role_less_error_when_nothing_ready_and_a_member_failed():
    model = _model(replicas=1)
    changed, _ = await _sync(model, [_instance(1, ModelInstanceStateEnum.ERROR)])

    assert changed is True
    assert model.state == ModelStateEnum.ERROR
    assert model.state_message == "1/1 instances failed"
    assert model.ready_replicas == 0


@pytest.mark.asyncio
async def test_a_failed_member_does_not_mask_a_serving_model():
    """Readiness is decided before failure: a deployment with members up is
    serving whatever else failed, so ERROR is reserved for "nothing ready"."""
    model = _model(replicas=2)
    changed, _ = await _sync(
        model, [_instance(1), _instance(2, ModelInstanceStateEnum.ERROR)]
    )

    assert changed is True
    assert model.state == ModelStateEnum.RUNNING
    assert model.degradations == [DegradationReasonEnum.RATIO_UNMET.value]
    assert model.ready_replicas == 1


@pytest.mark.asyncio
async def test_unreachable_members_read_as_not_ready_not_failed():
    """UNREACHABLE is a worker comms fault that clears when the worker comes
    back, so it is not a failure at the model level."""
    model = _model(replicas=1)
    changed, _ = await _sync(model, [_instance(1, ModelInstanceStateEnum.UNREACHABLE)])

    assert changed is True
    assert model.state == ModelStateEnum.PENDING
    assert model.ready_replicas == 0


@pytest.mark.asyncio
async def test_unchanged_status_is_not_written_back():
    """The change gate: a steady state produces no UPDATE, so watchers have
    nothing to churn on."""
    model = _model(replicas=1)
    instances = [_instance(1)]

    changed, update = await _sync(model, instances)
    assert changed is True
    assert update.await_count == 1

    changed, update = await _sync(model, instances)
    assert changed is False
    assert update.await_count == 0


@pytest.mark.asyncio
async def test_deleted_model_is_left_alone():
    model = _model(replicas=1, deleted_at=object())
    changed, update = await _sync(model, [_instance(1)])

    assert changed is False
    assert update.await_count == 0
    assert model.state is None


# --- groups: per-role readiness -------------------------------------------- #


@pytest.mark.asyncio
async def test_group_with_router_down_is_not_running():
    """3P1D with the router still down: four RUNNING instances, no service."""
    model = _model(replicas=1, roles=_pd_roles())
    instances = _members({"prefill": 3, "decode": 1})

    changed, _ = await _sync(model, instances)

    assert changed is True
    # The count keeps its meaning: it counts RUNNING instances, nothing else.
    assert model.ready_replicas == 4
    assert model.state == ModelStateEnum.PARTIAL
    assert model.state_message == "roles not ready: router"
    assert {name: (s.desired, s.ready) for name, s in model.role_status.items()} == {
        "prefill": (3, 3),
        "decode": (1, 1),
        "router": (1, 0),
    }
    assert is_model_servable(model) is False


@pytest.mark.asyncio
async def test_group_with_router_up_is_running():
    model = _model(replicas=1, roles=_pd_roles())
    instances = _members({"prefill": 3, "decode": 1, "router": 1})

    changed, _ = await _sync(model, instances)

    assert changed is True
    assert model.ready_replicas == 5
    assert model.state == ModelStateEnum.RUNNING
    assert model.state_message is None
    assert {name: (s.desired, s.ready) for name, s in model.role_status.items()} == {
        "prefill": (3, 3),
        "decode": (1, 1),
        "router": (1, 1),
    }
    assert is_model_servable(model) is True


@pytest.mark.asyncio
async def test_group_short_of_its_ratio_still_serves():
    """One ready member per role is the gate, not a full house: requiring
    every member would make a 3P1D scale-up unservable while it scales."""
    model = _model(replicas=1, roles=_pd_roles())
    instances = _members({"prefill": 1, "decode": 1, "router": 1})

    await _sync(model, instances)

    assert model.state == ModelStateEnum.RUNNING
    assert model.role_status["prefill"].desired == 3
    assert model.role_status["prefill"].ready == 1
    assert is_model_servable(model) is True


@pytest.mark.asyncio
async def test_group_desired_comes_from_the_spec_not_the_scan():
    """An instance that was never created has no state, so the denominator
    can only come from ``roles[].replicas``."""
    model = _model(replicas=1, roles=_pd_roles(prefill=4, decode=2))
    instances = _members({"router": 1})

    await _sync(model, instances)

    assert {name: (s.desired, s.ready) for name, s in model.role_status.items()} == {
        "prefill": (4, 0),
        "decode": (2, 0),
        "router": (1, 1),
    }
    assert model.state == ModelStateEnum.PARTIAL
    assert model.state_message == "roles not ready: decode, prefill"


@pytest.mark.asyncio
async def test_group_pending_when_no_member_is_ready():
    model = _model(replicas=1, roles=_pd_roles())
    instances = _members({"prefill": 3, "router": 1}, ModelInstanceStateEnum.STARTING)

    await _sync(model, instances)

    assert model.ready_replicas == 0
    assert model.state == ModelStateEnum.PENDING
    assert model.role_status["prefill"].ready == 0


@pytest.mark.asyncio
async def test_group_error_when_nothing_ready_and_members_failed():
    model = _model(replicas=1, roles=_pd_roles(prefill=1))
    instances = _members({"prefill": 1, "router": 1}, ModelInstanceStateEnum.ERROR)

    await _sync(model, instances)

    assert model.state == ModelStateEnum.ERROR
    assert model.state_message == "2/2 members failed"


@pytest.mark.asyncio
async def test_role_less_instances_are_counted_but_not_bucketed():
    """A member carrying no role still counts toward ``ready_replicas`` — the
    counter is a plain count — but it cannot satisfy a declared role."""
    model = _model(replicas=1, roles=_pd_roles(prefill=1))
    instances = _members({"prefill": 1, "decode": 1, "router": 1})
    instances.append(_instance(99))

    await _sync(model, instances)

    assert model.ready_replicas == 4
    assert sum(s.ready for s in model.role_status.values()) == 3
    assert model.state == ModelStateEnum.RUNNING


# --- degradation markers --------------------------------------------------- #


def _cache_config(injected: bool, reason=None) -> CacheConfigSnapshot:
    return CacheConfigSnapshot(cache_service_id=7, injected=injected, reason=reason)


@pytest.mark.asyncio
async def test_cache_not_injected_coexists_with_running():
    """A cache that never attached is a degradation, not a failure: the
    instance starts anyway, just slower."""
    model = _model(replicas=1)
    instances = [
        _instance(1, cache_config=_cache_config(False, "endpoint unreachable"))
    ]

    await _sync(model, instances)

    assert model.state == ModelStateEnum.RUNNING
    assert model.degradations == [DegradationReasonEnum.CACHE_NOT_INJECTED.value]
    assert model.state_message == ("shared cache not injected: endpoint unreachable")
    assert is_model_servable(model) is True


@pytest.mark.asyncio
async def test_cache_degradation_is_appended_to_the_state_message():
    model = _model(replicas=2)
    instances = [
        _instance(1, cache_config=_cache_config(False)),
        _instance(2, ModelInstanceStateEnum.PENDING),
    ]

    await _sync(model, instances)

    assert model.state == ModelStateEnum.RUNNING
    assert model.state_message == "1/2 replicas ready; shared cache not injected"
    # Two reasons coexist, and both coexist with RUNNING.
    assert model.degradations == [
        DegradationReasonEnum.RATIO_UNMET.value,
        DegradationReasonEnum.CACHE_NOT_INJECTED.value,
    ]


@pytest.mark.asyncio
async def test_no_cache_and_attached_cache_are_not_degradations():
    for cache_config in (None, _cache_config(True)):
        model = _model(replicas=1)
        await _sync(model, [_instance(1, cache_config=cache_config)])

        assert model.state == ModelStateEnum.RUNNING
        assert model.degradations is None
        assert model.state_message is None


@pytest.mark.asyncio
async def test_stale_is_left_unset():
    """``stale`` compares a member's ``spec_digest`` against the model's
    current digest, and nothing computes either yet. Deriving it from an
    unwritten column would mark every model stale."""
    model = _model(replicas=1)
    await _sync(model, [_instance(1)])

    assert model.stale is None


# --- the status actually persists ------------------------------------------ #


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Model.__table__.create)
        await conn.run_sync(ModelInstance.__table__.create)
        # The status owner resolves the model's workload namespace from
        # its owner Principal, so the round-trip needs that table too.
        await conn.run_sync(Principal.__table__.create)
    async with AsyncSession(engine) as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_status_round_trips_through_the_database(db_session):
    """The whole aggregate has to survive a write and a reload, and the
    reloaded row has to compare equal to a freshly computed one — otherwise
    the change gate fires on every pass and every watcher churns."""
    db_session.add(_model(replicas=1, roles=_pd_roles(prefill=2)))
    for instance in _members({"prefill": 2, "decode": 1, "router": 1}):
        db_session.add(instance)
    await db_session.commit()

    model = await Model.one_by_id(db_session, 1)
    assert await sync_model_status(db_session, model) is True

    reloaded = await Model.one_by_id(db_session, model.id)
    assert reloaded.state == ModelStateEnum.RUNNING
    assert reloaded.ready_replicas == 4
    assert {name: (s.desired, s.ready) for name, s in reloaded.role_status.items()} == {
        "prefill": (2, 2),
        "decode": (1, 1),
        "router": (1, 1),
    }

    # Second pass over an unchanged world: nothing to write.
    assert await sync_model_status(db_session, reloaded) is False


@pytest.mark.asyncio
async def test_degradations_round_trip_and_clear(db_session):
    db_session.add(_model(replicas=1))
    db_session.add(_instance(1, cache_config=_cache_config(False, "no route to host")))
    await db_session.commit()

    model = await Model.one_by_id(db_session, 1)
    assert await sync_model_status(db_session, model) is True
    reloaded = await Model.one_by_id(db_session, 1)
    assert reloaded.degradations == [DegradationReasonEnum.CACHE_NOT_INJECTED.value]

    # The marker is derived, so it clears itself once the cache attaches.
    instance = await ModelInstance.one_by_id(db_session, 1)
    instance.cache_config = _cache_config(True)
    db_session.add(instance)
    await db_session.commit()

    reloaded = await Model.one_by_id(db_session, 1)
    assert await sync_model_status(db_session, reloaded) is True
    reloaded = await Model.one_by_id(db_session, 1)
    assert reloaded.degradations is None
    assert reloaded.state_message is None


# --- the servability gate -------------------------------------------------- #


@pytest.mark.parametrize(
    "replicas, ready",
    [(1, 0), (1, 1), (3, 0), (3, 1), (3, 2), (3, 3), (0, 0), (0, 1)],
)
def test_gate_is_byte_for_byte_the_old_one_for_a_role_less_model(replicas, ready):
    """The ``ModelRouteTarget`` ACTIVE predicate moved from
    ``ready_replicas > 0`` to ``Model.state``. For a model with no roles the
    two must agree on every input, including a model scaling up (1 of 3
    ready) — taking that out of service would be a new outage, not a fix."""
    model = _model(replicas=replicas)
    model.ready_replicas = ready
    model.state, _ = derive_model_state(
        model,
        ready_replicas=ready,
        instance_count=max(replicas, ready),
        role_status=None,
        error_count=0,
    )

    assert is_model_servable(model) is (ready > 0)


def test_gate_falls_back_to_the_counter_for_rows_written_before_state_existed():
    """The migration adds ``state`` without a backfill, so a row can carry
    NULL until the owner first runs for it. Those rows must stay servable
    across the restart that introduces the field."""
    model = _model(replicas=1)
    model.state = None

    model.ready_replicas = 1
    assert is_model_servable(model) is True

    model.ready_replicas = 0
    assert is_model_servable(model) is False


@pytest.mark.parametrize(
    "state, expected",
    [
        (ModelStateEnum.RUNNING, True),
        (ModelStateEnum.PARTIAL, False),
        (ModelStateEnum.PENDING, False),
        (ModelStateEnum.ERROR, False),
    ],
)
def test_gate_requires_running_for_a_group(state, expected):
    """For a group PARTIAL means a role is at zero ready members, so the
    router cannot forward: not servable."""
    model = _model(replicas=1, roles=_pd_roles())
    model.state = state

    assert is_model_servable(model) is expected


# --- the upstream-registration seam ---------------------------------------- #


def test_upstream_registration_is_vacuous_without_a_router_role():
    assert upstream_registration_ready(_model(replicas=1)) is True
    assert (
        upstream_registration_ready(
            _model(replicas=1, roles=[RoleSpec(name="prefill", replicas=1)])
        )
        is True
    )


def test_a_group_parks_in_partial_when_the_upstream_is_not_registered():
    """The seam is the only place the second half of the RUNNING predicate is
    decided. Stub it out and the group must stay PARTIAL with a message
    rather than silently claiming to serve."""
    model = _model(replicas=1, roles=_pd_roles(prefill=1))
    role_status = {
        name: SimpleNamespace(desired=1, ready=1)
        for name in ("prefill", "decode", "router")
    }

    with patch(
        "gpustack.server.controllers.upstream_registration_ready",
        return_value=False,
    ):
        state, message = derive_model_state(
            model,
            ready_replicas=3,
            instance_count=3,
            role_status=role_status,
            error_count=0,
        )

    assert state == ModelStateEnum.PARTIAL
    assert message == "waiting for upstream registration"


# --- the gate as the route target sees it ---------------------------------- #


async def _run_sync_state(model, target):
    controller = ModelRouteTargetController(config=MagicMock())
    with (
        patch(
            "gpustack.server.controllers.ModelRouteTarget.one_by_id",
            AsyncMock(return_value=target),
        ),
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=model),
        ),
    ):
        await controller._sync_state(
            MagicMock(), target, Event(type=EventType.UPDATED, data=target)
        )


def _target():
    return SimpleNamespace(
        id=3,
        name="t",
        model_id=1,
        provider_id=None,
        state=TargetStateEnum.UNAVAILABLE,
        weight=42,
        is_default=True,
        update=AsyncMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state, expected",
    [
        (ModelStateEnum.RUNNING, TargetStateEnum.ACTIVE),
        # PARTIAL is unreachable for a plain model, and the gate is a plain
        # `state == RUNNING` — so were one ever written, it would correctly
        # read as unservable rather than as a special case.
        (ModelStateEnum.PARTIAL, TargetStateEnum.UNAVAILABLE),
        (ModelStateEnum.PENDING, TargetStateEnum.UNAVAILABLE),
        (ModelStateEnum.ERROR, TargetStateEnum.UNAVAILABLE),
    ],
)
async def test_route_target_state_for_a_plain_model(state, expected):
    model = _model(replicas=3)
    model.state = state
    target = _target()

    await _run_sync_state(model, target)

    assert target.state == expected
    # Weight and fallback mechanics are untouched by the gate change.
    assert target.weight == 42
    assert target.is_default is True


@pytest.mark.asyncio
async def test_route_target_goes_unavailable_when_a_group_loses_a_role():
    model = _model(replicas=1, roles=_pd_roles())
    model.state = ModelStateEnum.PARTIAL
    model.ready_replicas = 4
    target = _target()
    target.state = TargetStateEnum.ACTIVE

    await _run_sync_state(model, target)

    assert target.state == TargetStateEnum.UNAVAILABLE


# --- what the "state means servable, degradation sits beside it" choice buys


@pytest.mark.asyncio
async def test_partial_is_unreachable_for_a_role_less_model():
    """PARTIAL means "members up and still unservable". A role-less model
    serves on its first ready replica, so it has no such condition — which is
    what lets the servability gate be a plain `state == RUNNING` everywhere,
    with no per-shape special case."""
    for replicas in range(0, 4):
        for ready in range(0, 4):
            for errors in range(0, 3):
                model = _model(replicas=replicas)
                instances = [_instance(i) for i in range(ready)] + [
                    _instance(100 + i, ModelInstanceStateEnum.ERROR)
                    for i in range(errors)
                ]
                await _sync(model, instances)
                assert model.state != ModelStateEnum.PARTIAL, (
                    replicas,
                    ready,
                    errors,
                )
                # And the gate still answers exactly what it answered before
                # this field existed.
                assert is_model_servable(model) is (ready > 0)


@pytest.mark.asyncio
async def test_a_group_below_its_ratio_serves_and_is_marked():
    """Every role covered but one below strength: servable, because requiring
    full staffing would take a 3P1D out of service for the whole of a
    scale-up. The shortfall is the marker's job."""
    model = _model(replicas=1, roles=_pd_roles(prefill=3, decode=1, router=1))
    await _sync(
        model,
        [
            _instance(1, role="prefill"),
            _instance(2, role="decode"),
            _instance(3, role="router"),
        ],
    )

    assert model.state == ModelStateEnum.RUNNING
    assert model.degradations == [DegradationReasonEnum.RATIO_UNMET.value]
    assert model.role_status["prefill"] == RoleStatus(desired=3, ready=1)
    assert is_model_servable(model) is True


@pytest.mark.asyncio
async def test_a_group_with_a_role_at_zero_is_not_called_degraded():
    """The marker says "serving, with less of it". A group missing a whole
    role serves nothing — PARTIAL says so — and `ratio_unmet` beside it read
    as reduced capacity for a deployment answering no requests at all.

    A count cannot tell these apart: this is two RUNNING instances either way.
    """
    model = _model(replicas=1, roles=_pd_roles(prefill=1, decode=1, router=1))
    await _sync(
        model,
        [
            _instance(1, role="prefill"),
            _instance(2, role="decode"),
            _instance(3, ModelInstanceStateEnum.ERROR, role="router"),
        ],
    )

    assert model.state == ModelStateEnum.PARTIAL
    assert model.state_message == "roles not ready: router"
    assert model.degradations is None
    assert model.ready_replicas == 2
    assert is_model_servable(model) is False


@pytest.mark.asyncio
async def test_a_group_whose_router_has_not_been_created_yet_is_not_degraded():
    """The same window, and it opens on every PD start: the router is created
    only once every GPU role has a RUNNING member (`_role_dependencies`), so
    between those two events the group is up, unservable, and not degraded."""
    model = _model(replicas=1, roles=_pd_roles(prefill=1, decode=1, router=1))
    await _sync(model, [_instance(1, role="prefill"), _instance(2, role="decode")])

    assert model.state == ModelStateEnum.PARTIAL
    assert model.degradations is None


@pytest.mark.asyncio
async def test_a_role_below_strength_beside_a_role_at_zero_still_says_nothing():
    """Both conditions at once. The shortfall is real, but reporting it while
    the group cannot serve puts two contradictory sentences on one row — and
    the actionable one is the role that is missing entirely."""
    model = _model(replicas=1, roles=_pd_roles(prefill=3, decode=1, router=1))
    await _sync(
        model,
        [
            _instance(1, role="prefill"),
            _instance(2, role="decode"),
        ],
    )

    assert model.state == ModelStateEnum.PARTIAL
    assert model.degradations is None
    assert model.role_status["prefill"] == RoleStatus(desired=3, ready=1)


@pytest.mark.asyncio
async def test_a_placement_marker_still_reports_while_unservable(monkeypatch):
    """Only the ratio is gated on serving. The rest describe placement or
    configuration and are true whether or not the deployment answers
    requests — a group whose pairing is all-remote is worth saying so about
    before it starts serving, not after."""
    model = _model(replicas=1, roles=_pd_roles(prefill=1, decode=1, router=1))
    monkeypatch.setattr(controllers, "_pairing_remote", lambda *a, **kw: True)
    await _sync(model, [_instance(1, role="prefill"), _instance(2, role="decode")])

    assert model.state == ModelStateEnum.PARTIAL
    assert model.degradations == [DegradationReasonEnum.PAIRING_REMOTE.value]


@pytest.mark.asyncio
async def test_a_group_at_full_strength_is_not_marked():
    model = _model(replicas=1, roles=_pd_roles(prefill=2, decode=1, router=1))
    await _sync(
        model,
        [
            _instance(1, role="prefill"),
            _instance(2, role="prefill"),
            _instance(3, role="decode"),
            _instance(4, role="router"),
        ],
    )

    assert model.state == ModelStateEnum.RUNNING
    assert model.degradations is None


@pytest.mark.asyncio
async def test_nothing_ready_is_not_called_degraded():
    """Otherwise every model would carry the marker throughout its first
    start, which is not a degradation but a normal beginning."""
    model = _model(replicas=3)
    await _sync(model, [_instance(1, ModelInstanceStateEnum.PENDING)])

    assert model.state == ModelStateEnum.PENDING
    assert model.degradations is None


# --- the owner has to run on the spec side too


@pytest.mark.asyncio
async def test_a_group_with_no_instances_still_gets_its_declared_shape():
    """A group parked at `replicas: 0` has no instances, so no instance event
    ever fires for it. Without a spec-side pass its `role_status` would stay
    NULL and the UI would have no declared shape to render — which the design
    requires to be present on the list response, not just the detail one."""
    model = _model(replicas=0, roles=_pd_roles(prefill=3, decode=1, router=1))
    changed, _ = await _sync(model, [])

    assert changed is True
    assert model.state == ModelStateEnum.PENDING
    assert model.role_status == {
        "prefill": RoleStatus(desired=3, ready=0),
        "decode": RoleStatus(desired=1, ready=0),
        "router": RoleStatus(desired=1, ready=0),
    }
    assert model.ready_replicas == 0
    assert model.degradations is None
    assert is_model_servable(model) is False


@pytest.mark.asyncio
async def test_desired_follows_a_spec_edit_with_no_instance_change():
    """`desired` comes from the spec, so editing `roles[].replicas` has to move
    it. On a stopped group nothing else changes, so nothing else would trigger
    the recompute."""
    model = _model(replicas=0, roles=_pd_roles(prefill=3))
    await _sync(model, [])
    assert model.role_status["prefill"].desired == 3

    model.roles[0].replicas = 5
    changed, _ = await _sync(model, [])

    assert changed is True
    assert model.role_status["prefill"].desired == 5


@pytest.mark.asyncio
async def test_a_second_pass_over_an_unchanged_model_writes_nothing():
    """What makes it safe to call the owner from both the spec side and the
    instance side: an idempotent pass behind a change gate cannot loop."""
    model = _model(replicas=1, roles=_pd_roles(prefill=1, decode=1, router=1))
    instances = [
        _instance(1, role="prefill"),
        _instance(2, role="decode"),
        _instance(3, role="router"),
    ]
    first, first_update = await _sync(model, instances)
    second, second_update = await _sync(model, instances)

    assert first is True
    assert first_update.await_count == 1
    assert second is False
    assert second_update.await_count == 0


def test_the_migration_only_writes_values_the_writer_produces():
    """The seed and the writer have to agree, or the upgrade itself puts a row
    into a state the owner would never choose. `partial`, for instance, is not
    a value `derive_model_state` returns for a role-less model, and the
    servability gate reads it as unroutable — seeding it would take every
    partially-scaled model out of service until its first reconcile."""
    import pathlib
    import re

    import gpustack

    versions = pathlib.Path(gpustack.__file__).parent / "migrations" / "versions"
    path = next(versions.glob("*f4a5b6c7d8e9*database_changes.py"))
    body = path.read_text()
    backfill = body[
        body.index("def _backfill_state") : body.index("def _add_benchmark_target_mode")
    ]
    produced = set(re.findall(r"'(pending|partial|running|error)'", backfill))
    assert produced, "no state literals found in the backfill"

    reachable = set()
    for replicas in range(0, 4):
        for ready in range(0, 4):
            for errors in range(0, 3):
                state, _ = derive_model_state(
                    _model(replicas=replicas),
                    ready_replicas=ready,
                    instance_count=ready + errors,
                    role_status=None,
                    error_count=errors,
                )
                reachable.add(state.value)

    assert produced <= reachable, (
        f"the backfill writes {sorted(produced - reachable)}, which "
        f"derive_model_state never produces for a role-less model"
    )


# --- readiness: any_per_role vs all ----------------------------------------- #


def _readiness_state(readiness, ready, desired=4):
    """A 4P1D group with `ready` prefills up, judged under one policy."""
    from gpustack.schemas.models import DisaggregationSpec, PDModeEnum

    model = _model(replicas=1, roles=_pd_roles(prefill=desired))
    if readiness is not None:
        model.disaggregation = DisaggregationSpec(
            mode=PDModeEnum.VLLM_NIXL, readiness=readiness
        )
    role_status = {
        "prefill": SimpleNamespace(desired=desired, ready=ready),
        "decode": SimpleNamespace(desired=1, ready=1),
        "router": SimpleNamespace(desired=1, ready=1),
    }
    with patch(
        "gpustack.server.controllers.upstream_registration_ready", return_value=True
    ):
        return derive_model_state(
            model,
            ready_replicas=ready + 2,
            instance_count=desired + 2,
            role_status=role_status,
            error_count=0,
        )


def test_the_default_serves_a_group_that_is_short_of_its_ratio():
    """The reason this policy is the default: requiring full staffing would
    make every scale-up an outage for its whole duration."""
    state, _ = _readiness_state("any_per_role", ready=3)
    assert state == ModelStateEnum.RUNNING


def test_readiness_all_withholds_a_group_that_is_short_of_its_ratio():
    """`readiness="all"` has to change the judgement, not merely read back: a
    3/4 prefill group is PARTIAL under it, where `any_per_role` calls the same
    group RUNNING. A setting that reads back but changes nothing is worse than
    an absent one."""
    state, message = _readiness_state("all", ready=3)
    assert state == ModelStateEnum.PARTIAL
    assert "prefill" in message


def test_readiness_all_serves_once_every_member_is_up():
    state, _ = _readiness_state("all", ready=4)
    assert state == ModelStateEnum.RUNNING


def test_a_group_with_no_disaggregation_block_uses_the_default():
    """Role-only orchestration has no `disaggregation` at all, and a row
    written before the field existed deserialises without it. Both must read
    as the forgiving policy rather than raise."""
    state, _ = _readiness_state(None, ready=3)
    assert state == ModelStateEnum.RUNNING


# --- the restart guard's release ------------------------------------------- #


@pytest.mark.asyncio
async def test_reaching_running_releases_the_restart_guard():
    """The endpoint marks `restarting_since` before it tears the generation
    down; this is the only thing that clears it, and RUNNING is the only event
    that actually means the replacements are no longer at risk."""
    model = _model(roles=[RoleSpec(name="prefill", replicas=1)])
    model.restarting_since = datetime.now(timezone.utc)

    changed, _ = await _sync(model, [_instance(1, role="prefill")])

    assert changed is True
    assert model.state == ModelStateEnum.RUNNING
    assert model.restarting_since is None


@pytest.mark.asyncio
async def test_a_group_still_rebuilding_keeps_the_guard():
    """Releasing on anything short of RUNNING would reopen the window the
    guard exists for: the replacements are exactly the members that are not
    running yet."""
    model = _model(roles=[RoleSpec(name="prefill", replicas=1)])
    since = datetime.now(timezone.utc)
    model.restarting_since = since

    await _sync(model, [_instance(1, ModelInstanceStateEnum.STARTING, role="prefill")])

    assert model.state != ModelStateEnum.RUNNING
    assert model.restarting_since == since


# --- the drain window, as a list row sees it ------------------------------- #


@pytest.mark.asyncio
async def test_a_drained_member_is_not_ready_but_is_still_a_member():
    """3P1D scaled to 2P1D: the three counts have to disagree, correctly.

    A drained member is deliberately left RUNNING so the decodes mid-request
    can finish pulling its KV — but the router dropped it when the window
    opened, so it is running without being reachable. Counting it as ready is
    what made the role report `3 / 2` for the length of the window, which reads
    as an edit that did not take.
    """
    model = _model(replicas=1, roles=_pd_roles(prefill=2))
    instances = _members({"prefill": 3, "decode": 1, "router": 1})
    instances[0].draining_since = datetime.now(timezone.utc)

    changed, _ = await _sync(model, instances)

    assert changed is True
    prefill = model.role_status["prefill"]
    assert (prefill.desired, prefill.ready, prefill.draining) == (2, 2, 1)
    # The model-level count is the one that keeps its pre-PD meaning: a plain
    # count of RUNNING instances, draining members and router included.
    assert model.ready_replicas == 5
    # And the group is serving throughout: the window is the whole point.
    assert model.state == ModelStateEnum.RUNNING


@pytest.mark.asyncio
async def test_a_broken_member_can_also_be_draining():
    """Not a corner — the ordinary case, and the one the counts must survive.

    Victim selection scores a member whose worker or state is bad at zero, so
    a broken member is the *first* one a scale-down picks. It is then draining
    and not RUNNING at once — absent from `ready`, present in `draining` —
    which is why `draining` cannot be derived from the other two counts and
    has to be reported in its own right.
    """
    model = _model(replicas=1, roles=_pd_roles(prefill=1, decode=1))
    instances = _members({"decode": 1, "router": 1})
    serving = _instance(10, role="prefill")
    leaving = _instance(11, role="prefill")
    broken = _instance(12, ModelInstanceStateEnum.ERROR, role="prefill")
    leaving.draining_since = datetime.now(timezone.utc)
    broken.draining_since = datetime.now(timezone.utc)

    await _sync(model, instances + [serving, leaving, broken])

    prefill = model.role_status["prefill"]
    assert (prefill.desired, prefill.ready, prefill.draining) == (1, 1, 2)
    # One member is genuinely serving the role, so the group still serves.
    assert model.state == ModelStateEnum.RUNNING


@pytest.mark.asyncio
async def test_a_settled_role_reports_nothing_draining():
    """Every member up and none leaving: the cell prints the fraction alone,
    and the reader is never shown a discrepancy that is not there."""
    model = _model(replicas=1, roles=_pd_roles(prefill=2, decode=1))
    instances = _members({"prefill": 2, "decode": 1, "router": 1})

    await _sync(model, instances)

    prefill = model.role_status["prefill"]
    assert (prefill.ready, prefill.desired) == (2, 2)
    assert all(status.draining == 0 for status in model.role_status.values())


@pytest.mark.asyncio
async def test_the_window_opening_and_closing_both_publish():
    """Both edges ride the change gate.

    Without this the UI would light up when the drain starts and stay lit: the
    reap deletes a row, and if the resulting `role_status` compared equal to
    the stored one nothing would publish the return to normal.
    """
    model = _model(replicas=1, roles=_pd_roles(prefill=2, decode=1))
    instances = _members({"prefill": 3, "decode": 1, "router": 1})
    instances[0].draining_since = datetime.now(timezone.utc)
    await _sync(model, instances)
    opened = model.role_status["prefill"]

    # The window closes the only way it can: the member is gone.
    changed, _ = await _sync(model, instances[1:])

    assert opened.draining == 1
    assert changed is True
    assert model.role_status["prefill"].draining == 0
