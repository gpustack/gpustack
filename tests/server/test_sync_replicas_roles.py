"""Role-aware replica convergence.

``sync_replicas`` carries two rules behind one name. The role-less one is
unchanged and pinned here so it stays that way; the role-bearing one is new,
and the properties worth pinning are the ones whose absence produced the
active bug this replaces:

* a role-bearing model fans out to ``sum(roles[].replicas)`` members, not to
  ``Model.replicas`` — which for such a model is a 0/1 deployment switch;
* every member carries its ``role``, ``group_id`` and ``spec_digest``, because
  nothing downstream can tell a prefill from a decode without them;
* the router is NOT part of the atomic first formation and waits on a
  dependency gate, since its command line is rendered from peer addresses that
  do not exist until those peers are running;
* surplus is measured against the ROLE's count. The pre-PD expression
  ``len(candidates) - model.replicas`` deletes eight members of a 4P4D in one
  pass once it is scoped to a single role.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelStateEnum,
    RoleSpec,
    SourceEnum,
)
from gpustack.server.controllers import (
    _dependencies_ready,
    _gpu_roles,
    _role_dependencies,
    find_scale_down_candidates,
    model_spec_digest,
    sync_replicas,
)


def _model(replicas=1, roles=None, **kwargs) -> Model:
    return Model(
        id=1,
        name="m",
        replicas=replicas,
        ready_replicas=0,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        cluster_id=1,
        roles=roles,
        **kwargs,
    )


def _pd_roles(prefill=1, decode=1, router=1):
    return [
        RoleSpec(name="prefill", replicas=prefill),
        RoleSpec(name="decode", replicas=decode),
        RoleSpec(name="router", replicas=router),
    ]


def _instance(
    id,
    role=None,
    group_id=None,
    spec_digest=None,
    state=ModelInstanceStateEnum.RUNNING,
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
        group_id=group_id,
        spec_digest=spec_digest,
        created_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
    )


class _Recorder:
    """Stands in for ``ModelInstanceService``, recording what convergence did."""

    def __init__(self):
        self.created = []
        self.deleted = []

    def __call__(self, session):
        return SimpleNamespace(
            batch_create=self._batch_create,
            create=self._create,
            batch_delete=self._batch_delete,
        )

    async def _batch_create(self, instances):
        self.created.extend(instances)
        return instances

    async def _create(self, instance):
        self.created.append(instance)
        return instance

    async def _batch_delete(self, instances):
        self.deleted.extend(instances)
        return [i.name for i in instances]


async def _run(model, instances, instance_type_snapshot=None):
    recorder = _Recorder()
    matched = (
        [SimpleNamespace(snapshot=instance_type_snapshot)]
        if instance_type_snapshot
        else []
    )
    # The second read inside convergence re-lists the model's instances to pick
    # up the rows the atomic formation just wrote; returning the created rows
    # keeps that read honest without a database.
    reads = {"n": 0}

    async def _all_by_field(session, field, value, **kwargs):
        reads["n"] += 1
        if reads["n"] == 1:
            return instances
        return instances + [
            _instance(
                100 + i,
                role=c.role,
                group_id=c.group_id,
                spec_digest=c.spec_digest,
                state=ModelInstanceStateEnum.PENDING,
            )
            for i, c in enumerate(recorder.created)
        ]

    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(side_effect=_all_by_field),
        ),
        patch(
            "gpustack.server.controllers.GPUInstanceType.all_by_fields",
            AsyncMock(return_value=matched),
        ),
        patch(
            "gpustack.server.controllers.get_draft_model_source",
            AsyncMock(return_value=None),
        ),
        patch("gpustack.server.controllers.ModelInstanceService", recorder),
        # Soft scale-down marks the surplus instead of deleting it, and the
        # mark is a row write. Recorded rather than performed, for the same
        # reason the deletes are: this harness has no database.
        patch.object(ModelInstance, "update", AsyncMock()),
    ):
        await sync_replicas(MagicMock(), model)
    return recorder


def _by_role(created):
    counts = {}
    for c in created:
        counts[c.role] = counts.get(c.role, 0) + 1
    return counts


# --- the role-less baseline ------------------------------------------------ #


@pytest.mark.asyncio
async def test_role_less_still_fans_out_by_model_replicas():
    recorder = await _run(_model(replicas=3), [])

    assert len(recorder.created) == 3
    # Nothing role-shaped is invented for a plain deployment.
    assert all(c.role is None for c in recorder.created)
    assert all(c.group_id is None for c in recorder.created)


@pytest.mark.asyncio
async def test_role_less_scale_down_unchanged():
    model = _model(replicas=1)
    instances = [_instance(1), _instance(2)]
    with patch(
        "gpustack.server.controllers.find_scale_down_candidates",
        AsyncMock(
            return_value=[
                SimpleNamespace(model_instance=instances[0]),
                SimpleNamespace(model_instance=instances[1]),
            ]
        ),
    ):
        recorder = await _run(model, instances)

    assert len(recorder.deleted) == 1


# --- fan-out --------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_group_fans_out_to_the_sum_of_its_gpu_roles():
    """The bug this replaces: 1P1D+router against `replicas == 1` produced one
    anonymous instance. The GPU roles form together; the router does not."""
    recorder = await _run(_model(replicas=1, roles=_pd_roles()), [])

    assert _by_role(recorder.created) == {"prefill": 1, "decode": 1}
    assert not recorder.deleted


@pytest.mark.asyncio
async def test_xpyd_fans_out_by_each_roles_own_count():
    recorder = await _run(_model(replicas=1, roles=_pd_roles(prefill=3, decode=2)), [])

    assert _by_role(recorder.created) == {"prefill": 3, "decode": 2}


@pytest.mark.asyncio
async def test_every_member_carries_role_group_and_digest():
    recorder = await _run(_model(replicas=1, roles=_pd_roles()), [])

    group_ids = {c.group_id for c in recorder.created}
    digests = {c.spec_digest for c in recorder.created}
    assert len(group_ids) == 1, "one generation is one group_id"
    assert len(digests) == 1, "one generation is one spec_digest"
    assert group_ids.pop().startswith("1-")
    assert digests.pop().startswith("sha1:")
    assert all(c.role for c in recorder.created)


@pytest.mark.asyncio
async def test_member_names_say_which_role_they_are():
    recorder = await _run(_model(replicas=1, roles=_pd_roles()), [])

    assert any(c.name.startswith("m-prefill-") for c in recorder.created)
    assert any(c.name.startswith("m-decode-") for c in recorder.created)


@pytest.mark.asyncio
async def test_replicas_zero_parks_the_whole_group():
    """`Model.replicas` is a deployment switch for a role-bearing model, so
    zero is "the group is off", not "zero of each role"."""
    members = [
        _instance(1, role="prefill", group_id="1-abc"),
        _instance(2, role="decode", group_id="1-abc"),
    ]
    recorder = await _run(_model(replicas=0, roles=_pd_roles()), members)

    assert len(recorder.deleted) == 2
    assert not recorder.created


# --- the router's dependency gate ------------------------------------------ #


def test_a_router_depends_on_every_gpu_role_by_default():
    model = _model(roles=_pd_roles())
    router = model.roles[-1]

    assert _role_dependencies(model, router) == ["prefill", "decode"]
    assert [r.name for r in _gpu_roles(model)] == ["prefill", "decode"]


def test_an_explicit_dependency_list_wins():
    model = _model(
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(name="decode", replicas=1),
            RoleSpec(name="router", replicas=1, dependencies=["prefill"]),
        ]
    )
    assert _role_dependencies(model, model.roles[-1]) == ["prefill"]


def test_dependencies_need_running_not_merely_created():
    model = _model(roles=_pd_roles())
    router = model.roles[-1]
    pending = [
        _instance(1, role="prefill", state=ModelInstanceStateEnum.PENDING),
        _instance(2, role="decode", state=ModelInstanceStateEnum.PENDING),
    ]
    running = [
        _instance(1, role="prefill"),
        _instance(2, role="decode"),
    ]

    assert _dependencies_ready(model, router, pending) is False
    assert _dependencies_ready(model, router, running) is True


@pytest.mark.asyncio
async def test_the_router_is_not_created_before_its_peers_run():
    """Creating it early does not merely start it sooner — it renders a router
    whose peer flags have no addresses to carry."""
    members = [
        _instance(
            1,
            role="prefill",
            group_id="1-abc",
            spec_digest="sha1:abc",
            state=ModelInstanceStateEnum.PENDING,
        ),
        _instance(
            2,
            role="decode",
            group_id="1-abc",
            spec_digest="sha1:abc",
            state=ModelInstanceStateEnum.PENDING,
        ),
    ]
    recorder = await _run(_model(replicas=1, roles=_pd_roles()), members)

    assert not recorder.created


@pytest.mark.asyncio
async def test_the_router_appears_once_its_peers_are_running():
    members = [
        _instance(1, role="prefill", group_id="1-abc", spec_digest="sha1:abc"),
        _instance(2, role="decode", group_id="1-abc", spec_digest="sha1:abc"),
    ]
    recorder = await _run(_model(replicas=1, roles=_pd_roles()), members)

    assert _by_role(recorder.created) == {"router": 1}
    # It joins the generation its peers are already in, rather than starting
    # one of its own.
    assert recorder.created[0].group_id == "1-abc"
    assert recorder.created[0].spec_digest == "sha1:abc"


# --- per-role convergence -------------------------------------------------- #


@pytest.mark.asyncio
async def test_scaling_one_role_up_touches_only_that_role():
    """Group-granular convergence would express this as delete-group plus
    create-group — a full outage to add one prefill."""
    members = [
        _instance(1, role="prefill", group_id="1-abc", spec_digest="sha1:abc"),
        _instance(2, role="decode", group_id="1-abc", spec_digest="sha1:abc"),
        _instance(3, role="router", group_id="1-abc", spec_digest="sha1:abc"),
    ]
    recorder = await _run(_model(replicas=1, roles=_pd_roles(prefill=3)), members)

    assert _by_role(recorder.created) == {"prefill": 2}
    assert not recorder.deleted


@pytest.mark.asyncio
async def test_surplus_is_measured_against_the_roles_count():
    """The pre-PD expression was `len(candidates) - model.replicas`. Scoped to
    one role of a group where `model.replicas == 1`, it takes out every member
    the scorer returns.

    A group's surplus is *drained* rather than deleted — the arithmetic is
    what this asserts, so it reads the mark instead of the delete.
    """
    prefills = [
        _instance(i, role="prefill", group_id="1-abc", spec_digest="sha1:abc")
        for i in (1, 2, 3, 4)
    ]
    members = prefills + [
        _instance(5, role="decode", group_id="1-abc", spec_digest="sha1:abc"),
        _instance(6, role="router", group_id="1-abc", spec_digest="sha1:abc"),
    ]
    with patch(
        "gpustack.server.controllers.find_scale_down_candidates",
        AsyncMock(return_value=[SimpleNamespace(model_instance=p) for p in prefills]),
    ):
        await _run(_model(replicas=1, roles=_pd_roles(prefill=2)), members)

    drained = [i for i in members if i.draining_since is not None]
    assert len(drained) == 2, "4 prefills down to 2, not down to 0"
    assert {i.role for i in drained} == {"prefill"}


@pytest.mark.asyncio
async def test_a_scoring_failure_deletes_nothing():
    """`find_scale_down_candidates` returns [] on its internal exception, so an
    empty result cannot be told apart from "nothing to remove"."""
    prefills = [
        _instance(i, role="prefill", group_id="1-abc", spec_digest="sha1:abc")
        for i in (1, 2, 3)
    ]
    with patch(
        "gpustack.server.controllers.find_scale_down_candidates",
        AsyncMock(return_value=[]),
    ):
        recorder = await _run(_model(replicas=1, roles=_pd_roles(prefill=1)), prefills)

    assert not recorder.deleted


@pytest.mark.asyncio
async def test_mixed_roles_are_refused_by_the_scale_down_selector():
    """Its PlacementScorer aggregates per worker by `model_id`, so an
    eight-card prefill and a cpu_only router share one distribution and their
    scores are not comparable. Ranking them together picks a plausible wrong
    victim rather than failing."""
    with pytest.raises(ValueError, match="single role"):
        await find_scale_down_candidates(
            [_instance(1, role="prefill"), _instance(2, role="router")],
            _model(roles=_pd_roles()),
        )


# --- generation identity --------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_scale_does_not_change_the_digest():
    """Folding replica counts into the digest would make every scale a
    generation change, and a generation change is a full-group restart."""
    session = MagicMock()
    with patch(
        "gpustack.server.controllers.GPUInstanceType.all_by_fields",
        AsyncMock(return_value=[]),
    ):
        one = await model_spec_digest(session, _model(roles=_pd_roles(prefill=1)))
        three = await model_spec_digest(session, _model(roles=_pd_roles(prefill=3)))
        scaled = await model_spec_digest(session, _model(replicas=1, roles=_pd_roles()))
        parked = await model_spec_digest(session, _model(replicas=0, roles=_pd_roles()))

    assert one == three
    assert scaled == parked


@pytest.mark.asyncio
async def test_a_description_change_does_not_change_the_digest():
    session = MagicMock()
    with patch(
        "gpustack.server.controllers.GPUInstanceType.all_by_fields",
        AsyncMock(return_value=[]),
    ):
        plain = await model_spec_digest(session, _model(roles=_pd_roles()))
        described = await model_spec_digest(
            session, _model(roles=_pd_roles(), description="notes")
        )

    assert plain == described


@pytest.mark.asyncio
async def test_a_deployment_shaping_change_does_change_the_digest():
    session = MagicMock()
    with patch(
        "gpustack.server.controllers.GPUInstanceType.all_by_fields",
        AsyncMock(return_value=[]),
    ):
        plain = await model_spec_digest(session, _model(roles=_pd_roles()))
        env = await model_spec_digest(
            session, _model(roles=_pd_roles(), env={"A": "1"})
        )
        overridden = await model_spec_digest(
            session,
            _model(
                roles=[
                    RoleSpec(name="prefill", replicas=1, backend_parameters=["--x"]),
                    RoleSpec(name="decode", replicas=1),
                    RoleSpec(name="router", replicas=1),
                ]
            ),
        )

    assert plain != env
    assert plain != overridden


@pytest.mark.asyncio
async def test_the_instance_type_snapshot_is_part_of_the_digest():
    """D13: a `gpu_type_selector` records only the type's NAME, while the
    catalog behind that name is versioned by retire-and-insert. Without this,
    two members admitted at different moments can resolve one name to
    different card specs and the group looks like one generation."""
    model = _model(
        roles=[
            RoleSpec(
                name="prefill",
                replicas=1,
                gpu_type_selector={"type": "a100-slice"},
            ),
            RoleSpec(name="decode", replicas=1),
        ]
    )
    session = MagicMock()

    async def _snapshot(value):
        with patch(
            "gpustack.server.controllers.GPUInstanceType.all_by_fields",
            AsyncMock(return_value=[SimpleNamespace(snapshot=value)] if value else []),
        ):
            return await model_spec_digest(session, model)

    assert await _snapshot("sha1:gen1") != await _snapshot("sha1:gen2")
    # A type that vanished is itself a digest input, not a dropped one.
    assert await _snapshot(None) != await _snapshot("sha1:gen1")


@pytest.mark.asyncio
async def test_new_members_join_the_generation_their_peers_are_in():
    """A spec edit makes the running members stale; it does not retire them.
    Adding a member with the model's CURRENT digest would put two generations
    in one group — the exact pairing `group_id` and `spec_digest` exist to
    make structurally impossible."""
    members = [
        _instance(1, role="prefill", group_id="1-old", spec_digest="sha1:old"),
        _instance(2, role="decode", group_id="1-old", spec_digest="sha1:old"),
        _instance(3, role="router", group_id="1-old", spec_digest="sha1:old"),
    ]
    recorder = await _run(
        _model(replicas=1, roles=_pd_roles(prefill=2), env={"CHANGED": "1"}),
        members,
    )

    assert len(recorder.created) == 1
    assert recorder.created[0].group_id == "1-old"
    assert recorder.created[0].spec_digest == "sha1:old"


# --- staleness ------------------------------------------------------------- #


async def _sync_status(model, instances, snapshot=None):
    from gpustack.server.controllers import sync_model_status

    update = AsyncMock()
    service = MagicMock(return_value=SimpleNamespace(update=update))
    with (
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=instances),
        ),
        patch(
            "gpustack.server.controllers.GPUInstanceType.all_by_fields",
            AsyncMock(
                return_value=[SimpleNamespace(snapshot=snapshot)] if snapshot else []
            ),
        ),
        patch("gpustack.server.controllers.ModelService", service),
    ):
        await sync_model_status(MagicMock(), model)
    return model


@pytest.mark.asyncio
async def test_members_matching_the_current_spec_are_not_stale():
    model = _model(replicas=1, roles=_pd_roles())
    with patch(
        "gpustack.server.controllers.GPUInstanceType.all_by_fields",
        AsyncMock(return_value=[]),
    ):
        digest = await model_spec_digest(MagicMock(), model)

    await _sync_status(
        model,
        [
            _instance(1, role="prefill", group_id="g", spec_digest=digest),
            _instance(2, role="decode", group_id="g", spec_digest=digest),
            _instance(3, role="router", group_id="g", spec_digest=digest),
        ],
    )
    assert model.stale is False


@pytest.mark.asyncio
async def test_members_predating_a_spec_edit_are_stale_but_still_serving():
    """Orthogonal to `state`: this is exactly the case worth surfacing, since
    a user who edits a config and sees RUNNING has no other signal that the
    edit has not taken effect."""
    model = _model(replicas=1, roles=_pd_roles())
    await _sync_status(
        model,
        [
            _instance(1, role="prefill", group_id="g", spec_digest="sha1:old"),
            _instance(2, role="decode", group_id="g", spec_digest="sha1:old"),
            _instance(3, role="router", group_id="g", spec_digest="sha1:old"),
        ],
    )
    assert model.stale is True
    assert model.state == ModelStateEnum.RUNNING


@pytest.mark.asyncio
async def test_rows_predating_the_column_are_not_marked_stale():
    """A pre-upgrade instance carries no digest. Reading that as "differs"
    would mark every existing model stale on the first pass after an
    upgrade."""
    model = _model(replicas=1)
    await _sync_status(model, [_instance(1), _instance(2)])

    assert model.stale is None


# --- gateway registration -------------------------------------------------- #


def test_a_role_less_model_registers_every_instance():
    from gpustack.server.controllers import _gateway_registrable_instances

    instances = [_instance(1), _instance(2)]
    assert _gateway_registrable_instances(_model(), instances) == instances


def test_a_group_registers_only_its_router():
    """Every member serves an OpenAI-shaped API on its own port, so registering
    them all yields upstreams that answer the same requests wrongly: a request
    balanced onto a prefill returns after one token, one onto a decode runs
    without the prefix its KV was meant to carry. Both return 200."""
    from gpustack.server.controllers import _gateway_registrable_instances

    router = _instance(3, role="router")
    registrable = _gateway_registrable_instances(
        _model(roles=_pd_roles()),
        [_instance(1, role="prefill"), _instance(2, role="decode"), router],
    )
    assert registrable == [router]


def test_a_routerless_group_registers_nothing():
    """Falling back to the GPU members would be the same wrong answer by
    another route: no member of a group can serve a whole request alone."""
    from gpustack.server.controllers import _gateway_registrable_instances

    model = _model(
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(name="decode", replicas=1),
        ]
    )
    assert (
        _gateway_registrable_instances(
            model, [_instance(1, role="prefill"), _instance(2, role="decode")]
        )
        == []
    )


# --- turning disaggregation off -------------------------------------------- #


@pytest.mark.asyncio
async def test_removing_roles_retires_the_group_first():
    """The role-less rule ranks every instance in one comparison, which an
    eight-card prefill and a cpu_only router cannot share — and the gateway
    filter keys on `model.roles`, so the moment roles are gone every leftover
    member becomes a registered upstream answering whole requests it cannot
    serve."""
    members = [
        _instance(1, role="prefill", group_id="1-abc", spec_digest="sha1:abc"),
        _instance(2, role="decode", group_id="1-abc", spec_digest="sha1:abc"),
        _instance(3, role="router", group_id="1-abc", spec_digest="sha1:abc"),
    ]
    recorder = await _run(_model(replicas=1, roles=None), members)

    assert len(recorder.deleted) == 3
    # Nothing is built in the same pass: the deletion has to be settled before
    # anything counts what is left.
    assert not recorder.created


@pytest.mark.asyncio
async def test_plain_replicas_are_built_on_the_next_pass():
    recorder = await _run(_model(replicas=2, roles=None), [])

    assert len(recorder.created) == 2
    assert all(c.role is None for c in recorder.created)


@pytest.mark.asyncio
async def test_a_mix_of_leftover_and_plain_instances_only_retires_the_leftovers():
    members = [
        _instance(1, role="prefill", group_id="1-abc", spec_digest="sha1:abc"),
        _instance(2),
    ]
    recorder = await _run(_model(replicas=1, roles=None), members)

    assert [i.id for i in recorder.deleted] == [1]


# --- turning disaggregation on --------------------------------------------- #


@pytest.mark.asyncio
async def test_enabling_roles_retires_the_instances_that_predate_the_group():
    """Observed live on a two-card host: the pre-PD instance carried no role,
    so nothing counted it toward any role and the gateway had already stopped
    routing to it — but it still held a GPU, and holding it is what kept the
    new group's decode from being schedulable."""
    legacy = [_instance(1), _instance(2)]
    recorder = await _run(_model(replicas=1, roles=_pd_roles()), legacy)

    assert len(recorder.deleted) == 2
    assert all(i.group_id is None for i in recorder.deleted)


@pytest.mark.asyncio
async def test_the_group_still_forms_in_the_same_pass():
    """Retiring is not a reason to defer forming: the new generation is what
    the retirement is making room for."""
    recorder = await _run(_model(replicas=1, roles=_pd_roles()), [_instance(1)])

    assert len(recorder.deleted) == 1
    assert _by_role(recorder.created) == {"prefill": 1, "decode": 1}


@pytest.mark.asyncio
async def test_an_established_group_keeps_its_members():
    """Only role-less instances are orphans. A member of the current
    generation must not be swept up by the same rule."""
    members = [
        _instance(1, role="prefill", group_id="1-abc", spec_digest="sha1:abc"),
        _instance(2, role="decode", group_id="1-abc", spec_digest="sha1:abc"),
        _instance(3, role="router", group_id="1-abc", spec_digest="sha1:abc"),
    ]
    recorder = await _run(_model(replicas=1, roles=_pd_roles()), members)

    assert not recorder.deleted
    assert not recorder.created
