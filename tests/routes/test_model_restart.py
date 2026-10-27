"""The group restart endpoint.

`POST /v2/models/{id}/restart` exists because "restart" had no backend at all —
the whole API had zero restart routes, and restarting in practice meant
deleting an instance and letting replica convergence rebuild it. For a group
that sequence is a correctness bug rather than an inconvenience: between the
two deletes there is a new-generation prefill paired with an old-generation
decode, and the engines accept that pairing. A `max_model_len` mismatch
handshakes, transfers, and only surfaces on a long prompt — after prefill has
already been paid for.

So the properties pinned here are the ones that make that window impossible:
the whole generation goes down together, there is no way to ask for part of
it, and a restart already in flight is refused rather than restarted again.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack import envs
from gpustack.api.exceptions import ConflictException, InternalServerErrorException
from gpustack.routes.models import _restart_in_flight, restart_model
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    RoleSpec,
    SourceEnum,
)

TARGET = "sha1:current"


def _model(roles=None) -> Model:
    return Model(
        id=1,
        name="m",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        cluster_id=1,
        roles=roles,
    )


def _pd_roles():
    return [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="decode", replicas=1),
        RoleSpec(name="router", replicas=1, cpu_only=True),
    ]


def _instance(id, role=None, spec_digest=TARGET) -> ModelInstance:
    return ModelInstance(
        id=id,
        name=f"m-{id}",
        model_id=1,
        model_name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        state=ModelInstanceStateEnum.RUNNING,
        role=role,
        spec_digest=spec_digest,
    )


async def _restart(model, instances):
    deleted = []

    async def _batch_delete(rows):
        deleted.extend(rows)
        return [r.name for r in rows]

    async def _update(row, source=None, **kwargs):
        # The endpoint marks the guard through the service. Applied to the row
        # rather than recorded, so a test can restart the same model twice and
        # see the second call meet the mark the first one left.
        for key, value in (source or {}).items():
            setattr(row, key, value)

    service = MagicMock(return_value=SimpleNamespace(batch_delete=_batch_delete))
    model_service = MagicMock(return_value=SimpleNamespace(update=_update))
    with (
        patch("gpustack.routes.models.Model.one_by_id", AsyncMock(return_value=model)),
        patch("gpustack.routes.models.assert_resource_visible", MagicMock()),
        patch(
            "gpustack.routes.models.model_spec_digest",
            AsyncMock(return_value=TARGET),
        ),
        patch(
            "gpustack.routes.models.ModelInstance.all_by_fields",
            AsyncMock(return_value=instances),
        ),
        patch("gpustack.routes.models.ModelInstanceService", service),
        patch("gpustack.routes.models.ModelService", model_service),
    ):
        result = await restart_model(MagicMock(), MagicMock(), 1)
    return result, deleted


@pytest.mark.asyncio
async def test_a_stale_group_goes_down_whole():
    """All of it, in one batch. Deleting members one at a time is exactly how
    the cross-generation window is produced."""
    members = [
        _instance(1, role="prefill", spec_digest="sha1:old"),
        _instance(2, role="decode", spec_digest="sha1:old"),
        _instance(3, role="router", spec_digest="sha1:old"),
    ]
    result, deleted = await _restart(_model(roles=_pd_roles()), members)

    assert result.restarted is True
    assert result.spec_digest == TARGET
    assert len(deleted) == 3
    assert {i.role for i in deleted} == {"prefill", "decode", "router"}


@pytest.mark.asyncio
async def test_rebuilding_is_left_to_replica_convergence():
    """Convergence is the one place that knows a group forms its GPU roles
    atomically and holds the router back until they run. Recreating here would
    be a second implementation of the rule that matters most."""
    members = [_instance(1, role="prefill", spec_digest="sha1:old")]
    result, _ = await _restart(_model(roles=_pd_roles()), members)

    assert result.deleted_instances == ["m-1"]
    assert "re-form" in result.message


@pytest.mark.asyncio
async def test_a_group_already_on_the_current_spec_is_still_rebuilt():
    """This asserted the opposite until the endpoint stopped short-circuiting
    on a converged group.

    Two reasons it flipped. The word on the menu is "restart", and the state an
    operator reaches for it in — a process wedged behind a socket while the
    control plane still calls it RUNNING — is exactly the one a digest
    comparison cannot see. And only a group's members carry a `spec_digest` at
    all, so the short-circuit made the same button rebuild a role-less
    deployment while doing nothing to a PD group on its current spec."""
    members = [
        _instance(1, role="prefill"),
        _instance(2, role="decode"),
        _instance(3, role="router"),
    ]
    result, deleted = await _restart(_model(roles=_pd_roles()), members)

    assert result.restarted is True
    assert sorted(result.deleted_instances) == ["m-1", "m-2", "m-3"]


# --- the guard against a second teardown ----------------------------------- #


@pytest.mark.asyncio
async def test_a_restart_records_that_one_is_in_flight():
    """The fact has to be written down, because it cannot be read back off the
    rows: the teardown is synchronous and the reconcile rebuilds from the same
    target digest, so there is no moment when the table shows two
    generations."""
    model = _model(roles=_pd_roles())
    assert model.restarting_since is None

    await _restart(model, [_instance(1, role="prefill")])

    assert model.restarting_since is not None


@pytest.mark.asyncio
async def test_a_second_click_while_the_replacements_are_starting_is_refused():
    """The regression this guard exists for, measured on a live group: a
    second restart 4.5s after the first returned 200 and deleted the three
    replacements the first one had just created, costing the group another full
    startup with nothing in the UI to say why it was back at pending.

    The replacements are ordinary members carrying the target digest, which is
    why the digest comparison this replaces could never see them."""
    model = _model(roles=_pd_roles())
    await _restart(model, [_instance(1, role="prefill"), _instance(2, role="decode")])

    replacements = [_instance(3, role="prefill"), _instance(4, role="decode")]
    replacements[0].state = ModelInstanceStateEnum.STARTING
    with pytest.raises(ConflictException):
        await _restart(model, replacements)


@pytest.mark.asyncio
async def test_a_stale_guard_lapses_rather_than_wedging_the_button():
    """A group that never converges is exactly the one an operator needs to
    restart again. A guard with no expiry would answer 409 to them forever."""
    model = _model(roles=_pd_roles())
    model.restarting_since = datetime.now(timezone.utc) - timedelta(
        seconds=envs.RESTART_IN_FLIGHT_LAPSE_SECONDS + 1
    )

    result, deleted = await _restart(model, [_instance(1, role="prefill")])

    assert result.restarted is True
    assert len(deleted) == 1


@pytest.mark.asyncio
async def test_mixed_digests_alone_do_not_refuse_anything():
    """The old guard's whole condition, now inert. Kept as a test because the
    shape is tempting to reintroduce: it reads like "mid-replacement" and is
    not, and a group that somehow does hold two generations is the one most in
    need of the restart that would collapse them into one."""
    members = [
        _instance(1, role="prefill", spec_digest=TARGET),
        _instance(2, role="decode", spec_digest="sha1:old"),
    ]
    result, deleted = await _restart(_model(roles=_pd_roles()), members)

    assert result.restarted is True
    assert len(deleted) == 2


def test_the_guard_reads_naive_and_aware_timestamps_alike():
    """`UTCDateTime` hands back an aware value; a row that reached memory
    without passing through the column is naive. Subtracting the wrong one
    raises, and a TypeError here is a 500 on the restart button."""
    model = _model()
    for since in (
        datetime.now(timezone.utc),
        datetime.now(timezone.utc).replace(tzinfo=None),
    ):
        model.restarting_since = since
        assert _restart_in_flight(model) is True


@pytest.mark.asyncio
async def test_a_model_with_no_instances_reports_nothing_to_do():
    result, deleted = await _restart(_model(roles=_pd_roles()), [])

    assert result.restarted is False
    assert deleted == []
    assert result.spec_digest == TARGET


@pytest.mark.asyncio
async def test_a_role_less_model_restarts_the_same_way():
    """The mechanism is identical, so refusing here would be an arbitrary
    restriction — and the same "edited the config, nothing happened" problem
    exists for a single-instance model."""
    members = [_instance(1, spec_digest="sha1:old")]
    result, deleted = await _restart(_model(), members)

    assert result.restarted is True
    assert len(deleted) == 1


def test_the_endpoint_takes_no_role_parameter():
    """ "Restart only the decodes" is precisely the request that produces the
    cross-generation window, and the strongest rejection is to have no way to
    express it."""
    import inspect

    params = set(inspect.signature(restart_model).parameters)
    assert params == {"session", "ctx", "id"}


# --- a member that is not running anything --------------------------------- #


@pytest.mark.asyncio
async def test_a_group_with_a_failed_member_is_restarted():
    """Reporting "already run the current configuration" to someone whose
    group is half down is not merely unhelpful, it is untrue: a member in
    ERROR is not running that configuration, it is not running anything. And
    it left the one operation they reached for with nothing to do."""
    model = _model()
    instances = [
        _instance(1, spec_digest=TARGET),
        _instance(2, spec_digest=TARGET),
    ]
    instances[1].state = ModelInstanceStateEnum.ERROR

    result, deleted = await _restart(model, instances)

    assert result.restarted is True
    assert len(deleted) == 2
    # Named, because the reason it failed is usually still there and a bare
    # "restarted" invites an immediate retry of the same failure.
    assert instances[1].name in result.message
    assert "check its log" in result.message


@pytest.mark.asyncio
async def test_a_healthy_group_is_rebuilt_without_the_failure_wording():
    """A healthy group restarts too, but must not be handed the sentence
    written for a group with a dead member — that one tells the reader to go
    read a log before retrying, and there is no log to read."""
    model = _model()
    instances = [_instance(1, spec_digest=TARGET)]

    result, deleted = await _restart(model, instances)

    assert result.restarted is True
    assert len(deleted) == 1
    assert "check its log" not in result.message


@pytest.mark.asyncio
async def test_a_teardown_that_failed_leaves_no_guard_behind():
    """Nothing was torn down, so there are no replacements to protect. The
    failure the operator now has to retry is the worst possible moment to
    answer their retry with a 409."""
    model = _model(roles=_pd_roles())

    async def _explode(_rows):
        raise RuntimeError("boom")

    service = MagicMock(return_value=SimpleNamespace(batch_delete=_explode))

    async def _update(row, source=None, **kwargs):
        for key, value in (source or {}).items():
            setattr(row, key, value)

    model_service = MagicMock(return_value=SimpleNamespace(update=_update))
    with (
        patch("gpustack.routes.models.Model.one_by_id", AsyncMock(return_value=model)),
        patch("gpustack.routes.models.assert_resource_visible", MagicMock()),
        patch(
            "gpustack.routes.models.model_spec_digest", AsyncMock(return_value=TARGET)
        ),
        patch(
            "gpustack.routes.models.ModelInstance.all_by_fields",
            AsyncMock(return_value=[_instance(1, role="prefill")]),
        ),
        patch("gpustack.routes.models.ModelInstanceService", service),
        patch("gpustack.routes.models.ModelService", model_service),
    ):
        with pytest.raises(InternalServerErrorException):
            await restart_model(MagicMock(), MagicMock(), 1)

    assert model.restarting_since is None
