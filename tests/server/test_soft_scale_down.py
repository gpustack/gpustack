"""Scaling a prefill down without dropping the KV it still holds.

The engine has no "stop accepting work and exit once your blocks have been
fetched" — waiting for that is upstream WIP. So the wait happens in the
orchestration layer: the member leaves the router's registry at once, keeps
running for a window, and is deleted after it. Removing an address is measured
at 18ms with requests in flight, so the cost of changing a ratio is entirely
that window, not an interruption.
"""

import importlib
import pkgutil
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import sqlalchemy as sa
from sqlmodel import SQLModel

from gpustack.schemas.common import UTCDateTime
from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum, RoleNameEnum
from gpustack.server import pd_membership
from gpustack.server.controllers import (
    _next_drain_due,
    _reap_drained,
    _scale_down_role,
    cancel_drain,
)

WINDOW = 60


def _instance(name, role="prefill", draining_since=None, worker_id=1):
    instance = SimpleNamespace(
        id=name,
        name=name,
        role=role,
        worker_id=worker_id,
        worker_ip="10.0.0.1",
        port=40000,
        state=ModelInstanceStateEnum.RUNNING,
        draining_since=draining_since,
    )
    instance.update = AsyncMock()
    return instance


def _model(roles=("prefill", "decode")):
    return SimpleNamespace(
        id=1,
        name="m",
        roles=[SimpleNamespace(name=name) for name in roles],
    )


def _role(name="prefill", replicas=1):
    return SimpleNamespace(name=name, replicas=replicas)


class TestRegistryRemoval:
    """Step one, and it is the only step that changes what users see."""

    def test_a_draining_member_leaves_the_router(self):
        keep = _instance("p1")
        going = _instance("p2", draining_since=datetime.now(timezone.utc))
        going.port = 40001

        members = pd_membership.desired_members(_model(), [keep, going])

        assert members == {pd_membership.member_url(keep): "prefill"}

    def test_it_is_not_a_state_change(self):
        """The container must keep serving the decodes already pulling from
        it. Only *new* work has to stop arriving, and the address is exactly
        that distinction."""
        going = _instance("p2", draining_since=datetime.now(timezone.utc))
        assert going.state == ModelInstanceStateEnum.RUNNING


class TestPicking:
    async def _scale(
        self, model, role, have, window=WINDOW, candidates=None, generation=None
    ):
        picked = candidates if candidates is not None else have
        with (
            patch("gpustack.envs.SCHEDULER_DRAIN_WINDOW_SECONDS", window),
            patch(
                "gpustack.server.controllers.find_scale_down_candidates",
                AsyncMock(
                    return_value=[
                        SimpleNamespace(model_instance=i, score=0) for i in picked
                    ]
                ),
            ),
            patch(
                "gpustack.server.controllers._release_and_delete", AsyncMock()
            ) as release,
        ):
            await _scale_down_role(
                None, model, role, have, generation if generation is not None else have
            )
            return release

    @pytest.mark.asyncio
    async def test_the_surplus_is_marked_not_deleted(self):
        have = [_instance("p1"), _instance("p2")]
        release = await self._scale(_model(), _role(replicas=1), have)

        release.assert_not_called()
        assert sum(i.draining_since is not None for i in have) == 1

    @pytest.mark.asyncio
    async def test_a_second_pass_does_not_pick_another(self):
        """The debounce, and without it a burst of reconciles scales the
        role to zero one window at a time: the draining member still counts as
        surplus, so every pass picks one more."""
        draining = _instance("p1", draining_since=datetime.now(timezone.utc))
        have = [draining, _instance("p2")]

        await self._scale(_model(), _role(replicas=1), have)

        assert have[1].draining_since is None

    @pytest.mark.asyncio
    async def test_excess_is_measured_against_the_role(self):
        """Not against the model: `len(have) - model.replicas` scoped to one
        role of a 4P4D deletes eight instances in a pass."""
        have = [_instance(f"p{i}") for i in range(4)]
        await self._scale(_model(), _role(replicas=2), have)

        assert sum(i.draining_since is not None for i in have) == 2

    @pytest.mark.asyncio
    async def test_a_role_less_model_still_deletes_immediately(self):
        """No router registry to leave, so a window would only mean the member
        keeps taking new requests and then vanishes mid-request."""
        have = [_instance("i1"), _instance("i2")]
        release = await self._scale(
            SimpleNamespace(id=1, name="m", roles=None), _role(replicas=1), have
        )

        release.assert_called_once()
        assert all(i.draining_since is None for i in have)

    @pytest.mark.asyncio
    async def test_a_zero_window_deletes_immediately(self):
        have = [_instance("p1"), _instance("p2")]
        release = await self._scale(_model(), _role(replicas=1), have, window=0)

        release.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_scoring_failure_deletes_nothing(self):
        """`find_scale_down_candidates` returns [] on its internal exception,
        so empty is ambiguous and not deleting is the fail-safe reading."""
        have = [_instance("p1"), _instance("p2")]
        release = await self._scale(_model(), _role(replicas=1), have, candidates=[])

        release.assert_not_called()
        assert all(i.draining_since is None for i in have)


class TestReaping:
    async def _reap(self, instances, window=WINDOW):
        with (
            patch("gpustack.envs.SCHEDULER_DRAIN_WINDOW_SECONDS", window),
            patch(
                "gpustack.server.controllers._release_and_delete", AsyncMock()
            ) as release,
        ):
            survivors = await _reap_drained(None, instances)
            return survivors, release

    @pytest.mark.asyncio
    async def test_inside_the_window_it_keeps_running(self):
        recent = _instance(
            "p1",
            draining_since=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
        survivors, release = await self._reap([recent])

        release.assert_not_called()
        assert survivors == [recent]

    @pytest.mark.asyncio
    async def test_past_the_window_it_is_deleted(self):
        old = _instance(
            "p1",
            draining_since=datetime.now(timezone.utc) - timedelta(seconds=WINDOW + 1),
        )
        keep = _instance("p2")
        survivors, release = await self._reap([old, keep])

        release.assert_called_once()
        assert survivors == [keep]

    @pytest.mark.asyncio
    async def test_a_window_that_elapsed_while_the_server_was_down(self):
        """Why the timestamp is on the row. Held in memory, a restart
        mid-window leaves a member no router knows about and nothing will ever
        delete — serving nothing, holding its cards."""
        ancient = _instance(
            "p1", draining_since=datetime.now(timezone.utc) - timedelta(days=2)
        )
        _, release = await self._reap([ancient])

        release.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_naive_timestamp_is_read_as_utc(self):
        """SQLite hands the datetime back without a zone; it was written UTC,
        and comparing it against an aware `now` raises otherwise."""
        naive = _instance(
            "p1",
            draining_since=(
                datetime.now(timezone.utc) - timedelta(seconds=WINDOW + 1)
            ).replace(tzinfo=None),
        )
        _, release = await self._reap([naive])

        release.assert_called_once()

    @pytest.mark.asyncio
    async def test_survivors_never_include_a_deleted_member(self):
        """The per-role arithmetic that follows counts what comes back. A
        reaped member left in the list makes the role look satisfied and
        suppresses the replacement a re-scale-up is waiting for."""
        old = _instance(
            "p1",
            draining_since=datetime.now(timezone.utc) - timedelta(seconds=WINDOW + 1),
        )
        survivors, _ = await self._reap([old])

        assert survivors == []


class TestTheMarkCanActuallyBeStored:
    """Everything above mocks the write, and that is how this shipped broken.

    `_scale_down_role` builds `datetime.now(timezone.utc)` — an aware value.
    `draining_since` was declared as a bare `Optional[datetime]`, which maps to
    TIMESTAMP WITHOUT TIME ZONE. SQLite takes that combination without
    complaint, so every test here and every dev install passed; asyncpg refuses
    it outright, so on PostgreSQL the write raised after the victim had already
    been chosen, the reconcile died there, and the surplus member was never
    taken out of rotation — while `role_status` and `degradations` went on
    reporting the group as converged. Measured: 344s, two decodes, one wanted.
    """

    def test_the_column_normalizes_the_zone(self):
        column = ModelInstance.__table__.c.draining_since
        assert isinstance(column.type, UTCDateTime)

        aware = datetime.now(timezone.utc)
        stored = column.type.process_bind_param(aware, None)
        assert stored.tzinfo is None
        assert column.type.process_result_value(stored, None).tzinfo == timezone.utc

    def test_no_timestamp_column_anywhere_is_left_bare(self):
        """The invariant rather than the one column, because the bug is a
        declaration that looks entirely ordinary. A bare `Optional[datetime]`
        is the natural thing to write and is wrong on every table here."""
        import gpustack.schemas as schemas

        for module in pkgutil.iter_modules(schemas.__path__):
            importlib.import_module(f"{schemas.__name__}.{module.name}")

        bare = [
            f"{table.name}.{column.name}"
            for table in SQLModel.metadata.tables.values()
            for column in table.columns
            if isinstance(column.type, (sa.DateTime, sa.TIMESTAMP))
            and not isinstance(column.type, UTCDateTime)
        ]
        assert bare == []


class TestBookingTheReap:
    """`_reap_drained` runs off the reconcile loop, and nothing ticks it.

    A drained member changes no field the status pass publishes, and a settled
    deployment publishes nothing either, so without a booked pass the window
    expires against a reconcile that never arrives — the member keeps its
    accelerators for as long as the deployment goes unedited.
    """

    def test_nothing_draining_books_nothing(self):
        assert _next_drain_due([_instance("p1"), _instance("p2")]) is None

    def test_the_delay_is_what_is_left_of_the_window(self):
        with patch("gpustack.envs.SCHEDULER_DRAIN_WINDOW_SECONDS", WINDOW):
            due = _next_drain_due(
                [
                    _instance(
                        "p1",
                        draining_since=datetime.now(timezone.utc)
                        - timedelta(seconds=20),
                    )
                ]
            )
        assert 35 <= due <= 40

    def test_the_earliest_deadline_wins(self):
        now = datetime.now(timezone.utc)
        with patch("gpustack.envs.SCHEDULER_DRAIN_WINDOW_SECONDS", WINDOW):
            due = _next_drain_due(
                [
                    _instance("p1", draining_since=now - timedelta(seconds=10)),
                    _instance("p2", draining_since=now - timedelta(seconds=50)),
                ]
            )
        assert 5 <= due <= 15

    def test_an_elapsed_window_books_an_immediate_pass(self):
        """Not a negative delay: `add_after` treats <= 0 as "enqueue now",
        which is right, but only by accident — say it here so a window that
        elapsed while the server was down cannot turn into a skipped reap."""
        with patch("gpustack.envs.SCHEDULER_DRAIN_WINDOW_SECONDS", WINDOW):
            due = _next_drain_due(
                [
                    _instance(
                        "p1",
                        draining_since=datetime.now(timezone.utc) - timedelta(days=2),
                    )
                ]
            )
        assert due == 0


@pytest.mark.asyncio
async def test_cancelling_a_drain_puts_it_back():
    """The rollback is one field: the next membership reconcile sees an
    ordinary RUNNING member and re-registers it. Nothing restarts, because
    nothing was stopped."""
    instance = _instance("p1", draining_since=datetime.now(timezone.utc))

    await cancel_drain(None, instance)

    assert instance.draining_since is None
    assert pd_membership.desired_members(_model(), [instance]) == {
        pd_membership.member_url(instance): RoleNameEnum.PREFILL.value
    }
