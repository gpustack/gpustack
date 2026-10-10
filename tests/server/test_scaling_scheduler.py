from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy as sa
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.dialects import mysql, postgresql
from sqlmodel import col, select, text
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelCreate,
    ScalingSchedule,
    ScalingScheduleRule,
    SourceEnum,
)
from gpustack.routes.models import (
    _max_intended_replicas,
    apply_scaling_schedule_baseline,
    validate_model_in,
)
from gpustack.server.scaling_scheduler import compute_desired_replicas
from gpustack.server import scaling_scheduler

TZ = "Asia/Shanghai"
HOUR = 3600


@pytest.fixture(autouse=True)
def _pin_rollup_tz(monkeypatch):
    # compute_desired_replicas resolves the business timezone via
    # resolve_rollup_tz() (envs.TIMEZONE -> OS tz -> UTC). Pin it to TZ so the
    # cron-window assertions don't depend on the host's timezone.
    monkeypatch.setattr("gpustack.envs.TIMEZONE", TZ)


@pytest.fixture
def cache_invalidations(monkeypatch):
    all_models = AsyncMock()
    service = AsyncMock()
    monkeypatch.setattr(Model, "_invalidate_cached_all", all_models)
    monkeypatch.setattr(scaling_scheduler, "delete_cache_by_key", service)
    return all_models, service


def _at(hour: int, minute: int, day: int = 16) -> datetime:
    # 2026-07-16 is a Thursday; 07-18 Saturday, 07-19 Sunday, 07-20 Monday.
    return datetime(2026, 7, day, hour, minute, tzinfo=ZoneInfo(TZ))


def _schedule(**kwargs) -> ScalingSchedule:
    defaults = dict(enabled=True, baseline_replicas=0)
    defaults.update(kwargs)
    return ScalingSchedule(**defaults)


def _rule(start_cron: str, duration_seconds: int, replicas: int, **kw):
    return ScalingScheduleRule(
        start_cron=start_cron,
        duration_seconds=duration_seconds,
        replicas=replicas,
        **kw,
    )


# --- compute_desired_replicas: window (start + duration) semantics -----------


def test_disabled_schedule_returns_none():
    sched = _schedule(enabled=False, rules=[_rule("0 8 * * *", 12 * HOUR, 1)])
    assert compute_desired_replicas(sched, _at(10, 0)) is None


def test_before_window_uses_baseline_not_retroactive():
    """
    Before the window opens the count is the baseline; a window that fired on a
    previous day must not retroactively keep the model scaled up.
    """
    # Window opens 10:27, lasts 8h (→ 18:27).
    sched = _schedule(baseline_replicas=0, rules=[_rule("27 10 * * *", 8 * HOUR, 1)])
    assert compute_desired_replicas(sched, _at(2, 25)) == 0
    assert compute_desired_replicas(sched, _at(10, 26)) == 0


def test_inside_window_uses_rule_replicas():
    sched = _schedule(baseline_replicas=0, rules=[_rule("27 10 * * *", 8 * HOUR, 1)])
    assert compute_desired_replicas(sched, _at(10, 28)) == 1
    assert compute_desired_replicas(sched, _at(15, 0)) == 1
    # Last minute inside the window.
    assert compute_desired_replicas(sched, _at(18, 26)) == 1


def test_after_window_falls_back_to_baseline():
    sched = _schedule(baseline_replicas=0, rules=[_rule("27 10 * * *", 8 * HOUR, 1)])
    assert compute_desired_replicas(sched, _at(18, 28)) == 0
    assert compute_desired_replicas(sched, _at(23, 0)) == 0


def test_overlapping_windows_most_recent_start_wins():
    sched = _schedule(
        baseline_replicas=0,
        rules=[
            _rule("0 0 * * *", 24 * HOUR, 2),  # all-day at 2
            _rule("0 9 * * *", 3 * HOUR, 5),  # 09:00–12:00 at 5, starts later
        ],
    )
    assert compute_desired_replicas(sched, _at(8, 0)) == 2  # only all-day active
    assert compute_desired_replicas(sched, _at(10, 0)) == 5  # both active, later wins
    assert compute_desired_replicas(sched, _at(13, 0)) == 2  # back to all-day only


def test_cross_midnight_window():
    """22:00 + 8h spans midnight (→ 06:00 next day) without wrap-around bugs."""
    sched = _schedule(baseline_replicas=1, rules=[_rule("0 22 * * *", 8 * HOUR, 3)])
    assert compute_desired_replicas(sched, _at(21, 0)) == 1  # before open
    assert compute_desired_replicas(sched, _at(23, 0)) == 3  # inside, before midnight
    assert compute_desired_replicas(sched, _at(2, 0)) == 3  # inside, after midnight
    assert compute_desired_replicas(sched, _at(7, 0)) == 1  # after close


def test_weekend_window_spans_whole_weekend():
    """Saturday 00:00 + 48h covers the whole weekend (start + duration model)."""
    sched = _schedule(baseline_replicas=1, rules=[_rule("0 0 * * 6", 48 * HOUR, 3)])
    assert compute_desired_replicas(sched, _at(12, 0, day=17)) == 1  # Fri
    assert compute_desired_replicas(sched, _at(12, 0, day=18)) == 3  # Sat
    assert compute_desired_replicas(sched, _at(12, 0, day=19)) == 3  # Sun
    assert compute_desired_replicas(sched, _at(12, 0, day=20)) == 1  # Mon


def test_rule_missing_duration_is_skipped():
    # Defensive path: an enabled schedule can't reach the API without a duration
    # on every rule, but a row written by an older build (or edited directly)
    # could. Build it disabled, then flip the flag to exercise the guard.
    sched = ScalingSchedule(
        enabled=False,
        baseline_replicas=4,
        rules=[ScalingScheduleRule(start_cron="0 9 * * *", replicas=9)],
    )
    sched.enabled = True
    # Rule has no duration_seconds → skipped → falls back to baseline.
    assert compute_desired_replicas(sched, _at(9, 30)) == 4


def test_overlapping_windows_same_start_takes_largest_replicas():
    # Windows sharing a start instant must not resolve by list order.
    weekday = _rule("0 9 * * 1-5", HOUR, 10)
    daily = _rule("0 9 * * *", HOUR, 2)
    thursday_0930 = _at(9, 30)
    for rules in ([weekday, daily], [daily, weekday]):
        sched = _schedule(baseline_replicas=0, rules=rules)
        assert compute_desired_replicas(sched, thursday_0930) == 10


def test_window_does_not_open_before_its_start(monkeypatch):
    # Spring-forward in America/New_York (2026-03-08 02:00 -> 03:00): 02:30 never
    # happens, so croniter resolves the "previous" 02:30 fire to 03:30 — a start
    # later than now. The window must stay shut until that start is reached.
    monkeypatch.setattr("gpustack.envs.TIMEZONE", "America/New_York")
    sched = _schedule(baseline_replicas=0, rules=[_rule("30 2 * * *", 2 * HOUR, 1)])
    ny = ZoneInfo("America/New_York")
    assert compute_desired_replicas(sched, datetime(2026, 3, 8, 3, 1, tzinfo=ny)) == 0
    # Once the resolved start has passed, the window is active for its duration.
    assert compute_desired_replicas(sched, datetime(2026, 3, 8, 4, 0, tzinfo=ny)) == 1
    assert compute_desired_replicas(sched, datetime(2026, 3, 8, 6, 0, tzinfo=ny)) == 0


def test_duration_seconds_bounded_to_avoid_overflow():
    # An unbounded duration overflows the timedelta used for the window end,
    # which would surface as a 500 from the create/update routes.
    with pytest.raises(ValidationError):
        ScalingScheduleRule(start_cron="0 9 * * *", duration_seconds=10**14, replicas=1)


def test_unsatisfiable_cron_rejected():
    # croniter.is_valid accepts February 30th; such a window would never open
    # while the scheduler logged a failure every tick.
    with pytest.raises(ValidationError):
        ScalingScheduleRule(start_cron="0 0 30 2 *", duration_seconds=HOUR, replicas=1)


# --- apply_scaling_schedule_baseline: save-time replica reconciliation --------


def _model(replicas: int, schedule: ScalingSchedule) -> ModelCreate:
    return ModelCreate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="x/y",
        replicas=replicas,
        scaling_schedule=schedule,
    )


def test_apply_ignores_submitted_replicas_and_uses_window_value():
    # An always-active window (daily 00:00 + 24h) makes the effective value
    # deterministic regardless of wall-clock time. While a schedule is enabled
    # the submitted replicas is not a user setting — the schedule owns it.
    sched = _schedule(baseline_replicas=1, rules=[_rule("0 0 * * *", 24 * HOUR, 2)])
    m = _model(replicas=7, schedule=sched)
    apply_scaling_schedule_baseline(m)
    assert m.scaling_schedule.baseline_replicas == 1  # user input, untouched
    assert m.replicas == 2  # driven to the in-window value, not the submitted 7


def test_placement_validated_when_a_window_scales_above_zero():
    # Scaling to zero outside business hours is a headline use case, so a
    # submitted replicas of 0 says nothing about whether instances are ever
    # placed — the scheduler raises the count later. The gate that decides
    # whether to validate gpu_selector must look at the whole schedule.
    sched = _schedule(baseline_replicas=0, rules=[_rule("0 9 * * *", HOUR, 5)])
    m = _model(replicas=0, schedule=sched)
    assert _max_intended_replicas(m) == 5

    # Without a schedule, a zero count still means "no instances" as before.
    plain = ModelCreate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="x/y",
        replicas=0,
    )
    assert _max_intended_replicas(plain) == 0


@pytest.mark.asyncio
async def test_validation_sees_submitted_replicas_not_schedule_value():
    # Validation must run against caller intent: rules downstream of it compare
    # against replicas (custom-backend topology, GPU placement), and a
    # schedule-driven value would make those checks depend on the wall clock.
    # A custom backend with no gpu_selector returns before touching the session.
    sched = _schedule(baseline_replicas=1, rules=[_rule("0 0 * * *", 24 * HOUR, 9)])
    m = ModelCreate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="x/y",
        replicas=1,
        backend=BackendEnum.CUSTOM.value,
        scaling_schedule=sched,
    )
    await validate_model_in(None, m)
    assert m.replicas == 1  # untouched by validation
    # The schedule is applied afterwards, as a server-side assignment.
    apply_scaling_schedule_baseline(m)
    assert m.replicas == 9


def test_apply_outside_window_uses_baseline():
    # Pin the only window to the 1st of a month half a year out so it is never
    # active while the test runs, making the baseline the deterministic outcome.
    far_month = ((datetime.now().month + 5) % 12) + 1
    sched = _schedule(
        baseline_replicas=1, rules=[_rule(f"0 0 1 {far_month} *", HOUR, 5)]
    )
    m = _model(replicas=7, schedule=sched)
    apply_scaling_schedule_baseline(m)
    assert m.replicas == 1  # baseline, not the submitted 7 nor the window's 5


def test_apply_noop_when_disabled():
    # Disabled: replicas is an ordinary user setting again and is left alone.
    sched = ScalingSchedule(enabled=False, baseline_replicas=0, rules=[])
    m = _model(replicas=5, schedule=sched)
    apply_scaling_schedule_baseline(m)
    assert m.replicas == 5  # untouched


def test_baseline_required_when_enabled():
    # Without a baseline an out-of-window schedule has no count to enforce, so
    # an enabled schedule must carry one.
    with pytest.raises(ValidationError):
        ScalingSchedule(
            enabled=True,
            baseline_replicas=None,
            rules=[_rule("0 9 * * *", 8 * HOUR, 3)],
        )


def test_baseline_optional_when_disabled():
    # A disabled schedule may be incomplete (in-progress edits must not 422).
    sched = ScalingSchedule(enabled=False, baseline_replicas=None, rules=[])
    assert sched.baseline_replicas is None


def test_window_length_is_absolute_across_dst(monkeypatch):
    # Spring-forward night in America/New_York (2026-03-08 02:00 -> 03:00): an
    # 8h window opening 22:00 the prior evening must span 8h of real time, not
    # 7h of wall-clock. Start is 2026-03-08 03:00 UTC, so the window ends at
    # 11:00 UTC; 10:30 UTC is inside, 11:30 UTC is outside.
    monkeypatch.setattr("gpustack.envs.TIMEZONE", "America/New_York")
    sched = _schedule(baseline_replicas=0, rules=[_rule("0 22 * * *", 8 * HOUR, 1)])
    inside = datetime(2026, 3, 8, 10, 30, tzinfo=timezone.utc)
    outside = datetime(2026, 3, 8, 11, 30, tzinfo=timezone.utc)
    assert compute_desired_replicas(sched, inside) == 1
    assert compute_desired_replicas(sched, outside) == 0


# --- scheduler tick: SQL pre-filter ------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change, expected",
    [
        ("unchanged", 2),
        ("disabled", None),
        ("removed", None),
        ("rules", 5),
        ("baseline", 5),
        ("converged", None),
        ("deleted", None),
        ("paused", None),
        ("missing", None),
    ],
)
async def test_tick_locks_and_rechecks_latest_schedule(
    monkeypatch, change, expected, cache_invalidations
):
    schedule = _schedule(baseline_replicas=1, rules=[_rule("0 0 * * *", 24 * HOUR, 2)])
    snapshot = Model(
        id=1,
        name="scheduled",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="a/b",
        replicas=0,
        scaling_schedule=schedule,
    )
    latest = snapshot.model_copy(deep=True)
    if change == "disabled":
        latest.scaling_schedule.enabled = False
    elif change == "removed":
        latest.scaling_schedule = None
    elif change == "rules":
        latest.scaling_schedule.rules[0].replicas = 5
    elif change == "baseline":
        latest.scaling_schedule.baseline_replicas = 5
        latest.scaling_schedule.rules = [_rule("0 0 1 1 *", HOUR, 2)]
    elif change == "converged":
        latest.replicas = 2
    elif change == "paused":
        latest.scaling_schedule.paused = True
    elif change == "deleted":
        latest.deleted_at = _at(10, 0)
    elif change == "missing":
        latest = None

    read_session = MagicMock()
    write_session = MagicMock()
    result = MagicMock()
    result.all.return_value = [latest] if latest is not None else []
    write_session.exec = AsyncMock(return_value=result)
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock()
    transaction.__aexit__ = AsyncMock()
    write_session.begin.return_value = transaction
    service = MagicMock()
    service.batch_update = AsyncMock()
    sessions = iter([read_session, write_session])

    @asynccontextmanager
    async def session():
        yield next(sessions)

    monkeypatch.setattr(scaling_scheduler, "async_session", session)
    monkeypatch.setattr(Model, "all_by_fields", AsyncMock(return_value=[snapshot]))
    monkeypatch.setattr(scaling_scheduler, "ModelService", lambda s: service)
    monkeypatch.setattr(
        scaling_scheduler,
        "compute_desired_replicas",
        lambda schedule, now: compute_desired_replicas(schedule, _at(10, 0)),
    )

    await scaling_scheduler.ScalingScheduler()._sync_scheduled_replicas()

    statement = write_session.exec.call_args.args[0]
    for dialect in (postgresql.dialect(), mysql.dialect()):
        assert str(statement.compile(dialect=dialect)).endswith("FOR UPDATE")
    if expected is None:
        service.batch_update.assert_not_awaited()
        for invalidation in cache_invalidations:
            invalidation.assert_not_awaited()
    else:
        assert latest.replicas == expected
        service.batch_update.assert_awaited_once_with([latest], auto_commit=False)
        cache_invalidations[0].assert_awaited_once()
        assert cache_invalidations[1].await_count == 2
    transaction.__aexit__.assert_awaited_once_with(None, None, None)
    assert statement.get_execution_options()["populate_existing"] is True
    if change == "disabled":
        assert latest.replicas == 0


@pytest.mark.asyncio
async def test_tick_query_skips_models_without_a_schedule():
    # The tick narrows to rows carrying a schedule. A model without one stores
    # the JSON literal 'null', not SQL NULL, so a plain IS NOT NULL predicate
    # would match every row and load the whole table.
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Model.__table__.create)
    try:
        async with AsyncSession(engine) as s:
            enabled = ScalingSchedule(
                enabled=True,
                baseline_replicas=1,
                rules=[_rule("0 9 * * *", HOUR, 2)],
            )
            disabled = ScalingSchedule(enabled=False, baseline_replicas=0, rules=[])
            for name, sched in [
                ("no-sched", None),
                ("legacy-null", None),
                ("disabled", disabled),
                ("enabled", enabled),
            ]:
                s.add(
                    Model(
                        name=name,
                        source=SourceEnum.HUGGING_FACE,
                        huggingface_repo_id="a/b",
                        replicas=1,
                        scaling_schedule=sched,
                    )
                )
            await s.commit()
            # A row predating the column holds a true SQL NULL.
            await s.exec(
                text(
                    "UPDATE models SET scaling_schedule = NULL WHERE name = 'legacy-null'"
                )
            )
            await s.commit()

            rows = await Model.all_by_fields(
                s,
                extra_conditions=[
                    col(Model.deleted_at).is_(None),
                    sa.cast(col(Model.scaling_schedule), sa.Text) != "null",
                ],
            )
            assert sorted(m.name for m in rows) == ["disabled", "enabled"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_tick_locks_batch_in_one_transaction(monkeypatch, cache_invalidations):
    schedule = _schedule(baseline_replicas=1, rules=[_rule("0 0 * * *", 24 * HOUR, 2)])
    snapshots = [
        Model(
            id=i,
            name=f"scheduled-{i}",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="a/b",
            replicas=0,
            scaling_schedule=schedule,
        )
        for i in range(1, 4)
    ]
    latest = [model.model_copy(deep=True) for model in snapshots]
    latest[1].scaling_schedule.paused = True
    events = []
    sessions = []
    write_session = MagicMock()

    @asynccontextmanager
    async def transaction():
        events.append("begin")
        try:
            yield
        finally:
            events.append("end")

    async def execute(statement):
        events.append("lock")
        result = MagicMock()
        result.all.return_value = latest
        return result

    async def update(models, *, auto_commit):
        assert auto_commit is False
        assert [(model.id, model.replicas) for model in models] == [(1, 2), (3, 2)]
        events.append("update")

    write_session.begin.side_effect = transaction
    write_session.exec = AsyncMock(side_effect=execute)

    @asynccontextmanager
    async def session():
        sessions.append(MagicMock() if not sessions else write_session)
        yield sessions[-1]

    service = MagicMock()
    service.batch_update = AsyncMock(side_effect=update)
    service_factory = MagicMock(return_value=service)
    monkeypatch.setattr(scaling_scheduler, "async_session", session)
    monkeypatch.setattr(Model, "all_by_fields", AsyncMock(return_value=snapshots))
    monkeypatch.setattr(scaling_scheduler, "ModelService", service_factory)
    await scaling_scheduler.ScalingScheduler()._sync_scheduled_replicas()

    assert len(sessions) == 2
    service_factory.assert_called_once_with(write_session)
    assert events == ["begin", "lock", "update", "end"]
    statement = write_session.exec.call_args.args[0]
    assert statement.get_execution_options()["populate_existing"] is True
    for dialect in (postgresql.dialect(), mysql.dialect()):
        sql = str(statement.compile(dialect=dialect))
        assert "ORDER BY models.id" in sql
        assert sql.endswith("FOR UPDATE")


@pytest.mark.asyncio
async def test_tick_evicts_cache_repopulated_before_commit(
    monkeypatch, cache_invalidations
):
    model = Model(
        id=1,
        name="scheduled",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="a/b",
        replicas=0,
        scaling_schedule=_schedule(
            baseline_replicas=1, rules=[_rule("0 0 * * *", 24 * HOUR, 2)]
        ),
    )
    database = {"replicas": 0}
    cache = {}
    observed = []

    async def read():
        cache.setdefault("replicas", database["replicas"])
        return cache["replicas"]

    async def evict():
        cache.clear()

    @asynccontextmanager
    async def transaction():
        yield
        # A concurrent reader can cache the committed row during a pending write.
        observed.append(await read())
        database["replicas"] = model.replicas

    async def update(rows, *, auto_commit):
        assert auto_commit is False
        assert rows == [model]
        await evict()

    write_session = MagicMock()
    write_session.begin.side_effect = transaction
    result = MagicMock()
    result.all.return_value = [model]
    write_session.exec = AsyncMock(return_value=result)
    sessions = iter([MagicMock(), write_session])

    @asynccontextmanager
    async def session():
        yield next(sessions)

    service = MagicMock()
    service.batch_update = AsyncMock(side_effect=update)
    cache_invalidations[0].side_effect = evict
    monkeypatch.setattr(scaling_scheduler, "async_session", session)
    monkeypatch.setattr(Model, "all_by_fields", AsyncMock(return_value=[model]))
    monkeypatch.setattr(scaling_scheduler, "ModelService", lambda s: service)
    await scaling_scheduler.ScalingScheduler()._sync_scheduled_replicas()

    assert observed == [0]
    assert await read() == 2
    cache_invalidations[1].assert_any_await(service.get_by_id, model.id)
    cache_invalidations[1].assert_any_await(service.get_by_name, model.name)


@pytest.mark.asyncio
async def test_tick_skips_paused_snapshot_without_requesting_a_write(monkeypatch):
    model = Model(
        id=1,
        name='paused',
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id='org/model',
        replicas=0,
        scaling_schedule=_schedule(
            paused=True, baseline_replicas=1, rules=[_rule("0 9 * * *", HOUR, 2)]
        ),
    )
    sessions = []

    @asynccontextmanager
    async def session():
        sessions.append(MagicMock())
        yield sessions[-1]

    monkeypatch.setattr(scaling_scheduler, 'async_session', session)
    monkeypatch.setattr(Model, 'all_by_fields', AsyncMock(return_value=[model]))
    service = MagicMock()
    service.batch_update = AsyncMock()
    monkeypatch.setattr(scaling_scheduler, 'ModelService', lambda s: service)
    await scaling_scheduler.ScalingScheduler()._sync_scheduled_replicas()
    assert len(sessions) == 1
    service.batch_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_batches_changed_replicas_after_one_reload(monkeypatch):
    """A tick reloads candidates together and commits changed rows together."""
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Model.__table__.create)

    schedule = ScalingSchedule(
        enabled=True,
        baseline_replicas=0,
        rules=[_rule("0 0 * * *", 24 * HOUR, 2)],
    )
    async with AsyncSession(engine) as session:
        session.add_all(
            [
                Model(
                    name="changed-a",
                    source=SourceEnum.HUGGING_FACE,
                    huggingface_repo_id="a/b",
                    replicas=0,
                    scaling_schedule=schedule.model_copy(deep=True),
                ),
                Model(
                    name="changed-b",
                    source=SourceEnum.HUGGING_FACE,
                    huggingface_repo_id="a/b",
                    replicas=1,
                    scaling_schedule=schedule.model_copy(deep=True),
                ),
                Model(
                    name="already-at-target",
                    source=SourceEnum.HUGGING_FACE,
                    huggingface_repo_id="a/b",
                    replicas=2,
                    scaling_schedule=schedule.model_copy(deep=True),
                ),
                Model(
                    name="paused",
                    source=SourceEnum.HUGGING_FACE,
                    huggingface_repo_id="a/b",
                    replicas=0,
                    scaling_schedule=schedule.model_copy(update={"paused": True}),
                ),
            ]
        )
        await session.commit()

    original_batch_update = scaling_scheduler.ModelService.batch_update
    batch_calls = []

    async def record_batch_update(service, models, *, auto_commit):
        batch_calls.append([(model.name, model.replicas) for model in models])
        assert auto_commit is False
        return await original_batch_update(service, models, auto_commit=auto_commit)

    monkeypatch.setattr(
        scaling_scheduler.ModelService, "batch_update", record_batch_update
    )
    monkeypatch.setattr(
        scaling_scheduler,
        "async_session",
        lambda: AsyncSession(engine, expire_on_commit=False),
    )

    await scaling_scheduler.ScalingScheduler(interval=0)._sync_scheduled_replicas()

    assert batch_calls == [[("changed-a", 2), ("changed-b", 2)]]
    async with AsyncSession(engine) as session:
        rows = (await session.exec(select(Model).order_by(Model.name))).all()
    assert [(row.name, row.replicas) for row in rows] == [
        ("already-at-target", 2),
        ("changed-a", 2),
        ("changed-b", 2),
        ("paused", 0),
    ]
    await engine.dispose()
