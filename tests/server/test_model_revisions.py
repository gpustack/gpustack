from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy import func
from sqlalchemy.engine.result import IteratorResult, SimpleResultMetaData
from sqlalchemy.orm import Session
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.schemas.model_revisions import ModelRevision
from gpustack.schemas.models import (
    Model,
    ModelUpdate,
    RoleSpec,
    ScalingSchedule,
    ScalingScheduleRule,
    LoraListEntry,
)
from gpustack.server.model_revisions import (
    REVISION_FIELDS,
    append_revision,
    deployment_spec,
    ensure_baseline,
    latest_revision,
    lock_model,
    prepare_history_read,
    prune_revisions,
    record_update,
)


def model(**values):
    return Model(
        **dict(
            {
                "id": 1,
                "name": "qwen",
                "source": "huggingface",
                "huggingface_repo_id": "org/qwen",
                "owner_principal_id": 5,
            },
            **values,
        )
    )


class HistorySession:
    """In-memory client boundary for revision queries, without a database."""

    def __init__(self):
        self.rows = []
        self.statements = []
        self.flush = AsyncMock()
        self.commit = AsyncMock()
        self.rollback = AsyncMock()

    def add(self, row):
        row.id = len(self.rows) + 1
        self.rows.append(row)

    async def exec(self, statement):
        self.statements.append(statement)
        params = statement.compile().params
        rows = sorted(
            [r for r in self.rows if r.model_id == params['model_id_1']],
            key=lambda r: r.revision,
            reverse=True,
        )
        offset = (
            statement._offset_clause.value
            if statement._offset_clause is not None
            else 0
        )
        limit = statement._limit_clause.value
        rows = rows[offset : offset + limit]
        scalar = len(statement.selected_columns) == 1
        value = (rows[0].revision if scalar else rows[0]) if rows else None
        return SimpleNamespace(first=lambda: value)

    async def execute(self, statement):
        self.statements.append(statement)
        params = statement.compile().params
        self.rows = [
            r
            for r in self.rows
            if not (
                r.model_id == params['model_id_1'] and r.revision < params['revision_1']
            )
        ]


def test_revision_fields_explicitly_classify_model_update_fields():
    fields = set(ModelUpdate.model_fields)
    recorded = set(REVISION_FIELDS)
    excluded = {
        # Resource identity, ownership and access remain attached to the live model.
        "name",
        "owner_principal_id",
        "cluster_id",
        "access_policy",
        # Annotations and history policy do not change deployment configuration.
        "description",
        "revision_history_limit",
        # Runtime observations and derived model capabilities are not user intent.
        "ready_replicas",
        "meta",
        "distributable",
    }
    assert len(recorded) == len(REVISION_FIELDS), "Duplicate revision fields"
    assert not recorded - fields, f"Unknown revision fields: {recorded - fields}"
    assert not excluded - fields, f"Unknown exclusions: {excluded - fields}"
    assert not recorded & excluded, f"Conflicting exclusions: {recorded & excluded}"
    missing = fields - recorded - excluded
    assert not missing, f"Classify new fields as recorded or excluded: {missing}"


def test_projection_excludes_identity_status_and_retention():
    current = model(
        revision_history_limit=0,
        ready_replicas=8,
        description="text",
        meta={"n_params": 123},
        env={"EXAMPLE_TOKEN": "fixture-value"},
    )
    spec = deployment_spec(current)
    for key in (
        "name",
        "owner_principal_id",
        "cluster_id",
        "revision_history_limit",
        "ready_replicas",
        "meta",
        "description",
        "id",
        "state",
    ):
        assert key not in spec
    assert spec["env"] == {"EXAMPLE_TOKEN": "fixture-value"}
    spec["env"]["EXAMPLE_TOKEN"] = "changed"
    assert current.env["EXAMPLE_TOKEN"] == "fixture-value"


def test_lora_snapshot_uses_portable_names_without_runtime_files():
    current = model(
        lora_list=[
            LoraListEntry(
                lora_name="qwen:adapter",
                lora_repo_name="org/adapter",
                path="/runtime",
                model_file_id=9,
            )
        ]
    )
    adapter = deployment_spec(current)["lora_list"][0]
    assert adapter["lora_name"] == "adapter"
    assert "path" not in adapter and "model_file_id" not in adapter
    assert current.lora_list[0].lora_name == "qwen:adapter"


def test_role_empty_override_is_distinct_from_inheritance():
    inherited = model(roles=[RoleSpec(name="worker", env=None)])
    empty = model(roles=[RoleSpec(name="worker", env={})])
    assert deployment_spec(inherited) != deployment_spec(empty)


def test_schedule_snapshot_uses_baseline_instead_of_live_replica_count():
    current = model(
        replicas=9,
        scaling_schedule=ScalingSchedule(
            enabled=True,
            baseline_replicas=2,
            rules=[
                ScalingScheduleRule(
                    start_cron="0 8 * * *", duration_seconds=60, replicas=9
                )
            ],
        ),
    )
    before = deployment_spec(current)
    current.replicas = 0
    assert deployment_spec(current) == before
    assert before["replicas"] == 2


@pytest.mark.parametrize("value", [-1, -10])
def test_retention_rejects_negative_values(value):
    with pytest.raises(ValidationError):
        ModelUpdate(
            name="qwen",
            source="huggingface",
            huggingface_repo_id="org/qwen",
            revision_history_limit=value,
        )


@pytest.mark.asyncio
async def test_baseline_is_idempotent_and_records_no_actor():
    session = HistorySession()
    current = model()
    first = await ensure_baseline(session, current)
    assert await ensure_baseline(session, current) is first
    assert len(session.rows) == 1
    assert first.created_by is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1, 10])
async def test_retention_keeps_latest_and_versions_continue_increasing(limit):
    session = HistorySession()
    current = model(revision_history_limit=limit)
    latest = await append_revision(session, current, 1, 99)
    for number in range(2, 16):
        before = deployment_spec(current)
        current.backend_parameters = [f"--max-model-len={number}"]
        await record_update(session, current, before, latest, 99)
        latest = await latest_revision(session, current.id)
    assert [row.revision for row in session.rows] == list(range(15 - limit, 16))
    assert len(session.rows) == limit + 1
    assert latest.revision == 15
    assert latest.created_by == 99
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_noop_and_status_changes_do_not_add_versions():
    session = HistorySession()
    current = model()
    latest = await ensure_baseline(session, current)
    before = deployment_spec(current)
    current.ready_replicas = 1
    current.state_message = "ready"
    current.meta = {"runtime": "value"}
    current.description = "edited description"
    await record_update(session, current, before, latest, 99)
    assert len(session.rows) == 1


@pytest.mark.asyncio
async def test_generic_proxy_change_creates_revision():
    session = HistorySession()
    current = model(generic_proxy=False)
    latest = await ensure_baseline(session, current)
    before = deployment_spec(current)
    current.generic_proxy = True
    await record_update(session, current, before, latest, 99)
    assert [row.spec["generic_proxy"] for row in session.rows] == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statement,value",
    [
        (select(func.count()).select_from(ModelRevision), 5),
        (select(ModelRevision.revision), 3),
    ],
)
async def test_sqlmodel_single_column_queries_return_scalars(
    monkeypatch, statement, value
):
    # The driver boundary returns rows; SQLModel performs scalar unwrapping.
    monkeypatch.setattr(
        Session,
        "execute",
        lambda *args, **kwargs: IteratorResult(
            SimpleResultMetaData(["value"]), iter([(value,)])
        ),
    )
    async with AsyncSession() as session:
        assert (await session.exec(statement)).one() == value


@pytest.mark.asyncio
async def test_retention_change_prunes_only_this_deployment_without_new_version():
    session = HistorySession()
    current = model()
    for number in range(1, 8):
        await append_revision(session, current, number)
    await append_revision(session, model(id=2), 1)
    before = deployment_spec(current)
    current.revision_history_limit = 0
    await record_update(session, current, before, await latest_revision(session, 1))
    assert [(r.model_id, r.revision) for r in session.rows] == [(1, 7), (2, 1)]
    current.revision_history_limit = 10
    await prune_revisions(session, current)
    assert len(session.rows) == 2


@pytest.mark.asyncio
async def test_rollback_snapshot_creates_new_version_even_if_target_is_pruned():
    session = HistorySession()
    current = model(revision_history_limit=1, backend_parameters=["--a"])
    first = await append_revision(session, current, 1)
    before = deployment_spec(current)
    current.backend_parameters = ["--b"]
    await record_update(session, current, before, first)
    before = deployment_spec(current)
    current.backend_parameters = first.spec["backend_parameters"]
    await record_update(session, current, before, await latest_revision(session, 1))
    assert [r.revision for r in session.rows] == [2, 3]
    assert session.rows[-1].spec == first.spec


@pytest.mark.asyncio
async def test_lock_refreshes_parent_and_denies_other_tenant():
    session = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(one_or_none=lambda: model()))
    )
    ctx = TenantContext(
        user=None, is_platform_admin=False, current_principal_id=8, org_role=None
    )
    with pytest.raises(NotFoundException):
        await lock_model(session, ctx, 1)
    statement = session.exec.await_args.args[0]
    assert statement.get_execution_options()["populate_existing"]
    for dialect in (mysql.dialect(), postgresql.dialect()):
        assert "FOR UPDATE" in str(statement.compile(dialect=dialect))


@pytest.mark.asyncio
async def test_history_queries_use_current_reads_under_repeatable_read():
    session = HistorySession()
    await ensure_baseline(session, model())
    await prune_revisions(session, model())
    for statement in session.statements:
        assert "FOR UPDATE" in str(statement.compile(dialect=mysql.dialect()))


@pytest.fixture
def read_ctx():
    return TenantContext(
        user=None, is_platform_admin=False, current_principal_id=5, org_role=None
    )


def read_session(*values):
    return SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    first=lambda value=value: value,
                    one_or_none=lambda value=value: value,
                )
                for value in values
            ]
        ),
        add=Mock(),
        flush=AsyncMock(),
        commit=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_existing_history_is_read_without_locks_or_writes(read_ctx):
    session = read_session(model(), 7)
    await prepare_history_read(session, read_ctx, 1)
    assert session.exec.await_count == 2
    for call in session.exec.await_args_list:
        for dialect in (mysql.dialect(), postgresql.dialect()):
            assert "FOR UPDATE" not in str(call.args[0].compile(dialect=dialect))
    assert "model_revisions.spec" not in str(session.exec.await_args.args[0])
    session.add.assert_not_called()
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("parent", [None, model(owner_principal_id=8)])
async def test_history_read_checks_parent_before_querying_revisions(read_ctx, parent):
    session = read_session(parent)
    with pytest.raises(NotFoundException):
        await prepare_history_read(session, read_ctx, 1)
    assert session.exec.await_count == 1
    assert "model_revisions" not in str(session.exec.await_args.args[0])
    session.add.assert_not_called()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("initialized_by_another_request", [False, True])
async def test_missing_history_rechecks_under_lock_and_ends_read_snapshot(
    read_ctx, initialized_by_another_request
):
    current = model(replicas=3)
    concurrent_revision = (
        ModelRevision(model_id=1, revision=2, spec=deployment_spec(current))
        if initialized_by_another_request
        else None
    )
    session = read_session(model(replicas=1), None, current, concurrent_revision)
    await prepare_history_read(session, read_ctx, 1)
    calls = session.exec.await_args_list
    assert len(calls) == 4
    assert calls[2].args[0].get_execution_options()["populate_existing"]
    for index, call in enumerate(calls):
        for dialect in (mysql.dialect(), postgresql.dialect()):
            sql = str(call.args[0].compile(dialect=dialect))
            assert ("FOR UPDATE" in sql) == (index >= 2)
    if initialized_by_another_request:
        session.add.assert_not_called()
        session.flush.assert_not_awaited()
    else:
        session.add.assert_called_once()
        baseline = session.add.call_args.args[0]
        assert baseline.revision == 1
        assert baseline.spec["replicas"] == 3
        assert baseline.created_by is None
        session.flush.assert_awaited_once()
    # A fresh transaction must see the baseline even when the initial read
    # snapshot predates another request's committed initialization.
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("parent", [None, model(owner_principal_id=8)])
async def test_baseline_initialization_rechecks_parent_visibility(read_ctx, parent):
    session = read_session(model(), None, parent)
    with pytest.raises(NotFoundException):
        await prepare_history_read(session, read_ctx, 1)
    assert session.exec.await_count == 3
    session.add.assert_not_called()
    session.commit.assert_not_awaited()
