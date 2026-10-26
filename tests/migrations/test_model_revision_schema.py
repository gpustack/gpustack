"""History DDL must be repeatable and portable across supported SQL dialects."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.schema import CreateTable


@pytest.fixture
def migration(monkeypatch):
    path = next(
        (Path(__file__).resolve().parents[2] / "gpustack/migrations/versions").glob(
            "*367a3982fcde*.py"
        )
    )
    spec = importlib.util.spec_from_file_location("revision_schema_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "op", MagicMock())
    monkeypatch.setattr(module, "table_exists", lambda name: False)
    monkeypatch.setattr(module, "column_exists", lambda table, column: False)
    return module


@pytest.fixture
def router_migration(monkeypatch):
    paths = list(
        (Path(__file__).resolve().parents[2] / "gpustack/migrations/versions").glob(
            "*a7e2c9d4f681*.py"
        )
    )
    assert paths, "Router snapshots need a migration after the existing head"
    spec = importlib.util.spec_from_file_location("router_snapshot_migration", paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "op", MagicMock())
    monkeypatch.setattr(module, "table_exists", lambda name: True)
    monkeypatch.setattr(module, "column_exists", lambda table, column: False)
    return module


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
def test_history_schema_constraints_and_default(migration, dialect):
    migration._upgrade_model_revisions()
    table_name, column = migration.op.add_column.call_args.args
    assert table_name == "models"
    assert column.name == "revision_history_limit"
    assert column.nullable is False
    assert column.server_default.arg == '10'

    args = migration.op.create_table.call_args.args
    metadata = sa.MetaData()
    for name in ("models", "principals"):
        sa.Table(name, metadata, sa.Column("id", sa.Integer, primary_key=True))
    table = sa.Table(args[0], metadata, *args[1:])
    ddl = str(CreateTable(table).compile(dialect=dialect))
    assert "UNIQUE (model_id, revision)" in ddl
    assert "REFERENCES models (id) ON DELETE CASCADE" in ddl
    assert "REFERENCES principals (id) ON DELETE SET NULL" in ddl
    assert "spec JSON NOT NULL" in ddl
    assert not table.c.created_at.nullable


@pytest.mark.parametrize(
    "has_column,has_table", [(True, True), (True, False), (False, True)]
)
def test_partial_upgrade_only_creates_missing_objects(
    migration, monkeypatch, has_column, has_table
):
    monkeypatch.setattr(migration, "column_exists", lambda *args: has_column)
    monkeypatch.setattr(migration, "table_exists", lambda *args: has_table)
    migration._upgrade_model_revisions()
    assert migration.op.add_column.call_count == int(not has_column)
    assert migration.op.create_table.call_count == int(not has_table)


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
def test_router_snapshot_is_nullable_portable_json(
    router_migration, monkeypatch, dialect
):
    from sqlalchemy.schema import CreateColumn

    migration = router_migration
    migration.upgrade()
    table, column = migration.op.add_column.call_args.args
    assert table == "model_instances"
    assert column.nullable
    assert "router_peer_snapshot JSON" in str(
        CreateColumn(column).compile(dialect=dialect)
    )
    migration.op.reset_mock()
    monkeypatch.setattr(migration, "column_exists", lambda *args: True)
    migration.upgrade()
    migration.op.add_column.assert_not_called()
    migration.downgrade()
    migration.op.drop_column.assert_called_once_with(
        "model_instances", "router_peer_snapshot"
    )


def test_router_snapshot_revision_runs_after_an_existing_database_head():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "gpustack/migrations"),
    )
    scripts = ScriptDirectory.from_config(cfg)
    revision = scripts.get_revision("a7e2c9d4f681")
    assert revision.down_revision == "f4a5b6c7d8e9"
    pending = list(
        scripts.iterate_revisions(scripts.get_current_head(), "f4a5b6c7d8e9")
    )
    assert revision.revision in {step.revision for step in pending}


def test_router_snapshot_is_not_added_by_the_shipped_bundle(migration):
    assert not hasattr(migration, "_upgrade_router_peer_snapshot")


@pytest.mark.parametrize(
    "table_exists,column_exists", [(False, False), (False, True), (True, False)]
)
def test_router_snapshot_absent_objects_make_downgrade_a_noop(
    router_migration, monkeypatch, table_exists, column_exists
):
    monkeypatch.setattr(router_migration, "table_exists", lambda _: table_exists)
    monkeypatch.setattr(router_migration, "column_exists", lambda *args: column_exists)
    router_migration.downgrade()
    router_migration.op.drop_column.assert_not_called()
