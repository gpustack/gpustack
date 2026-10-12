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
