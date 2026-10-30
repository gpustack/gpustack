import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.schema import CreateColumn


@pytest.fixture
def migration(monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "gpustack/migrations/versions"
        / "2026_10_30_1600-c8d3e6f1a902_v2_3_1_database_changes.py"
    )
    assert path.exists(), "Instance configuration snapshots require a migration"
    spec = importlib.util.spec_from_file_location("instance_config_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "op", MagicMock())
    monkeypatch.setattr(module, "table_exists", lambda name: True)
    monkeypatch.setattr(module, "column_exists", lambda *args: False)
    return module


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
def test_snapshot_column_is_nullable_portable_json(migration, dialect):
    migration.upgrade()
    table, column = migration.op.add_column.call_args.args
    assert table == "model_instances"
    assert column.name == "model_config_snapshot"
    assert column.nullable
    assert column.default is None
    assert "model_config_snapshot JSON" in str(
        CreateColumn(column).compile(dialect=dialect)
    )


@pytest.mark.parametrize("has_table,has_column", [(True, True), (False, False)])
def test_upgrade_skips_existing_column_or_missing_table(
    migration, monkeypatch, has_table, has_column
):
    monkeypatch.setattr(migration, "table_exists", lambda *args: has_table)
    monkeypatch.setattr(migration, "column_exists", lambda *args: has_column)
    migration.upgrade()
    migration.op.add_column.assert_not_called()


@pytest.mark.parametrize(
    "has_table,has_column", [(True, True), (True, False), (False, False)]
)
def test_downgrade_only_drops_an_existing_column(
    migration, monkeypatch, has_table, has_column
):
    monkeypatch.setattr(migration, "table_exists", lambda *args: has_table)
    monkeypatch.setattr(migration, "column_exists", lambda *args: has_column)
    migration.downgrade()
    assert migration.op.drop_column.call_count == int(has_table and has_column)
    if has_table and has_column:
        migration.op.drop_column.assert_called_once_with(
            "model_instances", "model_config_snapshot"
        )
