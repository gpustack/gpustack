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


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
def test_site_settings_table_upgrade_is_portable_and_idempotent(
    migration, monkeypatch, dialect
):
    import sqlalchemy as sa
    from sqlalchemy.schema import CreateTable

    monkeypatch.setattr(migration, "table_exists", lambda name: name != "site_settings")
    migration.upgrade()
    name, *columns = migration.op.create_table.call_args.args
    table = sa.Table(name, sa.MetaData(), *columns)
    ddl = str(CreateTable(table).compile(dialect=dialect))
    assert "id = 1" in ddl
    assert table.c.site_name.type.length == 128
    assert table.c.updated_at.nullable
    assert "MEDIUMTEXT" in ddl if dialect.name == "mysql" else "TEXT" in ddl
    migration.op.create_table.reset_mock()
    monkeypatch.setattr(migration, "table_exists", lambda name: True)
    migration.upgrade()
    migration.op.create_table.assert_not_called()


def test_site_settings_downgrade_only_drops_existing_table(migration, monkeypatch):
    migration.downgrade()
    migration.op.drop_table.assert_called_once_with("site_settings")
    migration.op.drop_table.reset_mock()
    monkeypatch.setattr(migration, "table_exists", lambda name: False)
    migration.downgrade()
    migration.op.drop_table.assert_not_called()
