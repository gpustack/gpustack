import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.schema import CreateColumn

USAGE_TABLES = {"model_usages", "model_usage_details", "model_usage_details_archive"}
IMAGE_COLUMNS = {"image_count", "image_pixel_count"}


@pytest.fixture
def migration(monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "gpustack/migrations/versions"
        / "2026_10_30_1600-c8d3e6f1a902_v2_3_1_database_changes.py"
    )
    spec = importlib.util.spec_from_file_location("image_usage_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "op", MagicMock())
    monkeypatch.setattr(module, "table_exists", lambda name: True)
    monkeypatch.setattr(module, "column_exists", lambda *args: False)
    return module


def _usage_calls(op_method):
    return [c for c in op_method.call_args_list if c.args[0] in USAGE_TABLES]


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
def test_image_columns_are_non_null_bigint_defaulting_to_zero(migration, dialect):
    migration.upgrade()

    calls = _usage_calls(migration.op.add_column)
    added = {(c.args[0], c.args[1].name) for c in calls}
    assert added == {(t, col) for t in USAGE_TABLES for col in IMAGE_COLUMNS}
    for call in calls:
        column = call.args[1]
        assert column.nullable is False
        ddl = str(CreateColumn(column).compile(dialect=dialect))
        assert "BIGINT" in ddl and "DEFAULT '0'" in ddl and "NOT NULL" in ddl


def test_upgrade_skips_columns_that_already_exist(migration, monkeypatch):
    monkeypatch.setattr(
        migration, "column_exists", lambda table, col: col == "image_count"
    )
    migration.upgrade()

    added = {(c.args[0], c.args[1].name) for c in _usage_calls(migration.op.add_column)}
    assert added == {(t, "image_pixel_count") for t in USAGE_TABLES}


def test_upgrade_skips_missing_tables(migration, monkeypatch):
    monkeypatch.setattr(
        migration, "table_exists", lambda name: name != "model_usage_details_archive"
    )
    migration.upgrade()

    tables = {c.args[0] for c in _usage_calls(migration.op.add_column)}
    assert tables == {"model_usages", "model_usage_details"}


def test_downgrade_drops_only_existing_image_columns(migration, monkeypatch):
    monkeypatch.setattr(migration, "column_exists", lambda table, col: True)
    migration.downgrade()

    dropped = {c.args for c in _usage_calls(migration.op.drop_column)}
    assert dropped == {(t, col) for t in USAGE_TABLES for col in IMAGE_COLUMNS}
