import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql


def migration():
    path = next(
        (Path(__file__).resolve().parents[2] / "gpustack/migrations/versions").glob(
            "*d5e8f0a1b2c3*.py"
        )
    )
    spec = importlib.util.spec_from_file_location("cache_instance_resources", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
def test_instance_table_includes_portable_reservation_column(monkeypatch, dialect):
    revision = migration()
    operations = MagicMock()
    monkeypatch.setattr(revision, "op", operations)
    revision.upgrade()

    instance_table = next(
        call.args
        for call in operations.create_table.call_args_list
        if call.args[0] == "cache_service_instances"
    )
    column = next(
        column
        for column in instance_table[1:]
        if isinstance(column, sa.Column) and column.name == "computed_resource_claim"
    )
    assert isinstance(column.type, sa.JSON)
    assert column.type.compile(dialect=dialect).lower() == "json"
    assert column.nullable


def test_downgrade_removes_instance_table(monkeypatch):
    revision = migration()
    operations = MagicMock()
    monkeypatch.setattr(revision, "op", operations)
    revision.downgrade()
    operations.drop_table.assert_any_call("cache_service_instances")
