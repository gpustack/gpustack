"""The metering key migration accepts MySQL-style unique index reflection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / 'gpustack/migrations/versions'
    / '2026_07_15_1600-367a3982fcde_v2_3_0_database_changes.py'
)
spec = importlib.util.spec_from_file_location('v2_3_0_database_changes', MIGRATION)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def test_mysql_index_metadata_is_used_when_reflection_misses_indexes(monkeypatch):
    bind = SimpleNamespace(
        dialect=SimpleNamespace(name='mysql'),
        execute=Mock(
            return_value=[('meter_key',), ('resource_id',), ('bucket_start',)]
        ),
    )
    monkeypatch.setattr(migration, 'op', SimpleNamespace(get_bind=lambda: bind))

    assert migration._index_exists('metered_usage', 'uq_metered_usage')
    assert (
        migration._unique_constraint_columns('metered_usage', 'uq_metered_usage')
        == migration._UQ_NARROW
    )
    assert bind.execute.call_args.args[1] == {
        'table_name': 'metered_usage',
        'index_name': 'uq_metered_usage',
    }
    assert 'information_schema.STATISTICS' in bind.execute.call_args.args[0].text


def test_metered_usage_key_upgrade_can_be_retried(monkeypatch):
    engine = sa.create_engine('sqlite://')
    metadata = sa.MetaData()
    table = sa.Table(
        'metered_usage',
        metadata,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('meter_key', sa.String(255), nullable=False),
        sa.Column('resource_id', sa.Integer, nullable=False),
        sa.Column('bucket_start', sa.DateTime, nullable=False),
        sa.Column('sku', sa.String(255)),
        sa.Column('sku_count', sa.Numeric(20, 8), nullable=False, server_default='1'),
        sa.Column('definition_snapshot', sa.String(255)),
        sa.Column('instance_type_name', sa.String(255)),
        sa.UniqueConstraint(*migration._UQ_NARROW, name='uq_metered_usage'),
    )
    metadata.create_all(engine)

    with engine.connect() as connection:
        op = Operations(MigrationContext.configure(connection))
        monkeypatch.setattr(migration, 'op', op)
        monkeypatch.setattr(migration, 'table_exists', lambda name: name == table.name)
        monkeypatch.setattr(migration, 'column_exists', lambda table, col: True)

        migration._upgrade_metering_sku_shape()
        migration._upgrade_metering_sku_shape()

        constraints = sa.inspect(connection).get_unique_constraints(table.name)
        assert constraints == [
            {'name': 'uq_metered_usage', 'column_names': migration._UQ_WIDE}
        ]
