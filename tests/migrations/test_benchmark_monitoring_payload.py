"""The archive schema preserves live quota on retries."""

import importlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.schema import CreateTable

from gpustack.schemas.artifact_storage import ArtifactStorageCapacity as Capacity
from gpustack.schemas.benchmark_monitoring_payload import (
    BenchmarkMonitoringPayload as Payload,
)


@pytest.mark.parametrize(
    'dialect, binary', [(postgresql.dialect(), 'BYTEA'), (mysql.dialect(), 'LONGBLOB')]
)
def test_archive_bundle_column_compiles_as_native_binary(dialect, binary):
    ddl = str(CreateTable(Payload.__table__).compile(dialect=dialect))
    assert f'payload {binary}' in ddl
    assert 'FOREIGN KEY' not in ddl
    assert 'PRIMARY KEY (benchmark_id)' in ddl


@pytest.mark.parametrize('operation', ['bulk_insert', 'create_index'])
def test_archive_upgrade_retries_without_overwriting_quota(monkeypatch, operation):
    revision = importlib.import_module(
        'gpustack.migrations.versions.'
        '2026_11_19_0100-f8c4a291bd65_benchmark_monitoring_payloads'
    )
    with sa.create_engine('sqlite:///:memory:').begin() as connection:
        connection.execute(
            sa.text('CREATE TABLE benchmark_monitoring_chunks (id INTEGER)')
        )
        connection.execute(
            sa.text('INSERT INTO benchmark_monitoring_chunks VALUES (1)')
        )
        monkeypatch.setattr(
            revision, 'op', Operations(MigrationContext.configure(connection))
        )
        monkeypatch.setattr(
            revision,
            'table_exists',
            lambda name: sa.inspect(connection).has_table(name),
        )

        def interrupt(*args, **kwargs):
            raise RuntimeError('interrupted migration')

        with monkeypatch.context() as interrupted:
            interrupted.setattr(revision.op, operation, interrupt)
            with pytest.raises(RuntimeError, match='interrupted migration'):
                revision.upgrade()
        assert sa.inspect(connection).has_table(Capacity.__tablename__)

        revision.upgrade()
        capacity = sa.Table(
            Capacity.__tablename__, sa.MetaData(), autoload_with=connection
        )
        assert list(capacity.columns.keys()) == list(Capacity.__table__.columns.keys())
        assert connection.execute(sa.select(capacity)).one() == (1, 0, 0, False)
        reflected = sa.Table(
            Payload.__tablename__, sa.MetaData(), autoload_with=connection
        )
        assert [c.name for c in reflected.columns] == [
            c.name for c in Payload.__table__.columns
        ]
        for name in reflected.columns.keys():
            assert reflected.c[name].nullable == Payload.__table__.c[name].nullable
        assert {i.name for i in reflected.indexes} == {
            i.name for i in Payload.__table__.indexes
        }

        connection.execute(
            sa.insert(Payload.__table__).values(
                benchmark_id=1, run_id='new-run', compressed_bytes=357
            )
        )
        connection.execute(
            capacity.update()
            .where(capacity.c.id == 1)
            .values(max_bytes=4096, used_bytes=357, initialized=True)
        )
        revision.upgrade()
        assert tuple(connection.execute(sa.select(capacity)).one()) == (
            1,
            4096,
            357,
            True,
        )
        with pytest.raises(RuntimeError, match='monitoring archives'):
            revision.downgrade()
        connection.execute(sa.delete(Payload.__table__))
        revision.downgrade()
        assert not sa.inspect(connection).has_table(Payload.__tablename__)
        assert not sa.inspect(connection).has_table(Capacity.__tablename__)
        assert (
            connection.execute(
                sa.text('SELECT id FROM benchmark_monitoring_chunks')
            ).scalar_one()
            == 1
        )
