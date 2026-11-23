"""Archive DDL is idempotent and uses types shared by both SQL dialect families."""

import importlib
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.schema import CreateColumn, CreateTable

from gpustack.schemas.benchmark import Benchmark


def test_archive_upgrade_before_benchmarks_table_is_created(tmp_path):
    engine = sa.create_engine(f'sqlite:///{tmp_path / "bootstrap.db"}')
    try:
        config = migration_config(str(engine.url))
        command.stamp(config, 'c8d3e6f1a902')
        command.upgrade(config, 'f8c4a291bd65')
        with engine.begin() as connection:
            assert not sa.inspect(connection).has_table('benchmarks')
            Benchmark.__table__.create(connection)
            assert 'run_id' in {
                c['name'] for c in sa.inspect(connection).get_columns('benchmarks')
            }
            assert connection.execute(
                sa.text(
                    'SELECT id, max_bytes, used_bytes, initialized '
                    'FROM artifact_storage_capacity'
                )
            ).one() == (1, 0, 0, False)
    finally:
        engine.dispose()


def migration_config(url=None):
    config = Config()
    config.set_main_option(
        'script_location',
        str(Path(__file__).resolve().parents[2] / 'gpustack/migrations'),
    )
    if url is not None:
        config.set_main_option('sqlalchemy.url', url)
    return config


def test_archive_schema_upgrades_follow_the_existing_head():
    scripts = ScriptDirectory.from_config(migration_config())
    kv_transfer = scripts.get_revision('921586f4a09b')
    archive = scripts.get_revision('f8c4a291bd65')
    assert kv_transfer.down_revision == 'c8d3e6f1a902'
    assert archive.down_revision == kv_transfer.revision
    pending = scripts.iterate_revisions(scripts.get_current_head(), 'c8d3e6f1a902')
    assert {kv_transfer.revision, archive.revision} <= {
        step.revision for step in pending
    }


@pytest.mark.parametrize('backport_first', [False, True], ids=['v2.3.1', 'v2.3.x'])
def test_alembic_upgrade_to_monitoring_preserves_kv_settings(tmp_path, backport_first):
    path = tmp_path / 'already_at_c8.db'
    engine = sa.create_engine(f'sqlite:///{path}')
    with engine.begin() as connection:
        connection.execute(sa.text('CREATE TABLE workers (id INTEGER PRIMARY KEY)'))
        connection.execute(sa.text('INSERT INTO workers (id) VALUES (1)'))
        connection.execute(
            sa.text(
                'CREATE TABLE benchmarks (id INTEGER PRIMARY KEY, name VARCHAR(64))'
            )
        )
        connection.execute(
            sa.text(
                'CREATE TABLE model_instances ('
                'id INTEGER PRIMARY KEY, cluster_id INTEGER, model_id INTEGER, '
                'model_config_snapshot JSON)'
            )
        )
        for field in ('cluster_id', 'model_id'):
            connection.execute(
                sa.text(
                    f'CREATE INDEX ix_model_instances_{field} '
                    f'ON model_instances ({field})'
                )
            )
        connection.execute(sa.text("INSERT INTO benchmarks VALUES (1, 'existing-run')"))
        connection.execute(
            sa.text(
                'INSERT INTO model_instances VALUES '
                "(1, 2, 3, '{\"source\":\"captured-before-upgrade\"}')"
            )
        )
    config = migration_config(str(engine.url))
    command.stamp(config, 'c8d3e6f1a902')
    with engine.connect() as connection:
        assert (
            connection.execute(
                sa.text('SELECT version_num FROM alembic_version')
            ).scalar_one()
            == 'c8d3e6f1a902'
        )
        assert 'run_id' not in {
            column['name']
            for column in sa.inspect(connection).get_columns('benchmarks')
        }
    if backport_first:
        versions = tmp_path / 'v2.3-dev'
        versions.mkdir()
        scripts = ScriptDirectory.from_config(config)
        for step in scripts.walk_revisions(head='c8d3e6f1a902'):
            shutil.copyfile(step.path, versions / Path(step.path).name)
        kv_transfer = scripts.get_revision('921586f4a09b')
        shutil.copyfile(kv_transfer.path, versions / Path(kv_transfer.path).name)
        backport = migration_config(str(engine.url))
        backport.set_main_option('path_separator', 'os')
        backport.set_main_option('version_locations', str(versions))
        command.upgrade(backport, 'head')
        with engine.begin() as connection:
            assert (
                connection.execute(
                    sa.text('SELECT version_num FROM alembic_version')
                ).scalar_one()
                == kv_transfer.revision
            )
            assert not sa.inspect(connection).has_table('benchmark_monitoring_payloads')
            connection.execute(
                sa.text("UPDATE workers SET kv_transfer_ifname = 'ens5f0' WHERE id = 1")
            )
    command.upgrade(config, 'f8c4a291bd65')
    with engine.connect() as connection:
        inspector = sa.inspect(connection)
        assert (
            connection.execute(
                sa.text('SELECT version_num FROM alembic_version')
            ).scalar_one()
            == 'f8c4a291bd65'
        )
        archive_tables = {'artifact_storage_capacity', 'benchmark_monitoring_payloads'}
        assert set(inspector.get_table_names()) == archive_tables | {
            'alembic_version',
            'benchmarks',
            'model_instances',
            'workers',
        }
        assert connection.execute(
            sa.text('SELECT id, kv_transfer_ifname FROM workers')
        ).one() == (1, 'ens5f0' if backport_first else None)
        assert 'run_id' in {
            column['name'] for column in inspector.get_columns('benchmarks')
        }
        assert connection.execute(
            sa.text('SELECT name, run_id FROM benchmarks')
        ).one() == ('existing-run', None)
        assert (
            connection.execute(
                sa.text('SELECT model_config_snapshot FROM model_instances')
            ).scalar_one()
            == '{"source":"captured-before-upgrade"}'
        )
        assert {
            index['name'] for index in inspector.get_indexes('model_instances')
        } == {
            'ix_model_instances_cluster_id',
            'ix_model_instances_model_id',
        }
        assert connection.execute(
            sa.text(
                'SELECT id, max_bytes, used_bytes, initialized '
                'FROM artifact_storage_capacity'
            )
        ).one() == (1, 0, 0, False)
    command.downgrade(config, 'c8d3e6f1a902')
    with engine.connect() as connection:
        assert (
            connection.execute(
                sa.text('SELECT version_num FROM alembic_version')
            ).scalar_one()
            == 'c8d3e6f1a902'
        )
        assert not archive_tables & set(sa.inspect(connection).get_table_names())
        assert [
            column['name']
            for column in sa.inspect(connection).get_columns('benchmarks')
        ] == ['id', 'name']
        assert [
            column['name'] for column in sa.inspect(connection).get_columns('workers')
        ] == ['id']
        assert (
            connection.execute(
                sa.text('SELECT model_config_snapshot FROM model_instances')
            ).scalar_one()
            == '{"source":"captured-before-upgrade"}'
        )
    engine.dispose()


@pytest.mark.parametrize(
    'dialect, binary', [(postgresql.dialect(), 'BYTEA'), (mysql.dialect(), 'LONGBLOB')]
)
def test_new_revision_emits_portable_archive_ddl(monkeypatch, dialect, binary):
    migration = importlib.import_module(
        'gpustack.migrations.versions.'
        '2026_11_19_0100-f8c4a291bd65_benchmark_monitoring_payloads'
    )
    operations = MagicMock()
    operations.get_bind.return_value.execute.return_value.first.return_value = None
    metadata = sa.MetaData()
    emitted = []

    def create_table(name, *columns):
        emitted.append(sa.Table(name, metadata, *columns))

    operations.create_table.side_effect = create_table
    inspector = MagicMock()
    inspector.get_indexes.return_value = []
    monkeypatch.setattr(migration, 'op', operations)
    monkeypatch.setattr(migration, 'table_exists', lambda name: name == 'benchmarks')
    monkeypatch.setattr(migration, 'column_exists', lambda *args: False)
    monkeypatch.setattr(migration.sa, 'inspect', lambda bind: inspector)
    migration.upgrade()
    assert [table.name for table in emitted] == [
        'artifact_storage_capacity',
        'benchmark_monitoring_payloads',
    ]
    assert operations.add_column.call_count == 1
    table, column = operations.add_column.call_args.args
    assert table == 'benchmarks' and column.name == 'run_id'
    assert column.nullable
    assert 'run_id VARCHAR(64)' in str(CreateColumn(column).compile(dialect=dialect))
    for archive_table in emitted:
        statement = str(CreateTable(archive_table).compile(dialect=dialect))
        assert archive_table.name in statement
        assert 'JSONB' not in statement
        assert 'FOREIGN KEY' not in statement
    assert f'payload {binary}' in str(CreateTable(emitted[1]).compile(dialect=dialect))
    assert operations.create_index.call_count == 2
