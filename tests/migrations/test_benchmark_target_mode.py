"""`target_mode` lands on existing rows as what those rows measured.

The column is added to a table that already has runs in it, and every one of
them aimed at an instance — the mode did not exist to be chosen. A NULL there
would read as "unknown" on a report that is not unknown at all, so the
migration backfills rather than leaving the old rows blank.
"""

import importlib.util
import sqlite3
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

# Matched by revision id rather than by file name: the name carries a
# timestamp that moves whenever the revision is re-chained, and a rebase that
# renamed it should not break a test about what the migration does.
_PATH = next(
    (Path(__file__).resolve().parents[2] / "gpustack/migrations/versions").glob(
        "*f4a5b6c7d8e9*.py"
    )
)
_spec = importlib.util.spec_from_file_location(
    "pd_disaggregation_database_changes", _PATH
)
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A `benchmarks` table with one pre-existing run, wired to alembic's `op`."""
    path = tmp_path / "gpustack.db"
    engine = sa.create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE benchmarks (id INTEGER PRIMARY KEY)"))
        conn.execute(sa.text("INSERT INTO benchmarks (id) VALUES (1)"))

    connection = engine.connect()
    ctx = MigrationContext.configure(connection)
    ops = Operations(ctx)
    # The module reads the live connection through alembic's proxies, and its
    # own guards use the repo helpers, which read the same context.
    monkeypatch.setattr(migration, "op", ops)
    monkeypatch.setattr(migration, "table_exists", lambda name: name == "benchmarks")
    monkeypatch.setattr(
        migration,
        "column_exists",
        lambda table, column: column
        in {c["name"] for c in sa.inspect(connection).get_columns(table)},
    )
    yield connection, path
    connection.close()


def _columns(path):
    con = sqlite3.connect(path)
    try:
        return {c[1] for c in con.execute("PRAGMA table_info(benchmarks)")}
    finally:
        con.close()


def test_the_column_is_added_and_old_rows_say_instance(db):
    connection, path = db
    migration._add_benchmark_target_mode()
    connection.commit()

    assert "target_mode" in _columns(path)
    con = sqlite3.connect(path)
    try:
        assert con.execute("SELECT target_mode FROM benchmarks").fetchall() == [
            ("model_instance",)
        ]
    finally:
        con.close()


def test_upgrading_twice_is_not_an_error(db):
    # Re-running a migration is how a partially applied upgrade is recovered.
    connection, path = db
    migration._add_benchmark_target_mode()
    connection.commit()
    migration._add_benchmark_target_mode()
    connection.commit()
    assert "target_mode" in _columns(path)


def test_downgrade_takes_it_back_off(db):
    connection, path = db
    migration._add_benchmark_target_mode()
    connection.commit()
    migration._drop_benchmark_target_mode()
    connection.commit()
    assert "target_mode" not in _columns(path)
