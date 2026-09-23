"""The `target_mode` column has to hold the enum VALUE, not its member name.

`load_type` on the same table already carries the comment explaining why, and
the migration that adds `target_mode` writes the literal `'instance'`. When the
field omitted `sa_type=AutoString`, SQLModel gave it a native enum column keyed
on member NAMES, so every read of a row the migration had backfilled raised
``'instance' is not among the defined enum values``. Observed consequence: the
benchmark watch stream died on subscribe and the worker re-subscribed every
five seconds, which takes the whole benchmark feature out.
"""

import sqlalchemy as sa
from sqlmodel.sql.sqltypes import AutoString

from gpustack.schemas.benchmark import Benchmark, BenchmarkTargetModeEnum


def test_target_mode_column_stores_the_value_not_the_member_name():
    column = Benchmark.__table__.c["target_mode"]
    assert isinstance(column.type, AutoString)
    assert not isinstance(column.type, sa.Enum)


def test_target_mode_column_matches_load_type():
    """One convention per table, so a reader does not have to check each field."""
    target_mode = Benchmark.__table__.c["target_mode"]
    load_type = Benchmark.__table__.c["load_type"]
    assert type(target_mode.type) is type(load_type.type)


def test_the_migration_backfill_is_a_readable_value():
    """`'instance'` — what the migration writes — round-trips as the enum."""
    assert BenchmarkTargetModeEnum("instance") is BenchmarkTargetModeEnum.INSTANCE
    assert BenchmarkTargetModeEnum.INSTANCE.value == "instance"
    assert BenchmarkTargetModeEnum.ROUTE.value == "route"
