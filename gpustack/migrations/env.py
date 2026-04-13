from logging.config import fileConfig
import os
import re

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.dialects.postgresql import base as pg_base

from alembic import context

from gpustack import schemas
from sqlmodel import SQLModel

# Patch PGDialect to support openGauss version strings.
# openGauss returns "(openGauss X.Y.Z build ...)" instead of "PostgreSQL X.Y.Z",
# which SQLAlchemy's default regex cannot parse.
_orig_get_server_version_info = pg_base.PGDialect._get_server_version_info


def _patched_get_server_version_info(self, connection):
    try:
        return _orig_get_server_version_info(self, connection)
    except AssertionError:
        v = connection.exec_driver_sql("select pg_catalog.version()").scalar()
        m = re.search(r"openGauss (\d+)\.(\d+)(?:\.(\d+))?", v)
        if not m:
            raise
        return tuple([int(x) if x is not None else 0 for x in m.group(1, 2, 3)])


pg_base.PGDialect._get_server_version_info = _patched_get_server_version_info

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = SQLModel.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url", os.getenv('DATABASE_URL'))
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    url = config.get_main_option("sqlalchemy.url", os.getenv('DATABASE_URL'))
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=url,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            transactional_ddl=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
