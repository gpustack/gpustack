import argparse
import logging
import os
import re
import shutil
from glob import glob
import sys
from typing import Optional
from urllib.parse import parse_qs, urlparse, urlunparse

from sqlalchemy import MetaData, Table
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, text, func
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)

from gpustack.cmd.start import get_gpustack_env
from gpustack import envs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

revision = "d19176de3b74"  # 0.7.1

TARGET_TABLES = [
    "system_loads",
    "workers",
    "users",
    "api_keys",
    "model_files",
    "models",
    "model_instances",
    "modelinstancemodelfilelink",
    "model_usages",
]

migration_temp_file_prefix = "gpustack_migration_temp_"


def setup_migrate_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "migrate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--migration-data-dir",
        type=str,
        help="Data directory to include the original sqlite file database.db to migrate.",
        default=get_gpustack_env("MIGRATION_DATA_DIR"),
        required=True,
    )

    parser.add_argument(
        "--database-url",
        type=str,
        help="Target database URL, e.g. postgresql://user:password@host:port/db_name.",
        default=get_gpustack_env("DATABASE_URL"),
        required=True,
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace):
    import asyncio

    asyncio.run(_run(args))


async def _run(args):
    try:
        logger.info("Starting database migration...")
        sqlite_db_url, postgres_db_url, old_engine, new_engine = await prepare_env(args)
        await upgrade_schema(sqlite_db_url, postgres_db_url, old_engine)
        await migrate_all_data(old_engine, new_engine)
        await old_engine.dispose()
        await new_engine.dispose()
        clean_env(args)
        logger.info("Migration completed successfully.")

    except Exception as e:
        logger.fatal(f"Failed to migrate: {e}")
        sys.exit(1)


async def prepare_env(args):
    logger.info("=" * 30 + " Preparing " + "=" * 30)

    data_dir = args.migration_data_dir
    database_url = args.database_url
    migration_temp_sqlite_path = _copy_sqlite_file(data_dir)

    sqlite_db_url = f"sqlite:///{migration_temp_sqlite_path}"
    postgres_db_url = database_url

    old_engine = await init_db_engine(sqlite_db_url)
    new_engine = await init_db_engine(postgres_db_url)
    return sqlite_db_url, postgres_db_url, old_engine, new_engine


async def upgrade_schema(sqlite_db_url, postgres_db_url, old_engine):
    logger.info("=" * 30 + " Drop views " + "=" * 30)
    await _drop_view(old_engine)

    logger.info("=" * 30 + " SQLite Upgrade " + "=" * 30)
    _run_schema_upgrade(sqlite_db_url, revision)

    logger.info("=" * 30 + " Postgres Upgrade " + "=" * 30)
    _run_schema_upgrade(postgres_db_url, revision)


def clean_env(args):
    logger.info("=" * 30 + " Cleaning Up " + "=" * 30)

    data_dir = args.migration_data_dir
    for f in glob(os.path.join(data_dir, f"{migration_temp_file_prefix}*")):
        try:
            os.remove(f)
            logger.info(f"Cleaning up temporary files {f}")
        except Exception as e:
            logger.error(f"Failed to remove file {f}: {e}")


def _copy_sqlite_file(data_dir: str):
    sqlite_path = ""
    required_files = ["database.db"]
    optional_files = ["database.db-wal"]
    for f in required_files + optional_files:
        file_path = os.path.join(data_dir, f)
        if os.path.exists(file_path) is False:
            if f in required_files:
                raise FileNotFoundError(f"Required sqlite file {file_path} not found.")
            else:
                continue

        copied_file_path = os.path.join(data_dir, f"{migration_temp_file_prefix}{f}")
        try:
            shutil.copyfile(file_path, copied_file_path)
            logger.info(f"Copied sqlite file to {copied_file_path}")
            if f == "database.db":
                sqlite_path = copied_file_path
        except Exception as e:
            raise RuntimeError(f"Failed to copy sqlite file: {e}") from e

    return sqlite_path


def _run_schema_upgrade(db_url: str, revision: str = "head"):
    logger.info(f"Running schema upgrade for {db_url}.")

    from alembic import command
    from alembic.config import Config as AlembicConfig
    import importlib.util

    spec = importlib.util.find_spec("gpustack")
    if spec is None:
        raise ImportError("The 'gpustack' package is not found.")

    pkg_path = spec.submodule_search_locations[0]
    alembic_cfg = AlembicConfig()
    alembic_cfg.set_main_option("script_location", os.path.join(pkg_path, "migrations"))
    alembic_cfg.set_main_option("called_by_db_migration", "true")

    db_url_escaped = db_url.replace("%", "%%")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url_escaped)

    try:
        command.upgrade(alembic_cfg, revision)
    except Exception as e:
        raise RuntimeError(f"Database upgrade failed: {e}") from e
    logger.info(f"Database schema upgrade for {db_url} completed.")


async def _drop_view(engine: AsyncEngine):
    logger.info("Dropping views in the old database if any.")
    async with engine.begin() as conn:
        await conn.execute(text("DROP VIEW IF EXISTS gpu_devices_view"))


async def migrate_all_data(old_engine: AsyncEngine, new_engine: AsyncEngine):
    logger.info("=" * 30 + " Migrate Data " + "=" * 30)

    old_meta = MetaData()
    new_meta = MetaData()

    async with old_engine.begin() as conn:
        await conn.run_sync(old_meta.reflect, only=TARGET_TABLES)
    async with new_engine.begin() as conn:
        await conn.run_sync(new_meta.reflect, only=TARGET_TABLES)

    for table_name in TARGET_TABLES:
        await _migrate_table(table_name, old_meta, new_meta, old_engine, new_engine)

    await _sync_table_sequence(new_meta, new_engine)


async def _migrate_table(
    table_name: str,
    old_meta: MetaData,
    new_meta: MetaData,
    old_engine: AsyncEngine,
    new_engine: AsyncEngine,
):
    old_table: Optional[Table] = old_meta.tables.get(table_name)
    new_table: Optional[Table] = new_meta.tables.get(table_name)

    if old_table is None:
        logger.info(f"Old database lack of {table_name}, skip.")
        return
    if new_table is None:
        logger.info(f"New database lack of {table_name}, skip.")
        return

    common_cols = [c for c in old_table.columns.keys() if c in new_table.columns.keys()]
    if not common_cols:
        logger.info(f"Table {table_name} has no common columns, skipping.")
        return

    async with (
        AsyncSession(old_engine) as old_sess,
        AsyncSession(new_engine) as new_sess,
    ):
        stmt = select(*[old_table.c[col] for col in common_cols])
        result = await old_sess.execute(stmt)
        rows = result.fetchall()

        if not rows:
            logger.info(f"Old table {table_name} has no data, skipping.")
            return

        # Convert to dictionary
        data = [dict(zip(common_cols, row)) for row in rows]

        # Insert into new database
        await new_sess.execute(new_table.insert(), data)
        await new_sess.commit()

        logger.info(f"Table {table_name} has migrated {len(data)} records.")


async def _sync_table_sequence(new_meta: MetaData, new_engine: AsyncEngine):
    synced = 0
    async with AsyncSession(new_engine) as session:
        for table_name, table in new_meta.tables.items():
            if table_name not in TARGET_TABLES:
                continue

            id_col = table.columns.get("id")
            if id_col is None:
                continue

            stmt = select(func.max(id_col))
            result = await session.execute(stmt)
            max_id = result.scalar()

            if max_id is None:
                continue

            seq_name = f"{table_name}_id_seq"

            setval_stmt = text('SELECT setval(:seq_name, :max_id)')
            await session.execute(setval_stmt, {"seq_name": seq_name, "max_id": max_id})
            synced += 1

        await session.commit()

    logger.info(f"Synced {synced} sequences.")


async def init_db_engine(db_url: str):
    connect_args = {}
    if db_url.startswith("sqlite://"):
        connect_args = {"check_same_thread": False}
        # use async driver
        db_url = re.sub(r'^sqlite://', 'sqlite+aiosqlite://', db_url)
    elif db_url.startswith("postgresql://"):
        db_url = re.sub(r'^postgresql://', 'postgresql+asyncpg://', db_url)
        parsed = urlparse(db_url)
        # rewrite the parameters to use asyncpg with custom database schema
        query_params = parse_qs(parsed.query)
        qoptions = query_params.pop('options', None)
        schema_name = None
        if qoptions is not None and len(qoptions) > 0:
            option = qoptions[0]
            if option.startswith('-csearch_path='):
                schema_name = option[len('-csearch_path=') :]
        if schema_name:
            connect_args['server_settings'] = {'search_path': schema_name}
        new_parsed = parsed._replace(query={})
        db_url = urlunparse(new_parsed)

    elif db_url.startswith("mysql://"):
        db_url = re.sub(r'^mysql://', 'mysql+asyncmy://', db_url)
    else:
        raise Exception(f"Unsupported database URL: {db_url}")

    engine = create_async_engine(
        db_url,
        echo=envs.DB_ECHO,
        pool_size=envs.DB_POOL_SIZE,
        max_overflow=envs.DB_MAX_OVERFLOW,
        pool_timeout=envs.DB_POOL_TIMEOUT,
        connect_args=connect_args,
    )
    return engine
