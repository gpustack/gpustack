import argparse
import logging
import os
import shutil
from glob import glob
import sys

from gpustack.cmd.start import get_gpustack_env
from gpustack.schemas.system_load import SystemLoad
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.clusters import WorkerPool
from gpustack.schemas.workers import Worker
from gpustack.schemas.users import User
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.links import ModelInstanceModelFileLink, ModelUserLink
from gpustack.server.db import init_db_engine

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, text, func
from sqlalchemy.ext.asyncio import AsyncEngine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cls = [
    SystemLoad,
    WorkerPool,
    Worker,
    User,
    ApiKey,
    ModelFile,
    Model,
    ModelInstance,
    ModelInstanceModelFileLink,
    ModelUserLink,
    ModelUsage,
]

copied_database_file = "gpustack_copied_database.db"


def setup_migrate_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "migrate",
    )
    parser.add_argument(
        "--sqlite-path",
        type=str,
        help="Path to the original sqlite database file to migrate.",
        default=get_gpustack_env("DATA_DIR"),
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
        clean_env(args)
        logger.info("Migration completed successfully.")

    except Exception as e:
        logger.fatal(f"Failed to migrate: {e}")
        sys.exit(1)


async def prepare_env(args):
    logger.info("=" * 30 + " Preparing " + "=" * 30)

    sqlite_path = args.sqlite_path
    database_url = args.database_url
    copied_sqlite_path = _copy_sqlite_file(sqlite_path)

    sqlite_db_url = f"sqlite:///{copied_sqlite_path}"
    postgres_db_url = database_url

    old_engine = await init_db_engine(sqlite_db_url)
    new_engine = await init_db_engine(postgres_db_url)
    return sqlite_db_url, postgres_db_url, old_engine, new_engine


async def upgrade_schema(sqlite_db_url, postgres_db_url, old_engine):
    logger.info("=" * 30 + " Drop views " + "=" * 30)
    await _drop_view(old_engine)

    logger.info("=" * 30 + " SQLite Upgrade " + "=" * 30)
    _run_schema_upgrade(sqlite_db_url)

    logger.info("=" * 30 + " Postgres Upgrade " + "=" * 30)
    _run_schema_upgrade(postgres_db_url)


async def migrate_all_data(old_engine, new_engine):
    logger.info("=" * 30 + " Delete System Users " + "=" * 30)
    await drop_system_users(new_engine)

    logger.info("=" * 30 + " Migrate Data " + "=" * 30)
    await _run_migrate_from_sqlite_to_postgres(old_engine, new_engine)

    logger.info("=" * 30 + " Sync Sequences " + "=" * 30)
    await _sync_sequences(new_engine)


def clean_env(args):
    logger.info("=" * 30 + " Cleaning Up " + "=" * 30)

    sqlite_path = args.sqlite_path
    data_dir = os.path.dirname(sqlite_path)
    prefix = copied_database_file.split('.')[0]

    for f in glob(os.path.join(data_dir, f"{prefix}*")):
        try:
            os.remove(f)
            logger.info(f"Cleaning up temporary files {f}")
        except Exception as e:
            logger.error(f"Failed to remove file {f}: {e}")


def _copy_sqlite_file(sqlite_path: str):
    data_dir = os.path.dirname(sqlite_path)
    copied_file_path = os.path.join(data_dir, copied_database_file)
    try:
        shutil.copyfile(sqlite_path, copied_file_path)
        logger.info(f"Copied sqlite file to {copied_file_path}")
        return copied_file_path
    except Exception as e:
        raise RuntimeError(f"Failed to copy sqlite file: {e}") from e


def _run_schema_upgrade(db_url: str):
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

    db_url_escaped = db_url.replace("%", "%%")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url_escaped)

    try:
        command.upgrade(alembic_cfg, "head")
    except Exception as e:
        raise RuntimeError(f"Database upgrade failed: {e}") from e
    logger.info(f"Database schema upgrade for {db_url} completed.")


async def _drop_view(engine: AsyncEngine):
    logger.info("Dropping views in the old database if any.")
    async with engine.begin() as conn:
        await conn.execute(text("DROP VIEW IF EXISTS gpu_devices_view"))


async def _run_migrate_from_sqlite_to_postgres(
    old_engine: AsyncEngine, new_engine: AsyncEngine
):
    logger.info("Starting data migration from SQLite to Postgres.")
    async with (
        AsyncSession(old_engine) as old_sess,
        AsyncSession(new_engine) as new_sess,
    ):
        for model_cls in cls:
            await migrate_data_for_table(old_sess, new_sess, model_cls)

        await new_sess.commit()


async def drop_system_users(new_engine: AsyncEngine):
    # drop system user id from new db since they already exist in old db
    user_cluster_1_username = "system/cluster-1"
    user_worker_0_username = "system/worker-0"

    async with (AsyncSession(new_engine) as new_sess,):
        user_cluster_1 = await User.one_by_field(
            session=new_sess, field="username", value=user_cluster_1_username
        )
        if user_cluster_1 is not None:
            await user_cluster_1.delete(new_sess)

        user_worker_0 = await User.one_by_field(
            session=new_sess, field="username", value=user_worker_0_username
        )
        if user_worker_0 is not None:
            await user_worker_0.delete(new_sess)


async def migrate_data_for_table(
    old_sess: AsyncSession, new_sess: AsyncSession, model_cls
):
    logger.info(f"Migrating table {model_cls.__tablename__}")

    statement = select(model_cls)
    if hasattr(model_cls, "id"):
        statement = select(model_cls).order_by(model_cls.id)

    result = await old_sess.exec(statement)
    rows = result.all()

    for row in rows:
        old_sess.expunge(row)

    migrated = 0
    for row in rows:
        new_obj = model_cls()

        for col in model_cls.__table__.columns:
            col_name = col.name
            if hasattr(row, col_name):
                setattr(new_obj, col_name, getattr(row, col_name))

        new_sess.add(new_obj)
        migrated += 1

    await new_sess.flush()
    logger.info(f"Migrated {migrated} records into {model_cls.__tablename__}")


async def _sync_sequences(engine: AsyncEngine):
    synced = 0
    async with AsyncSession(engine) as session:
        for model_cls in cls:
            if not hasattr(model_cls, "id"):
                continue

            table_name = model_cls.__tablename__
            seq_name = f"{table_name}_id_seq"

            stmt = select(func.max(model_cls.id))
            result = await session.exec(stmt)
            max_id = result.first()

            if max_id is None:
                continue

            setval_stmt = text("SELECT setval(:seq_name, :max_id)")
            await session.execute(
                setval_stmt, params={"seq_name": seq_name, "max_id": max_id}
            )
            synced += 1
        await session.commit()

    logger.info(f"Synced {synced} sequences.")
