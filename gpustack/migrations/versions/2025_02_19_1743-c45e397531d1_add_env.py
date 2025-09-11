"""GPUStack v0.6.0

Revision ID: c45e397531d1
Revises: e6bf9e067296
Create Date: 2025-02-19 17:43:06.434145

"""
import shutil
from typing import Sequence, Union
import glob
import logging
import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import sqlmodel
import gpustack
from gpustack.config.config import get_global_config
from gpustack.migrations.utils import column_exists, table_exists


logger = logging.getLogger(__name__)

naming_convention = {
    "fk":
    "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}

# revision identifiers, used by Alembic.
revision: str = 'c45e397531d1'
down_revision: Union[str, None] = 'e6bf9e067296'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        if not column_exists('models', 'env'):
            batch_op.add_column(sa.Column('env', sa.JSON(), nullable=True))
            batch_op.add_column(sa.Column('restart_on_error', sa.Boolean(), nullable=True))

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        if not column_exists('model_instances', 'resolved_path'):
            batch_op.add_column(sa.Column('resolved_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
            batch_op.add_column(sa.Column('restart_count', sa.Integer(), nullable=True))
            batch_op.add_column(sa.Column('last_restart_time', gpustack.schemas.common.UTCDateTime(), nullable=True))

    conn = op.get_bind()
    if not table_exists('model_files'):
        if conn.dialect.name == 'postgresql':
            source_enum_type = postgresql.ENUM('HUGGING_FACE', 'OLLAMA_LIBRARY', 'MODEL_SCOPE', 'LOCAL_PATH', name='sourceenum', create_type=False)
        elif conn.dialect.name == 'mysql':
            source_enum_type = sa.Enum('HUGGING_FACE', 'OLLAMA_LIBRARY', 'MODEL_SCOPE', 'LOCAL_PATH', name='sourceenum', create_constraint=True)
        else:
            source_enum_type = sqlmodel.sql.sqltypes.AutoString()

        op.create_table('model_files',
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('source', source_enum_type, nullable=False),
        sa.Column('huggingface_repo_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('huggingface_filename', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('ollama_library_model_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('model_scope_model_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('model_scope_file_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('local_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('source_index', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('size', sa.BigInteger(), nullable=True),
        sa.Column('download_progress', sa.Float(), nullable=True),
        sa.Column('resolved_paths', sa.JSON(), nullable=True),
        sa.Column('state', sa.Enum('ERROR', 'DOWNLOADING', 'READY', name='modelfilestateenum'), nullable=False),
        sa.Column('state_message', sa.Text(), nullable=True),
        sa.Column('cleanup_on_delete', sa.Boolean(), nullable=True),
        sa.Column('local_dir', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('worker_id', sa.Integer(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )

    if not table_exists('modelinstancemodelfilelink'):
        op.create_table('modelinstancemodelfilelink',
        sa.Column('model_instance_id', sa.Integer(), nullable=False),
        sa.Column('model_file_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['model_file_id'], ['model_files.id'], name='fk_model_instance_model_file_link_model_files', ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['model_instance_id'], ['model_instances.id'], name='fk_model_instance_model_file_link_model_instances', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('model_instance_id', 'model_file_id')
        )

    migrate_legacy_hf_cache()
    remove_legacy_ms_cache_locks()
    remove_legacy_ollama_model_locks()
    alter_users_table_autoincrement_keyword(True)
    delete_orphan_keys()
    alter_api_keys_foreign_key(True)


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        if column_exists('models', 'env'):
            batch_op.drop_column('env')
            batch_op.drop_column('restart_on_error')

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        if column_exists('model_instances', 'resolved_path'):
            batch_op.drop_column('resolved_path')
            batch_op.drop_column('last_restart_time')
            batch_op.drop_column('restart_count')

    if table_exists('modelinstancemodelfilelink'):
        op.drop_table('modelinstancemodelfilelink')

    if table_exists('model_files'):
        op.drop_table('model_files')
    alter_users_table_autoincrement_keyword(False)
    alter_api_keys_foreign_key(False)

def migrate_legacy_hf_cache():
    config = get_global_config()
    if config is None:
        return
    hf_cache_base = os.path.join(config.cache_dir, "huggingface")
    model_dirs = glob.glob(os.path.join(hf_cache_base, "models--*--*"))

    for model_dir in model_dirs:
        parts = model_dir.split("--")
        if len(parts) < 3:
            continue

        org, model = parts[1], parts[2]
        snapshot_dir = os.path.join(model_dir, "snapshots")

        if not os.path.exists(snapshot_dir) or not os.path.isdir(snapshot_dir):
            continue

        snapshot_subdirs = sorted(os.listdir(snapshot_dir))
        if not snapshot_subdirs:
            continue

        first_snapshot = os.path.join(snapshot_dir, snapshot_subdirs[0])
        target_path = os.path.join(hf_cache_base, org, model)


        if os.path.exists(target_path):
            logger.info(f"Target path already exists, skipping: {target_path}")
            continue

        os.makedirs(target_path, exist_ok=True)

        for filename in os.listdir(first_snapshot):
            src_path = os.path.join(first_snapshot, filename)
            dst_path = os.path.join(target_path, filename)

            try:
                if os.path.islink(src_path):
                    real_path = os.path.realpath(src_path)
                    shutil.move(real_path, dst_path)
                else:
                    shutil.move(src_path, dst_path)
            except Exception as e:
                logger.warning(f"Failed to move {src_path} to {dst_path}: {e}")

        shutil.rmtree(model_dir, ignore_errors=True)
        logger.info(f"Migrated {first_snapshot} to {target_path}")


def remove_legacy_ms_cache_locks():
    config = get_global_config()
    if config is None:
        return
    model_scope_dir = os.path.join(config.cache_dir, "model_scope")

    if not os.path.isdir(model_scope_dir):
        return

    for org in os.listdir(model_scope_dir):
        org_path = os.path.join(model_scope_dir, org)
        if not os.path.isdir(org_path):
            continue

        for file in os.listdir(org_path):
            if file.endswith(".lock"):
                lock_path = os.path.join(org_path, file)
                if os.path.isfile(lock_path):
                    try:
                        os.remove(lock_path)
                        logger.info(f"Deleted lock file: {lock_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete lock file {lock_path}: {e}")

def remove_legacy_ollama_model_locks():
    config = get_global_config()
    if config is None:
        return
    ollama_dir = os.path.join(config.cache_dir, "ollama")

    if not os.path.isdir(ollama_dir):
        return

    for file in os.listdir(ollama_dir):
        if file.endswith(".lock"):
            lock_path = os.path.join(ollama_dir, file)
            if os.path.isfile(lock_path):
                try:
                    os.remove(lock_path)
                    logger.info(f"Deleted lock file: {lock_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete lock file {lock_path}: {e}")


def alter_users_table_autoincrement_keyword(auto_increment=False) -> None:
    conn = op.get_bind()
    if conn.engine.name != 'sqlite':
        return
    kwarg = dict()
    if auto_increment:
        kwarg['sqlite_autoincrement'] = True
    # Refer to the workaround here https://github.com/sqlalchemy/alembic/issues/380
    # batchop can only use table_kwargs to parse sqlite_autoincrement to create AUTOINCREMENT keywork for the primary key
    with op.batch_alter_table('users', table_kwargs=kwarg, recreate='always') as batch_op:
        pass

def alter_api_keys_foreign_key(foreign_key=False) -> None:
    foreign_key_name = 'fk_api_keys_user_id_users'
    with op.batch_alter_table('api_keys', naming_convention=naming_convention) as batch_op:
        if foreign_key:
            batch_op.create_foreign_key(foreign_key_name, 'users', ['user_id'], ['id'], ondelete='CASCADE')
        else:
            batch_op.drop_constraint(foreign_key_name, type_='foreignkey')

def delete_orphan_keys() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text(
            """
            DELETE FROM api_keys
            WHERE user_id NOT IN (SELECT id FROM users)
            """
        )
    )
