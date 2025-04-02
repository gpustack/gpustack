"""GPUStack v0.6.0

Revision ID: c45e397531d1
Revises: e6bf9e067296
Create Date: 2025-02-19 17:43:06.434145

"""
from typing import Sequence, Union
import glob
import logging
import os

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from gpustack.config.config import get_global_config
from gpustack.migrations.utils import column_exists, table_exists


logger = logging.getLogger(__name__)



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

    if not table_exists('model_files'):
        op.create_table('model_files',
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('source', sa.Enum('HUGGING_FACE', 'OLLAMA_LIBRARY', 'MODEL_SCOPE', 'LOCAL_PATH', name='sourceenum'), nullable=False),
        sa.Column('huggingface_repo_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('huggingface_filename', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('ollama_library_model_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('model_scope_model_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('model_scope_file_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('local_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('source_index', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('size', sa.Integer(), nullable=True),
        sa.Column('download_progress', sa.Float(), nullable=True),
        sa.Column('resolved_paths', sa.JSON(), nullable=True),
        sa.Column('state', sa.Enum('ERROR', 'DOWNLOADING', 'READY', name='modelfilestateenum'), nullable=False),
        sa.Column('state_message', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
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

    create_legacy_hf_cache_symlinks()


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

def create_legacy_hf_cache_symlinks():
    config = get_global_config()
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

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if os.path.exists(target_path) or os.path.islink(target_path):
            continue

        os.symlink(first_snapshot, target_path)
        logger.info(f"Created symlink: {target_path} -> {first_snapshot}")
