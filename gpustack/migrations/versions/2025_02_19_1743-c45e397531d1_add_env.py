"""add env

Revision ID: c45e397531d1
Revises: e6bf9e067296
Create Date: 2025-02-19 17:43:06.434145

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'c45e397531d1'
down_revision: Union[str, None] = 'e6bf9e067296'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('env', sa.JSON(), nullable=True))

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('resolved_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True))

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
    sa.Column('local_dir', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('worker_id', sa.Integer(), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
    sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('modelinstancemodelfilelink',
    sa.Column('model_instance_id', sa.Integer(), nullable=False),
    sa.Column('model_file_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['model_file_id'], ['model_files.id'], name='fk_model_instance_model_file_link_model_files', ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['model_instance_id'], ['model_instances.id'], name='fk_model_instance_model_file_link_model_instances', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('model_instance_id', 'model_file_id')
    )


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('env')

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('resolved_path')

    op.drop_table('modelinstancemodelfilelink')
    op.drop_table('model_files')
