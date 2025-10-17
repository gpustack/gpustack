"""add image_name and run_command to models table, add inference_backends table

Revision ID: nrxtab43e8j8
Revises: cbbc03c88985
Create Date: 2025-09-17 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers, used by Alembic.
revision: str = 'nrxtab43e8j8'
down_revision: Union[str, None] = 'eeacfbc6a2bf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('image_name', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('run_command', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('extended_kv_cache', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('speculative_config', sa.JSON(), nullable=True))

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('draft_model_download_progress', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('draft_model_file_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_model_instances_draft_model_file_id', 'model_files', ['draft_model_file_id'], ['id'])

    # Create inference_backends table
    op.create_table(
        'inference_backends',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('backend_name', sa.String(length=255), nullable=False),
        sa.Column('default_version', sa.String(length=255), nullable=True, default=''),
        sa.Column('description', sa.String(length=255), nullable=True),
        sa.Column('default_run_command', sa.String(length=255), nullable=True, default=''),
        sa.Column('health_check_path', sa.String(length=255), nullable=True, default=''),
        sa.Column('version_configs', sa.JSON(), nullable=True),
        sa.Column('default_backend_param', sa.JSON(), nullable=True),
        sa.Column('is_built_in', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('backend_name')
    )
    
    # Create indexes for inference_backends table
    op.create_index(op.f('ix_inference_backends_backend_name'), 'inference_backends', ['backend_name'], unique=True)


def downgrade() -> None:
    # Drop inference_backends table indexes
    op.drop_index(op.f('ix_inference_backends_backend_name'), table_name='inference_backends')
    
    # Drop inference_backends table
    op.drop_table('inference_backends')

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('draft_model_download_progress')
        batch_op.drop_constraint('fk_model_instances_draft_model_file_id', type_='foreignkey')
        batch_op.drop_column('draft_model_file_id')

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('run_command')
        batch_op.drop_column('image_name')
        batch_op.drop_column('extended_kv_cache')
        batch_op.drop_column('speculative_config')
