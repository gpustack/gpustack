"""support speculative config

Revision ID: 912d04e0d1d0
Revises: 7737c7eba9c4
Create Date: 2025-10-28 15:48:39.610731

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack

from gpustack.migrations.utils import  table_exists

# revision identifiers, used by Alembic.
revision: str = '912d04e0d1d0'
down_revision: Union[str, None] = '7737c7eba9c4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('speculative_config', sa.JSON(), nullable=True))

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('draft_model_download_progress', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('draft_model_resolved_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column('draft_model_source',
            gpustack.schemas.common.JSON(), nullable=True))

    if not table_exists('modelinstancedraftmodelfilelink'):
        op.create_table('modelinstancedraftmodelfilelink',
        sa.Column('model_instance_id', sa.Integer(), nullable=False),
        sa.Column('model_file_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['model_file_id'], ['model_files.id'], name='fk_model_instance_draft_model_file_link_model_files', ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['model_instance_id'], ['model_instances.id'], name='fk_model_instance_draft_model_file_link_model_instances', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('model_instance_id', 'model_file_id')
        )

def downgrade() -> None:
    if table_exists('modelinstancedraftmodelfilelink'):
        op.drop_table('modelinstancedraftmodelfilelink')

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('draft_model_download_progress')
        batch_op.drop_column('draft_model_resolved_path')
        batch_op.drop_column('draft_model_source')

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('speculative_config')
