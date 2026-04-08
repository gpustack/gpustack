"""v2.2.0 add worker version

Revision ID: 8bf38a6bb3b5
Revises: 8ad0f94c92e8
Create Date: 2026-03-10 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import gpustack

# revision identifiers, used by Alembic.
revision: str = '8bf38a6bb3b5'
down_revision: Union[str, None] = '8ad0f94c92e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('worker_version', sa.String(100), nullable=True))
    
    ### k8s volume mount
    with op.batch_alter_table('clusters', schema=None) as batch_op:
        batch_op.add_column(sa.Column('k8s_volume_mounts', gpustack.schemas.common.JSON(), nullable=True))
    ### end

    ### custom API_KEY
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_custom', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('scope', gpustack.schemas.common.JSON(), nullable=True))

    # Update existing API keys to have full access scope
    op.execute("UPDATE api_keys SET scope = '[\"*\"]' WHERE scope IS NULL")
    op.execute("UPDATE api_keys SET is_custom = false WHERE is_custom IS NULL")

    # Set scope to NOT NULL
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.alter_column('scope', nullable=False)
        batch_op.alter_column('is_custom', nullable=False)


def downgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('worker_version')

    ### k8s volume mount
    with op.batch_alter_table('clusters', schema=None) as batch_op:
        batch_op.drop_column('k8s_volume_mounts')
    ### end

    ### custom API_KEY
    op.execute("DELETE FROM api_keys WHERE is_custom = true")
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.drop_column('is_custom')
        batch_op.drop_column('scope')
    ### end
