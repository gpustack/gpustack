"""Model Access Control

Revision ID: bb23ccc18b80
Revises: f3d36b2d11f8
Create Date: 2025-09-26 22:53:18.606454

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from gpustack.migrations.utils import table_exists

# revision identifiers, used by Alembic.
revision: str = 'bb23ccc18b80'
down_revision: Union[str, None] = 'f3d36b2d11f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def access_control_upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('public', sa.Boolean(), nullable=True))
    op.execute(
        sa.text("UPDATE models SET public = true WHERE public IS NULL")
    )
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column('public', existing_type=sa.Boolean(), nullable=False)

    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.add_column(sa.Column('allowed_model_names', sa.JSON(), nullable=True))
    if not table_exists('modeluserlink'):
        op.create_table('modeluserlink',
            sa.Column('model_id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(['model_id'], ['models.id'], name='fk_model_user_link_models', ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_model_user_link_users', ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('model_id', 'user_id')
        )
    
def access_control_downgrade() -> None:
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.drop_column('allowed_model_names')
    if table_exists('modeluserlink'):
        op.drop_table('modeluserlink')
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('public')


def upgrade() -> None:
    access_control_upgrade()


def downgrade() -> None:
    access_control_downgrade()
