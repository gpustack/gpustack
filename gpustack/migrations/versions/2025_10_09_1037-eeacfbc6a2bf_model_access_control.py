"""Model Access Control

Revision ID: eeacfbc6a2bf
Revises: 2025_10_07_add_is_active
Create Date: 2025-10-09 10:37:20.646154

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from gpustack.migrations.utils import table_exists

# revision identifiers, used by Alembic.
revision: str = 'eeacfbc6a2bf'
down_revision: Union[str, None] = '2025_10_07_add_is_active'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def access_control_upgrade() -> None:
    access_policy_enum = sa.Enum(
        'PUBLIC',
        'AUTHED',
        'ALLOWED_USERS',
        name='accesspolicyenum',
    )
    bind = op.get_bind()
    if bind.dialect.name in ('postgresql', 'mysql'):
        access_policy_enum.create(bind, checkfirst=True)
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('access_policy', access_policy_enum, nullable=True, server_default='AUTHED'))
    op.execute(
        "UPDATE models SET access_policy='AUTHED' WHERE access_policy IS NULL"
    )
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column('access_policy', existing_type=access_policy_enum, nullable=False)

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
        batch_op.drop_column('access_policy')
    access_policy_enum = sa.Enum(
        'PUBLIC',
        'AUTHED',
        'ALLOWED_USERS',
        name='accesspolicyenum',
    )
    access_policy_enum.drop(op.get_bind(), checkfirst=True)


def upgrade() -> None:
    access_control_upgrade()


def downgrade() -> None:
    access_control_downgrade()
