"""Gateway Followup

Revision ID: 2ea2c247a117
Revises: 912d04e0d1d0
Create Date: 2025-11-01 00:34:19.224562

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = '2ea2c247a117'
down_revision: Union[str, None] = '912d04e0d1d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_usages", schema=None) as batch_op:
        batch_op.alter_column('operation',existing_type=sa.VARCHAR(length=16), nullable=True)
        batch_op.alter_column('user_id', existing_type=sa.Integer(), nullable=True)
        batch_op.add_column(sa.Column('access_key', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.add_column(sa.Column('generic_proxy', sa.Boolean(), nullable=True, server_default=sa.sql.expression.false()))

def downgrade() -> None:
    op.execute(
        "DELETE FROM model_usages WHERE access_key IS NOT NULL OR user_id IS NULL OR operation IS NULL"
    )
    with op.batch_alter_table("model_usages", schema=None) as batch_op:
        batch_op.drop_column('access_key')
        batch_op.alter_column('user_id', existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column('operation', existing_type=sa.VARCHAR(length=16), nullable=False)
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.drop_column('generic_proxy')
