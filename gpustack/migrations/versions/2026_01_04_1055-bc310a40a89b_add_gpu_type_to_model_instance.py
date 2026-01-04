"""add gpu_type to model_instance

Revision ID: bc310a40a89b
Revises: 2aed534bd7b2
Create Date: 2026-01-04 10:55:54.240705

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'bc310a40a89b'
down_revision: Union[str, None] = '2aed534bd7b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('gpu_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('gpu_type')
