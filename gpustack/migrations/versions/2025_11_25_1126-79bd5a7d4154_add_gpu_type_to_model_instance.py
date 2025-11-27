"""add gpu_type to model instance

Revision ID: 79bd5a7d4154
Revises: e30134bd18dc
Create Date: 2025-11-25 11:26:13.760245

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '79bd5a7d4154'
down_revision: Union[str, None] = 'e30134bd18dc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('gpu_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('gpu_type')
