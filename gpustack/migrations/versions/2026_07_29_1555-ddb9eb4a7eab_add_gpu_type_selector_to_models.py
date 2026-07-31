"""add gpu_type_selector to models

Revision ID: ddb9eb4a7eab
Revises: 367a3982fcde
Create Date: 2026-07-29 15:55:40.596358

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'ddb9eb4a7eab'
down_revision: Union[str, None] = '367a3982fcde'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('gpu_type_selector', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('gpu_type_selector')
