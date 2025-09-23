"""add worker metrics port

Revision ID: f3d36b2d11f8
Revises: 924c9a0b4c13
Create Date: 2025-09-17 18:12:15.025866

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'f3d36b2d11f8'
down_revision: Union[str, None] = '924c9a0b4c13'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('metrics_port', sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('metrics_port')
