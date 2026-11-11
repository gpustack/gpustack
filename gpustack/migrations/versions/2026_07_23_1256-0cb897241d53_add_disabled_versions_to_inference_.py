"""add disabled_versions to inference_backends

Revision ID: 0cb897241d53
Revises: c4d7e8f9a0b1
Create Date: 2026-07-23 12:56:21.126188

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0cb897241d53'
down_revision: Union[str, None] = 'c4d7e8f9a0b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.add_column(sa.Column('disabled_versions', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.drop_column('disabled_versions')
