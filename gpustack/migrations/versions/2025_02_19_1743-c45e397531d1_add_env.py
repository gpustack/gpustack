"""add env

Revision ID: c45e397531d1
Revises: e6bf9e067296
Create Date: 2025-02-19 17:43:06.434145

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'c45e397531d1'
down_revision: Union[str, None] = 'e6bf9e067296'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('env', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('env')
