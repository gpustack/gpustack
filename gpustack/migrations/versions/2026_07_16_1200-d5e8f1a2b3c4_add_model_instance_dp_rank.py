"""add model_instances.dp_rank

Add a nullable ``dp_rank`` integer column to ``model_instances`` for the vLLM
data-parallel node-per-instance path: each DP node is a standalone
``ModelInstance`` carrying its own rank (0 = coordinator). NULL for every
other instance, so the column is fully backward compatible.

Revision ID: d5e8f1a2b3c4
Revises: c4d7e8f9a0b1
Create Date: 2026-07-16 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5e8f1a2b3c4'
down_revision: Union[str, None] = 'c4d7e8f9a0b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'model_instances',
        sa.Column('dp_rank', sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('model_instances', 'dp_rank')
