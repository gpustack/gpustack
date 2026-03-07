"""add parameter_format and common_parameters to inference_backends

Revision ID: 3816b7d7a7a1
Revises: 53667f33f000
Create Date: 2026-02-24 14:21:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3816b7d7a7a1'
down_revision: Union[str, None] = '53667f33f000'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add parameter_format and common_parameters columns to inference_backends table."""
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('parameter_format', sa.String(length=255), nullable=True)
        )
        batch_op.add_column(
            sa.Column('common_parameters', sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    """Remove parameter_format and common_parameters columns from inference_backends table."""
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.drop_column('common_parameters')
        batch_op.drop_column('parameter_format')
