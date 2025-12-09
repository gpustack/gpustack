"""increase run_command length

Revision ID: a7c1b20f9d3e
Revises: e30134bd18dc
Create Date: 2025-12-09 12:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers, used by Alembic.
revision: str = 'a7c1b20f9d3e'
down_revision: Union[str, None] = 'e30134bd18dc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column('run_command', type_=sa.Text(), existing_type=sa.String(length=255), nullable=True)

    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.alter_column('default_run_command', type_=sa.Text(), existing_type=sa.String(length=255), nullable=True)


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column('run_command', type_=sa.String(length=255), existing_type=sa.Text(), nullable=True)

    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.alter_column('default_run_command', type_=sa.String(length=255), existing_type=sa.Text(), nullable=True)
